#!/usr/bin/env python3
"""
summarize_puncta_by_cell.py

For each cytoplasm mask image:

- Find the matching puncta mask and raw cytoplasm intensity image.
- For each cell (label) in the cytoplasm mask:
    * Check if any puncta pixels fall inside that cell.
    * Count puncta pixels.
    * Compute mean cytoplasm intensity for that cell.
- Save everything to a CSV with one row per cell.

Columns:
    image_location      (e.g. "0_Z004")
    cyto_file
    puncta_file
    intensity_file
    cell_label
    num_cyto_pixels
    num_puncta_pixels
    has_puncta          (0/1)
    cyto_mean_intensity
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
from skimage import measure


def parse_location(path: Path) -> str:
    """
    Extract the location token from the filename.

    Example:
        ".../XYPos/0_Z004.ome_cyto3_masks.tif" -> "0_Z004"
    """
    s = str(path)
    if "XYPos" in s:
        after = s.split("XYPos", 1)[1].lstrip("/\\")
        # after = "0_Z004.ome_cyto3_masks.tif"
        token = after.split(".ome", 1)[0]  # "0_Z004"
        token = token.split()[0]           # just in case there are trailing spaces
        return token
    else:
        # Fallback if pattern is different: use stem
        return path.stem


def ensure_2d(arr: np.ndarray) -> np.ndarray:
    """
    Reduce an array down to 2D (Y, X) by repeatedly taking arr[0]
    if there are extra leading dimensions (e.g. Z, C).
    """
    out = arr
    while out.ndim > 2:
        out = out[0]
    return out


def get_cyto_labels(cyto: np.ndarray) -> np.ndarray:
    """
    Ensure we have a labeled mask:
    - If it's already multi-label (Cellpose-style), return as-is.
    - If it's binary, label connected components.
    """
    vals = np.unique(cyto)
    # Binary-like (0/1 or 0/single nonzero)
    if len(vals) <= 2 and 0 in vals:
        return measure.label(cyto > 0, connectivity=1)
    else:
        return cyto.astype(int)


def main(cyto_dir, puncta_dir, intensity_dir, out_csv, min_puncta_pixels=1):
    cyto_dir = Path(cyto_dir)
    puncta_dir = Path(puncta_dir)
    intensity_dir = Path(intensity_dir)

    rows = []

    for cyto_path in sorted(cyto_dir.glob("*.tif")):
        location = parse_location(cyto_path)

        # --- Match puncta file using the location token ---
        puncta_candidates = sorted(puncta_dir.glob(f"*{location}*"))
        if not puncta_candidates:
            print(f"[WARN] No puncta file for {cyto_path.name} (location {location}), skipping")
            continue
        puncta_path = puncta_candidates[0]

        # --- Match intensity file using the same location token ---
        intensity_candidates = sorted(intensity_dir.glob(f"*{location}*"))
        if not intensity_candidates:
            print(f"[WARN] No intensity file for {cyto_path.name} (location {location}), skipping")
            continue
        intensity_path = intensity_candidates[0]

        # --- Load data ---
        cyto = ensure_2d(tiff.imread(cyto_path))
        puncta = ensure_2d(tiff.imread(puncta_path))
        img = ensure_2d(tiff.imread(intensity_path))

        if cyto.shape != img.shape:
            raise ValueError(
                f"Shape mismatch between cyto {cyto.shape} and intensity {img.shape} "
                f"for {cyto_path.name}"
            )

        if puncta.shape != cyto.shape:
            raise ValueError(
                f"Shape mismatch between cyto {cyto.shape} and puncta {puncta.shape} "
                f"for {cyto_path.name}"
            )

        labels = get_cyto_labels(cyto)
        max_label = int(labels.max())

        for lab in range(1, max_label + 1):
            cell_mask = labels == lab
            num_cyto_pixels = int(cell_mask.sum())
            if num_cyto_pixels == 0:
                continue

            # Puncta pixels inside this cell
            puncta_pixels = np.logical_and(cell_mask, puncta > 0)
            num_puncta_pixels = int(puncta_pixels.sum())
            has_puncta = int(num_puncta_pixels >= min_puncta_pixels)

            # Mean cytoplasm intensity inside this cell
            cyto_mean_intensity = float(img[cell_mask].mean())

            rows.append(
                {
                    "image_location": location,
                    "cyto_file": cyto_path.name,
                    "puncta_file": puncta_path.name,
                    "intensity_file": intensity_path.name,
                    "cell_label": lab,
                    "num_cyto_pixels": num_cyto_pixels,
                    "num_puncta_pixels": num_puncta_pixels,
                    "has_puncta": has_puncta,
                    "cyto_mean_intensity": cyto_mean_intensity,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cyto-dir", required=True, help="Folder with cytoplasm masks")
    parser.add_argument("--puncta-dir", required=True, help="Folder with puncta masks")
    parser.add_argument(
        "--intensity-dir",
        required=True,
        help="Folder with raw cytoplasm intensity images",
    )
    parser.add_argument("--out-csv", required=True, help="Path to output CSV file")
    parser.add_argument(
        "--min-puncta-pixels",
        type=int,
        default=1,
        help="Minimum overlapping puncta pixels to call a cell puncta-positive",
    )

    args = parser.parse_args()
    main(**vars(args))
