#!/usr/bin/env python3
"""
mean_intensity_and_puncta.py  (Csat-ready + triptychs + channel selection)

Per-cell summary for nucleus / puncta images:

- Uses a user-specified intensity channel from OME-TIFF.
- Computes per-cell mean nucleus intensity (raw).
- Calls puncta presence per cell using a cleaned puncta mask with
  a minimum puncta area threshold.
- Tracks saturation fraction, puncta area, and puncta density.
- (Optional) Generates triptych PNGs per image.

Usage
-----
    python mean_intensity_and_puncta.py \\
        --nuc-dir /path/to/nucleus_masks \\
        --puncta-dir /path/to/puncta_masks \\
        --intensity-dir /path/to/ome_tiffs \\
        --out-csv results.csv \\
        --make-triptychs --triptych-out-dir triptychs
"""

import sys
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
from skimage import measure, morphology
import matplotlib.pyplot as plt

# Import shared utilities
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from segmentation_utils import auto_lut_clip, ensure_2d, load_image_2d


# ------------------------------------------------------------------ #
#  Location-based file matching
# ------------------------------------------------------------------ #

def parse_location(path: Path) -> str:
    """
    Extract an image location token from the file path.

    Tries XYPos patterns first, then generic digit_Z patterns,
    then falls back to filename stem (stripping common mask suffixes).
    """
    s = str(path)
    m = re.search(r"XYPos[/\\]([^.]*)\.ome", s)
    if m:
        return m.group(1)
    m = re.search(r"(\d+_Z\d+)", s)
    if m:
        return m.group(1)
    stem = path.stem
    # Strip Cellpose _seg suffix and common mask suffixes
    for suffix in ("_seg", "_cyto3_masks", "_cell_masks", "_cyto_masks", "_masks"):
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
            break
    return stem


def build_location_map(root: Path, kind: str) -> dict:
    """Build {location_token: Path} for all .tif/.npy files under *root*.

    Supports both plain TIFF masks and Cellpose ``_seg.npy`` files.
    When both exist for the same location, ``_seg.npy`` takes priority
    (these may contain manually curated masks edited in Cellpose GUI).
    """
    mapping = {}
    for ext in ("*.tif", "*_seg.npy"):
        for p in root.rglob(ext):
            loc = parse_location(p)
            if loc in mapping:
                # _seg.npy takes priority over .tif
                if p.suffix == ".npy" and mapping[loc].suffix != ".npy":
                    mapping[loc] = p
                elif p.suffix == ".npy" or mapping[loc].suffix == ".npy":
                    pass  # keep existing npy
                else:
                    print(f"[WARN] {kind}: duplicate location '{loc}'")
                    print(f"       existing -> {mapping[loc]}")
                    print(f"       new      -> {p}")
            else:
                mapping[loc] = p
    print(f"[INFO] {kind}: indexed {len(mapping)} locations from {root}")
    return mapping


# ------------------------------------------------------------------ #
#  Label helpers
# ------------------------------------------------------------------ #

def get_labels(mask: np.ndarray) -> np.ndarray:
    """Return a label image; auto-label binary masks."""
    vals = np.unique(mask)
    if len(vals) <= 2 and 0 in vals:
        return measure.label(mask > 0, connectivity=1)
    return mask.astype(int)


def _load_mask_2d(path: Path) -> np.ndarray:
    """Load a 2D mask from a .tif or Cellpose _seg.npy file."""
    path = Path(path)
    if path.suffix == ".npy":
        dat = np.load(str(path), allow_pickle=True).item()
        return ensure_2d(np.asarray(dat["masks"]))
    return ensure_2d(tiff.imread(path))


def load_intensity_image(path: Path, channel: int) -> np.ndarray:
    """
    Load one 2D channel from an OME-TIFF or plain TIFF.

    Handles (C,Y,X), (Z,C,Y,X), and (Y,X) shapes.
    """
    arr = np.asarray(tiff.imread(path))
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if channel >= arr.shape[0]:
            raise ValueError(
                f"Requested channel {channel}, but array shape is {arr.shape}"
            )
        return arr[channel]
    if arr.ndim == 4:
        if channel >= arr.shape[1]:
            raise ValueError(
                f"Requested channel {channel}, but array shape is {arr.shape}"
            )
        return arr[0, channel]
    return ensure_2d(arr)


# ------------------------------------------------------------------ #
#  Triptych (4-panel QC image)
# ------------------------------------------------------------------ #

def make_triptych(img, img_puncta, labels, puncta_mask,
                  has_puncta_map, centroid_map, out_path):
    """
    Save a 4-panel QC image:
      [0] Nucleus channel (LUT-normalized)
      [1] Label map with cell IDs (red = puncta+, white = puncta-)
      [2] Puncta overlay on nucleus channel
      [3] Puncta channel (LUT-normalized)
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    cmap = plt.cm.get_cmap("tab20").copy()
    cmap.set_bad(color="black")

    # Panel 1
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Nucleus (LUT)")
    axes[0].axis("off")

    # Panel 2
    labels_masked = np.ma.masked_where(labels == 0, labels)
    axes[1].imshow(labels_masked, cmap=cmap)
    axes[1].set_title("Labels (red=puncta+)")
    axes[1].axis("off")
    for lab, (cy, cx) in centroid_map.items():
        if lab == 0:
            continue
        color = "red" if has_puncta_map.get(lab, 0) == 1 else "white"
        axes[1].text(cx, cy, str(lab), color=color,
                     ha="center", va="center", fontsize=6, weight="bold")

    # Panel 3
    axes[2].imshow(img, cmap="gray")
    p_masked = np.ma.masked_where(puncta_mask == 0, puncta_mask)
    axes[2].imshow(p_masked, cmap=cmap, alpha=0.5)
    axes[2].set_title("Puncta overlay")
    axes[2].axis("off")

    # Panel 4
    axes[3].imshow(img_puncta, cmap="gray")
    axes[3].set_title("Puncta (LUT)")
    axes[3].axis("off")

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ #
#  Main analysis
# ------------------------------------------------------------------ #

def main(
    nuc_dir,
    puncta_dir,
    intensity_dir,
    out_csv,
    min_puncta_area=5,
    puncta_open_radius=1,
    make_triptychs=False,
    triptych_out_dir=None,
    intensity_channel=2,
    puncta_channel=1,
):
    nuc_dir = Path(nuc_dir)
    puncta_dir = Path(puncta_dir)
    intensity_dir = Path(intensity_dir)
    triptych_out_dir = Path(triptych_out_dir) if triptych_out_dir else None

    puncta_map = build_location_map(puncta_dir, "puncta")
    intensity_map = build_location_map(intensity_dir, "intensity")
    rows = []

    nuc_files = sorted(
        list(nuc_dir.glob("*.tif")) + list(nuc_dir.glob("*_seg.npy"))
    )
    print(f"[INFO] Processing {len(nuc_files)} nucleus mask file(s)")

    for idx, cyto_path in enumerate(nuc_files, 1):
        location = parse_location(cyto_path)

        puncta_path = puncta_map.get(location)
        if puncta_path is None:
            print(f"[WARN] No puncta file for location '{location}', skipping")
            continue

        intensity_path = intensity_map.get(location)
        if intensity_path is None:
            print(f"[WARN] No intensity file for location '{location}', skipping")
            continue

        # Load data (supports .tif and Cellpose _seg.npy)
        nucleus = _load_mask_2d(cyto_path)
        puncta = _load_mask_2d(puncta_path)
        img = load_intensity_image(intensity_path, channel=intensity_channel)
        img_puncta = load_intensity_image(intensity_path, channel=puncta_channel)

        # Shape validation
        if nucleus.shape != img.shape:
            raise ValueError(
                f"Shape mismatch: nucleus mask {nucleus.shape} vs "
                f"intensity image {img.shape} for {cyto_path.name}"
            )
        if puncta.shape != nucleus.shape:
            raise ValueError(
                f"Shape mismatch: nucleus mask {nucleus.shape} vs "
                f"puncta mask {puncta.shape} for {cyto_path.name}"
            )

        labels = get_labels(nucleus)
        max_label = int(labels.max())
        if max_label == 0:
            print(f"[WARN] No cells in {cyto_path.name}, skipping")
            continue

        props = measure.regionprops(labels)
        centroid_map = {p.label: p.centroid for p in props}

        # Clean puncta mask
        if puncta_open_radius > 0:
            se = morphology.disk(puncta_open_radius)
            puncta_clean = morphology.opening(puncta > 0, se)
        else:
            puncta_clean = puncta > 0

        # Saturation bound
        if np.issubdtype(img.dtype, np.integer):
            imax = np.iinfo(img.dtype).max
        else:
            imax = np.nan

        has_puncta_map = {}

        for lab in range(1, max_label + 1):
            nuc_mask = labels == lab
            n_pix = int(nuc_mask.sum())
            if n_pix == 0:
                continue

            # Saturation fraction
            sat_frac = float((img[nuc_mask] >= imax).mean()) if np.isfinite(imax) else 0.0

            # Puncta overlap
            puncta_inside = np.logical_and(nuc_mask, puncta_clean)
            n_puncta_px = int(puncta_inside.sum())
            has_puncta = int(n_puncta_px >= min_puncta_area)
            has_puncta_map[lab] = has_puncta

            # Mean raw intensity
            nuc_mean_raw = float(img[nuc_mask].astype(np.float32).mean())

            cy, cx = centroid_map.get(lab, (np.nan, np.nan))

            rows.append({
                "image_location": location,
                "nuc_file": cyto_path.name,
                "puncta_file": puncta_path.name,
                "intensity_file": intensity_path.name,
                "intensity_channel": intensity_channel,
                "nucleus_label": lab,
                "centroid_y": cy,
                "centroid_x": cx,
                "num_nuc_pixels": n_pix,
                "num_puncta_pixels": n_puncta_px,
                "has_puncta": has_puncta,
                "puncta_area_in_nuc": n_puncta_px,
                "puncta_density": n_puncta_px / n_pix,
                "nuc_mean_raw": nuc_mean_raw,
                "sat_frac_nuc": sat_frac,
            })

        # Triptych
        if make_triptychs and triptych_out_dir is not None:
            trip_path = triptych_out_dir / f"{location}_triptych.png"
            make_triptych(
                img=auto_lut_clip(img),
                img_puncta=auto_lut_clip(img_puncta),
                labels=labels,
                puncta_mask=puncta_clean,
                has_puncta_map=has_puncta_map,
                centroid_map=centroid_map,
                out_path=trip_path,
            )

        if idx % 10 == 0 or idx == len(nuc_files):
            print(f"  Processed {idx}/{len(nuc_files)} images")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Per-nucleus puncta analysis with intensity metrics."
    )
    parser.add_argument("--nuc-dir", required=True,
                        help="Folder with nucleus masks")
    parser.add_argument("--puncta-dir", required=True,
                        help="Folder with puncta masks")
    parser.add_argument("--intensity-dir", required=True,
                        help="Folder with raw OME-TIFF images")
    parser.add_argument("--out-csv", required=True,
                        help="Path to output CSV file")
    parser.add_argument("--min-puncta-area", type=int, default=5,
                        help="Min puncta pixels to call has_puncta=1 (default: 5)")
    parser.add_argument("--puncta-open-radius", type=int, default=1,
                        help="Morphological opening radius for puncta (default: 1)")
    parser.add_argument("--make-triptychs", action="store_true",
                        help="Generate QC triptych PNGs")
    parser.add_argument("--triptych-out-dir", type=str, default=None,
                        help="Output folder for triptych PNGs")
    parser.add_argument("--intensity-channel", type=int, default=2,
                        help="Channel for nucleus intensity (default: 2)")
    parser.add_argument("--puncta-channel", type=int, default=1,
                        help="Channel for puncta intensity (default: 1)")

    args = parser.parse_args()
    main(**vars(args))
