#!/usr/bin/env python3
"""
mean_intensity_and_puncta.py  (Csat-ready + triptychs + channel selection)

Per-cell summary for cytoplasm / puncta images:

- Uses a user-specified intensity channel from OME-TIFF (e.g., channel 1 = whole cell / cytoplasm).
- Computes per-cell mean cytoplasm intensity (raw) over the cytoplasm mask.
- Calls puncta presence per cell using a cleaned puncta mask and (optionally) an
  eroded cell interior, with a minimum puncta area threshold.
- Tracks saturation fraction, puncta area, and puncta density.
- (Optional) Generates triptych PNGs per image:
    1) cytoplasm channel raw intensity
    2) cytoplasm labels with cell numbers (red if puncta+, white if puncta-)
    3) puncta overlay on cytoplasm channel
"""

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
import tifffile as tiff
from skimage import measure, morphology
import matplotlib.pyplot as plt



def percentile_norm(img2d, p1=1, p99=99):
    lo, hi = np.percentile(img2d, (p1, p99))
    if hi <= lo:
        return np.zeros_like(img2d, dtype=np.uint8)
    x = np.clip((img2d - lo) / (hi - lo), 0, 1)
    return x.astype(np.float32)


def auto_lut_clip(img, low_percentile=2.0, high_percentile=99.8):
    """
    Apply viewer-style LUT clipping:
    - Values below low_percentile -> set to 0
    - Values above high_percentile -> set to 1
    - Everything else scaled linearly between 0 and 1
    (no global stretching of full dynamic range)
    """
    img = img.astype(np.float32)
    lo = np.percentile(img, low_percentile)
    hi = np.percentile(img, high_percentile)

    img_clipped = np.clip(img, lo, hi)
    img_clipped = (img_clipped - lo) / (hi - lo + 1e-8)
    img_clipped[img < lo] = 0.0  # fully black background
    return img_clipped.astype(np.float32)


def parse_location(path: Path) -> str:
    """
    Try to extract image location token from the path.

    Examples:
        ".../XYPos/0_Z004.ome_cyto3_masks.tif" -> "0_Z004"
        If that fails, try generic "<digits>_Z<digits>" pattern.
        Fallback: stem of the filename.
    """
    s = str(path)
    m = re.search(r"XYPos[/\\]([^.]*)\.ome", s)
    if m:
        return m.group(1)

    m = re.search(r"(\d+_Z\d+)", s)
    if m:
        return m.group(1)

    return path.stem


def ensure_2d(arr: np.ndarray) -> np.ndarray:
    """
    Reduce an array to 2D (Y, X) by repeatedly taking arr[0] if there are
    extra leading dimensions (e.g., Z).
    NOTE: This should only be used when there is no explicit channel dimension.
    """
    out = arr
    while out.ndim > 2:
        out = out[0]
    return out


def load_intensity_image(path: Path, channel: int) -> np.ndarray:
    """
    Load an OME-TIFF and return a 2D cytoplasm/whole-cell channel.

    Assumes typical shapes:
        (C, Y, X)
        (Z, C, Y, X)
        (Y, X)  # already 2D

    If a channel axis is present, uses the given 'channel' index.
    """
    arr = tiff.imread(path)
    arr = np.asarray(arr)

    if arr.ndim == 2:
        # Already 2D (Y, X)
        return arr

    if arr.ndim == 3:
        # Ambiguous: assume (C, Y, X)
        if channel >= arr.shape[0]:
            raise ValueError(f"Requested channel {channel}, but array has shape {arr.shape}")
        return arr[channel]

    if arr.ndim == 4:
        # Assume (Z, C, Y, X). Take Z=0 for now; adjust if needed.
        if channel >= arr.shape[1]:
            raise ValueError(f"Requested channel {channel}, but array has shape {arr.shape}")
        return arr[0, channel]

    # Fallback: just squeeze down, but this shouldn't really happen
    return ensure_2d(arr)


def get_cyto_labels(cyto: np.ndarray) -> np.ndarray:
    """
    Ensure we have a labeled cytoplasm mask:
    - If it's binary-like (0 and 1/other single value), label connected components.
    - Otherwise assume it's already a label image.
    """
    vals = np.unique(cyto)
    if len(vals) <= 2 and 0 in vals:
        return measure.label(cyto > 0, connectivity=1)
    else:
        return cyto.astype(int)


def make_triptych(img, img_puncta, labels, puncta_mask, has_puncta_map, centroid_map, out_path):
    """
    Create a 1x3 triptych:
      [0] nucleus channel LUT-normalized intensity (grayscale)
      [1] label image with nucleus numbers (red if has_puncta=1, white otherwise)
      [2] nucleus channel with puncta overlay in red
      [3] puncta channel LUT-normalized intensity (grayscale)

    Saves to out_path as PNG.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(12, 4))

    cmap = plt.cm.get_cmap("tab20").copy()
    cmap.set_bad(color="black")
    # Panel 1: normalized nucleus channel
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Nucleus channel (LUT-normalized)")
    axes[0].axis("off")

    # Panel 2: labels + numbers
    labels_2 = np.ma.masked_where(labels == 0, labels)
    axes[1].imshow(labels_2, cmap=cmap)
    axes[1].set_title("Nucleus labels\n(red = puncta+, white = puncta-)")
    axes[1].axis("off")

    for lab, (cy, cx) in centroid_map.items():
        if lab == 0:
            continue
        has_puncta = has_puncta_map.get(lab, 0)
        txt_color = "red" if has_puncta == 1 else "white"
        axes[1].text(
            cx,
            cy,
            str(lab),
            color=txt_color,
            ha="center",
            va="center",
            fontsize=6,
            weight="bold",
        )

    # Panel 3: puncta overlay on cytoplasm channel
    axes[2].imshow(img, cmap="gray")
    masked = np.ma.masked_where(puncta_mask == 0, puncta_mask)
    axes[2].imshow(masked, cmap=cmap, alpha=0.5)
    axes[2].set_title("Puncta overlay")
    axes[2].axis("off")

    # Panel 4: puncta channel
    axes[3].imshow(img_puncta, cmap="gray")
    axes[3].set_title("Puncta channel (LUT-normalized)")
    axes[3].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def find_matching_file(root: Path, location: str, kind: str) -> Path | None:
    """
    Find a file under `root` that corresponds to the given location token.

    Filenames look like:
        .../XYPos:114_Z005.ome.tif

    so we want to match the segment:

        :114_Z005.ome.tif
    or
        /114_Z005.ome.tif
    or
        \114_Z005.ome.tif
    """
    candidates = list(root.rglob(f"*{location}*"))
    if not candidates:
        print(f"[WARN] No {kind} file found at all for location {location}")
        return None

    # allow :, /, or \ before the location
    pat = re.compile(rf"(?:[:\\/]){re.escape(location)}\.ome\.tif$", re.IGNORECASE)

    strict = [p for p in candidates if pat.search(str(p))]

    if not strict:
        print(f"[WARN] No strict {kind} match for location {location}; falling back to loose match:")
        print("       ", candidates[0])
        return candidates[0]

    if len(strict) > 1:
        print(f"[WARN] Multiple strict {kind} matches for location {location}; using first:")
        for p in strict:
            print("       ", p)

    return strict[0]


def build_location_map(root: Path, kind: str) -> dict[str, Path]:
    """
    Build a mapping {location_token -> Path} for all .tif files under `root`.

    Uses parse_location() to extract tokens like "0_Z005", "10_Z005", etc.
    Warns if the same location appears more than once.
    """
    mapping: dict[str, Path] = {}
    for p in root.rglob("*.tif"):
        loc = parse_location(p)
        if loc in mapping:
            print(f"[WARN] {kind}: duplicate location {loc}")
            print(f"       existing -> {mapping[loc]}")
            print(f"       new      -> {p}")
        else:
            mapping[loc] = p
    print(f"[INFO] {kind}: indexed {len(mapping)} locations from {root}")
    return mapping


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

    for cyto_path in sorted(nuc_dir.glob("*.tif")):
        location = parse_location(cyto_path)

        # Match puncta and intensity by location token
        puncta_path = puncta_map.get(location)
        if puncta_path is None:
            print(f"[WARN] No puncta file for {cyto_path.name} (location {location}), skipping")
            continue

        intensity_path = intensity_map.get(location)
        if intensity_path is None:
            print(f"[WARN] No intensity file for {intensity_path.name} (location {location}), skipping")
            continue

        # Load masks
        nucleus = ensure_2d(tiff.imread(cyto_path))
        puncta = ensure_2d(tiff.imread(puncta_path))

        # Load nucleus intensity channel from OME-TIFF
        img = load_intensity_image(intensity_path, channel=intensity_channel)
        img_puncta = load_intensity_image(intensity_path, channel=puncta_channel)

        if nucleus.shape != img.shape:
            raise ValueError(
                f"Shape mismatch between cyto {nucleus.shape} and intensity {img.shape} "
                f"for {cyto_path.name}"
            )
        if puncta.shape != nucleus.shape:
            raise ValueError(
                f"Shape mismatch between cyto {nucleus.shape} and puncta {puncta.shape} "
                f"for {cyto_path.name}"
            )

        labels = get_cyto_labels(nucleus)
        max_label = int(labels.max())
        if max_label == 0:
            print(f"[WARN] No cells found in {cyto_path.name}, skipping")
            continue

        # Regionprops once per image (for centroids, etc.)
        props = measure.regionprops(labels)
        centroid_map = {p.label: p.centroid for p in props}

        # Clean puncta once per image (remove tiny specks)
        if puncta_open_radius > 0:
            se_puncta = morphology.disk(puncta_open_radius)
            puncta_clean = morphology.opening(puncta > 0, se_puncta)
        else:
            puncta_clean = puncta > 0

        # Determine max intensity for saturation detection (if integer type)
        if np.issubdtype(img.dtype, np.integer):
            imax = np.iinfo(img.dtype).max
        else:
            imax = np.nan  # no obvious saturation bound

        # Per-image map from label -> has_puncta (for triptych coloring)
        has_puncta_map = {}

        for lab in range(1, max_label + 1):
            nucleus_mask = labels == lab
            n_pix = int(nucleus_mask.sum())
            if n_pix == 0:
                continue

            # ----- Saturation fraction in this cell -----
            if np.isfinite(imax):
                sat_frac_nuc = float((img[nucleus_mask] >= imax).mean())
            else:
                sat_frac_nuc = 0.0

            # ----- Puncta overlap (cleaned puncta, eroded cell core) -----

            puncta_inside = np.logical_and(nucleus_mask, puncta_clean)
            num_puncta_pixels = int(puncta_inside.sum())

            # Minimum area requirement
            min_area_thresh = min_puncta_area
            has_puncta = int(num_puncta_pixels >= min_area_thresh)
            has_puncta_map[lab] = has_puncta

            # ----- Intensities (raw) over nucleus mask -----
            nuc_vals_raw = img[nucleus_mask].astype(np.float32)
            nuc_mean_raw = float(nuc_vals_raw.mean())

            # Puncta morphology metrics
            puncta_area_in_nuc = num_puncta_pixels
            puncta_density = float(num_puncta_pixels / n_pix) if n_pix > 0 else 0.0

            # Centroid for this nucleus
            cy, cx = centroid_map.get(lab, (np.nan, np.nan))

            rows.append(
                {
                    "image_location": location,
                    "nuc_file": cyto_path.name,
                    "puncta_file": puncta_path.name,
                    "intensity_file": intensity_path.name,
                    "intensity_channel": intensity_channel,
                    "nucleus_label": lab,
                    "centroid_y": cy,
                    "centroid_x": cx,
                    "num_nuc_pixels": n_pix,
                    "num_puncta_pixels": num_puncta_pixels,
                    "has_puncta": has_puncta,
                    "puncta_area_in_nuc": puncta_area_in_nuc,
                    "puncta_density": puncta_density,
                    "nuc_mean_raw": nuc_mean_raw,
                    "sat_frac_nuc": sat_frac_nuc,
                }
            )

        # ---- After finishing all cells in this image: make triptych, if requested ----
        if make_triptychs and triptych_out_dir is not None:
            trip_name = f"{location}_triptych.png"
            trip_path = triptych_out_dir / trip_name
            make_triptych(
                img=auto_lut_clip(img),
                img_puncta = auto_lut_clip(img_puncta),
                labels=labels,
                puncta_mask=puncta_clean,
                has_puncta_map=has_puncta_map,
                centroid_map=centroid_map,
                out_path=trip_path,
            )

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nuc-dir", required=True, help="Folder with nucleus masks")
    parser.add_argument("--puncta-dir", required=True, help="Folder with puncta masks")
    parser.add_argument("--intensity-dir", required=True, help="Folder with raw OME-TIFF images")
    parser.add_argument("--out-csv", required=True, help="Path to output CSV file")

    parser.add_argument("--min-puncta-area", type=int, default=5,
                        help="Minimum puncta area (pixels) in the cell to call has_puncta=1")
    parser.add_argument("--puncta-open-radius", type=int, default=1,
                        help="Radius for opening puncta mask (0 = no opening)")
    parser.add_argument("--make-triptychs", action="store_true",
                        help="If set, generate triptych PNGs for each image")
    parser.add_argument("--triptych-out-dir", type=str, default=None,
                        help="Folder to save triptych PNGs (required if --make-triptychs)")

    parser.add_argument("--intensity-channel", type=int, default=2,
                        help="Channel index in OME-TIFF to use for nucleus intensity (e.g., 2 = nucleus)")
    parser.add_argument("--puncta-channel", type=int, default=1,
                        help="Channel index in OME-TIFF to use for puncta intensity (e.g., 1 = gfp)")

    args = parser.parse_args()
    main(**vars(args))
