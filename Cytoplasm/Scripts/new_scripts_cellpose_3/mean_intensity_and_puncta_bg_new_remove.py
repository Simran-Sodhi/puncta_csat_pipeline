#!/usr/bin/env python3
"""
summarize_puncta_by_cell.py  (local-background + Csat-ready + triptychs + channel selection)

Per-cell summary for cytoplasm / puncta images:

- Uses a user-specified intensity channel from OME-TIFF (e.g., channel 1 = whole cell / cytoplasm).
- Computes per-cell mean cytoplasm intensity (raw, global bg-sub, local bg-sub) over the cytoplasm mask.
- Estimates local background using an annulus (ring) around each cell:
    * excludes (dilated) other cells
    * excludes puncta pixels
- Calls puncta presence per cell using a cleaned puncta mask and (optionally) an
  eroded cell interior, with a minimum puncta area threshold.
- Tracks saturation fraction, puncta area, and puncta density.
- Exports a canonical "intensity_for_cs" column = hybrid bg-sub mean,
  where hybrid background = local if sane, else global.
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
    # return (x * 255).astype(np.uint8)
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

    # Fallback: just squeeze down, but this shouldn't really happen for your data
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
      [0] cytoplasm channel raw intensity (grayscale)
      [1] label image with cell numbers (red if has_puncta=1, white otherwise)
      [2] cytoplasm channel with puncta overlay in red

    Saves to out_path as PNG.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(12, 4))

    cmap = plt.cm.get_cmap("tab20").copy()
    cmap.set_bad(color="black")
    # Panel 1: normalized cytoplasm channel
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Cytoplasm channel (LUT-normalized)")
    axes[0].axis("off")

    # Panel 2: labels + numbers
    labels_2 = np.ma.masked_where(labels == 0, labels)
    axes[1].imshow(labels_2, cmap=cmap)
    axes[1].set_title("Cytoplasm labels\n(red = puncta+, white = puncta-)")
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

    Your filenames look like:
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
    cyto_dir,
    puncta_dir,
    intensity_dir,
    cell_dir,
    out_csv,
    min_puncta_pixels=1,
    min_puncta_area=5,
    bg_stat="median",
    norm_percentile=99.0,
    calib_a=None,
    calib_b=None,
    ring_inner_px=3,
    ring_outer_px=8,
    min_ring_pixels=200,
    other_cell_exclusion_px=1,
    puncta_open_radius=1,
    cell_erosion_px=1,
    make_triptychs=False,
    triptych_out_dir=None,
    intensity_channel=1,
    puncta_channel =2,
    local_bg_max_delta=800.0,  # NEW: max allowed (local_bg - global_bg) to accept local
):
    cyto_dir = Path(cyto_dir)
    puncta_dir = Path(puncta_dir)
    intensity_dir = Path(intensity_dir)
    cell_dir = Path(cell_dir)
    triptych_out_dir = Path(triptych_out_dir) if triptych_out_dir else None

    puncta_map = build_location_map(puncta_dir, "puncta")
    intensity_map = build_location_map(intensity_dir, "intensity")
    cell_map = build_location_map(cell_dir, "cyto")
    rows = []

    for cyto_path in sorted(cyto_dir.glob("*.tif")):
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

        cell_path = cell_map.get(location)
        if cell_path is None:
            print(f"[WARN] No whole cell mask file for {cell_path.name} (location {location}), skipping")
            continue

        # print(f"[DEBUG] location={location}")
        # print(f"        cyto     -> {cyto_path}")
        # print(f"        puncta   -> {puncta_path}")
        # print(f"        intensity-> {intensity_path}")
        # print(f"        whole cell-> {cell_path}")

        # Load masks
        cyto = ensure_2d(tiff.imread(cyto_path))  # this is your cytoplasm label mask
        cell = ensure_2d(tiff.imread(cell_path))
        puncta = ensure_2d(tiff.imread(puncta_path))

        # Load cytoplasm intensity channel from OME-TIFF
        img = load_intensity_image(intensity_path, channel=intensity_channel)
        img_puncta = load_intensity_image(intensity_path, channel=puncta_channel)

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
        if max_label == 0:
            print(f"[WARN] No cells found in {cyto_path.name}, skipping")
            continue

        cell_labels = get_cyto_labels(cell)

        # Regionprops once per image (for centroids, etc.)
        props = measure.regionprops(labels)
        centroid_map = {p.label: p.centroid for p in props}

        # ----- Global background from OUTSIDE all cells -----
        noncell_mask = cell_labels == 0
        if noncell_mask.any():
            noncell_vals = img[noncell_mask]
            if bg_stat == "median":
                img_bg_global = float(np.median(noncell_vals))
            else:
                img_bg_global = float(np.mean(noncell_vals))
        else:
            img_bg_global = float(np.median(img))

        # ----- Image P99 scale, based on bg-sub image (>0 only) -----
        img_bgsub_global = img.astype(np.float32) - img_bg_global
        pos_vals = img_bgsub_global[img_bgsub_global > 0]
        p_norm = float(np.percentile(pos_vals, norm_percentile)) if pos_vals.size else 1.0

        # Clean puncta once per image (remove tiny specks)
        if puncta_open_radius > 0:
            se_puncta = morphology.disk(puncta_open_radius)
            puncta_clean = morphology.opening(puncta > 0, se_puncta)
        else:
            puncta_clean = puncta > 0

        # Structuring elements for ring building and cell erosion
        selem_inner = morphology.disk(ring_inner_px)
        selem_outer = morphology.disk(ring_outer_px)
        selem_cell_erosion = morphology.disk(cell_erosion_px) if cell_erosion_px > 0 else None

        # Dilated "other cells" mask to exclude from background
        if other_cell_exclusion_px > 0:
            se_other = morphology.disk(other_cell_exclusion_px)
            other_cells_mask = morphology.binary_dilation(labels > 0, se_other)
        else:
            other_cells_mask = labels > 0

        # Determine max intensity for saturation detection (if integer type)
        if np.issubdtype(img.dtype, np.integer):
            imax = np.iinfo(img.dtype).max
        else:
            imax = np.nan  # no obvious saturation bound

        # Per-image map from label -> has_puncta (for triptych coloring)
        has_puncta_map = {}

        for lab in range(1, max_label + 1):
            cell_mask = labels == lab
            n_pix = int(cell_mask.sum())
            if n_pix == 0:
                continue

            # ----- Saturation fraction in this cell -----
            if np.isfinite(imax):
                sat_frac_cell = float((img[cell_mask] >= imax).mean())
            else:
                sat_frac_cell = 0.0

            # ----- Puncta overlap (cleaned puncta, eroded cell core) -----
            if selem_cell_erosion is not None:
                cell_mask_core = morphology.binary_erosion(cell_mask, selem_cell_erosion)
                # If erosion removes everything (tiny cell), fall back
                if not cell_mask_core.any():
                    cell_mask_core = cell_mask.copy()
            else:
                cell_mask_core = cell_mask

            puncta_inside = np.logical_and(cell_mask_core, puncta_clean)
            num_puncta_pixels = int(puncta_inside.sum())

            # Minimum area requirement
            min_area_thresh = max(min_puncta_pixels, min_puncta_area)
            has_puncta = int(num_puncta_pixels >= min_area_thresh)
            has_puncta_map[lab] = has_puncta

            # ----- Intensities (raw & global bg-sub) over cytoplasm mask -----
            cyto_vals_raw = img[cell_mask].astype(np.float32)
            cyto_mean_raw = float(cyto_vals_raw.mean())
            cyto_mean_bgsub_global = float((cyto_vals_raw - img_bg_global).mean())
            cyto_mean_norm_global = float(cyto_mean_bgsub_global / p_norm) if p_norm > 0 else 0.0

            # ----- Local ring background around this cell -----
            # ring = dilate by outer minus dilate by inner
            dil_outer = morphology.binary_dilation(cell_mask, selem_outer)
            dil_inner = morphology.binary_dilation(cell_mask, selem_inner)
            ring_mask = np.logical_and(dil_outer, np.logical_not(dil_inner))

            # exclude cells (with safety margin) and puncta from background estimation
            ring_mask = np.logical_and(ring_mask, np.logical_not(other_cells_mask))
            ring_mask = np.logical_and(ring_mask, np.logical_not(puncta_clean))

            local_bg_pixels_used = int(ring_mask.sum())

            # --- hybrid local/global background decision ---
            local_bg_raw = None
            local_bg_used = img_bg_global
            local_bg_source = "global"  # default

            # if too few ring pixels, expand ring outward once
            if local_bg_pixels_used < min_ring_pixels:
                selem_outer2 = morphology.disk(
                    max(ring_outer_px + ring_inner_px, ring_outer_px + 3)
                )
                dil_outer2 = morphology.binary_dilation(cell_mask, selem_outer2)
                ring_mask2 = np.logical_and(dil_outer2, np.logical_not(dil_outer))
                ring_mask2 = np.logical_and(ring_mask2, np.logical_not(other_cells_mask))
                ring_mask2 = np.logical_and(ring_mask2, np.logical_not(puncta_clean))

                if ring_mask2.sum() > ring_mask.sum():
                    ring_mask = ring_mask2

                local_bg_pixels_used = int(ring_mask.sum())

            if local_bg_pixels_used >= max(50, min_ring_pixels // 4):
                ring_vals = img[ring_mask].astype(np.float32)
                if bg_stat == "median":
                    local_bg_raw = float(np.median(ring_vals))
                else:
                    local_bg_raw = float(np.mean(ring_vals))

                # Hybrid sanity check: local if sane, else global
                if local_bg_raw - img_bg_global <= float(local_bg_max_delta):
                    local_bg_used = local_bg_raw
                    local_bg_source = "local"
                else:
                    local_bg_used = img_bg_global
                    local_bg_source = "global_fallback"
            else:
                # not enough pixels; use global
                local_bg_used = img_bg_global
                local_bg_source = "global_fallback"

            # Local-bg-sub mean (using raw local estimate if available, for debugging)
            if local_bg_raw is not None:
                cyto_mean_bgsub_local = float((cyto_vals_raw - local_bg_raw).mean())
            else:
                cyto_mean_bgsub_local = float((cyto_vals_raw - img_bg_global).mean())
            cyto_mean_norm_local = float(cyto_mean_bgsub_local / p_norm) if p_norm > 0 else 0.0

            # Hybrid-bg-sub mean (this is what we use for C_sat)
            cyto_mean_bgsub_hybrid = float((cyto_vals_raw - local_bg_used).mean())

            # Optional calibration using hybrid bg-sub intensity
            if calib_a is not None and calib_b is not None and calib_a != 0:
                conc_estimate_hybrid = (cyto_mean_bgsub_hybrid - float(calib_b)) / float(calib_a)
                conc_estimate_hybrid = float(max(conc_estimate_hybrid, 0.0))
            else:
                conc_estimate_hybrid = None

            # Puncta morphology metrics
            puncta_area_in_cell = num_puncta_pixels
            puncta_density = float(num_puncta_pixels / n_pix) if n_pix > 0 else 0.0

            # Canonical column for C_sat fitting: hybrid bg-sub
            intensity_for_cs = cyto_mean_bgsub_hybrid

            # Centroid for this cell
            cy, cx = centroid_map.get(lab, (np.nan, np.nan))

            rows.append(
                {
                    "image_location": location,
                    "cyto_file": cyto_path.name,
                    "puncta_file": puncta_path.name,
                    "intensity_file": intensity_path.name,
                    "intensity_channel": intensity_channel,
                    "cell_label": lab,
                    "centroid_y": cy,
                    "centroid_x": cx,
                    "num_cyto_pixels": n_pix,
                    "num_puncta_pixels": num_puncta_pixels,
                    "has_puncta": has_puncta,
                    "puncta_area_in_cell": puncta_area_in_cell,
                    "puncta_density": puncta_density,
                    "background_method": bg_stat,
                    "background_value_global": img_bg_global,
                    "norm_percentile": norm_percentile,
                    "norm_value_pXX": p_norm,
                    "local_bg_raw": local_bg_raw,
                    "local_bg_used": local_bg_used,
                    "local_bg_source": local_bg_source,
                    "local_bg_pixels_used": local_bg_pixels_used,
                    "cyto_mean_raw": cyto_mean_raw,
                    "cyto_mean_bgsub": cyto_mean_bgsub_global,
                    "cyto_mean_norm": cyto_mean_norm_global,
                    "cyto_mean_bgsub_local": cyto_mean_bgsub_local,
                    "cyto_mean_norm_local": cyto_mean_norm_local,
                    "cyto_mean_bgsub_hybrid": cyto_mean_bgsub_hybrid,
                    "sat_frac_cell": sat_frac_cell,
                    "conc_estimate_hybrid": conc_estimate_hybrid,
                    "intensity_for_cs": intensity_for_cs,
                }
            )

        # ---- After finishing all cells in this image: make triptych, if requested ----
        if make_triptychs and triptych_out_dir is not None:
            trip_name = f"{location}_triptych.png"
            trip_path = triptych_out_dir / trip_name
            make_triptych(
                img=percentile_norm(img),
                img_puncta = percentile_norm(img_puncta),
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
    parser.add_argument("--cyto-dir", required=True, help="Folder with cytoplasm masks")
    parser.add_argument("--puncta-dir", required=True, help="Folder with puncta masks")
    parser.add_argument("--intensity-dir", required=True, help="Folder with raw OME-TIFF images")
    parser.add_argument("--cell-dir", required=True, help="Folder with raw OME-TIFF images")
    parser.add_argument("--out-csv", required=True, help="Path to output CSV file")

    parser.add_argument("--min-puncta-pixels", type=int, default=1,
                        help="Legacy: minimum puncta pixels; combined with --min-puncta-area")
    parser.add_argument("--min-puncta-area", type=int, default=5,
                        help="Minimum puncta area (pixels) in the cell to call has_puncta=1")

    parser.add_argument("--bg-stat", choices=["median", "mean"], default="median",
                        help="Statistic for background estimation (global & local rings)")
    parser.add_argument("--norm-percentile", type=float, default=99.0,
                        help="Percentile of bg-sub image (>0) used for per-image normalization")

    parser.add_argument("--calib-a", type=float, default=None,
                        help="Slope a in I ≈ a*C + b (for optional concentration estimate)")
    parser.add_argument("--calib-b", type=float, default=None,
                        help="Intercept b in I ≈ a*C + b")

    parser.add_argument("--ring-inner-px", type=int, default=3,
                        help="Inner radius of local background ring (pixels)")
    parser.add_argument("--ring-outer-px", type=int, default=8,
                        help="Outer radius of local background ring (pixels)")
    parser.add_argument("--min-ring-pixels", type=int, default=200,
                        help="Minimum pixels needed to trust local background")

    parser.add_argument("--other-cell-exclusion-px", type=int, default=1,
                        help="Dilation radius (pixels) to exclude other cells from background rings")
    parser.add_argument("--puncta-open-radius", type=int, default=1,
                        help="Radius for opening puncta mask (0 = no opening)")
    parser.add_argument("--cell-erosion-px", type=int, default=1,
                        help="Erode cell mask before puncta overlap (0 = no erosion)")

    parser.add_argument("--make-triptychs", action="store_true",
                        help="If set, generate triptych PNGs for each image")
    parser.add_argument("--triptych-out-dir", type=str, default=None,
                        help="Folder to save triptych PNGs (required if --make-triptychs)")

    parser.add_argument("--intensity-channel", type=int, default=1,
                        help="Channel index in OME-TIFF to use for cytoplasm intensity (e.g., 1 = whole cell)")
    parser.add_argument("--puncta-channel", type=int, default=2,
                        help="Channel index in OME-TIFF to use for puncta intensity (e.g., 2 = gfp)")
    parser.add_argument("--local-bg-max-delta", type=float, default=800.0,
                        help="Max allowed (local_bg - global_bg) before falling back to global background")

    args = parser.parse_args()
    main(**vars(args))
