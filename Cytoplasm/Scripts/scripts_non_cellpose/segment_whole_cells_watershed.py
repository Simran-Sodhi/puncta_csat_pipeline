#!/usr/bin/env python3
"""
High-fidelity whole-cell segmentation for OME-TIFFs with:
  - Channel 0: nuclei (e.g., H2B)
  - Channel 1: cytoplasm/membrane (e.g., HOTag)

Pipeline:
  nuclei -> markers; cytoplasm -> edge/inverse-intensity cost; geodesic watershed.

Outputs (per OME-TIFF):
  - <name>_labels.tif          (uint16 label mask)
  - <name>_overlay.png         (optional RGB overlay of cyto with red boundaries)

Requires: numpy, tifffile, scikit-image, matplotlib (only for overlay), scipy
Install:
  pip install numpy tifffile scikit-image matplotlib scipy
"""

import argparse
from pathlib import Path
import numpy as np
import tifffile as tiff
from tifffile import TiffFile
from skimage import filters, exposure, morphology, measure, segmentation, util, feature
from scipy import ndimage as ndi

# --------------------------- Core segmentation ---------------------------

# def segment_cells_from_cyto_and_nuc(
#     cyto, nuc,
#     min_cell_area=600,
#     min_nuc_area=80,
#     hole_area=600,
#     clip_high_pct=99.8,
#     gaussian_sigma=1.0,
#     sobel_weight=0.65,
#     inv_intensity_weight=0.35,
#     sat_mask_erode=2,
#     boundary_smooth_disk=2,
# ):
#     """
#     Inputs:
#       cyto, nuc: 2D arrays (float or uint), same shape (Y, X).
#     Returns:
#       labels: int32 labeled mask (0 = background).
#     """

#     # --- 0) normalize to float [0,1] with gentle high clipping
#     def to_float01(x):
#         x = util.img_as_float32(x)
#         high = np.percentile(x, clip_high_pct)
#         x = np.clip(x, 0, high)
#         x = (x - x.min()) / (x.max() - x.min() + 1e-8)
#         return x

#     cy = to_float01(cyto)
#     nu = to_float01(nuc)

#     # --- 1) denoise + local contrast (cyto)
#     cy = filters.gaussian(cy, gaussian_sigma, preserve_range=True)
#     cy = exposure.equalize_adapthist(cy, clip_limit=0.01)

#     # enhance thin membranes / edges
#     selem = morphology.disk(25)
#     cy_enh = morphology.white_tophat(cy, selem)

#     # --- 2) protect saturated regions (prevents “flooding”)
#     sat = cyto >= np.percentile(cyto, 99.9)
#     if sat_mask_erode > 0:
#         sat = morphology.binary_erosion(sat, morphology.disk(sat_mask_erode))

#     # --- 3) seeds from nuclei
#     nu_s = filters.gaussian(nu, 1.0)
#     t = filters.threshold_otsu(nu_s)
#     nuc_bin = nu_s > t
#     nuc_bin = morphology.remove_small_objects(nuc_bin, min_nuc_area)
#     nuc_bin = morphology.binary_opening(nuc_bin, morphology.disk(1))

#     # split touching nuclei
#     dist = ndi.distance_transform_edt(nuc_bin)
#     peaks = feature.peak_local_max(
#         dist, labels=nuc_bin, footprint=np.ones((9, 9)), exclude_border=False
#     )
#     markers = np.zeros_like(dist, dtype=np.int32)
#     for i, (r, c) in enumerate(peaks, start=1):
#         markers[r, c] = i
#     if markers.max() == 0:
#         markers = measure.label(nuc_bin)

#     # --- 4) build geodesic cost from cyto
#     grad = filters.sobel(cy_enh)
#     grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)
#     inv_intensity = 1.0 - cy
#     inv_intensity = (inv_intensity - inv_intensity.min()) / (np.ptp(inv_intensity) + 1e-8)
#     cost = sobel_weight * grad + inv_intensity_weight * inv_intensity
#     cost = filters.gaussian(cost, 1.0)

#     # permissive mask where cells likely exist
#     try:
#         th_li = filters.threshold_li(cy)
#     except Exception:
#         th_li = np.percentile(cy, 60)
#     cy_mask = cy > th_li
#     cy_mask |= morphology.binary_dilation(nuc_bin, morphology.disk(10))
#     cy_mask &= ~sat

#     # --- 5) watershed
#     labels = segmentation.watershed(cost, markers=markers, mask=cy_mask)

#     # --- 6) clean and re-watershed for crisp borders
#     labels = measure.label(labels > 0)
#     props = measure.regionprops(labels)
#     small_ids = [p.label for p in props if p.area < min_cell_area]
#     if small_ids:
#         bad = np.isin(labels, small_ids)
#         labels[bad] = 0
#         labels = measure.label(labels > 0)

#     binm = labels > 0
#     binm = morphology.remove_small_holes(binm, area_threshold=hole_area)
#     binm = morphology.binary_closing(binm, morphology.disk(boundary_smooth_disk))

#     labels = segmentation.watershed(grad + 0.01, measure.label(nuc_bin), mask=binm)
#     return measure.label(labels, connectivity=1).astype(np.int32)
from skimage.filters import threshold_sauvola

def segment_cells_from_cyto_and_nuc(
    cyto, nuc,
    # --- core thresholds ---
    min_cell_area=900,
    min_nuc_area=120,
    hole_area=1200,
    clip_high_pct=99.7,
    gaussian_sigma=1.0,
    sobel_weight=0.60,
    inv_intensity_weight=0.55,
    # --- biological constraints ---
    near_nuc_dist=40,         # px, geodesic growth radius
    min_seed_overlap_px=20,   # px of overlap with nucleus seed
    cyto_intensity_q=0.55,    # mean(cyto) must be >= this quantile of near-nucleus cyto
    # --- misc ---
    sat_mask_erode=1,
    boundary_smooth_disk=2,
):
    import numpy as np
    from skimage import filters, exposure, morphology, measure, segmentation, util, feature
    from scipy import ndimage as ndi

    def to_float01(x):
        x = util.img_as_float32(x)
        high = np.percentile(x, clip_high_pct)
        x = np.clip(x, 0, high)
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        return x

    def geodesic_mask(seed_bin, cost_img, max_dist):
        reach = seed_bin.copy()
        front = seed_bin.copy()
        barrier = cost_img > np.percentile(cost_img, 90)
        for _ in range(int(max_dist)):
            front = morphology.binary_dilation(front, morphology.disk(1))
            front &= ~barrier
            new = front & ~reach
            if not new.any(): break
            reach |= new
        return reach

    cy_raw = util.img_as_float32(cyto)
    cy = to_float01(cyto); nu = to_float01(nuc)
    cy_smooth = filters.gaussian(cy, gaussian_sigma, preserve_range=True)
    cy_eq = exposure.equalize_adapthist(cy_smooth, clip_limit=0.01)
    cy_edges = filters.gaussian(cy_eq, 0.8)

    sat = cy_raw >= np.percentile(cy_raw, 99.9)
    if sat_mask_erode > 0:
        sat = morphology.binary_erosion(sat, morphology.disk(sat_mask_erode))

    # nuclei → seeds
    nu_s = filters.gaussian(nu, 1.0)
    t = filters.threshold_otsu(nu_s)
    nuc_bin = morphology.remove_small_objects(nu_s > t, min_nuc_area)
    nuc_bin = morphology.binary_opening(nuc_bin, morphology.disk(1))
    dist = ndi.distance_transform_edt(nuc_bin)
    peaks = feature.peak_local_max(dist, labels=nuc_bin, min_distance=6,
                                   footprint=np.ones((7,7)), exclude_border=False)
    markers = np.zeros_like(dist, dtype=np.int32)
    for i,(r,c) in enumerate(peaks, start=1): markers[r,c] = i
    seeds = segmentation.watershed(-dist, markers, mask=nuc_bin) if markers.max()>0 else measure.label(nuc_bin)
    seeds = measure.label(morphology.remove_small_objects(seeds>0, min_nuc_area))

    # support mask: Sauvola (local) ∧ near-nucleus geodesic
    win = 51 if min(cy.shape)>256 else 31
    sau = threshold_sauvola(cy_smooth, window_size=win, k=0.2)
    cy_mask_local = cy_smooth > sau
    cy_mask_local = morphology.remove_small_objects(cy_mask_local, 400)
    cy_mask_local = morphology.binary_opening(cy_mask_local, morphology.disk(2))

    tmp = filters.scharr(cy_edges); tmp = (tmp-tmp.min())/(tmp.max()-tmp.min()+1e-8)
    near_nuc = geodesic_mask(seeds>0, tmp, near_nuc_dist)

    cy_mask = cy_mask_local & near_nuc  # ← intersection (kills bubbles)

    # cost
    grad = filters.scharr(cy_edges); grad = (grad-grad.min())/(grad.max()-grad.min()+1e-8)
    invI = 1.0 - cy_smooth; invI = (invI-invI.min())/(np.ptp(invI)+1e-8)
    cost = sobel_weight*grad + inv_intensity_weight*invI + 0.15*sat.astype(float)
    cost = filters.gaussian(cost, 1.0)

    labels = segmentation.watershed(cost, markers=seeds, mask=cy_mask)
    labels = measure.label(labels>0)

    # post-filters: keep only nucleus-overlapping & bright-enough cells
    seed_mask = seeds>0
    overlap = np.bincount(labels[seed_mask], minlength=labels.max()+1)
    keep = np.zeros(labels.max()+1, bool); keep[overlap >= min_seed_overlap_px] = True
    labels[~keep[labels]] = 0; labels = measure.label(labels>0)

    ref_band = morphology.binary_dilation(seed_mask, morphology.disk(near_nuc_dist))
    ref_vals = cy[ref_band]
    if ref_vals.size:
        floor = np.quantile(ref_vals, cyto_intensity_q)
        for p in measure.regionprops(labels, intensity_image=cy):
            if p.mean_intensity < floor or p.area < min_cell_area:
                labels[labels==p.label] = 0
        labels = measure.label(labels>0)

    # fill holes + smooth + gentle re-watershed
    binm = labels>0
    binm = morphology.remove_small_holes(binm, area_threshold=hole_area)
    binm = morphology.binary_closing(binm, morphology.disk(boundary_smooth_disk))
    labels = segmentation.watershed(grad+0.02, seeds, mask=binm)

    return measure.label(labels, connectivity=1).astype(np.int32)


# --------------------------- I/O & orchestration ---------------------------

def parse_axes_from_ome(path: Path):
    """Return OME axes string if present, else None."""
    try:
        with TiffFile(path) as tf:
            if tf.ome_metadata is None:
                return None
            omexml = tf.ome_metadata
            # very light scrape for 'DimensionOrder' (e.g., 'TCZYX', 'CZYX', etc.)
            # tifffile also exposes series[0].axes (preferred)
            if tf.series and getattr(tf.series[0], "axes", None):
                return tf.series[0].axes
    except Exception:
        return None
    return None


def load_channels(path: Path, z_mode: str = "max"):
    """
    Load an OME-TIFF and return (cyto, nuc) 2D arrays ready for segmentation.
    Supports:
      - CYX
      - ZCYX / CZYX / ZYXC variants (via axes info; falls back to best guess)
    z_mode: 'max' (default) for max-projection, or 'slice' for middle Z slice.
    """
    arr = tiff.imread(path)
    axes = parse_axes_from_ome(path)

    def project_z(vol):
        if z_mode == "slice":
            mid = vol.shape[0] // 2
            return vol[mid]
        else:
            return np.max(vol, axis=0)

    # Heuristics if axes missing
    if axes is None:
        # Try to infer by ndim and sizes
        if arr.ndim == 3:
            # Assume CYX (common for 2D multichannel)
            C, Y, X = arr.shape
            cyto = arr[1]
            nuc = arr[0]
            return cyto, nuc
        elif arr.ndim == 4:
            # Guess ZCYX or CZYX
            if arr.shape[0] in (2, 3, 4):  # likely CZYX
                C, Z, Y, X = arr.shape
                cyto = project_z(arr[1])
                nuc = project_z(arr[0])
                return cyto, nuc
            else:  # likely ZCYX
                Z, C, Y, X = arr.shape
                cyto = project_z(arr[:, 1])
                nuc = project_z(arr[:, 0])
                return cyto, nuc
        else:
            raise ValueError(f"Unsupported shape for {path}: {arr.shape}")

    # Use axes to map
    # Normalize to a dict of axis -> index position
    axis_pos = {ax: i for i, ax in enumerate(axes)}
    # Ensure we have Y and X
    if "Y" not in axis_pos or "X" not in axis_pos:
        raise ValueError(f"Cannot find Y/X axes in {axes} for {path}")

    # Helper to index channels robustly
    def get_channel(img, chan_index):
        # Move channel axis to front for convenience
        c_axis = axis_pos.get("C", None)
        if c_axis is None:
            raise ValueError("No channel axis 'C' found; cannot split channels.")
        img_moved = np.moveaxis(img, c_axis, 0)  # C first
        return img_moved[chan_index]

    # Expand to consistent order
    if "Z" in axis_pos:
        # Bring Z to front, then C, then YX
        z_axis = axis_pos["Z"]
        imgZfirst = np.moveaxis(arr, z_axis, 0)  # Z first
        # Now extract channels on Z-first array:
        # To use get_channel, we need a view with C axis; recompute axis map post-move
        # Easiest: project per channel explicitly.
        # Build per-channel volumes:
        c_axis = axes.index("C")
        if c_axis < z_axis:
            # after moving Z, C index shifts if before Z
            c_axis_moved = c_axis
        else:
            c_axis_moved = c_axis - 1
        # Move C to second axis: (Z, C, ...)
        imgZC = np.moveaxis(imgZfirst, c_axis_moved, 1)

        cyto_vol = imgZC[:, 1]  # (Z, Y, X)
        nuc_vol = imgZC[:, 0]
        cyto = project_z(cyto_vol)
        nuc = project_z(nuc_vol)
        return cyto, nuc
    else:
        # No Z: just split channels
        cyto = get_channel(arr, 1)
        nuc = get_channel(arr, 0)
        return cyto, nuc


def save_overlay_png(cyto, labels, out_png_path: Path, dpi=200):
    """Save a quick RGB overlay with red boundaries on the cyto image."""
    from skimage.segmentation import find_boundaries
    import matplotlib.pyplot as plt

    cy_norm = (cyto - cyto.min()) / (cyto.max() - cyto.min() + 1e-8)
    rgb = np.dstack([cy_norm, cy_norm, cy_norm])
    bnd = find_boundaries(labels)
    rgb[bnd] = [1, 0, 0]

    plt.figure(figsize=(cyto.shape[1] / dpi, cyto.shape[0] / dpi), dpi=dpi)
    plt.imshow(rgb)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_png_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def process_file(path: Path, output_dir: Path, z_mode: str, save_overlays: bool, params: dict):
    cyto, nuc = load_channels(path, z_mode=z_mode)
    labels = segment_cells_from_cyto_and_nuc(cyto, nuc, **params)

    out_tif = output_dir / f"{path.stem}_labels.tif"
    tiff.imwrite(out_tif.as_posix(), labels.astype(np.uint16), photometric="minisblack")

    if save_overlays:
        out_png = output_dir / f"{path.stem}_overlay.png"
        save_overlay_png(cyto, labels, out_png)

    return out_tif


# --------------------------- CLI ---------------------------

def main():
    p = argparse.ArgumentParser(description="Whole-cell segmentation for OME-TIFFs (nuc=ch0, cyto=ch1).")
    p.add_argument("--input_dir", required=True, type=Path, help="Folder containing .ome.tif/.ome.tiff files.")
    p.add_argument("--output_dir", required=True, type=Path, help="Where to save masks/overlays.")
    p.add_argument("--z_mode", choices=["max", "slice"], default="max",
                   help="For Z-stacks: 'max' = max projection (default), 'slice' = middle slice.")
    p.add_argument("--save_overlays", action="store_true", help="Also save PNG overlays with red boundaries.")
    # Advanced knobs (you can usually keep defaults)
    p.add_argument("--min_cell_area", type=int, default=600)
    p.add_argument("--min_nuc_area", type=int, default=80)
    p.add_argument("--hole_area", type=int, default=600)
    p.add_argument("--clip_high_pct", type=float, default=99.8)
    p.add_argument("--gaussian_sigma", type=float, default=1.0)
    p.add_argument("--sobel_weight", type=float, default=0.65)
    p.add_argument("--inv_intensity_weight", type=float, default=0.35)
    p.add_argument("--sat_mask_erode", type=int, default=2)
    p.add_argument("--boundary_smooth_disk", type=int, default=2)
    p.add_argument("--near_nuc_dist", type=int, default=35)
    p.add_argument("--min_seed_overlap_px", type=int, default=15)
    p.add_argument("--cyto_intensity_q", type=float, default=0.55)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    params = dict(
        min_cell_area=args.min_cell_area,
        min_nuc_area=args.min_nuc_area,
        hole_area=args.hole_area,
        clip_high_pct=args.clip_high_pct,
        gaussian_sigma=args.gaussian_sigma,
        sobel_weight=args.sobel_weight,
        inv_intensity_weight=args.inv_intensity_weight,
        sat_mask_erode=args.sat_mask_erode,
        boundary_smooth_disk=args.boundary_smooth_disk,
        near_nuc_dist=args.near_nuc_dist,
        min_seed_overlap_px=args.min_seed_overlap_px,
        cyto_intensity_q=args.cyto_intensity_q,
    )

    ome_paths = sorted([
        p for p in args.input_dir.iterdir()
        if p.suffix.lower() in (".tif", ".tiff") and "ome" in p.name.lower()
    ])
    if not ome_paths:
        print(f"No OME-TIFFs found in {args.input_dir}")
        return

    print(f"Found {len(ome_paths)} file(s). Processing...")
    for path in ome_paths:
        print(f"  -> {path.name}")
        try:
            out_tif = process_file(path, args.output_dir, args.z_mode, args.save_overlays, params)
            print(f"     Saved: {out_tif.name}")
        except Exception as e:
            print(f"     ERROR processing {path.name}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
