#!/usr/bin/env python3
"""
segmentation_utils.py

Shared utilities for the Puncta-CSAT segmentation pipeline.
Consolidates duplicated functions from evaluate_nucleus.py,
evaluate_puncta.py, and mean_intensity_and_puncta.py.
"""

import os
from pathlib import Path

import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from skimage import morphology, exposure
from skimage.segmentation import relabel_sequential


# ------------------------------------------------------------------ #
#  Image loading
# ------------------------------------------------------------------ #

def load_image_2d(path, channel_index=0, z_index=0):
    """
    Load a single 2D plane from an OME-TIFF or regular TIFF.

    Handles OME axes metadata (TCZYX, CZYX, CYX, etc.) and falls
    back to shape-based heuristics for plain TIFFs.

    Parameters
    ----------
    path : str or Path
        Path to the image file.
    channel_index : int
        0-based channel index to extract.
    z_index : int
        0-based Z-plane index to extract.

    Returns
    -------
    img2d : np.ndarray (Y, X)
    """
    path = str(path)

    with tiff.TiffFile(path) as tf:
        series = tf.series[0]
        data = series.asarray()
        axes = getattr(series, "axes", None)

    # --- OME-style with known axes string ---
    if axes is not None:
        sl = [slice(None)] * len(axes)

        if "T" in axes:
            sl[axes.index("T")] = 0

        if "C" in axes:
            c_pos = axes.index("C")
            if channel_index >= data.shape[c_pos]:
                raise ValueError(
                    f"channel_index={channel_index} but image has "
                    f"only {data.shape[c_pos]} channels."
                )
            sl[c_pos] = channel_index

        if "Z" in axes:
            z_pos = axes.index("Z")
            if z_index >= data.shape[z_pos]:
                raise ValueError(
                    f"z_index={z_index} but image has "
                    f"only {data.shape[z_pos]} z-planes."
                )
            sl[z_pos] = z_index

        img2d = np.squeeze(data[tuple(sl)])
        if img2d.ndim != 2:
            raise ValueError(
                f"Expected 2D after slicing, got shape {img2d.shape} "
                f"(axes='{axes}')."
            )
        return img2d

    # --- Plain TIFF: infer from shape ---
    if data.ndim == 2:
        return data

    if data.ndim == 3:
        # Small first dim -> (C, Y, X)
        if data.shape[0] <= 4 and data.shape[0] <= data.shape[-1]:
            if channel_index >= data.shape[0]:
                raise ValueError(
                    f"channel_index={channel_index} but image has "
                    f"only {data.shape[0]} channels (C,Y,X)."
                )
            return data[channel_index]

        # Small last dim -> (Y, X, C)
        if data.shape[-1] <= 4:
            if channel_index >= data.shape[-1]:
                raise ValueError(
                    f"channel_index={channel_index} but image has "
                    f"only {data.shape[-1]} channels (Y,X,C)."
                )
            return data[:, :, channel_index]

        # Otherwise -> (Z, Y, X)
        if z_index >= data.shape[0]:
            raise ValueError(
                f"z_index={z_index} but image has "
                f"only {data.shape[0]} z-planes (Z,Y,X)."
            )
        return data[z_index]

    if data.ndim == 4:
        # Assume (Z, C, Y, X)
        if channel_index >= data.shape[1]:
            raise ValueError(
                f"channel_index={channel_index} but image has "
                f"only {data.shape[1]} channels (Z,C,Y,X)."
            )
        return data[z_index, channel_index]

    raise ValueError(
        f"Unsupported TIFF shape {data.shape} for file {path}"
    )


def ensure_2d(arr: np.ndarray) -> np.ndarray:
    """Reduce to 2D by taking arr[0] for each extra leading dimension."""
    out = arr
    while out.ndim > 2:
        out = out[0]
    return out


# ------------------------------------------------------------------ #
#  LUT / normalization
# ------------------------------------------------------------------ #

def auto_lut_clip(img, low_percentile=2.0, high_percentile=99.8):
    """
    Percentile-based LUT clipping (ImageJ-style "Auto" normalization).

    Returns a float32 image in [0, 1].
    """
    img = img.astype(np.float32)
    lo = np.percentile(img, low_percentile)
    hi = np.percentile(img, high_percentile)
    out = np.clip(img, lo, hi)
    out = (out - lo) / (hi - lo + 1e-8)
    out[img < lo] = 0.0
    return out


def percentile_norm(img2d, p_low=1, p_high=99):
    """Simple percentile normalization to [0, 1]."""
    lo, hi = np.percentile(img2d, (p_low, p_high))
    if hi <= lo:
        return np.zeros_like(img2d, dtype=np.float32)
    return np.clip((img2d - lo) / (hi - lo), 0, 1).astype(np.float32)


def normalize_dic(img2d, clip_limit=0.02, kernel_size=None):
    """
    Normalize a DIC / bright-field image for Cellpose segmentation.

    DIC images have low-contrast cell boundaries with halos and shadows.
    Standard percentile normalization doesn't work well.  Instead we use
    CLAHE (Contrast-Limited Adaptive Histogram Equalization) which boosts
    local contrast around cell edges.

    Parameters
    ----------
    img2d : np.ndarray (Y, X)
        Raw DIC image (any dtype).
    clip_limit : float
        CLAHE clip limit (higher = more contrast, default 0.02).
    kernel_size : int or None
        CLAHE tile size.  None = automatic (1/8 of image size).

    Returns
    -------
    img_norm : np.ndarray (Y, X), float32 in [0, 1]
    """
    img = img2d.astype(np.float32)
    # Percentile clip to remove extreme outliers
    lo, hi = np.percentile(img, (0.5, 99.5))
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    img = np.clip(img, lo, hi)
    img = (img - lo) / (hi - lo)
    # Apply CLAHE to boost local contrast at cell edges
    img_clahe = exposure.equalize_adapthist(
        img, clip_limit=clip_limit, kernel_size=kernel_size,
    )
    return img_clahe.astype(np.float32)


# ------------------------------------------------------------------ #
#  Mask post-processing
# ------------------------------------------------------------------ #

def filter_small_objects(masks, min_size):
    """Remove labeled regions smaller than *min_size* pixels."""
    if min_size is None or min_size <= 0:
        return masks
    filtered = morphology.remove_small_objects(
        masks, min_size=min_size, connectivity=1,
    )
    return filtered.astype(masks.dtype)


def postprocess_mask(masks, min_size=0, remove_edges=False):
    """
    Post-process a label mask:
    1. Remove objects smaller than *min_size*.
    2. Optionally remove objects touching the image border.
    3. Re-label consecutively.
    """
    labels = masks.astype(np.int32, copy=True)

    # BUG-FIX: original used ``or`` which always evaluated True
    if min_size is not None and min_size > 0:
        labels = morphology.remove_small_objects(
            labels, min_size=int(min_size), connectivity=1,
        )

    if remove_edges:
        border = np.zeros_like(labels, dtype=bool)
        border[0, :] = border[-1, :] = True
        border[:, 0] = border[:, -1] = True
        for lab in np.unique(labels[border]):
            if lab != 0:
                labels[labels == lab] = 0

    labels, _, _ = relabel_sequential(labels)
    return labels.astype(np.int32)


# ------------------------------------------------------------------ #
#  Cytoplasm mask via subtraction
# ------------------------------------------------------------------ #

def compute_cytoplasm_mask(cell_mask, nuc_mask, nuc_dilate_px=0,
                           min_nuc_pixels=10, min_overlap_frac=0.005):
    """
    Derive a cytoplasm-only label mask by subtracting nucleus from whole-cell.

    Parameters
    ----------
    cell_mask : np.ndarray (Y, X)
        Whole-cell label mask (from Cellpose cyto3).
    nuc_mask : np.ndarray (Y, X)
        Nucleus label mask.
    nuc_dilate_px : int
        Pixels to dilate the nucleus before subtraction (default 0).
    min_nuc_pixels : int
        Minimum nucleus pixels overlapping a cell to keep it (default 10).
    min_overlap_frac : float
        Minimum fraction of cell area that must overlap with nucleus
        to keep the cell (default 0.005 = 0.5%).

    Returns
    -------
    cyto_labels : np.ndarray (Y, X)
        Cytoplasm-only label mask (consecutively labelled).
    kept_labels : list[int]
        Original cell labels that were kept.
    orphan_labels : list[int]
        Original cell labels removed (insufficient nuclear overlap).
    """
    cell = cell_mask.astype(np.int32, copy=True)
    nuc_binary = nuc_mask > 0

    # Optional nucleus dilation
    if nuc_dilate_px > 0:
        se = morphology.disk(nuc_dilate_px)
        nuc_binary = morphology.dilation(nuc_binary, se)

    # Subtract nucleus region from whole-cell mask
    cyto = cell.copy()
    cyto[nuc_binary] = 0

    # Filter out cells without sufficient nuclear overlap
    unique_labels = np.unique(cell)
    unique_labels = unique_labels[unique_labels != 0]

    kept, orphans = [], []
    for lab in unique_labels:
        cell_pixels = cell == lab
        n_cell = int(cell_pixels.sum())
        if n_cell == 0:
            continue
        n_overlap = int(np.logical_and(cell_pixels, nuc_mask > 0).sum())
        frac = n_overlap / n_cell
        if n_overlap >= min_nuc_pixels and frac >= min_overlap_frac:
            kept.append(int(lab))
        else:
            orphans.append(int(lab))
            cyto[cell == lab] = 0

    # Relabel consecutively
    cyto_labels, _, _ = relabel_sequential(cyto)
    return cyto_labels.astype(np.int32), kept, orphans


def save_cytoplasm_triptych(img_norm, cell_mask, nuc_mask, cyto_mask, out_path):
    """
    Save a 4-panel QC image for cytoplasm extraction:
      [0] Image (LUT-normalized)
      [1] Whole-cell mask
      [2] Nucleus mask
      [3] Cytoplasm-only mask
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    cmap = plt.cm.get_cmap("tab20").copy()
    cmap.set_bad(color="black")

    axes[0].imshow(img_norm, cmap="gray")
    axes[0].set_title("Image (LUT)")
    axes[0].axis("off")

    cell_m = np.ma.masked_where(cell_mask == 0, cell_mask)
    axes[1].imshow(cell_m, cmap=cmap)
    axes[1].set_title("Whole-cell mask")
    axes[1].axis("off")

    nuc_m = np.ma.masked_where(nuc_mask == 0, nuc_mask)
    axes[2].imshow(nuc_m, cmap=cmap)
    axes[2].set_title("Nucleus mask")
    axes[2].axis("off")

    cyto_m = np.ma.masked_where(cyto_mask == 0, cyto_mask)
    axes[3].imshow(cyto_m, cmap=cmap)
    axes[3].set_title("Cytoplasm only")
    axes[3].axis("off")

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ #
#  Cellpose runner
# ------------------------------------------------------------------ #

def run_cellpose(img2d, model, diameter=None, batch_size=1, normalize=True):
    """
    Run a pre-initialised Cellpose model on a 2D image.

    Returns the label mask (Y, X).
    """
    if img2d.ndim == 3 and img2d.shape[-1] == 1:
        img2d = img2d[:, :, 0]
    if img2d.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {img2d.shape}")

    masks, _flows, _styles, _diams = model.eval(
        img2d,
        diameter=diameter,
        channels=[0, 0],
        batch_size=batch_size,
        normalize=normalize,
    )
    return masks


# ------------------------------------------------------------------ #
#  File I/O helpers
# ------------------------------------------------------------------ #

def save_mask(mask, out_path):
    """Save a label mask as 16-bit TIFF."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(str(out_path), mask.astype(np.uint16))


def collect_image_paths(input_path):
    """Collect all TIF / OME-TIF files from a file or directory."""
    p = Path(input_path)
    if p.is_file():
        return [p]
    if not p.is_dir():
        raise FileNotFoundError(f"{input_path} is not a file or directory")
    exts = (".tif", ".tiff", ".ome.tif", ".ome.tiff")
    files = []
    for ext in exts:
        files.extend(p.rglob(f"*{ext}"))
    return sorted(set(files))


# ------------------------------------------------------------------ #
#  Visualization: triptych
# ------------------------------------------------------------------ #

def save_triptych(img_norm, masks, out_path):
    """
    Save a 3-panel triptych: [image | mask labels | overlay].
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    cmap = plt.cm.get_cmap("tab20").copy()
    cmap.set_bad(color="black")

    axes[0].imshow(img_norm, cmap="gray")
    axes[0].set_title("Image (LUT-normalized)")
    axes[0].axis("off")

    masked = np.ma.masked_where(masks == 0, masks)
    axes[1].imshow(masked, cmap=cmap)
    axes[1].set_title("Masks")
    axes[1].axis("off")

    axes[2].imshow(img_norm, cmap="gray")
    axes[2].imshow(masked, cmap=cmap, alpha=0.5)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
