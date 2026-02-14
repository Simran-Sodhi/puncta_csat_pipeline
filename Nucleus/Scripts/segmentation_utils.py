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
from scipy.ndimage import binary_fill_holes, gaussian_filter
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


def smooth_labels(labels, smooth_radius=3):
    """
    Smooth the edges of each labeled object using morphological operations
    and Gaussian blurring.

    For each label:
    1. Extract binary mask for the label.
    2. Apply morphological closing then opening (disk structuring element)
       to remove small protrusions and fill small concavities.
    3. Apply Gaussian blur + 0.5 threshold to round off remaining
       jagged pixel-level edges.
    4. Fill interior holes created by the smoothing.

    Parameters
    ----------
    labels : np.ndarray (Y, X)
        Integer label mask (0 = background).
    smooth_radius : int
        Radius of the disk structuring element and sigma for the
        Gaussian blur.  Larger values = smoother edges.
        Typical range: 2-5.  Default 3.

    Returns
    -------
    smoothed : np.ndarray (Y, X), same dtype as input
    """
    if smooth_radius <= 0:
        return labels

    smoothed = np.zeros_like(labels)
    selem = morphology.disk(smooth_radius)

    for lab in np.unique(labels):
        if lab == 0:
            continue
        binary = labels == lab
        # Morphological close then open to regularise shape
        binary = morphology.binary_closing(binary, selem)
        binary = morphology.binary_opening(binary, selem)
        # Gaussian blur + threshold to round off pixelated edges
        blurred = gaussian_filter(binary.astype(np.float32), sigma=smooth_radius)
        binary = blurred > 0.5
        # Fill any interior holes
        binary = binary_fill_holes(binary)
        # Write back, later labels overwrite earlier ones at overlaps
        smoothed[binary] = lab

    return smoothed


def postprocess_mask(masks, min_size=0, remove_edges=False, smooth_radius=0,
                     edge_thresh=0.0):
    """
    Post-process a label mask:
    1. Remove objects smaller than *min_size*.
    2. Optionally remove objects touching the image border.
    3. Smooth mask edges (if smooth_radius > 0).
    4. Re-label consecutively.

    Parameters
    ----------
    edge_thresh : float
        Fraction of an object's perimeter that must lie on the image
        border before it is removed (0.0–1.0).  When 0.0 (default),
        any border contact removes the object.  Set to e.g. 0.25 for
        DIC images to keep cells that only slightly touch the edge.
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
            if lab == 0:
                continue
            if edge_thresh > 0:
                obj_mask = labels == lab
                # Perimeter = pixels in the object that have at least
                # one 4-connected background neighbour
                eroded = morphology.binary_erosion(obj_mask)
                perimeter = obj_mask & ~eroded
                n_perim = int(perimeter.sum())
                if n_perim == 0:
                    continue
                n_border = int((perimeter & border).sum())
                if n_border / n_perim < edge_thresh:
                    continue  # keep this object
            labels[labels == lab] = 0

    if smooth_radius > 0:
        labels = smooth_labels(labels, smooth_radius=smooth_radius)

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
#  Cellpose model loader
# ------------------------------------------------------------------ #

def _resolve_bioimageio_weights(source):
    """
    Resolve a BioImage.io model descriptor (rdf.yaml, .zip, or resource ID)
    to a local PyTorch weights path that Cellpose can load.

    Parameters
    ----------
    source : str
        Path to rdf.yaml / bioimageio.yaml / .zip, or a ``bioimage.io:<id>`` URI.

    Returns
    -------
    weights_path : str
        Path to the PyTorch state-dict file.
    """
    try:
        from bioimageio.spec import load_description
    except ImportError:
        raise ImportError(
            "The 'bioimageio.spec' package is required to load BioImage.io "
            "models.  Install it with:  pip install 'bioimageio.core'"
        )

    model_descr = load_description(source)
    weights = getattr(model_descr, "weights", None)
    if weights is None:
        raise ValueError(f"BioImage.io model '{source}' has no weights field.")

    pt_weights = getattr(weights, "pytorch_state_dict", None)
    if pt_weights is None:
        available = getattr(weights, "available_formats", [])
        raise ValueError(
            f"BioImage.io model '{source}' has no pytorch_state_dict weights. "
            f"Available weight formats: {available}"
        )

    return str(pt_weights.source)


def _resolve_model_spec(model_spec):
    """
    Resolve a model specification to a value Cellpose understands.

    Handles:
    - Built-in model names (cyto3, cpsam, etc.) -- returned as-is.
    - ``bioimage.io:<resource-id>`` URIs.
    - Paths to BioImage.io descriptors (rdf.yaml, bioimageio.yaml, .zip).
    - Paths to regular Cellpose model files -- returned as-is.
    """
    if model_spec is None:
        return None

    spec = str(model_spec).strip()

    # Explicit BioImage.io URI
    if spec.startswith("bioimage.io:"):
        return _resolve_bioimageio_weights(spec[len("bioimage.io:"):])

    spec_path = Path(spec)

    # Check if it's a BioImage.io descriptor file
    if spec_path.is_file() and spec_path.suffix.lower() in (".yaml", ".yml", ".zip"):
        name = spec_path.name.lower()
        if name in ("rdf.yaml", "bioimageio.yaml") or name.endswith(".zip"):
            return _resolve_bioimageio_weights(str(spec_path))
        # For other YAML files, peek inside for BioImage.io format markers
        if spec_path.suffix.lower() in (".yaml", ".yml"):
            try:
                with open(spec_path) as fh:
                    header = fh.read(512)
                if "format_version" in header and "weights" in header:
                    return _resolve_bioimageio_weights(str(spec_path))
            except Exception:
                pass

    return spec


def load_cellpose_model(gpu=False, model_type="cyto3"):
    """
    Load a Cellpose model, compatible with both Cellpose 3 and 4.

    Supports:
    - Built-in Cellpose model names (``cyto3``, ``cpsam``, etc.)
    - Custom Cellpose model file paths
    - BioImage.io models (rdf.yaml, bioimageio.yaml, .zip, or
      ``bioimage.io:<id>`` URIs)

    Cellpose 4 removed ``models.Cellpose`` and uses
    ``models.CellposeModel(pretrained_model=...)`` instead.
    """
    from cellpose import models

    resolved = _resolve_model_spec(model_type)

    # Cellpose 4+: use CellposeModel with pretrained_model
    if hasattr(models, "CellposeModel"):
        try:
            return models.CellposeModel(gpu=gpu, pretrained_model=resolved)
        except TypeError:
            # Fallback: older CellposeModel signature
            return models.CellposeModel(gpu=gpu, model_type=resolved)

    # Cellpose 3: use Cellpose class
    if hasattr(models, "Cellpose"):
        return models.Cellpose(gpu=gpu, model_type=resolved)

    raise ImportError(
        "Cannot find Cellpose model class. "
        "Please install cellpose >= 3.0: pip install cellpose"
    )


# ------------------------------------------------------------------ #
#  Cellpose runner
# ------------------------------------------------------------------ #

def run_cellpose(img2d, model, diameter=None, batch_size=1, normalize=True):
    """
    Run a pre-initialised Cellpose model on a 2D image.

    Compatible with Cellpose 3 (channels param) and 4 (no channels).
    Returns (masks, flows) where masks is (Y, X) label array and
    flows is the list of flow arrays from Cellpose.
    """
    if img2d.ndim == 3 and img2d.shape[-1] == 1:
        img2d = img2d[:, :, 0]
    if img2d.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {img2d.shape}")

    # Cellpose 4 removed the channels parameter
    import inspect
    eval_params = inspect.signature(model.eval).parameters

    kwargs = dict(
        diameter=diameter,
        batch_size=batch_size,
        normalize=normalize,
    )
    # Only pass channels if the model.eval() accepts it (Cellpose 3)
    if "channels" in eval_params:
        kwargs["channels"] = [0, 0]

    # Cellpose 4 returns 3 values (masks, flows, styles);
    # Cellpose 3 returns 4 values (masks, flows, styles, diams).
    result = model.eval(img2d, **kwargs)
    masks = result[0]
    flows = result[1] if len(result) > 1 else []
    return masks, flows


# ------------------------------------------------------------------ #
#  File I/O helpers
# ------------------------------------------------------------------ #

def save_mask(mask, out_path):
    """Save a label mask as 16-bit TIFF."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(str(out_path), mask.astype(np.uint16))


def load_mask(path):
    """
    Load a mask from a ``.tif`` or Cellpose ``_seg.npy`` file.

    For ``_seg.npy`` files, extracts the ``masks`` array from the
    dictionary. This allows downstream analysis to use masks that
    have been manually edited in the Cellpose GUI.

    Returns
    -------
    mask : np.ndarray (Y, X)
    """
    path = Path(path)
    if path.suffix == ".npy":
        dat = np.load(str(path), allow_pickle=True).item()
        return np.asarray(dat["masks"])
    return tiff.imread(str(path))


def save_seg_npy(img, masks, flows, filename, out_dir, diameter=None):
    """
    Save a Cellpose-compatible ``_seg.npy`` file for manual curation.

    The saved file can be opened in the Cellpose GUI for mask editing.
    The GUI looks for ``<imagename>_seg.npy`` alongside the image file.

    Parameters
    ----------
    img : np.ndarray
        The 2D image that was segmented (any dtype, 2D or 3D).
    masks : np.ndarray (Y, X)
        Integer label mask.
    flows : list
        Flow arrays returned by ``model.eval()``.
    filename : str or Path
        Original image filename (used as the stem for the .npy file).
    out_dir : str or Path
        Directory in which to save the ``_seg.npy`` file.
    diameter : float or None
        Estimated cell diameter (optional).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(filename).stem

    # Cellpose GUI expects img as (Y, X, 3) uint8 RGB stack.
    # Convert our single-channel image to the expected format.
    img_arr = np.asarray(img)
    if img_arr.ndim == 2:
        # Normalise to 0-255 uint8 for display
        fmin, fmax = img_arr.min(), img_arr.max()
        if fmax > fmin:
            img_u8 = ((img_arr - fmin) / (fmax - fmin) * 255).astype(np.uint8)
        else:
            img_u8 = np.zeros(img_arr.shape, dtype=np.uint8)
        # Stack to (Y, X, 3) so Cellpose GUI can unpack (Ly, Lx, _)
        img_rgb = np.stack([img_u8, img_u8, img_u8], axis=-1)
    elif img_arr.ndim == 3 and img_arr.shape[-1] == 3:
        # Already (Y, X, 3)
        if img_arr.dtype != np.uint8:
            fmin, fmax = img_arr.min(), img_arr.max()
            if fmax > fmin:
                img_rgb = ((img_arr - fmin) / (fmax - fmin) * 255).astype(np.uint8)
            else:
                img_rgb = np.zeros(img_arr.shape, dtype=np.uint8)
        else:
            img_rgb = img_arr
    else:
        # Multi-channel (C, Y, X) – take first channel, convert to RGB
        if img_arr.ndim == 3:
            ch0 = img_arr[0]
        else:
            ch0 = img_arr.reshape(img_arr.shape[-2], img_arr.shape[-1])
        fmin, fmax = ch0.min(), ch0.max()
        if fmax > fmin:
            img_u8 = ((ch0 - fmin) / (fmax - fmin) * 255).astype(np.uint8)
        else:
            img_u8 = np.zeros(ch0.shape, dtype=np.uint8)
        img_rgb = np.stack([img_u8, img_u8, img_u8], axis=-1)

    n_cells = int(masks.max()) if masks.max() > 0 else 0
    dat = {
        "img": img_rgb,
        "masks": masks.astype(np.uint16),
        "outlines": [],
        "flows": flows,
        "ismanual": np.zeros(n_cells, dtype=bool),
        "filename": str(filename),
        "est_diam": diameter if diameter else 0,
        "chan_choose": [0, 0],
    }

    npy_path = out_dir / f"{stem}_seg.npy"
    np.save(str(npy_path), dat)
    return npy_path


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
