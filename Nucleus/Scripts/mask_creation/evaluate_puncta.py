#!/usr/bin/env python3
"""
evaluate_puncta.py

Run Cellpose (cyto3 model) on OME-TIF or TIF/TIFF images to generate masks.

- Automatic LUT-style thresholding/normalization before running Cellpose
  (percentile-based contrast stretch, similar to ImageJ "Auto" LUT behavior).
- For each image, save:
    * Label mask as a TIFF
    * A triptych PNG (image, masks, overlay) in a separate "triptychs" subfolder.

Assumptions:
- Cellpose 3 is installed and provides the 'cyto3' pretrained model.
- Images can be:
    * OME-TIFF with axes like "TCZYX", "CZYX", "CYX", etc.
    * Regular TIFF with shapes like (Y, X), (Z, Y, X), (Y, X, C), or (C, Y, X).
- Puncta channel is channel index = 1 (second channel).
- If multiple Z planes exist, we use Z = 8.

Usage examples:
    python evaluate_puncta.py --input /path/to/image.ome.tif --outdir masks --gpu --diameter 20
    python evaluate_puncta.py --input /path/to/folder --outdir masks --gpu --diameter 20

"""

import os
import argparse
from pathlib import Path

import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from skimage import morphology

from cellpose import models


# ----------------- image loading helpers ----------------- #

def load_cyto_plane(path, channel_index=2, z_index=0):
    """
    Load a single 2D puncta image from an OME-TIFF or regular TIFF.

    Parameters
    ----------
    path : str or Path
        Path to the image file.
    channel_index : int
        Index of the puncta channel (0-based).
    z_index : int
        Index of the z-plane to use if multiple Z planes exist.

    Returns
    -------
    img2d : np.ndarray (Y, X)
        2D image for Cellpose.
    """
    path = str(path)

    with tiff.TiffFile(path) as tf:
        # Prefer series[0] which respects OME metadata / axes
        series = tf.series[0]
        data = series.asarray()
        axes = getattr(series, "axes", None)

    # Case 1: OME-style with known axes string
    if axes is not None:
        # Build a slice for each axis
        sl = [slice(None)] * len(axes)

        # Time: use first frame
        if "T" in axes:
            t_idx = axes.index("T")
            sl[t_idx] = 0

        # Channel: use requested puncta channel
        if "C" in axes:
            c_idx = axes.index("C")
            if channel_index >= data.shape[c_idx]:
                raise ValueError(
                    f"Requested channel_index={channel_index} "
                    f"but image has only {data.shape[c_idx]} channels."
                )
            sl[c_idx] = channel_index

        # Z: use z_index (0 by default)
        if "Z" in axes:
            z_idx_axis = axes.index("Z")
            if z_index >= data.shape[z_idx_axis]:
                raise ValueError(
                    f"Requested z_index={z_index} "
                    f"but image has only {data.shape[z_idx_axis]} z-planes."
                )
            sl[z_idx_axis] = z_index

        img = data[tuple(sl)]

        # Now reduce to 2D (Y, X)
        img2d = np.squeeze(img)
        if img2d.ndim != 2:
            raise ValueError(
                f"Expected 2D image after slicing, got shape {img2d.shape} "
                f"with axes '{axes}'."
            )
        return img2d

    # Case 2: Non-OME TIFF, infer layout from shape
    # Common shapes:
    #   (Y, X)                      -> already 2D
    #   (Z, Y, X)                   -> use Z=0
    #   (Y, X, C)                   -> use specified channel in last dim
    #   (C, Y, X)                   -> use specified channel in first dim

    if data.ndim == 2:
        return data

    if data.ndim == 3:
        # Heuristic: if first dimension is small (<=4) treat as channels
        if data.shape[0] <= 4 and data.shape[0] <= data.shape[-1]:
            # (C, Y, X)
            if channel_index >= data.shape[0]:
                raise ValueError(
                    f"Requested channel_index={channel_index} "
                    f"but image has only {data.shape[0]} channels (C, Y, X)."
                )
            img2d = data[channel_index, :, :]
            return img2d

        # If last dimension is small, treat as (Y, X, C)
        if data.shape[-1] <= 4:
            if channel_index >= data.shape[-1]:
                raise ValueError(
                    f"Requested channel_index={channel_index} "
                    f"but image has only {data.shape[-1]} channels (Y, X, C)."
                )
            img2d = data[:, :, channel_index]
            return img2d

        # Otherwise assume (Z, Y, X), use z_index
        if z_index >= data.shape[0]:
            raise ValueError(
                f"Requested z_index={z_index} but image has only {data.shape[0]} z-planes (Z, Y, X)."
            )
        img2d = data[z_index, :, :]
        return img2d

    raise ValueError(
        f"Unsupported TIFF/OME-TIFF shape {data.shape} for file {path}"
    )


# ----------------- LUT / normalization helpers ----------------- #

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


# Small object removal
def filter_small_objects(masks, min_size):
    """
    Remove small labeled regions from the mask.

    Parameters
    ----------
    masks : np.ndarray (Y, X)
        Label image from Cellpose (0 = background, >0 = objects).
    min_size : int
        Minimum area (in pixels). Objects with fewer pixels are removed.

    Returns
    -------
    masks_filtered : np.ndarray (Y, X)
        Mask with small objects removed (labels may no longer be consecutive).
    """
    if min_size is None or min_size <= 0:
        return masks

    # skimage.morphology.remove_small_objects works directly on label images:
    masks_filtered = morphology.remove_small_objects(
        masks,
        min_size=min_size,
        connectivity=1,
    )
    return masks_filtered.astype(masks.dtype)


# ----------------- cellpose runner ----------------- #

def run_cellpose_on_image(
    img2d,
    model,
    diameter=None,
    batch_size=1,
    normalize=True,
):
    """
    Run Cellpose (cyto3) on a single 2D image.

    Parameters
    ----------
    img2d : np.ndarray (Y, X)
        Input image (ideally normalized to [0, 1]).
    model : cellpose.models.Cellpose
        Pre-initialized Cellpose model.
    diameter : float or None
        Estimated cell diameter. If None, Cellpose will estimate.
    batch_size : int
        Batch size for Cellpose.
    normalize : bool
        If True, let Cellpose handle intensity normalization additionally.

    Returns
    -------
    masks : np.ndarray (Y, X)
        Label mask where each cell has a unique integer ID.
    """
    # Cellpose expects at least HxW or HxWx1
    if img2d.ndim == 2:
        img_cp = img2d
    elif img2d.ndim == 3 and img2d.shape[-1] == 1:
        img_cp = img2d[:, :, 0]
    else:
        raise ValueError(f"Expected 2D or (Y, X, 1) image, got {img2d.shape}")

    # channels=[0, 0] because we have a single "cyto" channel image now
    masks, flows, styles, diams = model.eval(
        img_cp,
        diameter=diameter,
        channels=[0, 0],
        batch_size=batch_size,
        normalize=normalize,
    )
    return masks


def save_mask(mask, out_path):
    """
    Save mask as a 16-bit TIFF with integer labels.
    """
    mask_uint16 = mask.astype(np.uint16)
    out_path = str(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tiff.imwrite(out_path, mask_uint16)


# ----------------- visualization: triptych ----------------- #

def save_triptych(img_norm, masks, out_path):
    """
    Save a triptych image with:
        [left]   normalized puncta image (grayscale)
        [middle] mask labels (color)
        [right]  overlay of masks on image

    Parameters
    ----------
    img_norm : np.ndarray (Y, X)
        Normalized image in [0, 1].
    masks : np.ndarray (Y, X)
        Label mask.
    out_path : str or Path
        Path to save PNG triptych.
    """
    out_path = str(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    cmap = plt.cm.get_cmap("tab20").copy()
    cmap.set_bad(color="black") 
    
    # Image
    axes[0].imshow(img_norm, cmap="gray")
    axes[0].set_title("Image (LUT-normalized)")
    axes[0].axis("off")

    # Masks
    masked_2 = np.ma.masked_where(masks == 0, masks)
    axes[1].imshow(masked_2, cmap=cmap)
    axes[1].set_title("Masks")
    axes[1].axis("off")

    # Overlay
    axes[2].imshow(img_norm, cmap="gray")
    # Mask only where labels > 0
    masked = np.ma.masked_where(masks == 0, masks)
    axes[2].imshow(masked, cmap=cmap, alpha=0.5)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------- main script logic ----------------- #

def collect_image_paths(input_path):
    """
    Given an input path (file or directory), collect all TIF/OME-TIF files.
    """
    p = Path(input_path)
    if p.is_file():
        return [p]

    if not p.is_dir():
        raise FileNotFoundError(f"{input_path} is neither a file nor a directory")

    exts = (".tif", ".tiff", ".ome.tif", ".ome.tiff")
    files = []
    for ext in exts:
        files.extend(p.rglob(f"*{ext}"))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(
        description="Run Cellpose cyto3 model on OME-TIF / TIFF images with LUT-normalization and triptych output."
    )
    parser.add_argument(
        "--input",
        help="Path to a single image file or a directory containing images",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="cellpose_masks",
        help="Directory to save mask TIFFs (default: cellpose_masks)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU if available (requires correct CuPy / CUDA setup)",
    )
    parser.add_argument(
        "--diameter",
        type=float,
        default=None,
        help="Approximate puncta diameter; if omitted, Cellpose will estimate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for Cellpose (default: 1)",
    )
    parser.add_argument(
        "--channel-index",
        type=int,
        default=1,
        help="Puncta channel index in the input image (default: 1 = second channel)",
    )
    parser.add_argument(
        "--z-index",
        type=int,
        default=8,
        help="Z-plane index to use when multiple Z planes exist (default: 8)",
    )
    parser.add_argument(
        "--lut-low",
        type=float,
        default=2.0,
        help="Low percentile for LUT normalization (default: 2.0)",
    )
    parser.add_argument(
        "--lut-high",
        type=float,
        default=99.8,
        help="High percentile for LUT normalization (default: 99.8)",
    )

    parser.add_argument(
        "--min-size",
        type=int,
        default=10000,
        help="Minimum object size in pixels; objects smaller than this "
             "are removed from the mask (default: 0 = no filtering).",
    )


    args = parser.parse_args()

    image_paths = collect_image_paths(args.input)
    if not image_paths:
        raise RuntimeError(f"No TIF/TIFF/OME-TIFF images found under {args.input}")

    print(f"Found {len(image_paths)} image(s). Initializing Cellpose (nuclei)...")

    model = models.Cellpose(gpu=args.gpu, model_type="cyto3")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    triptych_dir = outdir / "triptychs"
    triptych_dir.mkdir(parents=True, exist_ok=True)

    for img_path in image_paths:
        print(f"Processing: {img_path}")
        try:
            img2d = load_cyto_plane(
                img_path,
                channel_index=args.channel_index,
                z_index=args.z_index,
            )

            # Apply LUT-style clipping BEFORE Cellpose
            img_norm = auto_lut_clip(
                img2d,
                low_percentile=args.lut_low,
                high_percentile=args.lut_high,
            )

            masks = run_cellpose_on_image(
                img_norm,
                model=model,
                diameter=args.diameter,
                batch_size=args.batch_size,
            )
            # masks = filter_small_objects(masks, min_size=args.min_size)
        except Exception as e:
            print(f"  [ERROR] Failed on {img_path}: {e}")
            continue

        rel = Path(img_path).stem  # filename without extension(s)

        # Save mask
        mask_path = outdir / f"{rel}_cyto3_masks.tif"
        save_mask(masks, mask_path)
        print(f"  Saved mask to: {mask_path}")

        # Save triptych
        triptych_path = triptych_dir / f"{rel}_triptych.png"
        save_triptych(img_norm, masks, triptych_path)
        print(f"  Saved triptych to: {triptych_path}")

    print("Done.")


if __name__ == "__main__":
    main()
