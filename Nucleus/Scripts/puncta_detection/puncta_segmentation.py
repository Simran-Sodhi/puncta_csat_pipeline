#!/usr/bin/env python3
"""
puncta_segmentation.py — Classical 2D puncta detection for training masks.

Detects puncta (small bright spots) using classical image processing methods
instead of Cellpose, which is designed for cell-sized objects and often fails
on sub-cellular puncta.

Supported detection methods
---------------------------
- **Threshold** (Otsu, Yen, Triangle, Li, or a fixed value)
- **LoG** (Laplacian of Gaussian blob detection)
- **DoG** (Difference of Gaussians blob detection)

Each method produces a per-image label mask suitable for:
  1. Direct use in the Analysis pipeline.
  2. Curation in the Cellpose GUI (``_seg.npy`` format).
  3. Training a custom Cellpose or other segmentation model.

Usage (CLI)
-----------
    python puncta_segmentation.py \\
        --image-dir /path/to/ome_tiffs \\
        --out-dir /path/to/puncta_masks \\
        --channel 1 --method threshold --threshold-method otsu \\
        --min-size 3 --max-size 500 --sigma 1.0

Usage (from GUI)
----------------
    from puncta_detection.puncta_segmentation import segment_puncta_2d, batch_segment

    mask = segment_puncta_2d(img2d, method="threshold", ...)
    batch_segment(image_dir, out_dir, channel=1, ...)
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import tifffile as tiff
from skimage import filters, measure, morphology, exposure, feature
from skimage.segmentation import relabel_sequential

# Import shared utilities
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from segmentation_utils import (
    load_image_2d,
    auto_lut_clip,
    ensure_2d,
    save_mask,
    save_seg_npy,
    save_triptych,
    collect_image_paths,
)


# ------------------------------------------------------------------ #
#  Pre-processing
# ------------------------------------------------------------------ #

def preprocess_puncta(img2d, sigma=1.0, background_subtraction=True,
                      tophat_radius=15):
    """
    Pre-process a 2D image for puncta detection.

    Parameters
    ----------
    img2d : ndarray (Y, X)
        Raw 2D intensity image.
    sigma : float
        Gaussian smoothing sigma.  Small values (0.5–1.5) preserve puncta
        while suppressing pixel noise.
    background_subtraction : bool
        Apply white top-hat to remove uneven background illumination.
    tophat_radius : int
        Radius of the disk structuring element for white top-hat.
        Should be larger than the largest expected punctum.

    Returns
    -------
    ndarray float32 in [0, 1]
    """
    img = img2d.astype(np.float32)

    if background_subtraction and tophat_radius > 0:
        se = morphology.disk(tophat_radius)
        img = morphology.white_tophat(img, se)

    if sigma > 0:
        img = filters.gaussian(img, sigma=sigma)

    # Rescale to [0, 1]
    img = exposure.rescale_intensity(img, out_range=(0.0, 1.0))
    return img


# ------------------------------------------------------------------ #
#  Detection methods
# ------------------------------------------------------------------ #

def _detect_threshold(img, threshold_method="otsu", custom_value=None):
    """Threshold-based puncta detection.  Returns a binary mask."""
    if threshold_method == "custom" and custom_value is not None:
        # Custom value is in the normalised [0,1] space
        thresh = float(custom_value)
    elif threshold_method == "otsu":
        thresh = filters.threshold_otsu(img)
    elif threshold_method == "yen":
        thresh = filters.threshold_yen(img)
    elif threshold_method == "triangle":
        thresh = filters.threshold_triangle(img)
    elif threshold_method == "li":
        thresh = filters.threshold_li(img)
    else:
        thresh = filters.threshold_otsu(img)

    return img > thresh


def _detect_log(img, min_sigma=1.0, max_sigma=5.0, num_sigma=5,
                threshold_rel=0.1):
    """
    Laplacian-of-Gaussian blob detection.

    Returns a binary mask where each detected blob is filled as a disk
    with the detected radius.
    """
    blobs = feature.blob_log(
        img,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=int(num_sigma),
        threshold=threshold_rel,
    )
    mask = np.zeros_like(img, dtype=bool)
    for y, x, s in blobs:
        r = int(np.ceil(s * np.sqrt(2)))
        rr, cc = _disk_coords(int(y), int(x), r, img.shape)
        mask[rr, cc] = True
    return mask


def _detect_dog(img, min_sigma=1.0, max_sigma=5.0, threshold_rel=0.1):
    """
    Difference-of-Gaussians blob detection.

    Returns a binary mask similar to LoG.
    """
    blobs = feature.blob_dog(
        img,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        threshold=threshold_rel,
    )
    mask = np.zeros_like(img, dtype=bool)
    for y, x, s in blobs:
        r = int(np.ceil(s))
        rr, cc = _disk_coords(int(y), int(x), r, img.shape)
        mask[rr, cc] = True
    return mask


def _disk_coords(cy, cx, radius, shape):
    """Return (row, col) arrays for a filled disk, clipped to *shape*."""
    from skimage.draw import disk as skdisk
    rr, cc = skdisk((cy, cx), radius, shape=shape)
    return rr, cc


# ------------------------------------------------------------------ #
#  Size filtering & labelling
# ------------------------------------------------------------------ #

def _label_and_filter(binary, min_size=3, max_size=0, open_radius=0):
    """
    Label connected components, apply morphological opening, then
    filter by size.

    Returns a consecutively-labelled int32 mask.
    """
    if open_radius > 0:
        se = morphology.disk(open_radius)
        binary = morphology.binary_opening(binary, se)

    if min_size > 0:
        binary = morphology.remove_small_objects(binary, min_size=min_size)

    labels = measure.label(binary, connectivity=1)

    if max_size > 0:
        for prop in measure.regionprops(labels):
            if prop.area > max_size:
                labels[labels == prop.label] = 0

    labels, _, _ = relabel_sequential(labels)
    return labels.astype(np.int32)


# ------------------------------------------------------------------ #
#  Public API — single-image
# ------------------------------------------------------------------ #

def segment_puncta_2d(
    img2d,
    method="threshold",
    # Pre-processing
    sigma=1.0,
    background_subtraction=True,
    tophat_radius=15,
    # Threshold params
    threshold_method="otsu",
    custom_threshold=None,
    # Blob detection params
    min_sigma=1.0,
    max_sigma=5.0,
    num_sigma=5,
    blob_threshold=0.1,
    # Post-processing
    min_size=3,
    max_size=0,
    open_radius=0,
):
    """
    Detect puncta in a single 2D image.

    Parameters
    ----------
    img2d : ndarray (Y, X)
        Raw intensity image (any dtype).
    method : {"threshold", "log", "dog"}
        Detection algorithm.
    sigma : float
        Pre-processing Gaussian sigma.
    background_subtraction : bool
        Apply white top-hat background removal.
    tophat_radius : int
        Structuring element radius for top-hat.
    threshold_method : str
        One of "otsu", "yen", "triangle", "li", "custom".
    custom_threshold : float or None
        Fixed threshold value (normalised 0–1 range) when
        ``threshold_method="custom"``.
    min_sigma, max_sigma, num_sigma : float
        Blob-scale search range for LoG/DoG.
    blob_threshold : float
        Relative threshold for blob detection (lower = more blobs).
    min_size : int
        Remove objects smaller than this (pixels).
    max_size : int
        Remove objects larger than this (0 = no limit).
    open_radius : int
        Morphological opening radius to separate touching puncta.

    Returns
    -------
    labels : ndarray int32 (Y, X)
        Label mask where each punctum has a unique integer > 0.
    preprocessed : ndarray float32 (Y, X)
        The pre-processed image (useful for QC / triptychs).
    """
    preprocessed = preprocess_puncta(
        img2d, sigma=sigma,
        background_subtraction=background_subtraction,
        tophat_radius=tophat_radius,
    )

    if method == "threshold":
        binary = _detect_threshold(
            preprocessed,
            threshold_method=threshold_method,
            custom_value=custom_threshold,
        )
    elif method == "log":
        binary = _detect_log(
            preprocessed,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            num_sigma=num_sigma,
            threshold_rel=blob_threshold,
        )
    elif method == "dog":
        binary = _detect_dog(
            preprocessed,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            threshold_rel=blob_threshold,
        )
    else:
        raise ValueError(f"Unknown method: {method!r}")

    labels = _label_and_filter(
        binary, min_size=min_size, max_size=max_size, open_radius=open_radius,
    )
    return labels, preprocessed


# ------------------------------------------------------------------ #
#  Public API — batch
# ------------------------------------------------------------------ #

def batch_segment(
    image_dir,
    out_dir,
    channel=1,
    z_index=0,
    method="threshold",
    sigma=1.0,
    background_subtraction=True,
    tophat_radius=15,
    threshold_method="otsu",
    custom_threshold=None,
    min_sigma=1.0,
    max_sigma=5.0,
    num_sigma=5,
    blob_threshold=0.1,
    min_size=3,
    max_size=0,
    open_radius=0,
    save_cellpose_npy=True,
    save_triptychs=True,
    progress_callback=None,
):
    """
    Batch-segment puncta across all images in a directory.

    Parameters
    ----------
    image_dir : str or Path
        Folder containing OME-TIFFs or plain TIFFs.
    out_dir : str or Path
        Output folder for masks and (optional) _seg.npy files.
    channel : int
        Channel to use for puncta (e.g. 1 = mEGFP).
    z_index : int
        Z-slice to use (0-indexed).
    save_cellpose_npy : bool
        Save Cellpose-compatible ``_seg.npy`` files for manual curation
        in the Cellpose GUI.
    save_triptychs : bool
        Save QC triptych PNGs.
    progress_callback : callable or None
        Called with ``(index, total, filename, n_objects)`` after each image.
    **kwargs
        Passed through to :func:`segment_puncta_2d`.

    Returns
    -------
    list of dict
        Summary per image (filename, n_objects).
    """
    image_dir = Path(image_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trip_dir = out_dir / "triptychs"
    if save_triptychs:
        trip_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_image_paths(str(image_dir))
    if not image_paths:
        print("[WARN] No images found")
        return []

    summaries = []
    total = len(image_paths)
    print(f"[INFO] Processing {total} image(s) for puncta segmentation")

    for idx, img_path in enumerate(image_paths, 1):
        stem = img_path.stem
        print(f"  [{idx}/{total}] {img_path.name}")

        try:
            img2d = load_image_2d(img_path, channel_index=channel, z_index=z_index)

            labels, preprocessed = segment_puncta_2d(
                img2d,
                method=method,
                sigma=sigma,
                background_subtraction=background_subtraction,
                tophat_radius=tophat_radius,
                threshold_method=threshold_method,
                custom_threshold=custom_threshold,
                min_sigma=min_sigma,
                max_sigma=max_sigma,
                num_sigma=num_sigma,
                blob_threshold=blob_threshold,
                min_size=min_size,
                max_size=max_size,
                open_radius=open_radius,
            )

            n_objects = int(labels.max())

            # Save mask TIFF
            save_mask(labels, out_dir / f"{stem}_puncta_masks.tif")

            # Save Cellpose _seg.npy for GUI curation
            if save_cellpose_npy:
                save_seg_npy(
                    preprocessed, labels, [],
                    f"{stem}_puncta", out_dir, diameter=None,
                )

            # Save triptych
            if save_triptychs:
                save_triptych(
                    auto_lut_clip(img2d), labels,
                    trip_dir / f"{stem}_puncta_triptych.png",
                )

            summaries.append({
                "filename": img_path.name,
                "n_objects": n_objects,
                "status": "OK" if n_objects > 0 else "No puncta",
            })

        except Exception as exc:
            print(f"    [ERROR] {exc}")
            summaries.append({
                "filename": img_path.name,
                "n_objects": -1,
                "status": f"FAILED: {exc}",
            })

        if progress_callback is not None:
            progress_callback(idx, total, img_path.name,
                              summaries[-1]["n_objects"])

    print(f"[DONE] Processed {total} images -> {out_dir}")
    return summaries


# ------------------------------------------------------------------ #
#  CLI
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classical 2D puncta segmentation for training masks."
    )
    parser.add_argument("--image-dir", required=True,
                        help="Folder with OME-TIFF or TIFF images")
    parser.add_argument("--out-dir", required=True,
                        help="Output folder for puncta masks")
    parser.add_argument("--channel", type=int, default=1,
                        help="Intensity channel for puncta (default: 1 = mEGFP)")
    parser.add_argument("--z-index", type=int, default=0,
                        help="Z-slice index (default: 0)")
    parser.add_argument("--method", choices=["threshold", "log", "dog"],
                        default="threshold",
                        help="Detection method (default: threshold)")
    parser.add_argument("--sigma", type=float, default=1.0,
                        help="Pre-processing Gaussian sigma (default: 1.0)")
    parser.add_argument("--no-bg-sub", action="store_true",
                        help="Disable background subtraction (white top-hat)")
    parser.add_argument("--tophat-radius", type=int, default=15,
                        help="Top-hat structuring element radius (default: 15)")
    parser.add_argument("--threshold-method",
                        choices=["otsu", "yen", "triangle", "li", "custom"],
                        default="otsu",
                        help="Threshold algorithm (default: otsu)")
    parser.add_argument("--custom-threshold", type=float, default=None,
                        help="Fixed threshold (0-1) when --threshold-method=custom")
    parser.add_argument("--min-sigma", type=float, default=1.0,
                        help="Min blob sigma for LoG/DoG (default: 1.0)")
    parser.add_argument("--max-sigma", type=float, default=5.0,
                        help="Max blob sigma for LoG/DoG (default: 5.0)")
    parser.add_argument("--num-sigma", type=int, default=5,
                        help="Number of sigma steps for LoG (default: 5)")
    parser.add_argument("--blob-threshold", type=float, default=0.1,
                        help="Relative blob threshold (default: 0.1)")
    parser.add_argument("--min-size", type=int, default=3,
                        help="Min puncta size in pixels (default: 3)")
    parser.add_argument("--max-size", type=int, default=0,
                        help="Max puncta size in pixels (0 = no limit)")
    parser.add_argument("--open-radius", type=int, default=0,
                        help="Morphological opening radius (default: 0)")
    parser.add_argument("--no-cellpose-npy", action="store_true",
                        help="Skip saving _seg.npy files")
    parser.add_argument("--no-triptychs", action="store_true",
                        help="Skip saving triptych PNGs")

    args = parser.parse_args()
    batch_segment(
        image_dir=args.image_dir,
        out_dir=args.out_dir,
        channel=args.channel,
        z_index=args.z_index,
        method=args.method,
        sigma=args.sigma,
        background_subtraction=not args.no_bg_sub,
        tophat_radius=args.tophat_radius,
        threshold_method=args.threshold_method,
        custom_threshold=args.custom_threshold,
        min_sigma=args.min_sigma,
        max_sigma=args.max_sigma,
        num_sigma=args.num_sigma,
        blob_threshold=args.blob_threshold,
        min_size=args.min_size,
        max_size=args.max_size,
        open_radius=args.open_radius,
        save_cellpose_npy=not args.no_cellpose_npy,
        save_triptychs=not args.no_triptychs,
    )
