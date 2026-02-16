#!/usr/bin/env python3
"""
puncta_segmentation.py — Unified 2D puncta detection for training masks.

Integrates multiple detection methods through the HybridPuncta framework:

  * **Threshold** — fast global thresholding (Otsu / Yen / Triangle / Li / custom)
  * **LoG**       — Punctatools-style Laplacian-of-Gaussian blob detection
                     with background refinement and watershed segmentation
  * **DoG**       — Difference of Gaussians blob detection
  * **Intensity-Ratio** — PunctaFinder-style 3-criteria detector
                     (local ratio, global ratio, CV)
  * **Spotiflow** — deep-learning spot detection (requires ``pip install spotiflow``)
  * **Consensus** — combine any 2+ detectors with union / intersection /
                     majority-vote / weighted-confidence strategies

Each method produces per-image label masks suitable for:
  1. Direct use in the Analysis pipeline.
  2. Curation in the Cellpose GUI (``_seg.npy`` format).
  3. Training a custom segmentation model.

Usage (CLI)
-----------
    python puncta_segmentation.py \\
        --image-dir /path/to/ome_tiffs \\
        --out-dir /path/to/puncta_masks \\
        --channel 1 --method log \\
        --min-size 3 --max-size 500 --sigma 1.0

Usage (from GUI / Python)
-------------------------
    from puncta_detection.puncta_segmentation import segment_puncta_2d, batch_segment
    labels, preprocessed = segment_puncta_2d(img2d, method="log", ...)
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import tifffile as tiff
from skimage import filters, measure, morphology, exposure, feature
from skimage.segmentation import relabel_sequential

# Import shared segmentation utilities
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

# Import new detector framework
from .preprocessing import preprocess_pipeline
from .core import PunctaDetectionResult


# ------------------------------------------------------------------ #
#  Legacy simple detectors (threshold / DoG) — kept for backwards
#  compatibility.  LoG and intensity-ratio now delegate to the
#  framework detectors.
# ------------------------------------------------------------------ #

def _detect_threshold(img, threshold_method="otsu", custom_value=None):
    """Threshold-based puncta detection.  Returns a binary mask."""
    if threshold_method == "custom" and custom_value is not None:
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


def _detect_dog(img, min_sigma=1.0, max_sigma=5.0, threshold_rel=0.1):
    """Difference-of-Gaussians blob detection → binary mask."""
    blobs = feature.blob_dog(
        img, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold_rel,
    )
    mask = np.zeros_like(img, dtype=bool)
    from skimage.draw import disk as skdisk
    for y, x, s in blobs:
        r = max(1, int(np.ceil(s)))
        rr, cc = skdisk((int(y), int(x)), r, shape=img.shape)
        mask[rr, cc] = True
    return mask


def _label_and_filter(binary, min_size=3, max_size=0, open_radius=0):
    """Label connected components with size filtering."""
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
    bg_method="white_tophat",
    # Threshold params
    threshold_method="otsu",
    custom_threshold=None,
    # Blob detection params (LoG / DoG)
    min_sigma=1.0,
    max_sigma=5.0,
    num_sigma=5,
    blob_threshold=0.1,
    # Intensity-ratio params
    punctum_radius=3,
    t_local=1.5,
    t_global=1.5,
    t_cv=0.5,
    ir_step=1,
    # Post-processing
    min_size=3,
    max_size=0,
    open_radius=0,
    # LoG-specific
    use_watershed=True,
    bg_rejection=True,
    # Cell / ROI mask
    cell_mask=None,
    # Spotiflow params
    spotiflow_model="general",
    spotiflow_prob=0.5,
    # Consensus params
    consensus_detectors=None,
    consensus_strategy="weighted_confidence",
    consensus_weights=None,
    consensus_threshold=0.3,
    consensus_match_dist=3.0,
):
    """
    Detect puncta in a single 2D image.

    Parameters
    ----------
    img2d : ndarray (Y, X)
        Raw intensity image.
    method : str
        Detection method:
        ``"threshold"`` — global thresholding
        ``"log"`` — Punctatools-style LoG with watershed
        ``"dog"`` — Difference of Gaussians
        ``"intensity_ratio"`` — PunctaFinder-style 3-criteria
        ``"spotiflow"`` — deep-learning (requires spotiflow package)
        ``"consensus"`` — combine multiple detectors
    cell_mask : ndarray or None
        If provided, detections outside the mask are discarded.

    Returns
    -------
    labels : ndarray int32 (Y, X)
        Label mask (each punctum = unique integer > 0).
    preprocessed : ndarray float32 (Y, X)
        Pre-processed image (for QC triptychs).
    """
    preproc_kw = dict(
        bg_method=bg_method,
        bg_radius=tophat_radius,
        bg_enabled=background_subtraction,
        denoise_method="gaussian",
        denoise_sigma=sigma,
        denoise_enabled=(sigma > 0),
    )
    preprocessed = preprocess_pipeline(img2d, **preproc_kw)

    # ----- Framework detectors -------------------------------------
    if method == "log":
        from .log_detector import LoGDetector
        det = LoGDetector(
            log_threshold=blob_threshold,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            num_sigma=num_sigma,
            min_size=min_size,
            max_size=max_size,
            use_watershed=use_watershed,
            bg_rejection=bg_rejection,
            preprocess_kw=preproc_kw,
        )
        result = det.detect_2d(img2d, mask=cell_mask)
        labels = result.labels if result.labels is not None else np.zeros(img2d.shape, dtype=np.int32)
        return labels, preprocessed

    if method == "intensity_ratio":
        from .intensity_ratio_detector import IntensityRatioDetector
        det = IntensityRatioDetector(
            punctum_radius=punctum_radius,
            t_local=t_local,
            t_global=t_global,
            t_cv=t_cv,
            min_distance=min_size,
            step=ir_step,
            preprocess_kw=preproc_kw,
        )
        result = det.detect_2d(img2d, mask=cell_mask)
        labels = result.labels if result.labels is not None else np.zeros(img2d.shape, dtype=np.int32)
        return labels, preprocessed

    if method == "spotiflow":
        from .spotiflow_detector import SpotiflowDetector
        det = SpotiflowDetector(
            model_path=spotiflow_model,
            prob_threshold=spotiflow_prob,
        )
        result = det.detect_2d(img2d, mask=cell_mask)
        # Build label mask from coordinates
        labels = _coords_to_labels(
            result.coordinates, img2d.shape,
            radius=2, radii=result.radii,
        )
        return labels, preprocessed

    if method == "consensus":
        labels = _run_consensus(
            img2d, preprocessed, cell_mask, preproc_kw,
            consensus_detectors=consensus_detectors,
            consensus_strategy=consensus_strategy,
            consensus_weights=consensus_weights,
            consensus_threshold=consensus_threshold,
            consensus_match_dist=consensus_match_dist,
            # Pass through all detector params
            blob_threshold=blob_threshold,
            min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma,
            min_size=min_size, max_size=max_size,
            use_watershed=use_watershed, bg_rejection=bg_rejection,
            threshold_method=threshold_method, custom_threshold=custom_threshold,
            punctum_radius=punctum_radius,
            t_local=t_local, t_global=t_global, t_cv=t_cv,
            spotiflow_model=spotiflow_model, spotiflow_prob=spotiflow_prob,
        )
        return labels, preprocessed

    # ----- Legacy simple detectors ---------------------------------
    if method == "threshold":
        binary = _detect_threshold(preprocessed, threshold_method, custom_threshold)
    elif method == "dog":
        binary = _detect_dog(
            preprocessed, min_sigma=min_sigma, max_sigma=max_sigma,
            threshold_rel=blob_threshold,
        )
    else:
        raise ValueError(f"Unknown method: {method!r}")

    labels = _label_and_filter(binary, min_size=min_size, max_size=max_size,
                               open_radius=open_radius)

    # Apply cell mask
    if cell_mask is not None and labels.shape == cell_mask.shape:
        labels[cell_mask == 0] = 0
        labels, _, _ = relabel_sequential(labels)

    return labels, preprocessed


def _coords_to_labels(coords, shape, radius=2, radii=None):
    """Convert coordinate array to a label mask."""
    from skimage.draw import disk as skdisk
    labels = np.zeros(shape, dtype=np.int32)
    for i, (y, x) in enumerate(coords):
        r = int(round(radii[i])) if radii is not None else radius
        r = max(1, r)
        rr, cc = skdisk((int(round(y)), int(round(x))), r, shape=shape)
        labels[rr, cc] = i + 1
    return labels


def _run_consensus(
    img2d, preprocessed, cell_mask, preproc_kw,
    consensus_detectors=None,
    consensus_strategy="weighted_confidence",
    consensus_weights=None,
    consensus_threshold=0.3,
    consensus_match_dist=3.0,
    **detector_params,
):
    """Run multiple detectors and combine via consensus engine."""
    from .consensus import ConsensusEngine

    if consensus_detectors is None:
        consensus_detectors = ["threshold", "log"]

    results = {}

    for name in consensus_detectors:
        try:
            if name == "threshold":
                binary = _detect_threshold(
                    preprocessed,
                    detector_params.get("threshold_method", "otsu"),
                    detector_params.get("custom_threshold"),
                )
                lbl = _label_and_filter(
                    binary,
                    min_size=detector_params.get("min_size", 3),
                    max_size=detector_params.get("max_size", 0),
                )
                coords = _labels_to_coords(lbl)
                conf = np.ones(len(coords)) * 0.5
                results[name] = PunctaDetectionResult(
                    coordinates=coords, confidences=conf, labels=lbl,
                    metadata={"detector": "threshold"},
                )

            elif name == "log":
                from .log_detector import LoGDetector
                det = LoGDetector(
                    log_threshold=detector_params.get("blob_threshold", 0.01),
                    min_sigma=detector_params.get("min_sigma", 1.0),
                    max_sigma=detector_params.get("max_sigma", 5.0),
                    num_sigma=detector_params.get("num_sigma", 5),
                    min_size=detector_params.get("min_size", 3),
                    max_size=detector_params.get("max_size", 0),
                    use_watershed=detector_params.get("use_watershed", True),
                    bg_rejection=detector_params.get("bg_rejection", True),
                    preprocess_kw=preproc_kw,
                )
                results[name] = det.detect_2d(img2d, mask=cell_mask)

            elif name == "intensity_ratio":
                from .intensity_ratio_detector import IntensityRatioDetector
                det = IntensityRatioDetector(
                    punctum_radius=detector_params.get("punctum_radius", 3),
                    t_local=detector_params.get("t_local", 1.5),
                    t_global=detector_params.get("t_global", 1.5),
                    t_cv=detector_params.get("t_cv", 0.5),
                    preprocess_kw=preproc_kw,
                )
                results[name] = det.detect_2d(img2d, mask=cell_mask)

            elif name == "spotiflow":
                from .spotiflow_detector import SpotiflowDetector
                det = SpotiflowDetector(
                    model_path=detector_params.get("spotiflow_model", "general"),
                    prob_threshold=detector_params.get("spotiflow_prob", 0.5),
                )
                results[name] = det.detect_2d(img2d, mask=cell_mask)

        except Exception as exc:
            print(f"[WARN] Detector '{name}' failed: {exc}")
            continue

    if not results:
        return np.zeros(img2d.shape, dtype=np.int32)

    engine = ConsensusEngine(
        strategy=consensus_strategy,
        matching_distance=consensus_match_dist,
        weights=consensus_weights or {},
        confidence_threshold=consensus_threshold,
    )
    combined = engine.combine(results, image_shape=img2d.shape)
    return combined.labels if combined.labels is not None else np.zeros(img2d.shape, dtype=np.int32)


def _labels_to_coords(labels):
    """Extract centroids from a label mask."""
    props = measure.regionprops(labels)
    if not props:
        return np.empty((0, 2), dtype=np.float64)
    return np.array([p.centroid for p in props], dtype=np.float64)


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
    bg_method="white_tophat",
    threshold_method="otsu",
    custom_threshold=None,
    min_sigma=1.0,
    max_sigma=5.0,
    num_sigma=5,
    blob_threshold=0.1,
    punctum_radius=3,
    t_local=1.5,
    t_global=1.5,
    t_cv=0.5,
    ir_step=1,
    min_size=3,
    max_size=0,
    open_radius=0,
    use_watershed=True,
    bg_rejection=True,
    spotiflow_model="general",
    spotiflow_prob=0.5,
    consensus_detectors=None,
    consensus_strategy="weighted_confidence",
    consensus_weights=None,
    consensus_threshold=0.3,
    consensus_match_dist=3.0,
    save_cellpose_npy=True,
    save_triptychs=True,
    progress_callback=None,
):
    """Batch-segment puncta across all images in a directory.

    Parameters
    ----------
    progress_callback : callable or None
        Called with (index, total, filename, n_objects) after each image.

    Returns
    -------
    list of dict — per-image summary.
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
                bg_method=bg_method,
                threshold_method=threshold_method,
                custom_threshold=custom_threshold,
                min_sigma=min_sigma,
                max_sigma=max_sigma,
                num_sigma=num_sigma,
                blob_threshold=blob_threshold,
                punctum_radius=punctum_radius,
                t_local=t_local,
                t_global=t_global,
                t_cv=t_cv,
                ir_step=ir_step,
                min_size=min_size,
                max_size=max_size,
                open_radius=open_radius,
                use_watershed=use_watershed,
                bg_rejection=bg_rejection,
                spotiflow_model=spotiflow_model,
                spotiflow_prob=spotiflow_prob,
                consensus_detectors=consensus_detectors,
                consensus_strategy=consensus_strategy,
                consensus_weights=consensus_weights,
                consensus_threshold=consensus_threshold,
                consensus_match_dist=consensus_match_dist,
            )

            n_objects = int(labels.max())

            save_mask(labels, out_dir / f"{stem}_puncta_masks.tif")

            if save_cellpose_npy:
                save_seg_npy(preprocessed, labels, [], f"{stem}_puncta",
                             out_dir, diameter=None)

            if save_triptychs:
                save_triptych(auto_lut_clip(img2d), labels,
                              trip_dir / f"{stem}_puncta_triptych.png")

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
        description="2D puncta segmentation with multiple detection methods."
    )
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--channel", type=int, default=1)
    parser.add_argument("--z-index", type=int, default=0)
    parser.add_argument("--method",
                        choices=["threshold", "log", "dog",
                                 "intensity_ratio", "spotiflow", "consensus"],
                        default="threshold")
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--no-bg-sub", action="store_true")
    parser.add_argument("--tophat-radius", type=int, default=15)
    parser.add_argument("--bg-method", default="white_tophat",
                        choices=["white_tophat", "rolling_ball", "gaussian", "median"])
    parser.add_argument("--threshold-method",
                        choices=["otsu", "yen", "triangle", "li", "custom"],
                        default="otsu")
    parser.add_argument("--custom-threshold", type=float, default=None)
    parser.add_argument("--min-sigma", type=float, default=1.0)
    parser.add_argument("--max-sigma", type=float, default=5.0)
    parser.add_argument("--num-sigma", type=int, default=5)
    parser.add_argument("--blob-threshold", type=float, default=0.1)
    parser.add_argument("--punctum-radius", type=int, default=3)
    parser.add_argument("--t-local", type=float, default=1.5)
    parser.add_argument("--t-global", type=float, default=1.5)
    parser.add_argument("--t-cv", type=float, default=0.5)
    parser.add_argument("--min-size", type=int, default=3)
    parser.add_argument("--max-size", type=int, default=0)
    parser.add_argument("--open-radius", type=int, default=0)
    parser.add_argument("--no-cellpose-npy", action="store_true")
    parser.add_argument("--no-triptychs", action="store_true")
    # Consensus
    parser.add_argument("--consensus-detectors", nargs="+",
                        default=["threshold", "log"])
    parser.add_argument("--consensus-strategy", default="weighted_confidence")

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
        bg_method=args.bg_method,
        threshold_method=args.threshold_method,
        custom_threshold=args.custom_threshold,
        min_sigma=args.min_sigma,
        max_sigma=args.max_sigma,
        num_sigma=args.num_sigma,
        blob_threshold=args.blob_threshold,
        punctum_radius=args.punctum_radius,
        t_local=args.t_local,
        t_global=args.t_global,
        t_cv=args.t_cv,
        min_size=args.min_size,
        max_size=args.max_size,
        open_radius=args.open_radius,
        save_cellpose_npy=not args.no_cellpose_npy,
        save_triptychs=not args.no_triptychs,
        consensus_detectors=args.consensus_detectors,
        consensus_strategy=args.consensus_strategy,
    )
