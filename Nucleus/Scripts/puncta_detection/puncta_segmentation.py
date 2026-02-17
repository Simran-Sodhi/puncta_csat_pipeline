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
#  Spotiflow model cache — avoids reloading the DL model per image
# ------------------------------------------------------------------ #
_spotiflow_cache = {}


def _get_spotiflow_detector(model_path="general", prob_threshold=0.5):
    """Return a cached SpotiflowDetector for the given config."""
    key = (model_path, prob_threshold)
    if key not in _spotiflow_cache:
        from .spotiflow_detector import SpotiflowDetector
        _spotiflow_cache[key] = SpotiflowDetector(
            model_path=model_path,
            prob_threshold=prob_threshold,
        )
    return _spotiflow_cache[key]


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
    spot_radius=2,
    # Tight borders params
    tb_threshold_factor=4.0,
    tb_max_branch_length=10,
    tb_connect_distance=10,
    tb_min_eq_diameter=0,
    tb_min_border_strength=0,
    # Local thresholding params
    local_radius=50,
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
        ``"spotiflow+threshold"`` — Spotiflow seeds + Otsu morphology
        ``"tight_borders"`` — skeletonization-based precise boundaries
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
        det = _get_spotiflow_detector(spotiflow_model, spotiflow_prob)
        result = det.detect_2d(img2d, mask=cell_mask)
        # Build label mask from coordinates
        labels = _coords_to_labels(
            result.coordinates, img2d.shape,
            radius=spot_radius, radii=result.radii,
        )
        return labels, preprocessed

    if method == "spotiflow+threshold":
        labels = _spotiflow_threshold_hybrid(
            img2d, preprocessed, cell_mask, preproc_kw,
            spotiflow_model=spotiflow_model,
            spotiflow_prob=spotiflow_prob,
            threshold_method=threshold_method,
            custom_threshold=custom_threshold,
            min_size=min_size,
            max_size=max_size,
            local_radius=local_radius,
        )
        return labels, preprocessed

    if method == "spotiflow+log":
        labels = _spotiflow_log_hybrid(
            img2d, preprocessed, cell_mask, preproc_kw,
            spotiflow_model=spotiflow_model,
            spotiflow_prob=spotiflow_prob,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            num_sigma=num_sigma,
            min_size=min_size,
            max_size=max_size,
        )
        return labels, preprocessed

    if method == "tight_borders":
        labels = _detect_tight_borders(
            preprocessed,
            threshold_factor=tb_threshold_factor,
            max_branch_length=tb_max_branch_length,
            connect_distance=tb_connect_distance,
            min_eq_diameter=tb_min_eq_diameter,
            min_border_strength=tb_min_border_strength,
            raw_image=img2d,
            cell_mask=cell_mask,
            min_size=min_size,
            max_size=max_size,
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


def _detect_tight_borders(
    preprocessed,
    threshold_factor=4,
    max_branch_length=10,
    connect_distance=10,
    min_eq_diameter=0,
    min_border_strength=0,
    raw_image=None,
    cell_mask=None,
    min_size=3,
    max_size=0,
):
    """Tight Borders Detection — skeletonization-based precise boundaries.

    Mirrors the Icy TightBordersDetection plugin:

    1. **Threshold** — adaptive edge threshold to find border pixels.
    2. **Skeletonize** — reduce borders to 1-pixel-wide lines.
    3. **Prune branches** — remove spurious short branches (< max_branch_length).
    4. **Connect free endings** — bridge nearby skeleton endpoints to close gaps.
    5. **Fill** — flood-fill enclosed regions to produce label masks.
    6. **Filter** — remove objects by equivalent diameter / border strength.

    Parameters
    ----------
    preprocessed : ndarray (Y, X)
        Pre-processed intensity image.
    threshold_factor : float
        Multiplier on the mean gradient for thresholding edges.
    max_branch_length : int
        Skeleton branches shorter than this are pruned.
    connect_distance : int
        Free skeleton endpoints within this distance are connected.
    min_eq_diameter : float
        Vanish objects with equivalent diameter smaller than this.
    min_border_strength : float
        Keep objects with mean border intensity above this even if small.
    """
    from scipy import ndimage as ndi
    from skimage.morphology import skeletonize, remove_small_objects

    h, w = preprocessed.shape
    img = preprocessed.astype(np.float64)

    # --- Step 1: Edge-based thresholding ------------------------------
    # Compute gradient magnitude (Sobel)
    gy = filters.sobel_h(img)
    gx = filters.sobel_v(img)
    gradient = np.sqrt(gx**2 + gy**2)

    # Threshold: gradient > factor * mean(gradient)
    grad_mean = gradient.mean()
    if grad_mean > 0:
        border_binary = gradient > (threshold_factor * grad_mean)
    else:
        border_binary = np.zeros_like(gradient, dtype=bool)

    # Intersect with cell mask
    if cell_mask is not None:
        border_binary &= (cell_mask > 0)

    if not border_binary.any():
        return np.zeros((h, w), dtype=np.int32)

    # --- Step 2: Skeletonize ------------------------------------------
    skeleton = skeletonize(border_binary)

    # --- Step 3: Prune short branches ---------------------------------
    if max_branch_length > 0:
        skeleton = _prune_skeleton_branches(skeleton, max_branch_length)

    # --- Step 4: Connect free endings ---------------------------------
    if connect_distance > 0:
        skeleton = _connect_skeleton_endpoints(skeleton, connect_distance)

    # --- Step 5: Fill enclosed regions --------------------------------
    # Invert skeleton so enclosed regions become connected foreground
    # Add a 1-pixel border to prevent flood-fill from leaking
    padded = np.pad(~skeleton, 1, constant_values=False)
    # Label connected regions (background of skeleton = objects)
    filled_labels = measure.label(padded, connectivity=1)
    # Remove the border padding
    filled_labels = filled_labels[1:-1, 1:-1]

    # The largest region is typically the background — remove it
    if filled_labels.max() > 0:
        props = measure.regionprops(filled_labels)
        bg_label = max(props, key=lambda p: p.area).label
        filled_labels[filled_labels == bg_label] = 0

    # Intersect with cell mask
    if cell_mask is not None:
        filled_labels[cell_mask == 0] = 0

    # --- Step 6: Filter by size / eq. diameter / border strength ------
    if filled_labels.max() > 0:
        intensity_img = raw_image if raw_image is not None else preprocessed
        for prop in measure.regionprops(filled_labels,
                                         intensity_image=gradient):
            remove = False
            if min_size > 0 and prop.area < min_size:
                remove = True
            if max_size > 0 and prop.area > max_size:
                remove = True
            if min_eq_diameter > 0 and prop.equivalent_diameter < min_eq_diameter:
                # Check if border is strong enough to keep
                if min_border_strength > 0 and prop.mean_intensity >= min_border_strength:
                    remove = False
                else:
                    remove = True
            if remove:
                filled_labels[filled_labels == prop.label] = 0

    filled_labels, _, _ = relabel_sequential(filled_labels)
    return filled_labels.astype(np.int32)


def _prune_skeleton_branches(skeleton, max_length):
    """Remove skeleton branches shorter than max_length pixels.

    Endpoints have exactly 1 neighbour; branch points have 3+.
    We trace from each endpoint and remove if path length < max_length
    before hitting another branch point or endpoint.
    """
    skel = skeleton.copy()
    h, w = skel.shape

    def _neighbours(y, x):
        pts = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and skel[ny, nx]:
                    pts.append((ny, nx))
        return pts

    def _n_neighbours(y, x):
        return len(_neighbours(y, x))

    # Iteratively prune until no more short branches
    changed = True
    iterations = 0
    while changed and iterations < 20:
        changed = False
        iterations += 1
        endpoints = []
        for y in range(h):
            for x in range(w):
                if skel[y, x] and _n_neighbours(y, x) == 1:
                    endpoints.append((y, x))

        for ey, ex in endpoints:
            if not skel[ey, ex]:
                continue
            # Trace from endpoint
            path = [(ey, ex)]
            cy, cx = ey, ex
            for _ in range(max_length + 1):
                nbrs = _neighbours(cy, cx)
                # Exclude previous pixel
                nbrs = [(ny, nx) for ny, nx in nbrs if (ny, nx) not in path]
                if not nbrs:
                    break
                cy, cx = nbrs[0]
                nn = _n_neighbours(cy, cx)
                if nn >= 3:
                    # Reached a junction — this is a short branch
                    break
                path.append((cy, cx))

            # If branch is short (< max_length), remove it
            if len(path) <= max_length:
                for py, px in path[:-1]:  # Keep the junction point
                    skel[py, px] = False
                changed = True

    return skel


def _connect_skeleton_endpoints(skeleton, max_distance):
    """Connect nearby free endpoints in a skeleton.

    For each pair of endpoints within max_distance, draw a line
    connecting them on the skeleton.
    """
    from skimage.draw import line as skline

    skel = skeleton.copy()
    h, w = skel.shape

    # Find endpoints (pixels with exactly 1 skeleton neighbour)
    endpoints = []
    for y in range(h):
        for x in range(w):
            if not skel[y, x]:
                continue
            n = 0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and skel[ny, nx]:
                        n += 1
            if n == 1:
                endpoints.append((y, x))

    # Connect nearby pairs
    connected = set()
    for i, (y1, x1) in enumerate(endpoints):
        if i in connected:
            continue
        best_j = -1
        best_dist = max_distance + 1
        for j, (y2, x2) in enumerate(endpoints):
            if j <= i or j in connected:
                continue
            d = np.sqrt((y1 - y2)**2 + (x1 - x2)**2)
            if d < best_dist and d <= max_distance:
                best_dist = d
                best_j = j
        if best_j >= 0:
            y2, x2 = endpoints[best_j]
            rr, cc = skline(y1, x1, y2, x2)
            skel[rr, cc] = True
            connected.add(i)
            connected.add(best_j)

    return skel


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


def _spotiflow_threshold_hybrid(
    img2d, preprocessed, cell_mask, preproc_kw,
    spotiflow_model="general",
    spotiflow_prob=0.5,
    threshold_method="otsu",
    custom_threshold=None,
    min_size=3,
    max_size=0,
    local_radius=50,
):
    """Hybrid: Spotiflow seeds + **local** Otsu thresholding + watershed.

    Spotiflow-first pipeline that captures faint puncta missed by global
    thresholding:

    - **Spotiflow** detects all puncta centres — including dim spots that
      a global threshold would miss.
    - **Local Otsu** is computed in a window around each seed, so the
      threshold adapts to the local intensity profile and correctly
      captures spots that are locally bright but globally faint.
    - **Marker-controlled watershed** assigns the locally-thresholded
      foreground pixels to the nearest Spotiflow seed, producing clean
      label masks.

    Steps
    -----
    1. Spotiflow detects puncta centres (seeds).
    2. For each seed, Otsu (or user-chosen) threshold is computed on a
       local patch of radius ``local_radius`` around the seed and the
       resulting foreground pixels are accumulated into a combined mask.
    3. Each Spotiflow seed becomes a watershed marker placed inside its
       local foreground region.
    4. Watershed assigns foreground pixels to the nearest seed.
    5. Standard size filtering is applied.

    Parameters
    ----------
    local_radius : int
        Half-width (in pixels) of the square neighbourhood around each
        Spotiflow seed in which local Otsu thresholding is performed.
        Larger values give more context but approach global behaviour;
        smaller values are more adaptive but may under-segment.
        Default 50.
    """
    from skimage.segmentation import watershed

    # --- Step 1: Spotiflow point detection ----------------------------
    det = _get_spotiflow_detector(spotiflow_model, spotiflow_prob)
    result = det.detect_2d(img2d, mask=cell_mask)

    if result.coordinates is None or len(result.coordinates) == 0:
        return np.zeros(img2d.shape, dtype=np.int32)

    coords = np.asarray(result.coordinates, dtype=np.float64)

    # --- Step 2: Local Otsu around each Spotiflow seed ----------------
    h, w = preprocessed.shape
    binary = np.zeros((h, w), dtype=bool)
    lr = int(local_radius)

    for y, x in coords:
        yi, xi = int(round(y)), int(round(x))
        yi = np.clip(yi, 0, h - 1)
        xi = np.clip(xi, 0, w - 1)

        # Extract local patch
        y0 = max(0, yi - lr)
        y1 = min(h, yi + lr + 1)
        x0 = max(0, xi - lr)
        x1 = min(w, xi + lr + 1)
        patch = preprocessed[y0:y1, x0:x1]

        # Skip degenerate patches (uniform intensity)
        if patch.max() == patch.min():
            continue

        # Compute threshold on the local patch
        if threshold_method == "custom" and custom_threshold is not None:
            local_thresh = float(custom_threshold)
        elif threshold_method == "otsu":
            local_thresh = filters.threshold_otsu(patch)
        elif threshold_method == "yen":
            local_thresh = filters.threshold_yen(patch)
        elif threshold_method == "triangle":
            local_thresh = filters.threshold_triangle(patch)
        elif threshold_method == "li":
            local_thresh = filters.threshold_li(patch)
        else:
            local_thresh = filters.threshold_otsu(patch)

        # Apply local threshold and write into the global binary mask
        binary[y0:y1, x0:x1] |= (patch > local_thresh)

    # Intersect with cell mask if provided
    if cell_mask is not None:
        binary = binary & (cell_mask > 0)

    if not binary.any():
        # If no foreground, fall back to small disk labels from Spotiflow
        return _coords_to_labels(coords, img2d.shape, radius=2)

    # --- Step 3: Place Spotiflow seeds as markers ---------------------
    markers = np.zeros((h, w), dtype=np.int32)
    valid_seeds = 0
    for i, (y, x) in enumerate(coords):
        yi, xi = int(round(y)), int(round(x))
        yi = np.clip(yi, 0, h - 1)
        xi = np.clip(xi, 0, w - 1)
        # Seed should now be inside the locally-thresholded region
        if binary[yi, xi]:
            markers[yi, xi] = i + 1
            valid_seeds += 1
        else:
            # Seed just outside — check local neighbourhood (3x3)
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ny, nx = yi + dy, xi + dx
                    if 0 <= ny < h and 0 <= nx < w and binary[ny, nx]:
                        markers[ny, nx] = i + 1
                        valid_seeds += 1
                        break
                else:
                    continue
                break

    if valid_seeds == 0:
        return _coords_to_labels(coords, img2d.shape, radius=2)

    # --- Step 4: Watershed on locally-thresholded foreground ----------
    inv = preprocessed.max() - preprocessed
    labels = watershed(inv, markers=markers, mask=binary, compactness=1.0)

    # --- Step 5: Size filter ------------------------------------------
    if min_size > 0 or max_size > 0:
        for prop in measure.regionprops(labels):
            if min_size > 0 and prop.area < min_size:
                labels[labels == prop.label] = 0
            if max_size > 0 and prop.area > max_size:
                labels[labels == prop.label] = 0
        labels, _, _ = relabel_sequential(labels)

    return labels.astype(np.int32)


def _spotiflow_log_hybrid(
    img2d, preprocessed, cell_mask, preproc_kw,
    spotiflow_model="general",
    spotiflow_prob=0.5,
    min_sigma=1.0,
    max_sigma=5.0,
    num_sigma=5,
    min_size=3,
    max_size=0,
):
    """Hybrid: Spotiflow seeds + LoG scale estimation + watershed segmentation.

    1. Spotiflow provides accurate (y, x) point detections.
    2. A multi-scale LoG filter bank is evaluated at each seed to estimate
       the best-matching blob sigma (and thus radius).
    3. Marker-controlled watershed segments the actual blob boundaries
       using the Spotiflow seeds as markers.
    4. Standard size filtering is applied.
    """
    from skimage.segmentation import watershed

    # --- Step 1: Spotiflow point detection ----------------------------
    det = _get_spotiflow_detector(spotiflow_model, spotiflow_prob)
    result = det.detect_2d(img2d, mask=cell_mask)

    if result.coordinates is None or len(result.coordinates) == 0:
        return np.zeros(img2d.shape, dtype=np.int32)

    coords = np.asarray(result.coordinates, dtype=np.float64)

    # --- Step 2: LoG scale-space → radius per seed --------------------
    sigmas = np.linspace(min_sigma, max_sigma, num_sigma)
    h, w = preprocessed.shape
    # Pre-compute normalized LoG responses at every scale
    from scipy.ndimage import gaussian_laplace
    log_stack = np.empty((len(sigmas), h, w), dtype=np.float32)
    for si, s in enumerate(sigmas):
        # Normalized Laplacian of Gaussian (sigma^2 weighting)
        log_stack[si] = -(gaussian_laplace(preprocessed.astype(np.float64), sigma=s) * s * s)

    radii = np.empty(len(coords), dtype=np.float64)
    for i, (y, x) in enumerate(coords):
        yi, xi = int(round(y)), int(round(x))
        yi = np.clip(yi, 0, h - 1)
        xi = np.clip(xi, 0, w - 1)
        # Pick sigma with strongest LoG response at this location
        responses = log_stack[:, yi, xi]
        best_idx = int(np.argmax(responses))
        radii[i] = sigmas[best_idx] * np.sqrt(2)  # blob radius = sigma * sqrt(2)

    # --- Step 3: Marker-controlled watershed --------------------------
    markers = np.zeros(preprocessed.shape, dtype=np.int32)
    for i, (y, x) in enumerate(coords):
        yi, xi = int(round(y)), int(round(x))
        yi = np.clip(yi, 0, h - 1)
        xi = np.clip(xi, 0, w - 1)
        markers[yi, xi] = i + 1

    # Build a generous binary mask around all seeds so watershed doesn't
    # flood the entire image.  Each seed gets a disk of 3x its LoG radius.
    from skimage.draw import disk as skdisk
    ws_mask_local = np.zeros(preprocessed.shape, dtype=bool)
    for i, (y, x) in enumerate(coords):
        yi, xi = int(round(y)), int(round(x))
        yi = np.clip(yi, 0, h - 1)
        xi = np.clip(xi, 0, w - 1)
        r = max(3, int(round(radii[i] * 3)))
        rr, cc = skdisk((yi, xi), r, shape=preprocessed.shape)
        ws_mask_local[rr, cc] = True

    # Intersect with cell mask if provided
    if cell_mask is not None:
        ws_mask_local &= (cell_mask > 0)

    # Invert preprocessed image so watershed flows into bright regions
    inv = preprocessed.max() - preprocessed
    labels = watershed(inv, markers=markers, mask=ws_mask_local,
                       compactness=1.0)

    # --- Step 4: Size filter ------------------------------------------
    if min_size > 0 or max_size > 0:
        for prop in measure.regionprops(labels):
            if min_size > 0 and prop.area < min_size:
                labels[labels == prop.label] = 0
            if max_size > 0 and prop.area > max_size:
                labels[labels == prop.label] = 0
        labels, _, _ = relabel_sequential(labels)

    return labels.astype(np.int32)


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
                det = _get_spotiflow_detector(
                    detector_params.get("spotiflow_model", "general"),
                    detector_params.get("spotiflow_prob", 0.5),
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
#  Puncta-to-cell assignment and per-cell quantification
# ------------------------------------------------------------------ #

def assign_puncta_to_cells(puncta_labels, cell_labels):
    """Map each punctum to the cell it falls within.

    Parameters
    ----------
    puncta_labels : ndarray int32 (Y, X)
        Label mask where each punctum has a unique integer > 0.
    cell_labels : ndarray int32 (Y, X)
        Label mask where each cell has a unique integer > 0.

    Returns
    -------
    mapping : dict
        ``{cell_id: [list of puncta_ids]}``.
        Puncta that fall outside all cells are collected under key ``0``.
    """
    mapping = {}
    for prop in measure.regionprops(puncta_labels):
        cy, cx = int(round(prop.centroid[0])), int(round(prop.centroid[1]))
        cy = np.clip(cy, 0, cell_labels.shape[0] - 1)
        cx = np.clip(cx, 0, cell_labels.shape[1] - 1)
        cell_id = int(cell_labels[cy, cx])
        mapping.setdefault(cell_id, []).append(prop.label)
    return mapping


def per_cell_quantification(puncta_labels, cell_labels, raw_image=None):
    """Compute per-cell puncta statistics.

    Parameters
    ----------
    puncta_labels : ndarray int32 (Y, X)
        Puncta label mask.
    cell_labels : ndarray int32 (Y, X)
        Cell label mask.
    raw_image : ndarray or None
        If provided, intensity metrics are computed for each punctum.

    Returns
    -------
    list of dict
        One row per cell with columns:
        cell_id, cell_area, puncta_count, puncta_total_area,
        puncta_density, puncta_ids, [mean_puncta_intensity].
    """
    assignment = assign_puncta_to_cells(puncta_labels, cell_labels)

    # Pre-compute puncta properties
    puncta_props = {p.label: p for p in measure.regionprops(puncta_labels)}

    # Gather intensity per punctum if raw image is provided
    puncta_int_props = {}
    if raw_image is not None:
        for p in measure.regionprops(puncta_labels, intensity_image=raw_image):
            puncta_int_props[p.label] = p

    cell_props = {p.label: p for p in measure.regionprops(cell_labels)}

    rows = []
    for cell_id in sorted(cell_props.keys()):
        cell_area = cell_props[cell_id].area
        p_ids = assignment.get(cell_id, [])
        p_count = len(p_ids)
        p_total_area = sum(puncta_props[pid].area for pid in p_ids if pid in puncta_props)
        p_density = p_count / cell_area if cell_area > 0 else 0.0

        row = {
            "cell_id": cell_id,
            "cell_area": cell_area,
            "puncta_count": p_count,
            "puncta_total_area": p_total_area,
            "puncta_density": round(p_density, 6),
            "puncta_ids": p_ids,
        }

        if raw_image is not None:
            intensities = [
                puncta_int_props[pid].mean_intensity
                for pid in p_ids if pid in puncta_int_props
            ]
            row["mean_puncta_intensity"] = (
                round(float(np.mean(intensities)), 4) if intensities else 0.0
            )

        rows.append(row)

    return rows


def _load_cell_mask(path):
    """Load a cell mask from .tif or Cellpose _seg.npy."""
    path = Path(path)
    if path.suffix == ".npy":
        dat = np.load(str(path), allow_pickle=True).item()
        return ensure_2d(np.asarray(dat["masks"]))
    return ensure_2d(tiff.imread(str(path)))


def _parse_location(path):
    """Extract a location token for matching images across directories."""
    import re
    s = str(path)
    m = re.search(r"XYPos[/\\\\]([^.]*)\.ome", s)
    if m:
        return m.group(1)
    m = re.search(r"(\d+_Z\d+)", s)
    if m:
        return m.group(1)
    stem = Path(path).stem
    for suffix in ("_seg", "_cyto3_masks", "_cell_masks", "_cyto_masks", "_masks",
                    "_puncta_masks", "_puncta"):
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
            break
    return stem


def _build_cell_mask_map(cell_mask_dir):
    """Build {location_token: Path} for cell masks."""
    cell_mask_dir = Path(cell_mask_dir)
    mapping = {}
    for ext in ("*.tif", "*.tiff", "*_seg.npy"):
        for p in cell_mask_dir.rglob(ext):
            loc = _parse_location(p)
            if loc not in mapping or p.suffix == ".npy":
                mapping[loc] = p
    print(f"[INFO] Cell masks: indexed {len(mapping)} masks from {cell_mask_dir}")
    return mapping


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
    spot_radius=2,
    tb_threshold_factor=4.0,
    tb_max_branch_length=10,
    tb_connect_distance=10,
    tb_min_eq_diameter=0,
    tb_min_border_strength=0,
    local_radius=50,
    consensus_detectors=None,
    consensus_strategy="weighted_confidence",
    consensus_weights=None,
    consensus_threshold=0.3,
    consensus_match_dist=3.0,
    cell_mask_dir=None,
    save_cellpose_npy=True,
    save_triptychs=True,
    progress_callback=None,
):
    """Batch-segment puncta across all images in a directory.

    Parameters
    ----------
    cell_mask_dir : str or Path or None
        Directory containing cell/ROI label masks (.tif or _seg.npy).
        When provided, each image is matched to its cell mask by location
        token, puncta are assigned to individual cells, and a per-cell
        CSV is saved alongside the output masks.
    progress_callback : callable or None
        Called with (index, total, filename, n_objects) after each image.

    Returns
    -------
    list of dict — per-image summary.
    """
    import pandas as pd

    image_dir = Path(image_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trip_dir = out_dir / "triptychs"
    if save_triptychs:
        trip_dir.mkdir(parents=True, exist_ok=True)

    # Build cell mask lookup if directory provided
    cell_mask_map = {}
    if cell_mask_dir:
        cell_mask_map = _build_cell_mask_map(cell_mask_dir)

    image_paths = collect_image_paths(str(image_dir))
    if not image_paths:
        print("[WARN] No images found")
        return []

    summaries = []
    all_cell_rows = []
    total = len(image_paths)
    print(f"[INFO] Processing {total} image(s) for puncta segmentation")

    for idx, img_path in enumerate(image_paths, 1):
        stem = img_path.stem
        print(f"  [{idx}/{total}] {img_path.name}")

        try:
            img2d = load_image_2d(img_path, channel_index=channel, z_index=z_index)

            # Look up matching cell mask
            cell_mask = None
            location = _parse_location(img_path)
            if cell_mask_map and location in cell_mask_map:
                cell_mask = _load_cell_mask(cell_mask_map[location])
                if cell_mask.shape != img2d.shape:
                    print(f"    [WARN] Cell mask shape {cell_mask.shape} != "
                          f"image shape {img2d.shape}, skipping cell mask")
                    cell_mask = None
                else:
                    print(f"    Using cell mask: {cell_mask_map[location].name} "
                          f"({int(cell_mask.max())} cells)")

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
                spot_radius=spot_radius,
                cell_mask=cell_mask,
                local_radius=local_radius,
                tb_threshold_factor=tb_threshold_factor,
                tb_max_branch_length=tb_max_branch_length,
                tb_connect_distance=tb_connect_distance,
                tb_min_eq_diameter=tb_min_eq_diameter,
                tb_min_border_strength=tb_min_border_strength,
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

            # Per-cell quantification
            if cell_mask is not None and n_objects > 0:
                cell_rows = per_cell_quantification(
                    labels, cell_mask, raw_image=img2d,
                )
                for row in cell_rows:
                    row["filename"] = img_path.name
                    row["location"] = location
                    row["puncta_ids"] = str(row["puncta_ids"])
                all_cell_rows.extend(cell_rows)
                print(f"    {n_objects} puncta assigned to "
                      f"{sum(1 for r in cell_rows if r['puncta_count'] > 0)} cells")

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

    # Save per-cell CSV if we have cell data
    if all_cell_rows:
        csv_path = out_dir / "per_cell_puncta_summary.csv"
        df = pd.DataFrame(all_cell_rows)
        # Reorder columns
        col_order = [
            "filename", "location", "cell_id", "cell_area",
            "puncta_count", "puncta_total_area", "puncta_density",
        ]
        if "mean_puncta_intensity" in df.columns:
            col_order.append("mean_puncta_intensity")
        col_order.append("puncta_ids")
        df = df[[c for c in col_order if c in df.columns]]
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Per-cell summary saved: {csv_path} ({len(df)} rows)")

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
                                 "intensity_ratio", "spotiflow",
                                 "spotiflow+threshold", "spotiflow+log",
                                 "tight_borders", "consensus"],
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
    parser.add_argument("--spot-radius", type=int, default=2,
                        help="Radius (px) for Spotiflow spot masks (default: 2)")
    parser.add_argument("--local-radius", type=int, default=50,
                        help="Half-width (px) of local Otsu window around each "
                             "Spotiflow seed for spotiflow+threshold (default: 50)")
    parser.add_argument("--cell-mask-dir", default=None,
                        help="Directory of cell masks for per-cell quantification")
    # Tight borders
    parser.add_argument("--tb-threshold-factor", type=float, default=4.0,
                        help="Edge threshold multiplier for tight borders (default: 4)")
    parser.add_argument("--tb-max-branch-length", type=int, default=10,
                        help="Max skeleton branch length to prune (default: 10)")
    parser.add_argument("--tb-connect-distance", type=int, default=10,
                        help="Max distance to connect free skeleton endpoints (default: 10)")
    parser.add_argument("--tb-min-eq-diameter", type=float, default=0,
                        help="Min equivalent diameter to keep (default: 0 = off)")
    parser.add_argument("--tb-min-border-strength", type=float, default=0,
                        help="Keep small objects with border strength above this (default: 0)")
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
        spot_radius=args.spot_radius,
        local_radius=args.local_radius,
        cell_mask_dir=args.cell_mask_dir,
        tb_threshold_factor=args.tb_threshold_factor,
        tb_max_branch_length=args.tb_max_branch_length,
        tb_connect_distance=args.tb_connect_distance,
        tb_min_eq_diameter=args.tb_min_eq_diameter,
        tb_min_border_strength=args.tb_min_border_strength,
        save_cellpose_npy=not args.no_cellpose_npy,
        save_triptychs=not args.no_triptychs,
        consensus_detectors=args.consensus_detectors,
        consensus_strategy=args.consensus_strategy,
    )
