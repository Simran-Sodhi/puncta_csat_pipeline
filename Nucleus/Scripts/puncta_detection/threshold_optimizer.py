#!/usr/bin/env python3
"""
threshold_optimizer.py — Optimise detector thresholds for training.

Given a set of images with ground-truth puncta annotations, sweep
parameters for classical detectors (LoG, intensity-ratio) and find
the combination that maximises F1 (or another metric).
"""

import itertools
from typing import Optional

import numpy as np

from .core import PunctaMetrics, PunctaDetectionResult


def _run_and_score(detector, images, ground_truths, masks, metric, max_dist):
    """Run detector on all images and compute aggregate metric."""
    total_tp, total_fp, total_fn = 0, 0, 0
    for i, img in enumerate(images):
        m = masks[i] if masks is not None else None
        res = detector.detect_2d(img, mask=m)
        gt = ground_truths[i]
        tp, fp, fn = PunctaMetrics.match_detections(
            res.coordinates, gt, max_distance=max_dist,
        )
        total_tp += len(tp)
        total_fp += len(fp)
        total_fn += len(fn)

    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    if metric == "f1":
        return f1
    if metric == "precision":
        return prec
    if metric == "recall":
        return rec
    return f1


def optimize_log_detector(
    images: list,
    ground_truths: list,
    masks: list | None = None,
    param_grid: dict | None = None,
    metric: str = "f1",
    max_distance: float = 3.0,
    progress_callback=None,
) -> dict:
    """Grid-search over LoG detector parameters.

    Parameters
    ----------
    images : list of ndarray
        2-D intensity images.
    ground_truths : list of ndarray
        Ground truth coordinates per image, each (N, 2) as [y, x].
    masks : list of ndarray or None
        Cell / ROI masks per image.
    param_grid : dict or None
        ``{param_name: [values]}`` to sweep.  Defaults cover common
        ranges for LoG detection.
    metric : str
        ``"f1"``, ``"precision"``, or ``"recall"``.
    max_distance : float
        Matching distance for TP assignment.
    progress_callback : callable or None
        Called with (current_step, total_steps).

    Returns
    -------
    dict
        ``{"best_params": {...}, "best_score": float, "history": [...]}``
    """
    from .log_detector import LoGDetector

    if param_grid is None:
        param_grid = {
            "log_threshold": [0.005, 0.01, 0.02, 0.04, 0.08],
            "min_sigma": [0.5, 1.0, 1.5, 2.0],
            "max_sigma": [3.0, 5.0, 7.0],
            "min_size": [2, 3, 5],
        }

    keys = sorted(param_grid.keys())
    values = [param_grid[k] for k in keys]
    combos = list(itertools.product(*values))
    total = len(combos)

    best_score = -1.0
    best_params = {}
    history = []

    for step, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        det = LoGDetector(**params)
        score = _run_and_score(det, images, ground_truths, masks, metric, max_distance)
        history.append({"params": params, "score": score})

        if score > best_score:
            best_score = score
            best_params = params.copy()

        if progress_callback is not None:
            progress_callback(step + 1, total)

    return {
        "best_params": best_params,
        "best_score": best_score,
        "history": history,
    }


def optimize_intensity_ratio_detector(
    images: list,
    ground_truths: list,
    masks: list | None = None,
    param_grid: dict | None = None,
    metric: str = "f1",
    max_distance: float = 3.0,
    progress_callback=None,
) -> dict:
    """Grid-search over intensity-ratio detector parameters.

    Returns same structure as :func:`optimize_log_detector`.
    """
    from .intensity_ratio_detector import IntensityRatioDetector

    if param_grid is None:
        param_grid = {
            "punctum_radius": [2, 3, 4, 5],
            "t_local": [1.2, 1.5, 1.8, 2.0, 2.5],
            "t_global": [1.2, 1.5, 1.8, 2.0],
            "t_cv": [0.3, 0.5, 0.7, 1.0],
        }

    keys = sorted(param_grid.keys())
    values = [param_grid[k] for k in keys]
    combos = list(itertools.product(*values))
    total = len(combos)

    best_score = -1.0
    best_params = {}
    history = []

    for step, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        det = IntensityRatioDetector(**params)
        score = _run_and_score(det, images, ground_truths, masks, metric, max_distance)
        history.append({"params": params, "score": score})

        if score > best_score:
            best_score = score
            best_params = params.copy()

        if progress_callback is not None:
            progress_callback(step + 1, total)

    return {
        "best_params": best_params,
        "best_score": best_score,
        "history": history,
    }


def optimize_consensus_weights(
    images: list,
    ground_truths: list,
    detector_results: dict[str, list[PunctaDetectionResult]],
    weight_steps: int = 5,
    max_distance: float = 3.0,
) -> dict:
    """Find optimal consensus weights by grid search.

    Parameters
    ----------
    detector_results : dict
        ``{detector_name: [PunctaDetectionResult per image]}``.
    weight_steps : int
        Granularity of weight grid (e.g., 5 → weights 0.0, 0.25, 0.5, 0.75, 1.0).

    Returns
    -------
    dict with ``best_weights``, ``best_score``.
    """
    from .consensus import ConsensusEngine

    names = sorted(detector_results.keys())
    n = len(names)
    step_vals = np.linspace(0, 1, weight_steps)

    # Generate weight combinations that sum to ~1
    combos = list(itertools.product(step_vals, repeat=n))
    combos = [c for c in combos if sum(c) > 0]

    best_score = -1.0
    best_weights = {}

    for combo in combos:
        total = sum(combo)
        weights = {names[i]: combo[i] / total for i in range(n)}
        engine = ConsensusEngine(
            strategy="weighted_confidence",
            matching_distance=max_distance,
            weights=weights,
        )

        total_tp, total_fp, total_fn = 0, 0, 0
        for img_idx in range(len(images)):
            per_det = {
                name: detector_results[name][img_idx]
                for name in names
            }
            consensus = engine.combine(per_det, images[img_idx].shape)
            tp, fp, fn = PunctaMetrics.match_detections(
                consensus.coordinates,
                ground_truths[img_idx],
                max_distance=max_distance,
            )
            total_tp += len(tp)
            total_fp += len(fp)
            total_fn += len(fn)

        prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        if f1 > best_score:
            best_score = f1
            best_weights = weights.copy()

    return {"best_weights": best_weights, "best_score": best_score}
