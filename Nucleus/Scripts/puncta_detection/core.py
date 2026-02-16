#!/usr/bin/env python3
"""
core.py — Base detector interface and standardised detection result.

Every puncta detector (LoG, intensity-ratio, Spotiflow, threshold, etc.)
must implement the :class:`BaseDetector` interface so they are
interchangeable inside the consensus engine and the GUI.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


# ------------------------------------------------------------------ #
#  Standardised detection result
# ------------------------------------------------------------------ #

@dataclass
class PunctaDetectionResult:
    """Output from any detector — the lingua franca of the pipeline.

    Attributes
    ----------
    coordinates : ndarray (N, 2) or (N, 3)
        Detected puncta centres as ``[y, x]`` (2-D) or ``[z, y, x]`` (3-D).
    confidences : ndarray (N,)
        Per-punctum confidence / intensity score.
    radii : ndarray (N,) or None
        Estimated radius per punctum (pixels).  ``None`` if the detector
        does not estimate radii.
    labels : ndarray or None
        Segmentation label mask (same spatial shape as the input image).
        Each punctum has a unique integer > 0; background is 0.
    metadata : dict
        Detector-specific information: thresholds used, runtime, etc.
    """
    coordinates: np.ndarray
    confidences: np.ndarray
    radii: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)

    # Convenience --------------------------------------------------
    @property
    def n_detections(self) -> int:
        return len(self.coordinates)

    @property
    def is_3d(self) -> bool:
        return self.coordinates.ndim == 2 and self.coordinates.shape[1] == 3

    @staticmethod
    def empty(ndim: int = 2) -> "PunctaDetectionResult":
        """Return a result with zero detections."""
        cols = 3 if ndim == 3 else 2
        return PunctaDetectionResult(
            coordinates=np.empty((0, cols), dtype=np.float64),
            confidences=np.empty(0, dtype=np.float64),
            metadata={"status": "empty"},
        )


# ------------------------------------------------------------------ #
#  Abstract base detector
# ------------------------------------------------------------------ #

class BaseDetector(ABC):
    """All puncta detectors must implement this interface."""

    @abstractmethod
    def detect_2d(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        **kwargs,
    ) -> PunctaDetectionResult:
        """Detect puncta in a single 2-D image.

        Parameters
        ----------
        image : ndarray (Y, X)
            Intensity image (any numeric dtype).
        mask : ndarray (Y, X) or None
            Optional cell / ROI mask.  Detections outside the mask are
            discarded.
        """

    def detect_3d(
        self,
        volume: np.ndarray,
        mask: Optional[np.ndarray] = None,
        **kwargs,
    ) -> PunctaDetectionResult:
        """Detect puncta in a 3-D z-stack.

        Default implementation: per-slice 2-D detection.
        Subclasses may override for native 3-D support.
        """
        all_coords, all_conf, all_radii = [], [], []
        for z in range(volume.shape[0]):
            slice_mask = mask[z] if mask is not None else None
            res = self.detect_2d(volume[z], mask=slice_mask, **kwargs)
            if res.n_detections > 0:
                z_col = np.full((res.n_detections, 1), z, dtype=res.coordinates.dtype)
                coords_3d = np.hstack([z_col, res.coordinates])
                all_coords.append(coords_3d)
                all_conf.append(res.confidences)
                if res.radii is not None:
                    all_radii.append(res.radii)

        if all_coords:
            coords = np.vstack(all_coords)
            conf = np.concatenate(all_conf)
            radii = np.concatenate(all_radii) if all_radii else None
        else:
            coords = np.empty((0, 3), dtype=np.float64)
            conf = np.empty(0, dtype=np.float64)
            radii = None

        return PunctaDetectionResult(
            coordinates=coords, confidences=conf, radii=radii,
            metadata={"method": "per_slice_2d"},
        )

    @abstractmethod
    def get_default_params(self) -> dict:
        """Return default parameter dictionary."""

    @abstractmethod
    def get_name(self) -> str:
        """Return human-readable detector name."""


# ------------------------------------------------------------------ #
#  Evaluation metrics
# ------------------------------------------------------------------ #

class PunctaMetrics:
    """Matching and evaluation metrics for puncta detection."""

    @staticmethod
    def match_detections(
        predicted: np.ndarray,
        ground_truth: np.ndarray,
        max_distance: float = 3.0,
    ) -> tuple:
        """Hungarian matching between predicted and GT coordinates.

        Returns
        -------
        tp_pairs : list of (pred_idx, gt_idx)
        fp_indices : list of int
        fn_indices : list of int
        """
        if len(predicted) == 0 and len(ground_truth) == 0:
            return [], [], []
        if len(predicted) == 0:
            return [], [], list(range(len(ground_truth)))
        if len(ground_truth) == 0:
            return [], list(range(len(predicted))), []

        cost = cdist(predicted, ground_truth)
        row_ind, col_ind = linear_sum_assignment(cost)

        tp_pairs = []
        matched_pred = set()
        matched_gt = set()

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] <= max_distance:
                tp_pairs.append((r, c))
                matched_pred.add(r)
                matched_gt.add(c)

        fp_indices = [i for i in range(len(predicted)) if i not in matched_pred]
        fn_indices = [i for i in range(len(ground_truth)) if i not in matched_gt]

        return tp_pairs, fp_indices, fn_indices

    @staticmethod
    def compute_f1(
        predicted: np.ndarray,
        ground_truth: np.ndarray,
        max_distance: float = 3.0,
    ) -> dict:
        """Compute precision, recall, F1 at a given matching distance."""
        tp, fp, fn = PunctaMetrics.match_detections(
            predicted, ground_truth, max_distance,
        )
        n_tp = len(tp)
        n_fp = len(fp)
        n_fn = len(fn)
        precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0.0
        recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
               if (precision + recall) > 0 else 0.0)
        return {
            "tp": n_tp, "fp": n_fp, "fn": n_fn,
            "precision": precision, "recall": recall, "f1": f1,
        }

    @staticmethod
    def localization_rmse(
        predicted: np.ndarray,
        ground_truth: np.ndarray,
        max_distance: float = 3.0,
    ) -> float:
        """RMSE of matched detection positions."""
        tp_pairs, _, _ = PunctaMetrics.match_detections(
            predicted, ground_truth, max_distance,
        )
        if not tp_pairs:
            return float("nan")
        dists = [
            np.linalg.norm(predicted[p] - ground_truth[g])
            for p, g in tp_pairs
        ]
        return float(np.sqrt(np.mean(np.array(dists) ** 2)))

    @staticmethod
    def per_cell_metrics(
        predicted: np.ndarray,
        ground_truth: np.ndarray,
        cell_masks: np.ndarray,
        max_distance: float = 3.0,
    ) -> list:
        """Compute metrics per individual cell label.

        Returns a list of dicts, one per cell.
        """
        from skimage.measure import regionprops
        results = []
        for prop in regionprops(cell_masks):
            lab = prop.label
            mask = cell_masks == lab

            # Filter coords to this cell
            pred_in = np.array([
                c for c in predicted
                if mask[int(round(c[-2])), int(round(c[-1]))]
            ]) if len(predicted) > 0 else np.empty((0, predicted.shape[1]))

            gt_in = np.array([
                c for c in ground_truth
                if mask[int(round(c[-2])), int(round(c[-1]))]
            ]) if len(ground_truth) > 0 else np.empty((0, ground_truth.shape[1]))

            m = PunctaMetrics.compute_f1(pred_in, gt_in, max_distance)
            m["cell_label"] = lab
            m["n_predicted"] = len(pred_in)
            m["n_ground_truth"] = len(gt_in)
            results.append(m)
        return results
