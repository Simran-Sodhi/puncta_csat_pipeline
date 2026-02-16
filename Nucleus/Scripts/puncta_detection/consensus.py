#!/usr/bin/env python3
"""
consensus.py — Multi-detector consensus engine.

Combines results from 2+ detectors into a unified, high-confidence
detection set using configurable strategies.
"""

import numpy as np
from scipy.spatial import cKDTree
from skimage import measure
from skimage.draw import disk as skdisk
from skimage.segmentation import relabel_sequential

from .core import PunctaDetectionResult


class ConsensusEngine:
    """Combine puncta detections from multiple detectors.

    Parameters
    ----------
    strategy : str
        ``"union"`` — keep all unique detections from any detector.
        ``"intersection"`` — keep only detections found by *all* detectors.
        ``"majority_vote"`` — keep detections found by >= ceil(N/2) detectors.
        ``"weighted_confidence"`` — weighted-sum confidence; threshold on
        combined score.
    matching_distance : float
        Maximum distance (pixels) to consider two detections from
        different detectors as the same punctum.
    weights : dict or None
        ``{detector_name: weight}`` for the weighted-confidence strategy.
        If *None*, all detectors are weighted equally.
    confidence_threshold : float
        Minimum combined score for ``weighted_confidence`` strategy.
    """

    STRATEGIES = [
        "union",
        "intersection",
        "majority_vote",
        "weighted_confidence",
    ]

    def __init__(
        self,
        strategy: str = "weighted_confidence",
        matching_distance: float = 3.0,
        weights: dict | None = None,
        confidence_threshold: float = 0.3,
    ):
        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"Unknown strategy {strategy!r}.  Choose from {self.STRATEGIES}"
            )
        self.strategy = strategy
        self.matching_distance = matching_distance
        self.weights = weights or {}
        self.confidence_threshold = confidence_threshold

    # -------------------------------------------------------------- #
    #  Public API
    # -------------------------------------------------------------- #

    def combine(
        self,
        results: dict[str, PunctaDetectionResult],
        image_shape: tuple | None = None,
        default_radius: float = 2.0,
    ) -> PunctaDetectionResult:
        """Combine results from multiple detectors.

        Parameters
        ----------
        results : dict
            ``{detector_name: PunctaDetectionResult}``.
        image_shape : tuple or None
            ``(H, W)`` — needed to generate a consensus label mask.
        default_radius : float
            Fallback radius for mask generation when detectors don't
            return radii.

        Returns
        -------
        PunctaDetectionResult
            Unified result with per-detection agreement info in metadata.
        """
        # Flatten all detections into a master list
        all_points = []  # (y, x, confidence, detector_name, original_idx)
        for name, res in results.items():
            for i in range(res.n_detections):
                pt = res.coordinates[i]
                conf = float(res.confidences[i])
                r = float(res.radii[i]) if res.radii is not None else default_radius
                all_points.append({
                    "y": float(pt[-2]),
                    "x": float(pt[-1]),
                    "conf": conf,
                    "radius": r,
                    "detector": name,
                    "idx": i,
                })

        if not all_points:
            return PunctaDetectionResult.empty()

        # Build cross-detector match groups ----------------------------
        groups = self._build_groups(all_points, results.keys())

        # Apply strategy -----------------------------------------------
        if self.strategy == "union":
            selected = self._strategy_union(groups)
        elif self.strategy == "intersection":
            selected = self._strategy_intersection(groups, len(results))
        elif self.strategy == "majority_vote":
            selected = self._strategy_majority(groups, len(results))
        elif self.strategy == "weighted_confidence":
            selected = self._strategy_weighted(groups, results.keys())
        else:
            selected = groups  # fallback

        if not selected:
            return PunctaDetectionResult.empty()

        # Aggregate coordinates, confidences, radii --------------------
        coords, confs, radii_out, agreement = [], [], [], []
        for grp in selected:
            ys = [p["y"] for p in grp]
            xs = [p["x"] for p in grp]
            cs = [p["conf"] for p in grp]
            rs = [p["radius"] for p in grp]
            detectors_in_grp = list({p["detector"] for p in grp})

            # Average position, max confidence, mean radius
            coords.append([float(np.mean(ys)), float(np.mean(xs))])
            confs.append(float(np.max(cs)))
            radii_out.append(float(np.mean(rs)))
            agreement.append(detectors_in_grp)

        coords = np.array(coords, dtype=np.float64)
        confs = np.array(confs, dtype=np.float64)
        radii_arr = np.array(radii_out, dtype=np.float64)

        # Build consensus label mask -----------------------------------
        labels = None
        if image_shape is not None and len(image_shape) >= 2:
            labels = np.zeros(image_shape[:2], dtype=np.int32)
            for i, (y, x) in enumerate(coords):
                r = max(1, int(round(radii_arr[i])))
                rr, cc = skdisk((int(round(y)), int(round(x))), r,
                                shape=image_shape[:2])
                labels[rr, cc] = i + 1

        return PunctaDetectionResult(
            coordinates=coords,
            confidences=confs,
            radii=radii_arr,
            labels=labels,
            metadata={
                "strategy": self.strategy,
                "n_detectors": len(results),
                "per_detection_agreement": agreement,
                "matching_distance": self.matching_distance,
            },
        )

    # -------------------------------------------------------------- #
    #  Grouping logic
    # -------------------------------------------------------------- #

    def _build_groups(self, all_points, detector_names):
        """Cluster detections across detectors by spatial proximity.

        Each group contains detections from potentially multiple
        detectors that are within ``matching_distance`` of each other.
        """
        if not all_points:
            return []

        coords = np.array([[p["y"], p["x"]] for p in all_points])
        tree = cKDTree(coords)

        visited = set()
        groups = []

        # Sort by confidence descending so strongest detections seed groups
        order = sorted(range(len(all_points)), key=lambda i: -all_points[i]["conf"])

        for idx in order:
            if idx in visited:
                continue
            neighbours = tree.query_ball_point(coords[idx], self.matching_distance)
            grp = []
            for n in neighbours:
                if n not in visited:
                    visited.add(n)
                    grp.append(all_points[n])
            if grp:
                groups.append(grp)

        return groups

    # -------------------------------------------------------------- #
    #  Strategies
    # -------------------------------------------------------------- #

    def _strategy_union(self, groups):
        """Keep every group (all unique detections)."""
        return groups

    def _strategy_intersection(self, groups, n_detectors):
        """Keep groups detected by ALL detectors."""
        return [
            g for g in groups
            if len({p["detector"] for p in g}) == n_detectors
        ]

    def _strategy_majority(self, groups, n_detectors):
        """Keep groups detected by >= ceil(N/2) detectors."""
        import math
        threshold = math.ceil(n_detectors / 2)
        return [
            g for g in groups
            if len({p["detector"] for p in g}) >= threshold
        ]

    def _strategy_weighted(self, groups, detector_names):
        """Weighted confidence: score each group, threshold on combined score."""
        # Default equal weights if not specified
        names = list(detector_names)
        default_w = 1.0 / len(names) if names else 1.0
        weights = {n: self.weights.get(n, default_w) for n in names}
        total_weight = sum(weights.values()) or 1.0

        selected = []
        for grp in groups:
            score = 0.0
            for name in names:
                det_points = [p for p in grp if p["detector"] == name]
                if det_points:
                    best_conf = max(p["conf"] for p in det_points)
                    score += weights[name] * best_conf
            score /= total_weight
            if score >= self.confidence_threshold:
                selected.append(grp)

        return selected

    # -------------------------------------------------------------- #
    #  Adaptive strategy selection
    # -------------------------------------------------------------- #

    @staticmethod
    def estimate_snr(image: np.ndarray, mask: np.ndarray | None = None) -> float:
        """Estimate image signal-to-noise ratio.

        SNR = (mean_signal - mean_background) / std_background
        """
        if mask is not None:
            signal = image[mask > 0].astype(np.float64)
            background = image[mask == 0].astype(np.float64)
        else:
            # Use upper quartile as signal, lower quartile as background
            thresh = np.percentile(image, 75)
            signal = image[image >= thresh].astype(np.float64)
            background = image[image < thresh].astype(np.float64)

        if len(background) == 0 or background.std() == 0:
            return 0.0
        return float((signal.mean() - background.mean()) / background.std())

    def auto_select_strategy(self, snr: float) -> str:
        """Choose strategy based on image SNR.

        High SNR (>5): classical methods reliable → majority_vote
        Low SNR  (<3): deep-learning dominates → weighted_confidence
                       (with higher Spotiflow weight)
        Medium SNR:    balanced → weighted_confidence with defaults
        """
        if snr > 5:
            return "majority_vote"
        if snr < 3:
            return "weighted_confidence"
        return "weighted_confidence"
