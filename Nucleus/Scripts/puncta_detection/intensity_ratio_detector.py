#!/usr/bin/env python3
"""
intensity_ratio_detector.py — PunctaFinder-style intensity-ratio detector.

Detects puncta using three complementary criteria evaluated at every
candidate pixel position:

1. **Local intensity ratio** — punctum intensity vs. surrounding ring.
2. **Global intensity ratio** — punctum intensity vs. whole cell mean.
3. **Coefficient of variation (CV)** — homogeneity within the punctum disk.

This approach is robust to uneven cytoplasmic background and does not
require any training data for the basic mode.  An ``auto_threshold``
mode is available that sweeps thresholds on a labelled training set
to minimise a weighted false-positive / false-negative statistic.

Reference: Terpstra et al., 2024 — PunctaFinder algorithm.
"""

import numpy as np
from skimage import measure, morphology
from skimage.draw import disk as skdisk
from skimage.segmentation import relabel_sequential

from .core import BaseDetector, PunctaDetectionResult
from .preprocessing import preprocess_pipeline


class IntensityRatioDetector(BaseDetector):
    """PunctaFinder-style three-criteria puncta detector.

    Parameters
    ----------
    punctum_radius : int
        Expected punctum radius in pixels.  Determines the inner disk
        used for intensity measurement.
    ring_width : int
        Width of the surrounding annulus (ring) used for local
        background measurement (default: same as punctum_radius).
    t_local : float
        Threshold for the local intensity ratio
        (punctum_mean / ring_mean).
    t_global : float
        Threshold for the global intensity ratio
        (punctum_mean / cell_mean).
    t_cv : float
        Maximum coefficient of variation within the punctum disk
        (lower = more homogeneous puncta only).
    min_distance : int
        Minimum distance between detected puncta centres to suppress
        overlapping candidates.
    step : int
        Pixel step for scanning candidates (1 = every pixel, 2 = skip
        alternate pixels for speed, etc.).
    """

    def __init__(
        self,
        punctum_radius: int = 3,
        ring_width: int = 0,
        t_local: float = 1.5,
        t_global: float = 1.5,
        t_cv: float = 0.5,
        min_distance: int = 3,
        step: int = 1,
        preprocess_kw: dict | None = None,
    ):
        self.punctum_radius = punctum_radius
        self.ring_width = ring_width if ring_width > 0 else punctum_radius
        self.t_local = t_local
        self.t_global = t_global
        self.t_cv = t_cv
        self.min_distance = min_distance
        self.step = max(1, step)
        self.preprocess_kw = preprocess_kw or {}

    def get_name(self) -> str:
        return "Intensity-Ratio (PunctaFinder-style)"

    def get_default_params(self) -> dict:
        return {
            "punctum_radius": 3,
            "ring_width": 3,
            "t_local": 1.5,
            "t_global": 1.5,
            "t_cv": 0.5,
            "min_distance": 3,
            "step": 1,
        }

    # -------------------------------------------------------------- #
    #  Detection
    # -------------------------------------------------------------- #

    def detect_2d(
        self,
        image: np.ndarray,
        mask: np.ndarray | None = None,
        **kwargs,
    ) -> PunctaDetectionResult:
        raw = image.astype(np.float32)
        img = preprocess_pipeline(raw, **self.preprocess_kw)

        h, w = img.shape
        r = self.punctum_radius
        rw = self.ring_width
        outer_r = r + rw

        # Pre-compute disk / ring templates relative to (0, 0)
        yy, xx = np.ogrid[-outer_r:outer_r+1, -outer_r:outer_r+1]
        dist2 = yy**2 + xx**2
        inner_template = dist2 <= r**2
        ring_template = (dist2 > r**2) & (dist2 <= outer_r**2)

        # Global cell mean (within mask if provided)
        if mask is not None:
            cell_pixels = raw[mask > 0]
        else:
            cell_pixels = raw.ravel()
        if len(cell_pixels) == 0:
            return PunctaDetectionResult.empty()
        global_mean = float(cell_pixels.mean())
        if global_mean <= 0:
            global_mean = 1e-8

        # Candidate scan -----------------------------------------------
        candidates = []  # (y, x, confidence)
        pad = outer_r
        for cy in range(pad, h - pad, self.step):
            for cx in range(pad, w - pad, self.step):
                # Skip if outside mask
                if mask is not None and mask[cy, cx] == 0:
                    continue

                patch = raw[cy - outer_r:cy + outer_r + 1,
                            cx - outer_r:cx + outer_r + 1]

                inner_vals = patch[inner_template]
                ring_vals = patch[ring_template]

                if len(inner_vals) == 0 or len(ring_vals) == 0:
                    continue

                inner_mean = float(inner_vals.mean())
                ring_mean = float(ring_vals.mean())
                inner_std = float(inner_vals.std())

                # Criterion 1: local ratio
                if ring_mean <= 0:
                    ring_mean = 1e-8
                local_ratio = inner_mean / ring_mean
                if local_ratio < self.t_local:
                    continue

                # Criterion 2: global ratio
                global_ratio = inner_mean / global_mean
                if global_ratio < self.t_global:
                    continue

                # Criterion 3: CV (puncta should be relatively uniform)
                cv = inner_std / inner_mean if inner_mean > 0 else 999.0
                if cv > self.t_cv:
                    continue

                # Confidence = geometric mean of the two ratios
                conf = float(np.sqrt(local_ratio * global_ratio))
                candidates.append((cy, cx, conf))

        if not candidates:
            return PunctaDetectionResult.empty()

        # Non-maximum suppression (NMS) --------------------------------
        candidates.sort(key=lambda c: -c[2])
        selected = []
        suppressed = set()
        for i, (cy, cx, conf) in enumerate(candidates):
            if i in suppressed:
                continue
            selected.append((cy, cx, conf))
            for j in range(i + 1, len(candidates)):
                if j in suppressed:
                    continue
                oy, ox, _ = candidates[j]
                if (cy - oy) ** 2 + (cx - ox) ** 2 <= self.min_distance ** 2:
                    suppressed.add(j)

        coords = np.array([(y, x) for y, x, _ in selected], dtype=np.float64)
        confidences = np.array([c for _, _, c in selected], dtype=np.float64)
        radii = np.full(len(selected), self.punctum_radius, dtype=np.float64)

        # Build label mask from detected centres -----------------------
        labels = np.zeros(img.shape, dtype=np.int32)
        for i, (y, x) in enumerate(coords):
            rr, cc = skdisk((int(y), int(x)), r, shape=img.shape)
            labels[rr, cc] = i + 1

        return PunctaDetectionResult(
            coordinates=coords,
            confidences=confidences,
            radii=radii,
            labels=labels,
            metadata={
                "detector": self.get_name(),
                "t_local": self.t_local,
                "t_global": self.t_global,
                "t_cv": self.t_cv,
                "n_candidates_before_nms": len(candidates),
            },
        )
