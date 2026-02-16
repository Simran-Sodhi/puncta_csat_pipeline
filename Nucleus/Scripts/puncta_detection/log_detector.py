#!/usr/bin/env python3
"""
log_detector.py — Punctatools-style LoG-based puncta detector.

Uses Laplacian-of-Gaussian (LoG) blob detection followed by background
intensity refinement and Voronoi / watershed segmentation to produce
both puncta coordinates and a label mask.

This is the recommended classical detector for fluorescence microscopy
images where puncta appear as bright spots of varying size.
"""

import numpy as np
from skimage import feature, filters, measure, morphology, segmentation
from skimage.segmentation import relabel_sequential, watershed

from .core import BaseDetector, PunctaDetectionResult
from .preprocessing import preprocess_pipeline


class LoGDetector(BaseDetector):
    """Punctatools-style LoG blob detection with segmentation.

    Parameters
    ----------
    log_threshold : float
        Absolute threshold for LoG blob detection.  Lower values detect
        fainter puncta but increase false positives.
    min_sigma, max_sigma : float
        Scale range for multi-scale LoG (in pixels).
    num_sigma : int
        Number of intermediate sigma values.
    bg_percentile : float
        Percentile of local background used to filter detections.
        A detection is rejected if its peak intensity is below
        ``bg_percentile`` of the surrounding annulus.
    min_size, max_size : int
        Area filters applied after watershed segmentation.
    use_watershed : bool
        If True, segment each punctum with marker-controlled watershed;
        otherwise return only centroid coordinates with no label mask.

    Pre-processing
    ~~~~~~~~~~~~~~
    The detector applies its own preprocessing pipeline.  You can
    override individual steps via ``preprocess_kw``.
    """

    def __init__(
        self,
        log_threshold: float = 0.01,
        min_sigma: float = 1.0,
        max_sigma: float = 5.0,
        num_sigma: int = 5,
        bg_percentile: float = 50.0,
        bg_rejection: bool = True,
        min_size: int = 3,
        max_size: int = 500,
        use_watershed: bool = True,
        # Pre-processing defaults
        preprocess_kw: dict | None = None,
    ):
        self.log_threshold = log_threshold
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.num_sigma = num_sigma
        self.bg_percentile = bg_percentile
        self.bg_rejection = bg_rejection
        self.min_size = min_size
        self.max_size = max_size
        self.use_watershed = use_watershed
        self.preprocess_kw = preprocess_kw or {}

    # -------------------------------------------------------------- #
    #  BaseDetector interface
    # -------------------------------------------------------------- #

    def get_name(self) -> str:
        return "LoG (Punctatools-style)"

    def get_default_params(self) -> dict:
        return {
            "log_threshold": 0.01,
            "min_sigma": 1.0,
            "max_sigma": 5.0,
            "num_sigma": 5,
            "bg_percentile": 50.0,
            "bg_rejection": True,
            "min_size": 3,
            "max_size": 500,
            "use_watershed": True,
        }

    def detect_2d(
        self,
        image: np.ndarray,
        mask: np.ndarray | None = None,
        **kwargs,
    ) -> PunctaDetectionResult:
        # Pre-process
        img = preprocess_pipeline(image, **self.preprocess_kw)

        # LoG blob detection ----------------------------------------
        blobs = feature.blob_log(
            img,
            min_sigma=self.min_sigma,
            max_sigma=self.max_sigma,
            num_sigma=self.num_sigma,
            threshold=self.log_threshold,
        )
        if len(blobs) == 0:
            return PunctaDetectionResult.empty()

        coords = blobs[:, :2]    # (y, x)
        sigmas = blobs[:, 2]
        radii = sigmas * np.sqrt(2)

        # Filter by mask --------------------------------------------
        if mask is not None:
            keep = []
            for i, (y, x) in enumerate(coords):
                yi, xi = int(round(y)), int(round(x))
                if 0 <= yi < mask.shape[0] and 0 <= xi < mask.shape[1]:
                    if mask[yi, xi] > 0:
                        keep.append(i)
            if not keep:
                return PunctaDetectionResult.empty()
            coords = coords[keep]
            sigmas = sigmas[keep]
            radii = radii[keep]

        # Background rejection --------------------------------------
        if self.bg_rejection:
            keep = self._bg_filter(image.astype(np.float32), coords, radii)
            coords = coords[keep]
            sigmas = sigmas[keep]
            radii = radii[keep]
            if len(coords) == 0:
                return PunctaDetectionResult.empty()

        # Confidence = LoG response at detected scale ---------------
        confidences = np.array([
            img[int(round(y)), int(round(x))]
            for y, x in coords
        ], dtype=np.float64)

        # Segmentation via watershed --------------------------------
        labels = None
        if self.use_watershed:
            labels = self._watershed_segment(img, coords, radii, mask)

        return PunctaDetectionResult(
            coordinates=coords.astype(np.float64),
            confidences=confidences,
            radii=radii.astype(np.float64),
            labels=labels,
            metadata={
                "detector": self.get_name(),
                "log_threshold": self.log_threshold,
                "n_raw_blobs": len(blobs),
            },
        )

    # -------------------------------------------------------------- #
    #  Internals
    # -------------------------------------------------------------- #

    def _bg_filter(self, raw, coords, radii):
        """Reject detections whose peak is below local background."""
        keep = []
        h, w = raw.shape
        for i, (y, x) in enumerate(coords):
            r = max(int(round(radii[i])), 1)
            yi, xi = int(round(y)), int(round(x))

            # Inner disk (punctum)
            yy, xx = np.ogrid[-r:r+1, -r:r+1]
            inner = yy**2 + xx**2 <= r**2

            # Annulus (ring r .. 2r)
            outer_r = r * 2
            yy2, xx2 = np.ogrid[-outer_r:outer_r+1, -outer_r:outer_r+1]
            ring = (yy2**2 + xx2**2 > r**2) & (yy2**2 + xx2**2 <= outer_r**2)

            # Sample annulus
            ring_vals = []
            for dy in range(-outer_r, outer_r + 1):
                for dx in range(-outer_r, outer_r + 1):
                    ry, rx = yi + dy, xi + dx
                    if 0 <= ry < h and 0 <= rx < w:
                        if (dy * dy + dx * dx > r * r and
                                dy * dy + dx * dx <= outer_r * outer_r):
                            ring_vals.append(raw[ry, rx])

            if ring_vals:
                bg = float(np.percentile(ring_vals, self.bg_percentile))
                peak = float(raw[yi, xi])
                if peak > bg:
                    keep.append(i)
            else:
                keep.append(i)  # keep if we can't compute bg
        return keep

    def _watershed_segment(self, img, coords, radii, mask):
        """Marker-controlled watershed around detected centres."""
        markers = np.zeros(img.shape, dtype=np.int32)
        for i, (y, x) in enumerate(coords):
            yi, xi = int(round(y)), int(round(x))
            if 0 <= yi < img.shape[0] and 0 <= xi < img.shape[1]:
                markers[yi, xi] = i + 1

        # Invert image so watershed flows into bright regions
        inv = img.max() - img
        labels = watershed(inv, markers=markers, mask=(mask > 0) if mask is not None else None)

        # Size filter
        if self.min_size > 0 or self.max_size > 0:
            for prop in measure.regionprops(labels):
                if self.min_size > 0 and prop.area < self.min_size:
                    labels[labels == prop.label] = 0
                if self.max_size > 0 and prop.area > self.max_size:
                    labels[labels == prop.label] = 0
            labels, _, _ = relabel_sequential(labels)

        return labels.astype(np.int32)
