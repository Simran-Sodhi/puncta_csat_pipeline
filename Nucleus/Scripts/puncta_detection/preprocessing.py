#!/usr/bin/env python3
"""
preprocessing.py — Image preprocessing for puncta detection.

Provides background subtraction, normalisation, and denoising steps
that feed into any detector.
"""

import numpy as np
from skimage import exposure, filters, morphology


def normalize(
    image: np.ndarray,
    method: str = "percentile",
    low: float = 1.0,
    high: float = 99.8,
) -> np.ndarray:
    """Normalise intensity to [0, 1].

    Parameters
    ----------
    method : {"percentile", "minmax", "zscore", "clahe"}
    """
    img = image.astype(np.float32)
    if method == "percentile":
        lo = np.percentile(img, low)
        hi = np.percentile(img, high)
        if hi <= lo:
            return np.zeros_like(img)
        return np.clip((img - lo) / (hi - lo), 0, 1)
    if method == "minmax":
        lo, hi = img.min(), img.max()
        if hi <= lo:
            return np.zeros_like(img)
        return (img - lo) / (hi - lo)
    if method == "zscore":
        mu, sigma = img.mean(), img.std()
        if sigma == 0:
            return np.zeros_like(img)
        return (img - mu) / sigma
    if method == "clahe":
        img_u8 = exposure.rescale_intensity(img, out_range=(0, 1))
        return exposure.equalize_adapthist(img_u8, clip_limit=0.02)
    raise ValueError(f"Unknown normalisation method: {method!r}")


def subtract_background(
    image: np.ndarray,
    method: str = "white_tophat",
    radius: int = 15,
) -> np.ndarray:
    """Remove uneven background illumination.

    Parameters
    ----------
    method : {"white_tophat", "rolling_ball", "gaussian", "median"}
    radius : int
        Structuring element radius (white_tophat, rolling_ball) or
        sigma (gaussian) or kernel size (median).
    """
    img = image.astype(np.float32)
    if method == "white_tophat":
        se = morphology.disk(radius)
        return morphology.white_tophat(img, se)
    if method == "rolling_ball":
        # Approximate rolling-ball with morphological opening
        se = morphology.disk(radius)
        bg = morphology.opening(img, se)
        return np.clip(img - bg, 0, None)
    if method == "gaussian":
        bg = filters.gaussian(img, sigma=radius)
        return np.clip(img - bg, 0, None)
    if method == "median":
        from skimage.filters import median as _median
        se = morphology.disk(radius)
        bg = _median(img, se).astype(np.float32)
        return np.clip(img - bg, 0, None)
    raise ValueError(f"Unknown background method: {method!r}")


def denoise(
    image: np.ndarray,
    method: str = "gaussian",
    sigma: float = 0.5,
) -> np.ndarray:
    """Smooth / denoise the image.

    Parameters
    ----------
    method : {"gaussian", "median"}
    sigma : float
        Gaussian sigma or median disk radius.
    """
    img = image.astype(np.float32)
    if method == "gaussian":
        return filters.gaussian(img, sigma=sigma)
    if method == "median":
        se = morphology.disk(max(1, int(round(sigma))))
        from skimage.filters import median as _median
        return _median(img, se).astype(np.float32)
    raise ValueError(f"Unknown denoise method: {method!r}")


def preprocess_pipeline(
    image: np.ndarray,
    bg_method: str = "white_tophat",
    bg_radius: int = 15,
    bg_enabled: bool = True,
    denoise_method: str = "gaussian",
    denoise_sigma: float = 1.0,
    denoise_enabled: bool = True,
    norm_method: str = "percentile",
    norm_low: float = 1.0,
    norm_high: float = 99.8,
) -> np.ndarray:
    """Run the full preprocessing pipeline.

    Returns a float32 image in [0, 1].
    """
    img = image.astype(np.float32)
    if bg_enabled and bg_radius > 0:
        img = subtract_background(img, method=bg_method, radius=bg_radius)
    if denoise_enabled and denoise_sigma > 0:
        img = denoise(img, method=denoise_method, sigma=denoise_sigma)
    img = normalize(img, method=norm_method, low=norm_low, high=norm_high)
    return img
