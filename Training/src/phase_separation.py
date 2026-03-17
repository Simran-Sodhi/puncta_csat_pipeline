"""
phase_separation.py

Backend module for estimating the critical concentration (C*) for
protein phase separation from live-cell fluorescence images.

Pipeline:
  1. Extract per-cell measurements from fluorescence image + masks
  2. Detect droplets (condensates) within each cell
  3. Compute cytoplasmic intensity excluding droplets
  4. Fit logistic regression: P(droplet) vs cytoplasmic intensity
  5. Estimate C* (50% probability threshold) with bootstrap CI

Designed for GUI integration — all heavy computation in functions
that accept a progress_callback(current, total) for reporting.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, label as ndi_label
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects, binary_opening, disk

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  1. Image / mask loading helpers
# ------------------------------------------------------------------ #

def load_image_channel(path, channel_index=0, z_index=0):
    """Load a single 2D channel from a (possibly multi-dim) TIFF."""
    import tifffile as tiff
    with tiff.TiffFile(str(path)) as tf:
        data = tf.series[0].asarray()
        axes = getattr(tf.series[0], "axes", None)

    if axes is not None:
        sl = [slice(None)] * len(axes)
        if "T" in axes:
            sl[axes.index("T")] = 0
        if "C" in axes:
            sl[axes.index("C")] = channel_index
        if "Z" in axes:
            sl[axes.index("Z")] = z_index
        img = np.squeeze(data[tuple(sl)])
        if img.ndim == 2:
            return img.astype(np.float64)

    # Fallback: shape heuristics
    if data.ndim == 2:
        return data.astype(np.float64)
    if data.ndim == 3:
        if data.shape[0] <= 4:
            return data[channel_index].astype(np.float64)
        if data.shape[-1] <= 4:
            return data[:, :, channel_index].astype(np.float64)
    if data.ndim >= 4:
        return data[0, channel_index].astype(np.float64)
    raise ValueError(f"Cannot extract channel {channel_index} from shape {data.shape}")


def load_mask_2d(path):
    """Load a label mask and squeeze to 2D."""
    import tifffile as tiff
    arr = tiff.imread(str(path))
    while arr.ndim > 2:
        arr = arr[0]
    return arr.astype(np.int32)


# ------------------------------------------------------------------ #
#  2. Droplet detection
# ------------------------------------------------------------------ #

def detect_droplets_in_cell(fluorescence, cell_binary,
                            sigma=1.5,
                            threshold_factor=2.0,
                            min_droplet_area=5,
                            min_circularity=0.3):
    """
    Detect bright puncta (phase-separated droplets) within a single cell.

    Parameters
    ----------
    fluorescence : 2D array
        Fluorescence channel (full image).
    cell_binary : 2D bool array
        Binary mask of the cell region.
    sigma : float
        Gaussian smoothing sigma before thresholding.
    threshold_factor : float
        Droplets must exceed (background_mean + factor * background_std).
    min_droplet_area : int
        Minimum area in pixels for a valid droplet.
    min_circularity : float
        Minimum circularity (4*pi*area / perimeter^2) to keep.
        Set 0 to disable.

    Returns
    -------
    droplet_mask : 2D bool array (same shape as fluorescence)
        Binary mask of detected droplets within this cell.
    """
    cell_pixels = fluorescence[cell_binary]
    if cell_pixels.size == 0:
        return np.zeros_like(cell_binary)

    # Smooth and threshold
    smoothed = gaussian_filter(fluorescence, sigma=sigma)
    bg_mean = np.mean(cell_pixels)
    bg_std = np.std(cell_pixels)
    thresh = bg_mean + threshold_factor * bg_std

    bright = (smoothed > thresh) & cell_binary

    # Morphological opening to remove noise
    if bright.any():
        bright = binary_opening(bright, footprint=disk(1))

    # Label connected components and filter
    labeled, n_obj = ndi_label(bright)
    droplet_mask = np.zeros_like(cell_binary)

    if n_obj == 0:
        return droplet_mask

    # Remove small objects
    labeled = remove_small_objects(labeled, min_size=min_droplet_area)

    # Filter by circularity
    for prop in regionprops(labeled):
        if prop.perimeter > 0 and min_circularity > 0:
            circ = 4 * np.pi * prop.area / (prop.perimeter ** 2)
            if circ < min_circularity:
                continue
        droplet_mask[labeled == prop.label] = True

    return droplet_mask


# ------------------------------------------------------------------ #
#  3. Per-cell measurement extraction
# ------------------------------------------------------------------ #

def extract_cell_measurements(fluorescence_img, cell_mask,
                              nucleus_mask=None,
                              puncta_mask=None,
                              sigma=1.5,
                              threshold_factor=2.0,
                              min_droplet_area=5,
                              min_circularity=0.3,
                              progress_callback=None):
    """
    Extract per-cell measurements from a fluorescence image.

    Parameters
    ----------
    fluorescence_img : 2D array
        The fluorescence channel (e.g. mEGFP, channel 1).
    cell_mask : 2D int array
        Labeled cell mask (0=background, 1..N=cells).
    nucleus_mask : 2D int array or None
        Labeled nucleus mask. If provided, cytoplasm = cell - nucleus.
    puncta_mask : 2D int array or None
        Pre-computed puncta mask. If provided, used instead of
        auto-detection for droplet identification.
    sigma, threshold_factor, min_droplet_area, min_circularity
        Parameters for auto droplet detection (used when puncta_mask
        is None).
    progress_callback : callable(current, total) or None

    Returns
    -------
    df : pd.DataFrame
        Per-cell measurements.
    combined_droplet_mask : 2D bool array
        Union of all detected droplet masks (for visualization).
    """
    cell_ids = np.unique(cell_mask)
    cell_ids = cell_ids[cell_ids != 0]
    total = len(cell_ids)

    records = []
    combined_droplet_mask = np.zeros(cell_mask.shape, dtype=bool)

    for idx, cid in enumerate(cell_ids):
        if progress_callback and idx % 10 == 0:
            progress_callback(idx, total)

        cell_binary = cell_mask == cid
        cell_area = int(cell_binary.sum())

        # Total cell intensity
        total_intensity = float(fluorescence_img[cell_binary].sum())

        # Determine cytoplasm region
        if nucleus_mask is not None:
            nuc_binary = (nucleus_mask > 0) & cell_binary
            cyto_binary = cell_binary & ~nuc_binary
        else:
            cyto_binary = cell_binary

        # Detect or use pre-computed droplets
        if puncta_mask is not None:
            # Use pre-computed puncta mask within this cell
            cell_droplets = (puncta_mask > 0) & cell_binary
        else:
            cell_droplets = detect_droplets_in_cell(
                fluorescence_img, cyto_binary,
                sigma=sigma,
                threshold_factor=threshold_factor,
                min_droplet_area=min_droplet_area,
                min_circularity=min_circularity,
            )

        combined_droplet_mask |= cell_droplets

        # Droplet metrics
        labeled_drops, n_drops = ndi_label(cell_droplets)
        droplet_count = n_drops
        droplet_total_area = int(cell_droplets.sum())

        # Cytoplasm intensity EXCLUDING droplets (critical for unbiased C*)
        cyto_no_drops = cyto_binary & ~cell_droplets
        cyto_pixels = fluorescence_img[cyto_no_drops]
        if cyto_pixels.size > 0:
            cyto_mean_intensity = float(np.mean(cyto_pixels))
        else:
            cyto_mean_intensity = np.nan

        records.append({
            "cell_id": int(cid),
            "cell_area": cell_area,
            "total_cell_intensity": total_intensity,
            "cytoplasm_mean_intensity": cyto_mean_intensity,
            "droplet_present": 1 if droplet_count > 0 else 0,
            "droplet_count": droplet_count,
            "droplet_total_area": droplet_total_area,
        })

    if progress_callback:
        progress_callback(total, total)

    df = pd.DataFrame(records)
    return df, combined_droplet_mask


# ------------------------------------------------------------------ #
#  4. Data cleaning
# ------------------------------------------------------------------ #

def clean_data(df, z_threshold=3.0):
    """
    Clean per-cell data before analysis.

    1. Drop rows with NaN cytoplasm_mean_intensity.
    2. Remove outliers beyond z_threshold standard deviations.

    Returns a copy; does not modify the input.
    """
    out = df.dropna(subset=["cytoplasm_mean_intensity"]).copy()
    if len(out) < 3:
        return out

    mean_c = out["cytoplasm_mean_intensity"].mean()
    std_c = out["cytoplasm_mean_intensity"].std()
    if std_c > 0:
        z_scores = np.abs((out["cytoplasm_mean_intensity"] - mean_c) / std_c)
        out = out[z_scores <= z_threshold].copy()

    return out.reset_index(drop=True)


# ------------------------------------------------------------------ #
#  5. Critical concentration estimation (logistic regression)
# ------------------------------------------------------------------ #

def fit_logistic(df, n_bootstrap=1000, ci=0.95, random_state=42):
    """
    Fit a logistic regression: P(droplet_present) ~ cytoplasm_mean_intensity.

    Returns
    -------
    result : dict with keys:
        c_star        : float — estimated critical concentration
        slope         : float — logistic slope parameter
        ci_low        : float — lower bound of C* CI
        ci_high       : float — upper bound of C* CI
        model         : fitted LogisticRegression object
        n_cells       : int
        n_with_drops  : int
        n_no_drops    : int
    """
    from sklearn.linear_model import LogisticRegression

    X = df["cytoplasm_mean_intensity"].values.reshape(-1, 1)
    y = df["droplet_present"].values

    # Check we have both classes
    if len(np.unique(y)) < 2:
        return {
            "c_star": np.nan,
            "slope": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "model": None,
            "n_cells": len(df),
            "n_with_drops": int(y.sum()),
            "n_no_drops": int((y == 0).sum()),
            "error": "Need cells both with and without droplets.",
        }

    # Fit on full dataset
    model = LogisticRegression(solver="lbfgs", max_iter=1000)
    model.fit(X, y)

    slope = float(model.coef_[0, 0])
    intercept = float(model.intercept_[0])
    c_star = -intercept / slope if slope != 0 else np.nan

    # Bootstrap C*
    rng = np.random.RandomState(random_state)
    c_star_boots = []
    n = len(df)
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        X_b = X[idx]
        y_b = y[idx]
        if len(np.unique(y_b)) < 2:
            continue
        m = LogisticRegression(solver="lbfgs", max_iter=1000)
        m.fit(X_b, y_b)
        s = float(m.coef_[0, 0])
        if s != 0:
            c_star_boots.append(-float(m.intercept_[0]) / s)

    alpha = (1 - ci) / 2
    if c_star_boots:
        ci_low = float(np.percentile(c_star_boots, 100 * alpha))
        ci_high = float(np.percentile(c_star_boots, 100 * (1 - alpha)))
    else:
        ci_low = ci_high = np.nan

    return {
        "c_star": c_star,
        "slope": slope,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "model": model,
        "n_cells": len(df),
        "n_with_drops": int(y.sum()),
        "n_no_drops": int((y == 0).sum()),
    }


# ------------------------------------------------------------------ #
#  6. Plotting helpers (return matplotlib Figure objects)
# ------------------------------------------------------------------ #

def plot_intensity_histogram(df):
    """Histogram of cytoplasmic intensity distribution."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = Figure(figsize=(5, 3.5))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    vals = df["cytoplasm_mean_intensity"].dropna().values
    has_drops = df["droplet_present"].values == 1
    no_drops = ~has_drops & ~np.isnan(df["cytoplasm_mean_intensity"].values)

    ax.hist(vals[no_drops], bins=40, alpha=0.6, label="No droplets", color="#4C72B0")
    ax.hist(vals[has_drops], bins=40, alpha=0.6, label="With droplets", color="#DD8452")
    ax.set_xlabel("Cytoplasmic Mean Intensity")
    ax.set_ylabel("Cell Count")
    ax.set_title("Cytoplasmic Intensity Distribution")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_scatter_droplet_count(df):
    """Scatter: cytoplasm_mean_intensity vs droplet_count."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = Figure(figsize=(5, 3.5))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    x = df["cytoplasm_mean_intensity"].values
    y = df["droplet_count"].values
    ax.scatter(x, y, alpha=0.4, s=12, edgecolors="none")
    ax.set_xlabel("Cytoplasmic Mean Intensity")
    ax.set_ylabel("Droplet Count")
    ax.set_title("Intensity vs Droplet Count")
    fig.tight_layout()
    return fig


def plot_phase_transition(df, fit_result):
    """
    Phase transition plot:
      - Raw data points (jittered)
      - Binned fraction with droplets
      - Fitted logistic curve
      - Vertical line at C*
    """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = Figure(figsize=(6, 4))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    x = df["cytoplasm_mean_intensity"].values
    y = df["droplet_present"].values

    # Jittered raw data
    jitter = np.random.default_rng(0).uniform(-0.03, 0.03, size=len(y))
    ax.scatter(x, y + jitter, alpha=0.15, s=8, color="gray",
               edgecolors="none", label="Cells", zorder=1)

    # Binned fractions
    n_bins = min(20, max(5, len(df) // 20))
    bins = np.linspace(np.nanmin(x), np.nanmax(x), n_bins + 1)
    bin_centers = []
    bin_fractions = []
    for i in range(n_bins):
        mask = (x >= bins[i]) & (x < bins[i + 1])
        if mask.sum() >= 3:
            bin_centers.append((bins[i] + bins[i + 1]) / 2)
            bin_fractions.append(y[mask].mean())
    ax.scatter(bin_centers, bin_fractions, color="#DD8452", s=40,
               zorder=3, edgecolors="black", linewidths=0.5,
               label="Binned fraction")

    # Fitted logistic curve
    model = fit_result.get("model")
    c_star = fit_result.get("c_star", np.nan)
    if model is not None and not np.isnan(c_star):
        x_curve = np.linspace(np.nanmin(x), np.nanmax(x), 300)
        y_curve = model.predict_proba(x_curve.reshape(-1, 1))[:, 1]
        ax.plot(x_curve, y_curve, color="#C44E52", linewidth=2,
                label="Logistic fit", zorder=2)

        # C* vertical line
        ax.axvline(c_star, color="#C44E52", linestyle="--", linewidth=1.5,
                   label=f"C* = {c_star:.1f}", zorder=2)
        ci_lo = fit_result.get("ci_low", np.nan)
        ci_hi = fit_result.get("ci_high", np.nan)
        if not np.isnan(ci_lo) and not np.isnan(ci_hi):
            ax.axvspan(ci_lo, ci_hi, alpha=0.12, color="#C44E52")

    ax.set_xlabel("Cytoplasmic Mean Intensity (proxy for [Protein])")
    ax.set_ylabel("P(Phase Separation)")
    ax.set_title("Phase Transition Curve")
    ax.set_ylim(-0.08, 1.08)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


def plot_overlay(fluorescence_img, cell_mask, droplet_mask):
    """
    Overlay visualization: fluorescence + cell outlines + droplets.
    Returns a matplotlib Figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from skimage.segmentation import find_boundaries

    fig = Figure(figsize=(6, 5))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    # Normalize fluorescence for display
    vmin, vmax = np.percentile(fluorescence_img, (1, 99.5))
    disp = np.clip((fluorescence_img - vmin) / (vmax - vmin + 1e-8), 0, 1)
    ax.imshow(disp, cmap="gray")

    # Cell boundaries in cyan
    if cell_mask.max() > 0:
        boundaries = find_boundaries(cell_mask, mode="outer")
        overlay = np.zeros((*cell_mask.shape, 4), dtype=np.float32)
        overlay[boundaries] = [0, 1, 1, 0.6]  # cyan
        ax.imshow(overlay)

    # Droplets in magenta
    if droplet_mask.any():
        drop_overlay = np.zeros((*droplet_mask.shape, 4), dtype=np.float32)
        drop_overlay[droplet_mask] = [1, 0, 1, 0.5]  # magenta
        ax.imshow(drop_overlay)

    ax.set_title("Cells + Detected Droplets")
    ax.axis("off")
    fig.tight_layout()
    return fig
