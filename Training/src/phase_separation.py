"""
phase_separation.py

Backend module for estimating the critical concentration (Csat) for
protein phase separation from live-cell fluorescence images.

Two methods for Csat estimation:

  Method 1 (Binary):
    Logistic regression: P(puncta_present) vs cytoplasmic mean intensity
    Csat = intensity where P = 0.5

  Method 2 (Intensity):
    Sigmoid curve fit: puncta sum intensity vs average intensity
    of cytoplasm (cell - nucleus) OR nucleus (user-defined).
    Csat = intensity at 50% of the fitted maximum.

Pipeline:
  1. Load mEGFP fluorescence image + cell/nucleus/puncta masks
  2. Extract per-cell measurements using puncta mask directly
  3. Fit both Csat models with bootstrap confidence intervals

Designed for GUI integration — all heavy computation in functions
that accept a progress_callback(current, total) for reporting.
"""

import logging

import numpy as np
import pandas as pd
from scipy.ndimage import label as ndi_label

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
#  2. Per-cell measurement extraction (puncta mask required)
# ------------------------------------------------------------------ #

def extract_cell_measurements(fluorescence_img, cell_mask,
                              puncta_mask,
                              nucleus_mask=None,
                              progress_callback=None):
    """
    Extract per-cell measurements from mEGFP fluorescence + masks.

    Parameters
    ----------
    fluorescence_img : 2D array
        The mEGFP fluorescence channel.
    cell_mask : 2D int array
        Labeled cell mask (0=background, 1..N=cells).
    puncta_mask : 2D int array
        Labeled puncta (droplet) mask. Used directly — no auto-detection.
    nucleus_mask : 2D int array or None
        Labeled nucleus mask. If provided, cytoplasm = cell - nucleus.
    progress_callback : callable(current, total) or None

    Returns
    -------
    df : pd.DataFrame with columns:
        cell_id, cell_area, total_cell_intensity,
        cytoplasm_mean_intensity, nucleus_mean_intensity,
        puncta_present, puncta_count, puncta_total_area,
        puncta_sum_intensity
    puncta_binary : 2D bool array
        Combined puncta mask (for visualization).
    """
    cell_ids = np.unique(cell_mask)
    cell_ids = cell_ids[cell_ids != 0]
    total = len(cell_ids)

    puncta_binary = (puncta_mask > 0)
    records = []

    for idx, cid in enumerate(cell_ids):
        if progress_callback and idx % 10 == 0:
            progress_callback(idx, total)

        cell_binary = cell_mask == cid
        cell_area = int(cell_binary.sum())

        # Total cell intensity (mEGFP)
        total_intensity = float(fluorescence_img[cell_binary].sum())

        # Puncta within this cell
        cell_puncta = puncta_binary & cell_binary
        labeled_puncta, n_puncta = ndi_label(cell_puncta)
        puncta_area = int(cell_puncta.sum())

        # Sum intensity of puncta pixels in mEGFP channel
        if cell_puncta.any():
            puncta_sum_int = float(fluorescence_img[cell_puncta].sum())
        else:
            puncta_sum_int = 0.0

        # Nucleus region
        if nucleus_mask is not None:
            nuc_binary = (nucleus_mask > 0) & cell_binary
            cyto_binary = cell_binary & ~nuc_binary
        else:
            nuc_binary = None
            cyto_binary = cell_binary

        # Cytoplasm intensity EXCLUDING puncta (unbiased for Csat)
        cyto_no_puncta = cyto_binary & ~cell_puncta
        cyto_pixels = fluorescence_img[cyto_no_puncta]
        cyto_mean = float(np.mean(cyto_pixels)) if cyto_pixels.size > 0 else np.nan

        # Nucleus mean intensity
        if nuc_binary is not None:
            nuc_no_puncta = nuc_binary & ~cell_puncta
            nuc_pixels = fluorescence_img[nuc_no_puncta]
            nuc_mean = float(np.mean(nuc_pixels)) if nuc_pixels.size > 0 else np.nan
        else:
            nuc_mean = np.nan

        records.append({
            "cell_id": int(cid),
            "cell_area": cell_area,
            "total_cell_intensity": total_intensity,
            "cytoplasm_mean_intensity": cyto_mean,
            "nucleus_mean_intensity": nuc_mean,
            "puncta_present": 1 if n_puncta > 0 else 0,
            "puncta_count": n_puncta,
            "puncta_total_area": puncta_area,
            "puncta_sum_intensity": puncta_sum_int,
        })

    if progress_callback:
        progress_callback(total, total)

    df = pd.DataFrame(records)
    return df, puncta_binary


# ------------------------------------------------------------------ #
#  3. Data cleaning
# ------------------------------------------------------------------ #

def clean_data(df, intensity_col="cytoplasm_mean_intensity", z_threshold=3.0):
    """
    Clean per-cell data before analysis.

    1. Drop rows with NaN in intensity_col.
    2. Remove outliers beyond z_threshold standard deviations.

    Returns a copy; does not modify the input.
    """
    out = df.dropna(subset=[intensity_col]).copy()
    if len(out) < 3:
        return out

    mean_c = out[intensity_col].mean()
    std_c = out[intensity_col].std()
    if std_c > 0:
        z_scores = np.abs((out[intensity_col] - mean_c) / std_c)
        out = out[z_scores <= z_threshold].copy()

    return out.reset_index(drop=True)


# ------------------------------------------------------------------ #
#  4. Method 1: Binary logistic regression  P(puncta) ~ intensity
# ------------------------------------------------------------------ #

def fit_logistic_binary(df, x_col="cytoplasm_mean_intensity",
                        n_bootstrap=1000, ci=0.95, random_state=42):
    """
    Method 1: Logistic regression P(puncta_present) ~ x_col.
    Csat = x where P = 0.5 (the decision boundary).

    Returns dict with: csat, slope, ci_low, ci_high, model, counts, method.
    """
    from sklearn.linear_model import LogisticRegression

    X = df[x_col].values.reshape(-1, 1)
    y = df["puncta_present"].values

    base = {
        "method": "binary",
        "x_col": x_col,
        "n_cells": len(df),
        "n_with_puncta": int(y.sum()),
        "n_no_puncta": int((y == 0).sum()),
    }

    if len(np.unique(y)) < 2:
        return {**base, "csat": np.nan, "slope": np.nan,
                "ci_low": np.nan, "ci_high": np.nan, "model": None,
                "error": "Need cells both with and without puncta."}

    model = LogisticRegression(solver="lbfgs", max_iter=1000)
    model.fit(X, y)

    slope = float(model.coef_[0, 0])
    intercept = float(model.intercept_[0])
    csat = -intercept / slope if slope != 0 else np.nan

    # Bootstrap
    rng = np.random.RandomState(random_state)
    csat_boots = []
    n = len(df)
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        X_b, y_b = X[idx], y[idx]
        if len(np.unique(y_b)) < 2:
            continue
        m = LogisticRegression(solver="lbfgs", max_iter=1000)
        m.fit(X_b, y_b)
        s = float(m.coef_[0, 0])
        if s != 0:
            csat_boots.append(-float(m.intercept_[0]) / s)

    alpha = (1 - ci) / 2
    if csat_boots:
        ci_low = float(np.percentile(csat_boots, 100 * alpha))
        ci_high = float(np.percentile(csat_boots, 100 * (1 - alpha)))
    else:
        ci_low = ci_high = np.nan

    return {**base, "csat": csat, "slope": slope,
            "ci_low": ci_low, "ci_high": ci_high, "model": model}


# ------------------------------------------------------------------ #
#  5. Method 2: Sigmoid fit  puncta_sum_intensity ~ avg_intensity
# ------------------------------------------------------------------ #

def _sigmoid(x, L, k, x0, b):
    """Generalised logistic: y = b + L / (1 + exp(-k*(x - x0)))"""
    return b + L / (1.0 + np.exp(-k * (x - x0)))


def fit_sigmoid_intensity(df, x_col="cytoplasm_mean_intensity",
                          n_bootstrap=1000, ci=0.95, random_state=42):
    """
    Method 2: Fit sigmoid curve to puncta_sum_intensity vs x_col.
    Csat = x0, the midpoint where the response reaches 50% of its max.

    Returns dict with: csat, L, k, x0, b, ci_low, ci_high, method.
    """
    from scipy.optimize import curve_fit

    x = df[x_col].values.astype(np.float64)
    y = df["puncta_sum_intensity"].values.astype(np.float64)

    base = {
        "method": "intensity",
        "x_col": x_col,
        "n_cells": len(df),
    }

    if len(x) < 5:
        return {**base, "csat": np.nan, "L": np.nan, "k": np.nan,
                "x0": np.nan, "b": np.nan, "ci_low": np.nan,
                "ci_high": np.nan, "popt": None,
                "error": "Need at least 5 cells for sigmoid fit."}

    # Initial guesses
    y_range = np.max(y) - np.min(y)
    x_mid = np.median(x)
    p0 = [y_range, 0.01, x_mid, np.min(y)]
    bounds = ([0, -np.inf, np.min(x) - np.ptp(x), -np.inf],
              [np.inf, np.inf, np.max(x) + np.ptp(x), np.inf])

    try:
        popt, _ = curve_fit(_sigmoid, x, y, p0=p0, bounds=bounds,
                            maxfev=10000)
    except (RuntimeError, ValueError) as e:
        return {**base, "csat": np.nan, "L": np.nan, "k": np.nan,
                "x0": np.nan, "b": np.nan, "ci_low": np.nan,
                "ci_high": np.nan, "popt": None,
                "error": f"Sigmoid fit failed: {e}"}

    L, k, x0, b = popt
    csat = x0  # midpoint of sigmoid = 50% of L

    # Bootstrap
    rng = np.random.RandomState(random_state)
    csat_boots = []
    n = len(df)
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        x_b, y_b = x[idx], y[idx]
        try:
            popt_b, _ = curve_fit(_sigmoid, x_b, y_b, p0=popt,
                                  bounds=bounds, maxfev=5000)
            csat_boots.append(popt_b[2])  # x0
        except (RuntimeError, ValueError):
            continue

    alpha = (1 - ci) / 2
    if csat_boots:
        ci_low = float(np.percentile(csat_boots, 100 * alpha))
        ci_high = float(np.percentile(csat_boots, 100 * (1 - alpha)))
    else:
        ci_low = ci_high = np.nan

    return {**base, "csat": csat, "L": float(L), "k": float(k),
            "x0": float(x0), "b": float(b),
            "ci_low": ci_low, "ci_high": ci_high, "popt": popt}


# ------------------------------------------------------------------ #
#  6. Plotting helpers (return matplotlib Figure objects)
# ------------------------------------------------------------------ #

def _make_figure(w=5, h=3.5):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(w, h))
    FigureCanvasAgg(fig)
    return fig


def plot_intensity_histogram(df):
    """Histogram of cytoplasmic intensity, coloured by puncta presence."""
    fig = _make_figure()
    ax = fig.add_subplot(111)

    vals = df["cytoplasm_mean_intensity"].dropna().values
    has = df["puncta_present"].values == 1
    no = ~has & ~np.isnan(df["cytoplasm_mean_intensity"].values)

    ax.hist(vals[no], bins=40, alpha=0.6, label="No puncta", color="#4C72B0")
    ax.hist(vals[has], bins=40, alpha=0.6, label="With puncta", color="#DD8452")
    ax.set_xlabel("Cytoplasmic Mean Intensity (mEGFP)")
    ax.set_ylabel("Cell Count")
    ax.set_title("Cytoplasmic Intensity Distribution")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_scatter_puncta_count(df):
    """Scatter: cytoplasm_mean_intensity vs puncta_count."""
    fig = _make_figure()
    ax = fig.add_subplot(111)
    ax.scatter(df["cytoplasm_mean_intensity"].values,
               df["puncta_count"].values, alpha=0.4, s=12, edgecolors="none")
    ax.set_xlabel("Cytoplasmic Mean Intensity (mEGFP)")
    ax.set_ylabel("Puncta Count")
    ax.set_title("Intensity vs Puncta Count")
    fig.tight_layout()
    return fig


def plot_method1_phase_transition(df, result):
    """
    Method 1 plot: P(puncta_present) vs cytoplasm intensity.
    Jittered raw data + binned fractions + logistic curve + Csat line.
    """
    fig = _make_figure(6, 4)
    ax = fig.add_subplot(111)

    x_col = result.get("x_col", "cytoplasm_mean_intensity")
    x = df[x_col].values
    y = df["puncta_present"].values

    # Jittered raw data
    jitter = np.random.default_rng(0).uniform(-0.03, 0.03, size=len(y))
    ax.scatter(x, y + jitter, alpha=0.15, s=8, color="gray",
               edgecolors="none", label="Cells", zorder=1)

    # Binned fractions
    n_bins = min(20, max(5, len(df) // 20))
    bins = np.linspace(np.nanmin(x), np.nanmax(x), n_bins + 1)
    for i in range(n_bins):
        mask = (x >= bins[i]) & (x < bins[i + 1])
        if mask.sum() >= 3:
            cx = (bins[i] + bins[i + 1]) / 2
            ax.scatter(cx, y[mask].mean(), color="#DD8452", s=40, zorder=3,
                       edgecolors="black", linewidths=0.5)

    # Logistic curve
    model = result.get("model")
    csat = result.get("csat", np.nan)
    if model is not None and not np.isnan(csat):
        x_curve = np.linspace(np.nanmin(x), np.nanmax(x), 300)
        y_curve = model.predict_proba(x_curve.reshape(-1, 1))[:, 1]
        ax.plot(x_curve, y_curve, color="#C44E52", linewidth=2,
                label="Logistic fit", zorder=2)
        ax.axvline(csat, color="#C44E52", linestyle="--", linewidth=1.5,
                   label=f"Csat = {csat:.1f}", zorder=2)
        ci_lo = result.get("ci_low", np.nan)
        ci_hi = result.get("ci_high", np.nan)
        if not np.isnan(ci_lo) and not np.isnan(ci_hi):
            ax.axvspan(ci_lo, ci_hi, alpha=0.12, color="#C44E52")

    ax.set_xlabel("Cytoplasmic Mean Intensity (mEGFP)")
    ax.set_ylabel("P(Phase Separation)")
    ax.set_title("Method 1: Binary Phase Transition")
    ax.set_ylim(-0.08, 1.08)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


def plot_method2_sigmoid(df, result):
    """
    Method 2 plot: puncta_sum_intensity vs avg intensity (cyto or nuc).
    Raw scatter + fitted sigmoid + Csat line.
    """
    fig = _make_figure(6, 4)
    ax = fig.add_subplot(111)

    x_col = result.get("x_col", "cytoplasm_mean_intensity")
    x = df[x_col].values
    y = df["puncta_sum_intensity"].values

    ax.scatter(x, y, alpha=0.3, s=12, color="#4C72B0", edgecolors="none",
               label="Cells", zorder=1)

    popt = result.get("popt")
    csat = result.get("csat", np.nan)
    if popt is not None and not np.isnan(csat):
        x_curve = np.linspace(np.nanmin(x), np.nanmax(x), 300)
        y_curve = _sigmoid(x_curve, *popt)
        ax.plot(x_curve, y_curve, color="#C44E52", linewidth=2,
                label="Sigmoid fit", zorder=2)
        ax.axvline(csat, color="#C44E52", linestyle="--", linewidth=1.5,
                   label=f"Csat = {csat:.1f}", zorder=2)
        ci_lo = result.get("ci_low", np.nan)
        ci_hi = result.get("ci_high", np.nan)
        if not np.isnan(ci_lo) and not np.isnan(ci_hi):
            ax.axvspan(ci_lo, ci_hi, alpha=0.12, color="#C44E52")

    x_label = ("Nucleus Mean Intensity (mEGFP)" if "nucleus" in x_col
               else "Cytoplasmic Mean Intensity (mEGFP)")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Sum Puncta Intensity (mEGFP)")
    ax.set_title("Method 2: Intensity Sigmoid Fit")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


def plot_overlay(fluorescence_img, cell_mask, puncta_mask):
    """
    Overlay visualization: fluorescence + cell outlines + puncta.
    Returns a matplotlib Figure.
    """
    from skimage.segmentation import find_boundaries

    fig = _make_figure(6, 5)
    ax = fig.add_subplot(111)

    vmin, vmax = np.percentile(fluorescence_img, (1, 99.5))
    disp = np.clip((fluorescence_img - vmin) / (vmax - vmin + 1e-8), 0, 1)
    ax.imshow(disp, cmap="gray")

    if cell_mask.max() > 0:
        boundaries = find_boundaries(cell_mask, mode="outer")
        overlay = np.zeros((*cell_mask.shape, 4), dtype=np.float32)
        overlay[boundaries] = [0, 1, 1, 0.6]  # cyan
        ax.imshow(overlay)

    if isinstance(puncta_mask, np.ndarray):
        pm = puncta_mask > 0 if puncta_mask.dtype != bool else puncta_mask
        if pm.any():
            drop_overlay = np.zeros((*pm.shape, 4), dtype=np.float32)
            drop_overlay[pm] = [1, 0, 1, 0.5]  # magenta
            ax.imshow(drop_overlay)

    ax.set_title("Cells + Puncta (mEGFP)")
    ax.axis("off")
    fig.tight_layout()
    return fig
