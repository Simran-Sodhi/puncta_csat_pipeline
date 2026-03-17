"""
phase_separation.py

Backend module for estimating the critical concentration (Csat) for
protein phase separation from live-cell fluorescence images.

Supports bulk processing: provide directories of images and masks,
files are auto-matched by name stem.

Two methods for Csat estimation:

  Method 1 (Binary):
    Logistic regression: P(puncta_present) vs cytoplasmic mean intensity
    Csat = intensity where P = 0.5

  Method 2 (Intensity):
    Sigmoid curve fit: puncta sum intensity vs average intensity
    of cytoplasm (cell - nucleus) OR nucleus (user-defined).
    Csat = intensity at 50% of the fitted maximum.

Pipeline:
  1. Match files across directories by name stem
  2. For each matched set: load mEGFP channel + masks, extract per-cell
  3. Aggregate all cells into one DataFrame
  4. Fit both Csat models with bootstrap confidence intervals
"""

import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import label as ndi_label

logger = logging.getLogger(__name__)

TIFF_EXTENSIONS = {".tif", ".tiff"}


# ------------------------------------------------------------------ #
#  1. Image / mask loading helpers
# ------------------------------------------------------------------ #

def get_ome_channel_names(path):
    """Read channel names from OME-TIFF metadata.

    Returns
    -------
    list[str] : channel names in order, e.g. ["DIC", "Cy5", "GFP", "Cy3"].
                Returns empty list if no OME metadata is found.
    """
    import xml.etree.ElementTree as ET
    import tifffile as tiff

    path = Path(path)
    names = []
    try:
        with tiff.TiffFile(str(path)) as tf:
            if not tf.ome_metadata:
                return names
            root = ET.fromstring(tf.ome_metadata)
    except Exception:
        return names

    # Try common OME namespace URIs
    for ns_uri in [
        "http://www.openmicroscopy.org/Schemas/OME/2016-06",
        "",
    ]:
        ns_prefix = f"{{{ns_uri}}}" if ns_uri else ""
        # Find all Channel elements within the first Image/Pixels
        for pixels in root.iter(f"{ns_prefix}Pixels"):
            for ch in pixels.iter(f"{ns_prefix}Channel"):
                name = ch.get("Name") or ch.get("ID") or f"ch{len(names)}"
                names.append(name)
            break  # only first Pixels element
        if names:
            break
    return names


def resolve_channel_index(path, channel_index=0, channel_name=None):
    """Resolve a channel index, optionally by name from OME metadata.

    If *channel_name* is provided (e.g. ``"GFP"``), the OME channel names
    are read and a case-insensitive substring match is performed.  If the
    name is not found the function falls back to *channel_index*.

    Returns
    -------
    int : resolved 0-based channel index
    str or None : matched channel name (None if matched by index only)
    """
    if channel_name:
        ome_names = get_ome_channel_names(path)
        if ome_names:
            target = channel_name.strip().lower()
            for idx, n in enumerate(ome_names):
                if target in n.lower():
                    return idx, n
            logger.warning(
                "Channel name '%s' not found in OME metadata %s; "
                "falling back to index %d", channel_name, ome_names, channel_index)
    return channel_index, None


def load_image_channel(path, channel_index=0, z_index=0, channel_name=None):
    """Load a single 2D channel from a (possibly multi-dim) TIFF.

    Parameters
    ----------
    path : str or Path
    channel_index : int — 0-based fallback index
    z_index : int — Z-slice to extract
    channel_name : str or None — if given, resolve index from OME metadata
        (case-insensitive substring match, e.g. "GFP", "Cy5")
    """
    import tifffile as tiff

    resolved_idx, matched = resolve_channel_index(
        path, channel_index=channel_index, channel_name=channel_name)
    if matched:
        logger.info("Channel '%s' resolved to index %d (OME name: %s)",
                     channel_name, resolved_idx, matched)

    channel_index = resolved_idx

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
#  2. Bulk file matching
# ------------------------------------------------------------------ #

def _normalize_stem(filename):
    """
    Extract a normalised stem for matching files across directories.

    Strips common suffixes like _mask, _cp_masks, _seg, _nuclei, _puncta,
    _cell, channel indicators, and the file extension.
    """
    stem = Path(filename).stem
    # Remove .ome if present (e.g. file.ome.tif -> stem is file.ome)
    if stem.lower().endswith(".ome"):
        stem = stem[:-4]
    # Strip common mask/channel suffixes (greedy, order matters)
    stem = re.sub(
        r'(_cp_masks|_masks|_mask|_seg|_nuclei|_nucleus|_nuc'
        r'|_puncta|_cell|_cells|_labels?'
        r'|_ch\d+|_c\d+|_mEGFP|_mScarlet|_Cy5|_DIC)$',
        '', stem, flags=re.IGNORECASE)
    return stem.lower()


def list_tiffs(directory):
    """Return dict {normalized_stem: full_path} for all TIFFs in directory."""
    result = {}
    d = Path(directory)
    if not d.is_dir():
        return result
    for f in sorted(d.iterdir()):
        if f.is_file() and f.suffix.lower() in TIFF_EXTENSIONS:
            stem = _normalize_stem(f.name)
            result[stem] = str(f)
    return result


def match_files(image_dir, cell_mask_dir, puncta_mask_dir,
                nucleus_mask_dir=None):
    """
    Auto-match files across directories by normalised name stem.

    Returns
    -------
    matched : list of dict
        Each dict has keys: 'stem', 'image', 'cell_mask', 'puncta_mask',
        and optionally 'nucleus_mask'.
    warnings : list of str
        Files that could not be matched.
    """
    images = list_tiffs(image_dir)
    cells = list_tiffs(cell_mask_dir)
    puncta = list_tiffs(puncta_mask_dir)
    nuclei = list_tiffs(nucleus_mask_dir) if nucleus_mask_dir else {}

    # Intersect on stems present in all required directories
    common = set(images.keys()) & set(cells.keys()) & set(puncta.keys())
    if nucleus_mask_dir:
        common = common & set(nuclei.keys())

    matched = []
    for stem in sorted(common):
        entry = {
            "stem": stem,
            "image": images[stem],
            "cell_mask": cells[stem],
            "puncta_mask": puncta[stem],
        }
        if nucleus_mask_dir and stem in nuclei:
            entry["nucleus_mask"] = nuclei[stem]
        matched.append(entry)

    # Warnings for unmatched
    warnings = []
    all_stems = set(images.keys()) | set(cells.keys()) | set(puncta.keys())
    if nucleus_mask_dir:
        all_stems |= set(nuclei.keys())
    unmatched = all_stems - common
    for s in sorted(unmatched):
        present_in = []
        if s in images:
            present_in.append("images")
        if s in cells:
            present_in.append("cell_masks")
        if s in puncta:
            present_in.append("puncta_masks")
        if s in nuclei:
            present_in.append("nucleus_masks")
        warnings.append(f"'{s}' only in: {', '.join(present_in)}")

    return matched, warnings


# ------------------------------------------------------------------ #
#  3. Per-image measurement extraction (puncta mask required)
# ------------------------------------------------------------------ #

def extract_cell_measurements(fluorescence_img, cell_mask,
                              puncta_mask,
                              nucleus_mask=None,
                              image_name="",
                              progress_callback=None):
    """
    Extract per-cell measurements from mEGFP fluorescence + masks
    for a single image.

    Returns
    -------
    df : pd.DataFrame
    puncta_binary : 2D bool array
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
        puncta_sum_int = float(fluorescence_img[cell_puncta].sum()) if cell_puncta.any() else 0.0

        # Nucleus region
        if nucleus_mask is not None:
            nuc_binary = (nucleus_mask > 0) & cell_binary
            cyto_binary = cell_binary & ~nuc_binary
        else:
            nuc_binary = None
            cyto_binary = cell_binary

        # Cytoplasm mean intensity = Cell - Nucleus (mEGFP ch1)
        cyto_pixels = fluorescence_img[cyto_binary]
        cyto_mean = float(np.mean(cyto_pixels)) if cyto_pixels.size > 0 else np.nan

        # Nucleus mean intensity = Nucleus region only (mEGFP ch1,
        # nucleus mask from Cy5/miRFPnano3 ch3)
        if nuc_binary is not None:
            nuc_pixels = fluorescence_img[nuc_binary]
            nuc_mean = float(np.mean(nuc_pixels)) if nuc_pixels.size > 0 else np.nan
        else:
            nuc_mean = np.nan

        # Dilute phase: cytoplasm minus puncta (mEGFP ch1)
        cyto_dilute_binary = cyto_binary & ~cell_puncta
        cyto_dilute_pixels = fluorescence_img[cyto_dilute_binary]
        cyto_dilute_mean = float(np.mean(cyto_dilute_pixels)) if cyto_dilute_pixels.size > 0 else np.nan

        # Dilute phase: nucleus minus puncta (mEGFP ch1)
        if nuc_binary is not None:
            nuc_dilute_binary = nuc_binary & ~cell_puncta
            nuc_dilute_pixels = fluorescence_img[nuc_dilute_binary]
            nuc_dilute_mean = float(np.mean(nuc_dilute_pixels)) if nuc_dilute_pixels.size > 0 else np.nan
        else:
            nuc_dilute_mean = np.nan

        records.append({
            "image": image_name,
            "cell_id": int(cid),
            "cell_area": cell_area,
            "total_cell_intensity": total_intensity,
            "cytoplasm_mean_intensity": cyto_mean,
            "nucleus_mean_intensity": nuc_mean,
            "cyto_dilute_mean_intensity": cyto_dilute_mean,
            "nuc_dilute_mean_intensity": nuc_dilute_mean,
            "puncta_present": 1 if n_puncta > 0 else 0,
            "puncta_count": n_puncta,
            "puncta_total_area": puncta_area,
            "puncta_sum_intensity": puncta_sum_int,
        })

    if progress_callback:
        progress_callback(total, total)

    df = pd.DataFrame(records)
    return df, puncta_binary


def extract_bulk(matched_files, channel_index=1, channel_name=None,
                 log_callback=None, progress_callback=None):
    """
    Process all matched file sets and return aggregated DataFrame.

    Parameters
    ----------
    matched_files : list of dict from match_files()
    channel_index : int — fallback channel index (0-based)
    channel_name : str or None — channel name to resolve from OME metadata
        (e.g. "GFP").  If given, takes priority over channel_index.
    log_callback : callable(str) or None — for logging messages
    progress_callback : callable(current, total) or None

    Returns
    -------
    df_all : pd.DataFrame — aggregated per-cell measurements
    last_overlay_data : tuple (fluor, cell_mask, puncta_binary) — for overlay plot
    """
    all_dfs = []
    n_total = len(matched_files)
    last_overlay_data = None

    # Log channel resolution from the first image
    if log_callback and n_total > 0 and channel_name:
        first_path = matched_files[0]["image"]
        ome_names = get_ome_channel_names(first_path)
        if ome_names:
            log_callback(f"  OME channel order: {ome_names}")
        resolved_idx, matched_name = resolve_channel_index(
            first_path, channel_index=channel_index, channel_name=channel_name)
        if matched_name:
            log_callback(
                f"  Channel '{channel_name}' -> index {resolved_idx} "
                f"(OME name: '{matched_name}')")
        else:
            log_callback(
                f"  Channel name '{channel_name}' not found in OME; "
                f"using fallback index {channel_index}")

    for file_idx, entry in enumerate(matched_files):
        stem = entry["stem"]
        if progress_callback:
            progress_callback(file_idx, n_total)
        if log_callback:
            log_callback(f"[{file_idx+1}/{n_total}] Processing {stem}...")

        fluor_img = load_image_channel(
            entry["image"], channel_index=channel_index,
            channel_name=channel_name)
        cell_mask = load_mask_2d(entry["cell_mask"])
        puncta_mask = load_mask_2d(entry["puncta_mask"])

        nuc_mask = None
        if "nucleus_mask" in entry:
            nuc_mask = load_mask_2d(entry["nucleus_mask"])

        df, puncta_binary = extract_cell_measurements(
            fluorescence_img=fluor_img,
            cell_mask=cell_mask,
            puncta_mask=puncta_mask,
            nucleus_mask=nuc_mask,
            image_name=stem,
        )

        if log_callback:
            n_cells = len(df)
            n_with = int(df["puncta_present"].sum()) if n_cells > 0 else 0
            log_callback(f"  -> {n_cells} cells, {n_with} with puncta")

        all_dfs.append(df)
        last_overlay_data = (fluor_img, cell_mask, puncta_binary)

    if progress_callback:
        progress_callback(n_total, n_total)

    if all_dfs:
        df_all = pd.concat(all_dfs, ignore_index=True)
    else:
        df_all = pd.DataFrame()

    return df_all, last_overlay_data


# ------------------------------------------------------------------ #
#  4. Data cleaning
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
#  5. Method 1: Binary logistic regression  P(puncta) ~ intensity
# ------------------------------------------------------------------ #

def fit_logistic_binary(df, x_col="cytoplasm_mean_intensity",
                        n_bootstrap=1000, ci=0.95, random_state=42):
    """
    Method 1: Logistic regression P(puncta_present) ~ x_col.
    Csat = x where P = 0.5 (the decision boundary).
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
#  6. Method 2: Sigmoid fit  puncta_sum_intensity ~ avg_intensity
# ------------------------------------------------------------------ #

def _sigmoid(x, L, k, x0, b):
    """Generalised logistic: y = b + L / (1 + exp(-k*(x - x0)))"""
    z = np.clip(-k * (x - x0), -500, 500)
    return b + L / (1.0 + np.exp(z))


def fit_sigmoid_intensity(df, x_col="cytoplasm_mean_intensity",
                          n_bootstrap=1000, ci=0.95, random_state=42):
    """
    Method 2: Fit sigmoid curve to puncta_sum_intensity vs x_col.
    Csat = x0, the midpoint where the response reaches 50% of its max.
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
#  7. Plotting helpers (return matplotlib Figure objects)
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


# ------------------------------------------------------------------ #
#  8. DropFit Csat: droplet-size-aware probability analysis
# ------------------------------------------------------------------ #

def compute_dropfit_bins(df, x_col="cytoplasm_mean_intensity", n_bins=20):
    """Bin cells by intensity and compute per-bin droplet statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Per-cell DataFrame (from extract_bulk / extract_cell_measurements).
    x_col : str
        Column to bin on (cytoplasm or nucleus intensity).
    n_bins : int
        Number of equal-width intensity bins.

    Returns
    -------
    pd.DataFrame with columns:
        bin_center, bin_low, bin_high, total_cells, cells_with_droplets,
        droplet_probability, mean_droplet_count, mean_droplet_area,
        median_droplet_area, total_droplet_area
    """
    x = df[x_col].values
    edges = np.linspace(np.nanmin(x), np.nanmax(x), n_bins + 1)

    records = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i < n_bins - 1:
            mask = (x >= lo) & (x < hi)
        else:
            mask = (x >= lo) & (x <= hi)

        grp = df[mask]
        n_total = len(grp)
        if n_total == 0:
            continue

        n_with = int((grp["puncta_present"] == 1).sum())
        records.append({
            "bin_center": (lo + hi) / 2.0,
            "bin_low": lo,
            "bin_high": hi,
            "total_cells": n_total,
            "cells_with_droplets": n_with,
            "droplet_probability": n_with / n_total,
            "mean_droplet_count": float(grp["puncta_count"].mean()),
            "mean_droplet_area": float(grp["puncta_total_area"].mean()),
            "median_droplet_area": float(grp.loc[grp["puncta_present"] == 1,
                                                  "puncta_total_area"].median())
            if n_with > 0 else 0.0,
            "total_droplet_area": float(grp["puncta_total_area"].sum()),
        })

    return pd.DataFrame(records)


def fit_dropfit_csat(df, x_col="cytoplasm_mean_intensity",
                     n_bins=20, n_bootstrap=500, ci=0.95,
                     random_state=42):
    """DropFit-inspired Csat estimation.

    1. Bin cells by *x_col* intensity.
    2. Compute droplet probability per bin.
    3. Fit logistic regression on the *binned* probabilities
       (weighted by cell count per bin).
    4. Csat = intensity at P = 0.5.
    5. Bootstrap at the cell level to get confidence interval.

    Returns
    -------
    dict with keys:
        method, x_col, n_bins, n_cells, csat, slope,
        ci_low, ci_high, model, bin_df, error (if failed)
    """
    from sklearn.linear_model import LogisticRegression

    base = {
        "method": "dropfit",
        "x_col": x_col,
        "n_bins": n_bins,
        "n_cells": len(df),
    }

    X = df[x_col].values.reshape(-1, 1)
    y = df["puncta_present"].values

    if len(np.unique(y)) < 2:
        return {**base, "csat": np.nan, "slope": np.nan,
                "ci_low": np.nan, "ci_high": np.nan,
                "model": None, "bin_df": None,
                "error": "Need cells both with and without puncta."}

    # Fit on full cell-level data (same as Method 1)
    model = LogisticRegression(solver="lbfgs", max_iter=1000)
    model.fit(X, y)

    slope = float(model.coef_[0, 0])
    intercept = float(model.intercept_[0])
    csat = -intercept / slope if slope != 0 else np.nan

    # Compute binned statistics for visualization and CSV export
    bin_df = compute_dropfit_bins(df, x_col=x_col, n_bins=n_bins)

    # Bootstrap at cell level
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
            "ci_low": ci_low, "ci_high": ci_high,
            "model": model, "bin_df": bin_df}


def plot_dropfit_csat(df, result):
    """DropFit Csat plot: binned droplet probability with logistic fit.

    Shows:
      - Binned probabilities as sized circles (size ~ cell count)
      - Logistic fit curve
      - Csat vertical line with 95% CI shading
    """
    fig = _make_figure(6, 4)
    ax = fig.add_subplot(111)

    x_col = result.get("x_col", "cytoplasm_mean_intensity")
    bin_df = result.get("bin_df")

    # Plot raw jittered data
    x_raw = df[x_col].values
    y_raw = df["puncta_present"].values
    jitter = np.random.default_rng(0).uniform(-0.03, 0.03, size=len(y_raw))
    ax.scatter(x_raw, y_raw + jitter, alpha=0.08, s=6, color="gray",
               edgecolors="none", zorder=1)

    # Plot binned probabilities
    if bin_df is not None and len(bin_df) > 0:
        sizes = 30 + 200 * (bin_df["total_cells"].values /
                             max(bin_df["total_cells"].max(), 1))
        ax.scatter(bin_df["bin_center"].values,
                   bin_df["droplet_probability"].values,
                   s=sizes, color="#DD8452", edgecolors="black",
                   linewidths=0.7, zorder=3,
                   label="Binned probability")

    # Logistic curve and Csat
    model = result.get("model")
    csat = result.get("csat", np.nan)
    if model is not None and not np.isnan(csat):
        x_curve = np.linspace(np.nanmin(x_raw), np.nanmax(x_raw), 300)
        y_curve = model.predict_proba(x_curve.reshape(-1, 1))[:, 1]
        ax.plot(x_curve, y_curve, color="#C44E52", linewidth=2,
                label="Logistic fit", zorder=2)
        ax.axvline(csat, color="#C44E52", linestyle="--", linewidth=1.5,
                   label=f"Csat = {csat:.1f}", zorder=4)
        ci_lo = result.get("ci_low", np.nan)
        ci_hi = result.get("ci_high", np.nan)
        if not np.isnan(ci_lo) and not np.isnan(ci_hi):
            ax.axvspan(ci_lo, ci_hi, alpha=0.12, color="#C44E52",
                       label=f"95% CI [{ci_lo:.1f}, {ci_hi:.1f}]")

    ax.set_xlabel("Cytoplasmic Mean Intensity")
    ax.set_ylabel("P(Droplet Formation)")
    ax.set_title("DropFit: Droplet Probability vs Intensity")
    ax.set_ylim(-0.08, 1.08)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    return fig


def plot_dropfit_size_distribution(df, result):
    """Complementary plot: droplet size vs intensity with bin statistics.

    Shows mean droplet area per bin as a bar chart overlaid on a scatter
    of individual cell droplet areas.
    """
    fig = _make_figure(6, 4)
    ax = fig.add_subplot(111)

    x_col = result.get("x_col", "cytoplasm_mean_intensity")
    bin_df = result.get("bin_df")

    # Scatter: individual cells with droplets
    has = df["puncta_present"] == 1
    if has.any():
        ax.scatter(df.loc[has, x_col].values,
                   df.loc[has, "puncta_total_area"].values,
                   alpha=0.25, s=10, color="#4C72B0", edgecolors="none",
                   label="Individual cells", zorder=1)

    # Binned mean area as bars
    if bin_df is not None and len(bin_df) > 0:
        width = (bin_df["bin_high"].iloc[0] - bin_df["bin_low"].iloc[0]) * 0.7
        mask = bin_df["cells_with_droplets"] > 0
        if mask.any():
            ax.bar(bin_df.loc[mask, "bin_center"].values,
                   bin_df.loc[mask, "mean_droplet_area"].values,
                   width=width, alpha=0.4, color="#DD8452",
                   edgecolor="black", linewidth=0.5,
                   label="Mean area per bin", zorder=2)

    csat = result.get("csat", np.nan)
    if not np.isnan(csat):
        ax.axvline(csat, color="#C44E52", linestyle="--", linewidth=1.5,
                   label=f"Csat = {csat:.1f}", zorder=3)

    ax.set_xlabel("Cytoplasmic Mean Intensity")
    ax.set_ylabel("Droplet Total Area (px)")
    ax.set_title("DropFit: Droplet Size Distribution")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    return fig
