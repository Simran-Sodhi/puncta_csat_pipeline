#!/usr/bin/env python3
"""
csat_fit.py

Estimate C_sat (saturation concentration proxy in intensity units)
from per-cell summary CSV produced by summarize_puncta_by_cell.py.

Core idea:
    - Use a logistic model: P(puncta=1 | I) = sigmoid(b0 + b1 * I)
    - Define C_sat as the intensity where P = 0.5:
          C_sat = -b0 / b1
    - Optionally bootstrap this estimate to get a 95% confidence interval.
    - Also compute a nonparametric threshold via ROC Youden index for comparison.

By default, uses:
    intensity_column = "intensity_for_cs"
    puncta_column    = "has_puncta"

Usage example:

    python csat_fit.py \
        --csv puncta_summary.csv \
        --intensity-column intensity_for_cs \
        --puncta-column has_puncta \
        --max-sat-frac 0.02 \
        --min-cyto-pixels 50 \
        --bootstrap-iters 1000

"""

import argparse
import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve


def estimate_csat_logistic(intensity, puncta, C=None):
    """
    Fit a 1D logistic regression and return C_sat = -b0 / b1.

    intensity: 1D numpy array of intensities
    puncta   : 1D numpy array of binary labels (0/1)
    C        : penalty (inverse regularization strength) for LogisticRegression

    Returns:
        csat (float or np.nan), intercept (b0), coef (b1)
    """
    x = intensity.reshape(-1, 1)
    y = puncta.astype(int)

    # Check we actually have both classes
    if len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan

    # Logistic regression with minimal regularization (C large)
    if C is None:
        C = 1e6

    clf = LogisticRegression(
        solver="lbfgs",
        C=C,
        max_iter=1000,
        fit_intercept=True,
    )
    clf.fit(x, y)

    b0 = clf.intercept_[0]
    b1 = clf.coef_[0, 0]

    # Avoid division by zero or near-zero slopes
    if np.isclose(b1, 0.0):
        return np.nan, b0, b1

    csat = -b0 / b1
    return float(csat), float(b0), float(b1)


def estimate_csat_bootstrap(
    intensity,
    puncta,
    n_iters=1000,
    random_state=0,
    C=None,
    min_unique_labels=2,
):
    """
    Bootstrap C_sat by resampling rows with replacement.

    Returns:
        csat_median, (csat_lo, csat_hi), csat_samples (array)
    """
    rng = np.random.default_rng(random_state)
    n = len(intensity)

    csat_samples = []

    for _ in range(n_iters):
        idx = rng.integers(0, n, size=n)
        x_bs = intensity[idx]
        y_bs = puncta[idx]

        if len(np.unique(y_bs)) < min_unique_labels:
            continue

        csat_bs, _, _ = estimate_csat_logistic(x_bs, y_bs, C=C)
        if not np.isnan(csat_bs):
            csat_samples.append(csat_bs)

    if len(csat_samples) == 0:
        return np.nan, (np.nan, np.nan), np.array([])

    csat_samples = np.array(csat_samples)
    csat_median = float(np.median(csat_samples))
    csat_lo = float(np.percentile(csat_samples, 2.5))
    csat_hi = float(np.percentile(csat_samples, 97.5))

    return csat_median, (csat_lo, csat_hi), csat_samples


def estimate_csat_youden(intensity, puncta):
    """
    Estimate a threshold via ROC Youden index:
        J = TPR - FPR
    Returns:
        threshold (float or np.nan)
    """
    y = puncta.astype(int)
    if len(np.unique(y)) < 2:
        return np.nan

    fpr, tpr, thr = roc_curve(y, intensity)
    youden = tpr - fpr
    idx = np.argmax(youden)
    return float(thr[idx])


def main(
    csv,
    intensity_column,
    puncta_column,
    output_json,
    max_sat_frac,
    min_cyto_pixels,
    min_intensity=None,
    max_intensity=None,
    bootstrap_iters=1000,
    random_state=0,
):
    df = pd.read_csv(csv)

    if intensity_column not in df.columns:
        raise ValueError(f"Intensity column '{intensity_column}' not found in CSV.")
    if puncta_column not in df.columns:
        raise ValueError(f"Puncta column '{puncta_column}' not found in CSV.")

    # Optional filters
    mask = np.ones(len(df), dtype=bool)

    # Filter by saturation if sat_frac_cell exists
    if "sat_frac_cell" in df.columns and max_sat_frac is not None:
        mask &= df["sat_frac_cell"] <= max_sat_frac

    # Filter by cell size if num_cyto_pixels exists
    if "num_cyto_pixels" in df.columns and min_cyto_pixels is not None:
        mask &= df["num_cyto_pixels"] >= min_cyto_pixels

    # Filter by intensity range if provided
    if min_intensity is not None:
        mask &= df[intensity_column] >= min_intensity
    if max_intensity is not None:
        mask &= df[intensity_column] <= max_intensity

    df_filt = df[mask].copy()

    if len(df_filt) < 10:
        print(
            f"[WARN] After filtering, only {len(df_filt)} cells remain. "
            "Estimates may be unstable.",
            file=sys.stderr,
        )

    intensity = df_filt[intensity_column].to_numpy(dtype=float)
    puncta = df_filt[puncta_column].to_numpy(dtype=int)

    # Basic stats on labels
    uniq, counts = np.unique(puncta, return_counts=True)
    label_counts = dict(zip(map(int, uniq), map(int, counts)))

    # 1) Logistic regression C_sat
    csat_point, b0, b1 = estimate_csat_logistic(intensity, puncta)

    # 2) Bootstrap CI
    csat_median, (csat_lo, csat_hi), csat_samples = estimate_csat_bootstrap(
        intensity,
        puncta,
        n_iters=bootstrap_iters,
        random_state=random_state,
    )

    # 3) Nonparametric Youden threshold
    csat_youden = estimate_csat_youden(intensity, puncta)

    result = {
        "n_cells_total": int(len(df)),
        "n_cells_used": int(len(df_filt)),
        "label_counts_used": label_counts,
        "intensity_column": intensity_column,
        "puncta_column": puncta_column,
        "filters": {
            "max_sat_frac": max_sat_frac,
            "min_cyto_pixels": min_cyto_pixels,
            "min_intensity": min_intensity,
            "max_intensity": max_intensity,
        },
        "logistic_fit": {
            "b0_intercept": b0,
            "b1_coef": b1,
            "csat_point_estimate": csat_point,
        },
        "bootstrap": {
            "n_effective_samples": int(len(csat_samples)),
            "csat_median": csat_median,
            "csat_ci_95": [csat_lo, csat_hi],
        },
        "youden_threshold": csat_youden,
    }

    # Pretty-print to stdout
    print("===== C_sat estimation =====")
    print(f"CSV file              : {csv}")
    print(f"Cells (total / used)  : {result['n_cells_total']} / {result['n_cells_used']}")
    print(f"Label counts (used)   : {result['label_counts_used']}")
    print()
    print("Logistic regression:")
    print(f"  b0 (intercept)      : {b0}")
    print(f"  b1 (slope)          : {b1}")
    print(f"  C_sat (point)       : {csat_point}")
    print()
    print("Bootstrap (C_sat from logistic):")
    print(f"  iterations (requested)  : {bootstrap_iters}")
    print(f"  iterations (used)       : {result['bootstrap']['n_effective_samples']}")
    print(f"  C_sat median            : {csat_median}")
    print(f"  C_sat 95% CI            : [{csat_lo}, {csat_hi}]")
    print()
    print("Nonparametric Youden threshold:")
    print(f"  C_sat (Youden)      : {csat_youden}")
    print()

    if output_json is not None:
        with open(output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved JSON results to {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate C_sat from per-cell puncta CSV.")
    parser.add_argument("--csv", required=True, help="Path to per-cell CSV.")
    parser.add_argument(
        "--intensity-column",
        default="intensity_for_cs",
        help="Column name for intensity (default: intensity_for_cs).",
    )
    parser.add_argument(
        "--puncta-column",
        default="has_puncta",
        help="Column name for puncta presence (0/1, default: has_puncta).",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save results as JSON.",
    )

    parser.add_argument(
        "--max-sat-frac",
        type=float,
        default=0.02,
        help="Max allowed sat_frac_cell (if column exists) to keep a cell (default: 0.02).",
    )
    parser.add_argument(
        "--min-cyto-pixels",
        type=int,
        default=50,
        help="Min num_cyto_pixels (if column exists) to keep a cell (default: 50).",
    )
    parser.add_argument(
        "--min-intensity",
        type=float,
        default=None,
        help="Optional: minimum intensity cutoff.",
    )
    parser.add_argument(
        "--max-intensity",
        type=float,
        default=None,
        help="Optional: maximum intensity cutoff.",
    )

    parser.add_argument(
        "--bootstrap-iters",
        type=int,
        default=1000,
        help="Number of bootstrap iterations for C_sat CI (default: 1000).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed for bootstrapping (default: 0).",
    )

    args = parser.parse_args()
    main(**vars(args))
