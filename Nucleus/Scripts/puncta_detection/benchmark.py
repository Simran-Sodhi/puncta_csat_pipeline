#!/usr/bin/env python3
"""
benchmark.py — Compare puncta segmentation methods on the same images.

Runs every requested method on each image and computes:

  * Detection count per method
  * Pairwise agreement (intersection-over-union of detected sets)
  * If ground-truth masks are provided: Precision / Recall / F1 / RMSE
  * If cell masks are provided: per-cell detection counts per method
  * Visual comparison panels saved as PNG

Usage (CLI)
-----------
    python -m puncta_detection.benchmark \
        --image-dir /path/to/images \
        --out-dir /path/to/benchmark_results \
        --methods threshold spotiflow spotiflow+threshold \
        --channel 1

Usage (from Python / GUI)
-------------------------
    from puncta_detection.benchmark import run_benchmark
    results = run_benchmark(image_dir, out_dir, methods=[...], ...)
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
from skimage import measure
from skimage.segmentation import relabel_sequential

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from segmentation_utils import load_image_2d, auto_lut_clip, ensure_2d

from .core import PunctaMetrics


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #

def _labels_to_centroids(labels):
    """Extract (N, 2) centroids from a label mask."""
    props = measure.regionprops(labels)
    if not props:
        return np.empty((0, 2), dtype=np.float64)
    return np.array([p.centroid for p in props], dtype=np.float64)


def _iou_labels(labels_a, labels_b):
    """Intersection-over-Union between two binary foreground masks."""
    fg_a = labels_a > 0
    fg_b = labels_b > 0
    inter = np.count_nonzero(fg_a & fg_b)
    union = np.count_nonzero(fg_a | fg_b)
    return inter / union if union > 0 else 0.0


def _pairwise_detection_agreement(coords_dict, max_distance=3.0):
    """Pairwise precision/recall/F1 between all method pairs.

    Returns a list of dicts with keys:
    method_a, method_b, precision, recall, f1, iou.
    """
    rows = []
    names = list(coords_dict.keys())
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            metrics = PunctaMetrics.compute_f1(
                coords_dict[a], coords_dict[b], max_distance,
            )
            rows.append({
                "method_a": a,
                "method_b": b,
                "agreement_f1": round(metrics["f1"], 4),
                "a_vs_b_precision": round(metrics["precision"], 4),
                "a_vs_b_recall": round(metrics["recall"], 4),
            })
    return rows


def _make_comparison_panel(img2d, labels_dict, out_path):
    """Save a multi-panel PNG comparing all methods visually."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    methods = list(labels_dict.keys())
    n = len(methods) + 1  # +1 for raw image
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), dpi=120)
    if n == 1:
        axes = [axes]

    # Raw image
    disp = auto_lut_clip(img2d)
    axes[0].imshow(disp, cmap="gray")
    axes[0].set_title("Raw image", fontsize=9)
    axes[0].axis("off")

    # Each method
    rng = np.random.default_rng(42)
    for idx, name in enumerate(methods):
        labels = labels_dict[name]
        n_obj = int(labels.max())
        ax = axes[idx + 1]

        # Overlay: raw in gray, labels as coloured contours
        ax.imshow(disp, cmap="gray")
        if n_obj > 0:
            # Create random colormap for labels
            colors = rng.random((n_obj + 1, 4))
            colors[0] = [0, 0, 0, 0]  # background transparent
            colors[1:, 3] = 0.45
            overlay = colors[labels.clip(0, n_obj)]
            ax.imshow(overlay)
        ax.set_title(f"{name}\n({n_obj} puncta)", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ #
#  Public API
# ------------------------------------------------------------------ #

ALL_METHODS = [
    "threshold",
    "log",
    "dog",
    "intensity_ratio",
    "spotiflow",
    "spotiflow+threshold",
    "spotiflow+log",
    "tight_borders",
]


def run_benchmark(
    image_dir,
    out_dir,
    methods=None,
    channel=1,
    channel_name=None,
    z_index=0,
    gt_mask_dir=None,
    cell_mask_dir=None,
    match_distance=3.0,
    # pass-through params for segment_puncta_2d
    sigma=1.0,
    background_subtraction=True,
    tophat_radius=15,
    bg_method="white_tophat",
    threshold_method="otsu",
    min_sigma=1.0,
    max_sigma=5.0,
    num_sigma=5,
    blob_threshold=0.1,
    punctum_radius=3,
    t_local=1.5,
    t_global=1.5,
    t_cv=0.5,
    min_size=3,
    max_size=0,
    spotiflow_model="general",
    spotiflow_prob=0.5,
    spot_radius=2,
    save_panels=True,
    progress_callback=None,
):
    """Run multiple detection methods on every image and compare.

    Parameters
    ----------
    image_dir : str or Path
        Directory of input images.
    out_dir : str or Path
        Where to save comparison CSVs and panels.
    methods : list of str
        Methods to compare. Default: all available.
    gt_mask_dir : str or Path or None
        Ground-truth label masks for absolute evaluation (F1, RMSE).
    cell_mask_dir : str or Path or None
        Cell masks for per-cell comparison.
    match_distance : float
        Max pixel distance for matching detections across methods.
    save_panels : bool
        Save visual comparison PNGs.
    progress_callback : callable or None
        Called with (index, total, filename).

    Returns
    -------
    dict with keys:
        "per_image" — DataFrame (one row per image x method)
        "pairwise" — DataFrame (method-pair agreement scores)
        "summary"  — DataFrame (aggregated stats per method)
    """
    from .puncta_segmentation import (
        segment_puncta_2d, collect_image_paths,
        _parse_location, _build_cell_mask_map, _load_cell_mask,
    )

    image_dir = Path(image_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if methods is None:
        methods = ["threshold", "spotiflow", "spotiflow+threshold"]

    # GT masks
    gt_map = {}
    if gt_mask_dir:
        gt_map = _build_cell_mask_map(gt_mask_dir)

    # Cell masks
    cell_map = {}
    if cell_mask_dir:
        cell_map = _build_cell_mask_map(cell_mask_dir)

    image_paths = collect_image_paths(str(image_dir))
    if not image_paths:
        print("[WARN] No images found for benchmark")
        return {}

    panel_dir = out_dir / "comparison_panels"
    if save_panels:
        panel_dir.mkdir(parents=True, exist_ok=True)

    total = len(image_paths)
    per_image_rows = []
    pairwise_rows = []

    common_kw = dict(
        sigma=sigma,
        background_subtraction=background_subtraction,
        tophat_radius=tophat_radius,
        bg_method=bg_method,
        threshold_method=threshold_method,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        blob_threshold=blob_threshold,
        punctum_radius=punctum_radius,
        t_local=t_local,
        t_global=t_global,
        t_cv=t_cv,
        min_size=min_size,
        max_size=max_size,
        spotiflow_model=spotiflow_model,
        spotiflow_prob=spotiflow_prob,
        spot_radius=spot_radius,
    )

    print(f"[INFO] Benchmark: {len(methods)} methods x {total} images")
    print(f"       Methods: {', '.join(methods)}")

    for idx, img_path in enumerate(image_paths, 1):
        stem = img_path.stem
        location = _parse_location(img_path)
        print(f"  [{idx}/{total}] {img_path.name}")

        try:
            img2d = load_image_2d(img_path, channel_index=channel, z_index=z_index,
                                 channel_name=channel_name)
        except Exception as exc:
            print(f"    [ERROR] Could not load: {exc}")
            continue

        # Load cell mask if available
        cell_mask = None
        if cell_map and location in cell_map:
            try:
                cell_mask = _load_cell_mask(cell_map[location])
                if cell_mask.shape != img2d.shape:
                    cell_mask = None
            except Exception:
                cell_mask = None

        # Load GT if available
        gt_labels = None
        if gt_map and location in gt_map:
            try:
                gt_labels = _load_cell_mask(gt_map[location])
                if gt_labels.shape != img2d.shape:
                    gt_labels = None
            except Exception:
                gt_labels = None

        gt_coords = _labels_to_centroids(gt_labels) if gt_labels is not None else None

        # Run all methods
        labels_dict = {}
        coords_dict = {}
        for method in methods:
            try:
                labels, _ = segment_puncta_2d(
                    img2d, method=method, cell_mask=cell_mask, **common_kw,
                )
                labels_dict[method] = labels
                coords_dict[method] = _labels_to_centroids(labels)

                n_obj = int(labels.max())
                row = {
                    "filename": img_path.name,
                    "location": location,
                    "method": method,
                    "n_detections": n_obj,
                }

                # Morphology stats
                if n_obj > 0:
                    props = measure.regionprops(labels)
                    areas = [p.area for p in props]
                    row["mean_area"] = round(float(np.mean(areas)), 2)
                    row["median_area"] = round(float(np.median(areas)), 2)
                    row["std_area"] = round(float(np.std(areas)), 2)

                    # Intensity stats
                    int_props = measure.regionprops(labels, intensity_image=img2d)
                    intensities = [p.mean_intensity for p in int_props]
                    row["mean_intensity"] = round(float(np.mean(intensities)), 2)
                else:
                    row["mean_area"] = 0
                    row["median_area"] = 0
                    row["std_area"] = 0
                    row["mean_intensity"] = 0

                # GT metrics
                if gt_coords is not None:
                    m = PunctaMetrics.compute_f1(
                        coords_dict[method], gt_coords, match_distance,
                    )
                    row["gt_precision"] = round(m["precision"], 4)
                    row["gt_recall"] = round(m["recall"], 4)
                    row["gt_f1"] = round(m["f1"], 4)
                    row["gt_tp"] = m["tp"]
                    row["gt_fp"] = m["fp"]
                    row["gt_fn"] = m["fn"]
                    rmse = PunctaMetrics.localization_rmse(
                        coords_dict[method], gt_coords, match_distance,
                    )
                    row["gt_rmse_px"] = round(rmse, 3) if not np.isnan(rmse) else None

                # Foreground IoU with GT
                if gt_labels is not None:
                    row["gt_fg_iou"] = round(_iou_labels(labels, gt_labels), 4)

                per_image_rows.append(row)

            except Exception as exc:
                print(f"    [WARN] {method} failed: {exc}")
                per_image_rows.append({
                    "filename": img_path.name,
                    "location": location,
                    "method": method,
                    "n_detections": -1,
                    "error": str(exc),
                })

        # Pairwise agreement for this image
        pw = _pairwise_detection_agreement(coords_dict, match_distance)
        for r in pw:
            r["filename"] = img_path.name
        pairwise_rows.extend(pw)

        # Pairwise IoU
        method_names = list(labels_dict.keys())
        for i, a in enumerate(method_names):
            for b in method_names[i + 1:]:
                iou = _iou_labels(labels_dict[a], labels_dict[b])
                # Find matching pairwise row and add IoU
                for r in pairwise_rows:
                    if (r["filename"] == img_path.name
                            and r["method_a"] == a and r["method_b"] == b):
                        r["foreground_iou"] = round(iou, 4)

        # Save visual panel
        if save_panels and labels_dict:
            try:
                _make_comparison_panel(
                    img2d, labels_dict,
                    panel_dir / f"{stem}_comparison.png",
                )
            except Exception as exc:
                print(f"    [WARN] Panel save failed: {exc}")

        if progress_callback:
            progress_callback(idx, total, img_path.name)

    # Build DataFrames
    df_per_image = pd.DataFrame(per_image_rows)
    df_pairwise = pd.DataFrame(pairwise_rows)

    # Summary per method (aggregated across images)
    summary_rows = []
    if not df_per_image.empty:
        for method in methods:
            mdf = df_per_image[df_per_image["method"] == method]
            mdf_ok = mdf[mdf["n_detections"] >= 0]
            row = {
                "method": method,
                "images_processed": len(mdf_ok),
                "total_detections": int(mdf_ok["n_detections"].sum()),
                "mean_detections": round(float(mdf_ok["n_detections"].mean()), 1),
                "std_detections": round(float(mdf_ok["n_detections"].std()), 1),
                "mean_area": round(float(mdf_ok["mean_area"].mean()), 2),
                "mean_intensity": round(float(mdf_ok["mean_intensity"].mean()), 2),
            }
            if "gt_f1" in mdf.columns and mdf_ok["gt_f1"].notna().any():
                row["mean_gt_f1"] = round(float(mdf_ok["gt_f1"].mean()), 4)
                row["mean_gt_precision"] = round(float(mdf_ok["gt_precision"].mean()), 4)
                row["mean_gt_recall"] = round(float(mdf_ok["gt_recall"].mean()), 4)
            summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)

    # Save CSVs
    df_per_image.to_csv(out_dir / "benchmark_per_image.csv", index=False)
    df_pairwise.to_csv(out_dir / "benchmark_pairwise.csv", index=False)
    df_summary.to_csv(out_dir / "benchmark_summary.csv", index=False)

    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    if not df_summary.empty:
        for _, row in df_summary.iterrows():
            line = f"  {row['method']:25s}  "
            line += f"count={row['mean_detections']:6.1f} +/- {row['std_detections']:5.1f}  "
            line += f"area={row['mean_area']:5.1f}  "
            if "mean_gt_f1" in row and pd.notna(row.get("mean_gt_f1")):
                line += f"F1={row['mean_gt_f1']:.3f}  "
                line += f"P={row['mean_gt_precision']:.3f}  "
                line += f"R={row['mean_gt_recall']:.3f}"
            print(line)
    print(f"{'='*60}")
    print(f"Results saved to {out_dir}")

    return {
        "per_image": df_per_image,
        "pairwise": df_pairwise,
        "summary": df_summary,
    }


# ------------------------------------------------------------------ #
#  CLI
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark puncta segmentation methods side-by-side."
    )
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--methods", nargs="+",
                        default=["threshold", "spotiflow", "spotiflow+threshold"],
                        choices=ALL_METHODS)
    parser.add_argument("--channel", type=int, default=1)
    parser.add_argument("--z-index", type=int, default=0)
    parser.add_argument("--gt-mask-dir", default=None,
                        help="Ground-truth masks for F1/RMSE evaluation")
    parser.add_argument("--cell-mask-dir", default=None,
                        help="Cell masks for per-cell comparison")
    parser.add_argument("--match-distance", type=float, default=3.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--threshold-method", default="otsu")
    parser.add_argument("--min-size", type=int, default=3)
    parser.add_argument("--max-size", type=int, default=0)
    parser.add_argument("--spotiflow-model", default="general")
    parser.add_argument("--spotiflow-prob", type=float, default=0.5)
    parser.add_argument("--no-panels", action="store_true")

    args = parser.parse_args()
    run_benchmark(
        image_dir=args.image_dir,
        out_dir=args.out_dir,
        methods=args.methods,
        channel=args.channel,
        z_index=args.z_index,
        gt_mask_dir=args.gt_mask_dir,
        cell_mask_dir=args.cell_mask_dir,
        match_distance=args.match_distance,
        sigma=args.sigma,
        threshold_method=args.threshold_method,
        min_size=args.min_size,
        max_size=args.max_size,
        spotiflow_model=args.spotiflow_model,
        spotiflow_prob=args.spotiflow_prob,
        save_panels=not args.no_panels,
    )
