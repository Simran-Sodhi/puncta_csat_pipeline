"""
Compare two segmentation masks against an original image.

Loads an image (.ome.tif / .tif) and two masks (_seg.npy or .tif),
then computes per-object and global metrics:
  - IoU (Jaccard index)
  - Dice coefficient
  - Average Precision at multiple thresholds
  - Object counts, matched/unmatched objects
  - Intensity statistics under each mask
  - Per-object size comparison
  - Visual side-by-side comparison figure

Usage:
    python compare_masks.py \
        --image  sample.ome.tif \
        --mask1  sample_manual_seg.npy \
        --mask2  sample_model_seg.npy \
        --output comparison_report
"""

import argparse
import csv
import logging
import os
from pathlib import Path

import numpy as np
import tifffile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Loaders
# ------------------------------------------------------------------ #
def load_image(path: str) -> np.ndarray:
    """Load a .tif / .ome.tif image and return 2-D array (first channel/z)."""
    img = tifffile.imread(path)
    while img.ndim > 2:
        img = img[0]
    return img


def load_mask(path: str) -> np.ndarray:
    """Load a mask from _seg.npy (Cellpose) or .tif file."""
    path = str(path)
    if path.endswith(".npy"):
        dat = np.load(path, allow_pickle=True).item()
        return np.asarray(dat["masks"], dtype=np.int32)
    img = tifffile.imread(path)
    while img.ndim > 2:
        img = img[0]
    return img.astype(np.int32)


# ------------------------------------------------------------------ #
#  Global (binary) metrics
# ------------------------------------------------------------------ #
def binary_metrics(mask1: np.ndarray, mask2: np.ndarray) -> dict:
    """Compute binary IoU and Dice between two masks (foreground vs background)."""
    m1 = (mask1 > 0).astype(np.float64)
    m2 = (mask2 > 0).astype(np.float64)
    intersection = np.sum(m1 * m2)
    union = np.sum(m1) + np.sum(m2) - intersection
    iou = intersection / union if union > 0 else 0.0
    dice_denom = np.sum(m1) + np.sum(m2)
    dice = 2.0 * intersection / dice_denom if dice_denom > 0 else 0.0
    return {"binary_iou": iou, "binary_dice": dice}


# ------------------------------------------------------------------ #
#  Per-object matching and IoU
# ------------------------------------------------------------------ #
def match_objects(mask1: np.ndarray, mask2: np.ndarray, iou_thresh: float = 0.1):
    """Match objects between two label masks using IoU overlap.

    Returns
    -------
    matches : list of (label1, label2, iou)
    unmatched_1 : set of labels in mask1 with no match
    unmatched_2 : set of labels in mask2 with no match
    """
    labels1 = set(np.unique(mask1)) - {0}
    labels2 = set(np.unique(mask2)) - {0}

    # Pre-compute bounding boxes for speed
    def _bboxes(mask, labels):
        bbs = {}
        for lbl in labels:
            ys, xs = np.where(mask == lbl)
            bbs[lbl] = (ys.min(), ys.max(), xs.min(), xs.max())
        return bbs

    bb1 = _bboxes(mask1, labels1)
    bb2 = _bboxes(mask2, labels2)

    matches = []
    used2 = set()

    for l1 in sorted(labels1):
        best_iou = 0.0
        best_l2 = None
        r1_min, r1_max, c1_min, c1_max = bb1[l1]
        roi1 = mask1[r1_min:r1_max + 1, c1_min:c1_max + 1] == l1

        for l2 in sorted(labels2 - used2):
            r2_min, r2_max, c2_min, c2_max = bb2[l2]
            # Quick bbox overlap check
            if r1_max < r2_min or r2_max < r1_min or c1_max < c2_min or c2_max < c1_min:
                continue

            # Compute IoU in the union bounding box
            rmin = min(r1_min, r2_min)
            rmax = max(r1_max, r2_max)
            cmin = min(c1_min, c2_min)
            cmax = max(c1_max, c2_max)
            region1 = mask1[rmin:rmax + 1, cmin:cmax + 1] == l1
            region2 = mask2[rmin:rmax + 1, cmin:cmax + 1] == l2
            inter = np.sum(region1 & region2)
            union = np.sum(region1 | region2)
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best_l2 = l2

        if best_l2 is not None and best_iou >= iou_thresh:
            matches.append((l1, best_l2, best_iou))
            used2.add(best_l2)

    unmatched_1 = labels1 - {m[0] for m in matches}
    unmatched_2 = labels2 - used2
    return matches, unmatched_1, unmatched_2


# ------------------------------------------------------------------ #
#  Intensity statistics
# ------------------------------------------------------------------ #
def intensity_stats(image: np.ndarray, mask: np.ndarray) -> dict:
    """Compute intensity statistics of image under the mask."""
    fg = image[mask > 0].astype(np.float64)
    bg = image[mask == 0].astype(np.float64)
    labels = set(np.unique(mask)) - {0}

    per_object = []
    for lbl in sorted(labels):
        region = image[mask == lbl].astype(np.float64)
        per_object.append({
            "label": int(lbl),
            "area_px": int(region.size),
            "mean_intensity": float(np.mean(region)),
            "std_intensity": float(np.std(region)),
            "min_intensity": float(np.min(region)),
            "max_intensity": float(np.max(region)),
        })

    return {
        "n_objects": len(labels),
        "total_fg_px": int(fg.size),
        "total_bg_px": int(bg.size),
        "fg_mean": float(np.mean(fg)) if fg.size else 0.0,
        "fg_std": float(np.std(fg)) if fg.size else 0.0,
        "bg_mean": float(np.mean(bg)) if bg.size else 0.0,
        "per_object": per_object,
    }


# ------------------------------------------------------------------ #
#  Average Precision
# ------------------------------------------------------------------ #
def average_precision(mask_gt, mask_pred, thresholds=None):
    """Compute AP at given IoU thresholds."""
    if thresholds is None:
        thresholds = [0.5, 0.75, 0.9]
    matches, unmatched_gt, unmatched_pred = match_objects(mask_gt, mask_pred, iou_thresh=0.0)
    results = {}
    for t in thresholds:
        tp = sum(1 for _, _, iou in matches if iou >= t)
        fp = sum(1 for _, _, iou in matches if iou < t) + len(unmatched_pred)
        fn = sum(1 for _, _, iou in matches if iou < t) + len(unmatched_gt)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
        results[t] = {"precision": prec, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
    return results


# ------------------------------------------------------------------ #
#  Visualization
# ------------------------------------------------------------------ #
def plot_comparison(image, mask1, mask2, matches, stats1, stats2, ap_results, binary, output_path):
    """Generate a side-by-side comparison figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)

    # Row 1: Image, Mask1, Mask2, Overlay diff
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image, cmap="gray")
    ax1.set_title("Original Image")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(mask1, cmap="nipy_spectral", interpolation="nearest")
    ax2.set_title(f"Mask 1 ({stats1['n_objects']} objects)")
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(mask2, cmap="nipy_spectral", interpolation="nearest")
    ax3.set_title(f"Mask 2 ({stats2['n_objects']} objects)")
    ax3.axis("off")

    # Difference overlay: green = agree, red = mask1 only, blue = mask2 only
    ax4 = fig.add_subplot(gs[0, 3])
    m1_bin = mask1 > 0
    m2_bin = mask2 > 0
    diff_rgb = np.zeros((*image.shape[:2], 3), dtype=np.float32)
    diff_rgb[m1_bin & m2_bin] = [0, 1, 0]    # green = overlap
    diff_rgb[m1_bin & ~m2_bin] = [1, 0, 0]   # red = mask1 only
    diff_rgb[~m1_bin & m2_bin] = [0, 0, 1]   # blue = mask2 only
    ax4.imshow(diff_rgb)
    ax4.set_title("Difference\n(green=agree, red=M1 only, blue=M2 only)")
    ax4.axis("off")

    # Row 2: Per-object IoU histogram, Object size scatter, AP bar chart
    matched_ious = [iou for _, _, iou in matches]

    ax5 = fig.add_subplot(gs[1, 0:2])
    if matched_ious:
        ax5.hist(matched_ious, bins=20, range=(0, 1), color="steelblue", edgecolor="black")
        ax5.axvline(np.mean(matched_ious), color="red", linestyle="--",
                     label=f"Mean IoU = {np.mean(matched_ious):.3f}")
        ax5.legend()
    ax5.set_xlabel("Per-Object IoU")
    ax5.set_ylabel("Count")
    ax5.set_title("Per-Object IoU Distribution")

    ax6 = fig.add_subplot(gs[1, 2:4])
    sizes1 = {obj["label"]: obj["area_px"] for obj in stats1["per_object"]}
    sizes2 = {obj["label"]: obj["area_px"] for obj in stats2["per_object"]}
    if matches:
        s1 = [sizes1.get(l1, 0) for l1, l2, _ in matches]
        s2 = [sizes2.get(l2, 0) for l1, l2, _ in matches]
        colors = [iou for _, _, iou in matches]
        sc = ax6.scatter(s1, s2, c=colors, cmap="RdYlGn", vmin=0, vmax=1, alpha=0.7, edgecolors="black", s=30)
        plt.colorbar(sc, ax=ax6, label="IoU")
        max_sz = max(max(s1), max(s2)) * 1.1
        ax6.plot([0, max_sz], [0, max_sz], "k--", alpha=0.3, label="y=x")
        ax6.legend()
    ax6.set_xlabel("Mask 1 Object Area (px)")
    ax6.set_ylabel("Mask 2 Object Area (px)")
    ax6.set_title("Matched Object Sizes")

    # Row 3: Summary table, Intensity comparison
    ax7 = fig.add_subplot(gs[2, 0:2])
    ax7.axis("off")
    n_matched = len(matches)
    table_data = [
        ["Metric", "Value"],
        ["Objects in Mask 1", f"{stats1['n_objects']}"],
        ["Objects in Mask 2", f"{stats2['n_objects']}"],
        ["Matched objects", f"{n_matched}"],
        ["Unmatched in M1", f"{stats1['n_objects'] - n_matched}"],
        ["Unmatched in M2", f"{stats2['n_objects'] - n_matched}"],
        ["Binary IoU", f"{binary['binary_iou']:.4f}"],
        ["Binary Dice", f"{binary['binary_dice']:.4f}"],
        ["Mean per-obj IoU", f"{np.mean(matched_ious):.4f}" if matched_ious else "N/A"],
    ]
    for t, vals in ap_results.items():
        table_data.append([f"AP@{t}", f"{vals['precision']:.4f}"])
        table_data.append([f"F1@{t}", f"{vals['f1']:.4f}"])

    tbl = ax7.table(
        cellText=table_data, loc="center", cellLoc="left",
        colWidths=[0.5, 0.3],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.3)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")

    ax8 = fig.add_subplot(gs[2, 2:4])
    int1 = [obj["mean_intensity"] for obj in stats1["per_object"]]
    int2 = [obj["mean_intensity"] for obj in stats2["per_object"]]
    if int1:
        ax8.hist(int1, bins=30, alpha=0.6, label=f"Mask 1 (mean={np.mean(int1):.1f})", color="coral")
    if int2:
        ax8.hist(int2, bins=30, alpha=0.6, label=f"Mask 2 (mean={np.mean(int2):.1f})", color="steelblue")
    ax8.set_xlabel("Mean Object Intensity")
    ax8.set_ylabel("Count")
    ax8.set_title("Object Intensity Distribution")
    ax8.legend()

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Comparison figure saved to {output_path}")


# ------------------------------------------------------------------ #
#  CSV report
# ------------------------------------------------------------------ #
def save_csv_report(matches, stats1, stats2, binary, ap_results, output_path):
    """Save per-object match details and summary to CSV."""
    summary_path = str(output_path).replace(".csv", "_summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Value"])
        w.writerow(["Objects_Mask1", stats1["n_objects"]])
        w.writerow(["Objects_Mask2", stats2["n_objects"]])
        w.writerow(["Matched_Objects", len(matches)])
        w.writerow(["Binary_IoU", f"{binary['binary_iou']:.6f}"])
        w.writerow(["Binary_Dice", f"{binary['binary_dice']:.6f}"])
        w.writerow(["FG_Mean_Mask1", f"{stats1['fg_mean']:.2f}"])
        w.writerow(["FG_Mean_Mask2", f"{stats2['fg_mean']:.2f}"])
        for t, v in ap_results.items():
            w.writerow([f"Precision@{t}", f"{v['precision']:.6f}"])
            w.writerow([f"Recall@{t}", f"{v['recall']:.6f}"])
            w.writerow([f"F1@{t}", f"{v['f1']:.6f}"])

    sizes1 = {obj["label"]: obj for obj in stats1["per_object"]}
    sizes2 = {obj["label"]: obj for obj in stats2["per_object"]}

    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "mask1_label", "mask2_label", "iou",
            "area_m1", "area_m2", "area_diff",
            "mean_int_m1", "mean_int_m2", "int_diff",
        ])
        for l1, l2, iou in sorted(matches, key=lambda x: -x[2]):
            o1 = sizes1.get(l1, {})
            o2 = sizes2.get(l2, {})
            a1 = o1.get("area_px", 0)
            a2 = o2.get("area_px", 0)
            i1 = o1.get("mean_intensity", 0)
            i2 = o2.get("mean_intensity", 0)
            w.writerow([l1, l2, f"{iou:.4f}", a1, a2, a2 - a1,
                         f"{i1:.2f}", f"{i2:.2f}", f"{i2 - i1:.2f}"])

    logger.info(f"CSV report saved to {output_path} and {summary_path}")


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #
def compare(image_path, mask1_path, mask2_path, output_dir="comparison_report"):
    """Run the full comparison pipeline."""
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Loading image: {image_path}")
    image = load_image(image_path)

    logger.info(f"Loading mask 1: {mask1_path}")
    mask1 = load_mask(mask1_path)

    logger.info(f"Loading mask 2: {mask2_path}")
    mask2 = load_mask(mask2_path)

    if mask1.shape != mask2.shape:
        raise ValueError(
            f"Mask shapes differ: {mask1.shape} vs {mask2.shape}. "
            "They must be the same size."
        )

    # Global binary metrics
    binary = binary_metrics(mask1, mask2)
    logger.info(f"Binary IoU:  {binary['binary_iou']:.4f}")
    logger.info(f"Binary Dice: {binary['binary_dice']:.4f}")

    # Per-object matching
    matches, unmatched1, unmatched2 = match_objects(mask1, mask2)
    matched_ious = [iou for _, _, iou in matches]
    logger.info(f"Matched objects: {len(matches)}")
    if matched_ious:
        logger.info(f"Mean per-object IoU: {np.mean(matched_ious):.4f}")
    logger.info(f"Unmatched in mask1: {len(unmatched1)}")
    logger.info(f"Unmatched in mask2: {len(unmatched2)}")

    # AP at standard thresholds
    ap = average_precision(mask1, mask2)
    for t, v in ap.items():
        logger.info(f"  AP@{t}: P={v['precision']:.3f}  R={v['recall']:.3f}  F1={v['f1']:.3f}")

    # Intensity stats
    stats1 = intensity_stats(image, mask1)
    stats2 = intensity_stats(image, mask2)
    logger.info(f"Mask 1: {stats1['n_objects']} objects, FG mean intensity = {stats1['fg_mean']:.1f}")
    logger.info(f"Mask 2: {stats2['n_objects']} objects, FG mean intensity = {stats2['fg_mean']:.1f}")

    # Save outputs
    fig_path = os.path.join(output_dir, "comparison.png")
    plot_comparison(image, mask1, mask2, matches, stats1, stats2, ap, binary, fig_path)

    csv_path = os.path.join(output_dir, "per_object_matches.csv")
    save_csv_report(matches, stats1, stats2, binary, ap, csv_path)

    logger.info("Comparison complete.")
    return {
        "binary": binary,
        "matches": matches,
        "ap": ap,
        "stats_mask1": stats1,
        "stats_mask2": stats2,
    }


# ------------------------------------------------------------------ #
#  Auto-matching & Batch comparison
# ------------------------------------------------------------------ #
import re

_IMAGE_EXTS = {".tif", ".tiff", ".ome.tif", ".png"}
_MASK_EXTS = {".tif", ".tiff", ".png", ".npy"}


def _extract_key(filename: str) -> str:
    """Extract a matching key from a filename.

    Strategy (in order):
      1. Strip known suffixes (_img, _masks, _seg, _cell_masks, etc.)
         and use the remaining stem as the key.
      2. Fall back to the full stem (minus extension).

    This allows matching across naming conventions:
      dic_001_img.tif  <->  dic_001_masks.tif  <->  dic_001_seg.npy
    """
    stem = filename
    # Strip extension(s) — handle .ome.tif
    if stem.lower().endswith(".ome.tif"):
        stem = stem[:-8]
    else:
        stem = os.path.splitext(stem)[0]

    # Strip known suffixes (order matters — longest first)
    for suffix in [
        "_cell_masks", "_cyto3_masks", "_cyto_masks",
        "_nucleus_masks", "_puncta_masks",
        "_masks", "_seg", "_img",
    ]:
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem


def _scan_dir(directory: str, extensions: set) -> dict[str, str]:
    """Scan a directory and return {matching_key: filepath}."""
    result = {}
    for fname in sorted(os.listdir(directory)):
        fpath = os.path.join(directory, fname)
        if not os.path.isfile(fpath):
            continue
        flow = fname.lower()
        # Check extension match
        matched_ext = False
        for ext in extensions:
            if flow.endswith(ext):
                matched_ext = True
                break
        if not matched_ext:
            continue
        key = _extract_key(fname)
        result[key] = fpath
    return result


def auto_match(
    image_dir: str,
    mask1_dir: str,
    mask2_dir: str,
) -> tuple[list[tuple[str, str, str]], list[str]]:
    """Auto-match images with masks from two directories.

    Returns
    -------
    matched : list of (image_path, mask1_path, mask2_path)
    warnings : list of warning messages for unmatched files
    """
    images = _scan_dir(image_dir, _IMAGE_EXTS)
    masks1 = _scan_dir(mask1_dir, _MASK_EXTS)
    masks2 = _scan_dir(mask2_dir, _MASK_EXTS)

    matched = []
    warnings = []

    all_keys = sorted(set(images) | set(masks1) | set(masks2))
    for key in all_keys:
        img = images.get(key)
        m1 = masks1.get(key)
        m2 = masks2.get(key)
        if img and m1 and m2:
            matched.append((img, m1, m2))
        else:
            parts = []
            if not img:
                parts.append("no image")
            if not m1:
                parts.append("no mask1")
            if not m2:
                parts.append("no mask2")
            warnings.append(f"Key '{key}': {', '.join(parts)}")

    return matched, warnings


def compare_batch(
    image_dir: str,
    mask1_dir: str,
    mask2_dir: str,
    output_dir: str = "batch_comparison",
    progress_callback=None,
) -> dict:
    """Run comparison on all matched image/mask triplets in directories.

    Parameters
    ----------
    image_dir : str
        Directory containing original images.
    mask1_dir : str
        Directory containing ground-truth masks.
    mask2_dir : str
        Directory containing model-output masks.
    output_dir : str
        Directory for all reports.
    progress_callback : callable, optional
        Called with (current_index, total, key, results_dict) after each pair.

    Returns
    -------
    dict with keys:
        "per_pair" : list of {key, image, mask1, mask2, results}
        "aggregate" : aggregated metrics across all pairs
        "warnings" : list of warning strings
    """
    os.makedirs(output_dir, exist_ok=True)

    matched, warnings = auto_match(image_dir, mask1_dir, mask2_dir)
    if not matched:
        raise ValueError(
            f"No matched triplets found.\n"
            f"  Images in '{image_dir}': {len(_scan_dir(image_dir, _IMAGE_EXTS))}\n"
            f"  Masks1 in '{mask1_dir}': {len(_scan_dir(mask1_dir, _MASK_EXTS))}\n"
            f"  Masks2 in '{mask2_dir}': {len(_scan_dir(mask2_dir, _MASK_EXTS))}\n"
            f"Check that filenames share a common stem "
            f"(e.g. dic_001_img.tif / dic_001_masks.tif / dic_001_seg.npy)."
        )

    per_pair = []
    all_binary_iou = []
    all_binary_dice = []
    all_obj_ious = []
    all_ap = {}

    total = len(matched)
    for idx, (img_path, m1_path, m2_path) in enumerate(matched):
        key = _extract_key(os.path.basename(img_path))
        logger.info(f"[{idx + 1}/{total}] Comparing: {key}")

        try:
            pair_dir = os.path.join(output_dir, key)
            results = compare(img_path, m1_path, m2_path, pair_dir)

            per_pair.append({
                "key": key,
                "image": img_path,
                "mask1": m1_path,
                "mask2": m2_path,
                "results": results,
            })

            all_binary_iou.append(results["binary"]["binary_iou"])
            all_binary_dice.append(results["binary"]["binary_dice"])
            matched_ious = [iou for _, _, iou in results["matches"]]
            all_obj_ious.extend(matched_ious)

            # Accumulate AP
            for t, v in results["ap"].items():
                if t not in all_ap:
                    all_ap[t] = {"tp": 0, "fp": 0, "fn": 0}
                all_ap[t]["tp"] += v["tp"]
                all_ap[t]["fp"] += v["fp"]
                all_ap[t]["fn"] += v["fn"]

        except Exception as e:
            logger.error(f"Failed on '{key}': {e}")
            warnings.append(f"Error on '{key}': {e}")

        if progress_callback:
            progress_callback(idx + 1, total, key,
                              per_pair[-1] if per_pair else None)

    # Aggregate
    aggregate = {}
    if all_binary_iou:
        aggregate["mean_binary_iou"] = float(np.mean(all_binary_iou))
        aggregate["mean_binary_dice"] = float(np.mean(all_binary_dice))
        aggregate["std_binary_iou"] = float(np.std(all_binary_iou))
        aggregate["n_pairs"] = len(per_pair)
    if all_obj_ious:
        aggregate["mean_object_iou"] = float(np.mean(all_obj_ious))
        aggregate["median_object_iou"] = float(np.median(all_obj_ious))
        aggregate["total_matched_objects"] = len(all_obj_ious)
    for t, v in all_ap.items():
        tp, fp, fn = v["tp"], v["fp"], v["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
        aggregate[f"precision@{t}"] = prec
        aggregate[f"recall@{t}"] = recall
        aggregate[f"f1@{t}"] = f1

    # Save batch summary CSV
    summary_csv = os.path.join(output_dir, "batch_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "binary_iou", "binary_dice", "n_objects_m1",
                     "n_objects_m2", "n_matched", "mean_obj_iou"])
        for p in per_pair:
            r = p["results"]
            matched_ious = [iou for _, _, iou in r["matches"]]
            w.writerow([
                p["key"],
                f"{r['binary']['binary_iou']:.4f}",
                f"{r['binary']['binary_dice']:.4f}",
                r["stats_mask1"]["n_objects"],
                r["stats_mask2"]["n_objects"],
                len(r["matches"]),
                f"{np.mean(matched_ious):.4f}" if matched_ious else "N/A",
            ])

    logger.info(f"Batch summary saved to {summary_csv}")

    return {
        "per_pair": per_pair,
        "aggregate": aggregate,
        "warnings": warnings,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare two segmentation masks against an original image"
    )
    sub = parser.add_subparsers(dest="command")

    # Single-pair mode
    single = sub.add_parser("single", help="Compare a single image/mask pair")
    single.add_argument("--image", required=True, help="Path to original image")
    single.add_argument("--mask1", required=True, help="Path to mask 1 (ground truth)")
    single.add_argument("--mask2", required=True, help="Path to mask 2 (model output)")
    single.add_argument("--output", default="comparison_report", help="Output directory")

    # Batch mode
    batch = sub.add_parser("batch", help="Compare all matched pairs in directories")
    batch.add_argument("--images", required=True, help="Directory of images")
    batch.add_argument("--masks1", required=True, help="Directory of ground-truth masks")
    batch.add_argument("--masks2", required=True, help="Directory of model-output masks")
    batch.add_argument("--output", default="batch_comparison", help="Output directory")

    # Default: legacy single mode
    parser.add_argument("--image", help="Path to original image")
    parser.add_argument("--mask1", help="Path to mask 1")
    parser.add_argument("--mask2", help="Path to mask 2")
    parser.add_argument("--output", default="comparison_report", help="Output directory")

    args = parser.parse_args()

    if args.command == "batch":
        compare_batch(args.images, args.masks1, args.masks2, args.output)
    elif args.command == "single":
        compare(args.image, args.mask1, args.mask2, args.output)
    elif args.image and args.mask1 and args.mask2:
        compare(args.image, args.mask1, args.mask2, args.output)
    else:
        parser.print_help()
