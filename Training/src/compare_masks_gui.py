"""
Standalone GUI for comparing segmentation masks against original images.

Supports:
  - Single-pair mode: one image + two masks
  - Bulk mode: directories of images, ground-truth masks, and model masks
    with automatic filename matching

Image formats: .tif, .ome.tif, .tiff, .png
Mask formats:  _seg.npy (Cellpose), .tif, .tiff, .png

Run:
    python compare_masks_gui.py
"""

import os
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path

# ---------------------------------------------------------------------------
# Imports from comparison backend
# ---------------------------------------------------------------------------
try:
    from compare_masks import (
        compare,
        compare_batch,
        auto_match,
        load_image,
        load_mask,
        binary_metrics,
        match_objects,
        intensity_stats,
        average_precision,
    )
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from compare_masks import (
        compare,
        compare_batch,
        auto_match,
        load_image,
        load_mask,
        binary_metrics,
        match_objects,
        intensity_stats,
        average_precision,
    )


# ====================================================================== #
#  Shared formatting helpers
# ====================================================================== #
def format_single_results(results: dict) -> str:
    import numpy as np
    b = results["binary"]
    matches = results["matches"]
    ap = results["ap"]
    s1 = results["stats_mask1"]
    s2 = results["stats_mask2"]
    matched_ious = [iou for _, _, iou in matches]

    lines = []
    lines.append("=" * 55)
    lines.append("          MASK COMPARISON RESULTS")
    lines.append("=" * 55)
    lines.append("")
    lines.append("  GLOBAL METRICS")
    lines.append(f"  {'Binary IoU:':<28} {b['binary_iou']:.4f}")
    lines.append(f"  {'Binary Dice:':<28} {b['binary_dice']:.4f}")
    lines.append("")
    lines.append("  OBJECT COUNTS")
    lines.append(f"  {'Objects in Mask 1:':<28} {s1['n_objects']}")
    lines.append(f"  {'Objects in Mask 2:':<28} {s2['n_objects']}")
    lines.append(f"  {'Matched objects:':<28} {len(matches)}")
    lines.append(f"  {'Unmatched in Mask 1:':<28} {s1['n_objects'] - len(matches)}")
    lines.append(f"  {'Unmatched in Mask 2:':<28} {s2['n_objects'] - len(matches)}")
    lines.append("")
    lines.append("  PER-OBJECT IoU")
    if matched_ious:
        lines.append(f"  {'Mean IoU:':<28} {np.mean(matched_ious):.4f}")
        lines.append(f"  {'Median IoU:':<28} {np.median(matched_ious):.4f}")
        lines.append(f"  {'Min IoU:':<28} {np.min(matched_ious):.4f}")
        lines.append(f"  {'Max IoU:':<28} {np.max(matched_ious):.4f}")
        lines.append(f"  {'Std IoU:':<28} {np.std(matched_ious):.4f}")
    else:
        lines.append("  No matched objects found.")
    lines.append("")
    lines.append("  AVERAGE PRECISION")
    for t, v in ap.items():
        lines.append(f"  IoU >= {t}:")
        lines.append(f"    {'Precision:':<24} {v['precision']:.4f}")
        lines.append(f"    {'Recall:':<24} {v['recall']:.4f}")
        lines.append(f"    {'F1 Score:':<24} {v['f1']:.4f}")
        lines.append(f"    {'TP / FP / FN:':<24} {v['tp']} / {v['fp']} / {v['fn']}")
    lines.append("")
    lines.append("  INTENSITY (under mask)")
    lines.append(f"  {'Mask 1 FG mean:':<28} {s1['fg_mean']:.1f}")
    lines.append(f"  {'Mask 2 FG mean:':<28} {s2['fg_mean']:.1f}")
    lines.append(f"  {'Mask 1 FG pixels:':<28} {s1['total_fg_px']}")
    lines.append(f"  {'Mask 2 FG pixels:':<28} {s2['total_fg_px']}")
    lines.append("")

    if matched_ious:
        sorted_m = sorted(matches, key=lambda x: x[2], reverse=True)
        lines.append("  TOP 10 BEST MATCHED OBJECTS")
        lines.append(f"  {'M1 Label':<12}{'M2 Label':<12}{'IoU':<10}")
        lines.append("  " + "-" * 34)
        for l1, l2, iou in sorted_m[:10]:
            lines.append(f"  {l1:<12}{l2:<12}{iou:.4f}")
        lines.append("")

        if len(sorted_m) > 10:
            lines.append("  BOTTOM 10 WORST MATCHED OBJECTS")
            lines.append(f"  {'M1 Label':<12}{'M2 Label':<12}{'IoU':<10}")
            lines.append("  " + "-" * 34)
            for l1, l2, iou in sorted_m[-10:]:
                lines.append(f"  {l1:<12}{l2:<12}{iou:.4f}")

    lines.append("")
    lines.append("=" * 55)
    return "\n".join(lines)


def format_batch_results(batch: dict) -> str:
    import numpy as np
    agg = batch["aggregate"]
    pairs = batch["per_pair"]
    warnings = batch["warnings"]

    lines = []
    lines.append("=" * 65)
    lines.append("            BATCH COMPARISON RESULTS")
    lines.append("=" * 65)
    lines.append("")
    lines.append(f"  {'Pairs compared:':<30} {agg.get('n_pairs', 0)}")
    lines.append(f"  {'Mean Binary IoU:':<30} {agg.get('mean_binary_iou', 0):.4f}")
    lines.append(f"  {'Mean Binary Dice:':<30} {agg.get('mean_binary_dice', 0):.4f}")
    lines.append(f"  {'Std Binary IoU:':<30} {agg.get('std_binary_iou', 0):.4f}")
    lines.append(f"  {'Total matched objects:':<30} {agg.get('total_matched_objects', 0)}")
    lines.append(f"  {'Mean per-object IoU:':<30} {agg.get('mean_object_iou', 0):.4f}")
    lines.append(f"  {'Median per-object IoU:':<30} {agg.get('median_object_iou', 0):.4f}")
    lines.append("")
    lines.append("  AGGREGATE AVERAGE PRECISION")
    for t in [0.5, 0.75, 0.9]:
        p = agg.get(f"precision@{t}", 0)
        r = agg.get(f"recall@{t}", 0)
        f1 = agg.get(f"f1@{t}", 0)
        lines.append(f"  IoU >= {t}:  P={p:.4f}  R={r:.4f}  F1={f1:.4f}")
    lines.append("")

    # Per-pair summary table
    lines.append("  PER-PAIR SUMMARY")
    lines.append(f"  {'Key':<20}{'BinIoU':<10}{'BinDice':<10}"
                 f"{'Obj M1':<8}{'Obj M2':<8}{'Match':<8}{'MeanIoU':<10}")
    lines.append("  " + "-" * 74)
    for p in pairs:
        r = p["results"]
        matched_ious = [iou for _, _, iou in r["matches"]]
        mean_iou = f"{np.mean(matched_ious):.4f}" if matched_ious else "N/A"
        lines.append(
            f"  {p['key']:<20}"
            f"{r['binary']['binary_iou']:<10.4f}"
            f"{r['binary']['binary_dice']:<10.4f}"
            f"{r['stats_mask1']['n_objects']:<8}"
            f"{r['stats_mask2']['n_objects']:<8}"
            f"{len(r['matches']):<8}"
            f"{mean_iou:<10}"
        )
    lines.append("")

    if warnings:
        lines.append("  WARNINGS")
        for w in warnings:
            lines.append(f"    {w}")
        lines.append("")

    lines.append("=" * 65)
    return "\n".join(lines)


def open_figure(path: str):
    """Open a file with the system default viewer."""
    import subprocess
    import platform
    try:
        if platform.system() == "Darwin":
            subprocess.Popen(["open", path])
        elif platform.system() == "Windows":
            os.startfile(path)
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception:
        pass


# ====================================================================== #
#  Main GUI
# ====================================================================== #
class CompareMasksGUI(tk.Tk):
    """Standalone mask comparison application with single and bulk modes."""

    def __init__(self):
        super().__init__()
        self.title("Mask Comparison Tool")
        self.geometry("900x800")
        self.minsize(750, 650)

        self.log_queue: queue.Queue[str] = queue.Queue()
        self._build_ui()
        self._poll_queue()

    # ------------------------------------------------------------------ #
    #  UI
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self._single_frame = ttk.Frame(self.notebook)
        self._bulk_frame = ttk.Frame(self.notebook)
        self.notebook.add(self._single_frame, text="  Single Pair  ")
        self.notebook.add(self._bulk_frame, text="  Bulk (Directories)  ")

        self._build_single_tab()
        self._build_bulk_tab()

    # ================================================================== #
    #  SINGLE PAIR TAB
    # ================================================================== #
    def _build_single_tab(self):
        tab = self._single_frame

        ttk.Label(
            tab,
            text="Compare one image against two masks (ground truth vs model).",
            foreground="gray",
        ).pack(anchor=tk.W, padx=10, pady=(10, 5))

        # Input files
        io_frame = ttk.LabelFrame(tab, text="Input Files", padding=10)
        io_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(io_frame, text="Original Image:").grid(row=0, column=0, sticky=tk.W, pady=3)
        self.image_var = tk.StringVar()
        ttk.Entry(io_frame, textvariable=self.image_var, width=55).grid(
            row=0, column=1, padx=5, pady=3, sticky=tk.EW)
        ttk.Button(io_frame, text="Browse...", command=self._browse_image).grid(
            row=0, column=2, pady=3)

        ttk.Label(io_frame, text="Mask 1 (Ground Truth):").grid(row=1, column=0, sticky=tk.W, pady=3)
        self.mask1_var = tk.StringVar()
        ttk.Entry(io_frame, textvariable=self.mask1_var, width=55).grid(
            row=1, column=1, padx=5, pady=3, sticky=tk.EW)
        ttk.Button(io_frame, text="Browse...", command=self._browse_mask1).grid(
            row=1, column=2, pady=3)

        ttk.Label(io_frame, text="Mask 2 (Model Output):").grid(row=2, column=0, sticky=tk.W, pady=3)
        self.mask2_var = tk.StringVar()
        ttk.Entry(io_frame, textvariable=self.mask2_var, width=55).grid(
            row=2, column=1, padx=5, pady=3, sticky=tk.EW)
        ttk.Button(io_frame, text="Browse...", command=self._browse_mask2).grid(
            row=2, column=2, pady=3)

        io_frame.columnconfigure(1, weight=1)

        # Output
        out_frame = ttk.LabelFrame(tab, text="Output", padding=10)
        out_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(out_frame, text="Report Directory:").grid(row=0, column=0, sticky=tk.W, pady=3)
        self.output_var = tk.StringVar(value="comparison_report")
        ttk.Entry(out_frame, textvariable=self.output_var, width=55).grid(
            row=0, column=1, padx=5, pady=3, sticky=tk.EW)
        ttk.Button(out_frame, text="Browse...",
                   command=lambda: self._browse_dir(self.output_var)).grid(
            row=0, column=2, pady=3)

        self.open_figure = tk.BooleanVar(value=True)
        ttk.Checkbutton(out_frame, text="Open comparison figure when done",
                        variable=self.open_figure).grid(
            row=1, column=0, columnspan=3, sticky=tk.W, pady=2)

        out_frame.columnconfigure(1, weight=1)

        # Buttons
        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)

        self.btn_run = ttk.Button(btn_frame, text="Run Comparison", command=self._run_single)
        self.btn_run.pack(side=tk.LEFT, padx=5)

        self.btn_quick = ttk.Button(btn_frame, text="Quick Summary (no figure)",
                                    command=self._run_single_quick)
        self.btn_quick.pack(side=tk.LEFT, padx=5)

        # Progress
        self.single_progress = ttk.Progressbar(tab, mode="indeterminate")
        self.single_progress.pack(fill=tk.X, padx=10, pady=5)

        self.single_status = tk.StringVar(value="Ready")
        ttk.Label(tab, textvariable=self.single_status).pack(padx=10, anchor=tk.W)

        # Results
        results_frame = ttk.LabelFrame(tab, text="Results", padding=5)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))

        self.single_results = scrolledtext.ScrolledText(
            results_frame, wrap=tk.WORD, height=15, state=tk.DISABLED,
            font=("Courier", 9))
        self.single_results.pack(fill=tk.BOTH, expand=True)

    # ================================================================== #
    #  BULK TAB
    # ================================================================== #
    def _build_bulk_tab(self):
        tab = self._bulk_frame

        ttk.Label(
            tab,
            text="Compare all matching image/mask pairs across directories.\n"
                 "Files are auto-matched by filename stem "
                 "(e.g. dic_001_img.tif <-> dic_001_masks.tif <-> dic_001_seg.npy).",
            foreground="gray",
        ).pack(anchor=tk.W, padx=10, pady=(10, 5))

        # Directory inputs
        dir_frame = ttk.LabelFrame(tab, text="Input Directories", padding=10)
        dir_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(dir_frame, text="Images Directory:").grid(row=0, column=0, sticky=tk.W, pady=3)
        self.bulk_img_dir = tk.StringVar()
        ttk.Entry(dir_frame, textvariable=self.bulk_img_dir, width=55).grid(
            row=0, column=1, padx=5, pady=3, sticky=tk.EW)
        ttk.Button(dir_frame, text="Browse...",
                   command=lambda: self._browse_dir(self.bulk_img_dir)).grid(
            row=0, column=2, pady=3)

        ttk.Label(dir_frame, text="Mask 1 Dir (Ground Truth):").grid(row=1, column=0, sticky=tk.W, pady=3)
        self.bulk_mask1_dir = tk.StringVar()
        ttk.Entry(dir_frame, textvariable=self.bulk_mask1_dir, width=55).grid(
            row=1, column=1, padx=5, pady=3, sticky=tk.EW)
        ttk.Button(dir_frame, text="Browse...",
                   command=lambda: self._browse_dir(self.bulk_mask1_dir)).grid(
            row=1, column=2, pady=3)

        ttk.Label(dir_frame, text="Mask 2 Dir (Model Output):").grid(row=2, column=0, sticky=tk.W, pady=3)
        self.bulk_mask2_dir = tk.StringVar()
        ttk.Entry(dir_frame, textvariable=self.bulk_mask2_dir, width=55).grid(
            row=2, column=1, padx=5, pady=3, sticky=tk.EW)
        ttk.Button(dir_frame, text="Browse...",
                   command=lambda: self._browse_dir(self.bulk_mask2_dir)).grid(
            row=2, column=2, pady=3)

        dir_frame.columnconfigure(1, weight=1)

        # Output
        out_frame = ttk.LabelFrame(tab, text="Output", padding=10)
        out_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(out_frame, text="Report Directory:").grid(row=0, column=0, sticky=tk.W, pady=3)
        self.bulk_output_var = tk.StringVar(value="batch_comparison")
        ttk.Entry(out_frame, textvariable=self.bulk_output_var, width=55).grid(
            row=0, column=1, padx=5, pady=3, sticky=tk.EW)
        ttk.Button(out_frame, text="Browse...",
                   command=lambda: self._browse_dir(self.bulk_output_var)).grid(
            row=0, column=2, pady=3)
        out_frame.columnconfigure(1, weight=1)

        # Buttons
        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)

        self.btn_bulk_preview = ttk.Button(
            btn_frame, text="Preview Matches", command=self._bulk_preview)
        self.btn_bulk_preview.pack(side=tk.LEFT, padx=5)

        self.btn_bulk_run = ttk.Button(
            btn_frame, text="Run Batch Comparison", command=self._run_bulk)
        self.btn_bulk_run.pack(side=tk.LEFT, padx=5)

        # Progress
        self.bulk_progress = ttk.Progressbar(tab, mode="determinate", maximum=100)
        self.bulk_progress.pack(fill=tk.X, padx=10, pady=5)

        self.bulk_status = tk.StringVar(value="Ready")
        ttk.Label(tab, textvariable=self.bulk_status).pack(padx=10, anchor=tk.W)

        # Results
        results_frame = ttk.LabelFrame(tab, text="Results", padding=5)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))

        self.bulk_results = scrolledtext.ScrolledText(
            results_frame, wrap=tk.WORD, height=15, state=tk.DISABLED,
            font=("Courier", 9))
        self.bulk_results.pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------ #
    #  Browse helpers
    # ------------------------------------------------------------------ #
    _IMAGE_TYPES = [
        ("Image files", "*.tif *.tiff *.ome.tif *.png"),
        ("All files", "*.*"),
    ]
    _MASK_TYPES = [
        ("Mask files", "*.npy *.tif *.tiff *.png"),
        ("All files", "*.*"),
    ]

    def _browse_image(self):
        p = filedialog.askopenfilename(title="Select Original Image", filetypes=self._IMAGE_TYPES)
        if p:
            self.image_var.set(p)
            self.output_var.set(str(Path(p).parent / "comparison_report"))

    def _browse_mask1(self):
        p = filedialog.askopenfilename(title="Select Mask 1 (Ground Truth)", filetypes=self._MASK_TYPES)
        if p:
            self.mask1_var.set(p)

    def _browse_mask2(self):
        p = filedialog.askopenfilename(title="Select Mask 2 (Model Output)", filetypes=self._MASK_TYPES)
        if p:
            self.mask2_var.set(p)

    def _browse_dir(self, var):
        d = filedialog.askdirectory(title="Select Directory")
        if d:
            var.set(d)

    # ------------------------------------------------------------------ #
    #  Single pair: validation & run
    # ------------------------------------------------------------------ #
    def _validate_single(self) -> bool:
        for var, name in [
            (self.image_var, "original image"),
            (self.mask1_var, "Mask 1"),
            (self.mask2_var, "Mask 2"),
        ]:
            if not var.get() or not os.path.isfile(var.get()):
                messagebox.showwarning("Missing", f"Select a valid {name} file.")
                return False
        return True

    def _single_set_running(self, running: bool):
        state = tk.DISABLED if running else tk.NORMAL
        self.btn_run.config(state=state)
        self.btn_quick.config(state=state)
        if running:
            self.single_progress.start(10)
        else:
            self.single_progress.stop()

    def _run_single(self):
        if not self._validate_single():
            return
        self._single_set_running(True)
        self.single_status.set("Running comparison...")
        self._clear_text(self.single_results)

        def worker():
            try:
                results = compare(
                    self.image_var.get(),
                    self.mask1_var.get(),
                    self.mask2_var.get(),
                    self.output_var.get(),
                )
                text = format_single_results(results)
                self.log_queue.put(f"__SCMP_DONE__{text}")

                if self.open_figure.get():
                    fig_path = os.path.join(self.output_var.get(), "comparison.png")
                    if os.path.isfile(fig_path):
                        self.log_queue.put(f"__CMP_OPEN__{fig_path}")
            except Exception as e:
                self.log_queue.put(f"__SCMP_ERROR__{e}")

        threading.Thread(target=worker, daemon=True).start()

    def _run_single_quick(self):
        if not self._validate_single():
            return
        self._single_set_running(True)
        self.single_status.set("Computing metrics...")
        self._clear_text(self.single_results)

        def worker():
            try:
                image = load_image(self.image_var.get())
                mask1 = load_mask(self.mask1_var.get())
                mask2 = load_mask(self.mask2_var.get())
                binary = binary_metrics(mask1, mask2)
                matches, _, _ = match_objects(mask1, mask2)
                ap = average_precision(mask1, mask2)
                stats1 = intensity_stats(image, mask1)
                stats2 = intensity_stats(image, mask2)
                results = {
                    "binary": binary, "matches": matches, "ap": ap,
                    "stats_mask1": stats1, "stats_mask2": stats2,
                }
                text = format_single_results(results)
                self.log_queue.put(f"__SCMP_DONE__{text}")
            except Exception as e:
                self.log_queue.put(f"__SCMP_ERROR__{e}")

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------ #
    #  Bulk: validation, preview & run
    # ------------------------------------------------------------------ #
    def _validate_bulk(self) -> bool:
        for var, name in [
            (self.bulk_img_dir, "Images Directory"),
            (self.bulk_mask1_dir, "Mask 1 Directory"),
            (self.bulk_mask2_dir, "Mask 2 Directory"),
        ]:
            if not var.get() or not os.path.isdir(var.get()):
                messagebox.showwarning("Missing", f"Select a valid {name}.")
                return False
        return True

    def _bulk_set_running(self, running: bool):
        state = tk.DISABLED if running else tk.NORMAL
        self.btn_bulk_run.config(state=state)
        self.btn_bulk_preview.config(state=state)
        if not running:
            self.bulk_progress.config(value=0)

    def _bulk_preview(self):
        """Show which files will be matched without running comparison."""
        if not self._validate_bulk():
            return
        try:
            matched, warnings = auto_match(
                self.bulk_img_dir.get(),
                self.bulk_mask1_dir.get(),
                self.bulk_mask2_dir.get(),
            )
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        lines = []
        lines.append(f"Found {len(matched)} matched triplets:\n")
        for img, m1, m2 in matched:
            lines.append(f"  {os.path.basename(img)}")
            lines.append(f"    M1: {os.path.basename(m1)}")
            lines.append(f"    M2: {os.path.basename(m2)}")
            lines.append("")

        if warnings:
            lines.append(f"\nWarnings ({len(warnings)}):")
            for w in warnings:
                lines.append(f"  {w}")

        self._clear_text(self.bulk_results)
        self._append_text(self.bulk_results, "\n".join(lines))
        self.bulk_status.set(f"{len(matched)} pairs matched, {len(warnings)} warnings")

    def _run_bulk(self):
        if not self._validate_bulk():
            return
        self._bulk_set_running(True)
        self.bulk_status.set("Running batch comparison...")
        self._clear_text(self.bulk_results)

        def progress_cb(current, total, key, _result):
            pct = int(100 * current / total) if total > 0 else 0
            self.log_queue.put(f"__BCMP_PROGRESS__{pct}||{current}||{total}||{key}")

        def worker():
            try:
                batch = compare_batch(
                    self.bulk_img_dir.get(),
                    self.bulk_mask1_dir.get(),
                    self.bulk_mask2_dir.get(),
                    self.bulk_output_var.get(),
                    progress_callback=progress_cb,
                )
                text = format_batch_results(batch)
                self.log_queue.put(f"__BCMP_DONE__{text}")
            except Exception as e:
                self.log_queue.put(f"__BCMP_ERROR__{e}")

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------ #
    #  Text widget helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _clear_text(widget):
        widget.config(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.config(state=tk.DISABLED)

    @staticmethod
    def _append_text(widget, text):
        widget.config(state=tk.NORMAL)
        widget.insert(tk.END, text)
        widget.see(tk.END)
        widget.config(state=tk.DISABLED)

    # ------------------------------------------------------------------ #
    #  Log queue polling
    # ------------------------------------------------------------------ #
    def _poll_queue(self):
        while True:
            try:
                msg = self.log_queue.get_nowait()
            except queue.Empty:
                break

            # Single pair
            if msg.startswith("__SCMP_DONE__"):
                self._single_set_running(False)
                self.single_status.set("Comparison complete")
                self._append_text(self.single_results, msg[len("__SCMP_DONE__"):])
                continue
            if msg.startswith("__SCMP_ERROR__"):
                self._single_set_running(False)
                err = msg[len("__SCMP_ERROR__"):]
                self.single_status.set(f"Error: {err}")
                messagebox.showerror("Comparison Error", str(err))
                continue

            # Bulk
            if msg.startswith("__BCMP_PROGRESS__"):
                parts = msg[len("__BCMP_PROGRESS__"):].split("||")
                pct = int(parts[0])
                cur, tot, key = parts[1], parts[2], parts[3]
                self.bulk_progress.config(value=pct)
                self.bulk_status.set(f"Comparing {cur}/{tot}: {key}")
                continue
            if msg.startswith("__BCMP_DONE__"):
                self._bulk_set_running(False)
                self.bulk_status.set("Batch comparison complete")
                self._append_text(self.bulk_results, msg[len("__BCMP_DONE__"):])
                continue
            if msg.startswith("__BCMP_ERROR__"):
                self._bulk_set_running(False)
                err = msg[len("__BCMP_ERROR__"):]
                self.bulk_status.set(f"Error: {err}")
                messagebox.showerror("Batch Error", str(err))
                continue

            # Open figure
            if msg.startswith("__CMP_OPEN__"):
                open_figure(msg[len("__CMP_OPEN__"):])
                continue

        self.after(100, self._poll_queue)


def main():
    app = CompareMasksGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
