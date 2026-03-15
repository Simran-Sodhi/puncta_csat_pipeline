"""
Standalone GUI for comparing two segmentation masks against an original image.

Supports:
  - Image formats: .tif, .ome.tif, .tiff, .png
  - Mask formats:  _seg.npy (Cellpose), .tif, .tiff, .png

Computes IoU, Dice, Average Precision, per-object matching,
intensity statistics, and generates visual reports.

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
# Embedded comparison from PIL-free matplotlib
# ---------------------------------------------------------------------------
try:
    from compare_masks import (
        compare,
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
        load_image,
        load_mask,
        binary_metrics,
        match_objects,
        intensity_stats,
        average_precision,
    )


class CompareMasksGUI(tk.Tk):
    """Standalone mask comparison application."""

    def __init__(self):
        super().__init__()
        self.title("Mask Comparison Tool")
        self.geometry("820x750")
        self.minsize(700, 600)

        self.log_queue: queue.Queue[str] = queue.Queue()
        self._build_ui()
        self._poll_queue()

    # ------------------------------------------------------------------ #
    #  UI
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        # ---- Description ----
        ttk.Label(
            self,
            text="Compare two segmentation masks (ground truth vs model) "
                 "with per-object IoU, intensity, and visual reports.",
            foreground="gray",
        ).pack(anchor=tk.W, padx=10, pady=(10, 5))

        # ---- Input files ----
        io_frame = ttk.LabelFrame(self, text="Input Files", padding=10)
        io_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(io_frame, text="Original Image:").grid(row=0, column=0, sticky=tk.W, pady=3)
        self.image_var = tk.StringVar()
        ttk.Entry(io_frame, textvariable=self.image_var, width=55).grid(
            row=0, column=1, padx=5, pady=3, sticky=tk.EW
        )
        ttk.Button(io_frame, text="Browse...", command=self._browse_image).grid(
            row=0, column=2, pady=3
        )

        ttk.Label(io_frame, text="Mask 1 (Ground Truth):").grid(row=1, column=0, sticky=tk.W, pady=3)
        self.mask1_var = tk.StringVar()
        ttk.Entry(io_frame, textvariable=self.mask1_var, width=55).grid(
            row=1, column=1, padx=5, pady=3, sticky=tk.EW
        )
        ttk.Button(io_frame, text="Browse...", command=self._browse_mask1).grid(
            row=1, column=2, pady=3
        )

        ttk.Label(io_frame, text="Mask 2 (Model Output):").grid(row=2, column=0, sticky=tk.W, pady=3)
        self.mask2_var = tk.StringVar()
        ttk.Entry(io_frame, textvariable=self.mask2_var, width=55).grid(
            row=2, column=1, padx=5, pady=3, sticky=tk.EW
        )
        ttk.Button(io_frame, text="Browse...", command=self._browse_mask2).grid(
            row=2, column=2, pady=3
        )

        io_frame.columnconfigure(1, weight=1)

        # ---- Output ----
        out_frame = ttk.LabelFrame(self, text="Output", padding=10)
        out_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(out_frame, text="Report Directory:").grid(row=0, column=0, sticky=tk.W, pady=3)
        self.output_var = tk.StringVar(value="comparison_report")
        ttk.Entry(out_frame, textvariable=self.output_var, width=55).grid(
            row=0, column=1, padx=5, pady=3, sticky=tk.EW
        )
        ttk.Button(out_frame, text="Browse...", command=self._browse_output).grid(
            row=0, column=2, pady=3
        )

        self.open_figure = tk.BooleanVar(value=True)
        ttk.Checkbutton(out_frame, text="Open comparison figure when done", variable=self.open_figure).grid(
            row=1, column=0, columnspan=3, sticky=tk.W, pady=2
        )

        out_frame.columnconfigure(1, weight=1)

        # ---- Buttons ----
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)

        self.btn_run = ttk.Button(btn_frame, text="Run Comparison", command=self._run)
        self.btn_run.pack(side=tk.LEFT, padx=5)

        self.btn_quick = ttk.Button(btn_frame, text="Quick Summary (no figure)", command=self._run_quick)
        self.btn_quick.pack(side=tk.LEFT, padx=5)

        # ---- Progress ----
        self.progress = ttk.Progressbar(self, mode="indeterminate")
        self.progress.pack(fill=tk.X, padx=10, pady=5)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self, textvariable=self.status_var).pack(padx=10, anchor=tk.W)

        # ---- Results ----
        results_frame = ttk.LabelFrame(self, text="Results", padding=5)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))

        self.results_text = scrolledtext.ScrolledText(
            results_frame, wrap=tk.WORD, height=18, state=tk.DISABLED,
            font=("Courier", 9),
        )
        self.results_text.pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------ #
    #  Browse helpers
    # ------------------------------------------------------------------ #
    _IMAGE_TYPES = [
        ("Image files", "*.tif *.tiff *.ome.tif *.png"),
        ("All files", "*.*"),
    ]
    _MASK_TYPES = [
        ("Mask files", "*.npy *.tif *.tiff *.png"),
        ("Cellpose _seg.npy", "*.npy"),
        ("TIFF masks", "*.tif *.tiff"),
        ("All files", "*.*"),
    ]

    def _browse_image(self):
        p = filedialog.askopenfilename(title="Select Original Image", filetypes=self._IMAGE_TYPES)
        if p:
            self.image_var.set(p)
            # Auto-set output dir next to the image
            self.output_var.set(str(Path(p).parent / "comparison_report"))

    def _browse_mask1(self):
        p = filedialog.askopenfilename(title="Select Mask 1 (Ground Truth)", filetypes=self._MASK_TYPES)
        if p:
            self.mask1_var.set(p)

    def _browse_mask2(self):
        p = filedialog.askopenfilename(title="Select Mask 2 (Model Output)", filetypes=self._MASK_TYPES)
        if p:
            self.mask2_var.set(p)

    def _browse_output(self):
        d = filedialog.askdirectory(title="Select Output Directory")
        if d:
            self.output_var.set(d)

    # ------------------------------------------------------------------ #
    #  Validation
    # ------------------------------------------------------------------ #
    def _validate(self) -> bool:
        if not self.image_var.get() or not os.path.isfile(self.image_var.get()):
            messagebox.showwarning("Missing", "Select a valid original image file.")
            return False
        if not self.mask1_var.get() or not os.path.isfile(self.mask1_var.get()):
            messagebox.showwarning("Missing", "Select a valid Mask 1 file.")
            return False
        if not self.mask2_var.get() or not os.path.isfile(self.mask2_var.get()):
            messagebox.showwarning("Missing", "Select a valid Mask 2 file.")
            return False
        return True

    # ------------------------------------------------------------------ #
    #  Run: full comparison with figure
    # ------------------------------------------------------------------ #
    def _run(self):
        if not self._validate():
            return
        self._set_running(True)
        self.status_var.set("Running comparison...")
        self._clear_results()

        def worker():
            try:
                results = compare(
                    self.image_var.get(),
                    self.mask1_var.get(),
                    self.mask2_var.get(),
                    self.output_var.get(),
                )
                text = self._format_results(results)
                self.log_queue.put(f"__CMP_DONE__{text}")

                if self.open_figure.get():
                    fig_path = os.path.join(self.output_var.get(), "comparison.png")
                    if os.path.isfile(fig_path):
                        self.log_queue.put(f"__CMP_OPEN__{fig_path}")
            except Exception as e:
                self.log_queue.put(f"__CMP_ERROR__{e}")

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------ #
    #  Run: quick summary (no figure, in-memory only)
    # ------------------------------------------------------------------ #
    def _run_quick(self):
        if not self._validate():
            return
        self._set_running(True)
        self.status_var.set("Computing metrics...")
        self._clear_results()

        def worker():
            try:
                import numpy as np
                image = load_image(self.image_var.get())
                mask1 = load_mask(self.mask1_var.get())
                mask2 = load_mask(self.mask2_var.get())

                binary = binary_metrics(mask1, mask2)
                matches, u1, u2 = match_objects(mask1, mask2)
                ap = average_precision(mask1, mask2)
                stats1 = intensity_stats(image, mask1)
                stats2 = intensity_stats(image, mask2)

                results = {
                    "binary": binary,
                    "matches": matches,
                    "ap": ap,
                    "stats_mask1": stats1,
                    "stats_mask2": stats2,
                }
                text = self._format_results(results)
                self.log_queue.put(f"__CMP_DONE__{text}")
            except Exception as e:
                self.log_queue.put(f"__CMP_ERROR__{e}")

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------ #
    #  Format results for display
    # ------------------------------------------------------------------ #
    @staticmethod
    def _format_results(results: dict) -> str:
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

        # Top 10 best/worst matched objects
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

    # ------------------------------------------------------------------ #
    #  UI state helpers
    # ------------------------------------------------------------------ #
    def _set_running(self, running: bool):
        state = tk.DISABLED if running else tk.NORMAL
        self.btn_run.config(state=state)
        self.btn_quick.config(state=state)
        if running:
            self.progress.start(10)
        else:
            self.progress.stop()

    def _clear_results(self):
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete("1.0", tk.END)
        self.results_text.config(state=tk.DISABLED)

    def _append_results(self, text: str):
        self.results_text.config(state=tk.NORMAL)
        self.results_text.insert(tk.END, text)
        self.results_text.see(tk.END)
        self.results_text.config(state=tk.DISABLED)

    # ------------------------------------------------------------------ #
    #  Log queue polling
    # ------------------------------------------------------------------ #
    def _poll_queue(self):
        while True:
            try:
                msg = self.log_queue.get_nowait()
            except queue.Empty:
                break

            if msg.startswith("__CMP_DONE__"):
                self._set_running(False)
                self.status_var.set("Comparison complete")
                self._append_results(msg[len("__CMP_DONE__"):])
                continue
            if msg.startswith("__CMP_ERROR__"):
                self._set_running(False)
                err = msg[len("__CMP_ERROR__"):]
                self.status_var.set(f"Error: {err}")
                messagebox.showerror("Comparison Error", str(err))
                continue
            if msg.startswith("__CMP_OPEN__"):
                fig_path = msg[len("__CMP_OPEN__"):]
                self._open_figure(fig_path)
                continue

        self.after(100, self._poll_queue)

    @staticmethod
    def _open_figure(path: str):
        """Open the comparison figure with the system default viewer."""
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


def main():
    app = CompareMasksGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
