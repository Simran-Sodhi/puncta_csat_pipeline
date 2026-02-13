#!/usr/bin/env python3
"""
Tkinter GUI for the Cellpose Segmentation Training Pipeline.

Provides tabs for:
  1. File Renaming  — Rename images/masks to pipeline convention
  2. Configuration  — Edit training parameters from YAML configs
  3. Training       — Launch and monitor model training
  4. Evaluation     — Run inference and view results
"""

import logging
import os
import queue
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path

import yaml

# Ensure src/ modules are importable
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from rename_files import (
    list_image_files,
    rename_files,
    rename_image_mask_pair,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging handler that writes to a tkinter Text widget via a thread-safe queue
# ---------------------------------------------------------------------------
class QueueHandler(logging.Handler):
    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------
class SegmentationGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Cellpose Segmentation Pipeline")
        self.geometry("960x720")
        self.minsize(800, 600)

        # Thread-safe log queue
        self.log_queue: queue.Queue[str] = queue.Queue()

        # Currently loaded YAML config (named to avoid shadowing tk.Tk.config)
        self.yaml_config: dict = {}
        self.yaml_config_path: str = ""

        self._build_menu()
        self._build_tabs()
        self._setup_logging()
        self._poll_log_queue()

    # ----- Menu bar -----
    def _build_menu(self):
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Config...", command=self._open_config)
        file_menu.add_command(label="Save Config", command=self._save_config)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=menubar)

    # ----- Notebook tabs -----
    def _build_tabs(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.tab_preprocess = ttk.Frame(self.notebook)
        self.tab_rename = ttk.Frame(self.notebook)
        self.tab_config = ttk.Frame(self.notebook)
        self.tab_train = ttk.Frame(self.notebook)
        self.tab_eval = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_preprocess, text="  Preprocessing  ")
        self.notebook.add(self.tab_rename, text="  File Renaming  ")
        self.notebook.add(self.tab_config, text="  Configuration  ")
        self.notebook.add(self.tab_train, text="  Training  ")
        self.notebook.add(self.tab_eval, text="  Evaluation  ")

        self._build_preprocess_tab()
        self._build_rename_tab()
        self._build_config_tab()
        self._build_train_tab()
        self._build_eval_tab()

    # ==================================================================
    # TAB 0: PREPROCESSING
    # ==================================================================
    def _build_preprocess_tab(self):
        tab = self.tab_preprocess

        # ---- Input / Output ----
        io_frame = ttk.LabelFrame(tab, text="Directories", padding=10)
        io_frame.pack(fill=tk.X, padx=10, pady=(10, 5))

        ttk.Label(io_frame, text="Image Directory:").grid(
            row=0, column=0, sticky=tk.W, pady=2
        )
        self.pp_img_dir = tk.StringVar()
        ttk.Entry(io_frame, textvariable=self.pp_img_dir, width=55).grid(
            row=0, column=1, padx=5, pady=2
        )
        ttk.Button(io_frame, text="Browse...", command=self._pp_browse_img).grid(
            row=0, column=2, pady=2
        )

        ttk.Label(io_frame, text="Mask Output Directory:").grid(
            row=1, column=0, sticky=tk.W, pady=2
        )
        self.pp_out_dir = tk.StringVar(
            value=str(PROJECT_ROOT / "data" / "draft_masks")
        )
        ttk.Entry(io_frame, textvariable=self.pp_out_dir, width=55).grid(
            row=1, column=1, padx=5, pady=2
        )
        ttk.Button(io_frame, text="Browse...", command=self._pp_browse_out).grid(
            row=1, column=2, pady=2
        )

        # ---- Channel & Normalization Settings ----
        settings_frame = ttk.LabelFrame(tab, text="Channel & Normalization", padding=10)
        settings_frame.pack(fill=tk.X, padx=10, pady=5)

        # Segment channel
        ttk.Label(settings_frame, text="Segment Channel:").grid(
            row=0, column=0, sticky=tk.W, pady=2
        )
        self.pp_seg_channel = tk.IntVar(value=0)
        seg_combo = ttk.Combobox(
            settings_frame,
            textvariable=self.pp_seg_channel,
            values=[0, 1, 2, 3],
            width=5,
            state="readonly",
        )
        seg_combo.grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        self.pp_seg_label = tk.StringVar(value="DIC")
        ttk.Label(settings_frame, textvariable=self.pp_seg_label, foreground="gray").grid(
            row=0, column=2, sticky=tk.W, padx=5
        )
        seg_combo.bind("<<ComboboxSelected>>", self._pp_update_channel_labels)

        # Nuclear channel
        ttk.Label(settings_frame, text="Nuclear Channel:").grid(
            row=1, column=0, sticky=tk.W, pady=2
        )
        self.pp_nuc_channel = tk.IntVar(value=0)
        nuc_combo = ttk.Combobox(
            settings_frame,
            textvariable=self.pp_nuc_channel,
            values=[0, 1, 2, 3],
            width=5,
            state="readonly",
        )
        nuc_combo.grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)
        self.pp_nuc_label = tk.StringVar(value="None (grayscale)")
        ttk.Label(settings_frame, textvariable=self.pp_nuc_label, foreground="gray").grid(
            row=1, column=2, sticky=tk.W, padx=5
        )
        nuc_combo.bind("<<ComboboxSelected>>", self._pp_update_channel_labels)

        # Percentile range
        ttk.Label(settings_frame, text="Normalization Percentile:").grid(
            row=2, column=0, sticky=tk.W, pady=2
        )
        pct_frame = ttk.Frame(settings_frame)
        pct_frame.grid(row=2, column=1, columnspan=2, sticky=tk.W, padx=5)
        ttk.Label(pct_frame, text="Low:").pack(side=tk.LEFT)
        self.pp_lower_pct = tk.DoubleVar(value=1.0)
        ttk.Entry(pct_frame, textvariable=self.pp_lower_pct, width=6).pack(
            side=tk.LEFT, padx=(2, 10)
        )
        ttk.Label(pct_frame, text="High:").pack(side=tk.LEFT)
        self.pp_upper_pct = tk.DoubleVar(value=99.0)
        ttk.Entry(pct_frame, textvariable=self.pp_upper_pct, width=6).pack(
            side=tk.LEFT, padx=2
        )

        # Tile normalization
        ttk.Label(settings_frame, text="DIC Tile Blocksize:").grid(
            row=3, column=0, sticky=tk.W, pady=2
        )
        self.pp_tile_bs = tk.IntVar(value=128)
        ttk.Entry(settings_frame, textvariable=self.pp_tile_bs, width=8).grid(
            row=3, column=1, padx=5, pady=2, sticky=tk.W
        )
        ttk.Label(
            settings_frame,
            text="(0 = global normalization, >0 = tile-based for uneven illumination)",
            foreground="gray",
        ).grid(row=3, column=2, sticky=tk.W, padx=5)

        # Z-slice for ND2 Z-stacks
        ttk.Label(settings_frame, text="Z-Slice (ND2):").grid(
            row=4, column=0, sticky=tk.W, pady=2
        )
        z_frame = ttk.Frame(settings_frame)
        z_frame.grid(row=4, column=1, columnspan=2, sticky=tk.W, padx=5)
        self.pp_z_slice = tk.StringVar(value="0")
        ttk.Entry(z_frame, textvariable=self.pp_z_slice, width=6).pack(
            side=tk.LEFT, padx=(0, 10)
        )
        ttk.Label(
            z_frame,
            text="(0-indexed Z-slice for Z-stack .nd2 files)",
            foreground="gray",
        ).pack(side=tk.LEFT)

        # Invert DIC
        self.pp_invert_dic = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            settings_frame,
            text="Invert DIC (cells dark on bright background)",
            variable=self.pp_invert_dic,
        ).grid(row=5, column=0, columnspan=3, sticky=tk.W, pady=2)

        # ---- Cellpose Settings ----
        cp_frame = ttk.LabelFrame(tab, text="Cellpose Settings", padding=10)
        cp_frame.pack(fill=tk.X, padx=10, pady=5)

        # Model
        ttk.Label(cp_frame, text="Model:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.pp_model_var = tk.StringVar(value="cpsam (default)")
        pp_model_combo = ttk.Combobox(
            cp_frame,
            textvariable=self.pp_model_var,
            values=["cpsam (default)", "Custom model...", "BioImage.io model..."],
            width=22,
            state="readonly",
        )
        pp_model_combo.grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        pp_model_combo.bind("<<ComboboxSelected>>", self._pp_on_model_changed)

        self.pp_custom_model = tk.StringVar()
        self.pp_model_entry = ttk.Entry(
            cp_frame, textvariable=self.pp_custom_model, width=35
        )
        self.pp_model_entry.grid(row=0, column=2, padx=5, pady=2)
        self.pp_model_entry.config(state=tk.DISABLED)
        self.pp_model_btn = ttk.Button(
            cp_frame, text="Browse...", command=self._pp_browse_model
        )
        self.pp_model_btn.grid(row=0, column=3, pady=2)
        self.pp_model_btn.config(state=tk.DISABLED)

        # Diameter
        ttk.Label(cp_frame, text="Diameter:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.pp_diameter = tk.StringVar(value="auto")
        ttk.Entry(cp_frame, textvariable=self.pp_diameter, width=10).grid(
            row=1, column=1, padx=5, pady=2, sticky=tk.W
        )
        ttk.Label(cp_frame, text="(pixels, or 'auto')", foreground="gray").grid(
            row=1, column=2, sticky=tk.W, padx=5
        )

        # Flow threshold
        ttk.Label(cp_frame, text="Flow Threshold:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.pp_flow_thr = tk.DoubleVar(value=0.4)
        ttk.Entry(cp_frame, textvariable=self.pp_flow_thr, width=10).grid(
            row=2, column=1, padx=5, pady=2, sticky=tk.W
        )

        # Cell prob threshold
        ttk.Label(cp_frame, text="Cell Prob Threshold:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.pp_cellprob_thr = tk.DoubleVar(value=0.0)
        ttk.Entry(cp_frame, textvariable=self.pp_cellprob_thr, width=10).grid(
            row=3, column=1, padx=5, pady=2, sticky=tk.W
        )

        # ---- Buttons ----
        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(
            btn_frame, text="Preview Channels", command=self._pp_preview_channels
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            btn_frame, text="Expand ND2 Positions", command=self._pp_expand_nd2
        ).pack(side=tk.LEFT, padx=5)

        self.btn_pp_run = ttk.Button(
            btn_frame, text="Generate Masks", command=self._pp_generate
        )
        self.btn_pp_run.pack(side=tk.LEFT, padx=5)

        # Progress
        self.pp_progress = ttk.Progressbar(tab, mode="determinate")
        self.pp_progress.pack(fill=tk.X, padx=10, pady=5)

        self.pp_status = tk.StringVar(value="Ready")
        ttk.Label(tab, textvariable=self.pp_status).pack(padx=10, anchor=tk.W)

        # Results table
        result_frame = ttk.LabelFrame(tab, text="Results", padding=5)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))

        columns = ("filename", "objects", "status")
        self.pp_tree = ttk.Treeview(
            result_frame, columns=columns, show="headings", height=8
        )
        self.pp_tree.heading("filename", text="Filename")
        self.pp_tree.heading("objects", text="Objects Found")
        self.pp_tree.heading("status", text="Status")
        self.pp_tree.column("filename", width=350)
        self.pp_tree.column("objects", width=120, anchor=tk.CENTER)
        self.pp_tree.column("status", width=120, anchor=tk.CENTER)

        pp_scroll = ttk.Scrollbar(
            result_frame, orient=tk.VERTICAL, command=self.pp_tree.yview
        )
        self.pp_tree.configure(yscrollcommand=pp_scroll.set)
        self.pp_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        pp_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _pp_browse_img(self):
        d = filedialog.askdirectory(title="Select Image Directory")
        if d:
            self.pp_img_dir.set(d)

    def _pp_browse_out(self):
        d = filedialog.askdirectory(title="Select Mask Output Directory")
        if d:
            self.pp_out_dir.set(d)

    def _pp_on_model_changed(self, event=None):
        choice = self.pp_model_var.get()
        if choice == "Custom model...":
            self.pp_model_entry.config(state=tk.NORMAL)
            self.pp_model_btn.config(state=tk.NORMAL)
            self.pp_custom_model.set("")
        elif choice == "BioImage.io model...":
            self.pp_model_entry.config(state=tk.NORMAL)
            self.pp_model_btn.config(state=tk.NORMAL)
            self.pp_custom_model.set("bioimage.io:")
        else:
            self.pp_model_entry.config(state=tk.DISABLED)
            self.pp_model_btn.config(state=tk.DISABLED)
            self.pp_custom_model.set("")

    def _pp_browse_model(self):
        if self.pp_model_var.get() == "BioImage.io model...":
            path = filedialog.askopenfilename(
                title="Select BioImage.io Model (rdf.yaml / .zip)",
                initialdir=str(PROJECT_ROOT / "models"),
                filetypes=[
                    ("BioImage.io", "*.yaml *.yml *.zip"),
                    ("All", "*.*"),
                ],
            )
        else:
            path = filedialog.askopenfilename(
                title="Select Cellpose Model",
                initialdir=str(PROJECT_ROOT / "models"),
            )
        if path:
            self.pp_custom_model.set(path)

    _CHANNEL_NAMES = {0: "DIC", 1: "mEGFP", 2: "mScarlet", 3: "miRFPnano3"}

    def _pp_update_channel_labels(self, event=None):
        seg = self.pp_seg_channel.get()
        nuc = self.pp_nuc_channel.get()
        self.pp_seg_label.set(self._CHANNEL_NAMES.get(seg, f"Channel {seg}"))
        if nuc == 0:
            self.pp_nuc_label.set("None (grayscale)")
        else:
            self.pp_nuc_label.set(self._CHANNEL_NAMES.get(nuc, f"Channel {nuc}"))

    def _pp_preview_channels(self):
        """Save and open a channel preview for the first image."""
        img_dir = self.pp_img_dir.get()
        if not img_dir:
            messagebox.showwarning("Missing", "Select an image directory first.")
            return

        from preprocess import save_channel_preview
        import glob as _glob

        extensions = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.nd2")
        files = []
        for ext in extensions:
            files.extend(_glob.glob(os.path.join(img_dir, ext)))
        files = sorted(files)

        if not files:
            messagebox.showinfo("Empty", "No images found in the directory.")
            return

        preview_dir = str(PROJECT_ROOT / "results" / "channel_previews")
        z_val = self.pp_z_slice.get().strip()
        z_slice = int(z_val) if z_val else None

        try:
            path = save_channel_preview(
                files[0],
                preview_dir,
                lower_percentile=self.pp_lower_pct.get(),
                upper_percentile=self.pp_upper_pct.get(),
                tile_blocksize_dic=self.pp_tile_bs.get(),
                z_slice=z_slice,
            )
            self.pp_status.set(f"Channel preview saved: {path}")
            logger.info(f"Channel preview saved to {path}")
            messagebox.showinfo(
                "Preview Saved",
                f"Channel preview for {Path(files[0]).name} saved to:\n{path}\n\n"
                "Open this file to inspect channel quality and normalization.",
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate preview:\n{e}")
            logger.exception("Channel preview failed")

    def _pp_expand_nd2(self):
        """Expand multi-position .nd2 files in the image directory into TIFFs."""
        img_dir = self.pp_img_dir.get()
        if not img_dir:
            messagebox.showwarning("Missing", "Select an image directory first.")
            return

        import glob as _glob
        nd2_files = sorted(_glob.glob(os.path.join(img_dir, "*.nd2")))
        if not nd2_files:
            messagebox.showinfo("No ND2", "No .nd2 files found in the directory.")
            return

        from data_preparation import expand_nd2_positions, get_nd2_info

        output_dir = os.path.join(img_dir, "_nd2_expanded")
        total_saved = []
        multi_pos_count = 0

        for nd2_file in nd2_files:
            try:
                info = get_nd2_info(nd2_file)
                n_pos = info["n_positions"]
                if n_pos > 1:
                    multi_pos_count += 1
                    logger.info(
                        f"Expanding {Path(nd2_file).name}: "
                        f"{n_pos} positions, shape={info['shape']}"
                    )
                saved = expand_nd2_positions(nd2_file, output_dir)
                total_saved.extend(saved)
            except Exception as e:
                logger.error(f"Failed to expand {Path(nd2_file).name}: {e}")
                messagebox.showerror(
                    "ND2 Error",
                    f"Failed to expand {Path(nd2_file).name}:\n{e}",
                )

        if total_saved:
            messagebox.showinfo(
                "ND2 Expanded",
                f"Processed {len(nd2_files)} ND2 file(s) "
                f"({multi_pos_count} multi-position).\n"
                f"Saved {len(total_saved)} individual TIFFs to:\n{output_dir}\n\n"
                "You can now use this directory as the image input, or the "
                "pipeline will auto-expand during mask generation.",
            )
            self.pp_status.set(
                f"Expanded {len(nd2_files)} ND2 -> {len(total_saved)} TIFFs"
            )

    def _pp_generate(self):
        """Generate draft masks in a background thread."""
        img_dir = self.pp_img_dir.get()
        out_dir = self.pp_out_dir.get()
        if not img_dir:
            messagebox.showwarning("Missing", "Select an image directory.")
            return

        self.btn_pp_run.config(state=tk.DISABLED)
        self.pp_tree.delete(*self.pp_tree.get_children())
        self.pp_progress.config(mode="determinate", value=0)
        self.pp_status.set("Generating masks...")

        # Parse diameter
        diam_str = self.pp_diameter.get().strip().lower()
        diameter = None if diam_str in ("auto", "none", "") else float(diam_str)

        # Parse Z-slice
        z_val = self.pp_z_slice.get().strip()
        z_slice = int(z_val) if z_val else None

        # Parse model (supports local paths and BioImage.io identifiers)
        model_path = None
        if self.pp_model_var.get() in ("Custom model...", "BioImage.io model..."):
            model_path = self.pp_custom_model.get() or None

        def progress_cb(current, total, filename):
            if total > 0:
                pct = int(100 * current / total)
                # Use thread-safe queue to update GUI
                self.log_queue.put(f"__PP_PROGRESS__{pct}||{current}/{total}: {filename}")

        def worker():
            try:
                from preprocess import generate_masks
                results = generate_masks(
                    image_dir=img_dir,
                    output_dir=out_dir,
                    segment_channel=self.pp_seg_channel.get(),
                    nuclear_channel=self.pp_nuc_channel.get(),
                    model_path=model_path,
                    diameter=diameter,
                    flow_threshold=self.pp_flow_thr.get(),
                    cellprob_threshold=self.pp_cellprob_thr.get(),
                    lower_percentile=self.pp_lower_pct.get(),
                    upper_percentile=self.pp_upper_pct.get(),
                    tile_blocksize_dic=self.pp_tile_bs.get(),
                    invert_dic=self.pp_invert_dic.get(),
                    use_gpu=True,
                    z_slice=z_slice,
                    progress_callback=progress_cb,
                )
                # Encode results for the GUI thread
                encoded = "|".join(f"{name},{n}" for name, n in results)
                self.log_queue.put(f"__PP_DONE__{encoded}")
            except Exception as e:
                logger.exception("Mask generation failed")
                self.log_queue.put(f"__PP_ERROR__{e}")

        threading.Thread(target=worker, daemon=True).start()

    def _on_pp_finished(self, results_str: str | None = None, error: str | None = None):
        self.btn_pp_run.config(state=tk.NORMAL)
        self.pp_progress.config(value=100)

        if error:
            self.pp_status.set(f"Error: {error}")
            messagebox.showerror("Preprocessing Error", str(error))
            return

        if results_str:
            entries = results_str.split("|")
            total_objects = 0
            for entry in entries:
                if not entry:
                    continue
                name, n_str = entry.rsplit(",", 1)
                n = int(n_str)
                if n < 0:
                    status = "FAILED"
                elif n == 0:
                    status = "No signal"
                else:
                    status = "OK"
                    total_objects += n
                self.pp_tree.insert("", tk.END, values=(name, n if n >= 0 else "-", status))

            self.pp_status.set(
                f"Done: {len(entries)} images processed, {total_objects} total objects. "
                f"Masks saved to: {self.pp_out_dir.get()}"
            )
            messagebox.showinfo(
                "Preprocessing Complete",
                f"Generated draft masks for {len(entries)} images.\n"
                f"Total objects detected: {total_objects}\n\n"
                f"Masks saved to:\n{self.pp_out_dir.get()}\n\n"
                "Review and curate the masks manually, then use the\n"
                "File Renaming tab to prepare them for training.",
            )

    # ==================================================================
    # TAB 1: FILE RENAMING
    # ==================================================================
    def _build_rename_tab(self):
        tab = self.tab_rename

        # ---- Top: directory selectors ----
        dir_frame = ttk.LabelFrame(tab, text="Directories", padding=10)
        dir_frame.pack(fill=tk.X, padx=10, pady=(10, 5))

        # Image directory
        ttk.Label(dir_frame, text="Image Directory:").grid(
            row=0, column=0, sticky=tk.W, pady=2
        )
        self.rename_img_dir = tk.StringVar()
        ttk.Entry(dir_frame, textvariable=self.rename_img_dir, width=60).grid(
            row=0, column=1, padx=5, pady=2
        )
        ttk.Button(dir_frame, text="Browse...", command=self._browse_img_dir).grid(
            row=0, column=2, pady=2
        )

        # Mask directory
        ttk.Label(dir_frame, text="Mask Directory:").grid(
            row=1, column=0, sticky=tk.W, pady=2
        )
        self.rename_mask_dir = tk.StringVar()
        ttk.Entry(dir_frame, textvariable=self.rename_mask_dir, width=60).grid(
            row=1, column=1, padx=5, pady=2
        )
        ttk.Button(dir_frame, text="Browse...", command=self._browse_mask_dir).grid(
            row=1, column=2, pady=2
        )

        # ---- Options ----
        opts_frame = ttk.LabelFrame(tab, text="Rename Options", padding=10)
        opts_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(opts_frame, text="Prefix:").grid(row=0, column=0, sticky=tk.W)
        self.rename_prefix = tk.StringVar(value="dic_")
        prefix_combo = ttk.Combobox(
            opts_frame,
            textvariable=self.rename_prefix,
            values=["dic_", "fluor_", "cell_", "nuc_", ""],
            width=15,
        )
        prefix_combo.grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)

        ttk.Label(opts_frame, text="Target Extension:").grid(
            row=0, column=2, sticky=tk.W, padx=(20, 0)
        )
        self.rename_ext = tk.StringVar(value=".tif")
        ext_combo = ttk.Combobox(
            opts_frame,
            textvariable=self.rename_ext,
            values=[".tif", ".png", "(keep original)"],
            width=15,
        )
        ext_combo.grid(row=0, column=3, padx=5, pady=2, sticky=tk.W)

        self.rename_copy_mode = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            opts_frame, text="Copy (don't rename in place)", variable=self.rename_copy_mode
        ).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=2)

        # Output dirs (for copy mode)
        ttk.Label(opts_frame, text="Image Output Dir:").grid(
            row=2, column=0, sticky=tk.W, pady=2
        )
        self.rename_img_out = tk.StringVar(
            value=str(PROJECT_ROOT / "data" / "train" / "dic_raw")
        )
        ttk.Entry(opts_frame, textvariable=self.rename_img_out, width=50).grid(
            row=2, column=1, columnspan=2, padx=5, pady=2, sticky=tk.W
        )
        ttk.Button(
            opts_frame, text="Browse...", command=self._browse_img_out
        ).grid(row=2, column=3, pady=2)

        ttk.Label(opts_frame, text="Mask Output Dir:").grid(
            row=3, column=0, sticky=tk.W, pady=2
        )
        self.rename_mask_out = tk.StringVar(
            value=str(PROJECT_ROOT / "data" / "train" / "dic_labels")
        )
        ttk.Entry(opts_frame, textvariable=self.rename_mask_out, width=50).grid(
            row=3, column=1, columnspan=2, padx=5, pady=2, sticky=tk.W
        )
        ttk.Button(
            opts_frame, text="Browse...", command=self._browse_mask_out
        ).grid(row=3, column=3, pady=2)

        # ---- Buttons ----
        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(btn_frame, text="Preview", command=self._rename_preview).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(btn_frame, text="Rename / Copy", command=self._rename_execute).pack(
            side=tk.LEFT, padx=5
        )

        # ---- Preview table ----
        table_frame = ttk.LabelFrame(tab, text="Preview", padding=5)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))

        columns = ("type", "original", "new_name")
        self.rename_tree = ttk.Treeview(
            table_frame, columns=columns, show="headings", height=12
        )
        self.rename_tree.heading("type", text="Type")
        self.rename_tree.heading("original", text="Original Name")
        self.rename_tree.heading("new_name", text="New Name")
        self.rename_tree.column("type", width=80, anchor=tk.CENTER)
        self.rename_tree.column("original", width=300)
        self.rename_tree.column("new_name", width=300)

        scrollbar = ttk.Scrollbar(
            table_frame, orient=tk.VERTICAL, command=self.rename_tree.yview
        )
        self.rename_tree.configure(yscrollcommand=scrollbar.set)
        self.rename_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _browse_img_dir(self):
        d = filedialog.askdirectory(title="Select Image Directory")
        if d:
            self.rename_img_dir.set(d)

    def _browse_mask_dir(self):
        d = filedialog.askdirectory(title="Select Mask Directory")
        if d:
            self.rename_mask_dir.set(d)

    def _browse_img_out(self):
        d = filedialog.askdirectory(title="Select Image Output Directory")
        if d:
            self.rename_img_out.set(d)

    def _browse_mask_out(self):
        d = filedialog.askdirectory(title="Select Mask Output Directory")
        if d:
            self.rename_mask_out.set(d)

    def _get_rename_ext(self) -> str | None:
        ext = self.rename_ext.get()
        if ext == "(keep original)":
            return None
        return ext

    def _rename_preview(self):
        """Dry-run rename and populate the preview table."""
        self.rename_tree.delete(*self.rename_tree.get_children())

        img_dir = self.rename_img_dir.get()
        mask_dir = self.rename_mask_dir.get()
        if not img_dir or not mask_dir:
            messagebox.showwarning("Missing", "Select both image and mask directories.")
            return

        prefix = self.rename_prefix.get()
        ext = self._get_rename_ext()

        img_renames, mask_renames = rename_image_mask_pair(
            image_dir=img_dir,
            mask_dir=mask_dir,
            prefix=prefix,
            target_extension=ext,
            dry_run=True,
            copy_mode=self.rename_copy_mode.get(),
            image_output_dir=self.rename_img_out.get() if self.rename_copy_mode.get() else None,
            mask_output_dir=self.rename_mask_out.get() if self.rename_copy_mode.get() else None,
        )

        for old, new in img_renames:
            self.rename_tree.insert("", tk.END, values=("Image", old, new))
        for old, new in mask_renames:
            self.rename_tree.insert("", tk.END, values=("Mask", old, new))

        total = len(img_renames) + len(mask_renames)
        if total == 0:
            messagebox.showinfo("Empty", "No image files found in the selected directories.")
        else:
            logger.info(f"Preview: {len(img_renames)} images, {len(mask_renames)} masks")

    def _rename_execute(self):
        """Execute the rename/copy operation."""
        img_dir = self.rename_img_dir.get()
        mask_dir = self.rename_mask_dir.get()
        if not img_dir or not mask_dir:
            messagebox.showwarning("Missing", "Select both image and mask directories.")
            return

        action = "copy" if self.rename_copy_mode.get() else "rename"
        count_img = len(list_image_files(img_dir))
        count_mask = len(list_image_files(mask_dir))

        if not messagebox.askyesno(
            "Confirm",
            f"This will {action} {count_img} images and {count_mask} masks.\n\nProceed?",
        ):
            return

        prefix = self.rename_prefix.get()
        ext = self._get_rename_ext()

        try:
            img_renames, mask_renames = rename_image_mask_pair(
                image_dir=img_dir,
                mask_dir=mask_dir,
                prefix=prefix,
                target_extension=ext,
                dry_run=False,
                copy_mode=self.rename_copy_mode.get(),
                image_output_dir=self.rename_img_out.get() if self.rename_copy_mode.get() else None,
                mask_output_dir=self.rename_mask_out.get() if self.rename_copy_mode.get() else None,
            )
            messagebox.showinfo(
                "Done",
                f"Successfully processed {len(img_renames)} images and "
                f"{len(mask_renames)} masks.",
            )
            # Refresh preview
            self._rename_preview()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            logger.exception("Rename failed")

    # ==================================================================
    # TAB 2: CONFIGURATION
    # ==================================================================
    def _build_config_tab(self):
        tab = self.tab_config

        # Config file selector
        top = ttk.Frame(tab)
        top.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(top, text="Config File:").pack(side=tk.LEFT)
        self.cfg_path_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.cfg_path_var, width=50).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(top, text="Open...", command=self._open_config).pack(side=tk.LEFT)
        ttk.Button(top, text="Save", command=self._save_config).pack(
            side=tk.LEFT, padx=5
        )

        # Preset buttons
        preset_frame = ttk.Frame(tab)
        preset_frame.pack(fill=tk.X, padx=10)
        ttk.Label(preset_frame, text="Presets:").pack(side=tk.LEFT)
        ttk.Button(
            preset_frame,
            text="DIC Whole-Cell",
            command=lambda: self._load_preset("dic"),
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            preset_frame,
            text="Fluorescence Nucleus",
            command=lambda: self._load_preset("fluor"),
        ).pack(side=tk.LEFT, padx=5)

        # ---- Model selection ----
        model_frame = ttk.LabelFrame(tab, text="Model", padding=10)
        model_frame.pack(fill=tk.X, padx=10, pady=(10, 0))

        ttk.Label(model_frame, text="Pretrained Model:").grid(
            row=0, column=0, sticky=tk.W, pady=2
        )
        self.model_source_var = tk.StringVar(value="cpsam (default)")
        model_combo = ttk.Combobox(
            model_frame,
            textvariable=self.model_source_var,
            values=["cpsam (default)", "Custom model...", "BioImage.io model..."],
            width=25,
            state="readonly",
        )
        model_combo.grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        model_combo.bind("<<ComboboxSelected>>", self._on_model_source_changed)

        self.custom_model_path_var = tk.StringVar()
        self.custom_model_entry = ttk.Entry(
            model_frame, textvariable=self.custom_model_path_var, width=40
        )
        self.custom_model_entry.grid(row=0, column=2, padx=5, pady=2)
        self.custom_model_entry.config(state=tk.DISABLED)

        self.custom_model_btn = ttk.Button(
            model_frame, text="Browse...", command=self._browse_custom_model
        )
        self.custom_model_btn.grid(row=0, column=3, pady=2)
        self.custom_model_btn.config(state=tk.DISABLED)

        self.model_info_var = tk.StringVar(
            value="Using built-in Cellpose-SAM (cpsam) model"
        )
        ttk.Label(
            model_frame, textvariable=self.model_info_var, foreground="gray"
        ).grid(row=1, column=0, columnspan=4, sticky=tk.W, pady=(2, 0))

        # Scrolled parameter editor
        param_frame = ttk.LabelFrame(tab, text="Parameters", padding=10)
        param_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        canvas = tk.Canvas(param_frame)
        scrollbar = ttk.Scrollbar(param_frame, orient=tk.VERTICAL, command=canvas.yview)
        self.param_inner = ttk.Frame(canvas)

        self.param_inner.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=self.param_inner, anchor=tk.NW)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Store param widgets for reading back
        self.param_vars: dict[str, tk.Variable] = {}

    def _open_config(self):
        path = filedialog.askopenfilename(
            title="Open Configuration",
            filetypes=[("YAML", "*.yaml *.yml"), ("All", "*.*")],
            initialdir=str(PROJECT_ROOT / "configs"),
        )
        if path:
            self._load_config_file(path)

    def _load_preset(self, task: str):
        presets = {
            "dic": PROJECT_ROOT / "configs" / "dic_wholecell.yaml",
            "fluor": PROJECT_ROOT / "configs" / "fluor_nucleus.yaml",
        }
        path = presets.get(task)
        if path and path.exists():
            self._load_config_file(str(path))
        else:
            messagebox.showerror("Error", f"Preset config not found: {path}")

    def _load_config_file(self, path: str):
        try:
            with open(path) as f:
                self.yaml_config = yaml.safe_load(f)
            self.yaml_config_path = path
            self.cfg_path_var.set(path)
            self._populate_param_editor()
            self._sync_model_selector_from_config()
            logger.info(f"Loaded config: {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load config:\n{e}")

    def _on_model_source_changed(self, event=None):
        """Toggle custom model path entry based on dropdown selection."""
        choice = self.model_source_var.get()
        if choice == "Custom model...":
            self.custom_model_entry.config(state=tk.NORMAL)
            self.custom_model_btn.config(state=tk.NORMAL)
            self.custom_model_path_var.set("")
            self.model_info_var.set("Select a custom model file below")
        elif choice == "BioImage.io model...":
            self.custom_model_entry.config(state=tk.NORMAL)
            self.custom_model_btn.config(state=tk.NORMAL)
            self.custom_model_path_var.set("bioimage.io:")
            self.model_info_var.set(
                "Enter a BioImage.io resource ID (e.g. bioimage.io:affable-shark) "
                "or browse for a rdf.yaml / .zip file"
            )
        else:
            self.custom_model_entry.config(state=tk.DISABLED)
            self.custom_model_btn.config(state=tk.DISABLED)
            self.custom_model_path_var.set("")
            self.model_info_var.set("Using built-in Cellpose-SAM (cpsam) model")
            # Update config to use default
            if self.yaml_config and "MODEL" in self.yaml_config:
                self.yaml_config["MODEL"]["PRETRAINED_MODEL"] = None

    def _browse_custom_model(self):
        """Browse for a custom Cellpose or BioImage.io model file."""
        if self.model_source_var.get() == "BioImage.io model...":
            path = filedialog.askopenfilename(
                title="Select BioImage.io Model (rdf.yaml / .zip)",
                initialdir=str(PROJECT_ROOT / "models"),
                filetypes=[
                    ("BioImage.io", "*.yaml *.yml *.zip"),
                    ("All", "*.*"),
                ],
            )
        else:
            path = filedialog.askopenfilename(
                title="Select Cellpose Model",
                initialdir=str(PROJECT_ROOT / "models"),
            )
        if path:
            self.custom_model_path_var.set(path)
            model_name = Path(path).name
            self.model_info_var.set(f"Custom model: {model_name}")
            # Update config
            if self.yaml_config:
                if "MODEL" not in self.yaml_config:
                    self.yaml_config["MODEL"] = {}
                self.yaml_config["MODEL"]["PRETRAINED_MODEL"] = path
            logger.info(f"Custom model selected: {path}")

    def _sync_model_selector_from_config(self):
        """Update model selector widgets to reflect the loaded config."""
        if not self.yaml_config:
            return
        model_cfg = self.yaml_config.get("MODEL", {})
        pretrained = model_cfg.get("PRETRAINED_MODEL")
        if pretrained:
            pretrained_str = str(pretrained)
            is_bioimage = (
                pretrained_str.startswith("bioimage.io:")
                or pretrained_str.endswith((".yaml", ".yml", ".zip"))
            )
            if is_bioimage:
                self.model_source_var.set("BioImage.io model...")
                self.model_info_var.set(f"BioImage.io model: {pretrained_str}")
            else:
                self.model_source_var.set("Custom model...")
                self.model_info_var.set(f"Custom model: {Path(pretrained_str).name}")
            self.custom_model_path_var.set(pretrained_str)
            self.custom_model_entry.config(state=tk.NORMAL)
            self.custom_model_btn.config(state=tk.NORMAL)
        else:
            self.model_source_var.set("cpsam (default)")
            self.custom_model_path_var.set("")
            self.custom_model_entry.config(state=tk.DISABLED)
            self.custom_model_btn.config(state=tk.DISABLED)
            self.model_info_var.set("Using built-in Cellpose-SAM (cpsam) model")

    def _populate_param_editor(self):
        """Build parameter widgets from the loaded config."""
        # Clear old widgets
        for w in self.param_inner.winfo_children():
            w.destroy()
        self.param_vars.clear()

        row = 0
        row = self._add_section("TRAIN", row, [
            ("EPOCHS", "Epochs", "int"),
            ("LEARNING_RATE", "Learning Rate", "float"),
            ("WEIGHT_DECAY", "Weight Decay", "float"),
            ("BATCH_SIZE", "Batch Size", "int"),
            ("SAVE_EVERY", "Save Every N Epochs", "int"),
            ("MIN_TRAIN_MASKS", "Min Train Masks", "int"),
        ])

        row = self._add_section("DATA", row, [
            ("CHANNELS", "Channels [segment, nuclear]  (0=DIC, 1=mEGFP, 2=mScarlet, 3=miRFPnano3)", "str"),
            ("Z_SLICE", "Z-Slice (null=first, or 0-indexed integer)", "str"),
        ])

        row = self._add_section("INFERENCE", row, [
            ("DIAMETER", "Diameter (null=auto)", "str"),
            ("FLOW_THRESHOLD", "Flow Threshold", "float"),
            ("CELLPROB_THRESHOLD", "Cell Prob Threshold", "float"),
        ])

        row = self._add_section("AUGMENTATION", row, [
            ("ENABLE", "Enable Augmentation", "bool"),
            ("RANDOM_FLIP", "Random Flip", "bool"),
        ])

        row = self._add_section("PATHS", row, [
            ("MODEL_DIR", "Model Directory", "str"),
            ("RESULT_DIR", "Results Directory", "str"),
            ("MODEL_NAME", "Model Name", "str"),
        ])

    def _add_section(
        self, section: str, start_row: int, fields: list[tuple[str, str, str]]
    ) -> int:
        """Add a section header and its parameter fields."""
        cfg_section = self.yaml_config.get(section, {})

        ttk.Label(
            self.param_inner,
            text=f"--- {section} ---",
            font=("TkDefaultFont", 10, "bold"),
        ).grid(row=start_row, column=0, columnspan=2, sticky=tk.W, pady=(10, 2))
        row = start_row + 1

        for key, label, dtype in fields:
            val = cfg_section.get(key, "")
            ttk.Label(self.param_inner, text=f"  {label}:").grid(
                row=row, column=0, sticky=tk.W, padx=(10, 5), pady=1
            )

            full_key = f"{section}.{key}"

            if dtype == "bool":
                var = tk.BooleanVar(value=bool(val))
                ttk.Checkbutton(self.param_inner, variable=var).grid(
                    row=row, column=1, sticky=tk.W, pady=1
                )
            elif dtype == "int":
                var = tk.StringVar(value=str(val))
                ttk.Entry(self.param_inner, textvariable=var, width=20).grid(
                    row=row, column=1, sticky=tk.W, pady=1
                )
            elif dtype == "float":
                var = tk.StringVar(value=str(val))
                ttk.Entry(self.param_inner, textvariable=var, width=20).grid(
                    row=row, column=1, sticky=tk.W, pady=1
                )
            else:
                var = tk.StringVar(value=str(val))
                ttk.Entry(self.param_inner, textvariable=var, width=40).grid(
                    row=row, column=1, sticky=tk.W, pady=1
                )

            self.param_vars[full_key] = var
            row += 1

        return row

    def _read_params_into_config(self):
        """Read GUI parameter values back into self.yaml_config."""
        # Sync model selection
        if self.yaml_config:
            if "MODEL" not in self.yaml_config:
                self.yaml_config["MODEL"] = {}
            if self.model_source_var.get() in ("Custom model...", "BioImage.io model..."):
                custom_path = self.custom_model_path_var.get()
                self.yaml_config["MODEL"]["PRETRAINED_MODEL"] = custom_path if custom_path else None
            else:
                self.yaml_config["MODEL"]["PRETRAINED_MODEL"] = None

        type_map = {
            "TRAIN.EPOCHS": int,
            "TRAIN.BATCH_SIZE": int,
            "TRAIN.SAVE_EVERY": int,
            "TRAIN.MIN_TRAIN_MASKS": int,
            "TRAIN.LEARNING_RATE": float,
            "TRAIN.WEIGHT_DECAY": float,
            "INFERENCE.FLOW_THRESHOLD": float,
            "INFERENCE.CELLPROB_THRESHOLD": float,
        }

        for full_key, var in self.param_vars.items():
            section, key = full_key.split(".", 1)
            raw = var.get()

            # Parse value
            if full_key in type_map:
                try:
                    val = type_map[full_key](raw)
                except (ValueError, TypeError):
                    val = raw
            elif isinstance(var, tk.BooleanVar):
                val = var.get()
            elif raw.lower() in ("null", "none", ""):
                val = None
            elif raw.startswith("["):
                # Parse list like [0, 0]
                try:
                    val = yaml.safe_load(raw)
                except yaml.YAMLError:
                    val = raw
            else:
                val = raw

            if section not in self.yaml_config:
                self.yaml_config[section] = {}
            self.yaml_config[section][key] = val

    def _save_config(self):
        if not self.yaml_config:
            messagebox.showwarning("No Config", "Load a config first.")
            return

        self._read_params_into_config()

        path = self.yaml_config_path or filedialog.asksaveasfilename(
            title="Save Configuration",
            filetypes=[("YAML", "*.yaml")],
            initialdir=str(PROJECT_ROOT / "configs"),
            defaultextension=".yaml",
        )
        if not path:
            return

        try:
            with open(path, "w") as f:
                yaml.dump(self.yaml_config, f, default_flow_style=False, sort_keys=False)
            self.yaml_config_path = path
            self.cfg_path_var.set(path)
            logger.info(f"Config saved to {path}")
            messagebox.showinfo("Saved", f"Configuration saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save config:\n{e}")

    # ==================================================================
    # TAB 3: TRAINING
    # ==================================================================
    def _build_train_tab(self):
        tab = self.tab_train

        # Controls
        ctrl_frame = ttk.Frame(tab)
        ctrl_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(ctrl_frame, text="Task:").pack(side=tk.LEFT)
        self.train_task = tk.StringVar(value="dic")
        task_combo = ttk.Combobox(
            ctrl_frame,
            textvariable=self.train_task,
            values=["dic", "fluor", "both", "custom"],
            width=12,
            state="readonly",
        )
        task_combo.pack(side=tk.LEFT, padx=5)

        self.btn_train = ttk.Button(
            ctrl_frame, text="Start Training", command=self._start_training
        )
        self.btn_train.pack(side=tk.LEFT, padx=10)

        self.btn_stop = ttk.Button(
            ctrl_frame, text="Stop", command=self._stop_training, state=tk.DISABLED
        )
        self.btn_stop.pack(side=tk.LEFT, padx=5)

        # Progress
        self.train_progress = ttk.Progressbar(tab, mode="indeterminate")
        self.train_progress.pack(fill=tk.X, padx=10, pady=5)

        self.train_status = tk.StringVar(value="Ready")
        ttk.Label(tab, textvariable=self.train_status).pack(padx=10, anchor=tk.W)

        # Log output
        log_frame = ttk.LabelFrame(tab, text="Training Log", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))

        self.train_log = scrolledtext.ScrolledText(
            log_frame, wrap=tk.WORD, height=20, state=tk.DISABLED, font=("Courier", 9)
        )
        self.train_log.pack(fill=tk.BOTH, expand=True)

        self._training_thread = None
        self._stop_event = threading.Event()

    def _start_training(self):
        task = self.train_task.get()

        if task == "custom" and not self.yaml_config:
            messagebox.showwarning(
                "No Config", "Load a config in the Configuration tab first."
            )
            return

        self.btn_train.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.train_progress.start(10)
        self._stop_event.clear()
        self.train_status.set("Training...")

        # Clear log
        self.train_log.config(state=tk.NORMAL)
        self.train_log.delete("1.0", tk.END)
        self.train_log.config(state=tk.DISABLED)

        self._training_thread = threading.Thread(
            target=self._training_worker, args=(task,), daemon=True
        )
        self._training_thread.start()

    def _training_worker(self, task: str):
        try:
            from train_cellpose import load_config, train_cellpose_model

            configs_to_run = []
            if task == "dic":
                configs_to_run.append(str(PROJECT_ROOT / "configs" / "dic_wholecell.yaml"))
            elif task == "fluor":
                configs_to_run.append(str(PROJECT_ROOT / "configs" / "fluor_nucleus.yaml"))
            elif task == "both":
                configs_to_run.append(str(PROJECT_ROOT / "configs" / "dic_wholecell.yaml"))
                configs_to_run.append(str(PROJECT_ROOT / "configs" / "fluor_nucleus.yaml"))
            elif task == "custom":
                # Use config from GUI editor
                self._read_params_into_config()
                configs_to_run.append(self.yaml_config_path)

            for cfg_path in configs_to_run:
                if self._stop_event.is_set():
                    logger.info("Training stopped by user.")
                    break
                logger.info(f"Loading config: {cfg_path}")
                cfg = load_config(cfg_path)
                train_cellpose_model(cfg)

            self.log_queue.put("__TRAINING_DONE__")
        except Exception as e:
            logger.exception("Training failed")
            self.log_queue.put(f"__TRAINING_ERROR__{e}")

    def _stop_training(self):
        self._stop_event.set()
        self.train_status.set("Stopping...")

    def _on_training_finished(self, error: str | None = None):
        self.train_progress.stop()
        self.btn_train.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        if error:
            self.train_status.set(f"Error: {error}")
            messagebox.showerror("Training Error", str(error))
        else:
            self.train_status.set("Training complete")
            messagebox.showinfo("Done", "Training finished successfully.")

    # ==================================================================
    # TAB 4: EVALUATION
    # ==================================================================
    def _build_eval_tab(self):
        tab = self.tab_eval

        ctrl_frame = ttk.LabelFrame(tab, text="Evaluation Settings", padding=10)
        ctrl_frame.pack(fill=tk.X, padx=10, pady=10)

        # Model path
        ttk.Label(ctrl_frame, text="Trained Model:").grid(
            row=0, column=0, sticky=tk.W, pady=2
        )
        self.eval_model_path = tk.StringVar()
        ttk.Entry(ctrl_frame, textvariable=self.eval_model_path, width=50).grid(
            row=0, column=1, padx=5, pady=2
        )
        ttk.Button(
            ctrl_frame, text="Browse...", command=self._browse_eval_model
        ).grid(row=0, column=2, pady=2)

        # Image dir for inference
        ttk.Label(ctrl_frame, text="Image Directory:").grid(
            row=1, column=0, sticky=tk.W, pady=2
        )
        self.eval_img_dir = tk.StringVar()
        ttk.Entry(ctrl_frame, textvariable=self.eval_img_dir, width=50).grid(
            row=1, column=1, padx=5, pady=2
        )
        ttk.Button(
            ctrl_frame, text="Browse...", command=self._browse_eval_img_dir
        ).grid(row=1, column=2, pady=2)

        # Output dir
        ttk.Label(ctrl_frame, text="Output Directory:").grid(
            row=2, column=0, sticky=tk.W, pady=2
        )
        self.eval_output_dir = tk.StringVar(
            value=str(PROJECT_ROOT / "results" / "inference")
        )
        ttk.Entry(ctrl_frame, textvariable=self.eval_output_dir, width=50).grid(
            row=2, column=1, padx=5, pady=2
        )
        ttk.Button(
            ctrl_frame, text="Browse...", command=self._browse_eval_out_dir
        ).grid(row=2, column=2, pady=2)

        # Buttons
        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)

        self.btn_eval = ttk.Button(
            btn_frame, text="Run Inference", command=self._start_eval
        )
        self.btn_eval.pack(side=tk.LEFT, padx=5)

        self.btn_eval_full = ttk.Button(
            btn_frame,
            text="Full Evaluation (with metrics)",
            command=self._start_full_eval,
        )
        self.btn_eval_full.pack(side=tk.LEFT, padx=5)

        self.eval_progress = ttk.Progressbar(tab, mode="indeterminate")
        self.eval_progress.pack(fill=tk.X, padx=10, pady=5)

        self.eval_status = tk.StringVar(value="Ready")
        ttk.Label(tab, textvariable=self.eval_status).pack(padx=10, anchor=tk.W)

        # Results display
        results_frame = ttk.LabelFrame(tab, text="Results", padding=5)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))

        self.eval_results = scrolledtext.ScrolledText(
            results_frame, wrap=tk.WORD, height=15, state=tk.DISABLED, font=("Courier", 9)
        )
        self.eval_results.pack(fill=tk.BOTH, expand=True)

    def _browse_eval_model(self):
        path = filedialog.askopenfilename(
            title="Select Trained Model",
            initialdir=str(PROJECT_ROOT / "models"),
        )
        if path:
            self.eval_model_path.set(path)

    def _browse_eval_img_dir(self):
        d = filedialog.askdirectory(title="Select Image Directory")
        if d:
            self.eval_img_dir.set(d)

    def _browse_eval_out_dir(self):
        d = filedialog.askdirectory(title="Select Output Directory")
        if d:
            self.eval_output_dir.set(d)

    def _start_eval(self):
        model = self.eval_model_path.get()
        img_dir = self.eval_img_dir.get()
        output = self.eval_output_dir.get()

        if not model or not img_dir:
            messagebox.showwarning("Missing", "Select a model and image directory.")
            return

        self.btn_eval.config(state=tk.DISABLED)
        self.eval_progress.start(10)
        self.eval_status.set("Running inference...")

        def worker():
            try:
                from evaluate import run_inference
                masks, names = run_inference(
                    model_path=model,
                    image_dir=img_dir,
                    output_dir=output,
                )
                self.log_queue.put(f"__EVAL_DONE__Inference complete: {len(masks)} images processed")
            except Exception as e:
                logger.exception("Inference failed")
                self.log_queue.put(f"__EVAL_ERROR__{e}")

        threading.Thread(target=worker, daemon=True).start()

    def _start_full_eval(self):
        """Run full evaluation using the loaded config."""
        model = self.eval_model_path.get()
        if not model:
            messagebox.showwarning("Missing", "Select a trained model.")
            return
        if not self.yaml_config:
            messagebox.showwarning(
                "No Config", "Load a config in the Configuration tab first."
            )
            return

        self.btn_eval_full.config(state=tk.DISABLED)
        self.eval_progress.start(10)
        self.eval_status.set("Running full evaluation...")

        def worker():
            try:
                from evaluate import evaluate_model
                self._read_params_into_config()
                results = evaluate_model(self.yaml_config, model)
                msg = "Full Evaluation Results:\n\n"
                for k, v in results.items():
                    if not isinstance(v, (list, dict)):
                        msg += f"  {k}: {v}\n"
                self.log_queue.put(f"__EVAL_DONE__{msg}")
            except Exception as e:
                logger.exception("Evaluation failed")
                self.log_queue.put(f"__EVAL_ERROR__{e}")

        threading.Thread(target=worker, daemon=True).start()

    def _on_eval_finished(self, message: str | None = None, error: str | None = None):
        self.eval_progress.stop()
        self.btn_eval.config(state=tk.NORMAL)
        self.btn_eval_full.config(state=tk.NORMAL)

        if error:
            self.eval_status.set(f"Error: {error}")
            messagebox.showerror("Evaluation Error", str(error))
        else:
            self.eval_status.set("Evaluation complete")
            if message:
                self.eval_results.config(state=tk.NORMAL)
                self.eval_results.delete("1.0", tk.END)
                self.eval_results.insert(tk.END, message)
                self.eval_results.config(state=tk.DISABLED)

    # ==================================================================
    # LOGGING
    # ==================================================================
    def _setup_logging(self):
        handler = QueueHandler(self.log_queue)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(handler)

    def _poll_log_queue(self):
        """Periodically drain the log queue and update text widgets."""
        while True:
            try:
                msg = self.log_queue.get_nowait()
            except queue.Empty:
                break

            # Check for control messages
            if msg == "__TRAINING_DONE__":
                self._on_training_finished()
                continue
            if msg.startswith("__TRAINING_ERROR__"):
                self._on_training_finished(error=msg[len("__TRAINING_ERROR__"):])
                continue
            if msg.startswith("__EVAL_DONE__"):
                self._on_eval_finished(message=msg[len("__EVAL_DONE__"):])
                continue
            if msg.startswith("__EVAL_ERROR__"):
                self._on_eval_finished(error=msg[len("__EVAL_ERROR__"):])
                continue
            if msg.startswith("__PP_PROGRESS__"):
                payload = msg[len("__PP_PROGRESS__"):]
                pct_str, status_text = payload.split("||", 1)
                self.pp_progress.config(value=int(pct_str))
                self.pp_status.set(status_text)
                continue
            if msg.startswith("__PP_DONE__"):
                self._on_pp_finished(results_str=msg[len("__PP_DONE__"):])
                continue
            if msg.startswith("__PP_ERROR__"):
                self._on_pp_finished(error=msg[len("__PP_ERROR__"):])
                continue

            # Append to training log
            self.train_log.config(state=tk.NORMAL)
            self.train_log.insert(tk.END, msg + "\n")
            self.train_log.see(tk.END)
            self.train_log.config(state=tk.DISABLED)

        self.after(100, self._poll_log_queue)

    # ==================================================================
    # MISC
    # ==================================================================
    def _show_about(self):
        messagebox.showinfo(
            "About",
            "Cellpose Segmentation Pipeline GUI\n\n"
            "BiaPy-inspired workflow for training Cellpose models.\n\n"
            "Tasks:\n"
            "  - DIC Brightfield Whole-Cell Segmentation\n"
            "  - Fluorescence Nucleus Segmentation\n\n"
            "Built with Cellpose 4 (cpsam) + tkinter",
        )


def main():
    app = SegmentationGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
