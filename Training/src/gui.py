#!/usr/bin/env python3
"""
Unified GUI for the Puncta-CSAT Segmentation Pipeline.

Combines Training workflow and Analysis workflow in a single interface.

Provides tabs for:
  -- Analysis Workflow --
  1. ND2 Conversion   — Convert ND2 files to OME-TIFF
  2. Segmentation     — Run Cellpose (nucleus / puncta / cell / cytoplasm)
                        with model selection, channel & normalization controls
  3. Analysis         — Per-cell intensity & puncta analysis

  -- Training Workflow --
  4. File Renaming    — Rename images/masks to pipeline convention
  5. Configuration    — Edit training parameters from YAML configs
  6. Training         — Launch and monitor model training
  7. Evaluation       — Run inference and view results
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

# ------------------------------------------------------------------ #
#  Locate Nucleus/Scripts directory for analysis modules
# ------------------------------------------------------------------ #
def _find_nucleus_scripts_dir():
    """Return the path to Nucleus/Scripts, searching common locations."""
    candidates = [
        PROJECT_ROOT.parent / "Nucleus" / "Scripts",       # Segmentation/Nucleus/Scripts
        SRC_DIR.parent.parent / "Nucleus" / "Scripts",     # alternate
        Path(__file__).resolve().parent.parent.parent / "Nucleus" / "Scripts",
    ]
    for p in candidates:
        if (p / "segmentation_utils.py").exists():
            return p.resolve()
    return None

NUCLEUS_SCRIPTS_DIR = _find_nucleus_scripts_dir()
if NUCLEUS_SCRIPTS_DIR is not None:
    sys.path.insert(0, str(NUCLEUS_SCRIPTS_DIR))

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
        self.title("Puncta-CSAT Segmentation Pipeline")
        self.geometry("980x800")
        self.minsize(800, 650)

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

        # -- Analysis workflow tabs --
        self.tab_nd2 = ttk.Frame(self.notebook)
        self.tab_segmentation = ttk.Frame(self.notebook)
        self.tab_puncta_seg = ttk.Frame(self.notebook)
        self.tab_analysis = ttk.Frame(self.notebook)

        # -- Training workflow tabs --
        self.tab_rename = ttk.Frame(self.notebook)
        self.tab_config = ttk.Frame(self.notebook)
        self.tab_train = ttk.Frame(self.notebook)
        self.tab_eval = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_nd2, text="  ND2 Conversion  ")
        self.notebook.add(self.tab_segmentation, text="  Segmentation  ")
        self.notebook.add(self.tab_puncta_seg, text="  Puncta Segmentation  ")
        self.notebook.add(self.tab_analysis, text="  Analysis  ")
        self.notebook.add(self.tab_rename, text="  File Renaming  ")
        self.notebook.add(self.tab_config, text="  Configuration  ")
        self.notebook.add(self.tab_train, text="  Training  ")
        self.notebook.add(self.tab_eval, text="  Evaluation  ")

        self._build_nd2_tab()
        self._build_segmentation_tab()
        self._build_puncta_seg_tab()
        self._build_analysis_tab()
        self._build_rename_tab()
        self._build_config_tab()
        self._build_train_tab()
        self._build_eval_tab()

    # ==================================================================
    # TAB: ND2 CONVERSION
    # ==================================================================
    def _build_nd2_tab(self):
        tab = self.tab_nd2

        info = ttk.Label(
            tab,
            text="Convert ND2 microscopy files to per-position OME-TIFF, "
                 "extracting a single Z-plane.\n"
                 "Split channels creates per-channel folders with single-channel "
                 "TIFFs for Cellpose GUI, training, and analysis.",
            foreground="gray",
        )
        info.pack(anchor=tk.W, padx=10, pady=(10, 5))

        io_frame = ttk.LabelFrame(tab, text="Input / Output", padding=10)
        io_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(io_frame, text="ND2 File:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.nd2_file_var = tk.StringVar()
        ttk.Entry(io_frame, textvariable=self.nd2_file_var, width=55).grid(
            row=0, column=1, padx=5, pady=2
        )
        ttk.Button(io_frame, text="Browse...", command=self._nd2_browse_file).grid(
            row=0, column=2, pady=2
        )

        ttk.Label(io_frame, text="Output Directory:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.nd2_out_dir = tk.StringVar()
        ttk.Entry(io_frame, textvariable=self.nd2_out_dir, width=55).grid(
            row=1, column=1, padx=5, pady=2
        )
        ttk.Button(io_frame, text="Browse...", command=self._nd2_browse_out).grid(
            row=1, column=2, pady=2
        )

        param_frame = ttk.LabelFrame(tab, text="Parameters", padding=10)
        param_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(param_frame, text="Z-plane index:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.nd2_z_index = tk.IntVar(value=8)
        ttk.Entry(param_frame, textvariable=self.nd2_z_index, width=8).grid(
            row=0, column=1, padx=5, pady=2, sticky=tk.W
        )
        ttk.Label(param_frame, text="(0-based index of Z-plane to extract)", foreground="gray").grid(
            row=0, column=2, sticky=tk.W, padx=5
        )

        self.nd2_split_channels = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            param_frame, text="Split channels into separate folders",
            variable=self.nd2_split_channels,
        ).grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=2)
        ttk.Label(
            param_frame,
            text="Creates DIC/, mEGFP/, mScarlet/, etc. folders with single-channel TIFFs\n"
                 "compatible with Cellpose GUI for training, curation, and evaluation.",
            foreground="gray",
        ).grid(row=2, column=0, columnspan=3, sticky=tk.W, padx=20, pady=(0, 2))

        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        self.btn_nd2_run = ttk.Button(btn_frame, text="Convert", command=self._nd2_run)
        self.btn_nd2_run.pack(side=tk.LEFT, padx=5)

        self.nd2_status = tk.StringVar(value="Ready")
        ttk.Label(tab, textvariable=self.nd2_status).pack(padx=10, anchor=tk.W)

        # Log for ND2 tab
        log_frame = ttk.LabelFrame(tab, text="Log", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))
        self.nd2_log = scrolledtext.ScrolledText(
            log_frame, wrap=tk.WORD, height=12, state=tk.DISABLED, font=("Courier", 9)
        )
        self.nd2_log.pack(fill=tk.BOTH, expand=True)

    def _nd2_browse_file(self):
        path = filedialog.askopenfilename(
            title="Select ND2 File",
            filetypes=[("ND2 files", "*.nd2"), ("All files", "*.*")],
        )
        if path:
            self.nd2_file_var.set(path)

    def _nd2_browse_out(self):
        d = filedialog.askdirectory(title="Select Output Directory")
        if d:
            self.nd2_out_dir.set(d)

    def _nd2_run(self):
        nd2_path = self.nd2_file_var.get()
        out_dir = self.nd2_out_dir.get()
        if not nd2_path or not out_dir:
            messagebox.showwarning("Missing input", "Select both the ND2 file and output directory.")
            return
        z = self.nd2_z_index.get()
        split_ch = self.nd2_split_channels.get()
        self.btn_nd2_run.config(state=tk.DISABLED)
        mode_str = "OME-TIFF + split channels" if split_ch else "OME-TIFF"
        self.nd2_status.set(f"Converting (z={z}, {mode_str})...")
        self._nd2_log_append(f"Converting ND2 -> {mode_str}  (z={z})")

        def task():
            try:
                from preprocessing.nd2_to_ome_tif import convert_nd2
                convert_nd2(nd2_path, out_dir, z_index=z, split_channels=split_ch)
                self.log_queue.put("__ND2_DONE__")
            except ImportError as exc:
                self.log_queue.put(f"__ND2_ERROR__Missing dependency: {exc}\n  pip install nd2 tifffile numpy")
            except Exception as exc:
                import traceback
                self.log_queue.put(f"__ND2_ERROR__{exc}\n{traceback.format_exc()}")

        threading.Thread(target=task, daemon=True).start()

    def _nd2_log_append(self, msg):
        self.nd2_log.config(state=tk.NORMAL)
        self.nd2_log.insert(tk.END, msg + "\n")
        self.nd2_log.see(tk.END)
        self.nd2_log.config(state=tk.DISABLED)

    def _on_nd2_finished(self, error=None):
        self.btn_nd2_run.config(state=tk.NORMAL)
        if error:
            self.nd2_status.set(f"Error: {error[:80]}")
            self._nd2_log_append(f"[ERROR] {error}")
        else:
            self.nd2_status.set("Conversion complete")
            self._nd2_log_append("[DONE] ND2 conversion complete.")

    # ==================================================================
    # TAB: SEGMENTATION (unified: nucleus / puncta / cell / cytoplasm)
    #   Merges old Segmentation + Preprocessing into one tab with:
    #   - Mode presets (cell/nucleus/puncta/cytoplasm)
    #   - Model selector (cpsam / custom / BioImage.io)
    #   - Channel & normalization settings
    #   - Cellpose parameters (diameter, flow, cellprob)
    #   - Cytoplasm subtraction options
    #   - Progress, results table, log
    # ==================================================================
    def _build_segmentation_tab(self):
        tab = self.tab_segmentation

        # Use a canvas + scrollbar so the tab is scrollable
        canvas = tk.Canvas(tab, highlightthickness=0)
        sb = ttk.Scrollbar(tab, orient=tk.VERTICAL, command=canvas.yview)
        self.seg_body = ttk.Frame(canvas)
        self.seg_body.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.seg_body, anchor=tk.NW)
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        # Mouse-wheel scrolling
        def _on_scroll(event):
            if event.num == 4:
                canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                canvas.yview_scroll(1, "units")
            else:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_scroll)
        canvas.bind_all("<Button-4>", _on_scroll)
        canvas.bind_all("<Button-5>", _on_scroll)

        body = self.seg_body

        ch_info = ttk.Label(
            body,
            text="Channels:  0 = DIC (bright-field)    1 = GFP/mEGFP (puncta)    "
                 "2 = mScarlet (nucleus)    3 = miRFPnano3",
            foreground="gray",
        )
        ch_info.pack(anchor=tk.W, padx=10, pady=(10, 5))

        # ---- Input / Output ----
        io_frame = ttk.LabelFrame(body, text="Input / Output", padding=10)
        io_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(io_frame, text="Image Directory:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.seg_input_dir = tk.StringVar()
        ttk.Entry(io_frame, textvariable=self.seg_input_dir, width=55).grid(
            row=0, column=1, padx=5, pady=2
        )
        ttk.Button(io_frame, text="Browse...", command=lambda: self._browse_dir(self.seg_input_dir)).grid(
            row=0, column=2, pady=2
        )

        ttk.Label(io_frame, text="Output Masks Directory:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.seg_out_dir = tk.StringVar()
        ttk.Entry(io_frame, textvariable=self.seg_out_dir, width=55).grid(
            row=1, column=1, padx=5, pady=2
        )
        ttk.Button(io_frame, text="Browse...", command=lambda: self._browse_dir(self.seg_out_dir)).grid(
            row=1, column=2, pady=2
        )

        # ---- Mode selector ----
        mode_frame = ttk.LabelFrame(body, text="Segmentation Mode", padding=10)
        mode_frame.pack(fill=tk.X, padx=10, pady=5)

        self.seg_mode_var = tk.StringVar(value="cell")
        modes_row = ttk.Frame(mode_frame)
        modes_row.pack(fill=tk.X)
        for label, val in [("Cell (DIC)", "cell"), ("Nucleus", "nucleus"),
                           ("Puncta", "puncta"), ("Cytoplasm", "cytoplasm")]:
            ttk.Radiobutton(modes_row, text=label, variable=self.seg_mode_var,
                            value=val, command=self._seg_on_mode_change).pack(side=tk.LEFT, padx=8)

        # ---- Channel & Normalization ----
        norm_frame = ttk.LabelFrame(body, text="Channel & Normalization", padding=10)
        norm_frame.pack(fill=tk.X, padx=10, pady=5)

        # Segment channel with label
        ttk.Label(norm_frame, text="Segment Channel:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.seg_channel = tk.IntVar(value=0)
        seg_combo = ttk.Combobox(
            norm_frame, textvariable=self.seg_channel, values=[0, 1, 2, 3],
            width=5, state="readonly",
        )
        seg_combo.grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        self.seg_ch_label = tk.StringVar(value="DIC")
        ttk.Label(norm_frame, textvariable=self.seg_ch_label, foreground="gray").grid(
            row=0, column=2, sticky=tk.W, padx=5
        )
        seg_combo.bind("<<ComboboxSelected>>", self._seg_update_channel_labels)

        # Nuclear channel (for dual-channel input)
        ttk.Label(norm_frame, text="Nuclear Channel:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.seg_nuc_channel = tk.IntVar(value=0)
        nuc_combo = ttk.Combobox(
            norm_frame, textvariable=self.seg_nuc_channel, values=[0, 1, 2, 3],
            width=5, state="readonly",
        )
        nuc_combo.grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)
        self.seg_nuc_label = tk.StringVar(value="None (grayscale)")
        ttk.Label(norm_frame, textvariable=self.seg_nuc_label, foreground="gray").grid(
            row=1, column=2, sticky=tk.W, padx=5
        )
        nuc_combo.bind("<<ComboboxSelected>>", self._seg_update_channel_labels)

        # Percentile range
        ttk.Label(norm_frame, text="Normalization Percentile:").grid(row=2, column=0, sticky=tk.W, pady=2)
        pct_frame = ttk.Frame(norm_frame)
        pct_frame.grid(row=2, column=1, columnspan=2, sticky=tk.W, padx=5)
        ttk.Label(pct_frame, text="Low:").pack(side=tk.LEFT)
        self.seg_lower_pct = tk.DoubleVar(value=1.0)
        ttk.Entry(pct_frame, textvariable=self.seg_lower_pct, width=6).pack(side=tk.LEFT, padx=(2, 10))
        ttk.Label(pct_frame, text="High:").pack(side=tk.LEFT)
        self.seg_upper_pct = tk.DoubleVar(value=99.0)
        ttk.Entry(pct_frame, textvariable=self.seg_upper_pct, width=6).pack(side=tk.LEFT, padx=2)

        # DIC tile blocksize
        ttk.Label(norm_frame, text="DIC Tile Blocksize:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.seg_tile_bs = tk.IntVar(value=128)
        ttk.Entry(norm_frame, textvariable=self.seg_tile_bs, width=8).grid(
            row=3, column=1, padx=5, pady=2, sticky=tk.W
        )
        ttk.Label(norm_frame, text="(0 = global, >0 = tile-based for uneven illumination)",
                  foreground="gray").grid(row=3, column=2, sticky=tk.W, padx=5)

        # Z-slice
        ttk.Label(norm_frame, text="Z-Slice:").grid(row=4, column=0, sticky=tk.W, pady=2)
        z_frame = ttk.Frame(norm_frame)
        z_frame.grid(row=4, column=1, columnspan=2, sticky=tk.W, padx=5)
        self.seg_z_idx = tk.StringVar(value="0")
        ttk.Entry(z_frame, textvariable=self.seg_z_idx, width=6).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(z_frame, text="(0-indexed)", foreground="gray").pack(side=tk.LEFT)

        # Invert DIC + DIC normalization checkboxes
        chk_frame = ttk.Frame(norm_frame)
        chk_frame.grid(row=5, column=0, columnspan=3, sticky=tk.W, pady=2)

        self.seg_dic_norm_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(chk_frame, text="DIC normalization (CLAHE)",
                        variable=self.seg_dic_norm_var).pack(side=tk.LEFT, padx=(0, 15))

        self.seg_invert_dic = tk.BooleanVar(value=False)
        ttk.Checkbutton(chk_frame, text="Invert DIC (cells dark on bright background)",
                        variable=self.seg_invert_dic).pack(side=tk.LEFT)

        # CLAHE clip limit
        ttk.Label(norm_frame, text="CLAHE Clip Limit:").grid(row=6, column=0, sticky=tk.W, pady=2)
        self.seg_clahe_clip = tk.DoubleVar(value=0.02)
        clahe_frame = ttk.Frame(norm_frame)
        clahe_frame.grid(row=6, column=1, columnspan=2, sticky=tk.W, padx=5)
        ttk.Scale(clahe_frame, from_=0.005, to=0.10, variable=self.seg_clahe_clip,
                  orient=tk.HORIZONTAL, length=180).pack(side=tk.LEFT)
        ttk.Label(clahe_frame, textvariable=self.seg_clahe_clip, width=6).pack(side=tk.LEFT, padx=5)
        ttk.Label(clahe_frame, text="(higher = more contrast)", foreground="gray").pack(side=tk.LEFT)

        # ---- Model Selection ----
        model_frame = ttk.LabelFrame(body, text="Model", padding=10)
        model_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(model_frame, text="Model:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.seg_model_var = tk.StringVar(value="cyto3 (default)")
        seg_model_combo = ttk.Combobox(
            model_frame, textvariable=self.seg_model_var,
            values=["cyto3 (default)", "cpsam", "Custom model...", "BioImage.io model..."],
            width=22, state="readonly",
        )
        seg_model_combo.grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        seg_model_combo.bind("<<ComboboxSelected>>", self._seg_on_model_changed)

        self.seg_custom_model = tk.StringVar()
        self.seg_model_entry = ttk.Entry(model_frame, textvariable=self.seg_custom_model, width=35)
        self.seg_model_entry.grid(row=0, column=2, padx=5, pady=2)
        self.seg_model_entry.config(state=tk.DISABLED)
        self.seg_model_btn = ttk.Button(model_frame, text="Browse...", command=self._seg_browse_model)
        self.seg_model_btn.grid(row=0, column=3, pady=2)
        self.seg_model_btn.config(state=tk.DISABLED)

        # ---- Cellpose Parameters ----
        cp_frame = ttk.LabelFrame(body, text="Cellpose Parameters", padding=10)
        cp_frame.pack(fill=tk.X, padx=10, pady=5)

        row0 = ttk.Frame(cp_frame)
        row0.pack(fill=tk.X, pady=2)

        ttk.Label(row0, text="Diameter (px):").pack(side=tk.LEFT, padx=(0, 5))
        self.seg_diameter = tk.StringVar(value="auto")
        ttk.Entry(row0, textvariable=self.seg_diameter, width=8).pack(side=tk.LEFT, padx=(0, 15))

        ttk.Label(row0, text="Flow Threshold:").pack(side=tk.LEFT, padx=(0, 5))
        self.seg_flow_thr = tk.DoubleVar(value=0.4)
        ttk.Entry(row0, textvariable=self.seg_flow_thr, width=8).pack(side=tk.LEFT, padx=(0, 15))

        ttk.Label(row0, text="Cell Prob Threshold:").pack(side=tk.LEFT, padx=(0, 5))
        self.seg_cellprob_thr = tk.DoubleVar(value=0.0)
        ttk.Entry(row0, textvariable=self.seg_cellprob_thr, width=8).pack(side=tk.LEFT)

        row1 = ttk.Frame(cp_frame)
        row1.pack(fill=tk.X, pady=2)

        ttk.Label(row1, text="Min object size (px):").pack(side=tk.LEFT, padx=(0, 5))
        self.seg_min_size = tk.IntVar(value=50000)
        ttk.Entry(row1, textvariable=self.seg_min_size, width=10).pack(side=tk.LEFT, padx=(0, 15))

        self.seg_gpu_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row1, text="Use GPU", variable=self.seg_gpu_var).pack(side=tk.LEFT, padx=8)

        self.seg_augment_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row1, text="Test-time augmentation",
                        variable=self.seg_augment_var).pack(side=tk.LEFT, padx=8)

        self.seg_edges_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row1, text="Remove edge objects", variable=self.seg_edges_var).pack(side=tk.LEFT, padx=8)

        ttk.Label(row1, text="Smooth edges (radius):").pack(side=tk.LEFT, padx=(10, 5))
        self.seg_smooth_radius = tk.IntVar(value=3)
        ttk.Spinbox(row1, from_=0, to=10, textvariable=self.seg_smooth_radius, width=4).pack(side=tk.LEFT)

        # Row 2: max size and solidity filters
        row2 = ttk.Frame(cp_frame)
        row2.pack(fill=tk.X, pady=2)

        ttk.Label(row2, text="Max object size (px):").pack(side=tk.LEFT, padx=(0, 5))
        self.seg_max_size = tk.IntVar(value=0)
        ttk.Entry(row2, textvariable=self.seg_max_size, width=10).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(row2, text="(0 = no limit)", foreground="gray").pack(side=tk.LEFT, padx=(0, 15))

        ttk.Label(row2, text="Min solidity:").pack(side=tk.LEFT, padx=(0, 5))
        self.seg_min_solidity = tk.DoubleVar(value=0.0)
        ttk.Entry(row2, textvariable=self.seg_min_solidity, width=6).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(row2, text="(0 = off, 0.8 = filter irregular shapes)",
                  foreground="gray").pack(side=tk.LEFT)

        # ---- Cytoplasm-specific options (shown/hidden) ----
        self.seg_cyto_frame = ttk.LabelFrame(body, text="Cytoplasm Options (cell - nucleus = cytoplasm)", padding=10)

        ttk.Label(self.seg_cyto_frame, text="Nucleus Masks Folder:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.seg_nuc_mask_dir = tk.StringVar()
        ttk.Entry(self.seg_cyto_frame, textvariable=self.seg_nuc_mask_dir, width=45).grid(
            row=0, column=1, padx=5, pady=2
        )
        ttk.Button(self.seg_cyto_frame, text="Browse...",
                    command=lambda: self._browse_dir(self.seg_nuc_mask_dir)).grid(row=0, column=2, pady=2)

        cyto_params = ttk.Frame(self.seg_cyto_frame)
        cyto_params.grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=5)

        ttk.Label(cyto_params, text="Nuc dilation (px):").pack(side=tk.LEFT, padx=(0, 5))
        self.seg_nuc_dilate = tk.IntVar(value=0)
        ttk.Entry(cyto_params, textvariable=self.seg_nuc_dilate, width=6).pack(side=tk.LEFT, padx=(0, 15))

        ttk.Label(cyto_params, text="Min nuc overlap (px):").pack(side=tk.LEFT, padx=(0, 5))
        self.seg_min_nuc_px = tk.IntVar(value=10)
        ttk.Entry(cyto_params, textvariable=self.seg_min_nuc_px, width=6).pack(side=tk.LEFT, padx=(0, 15))

        ttk.Label(cyto_params, text="Min overlap frac:").pack(side=tk.LEFT, padx=(0, 5))
        self.seg_min_overlap_frac = tk.DoubleVar(value=0.005)
        ttk.Entry(cyto_params, textvariable=self.seg_min_overlap_frac, width=8).pack(side=tk.LEFT)

        # ---- Buttons ----
        btn_frame = ttk.Frame(body)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(btn_frame, text="Preview Channels", command=self._seg_preview_channels).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(btn_frame, text="Expand ND2 Positions", command=self._seg_expand_nd2).pack(
            side=tk.LEFT, padx=5
        )
        self.btn_seg_run = ttk.Button(btn_frame, text="Run Segmentation", command=self._seg_run)
        self.btn_seg_run.pack(side=tk.LEFT, padx=5)

        # Progress
        self.seg_progress = ttk.Progressbar(body, mode="determinate")
        self.seg_progress.pack(fill=tk.X, padx=10, pady=5)

        self.seg_status = tk.StringVar(value="Ready")
        ttk.Label(body, textvariable=self.seg_status).pack(padx=10, anchor=tk.W)

        # Results table
        result_frame = ttk.LabelFrame(body, text="Results", padding=5)
        result_frame.pack(fill=tk.X, padx=10, pady=(5, 5))

        columns = ("filename", "objects", "status")
        self.seg_tree = ttk.Treeview(
            result_frame, columns=columns, show="headings", height=6
        )
        self.seg_tree.heading("filename", text="Filename")
        self.seg_tree.heading("objects", text="Objects Found")
        self.seg_tree.heading("status", text="Status")
        self.seg_tree.column("filename", width=350)
        self.seg_tree.column("objects", width=120, anchor=tk.CENTER)
        self.seg_tree.column("status", width=120, anchor=tk.CENTER)

        seg_tree_scroll = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.seg_tree.yview)
        self.seg_tree.configure(yscrollcommand=seg_tree_scroll.set)
        self.seg_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        seg_tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Log
        log_frame = ttk.LabelFrame(body, text="Log", padding=5)
        log_frame.pack(fill=tk.X, padx=10, pady=(5, 10))
        self.seg_log = scrolledtext.ScrolledText(
            log_frame, wrap=tk.WORD, height=8, state=tk.DISABLED, font=("Courier", 9)
        )
        self.seg_log.pack(fill=tk.BOTH, expand=True)

        # Apply initial mode presets
        self._seg_on_mode_change()

    _CHANNEL_NAMES = {0: "DIC", 1: "mEGFP", 2: "mScarlet", 3: "miRFPnano3"}

    def _seg_update_channel_labels(self, event=None):
        seg = self.seg_channel.get()
        nuc = self.seg_nuc_channel.get()
        self.seg_ch_label.set(self._CHANNEL_NAMES.get(seg, f"Channel {seg}"))
        if nuc == 0:
            self.seg_nuc_label.set("None (grayscale)")
        else:
            self.seg_nuc_label.set(self._CHANNEL_NAMES.get(nuc, f"Channel {nuc}"))

    def _seg_on_mode_change(self):
        mode = self.seg_mode_var.get()
        if mode == "cell":
            self.seg_diameter.set("auto")
            self.seg_channel.set(0)
            self.seg_nuc_channel.set(0)
            self.seg_min_size.set(50000)
            self.seg_edges_var.set(True)
            self.seg_smooth_radius.set(3)
            self.seg_dic_norm_var.set(True)
            self.seg_cyto_frame.pack_forget()
        elif mode == "nucleus":
            self.seg_diameter.set("200")
            self.seg_channel.set(2)
            self.seg_nuc_channel.set(0)
            self.seg_min_size.set(10000)
            self.seg_edges_var.set(True)
            self.seg_smooth_radius.set(3)
            self.seg_dic_norm_var.set(False)
            self.seg_cyto_frame.pack_forget()
        elif mode == "puncta":
            self.seg_diameter.set("20")
            self.seg_channel.set(1)
            self.seg_nuc_channel.set(0)
            self.seg_min_size.set(0)
            self.seg_edges_var.set(False)
            self.seg_smooth_radius.set(0)
            self.seg_dic_norm_var.set(False)
            self.seg_cyto_frame.pack_forget()
        elif mode == "cytoplasm":
            self.seg_diameter.set("auto")
            self.seg_channel.set(0)
            self.seg_nuc_channel.set(0)
            self.seg_min_size.set(50000)
            self.seg_edges_var.set(True)
            self.seg_smooth_radius.set(3)
            self.seg_dic_norm_var.set(True)
            self.seg_cyto_frame.pack(fill=tk.X, padx=10, pady=5, before=self.btn_seg_run.master)
        self._seg_update_channel_labels()

    def _seg_on_model_changed(self, event=None):
        choice = self.seg_model_var.get()
        if choice in ("Custom model...", "BioImage.io model..."):
            self.seg_model_entry.config(state=tk.NORMAL)
            self.seg_model_btn.config(state=tk.NORMAL)
            if choice == "BioImage.io model...":
                self.seg_custom_model.set("bioimage.io:")
            else:
                self.seg_custom_model.set("")
        else:
            self.seg_model_entry.config(state=tk.DISABLED)
            self.seg_model_btn.config(state=tk.DISABLED)
            self.seg_custom_model.set("")

    def _seg_browse_model(self):
        if self.seg_model_var.get() == "BioImage.io model...":
            path = filedialog.askopenfilename(
                title="Select BioImage.io Model (rdf.yaml / .zip)",
                initialdir=str(PROJECT_ROOT / "models"),
                filetypes=[("BioImage.io", "*.yaml *.yml *.zip"), ("All", "*.*")],
            )
        else:
            path = filedialog.askopenfilename(
                title="Select Cellpose Model",
                initialdir=str(PROJECT_ROOT / "models"),
            )
        if path:
            self.seg_custom_model.set(path)

    def _seg_log_append(self, msg):
        self.seg_log.config(state=tk.NORMAL)
        self.seg_log.insert(tk.END, msg + "\n")
        self.seg_log.see(tk.END)
        self.seg_log.config(state=tk.DISABLED)

    def _seg_log_append_q(self, msg):
        """Thread-safe log append via queue."""
        self.log_queue.put(f"__SEG_LOG__{msg}")

    def _seg_preview_channels(self):
        """Save and open a channel preview for the first image."""
        img_dir = self.seg_input_dir.get()
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
        z_val = self.seg_z_idx.get().strip()
        z_slice = int(z_val) if z_val else None

        try:
            path = save_channel_preview(
                files[0], preview_dir,
                lower_percentile=self.seg_lower_pct.get(),
                upper_percentile=self.seg_upper_pct.get(),
                tile_blocksize_dic=self.seg_tile_bs.get(),
                z_slice=z_slice,
            )
            self.seg_status.set(f"Channel preview saved: {path}")
            messagebox.showinfo(
                "Preview Saved",
                f"Channel preview for {Path(files[0]).name} saved to:\n{path}\n\n"
                "Open this file to inspect channel quality and normalization.",
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate preview:\n{e}")

    def _seg_expand_nd2(self):
        """Expand multi-position .nd2 files in the image directory into TIFFs."""
        img_dir = self.seg_input_dir.get()
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
                    logger.info(f"Expanding {Path(nd2_file).name}: {n_pos} positions")
                saved = expand_nd2_positions(nd2_file, output_dir)
                total_saved.extend(saved)
            except Exception as e:
                logger.error(f"Failed to expand {Path(nd2_file).name}: {e}")
                messagebox.showerror("ND2 Error", f"Failed to expand {Path(nd2_file).name}:\n{e}")

        if total_saved:
            messagebox.showinfo(
                "ND2 Expanded",
                f"Processed {len(nd2_files)} ND2 file(s) ({multi_pos_count} multi-position).\n"
                f"Saved {len(total_saved)} individual TIFFs to:\n{output_dir}",
            )
            self.seg_status.set(f"Expanded {len(nd2_files)} ND2 -> {len(total_saved)} TIFFs")

    @staticmethod
    def _find_nuc_mask(nuc_mask_dir, image_stem):
        """Find nucleus mask matching an image stem."""
        import re
        nuc_dir = Path(nuc_mask_dir)
        for p in nuc_dir.glob("*.tif"):
            if image_stem in p.stem or p.stem in image_stem:
                return p
        m = re.search(r"(\d+_Z\d+)", image_stem)
        if m:
            token = m.group(1)
            for p in nuc_dir.glob("*.tif"):
                if token in p.stem:
                    return p
        return None

    def _seg_get_model_type(self):
        """Return the model type/path string based on the model selector."""
        choice = self.seg_model_var.get()
        if choice == "cpsam":
            return "cpsam"
        if choice in ("Custom model...", "BioImage.io model..."):
            return self.seg_custom_model.get() or "cyto3"
        return "cyto3"

    def _seg_run(self):
        input_path = self.seg_input_dir.get()
        out_dir = self.seg_out_dir.get()
        if not input_path or not out_dir:
            messagebox.showwarning("Missing input", "Select input images and output directory.")
            return
        if NUCLEUS_SCRIPTS_DIR is None:
            messagebox.showerror(
                "Nucleus/Scripts not found",
                "Cannot find Nucleus/Scripts/segmentation_utils.py.\n"
                "Make sure the Nucleus/ folder is in the repository root."
            )
            return

        mode = self.seg_mode_var.get()
        diam_str = self.seg_diameter.get().strip().lower()
        diameter = None if diam_str in ("0", "", "auto", "none") else float(diam_str)
        channel = self.seg_channel.get()
        z_val = self.seg_z_idx.get().strip()
        z = int(z_val) if z_val else 0
        min_sz = self.seg_min_size.get()
        max_sz = self.seg_max_size.get()
        min_sol = self.seg_min_solidity.get()
        gpu = self.seg_gpu_var.get()
        augment = self.seg_augment_var.get()
        rm_edges = self.seg_edges_var.get()
        smooth_r = self.seg_smooth_radius.get()
        use_dic_norm = self.seg_dic_norm_var.get()
        clahe_clip = self.seg_clahe_clip.get()
        flow_thr = self.seg_flow_thr.get()
        cellprob_thr = self.seg_cellprob_thr.get()
        model_type = self._seg_get_model_type()

        is_cyto = mode == "cytoplasm"
        nuc_mask_dir = None
        nuc_dilate_px = 0
        min_nuc_pixels = 10
        min_overlap_frac = 0.005

        if is_cyto:
            nuc_mask_dir = self.seg_nuc_mask_dir.get()
            if not nuc_mask_dir:
                messagebox.showwarning(
                    "Missing input",
                    "Cytoplasm mode requires a Nucleus Masks Folder.\n"
                    "Run nucleus segmentation first."
                )
                return
            nuc_dilate_px = self.seg_nuc_dilate.get()
            min_nuc_pixels = self.seg_min_nuc_px.get()
            min_overlap_frac = self.seg_min_overlap_frac.get()

        norm_label = f"DIC (CLAHE clip={clahe_clip:.3f})" if use_dic_norm else "LUT"
        self._seg_log_append(
            f"Segmentation ({mode}): model={model_type}, diameter={diameter}, "
            f"channel={channel}, z={z}, flow={flow_thr}, cellprob={cellprob_thr}, "
            f"min_size={min_sz}, max_size={max_sz}, solidity={min_sol}, "
            f"smooth={smooth_r}, norm={norm_label}, augment={augment}"
        )
        self.btn_seg_run.config(state=tk.DISABLED)
        self.seg_tree.delete(*self.seg_tree.get_children())
        self.seg_progress.config(mode="determinate", value=0)
        self.seg_status.set(f"Running {mode} segmentation...")

        def task():
            try:
                from segmentation_utils import (
                    load_image_2d, auto_lut_clip, normalize_dic,
                    ensure_2d,
                    load_cellpose_model, run_cellpose,
                    postprocess_mask,
                    save_mask, save_seg_npy, save_triptych,
                    save_cytoplasm_triptych,
                    collect_image_paths,
                    compute_cytoplasm_mask,
                )
                import numpy as np
                import tifffile as tiff_io

                image_paths = collect_image_paths(input_path)
                if not image_paths:
                    self.log_queue.put("__SEG_ERROR__No TIFF images found.")
                    return

                self._seg_log_append_q(f"Found {len(image_paths)} image(s). Loading model ({model_type})...")
                model = load_cellpose_model(gpu=gpu, model_type=model_type)
                outdir = Path(out_dir)
                outdir.mkdir(parents=True, exist_ok=True)
                trip_dir = outdir / "triptychs"
                trip_dir.mkdir(parents=True, exist_ok=True)
                total = len(image_paths)

                for i, img_path in enumerate(image_paths, 1):
                    self._seg_log_append_q(f"  [{i}/{total}] {img_path.name}")
                    pct = int(100 * i / total)
                    self.log_queue.put(f"__SEG_PROGRESS__{pct}")
                    n_objects = -1

                    try:
                        img2d = load_image_2d(img_path, channel_index=channel, z_index=z)
                        if use_dic_norm:
                            img_norm = normalize_dic(img2d, clip_limit=clahe_clip)
                        else:
                            img_norm = auto_lut_clip(img2d)

                        masks, flows = run_cellpose(
                            img_norm, model=model, diameter=diameter,
                            flow_threshold=flow_thr,
                            cellprob_threshold=cellprob_thr,
                            augment=augment,
                        )
                        edge_thr = 0.25 if use_dic_norm else 0.0
                        masks = postprocess_mask(
                            masks, min_size=min_sz, max_size=max_sz,
                            remove_edges=rm_edges, smooth_radius=smooth_r,
                            edge_thresh=edge_thr, min_solidity=min_sol,
                        )
                        n_objects = int(masks.max())
                        stem = img_path.stem

                        if is_cyto:
                            nuc_path = self._find_nuc_mask(nuc_mask_dir, stem)
                            if nuc_path is None:
                                self._seg_log_append_q(f"    [WARN] No nucleus mask for {stem}")
                                save_mask(masks, outdir / f"{stem}_cell_masks.tif")
                                save_seg_npy(img_norm, masks, flows, img_path.name, img_path.parent, diameter)
                                save_triptych(img_norm, masks, trip_dir / f"{stem}_cell_triptych.png")
                            else:
                                nuc_m = ensure_2d(tiff_io.imread(str(nuc_path)))
                                if nuc_m.shape != masks.shape:
                                    self._seg_log_append_q(f"    [WARN] Shape mismatch, skipping cyto")
                                    save_mask(masks, outdir / f"{stem}_cell_masks.tif")
                                    save_seg_npy(img_norm, masks, flows, img_path.name, img_path.parent, diameter)
                                else:
                                    cyto_mask, kept, orphans = compute_cytoplasm_mask(
                                        masks, nuc_m,
                                        nuc_dilate_px=nuc_dilate_px,
                                        min_nuc_pixels=min_nuc_pixels,
                                        min_overlap_frac=min_overlap_frac,
                                    )
                                    self._seg_log_append_q(
                                        f"    Kept {len(kept)} cells, removed {len(orphans)} orphans"
                                    )
                                    save_mask(masks, outdir / f"{stem}_cell_masks.tif")
                                    save_mask(cyto_mask, outdir / f"{stem}_cyto_masks.tif")
                                    save_seg_npy(img_norm, masks, flows, img_path.name, img_path.parent, diameter)
                                    save_seg_npy(img_norm, cyto_mask, flows, f"{stem}_cyto", outdir, diameter)
                                    save_cytoplasm_triptych(
                                        img_norm, masks, nuc_m, cyto_mask,
                                        trip_dir / f"{stem}_cyto_triptych.png"
                                    )
                        else:
                            save_mask(masks, outdir / f"{stem}_cyto3_masks.tif")
                            save_seg_npy(img_norm, masks, flows, img_path.name, img_path.parent, diameter)
                            save_triptych(img_norm, masks, trip_dir / f"{stem}_triptych.png")

                        status = "OK" if n_objects > 0 else "No signal"
                    except Exception as e:
                        self._seg_log_append_q(f"    [ERROR] {e}")
                        status = "FAILED"

                    self.log_queue.put(f"__SEG_RESULT__{img_path.name}||{n_objects}||{status}")

                self.log_queue.put("__SEG_DONE__")
            except Exception as exc:
                self.log_queue.put(f"__SEG_ERROR__{exc}")

        threading.Thread(target=task, daemon=True).start()

    def _on_seg_finished(self, error=None):
        self.btn_seg_run.config(state=tk.NORMAL)
        self.seg_progress.config(value=100)
        if error:
            self.seg_status.set(f"Error: {str(error)[:80]}")
            self._seg_log_append(f"[ERROR] {error}")
        else:
            self.seg_status.set("Segmentation complete")
            self._seg_log_append("[DONE] Segmentation complete.")

    # ==================================================================
    # TAB: PUNCTA SEGMENTATION (hybrid detection framework)
    # ==================================================================
    def _build_puncta_seg_tab(self):
        tab = self.tab_puncta_seg

        # Scrollable canvas
        canvas = tk.Canvas(tab, highlightthickness=0)
        sb = ttk.Scrollbar(tab, orient=tk.VERTICAL, command=canvas.yview)
        self.pseg_body = ttk.Frame(canvas)
        self.pseg_body.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.pseg_body, anchor=tk.NW)
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        body = self.pseg_body

        info = ttk.Label(
            body,
            text="Detect puncta using multiple methods: Threshold, LoG (Punctatools-style),\n"
                 "DoG, Intensity-Ratio (PunctaFinder-style), Spotiflow (deep-learning),\n"
                 "or Consensus (combine 2+ detectors).  Produces label masks for training.\n"
                 "Saves Cellpose-compatible _seg.npy for curation in the Cellpose GUI.",
            foreground="gray",
        )
        info.pack(anchor=tk.W, padx=10, pady=(10, 5))

        # ---- Input / Output ----
        io_frame = ttk.LabelFrame(body, text="Input / Output", padding=10)
        io_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(io_frame, text="Image Directory:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.pseg_input_dir = tk.StringVar()
        ttk.Entry(io_frame, textvariable=self.pseg_input_dir, width=55).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(io_frame, text="Browse...",
                    command=lambda: self._browse_dir(self.pseg_input_dir)).grid(row=0, column=2, pady=2)

        ttk.Label(io_frame, text="Output Masks Directory:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.pseg_out_dir = tk.StringVar()
        ttk.Entry(io_frame, textvariable=self.pseg_out_dir, width=55).grid(row=1, column=1, padx=5, pady=2)
        ttk.Button(io_frame, text="Browse...",
                    command=lambda: self._browse_dir(self.pseg_out_dir)).grid(row=1, column=2, pady=2)

        ttk.Label(io_frame, text="Cell Masks Folder (optional):").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.pseg_cell_mask_dir = tk.StringVar()
        ttk.Entry(io_frame, textvariable=self.pseg_cell_mask_dir, width=55).grid(row=2, column=1, padx=5, pady=2)
        ttk.Button(io_frame, text="Browse...",
                    command=lambda: self._browse_dir(self.pseg_cell_mask_dir)).grid(row=2, column=2, pady=2)
        ttk.Label(io_frame, text="Assigns puncta to cells and exports per-cell CSV",
                  foreground="gray").grid(row=2, column=3, sticky=tk.W, padx=5)

        # ---- Channel ----
        ch_frame = ttk.LabelFrame(body, text="Channel Selection", padding=10)
        ch_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(ch_frame, text="Puncta Channel:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.pseg_channel = tk.IntVar(value=1)
        ttk.Combobox(ch_frame, textvariable=self.pseg_channel, values=[0, 1, 2, 3],
                     width=5, state="readonly").grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        ttk.Label(ch_frame, text="(1 = mEGFP for most datasets)", foreground="gray").grid(
            row=0, column=2, sticky=tk.W, padx=5)

        ttk.Label(ch_frame, text="Z-Slice:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.pseg_z_idx = tk.IntVar(value=0)
        ttk.Entry(ch_frame, textvariable=self.pseg_z_idx, width=6).grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)

        # ---- Detection Method ----
        method_frame = ttk.LabelFrame(body, text="Detection Method", padding=10)
        method_frame.pack(fill=tk.X, padx=10, pady=5)

        self.pseg_method = tk.StringVar(value="threshold")
        method_row1 = ttk.Frame(method_frame)
        method_row1.pack(fill=tk.X)
        for label, val in [("Threshold", "threshold"),
                           ("LoG (Punctatools)", "log"),
                           ("DoG", "dog")]:
            ttk.Radiobutton(method_row1, text=label, variable=self.pseg_method,
                            value=val, command=self._pseg_on_method_change).pack(side=tk.LEFT, padx=6)

        method_row2 = ttk.Frame(method_frame)
        method_row2.pack(fill=tk.X, pady=(2, 0))
        for label, val in [("Intensity-Ratio (PunctaFinder)", "intensity_ratio"),
                           ("Spotiflow (DL)", "spotiflow"),
                           ("Spotiflow + Otsu (hybrid)", "spotiflow+threshold"),
                           ("Spotiflow + LoG (hybrid)", "spotiflow+log")]:
            ttk.Radiobutton(method_row2, text=label, variable=self.pseg_method,
                            value=val, command=self._pseg_on_method_change).pack(side=tk.LEFT, padx=6)

        method_row3 = ttk.Frame(method_frame)
        method_row3.pack(fill=tk.X, pady=(2, 0))
        for label, val in [("Tight Borders", "tight_borders"),
                           ("Consensus (multi-detector)", "consensus")]:
            ttk.Radiobutton(method_row3, text=label, variable=self.pseg_method,
                            value=val, command=self._pseg_on_method_change).pack(side=tk.LEFT, padx=6)

        # -- Threshold sub-panel --
        self.pseg_thresh_frame = ttk.Frame(method_frame)
        ttk.Label(self.pseg_thresh_frame, text="Algorithm:").pack(side=tk.LEFT, padx=(0, 5))
        self.pseg_thresh_method = tk.StringVar(value="otsu")
        ttk.Combobox(self.pseg_thresh_frame, textvariable=self.pseg_thresh_method,
                     values=["otsu", "yen", "triangle", "li", "custom"],
                     width=10, state="readonly").pack(side=tk.LEFT, padx=(0, 15))
        ttk.Label(self.pseg_thresh_frame, text="Custom value (0-1):").pack(side=tk.LEFT, padx=(0, 5))
        self.pseg_custom_thresh = tk.DoubleVar(value=0.3)
        ttk.Entry(self.pseg_thresh_frame, textvariable=self.pseg_custom_thresh, width=6).pack(side=tk.LEFT)

        # -- Blob sub-panel (LoG / DoG) --
        self.pseg_blob_frame = ttk.Frame(method_frame)
        ttk.Label(self.pseg_blob_frame, text="Min sigma:").pack(side=tk.LEFT, padx=(0, 5))
        self.pseg_min_sigma = tk.DoubleVar(value=1.0)
        ttk.Entry(self.pseg_blob_frame, textvariable=self.pseg_min_sigma, width=6).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(self.pseg_blob_frame, text="Max sigma:").pack(side=tk.LEFT, padx=(0, 5))
        self.pseg_max_sigma = tk.DoubleVar(value=5.0)
        ttk.Entry(self.pseg_blob_frame, textvariable=self.pseg_max_sigma, width=6).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(self.pseg_blob_frame, text="Blob threshold:").pack(side=tk.LEFT, padx=(0, 5))
        self.pseg_blob_thresh = tk.DoubleVar(value=0.1)
        ttk.Entry(self.pseg_blob_frame, textvariable=self.pseg_blob_thresh, width=6).pack(side=tk.LEFT)

        # -- Intensity-Ratio sub-panel (PunctaFinder) --
        self.pseg_ir_frame = ttk.Frame(method_frame)
        ttk.Label(self.pseg_ir_frame, text="Punctum radius:").pack(side=tk.LEFT, padx=(0, 5))
        self.pseg_ir_radius = tk.IntVar(value=3)
        ttk.Entry(self.pseg_ir_frame, textvariable=self.pseg_ir_radius, width=4).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(self.pseg_ir_frame, text="Local ratio:").pack(side=tk.LEFT, padx=(0, 5))
        self.pseg_ir_local = tk.DoubleVar(value=1.5)
        ttk.Entry(self.pseg_ir_frame, textvariable=self.pseg_ir_local, width=5).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(self.pseg_ir_frame, text="Global ratio:").pack(side=tk.LEFT, padx=(0, 5))
        self.pseg_ir_global = tk.DoubleVar(value=1.5)
        ttk.Entry(self.pseg_ir_frame, textvariable=self.pseg_ir_global, width=5).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(self.pseg_ir_frame, text="Max CV:").pack(side=tk.LEFT, padx=(0, 5))
        self.pseg_ir_cv = tk.DoubleVar(value=0.5)
        ttk.Entry(self.pseg_ir_frame, textvariable=self.pseg_ir_cv, width=5).pack(side=tk.LEFT)

        # -- Spotiflow sub-panel --
        self.pseg_spoti_frame = ttk.Frame(method_frame)
        ttk.Label(self.pseg_spoti_frame, text="Model:").pack(side=tk.LEFT, padx=(0, 5))
        self.pseg_spoti_model = tk.StringVar(value="general")
        ttk.Combobox(self.pseg_spoti_frame, textvariable=self.pseg_spoti_model,
                     values=["general", "Custom..."], width=12).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(self.pseg_spoti_frame, text="Prob threshold:").pack(side=tk.LEFT, padx=(0, 5))
        self.pseg_spoti_prob = tk.DoubleVar(value=0.5)
        ttk.Entry(self.pseg_spoti_frame, textvariable=self.pseg_spoti_prob, width=5).pack(side=tk.LEFT)
        ttk.Label(self.pseg_spoti_frame, text="  Spot radius (px):").pack(side=tk.LEFT, padx=(10, 5))
        self.pseg_spoti_radius = tk.IntVar(value=2)
        ttk.Spinbox(self.pseg_spoti_frame, from_=1, to=20,
                     textvariable=self.pseg_spoti_radius, width=4).pack(side=tk.LEFT)
        ttk.Label(self.pseg_spoti_frame, text="  (requires: pip install spotiflow)",
                  foreground="gray").pack(side=tk.LEFT, padx=5)

        # -- Tight Borders sub-panel --
        self.pseg_tb_frame = ttk.Frame(method_frame)
        tb_row1 = ttk.Frame(self.pseg_tb_frame)
        tb_row1.pack(fill=tk.X, pady=2)
        ttk.Label(tb_row1, text="Threshold factor:").pack(side=tk.LEFT, padx=(0, 5))
        self.pseg_tb_threshold = tk.DoubleVar(value=4.0)
        ttk.Entry(tb_row1, textvariable=self.pseg_tb_threshold, width=5).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(tb_row1, text="Max branch length (px):").pack(side=tk.LEFT, padx=(0, 5))
        self.pseg_tb_branch_len = tk.IntVar(value=10)
        ttk.Spinbox(tb_row1, from_=0, to=100,
                     textvariable=self.pseg_tb_branch_len, width=4).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(tb_row1, text="Connect free endings (px):").pack(side=tk.LEFT, padx=(0, 5))
        self.pseg_tb_connect = tk.IntVar(value=10)
        ttk.Spinbox(tb_row1, from_=0, to=100,
                     textvariable=self.pseg_tb_connect, width=4).pack(side=tk.LEFT)

        tb_row2 = ttk.Frame(self.pseg_tb_frame)
        tb_row2.pack(fill=tk.X, pady=2)
        ttk.Label(tb_row2, text="Min eq. diameter:").pack(side=tk.LEFT, padx=(0, 5))
        self.pseg_tb_min_eq_dia = tk.DoubleVar(value=0)
        ttk.Entry(tb_row2, textvariable=self.pseg_tb_min_eq_dia, width=5).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(tb_row2, text="Keep if border > :").pack(side=tk.LEFT, padx=(0, 5))
        self.pseg_tb_border_strength = tk.DoubleVar(value=0)
        ttk.Entry(tb_row2, textvariable=self.pseg_tb_border_strength, width=5).pack(side=tk.LEFT)

        # -- Consensus sub-panel --
        self.pseg_cons_frame = ttk.Frame(method_frame)
        cons_row1 = ttk.Frame(self.pseg_cons_frame)
        cons_row1.pack(fill=tk.X, pady=2)
        ttk.Label(cons_row1, text="Detectors to combine:").pack(side=tk.LEFT, padx=(0, 5))
        self.pseg_cons_thresh = tk.BooleanVar(value=True)
        ttk.Checkbutton(cons_row1, text="Threshold", variable=self.pseg_cons_thresh).pack(side=tk.LEFT, padx=3)
        self.pseg_cons_log = tk.BooleanVar(value=True)
        ttk.Checkbutton(cons_row1, text="LoG", variable=self.pseg_cons_log).pack(side=tk.LEFT, padx=3)
        self.pseg_cons_ir = tk.BooleanVar(value=False)
        ttk.Checkbutton(cons_row1, text="Intensity-Ratio", variable=self.pseg_cons_ir).pack(side=tk.LEFT, padx=3)
        self.pseg_cons_spoti = tk.BooleanVar(value=False)
        ttk.Checkbutton(cons_row1, text="Spotiflow", variable=self.pseg_cons_spoti).pack(side=tk.LEFT, padx=3)

        cons_row2 = ttk.Frame(self.pseg_cons_frame)
        cons_row2.pack(fill=tk.X, pady=2)
        ttk.Label(cons_row2, text="Strategy:").pack(side=tk.LEFT, padx=(0, 5))
        self.pseg_cons_strategy = tk.StringVar(value="weighted_confidence")
        ttk.Combobox(cons_row2, textvariable=self.pseg_cons_strategy,
                     values=["union", "intersection", "majority_vote", "weighted_confidence"],
                     width=22, state="readonly").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(cons_row2, text="Match distance:").pack(side=tk.LEFT, padx=(0, 5))
        self.pseg_cons_match_dist = tk.DoubleVar(value=3.0)
        ttk.Entry(cons_row2, textvariable=self.pseg_cons_match_dist, width=5).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(cons_row2, text="Confidence threshold:").pack(side=tk.LEFT, padx=(0, 5))
        self.pseg_cons_conf_thresh = tk.DoubleVar(value=0.3)
        ttk.Entry(cons_row2, textvariable=self.pseg_cons_conf_thresh, width=5).pack(side=tk.LEFT)

        # ---- Pre-processing ----
        preproc_frame = ttk.LabelFrame(body, text="Pre-processing", padding=10)
        preproc_frame.pack(fill=tk.X, padx=10, pady=5)

        row0 = ttk.Frame(preproc_frame)
        row0.pack(fill=tk.X, pady=2)
        ttk.Label(row0, text="Gaussian sigma:").pack(side=tk.LEFT, padx=(0, 5))
        self.pseg_sigma = tk.DoubleVar(value=1.0)
        ttk.Entry(row0, textvariable=self.pseg_sigma, width=6).pack(side=tk.LEFT, padx=(0, 15))

        self.pseg_bg_sub = tk.BooleanVar(value=True)
        ttk.Checkbutton(row0, text="Background subtraction", variable=self.pseg_bg_sub).pack(
            side=tk.LEFT, padx=(0, 10))

        ttk.Label(row0, text="Method:").pack(side=tk.LEFT, padx=(0, 5))
        self.pseg_bg_method = tk.StringVar(value="white_tophat")
        ttk.Combobox(row0, textvariable=self.pseg_bg_method,
                     values=["white_tophat", "rolling_ball", "gaussian", "median"],
                     width=14, state="readonly").pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(row0, text="Radius:").pack(side=tk.LEFT, padx=(0, 5))
        self.pseg_tophat_radius = tk.IntVar(value=15)
        ttk.Entry(row0, textvariable=self.pseg_tophat_radius, width=5).pack(side=tk.LEFT)

        # ---- Post-processing ----
        post_frame = ttk.LabelFrame(body, text="Post-processing (size filters)", padding=10)
        post_frame.pack(fill=tk.X, padx=10, pady=5)

        row1 = ttk.Frame(post_frame)
        row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="Min size (px):").pack(side=tk.LEFT, padx=(0, 5))
        self.pseg_min_size = tk.IntVar(value=3)
        ttk.Entry(row1, textvariable=self.pseg_min_size, width=8).pack(side=tk.LEFT, padx=(0, 15))
        ttk.Label(row1, text="Max size (px):").pack(side=tk.LEFT, padx=(0, 5))
        self.pseg_max_size = tk.IntVar(value=0)
        ttk.Entry(row1, textvariable=self.pseg_max_size, width=8).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(row1, text="(0 = no limit)", foreground="gray").pack(side=tk.LEFT, padx=(0, 15))
        ttk.Label(row1, text="Open radius:").pack(side=tk.LEFT, padx=(0, 5))
        self.pseg_open_radius = tk.IntVar(value=0)
        ttk.Entry(row1, textvariable=self.pseg_open_radius, width=5).pack(side=tk.LEFT)

        # ---- Output options ----
        out_opts = ttk.LabelFrame(body, text="Output Options", padding=10)
        out_opts.pack(fill=tk.X, padx=10, pady=5)

        self.pseg_save_npy = tk.BooleanVar(value=True)
        ttk.Checkbutton(out_opts, text="Save Cellpose _seg.npy (for curation / training)",
                        variable=self.pseg_save_npy).pack(anchor=tk.W, pady=1)
        self.pseg_save_trip = tk.BooleanVar(value=True)
        ttk.Checkbutton(out_opts, text="Save QC triptych PNGs",
                        variable=self.pseg_save_trip).pack(anchor=tk.W, pady=1)

        # ---- Buttons ----
        btn_frame = ttk.Frame(body)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)
        self.btn_pseg_run = ttk.Button(btn_frame, text="Run Puncta Segmentation",
                                       command=self._pseg_run)
        self.btn_pseg_run.pack(side=tk.LEFT, padx=5)

        self.btn_pseg_benchmark = ttk.Button(
            btn_frame, text="Benchmark Methods",
            command=self._pseg_benchmark)
        self.btn_pseg_benchmark.pack(side=tk.LEFT, padx=5)

        # Progress
        self.pseg_progress = ttk.Progressbar(body, mode="determinate")
        self.pseg_progress.pack(fill=tk.X, padx=10, pady=5)
        self.pseg_status = tk.StringVar(value="Ready")
        ttk.Label(body, textvariable=self.pseg_status).pack(padx=10, anchor=tk.W)

        # Results table
        result_frame = ttk.LabelFrame(body, text="Results", padding=5)
        result_frame.pack(fill=tk.X, padx=10, pady=(5, 5))
        columns = ("filename", "objects", "status")
        self.pseg_tree = ttk.Treeview(result_frame, columns=columns, show="headings", height=6)
        self.pseg_tree.heading("filename", text="Filename")
        self.pseg_tree.heading("objects", text="Puncta Found")
        self.pseg_tree.heading("status", text="Status")
        self.pseg_tree.column("filename", width=350)
        self.pseg_tree.column("objects", width=120, anchor=tk.CENTER)
        self.pseg_tree.column("status", width=120, anchor=tk.CENTER)
        pseg_tree_scroll = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.pseg_tree.yview)
        self.pseg_tree.configure(yscrollcommand=pseg_tree_scroll.set)
        self.pseg_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        pseg_tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Log
        log_frame = ttk.LabelFrame(body, text="Log", padding=5)
        log_frame.pack(fill=tk.X, padx=10, pady=(5, 10))
        self.pseg_log = scrolledtext.ScrolledText(
            log_frame, wrap=tk.WORD, height=8, state=tk.DISABLED, font=("Courier", 9)
        )
        self.pseg_log.pack(fill=tk.BOTH, expand=True)

        # Initial method state
        self._pseg_on_method_change()

    def _pseg_on_method_change(self):
        method = self.pseg_method.get()
        # Hide all sub-panels
        for frame in (self.pseg_thresh_frame, self.pseg_blob_frame,
                      self.pseg_ir_frame, self.pseg_spoti_frame,
                      self.pseg_tb_frame, self.pseg_cons_frame):
            frame.pack_forget()
        # Show relevant sub-panel
        if method == "threshold":
            self.pseg_thresh_frame.pack(fill=tk.X, pady=(5, 0))
        elif method in ("log", "dog"):
            self.pseg_blob_frame.pack(fill=tk.X, pady=(5, 0))
        elif method == "intensity_ratio":
            self.pseg_ir_frame.pack(fill=tk.X, pady=(5, 0))
        elif method == "spotiflow":
            self.pseg_spoti_frame.pack(fill=tk.X, pady=(5, 0))
        elif method == "spotiflow+threshold":
            self.pseg_spoti_frame.pack(fill=tk.X, pady=(5, 0))
            self.pseg_thresh_frame.pack(fill=tk.X, pady=(5, 0))
        elif method == "spotiflow+log":
            self.pseg_spoti_frame.pack(fill=tk.X, pady=(5, 0))
            self.pseg_blob_frame.pack(fill=tk.X, pady=(5, 0))
        elif method == "tight_borders":
            self.pseg_tb_frame.pack(fill=tk.X, pady=(5, 0))
        elif method == "consensus":
            self.pseg_cons_frame.pack(fill=tk.X, pady=(5, 0))

    def _pseg_log_append(self, msg):
        self.pseg_log.config(state=tk.NORMAL)
        self.pseg_log.insert(tk.END, msg + "\n")
        self.pseg_log.see(tk.END)
        self.pseg_log.config(state=tk.DISABLED)

    def _pseg_run(self):
        img_dir = self.pseg_input_dir.get()
        out_dir = self.pseg_out_dir.get()
        if not img_dir or not out_dir:
            messagebox.showwarning("Missing input",
                                   "Select both image directory and output directory.")
            return
        if NUCLEUS_SCRIPTS_DIR is None:
            messagebox.showerror("Nucleus/Scripts not found",
                                 "Cannot find Nucleus/Scripts/.\nMake sure the Nucleus/ folder is in the repository root.")
            return

        method = self.pseg_method.get()
        channel = self.pseg_channel.get()
        z_idx = self.pseg_z_idx.get()
        sigma = self.pseg_sigma.get()
        bg_sub = self.pseg_bg_sub.get()
        bg_method = self.pseg_bg_method.get()
        tophat_r = self.pseg_tophat_radius.get()
        min_sz = self.pseg_min_size.get()
        max_sz = self.pseg_max_size.get()
        open_r = self.pseg_open_radius.get()
        save_npy = self.pseg_save_npy.get()
        save_trip = self.pseg_save_trip.get()

        # Method-specific params
        thresh_method = self.pseg_thresh_method.get()
        custom_thresh = self.pseg_custom_thresh.get()
        min_sig = self.pseg_min_sigma.get()
        max_sig = self.pseg_max_sigma.get()
        blob_thr = self.pseg_blob_thresh.get()

        # Intensity-ratio params
        ir_radius = self.pseg_ir_radius.get()
        ir_local = self.pseg_ir_local.get()
        ir_global = self.pseg_ir_global.get()
        ir_cv = self.pseg_ir_cv.get()

        # Spotiflow params
        spoti_model = self.pseg_spoti_model.get()
        spoti_prob = self.pseg_spoti_prob.get()
        spoti_radius = self.pseg_spoti_radius.get()

        # Tight borders params
        tb_threshold = self.pseg_tb_threshold.get()
        tb_branch_len = self.pseg_tb_branch_len.get()
        tb_connect = self.pseg_tb_connect.get()
        tb_min_eq_dia = self.pseg_tb_min_eq_dia.get()
        tb_border_str = self.pseg_tb_border_strength.get()

        # Consensus params
        cons_detectors = []
        if self.pseg_cons_thresh.get():
            cons_detectors.append("threshold")
        if self.pseg_cons_log.get():
            cons_detectors.append("log")
        if self.pseg_cons_ir.get():
            cons_detectors.append("intensity_ratio")
        if self.pseg_cons_spoti.get():
            cons_detectors.append("spotiflow")
        cons_strategy = self.pseg_cons_strategy.get()
        cons_match_dist = self.pseg_cons_match_dist.get()
        cons_conf_thresh = self.pseg_cons_conf_thresh.get()

        # Cell mask dir (optional — for per-cell quantification)
        cell_mask_dir = self.pseg_cell_mask_dir.get() or None

        self._pseg_log_append(
            f"Puncta segmentation: method={method}, channel={channel}, z={z_idx}, "
            f"sigma={sigma}, bg={bg_method if bg_sub else 'off'}, "
            f"min_size={min_sz}, max_size={max_sz}"
        )
        if cell_mask_dir:
            self._pseg_log_append(f"  Cell masks: {cell_mask_dir}")
        self.btn_pseg_run.config(state=tk.DISABLED)
        self.pseg_tree.delete(*self.pseg_tree.get_children())
        self.pseg_progress.config(mode="determinate", value=0)
        self.pseg_status.set("Running puncta segmentation...")

        def task():
            try:
                from puncta_detection.puncta_segmentation import batch_segment

                def _on_progress(idx, total, fname, n_obj):
                    pct = int(100 * idx / total)
                    self.log_queue.put(f"__PSEG_PROGRESS__{pct}")
                    self.log_queue.put(
                        f"__PSEG_RESULT__{fname}||{n_obj}||"
                        f"{'OK' if n_obj > 0 else 'No puncta' if n_obj == 0 else 'FAILED'}"
                    )

                batch_segment(
                    image_dir=img_dir,
                    out_dir=out_dir,
                    channel=channel,
                    z_index=z_idx,
                    method=method,
                    sigma=sigma,
                    background_subtraction=bg_sub,
                    tophat_radius=tophat_r,
                    bg_method=bg_method,
                    threshold_method=thresh_method,
                    custom_threshold=custom_thresh if thresh_method == "custom" else None,
                    min_sigma=min_sig,
                    max_sigma=max_sig,
                    blob_threshold=blob_thr,
                    punctum_radius=ir_radius,
                    t_local=ir_local,
                    t_global=ir_global,
                    t_cv=ir_cv,
                    min_size=min_sz,
                    max_size=max_sz,
                    open_radius=open_r,
                    spotiflow_model=spoti_model,
                    spotiflow_prob=spoti_prob,
                    spot_radius=spoti_radius,
                    tb_threshold_factor=tb_threshold,
                    tb_max_branch_length=tb_branch_len,
                    tb_connect_distance=tb_connect,
                    tb_min_eq_diameter=tb_min_eq_dia,
                    tb_min_border_strength=tb_border_str,
                    consensus_detectors=cons_detectors if method == "consensus" else None,
                    consensus_strategy=cons_strategy,
                    consensus_threshold=cons_conf_thresh,
                    consensus_match_dist=cons_match_dist,
                    cell_mask_dir=cell_mask_dir,
                    save_cellpose_npy=save_npy,
                    save_triptychs=save_trip,
                    progress_callback=_on_progress,
                )
                self.log_queue.put("__PSEG_DONE__")
            except Exception as exc:
                import traceback
                self.log_queue.put(f"__PSEG_ERROR__{exc}\n{traceback.format_exc()}")

        threading.Thread(target=task, daemon=True).start()

    def _on_pseg_finished(self, error=None):
        self.btn_pseg_run.config(state=tk.NORMAL)
        self.pseg_progress.config(value=100)
        if error:
            self.pseg_status.set(f"Error: {str(error)[:80]}")
            self._pseg_log_append(f"[ERROR] {error}")
        else:
            self.pseg_status.set("Puncta segmentation complete")
            self._pseg_log_append("[DONE] Puncta segmentation complete.")

    def _pseg_benchmark(self):
        """Run the benchmarking pipeline comparing multiple methods."""
        img_dir = self.pseg_input_dir.get()
        out_dir = self.pseg_out_dir.get()
        if not img_dir or not out_dir:
            messagebox.showwarning("Missing input",
                                   "Select both image directory and output directory.")
            return
        if NUCLEUS_SCRIPTS_DIR is None:
            messagebox.showerror("Nucleus/Scripts not found",
                                 "Cannot find Nucleus/Scripts/.")
            return

        channel = self.pseg_channel.get()
        z_idx = self.pseg_z_idx.get()
        cell_mask_dir = self.pseg_cell_mask_dir.get() or None

        # Ask user which methods to benchmark
        methods_win = tk.Toplevel(self)
        methods_win.title("Select methods to compare")
        methods_win.geometry("380x350")
        ttk.Label(methods_win, text="Select methods to include in benchmark:",
                  font=("", 10, "bold")).pack(pady=(10, 5))

        method_vars = {}
        defaults = {
            "threshold": True,
            "spotiflow": True,
            "spotiflow+threshold": True,
            "spotiflow+log": False,
            "tight_borders": True,
            "log": False,
            "dog": False,
            "intensity_ratio": False,
        }
        for name, default in defaults.items():
            var = tk.BooleanVar(value=default)
            method_vars[name] = var
            ttk.Checkbutton(methods_win, text=name, variable=var).pack(
                anchor=tk.W, padx=20, pady=1)

        ttk.Label(methods_win, text="\nGround-truth masks (optional):").pack(anchor=tk.W, padx=10)
        gt_var = tk.StringVar()
        gt_frame = ttk.Frame(methods_win)
        gt_frame.pack(fill=tk.X, padx=10, pady=2)
        ttk.Entry(gt_frame, textvariable=gt_var, width=30).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(gt_frame, text="Browse...",
                    command=lambda: self._browse_dir(gt_var)).pack(side=tk.LEFT)

        def _run_benchmark():
            methods = [n for n, v in method_vars.items() if v.get()]
            if len(methods) < 2:
                messagebox.showwarning("Need methods",
                                       "Select at least 2 methods to compare.")
                return
            methods_win.destroy()

            bench_dir = str(Path(out_dir) / "benchmark")
            gt_dir = gt_var.get() or None

            self._pseg_log_append(f"Benchmark: comparing {', '.join(methods)}")
            self.btn_pseg_run.config(state=tk.DISABLED)
            self.btn_pseg_benchmark.config(state=tk.DISABLED)
            self.pseg_status.set("Running benchmark...")
            self.pseg_progress.config(mode="indeterminate")
            self.pseg_progress.start(20)

            def _task():
                try:
                    import pandas as pd
                    from puncta_detection.benchmark import run_benchmark
                    result = run_benchmark(
                        image_dir=img_dir,
                        out_dir=bench_dir,
                        methods=methods,
                        channel=channel,
                        z_index=z_idx,
                        gt_mask_dir=gt_dir,
                        cell_mask_dir=cell_mask_dir,
                        sigma=self.pseg_sigma.get(),
                        threshold_method=self.pseg_thresh_method.get(),
                        min_size=self.pseg_min_size.get(),
                        max_size=self.pseg_max_size.get(),
                        spotiflow_model=self.pseg_spoti_model.get(),
                        spotiflow_prob=self.pseg_spoti_prob.get(),
                        spot_radius=self.pseg_spoti_radius.get(),
                    )
                    # Log summary
                    if result and "summary" in result:
                        df = result["summary"]
                        for _, row in df.iterrows():
                            line = (f"  {row['method']:25s}  "
                                    f"count={row['mean_detections']:6.1f} +/- "
                                    f"{row['std_detections']:5.1f}  "
                                    f"area={row['mean_area']:5.1f}")
                            if "mean_gt_f1" in row and not pd.isna(row.get("mean_gt_f1")):
                                line += f"  F1={row['mean_gt_f1']:.3f}"
                            self.log_queue.put(f"__PSEG_LOG__{line}")
                    self.log_queue.put(f"__PSEG_LOG__Benchmark results saved to {bench_dir}")
                    self.log_queue.put("__PSEG_BENCH_DONE__")
                except Exception as exc:
                    import traceback
                    self.log_queue.put(f"__PSEG_ERROR__{exc}\n{traceback.format_exc()}")

            threading.Thread(target=_task, daemon=True).start()

        ttk.Button(methods_win, text="Run Benchmark", command=_run_benchmark).pack(pady=10)

    # ==================================================================
    # TAB: INTENSITY & PUNCTA ANALYSIS
    # ==================================================================
    def _build_analysis_tab(self):
        tab = self.tab_analysis

        info = ttk.Label(
            tab,
            text="Post-mask analysis: compute per-cell metrics from pre-computed masks.\n"
                 "Provide nucleus masks, puncta masks, and raw images. Optionally add cell masks\n"
                 "for cytoplasm metrics. Outputs per-cell CSV and optional per-puncta CSV.",
            foreground="gray",
        )
        info.pack(anchor=tk.W, padx=10, pady=(10, 5))

        # Inputs
        io_frame = ttk.LabelFrame(tab, text="Input Directories", padding=10)
        io_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(io_frame, text="Nucleus Masks Folder:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.ana_nuc_dir = tk.StringVar()
        ttk.Entry(io_frame, textvariable=self.ana_nuc_dir, width=50).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(io_frame, text="Browse...", command=lambda: self._browse_dir(self.ana_nuc_dir)).grid(
            row=0, column=2, pady=2
        )

        ttk.Label(io_frame, text="Puncta Masks Folder:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.ana_puncta_dir = tk.StringVar()
        ttk.Entry(io_frame, textvariable=self.ana_puncta_dir, width=50).grid(row=1, column=1, padx=5, pady=2)
        ttk.Button(io_frame, text="Browse...", command=lambda: self._browse_dir(self.ana_puncta_dir)).grid(
            row=1, column=2, pady=2
        )

        ttk.Label(io_frame, text="Raw OME-TIFF Images Folder:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.ana_intensity_dir = tk.StringVar()
        ttk.Entry(io_frame, textvariable=self.ana_intensity_dir, width=50).grid(row=2, column=1, padx=5, pady=2)
        ttk.Button(io_frame, text="Browse...", command=lambda: self._browse_dir(self.ana_intensity_dir)).grid(
            row=2, column=2, pady=2
        )

        ttk.Label(io_frame, text="Cell Masks Folder (optional):").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.ana_cell_dir = tk.StringVar()
        ttk.Entry(io_frame, textvariable=self.ana_cell_dir, width=50).grid(row=3, column=1, padx=5, pady=2)
        ttk.Button(io_frame, text="Browse...", command=lambda: self._browse_dir(self.ana_cell_dir)).grid(
            row=3, column=2, pady=2
        )

        # Output
        out_frame = ttk.LabelFrame(tab, text="Output", padding=10)
        out_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(out_frame, text="Per-cell CSV:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.ana_csv_path = tk.StringVar()
        ttk.Entry(out_frame, textvariable=self.ana_csv_path, width=50).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(out_frame, text="Browse...", command=self._ana_browse_csv).grid(
            row=0, column=2, pady=2
        )

        ttk.Label(out_frame, text="Triptych Output Folder:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.ana_trip_dir = tk.StringVar()
        ttk.Entry(out_frame, textvariable=self.ana_trip_dir, width=50).grid(row=1, column=1, padx=5, pady=2)
        ttk.Button(out_frame, text="Browse...", command=lambda: self._browse_dir(self.ana_trip_dir)).grid(
            row=1, column=2, pady=2
        )

        # Parameters
        param_frame = ttk.LabelFrame(tab, text="Parameters", padding=10)
        param_frame.pack(fill=tk.X, padx=10, pady=5)

        row0 = ttk.Frame(param_frame)
        row0.pack(fill=tk.X, pady=2)

        ttk.Label(row0, text="Nucleus channel:").pack(side=tk.LEFT, padx=(0, 5))
        self.ana_int_ch = tk.IntVar(value=2)
        ttk.Entry(row0, textvariable=self.ana_int_ch, width=5).pack(side=tk.LEFT, padx=(0, 15))

        ttk.Label(row0, text="Puncta channel:").pack(side=tk.LEFT, padx=(0, 5))
        self.ana_pun_ch = tk.IntVar(value=1)
        ttk.Entry(row0, textvariable=self.ana_pun_ch, width=5).pack(side=tk.LEFT, padx=(0, 15))

        ttk.Label(row0, text="Min puncta area:").pack(side=tk.LEFT, padx=(0, 5))
        self.ana_min_area = tk.IntVar(value=5)
        ttk.Entry(row0, textvariable=self.ana_min_area, width=5).pack(side=tk.LEFT, padx=(0, 15))

        ttk.Label(row0, text="Open radius:").pack(side=tk.LEFT, padx=(0, 5))
        self.ana_open_radius = tk.IntVar(value=1)
        ttk.Entry(row0, textvariable=self.ana_open_radius, width=5).pack(side=tk.LEFT)

        row1 = ttk.Frame(param_frame)
        row1.pack(fill=tk.X, pady=2)

        self.ana_trip_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row1, text="Generate QC triptychs", variable=self.ana_trip_var).pack(
            side=tk.LEFT, padx=8
        )

        self.ana_per_puncta_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            row1, text="Export per-puncta CSV (individual puncta metrics)",
            variable=self.ana_per_puncta_var
        ).pack(side=tk.LEFT, padx=8)

        # Metrics info
        metrics_frame = ttk.LabelFrame(tab, text="Output Metrics", padding=10)
        metrics_frame.pack(fill=tk.X, padx=10, pady=5)
        metrics_text = (
            "Per-cell CSV columns:\n"
            "  Shape: num_nuc_pixels, eccentricity, solidity\n"
            "  Puncta: num_puncta_objects, puncta_density, has_puncta, puncta_area_in_nuc\n"
            "  Nuc intensity: nuc_mean_raw, nuc_median_raw, nuc_std_raw, nuc_mean_bgsub\n"
            "  Puncta ch intensity: puncta_ch_mean, puncta_ch_median, puncta_ch_std, puncta_ch_mean_bgsub\n"
            "  Puncta spot intensity: puncta_mean_intensity, puncta_max_intensity, puncta_median_intensity\n"
            "  Cell (if cell masks): cell_area, cyto_area, cyto_nuc_mean, cyto_puncta_ch_mean, nuc_cyto_ratio\n"
            "\n"
            "Per-puncta CSV (optional): puncta_area, puncta_mean/max_intensity, centroid, eccentricity"
        )
        ttk.Label(metrics_frame, text=metrics_text, foreground="gray", justify=tk.LEFT,
                  font=("Courier", 8)).pack(anchor=tk.W)

        # Run button
        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)
        self.btn_ana_run = ttk.Button(btn_frame, text="Run Analysis", command=self._ana_run)
        self.btn_ana_run.pack(side=tk.LEFT, padx=5)

        # Progress
        self.ana_progress = ttk.Progressbar(tab, mode="determinate")
        self.ana_progress.pack(fill=tk.X, padx=10, pady=5)

        self.ana_status = tk.StringVar(value="Ready")
        ttk.Label(tab, textvariable=self.ana_status).pack(padx=10, anchor=tk.W)

        # Log
        log_frame = ttk.LabelFrame(tab, text="Log", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))
        self.ana_log = scrolledtext.ScrolledText(
            log_frame, wrap=tk.WORD, height=8, state=tk.DISABLED, font=("Courier", 9)
        )
        self.ana_log.pack(fill=tk.BOTH, expand=True)

    def _ana_browse_csv(self):
        path = filedialog.asksaveasfilename(
            title="Save CSV As",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            defaultextension=".csv",
        )
        if path:
            self.ana_csv_path.set(path)

    def _ana_log_append(self, msg):
        self.ana_log.config(state=tk.NORMAL)
        self.ana_log.insert(tk.END, msg + "\n")
        self.ana_log.see(tk.END)
        self.ana_log.config(state=tk.DISABLED)

    def _ana_run(self):
        nuc = self.ana_nuc_dir.get()
        puncta = self.ana_puncta_dir.get()
        intensity = self.ana_intensity_dir.get()
        csv_path = self.ana_csv_path.get()

        if not all([nuc, puncta, intensity, csv_path]):
            messagebox.showwarning(
                "Missing input",
                "Please fill in all required fields:\n"
                "- Nucleus masks folder\n"
                "- Puncta masks folder\n"
                "- Raw images folder\n"
                "- Output CSV path"
            )
            return
        if NUCLEUS_SCRIPTS_DIR is None:
            messagebox.showerror(
                "Nucleus/Scripts not found",
                "Cannot find Nucleus/Scripts/.\n"
                "Make sure the Nucleus/ folder is in the repository root."
            )
            return

        int_ch = self.ana_int_ch.get()
        pun_ch = self.ana_pun_ch.get()
        min_a = self.ana_min_area.get()
        open_r = self.ana_open_radius.get()
        trip = self.ana_trip_var.get()
        trip_dir = self.ana_trip_dir.get() or None
        cell_dir = self.ana_cell_dir.get() or None
        per_puncta = self.ana_per_puncta_var.get()

        self._ana_log_append(f"Analysis: nuc_ch={int_ch}, puncta_ch={pun_ch}, min_area={min_a}")
        if cell_dir:
            self._ana_log_append(f"  Cell masks: {cell_dir}")
        if per_puncta:
            self._ana_log_append("  Per-puncta CSV export enabled")
        self.btn_ana_run.config(state=tk.DISABLED)
        self.ana_status.set("Running analysis...")

        def task():
            try:
                from puncta_detection.mean_intensity_and_puncta import main as run_analysis

                def _on_progress(current, total):
                    pct = int(100 * current / total)
                    self.log_queue.put(f"__ANA_PROGRESS__{pct}")

                run_analysis(
                    nuc_dir=nuc,
                    puncta_dir=puncta,
                    intensity_dir=intensity,
                    out_csv=csv_path,
                    cell_dir=cell_dir,
                    min_puncta_area=min_a,
                    puncta_open_radius=open_r,
                    make_triptychs=trip,
                    triptych_out_dir=trip_dir,
                    intensity_channel=int_ch,
                    puncta_channel=pun_ch,
                    export_per_puncta=per_puncta,
                    progress_callback=_on_progress,
                )
                self.log_queue.put(f"__ANA_DONE__{csv_path}")
            except Exception as exc:
                self.log_queue.put(f"__ANA_ERROR__{exc}")

        threading.Thread(target=task, daemon=True).start()

    def _on_ana_finished(self, csv_path=None, error=None):
        self.btn_ana_run.config(state=tk.NORMAL)
        self.ana_progress.config(value=100)
        if error:
            self.ana_status.set(f"Error: {str(error)[:80]}")
            self._ana_log_append(f"[ERROR] {error}")
        else:
            self.ana_status.set("Analysis complete")
            self._ana_log_append(f"[DONE] Analysis complete. CSV: {csv_path}")

    # ==================================================================
    # Helper: generic directory browse
    # ==================================================================
    def _browse_dir(self, string_var):
        d = filedialog.askdirectory()
        if d:
            string_var.set(d)

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

            # -- ND2 Conversion --
            if msg == "__ND2_DONE__":
                self._on_nd2_finished()
                continue
            if msg.startswith("__ND2_ERROR__"):
                self._on_nd2_finished(error=msg[len("__ND2_ERROR__"):])
                continue

            # -- Segmentation --
            if msg == "__SEG_DONE__":
                self._on_seg_finished()
                continue
            if msg.startswith("__SEG_ERROR__"):
                self._on_seg_finished(error=msg[len("__SEG_ERROR__"):])
                continue
            if msg.startswith("__SEG_LOG__"):
                self._seg_log_append(msg[len("__SEG_LOG__"):])
                continue
            if msg.startswith("__SEG_PROGRESS__"):
                pct = int(msg[len("__SEG_PROGRESS__"):])
                self.seg_progress.config(value=pct)
                continue
            if msg.startswith("__SEG_RESULT__"):
                payload = msg[len("__SEG_RESULT__"):]
                name, n_str, status = payload.split("||", 2)
                n = int(n_str)
                self.seg_tree.insert("", tk.END, values=(name, n if n >= 0 else "-", status))
                continue

            # -- Puncta Segmentation --
            if msg == "__PSEG_DONE__":
                self._on_pseg_finished()
                continue
            if msg == "__PSEG_BENCH_DONE__":
                self.btn_pseg_run.config(state=tk.NORMAL)
                self.btn_pseg_benchmark.config(state=tk.NORMAL)
                self.pseg_progress.stop()
                self.pseg_progress.config(mode="determinate", value=100)
                self.pseg_status.set("Benchmark complete")
                self._pseg_log_append("[DONE] Benchmark complete.")
                continue
            if msg.startswith("__PSEG_LOG__"):
                self._pseg_log_append(msg[len("__PSEG_LOG__"):])
                continue
            if msg.startswith("__PSEG_ERROR__"):
                self._on_pseg_finished(error=msg[len("__PSEG_ERROR__"):])
                # Also re-enable benchmark button
                self.btn_pseg_benchmark.config(state=tk.NORMAL)
                continue
            if msg.startswith("__PSEG_PROGRESS__"):
                pct = int(msg[len("__PSEG_PROGRESS__"):])
                self.pseg_progress.config(value=pct)
                self.pseg_status.set(f"Processing... {pct}%")
                continue
            if msg.startswith("__PSEG_RESULT__"):
                payload = msg[len("__PSEG_RESULT__"):]
                name, n_str, status = payload.split("||", 2)
                n = int(n_str)
                self.pseg_tree.insert("", tk.END,
                                      values=(name, n if n >= 0 else "-", status))
                continue

            # -- Analysis --
            if msg.startswith("__ANA_PROGRESS__"):
                pct = int(msg[len("__ANA_PROGRESS__"):])
                self.ana_progress.config(value=pct)
                self.ana_status.set(f"Analysing... {pct}%")
                continue
            if msg.startswith("__ANA_DONE__"):
                self._on_ana_finished(csv_path=msg[len("__ANA_DONE__"):])
                continue
            if msg.startswith("__ANA_ERROR__"):
                self._on_ana_finished(error=msg[len("__ANA_ERROR__"):])
                continue

            # -- Training --
            if msg == "__TRAINING_DONE__":
                self._on_training_finished()
                continue
            if msg.startswith("__TRAINING_ERROR__"):
                self._on_training_finished(error=msg[len("__TRAINING_ERROR__"):])
                continue

            # -- Evaluation --
            if msg.startswith("__EVAL_DONE__"):
                self._on_eval_finished(message=msg[len("__EVAL_DONE__"):])
                continue
            if msg.startswith("__EVAL_ERROR__"):
                self._on_eval_finished(error=msg[len("__EVAL_ERROR__"):])
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
        nuc_status = f"Found: {NUCLEUS_SCRIPTS_DIR}" if NUCLEUS_SCRIPTS_DIR else "NOT FOUND"
        messagebox.showinfo(
            "About",
            "Puncta-CSAT Segmentation Pipeline\n\n"
            "Unified GUI for analysis and training workflows.\n\n"
            "Analysis Workflow:\n"
            "  - ND2 to OME-TIFF conversion\n"
            "  - Cellpose segmentation (nucleus / puncta / cell / cytoplasm)\n"
            "    with model selection (cyto3 / cpsam / custom / BioImage.io)\n"
            "    and channel & normalization controls\n"
            "  - Per-cell intensity & puncta analysis\n\n"
            "Training Workflow:\n"
            "  - Cellpose model fine-tuning (DIC / fluorescence)\n"
            "  - Model evaluation with metrics\n\n"
            f"Nucleus/Scripts: {nuc_status}\n"
            "Built with Cellpose + tkinter",
        )


def main():
    app = SegmentationGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
