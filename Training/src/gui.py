#!/usr/bin/env python3
"""
Unified GUI for the Puncta-CSAT Segmentation Pipeline.

Combines Training workflow and Analysis workflow in a single interface.

Provides tabs for:
  -- Analysis Workflow --
  1. ND2 Conversion   — Convert ND2 files to OME-TIFF
  2. Deconvolution    — Pre-processing: Richardson-Lucy (with PSF, TV
                        regularisation, pre-denoising) or CARE/CSBDeep (DL)
  3. Segmentation     — Run Cellpose (nucleus / puncta / cell / cytoplasm)
                        with model selection, channel & normalization controls
  4. Puncta Segmentation — Multi-method puncta detection & segmentation
  5. Analysis         — Per-cell intensity & puncta analysis

  -- Training Workflow --
  6. File Renaming    — Rename images/masks to pipeline convention
  7. Configuration    — Edit training parameters from YAML configs
  8. Training         — Launch and monitor model training
  9. Evaluation       — Run inference and view results
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
        self.tab_deconv = ttk.Frame(self.notebook)
        self.tab_segmentation = ttk.Frame(self.notebook)
        self.tab_puncta_seg = ttk.Frame(self.notebook)
        self.tab_analysis = ttk.Frame(self.notebook)

        # -- Training workflow tabs --
        self.tab_rename = ttk.Frame(self.notebook)
        self.tab_config = ttk.Frame(self.notebook)
        self.tab_train = ttk.Frame(self.notebook)
        self.tab_eval = ttk.Frame(self.notebook)
        self.tab_compare = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_nd2, text="  ND2 Conversion  ")
        self.notebook.add(self.tab_deconv, text="  Deconvolution  ")
        self.notebook.add(self.tab_segmentation, text="  Segmentation  ")
        self.notebook.add(self.tab_puncta_seg, text="  Puncta Segmentation  ")
        self.notebook.add(self.tab_analysis, text="  Phase Separation Analysis  ")
        self.notebook.add(self.tab_rename, text="  File Renaming  ")
        self.notebook.add(self.tab_config, text="  Configuration  ")
        self.notebook.add(self.tab_train, text="  Training  ")
        self.notebook.add(self.tab_eval, text="  Evaluation  ")
        self.notebook.add(self.tab_compare, text="  Mask Comparison  ")

        self._build_nd2_tab()
        self._build_deconv_tab()
        self._build_segmentation_tab()
        self._build_puncta_seg_tab()
        self._build_analysis_tab()
        self._build_rename_tab()
        self._build_config_tab()
        self._build_train_tab()
        self._build_eval_tab()
        self._build_compare_tab()

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
    # TAB: DECONVOLUTION (pre-processing)
    #   Two back-ends:
    #   - Richardson-Lucy (classical, with PSF + TV regularisation)
    #   - CARE / CSBDeep (deep-learning, open-source)
    # ==================================================================
    def _build_deconv_tab(self):
        tab = self.tab_deconv

        # Scrollable canvas (many parameters)
        canvas = tk.Canvas(tab, highlightthickness=0)
        sb_canvas = ttk.Scrollbar(tab, orient=tk.VERTICAL, command=canvas.yview)
        self.deconv_body = ttk.Frame(canvas)
        self.deconv_body.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.deconv_body, anchor=tk.NW)
        canvas.configure(yscrollcommand=sb_canvas.set)
        sb_canvas.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        body = self.deconv_body

        info = ttk.Label(
            body,
            text="Deconvolution pre-processing for fluorescence images.\n"
                 "Richardson-Lucy: classical iterative (no external deps).  "
                 "CARE/CSBDeep: pip install csbdeep tensorflow",
            foreground="gray",
        )
        info.pack(anchor=tk.W, padx=10, pady=(10, 5))

        # -- Input / Output --
        io_frame = ttk.LabelFrame(body, text="Input / Output", padding=10)
        io_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(io_frame, text="Image directory:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.deconv_input_dir = tk.StringVar()
        ttk.Entry(io_frame, textvariable=self.deconv_input_dir, width=55).grid(
            row=0, column=1, padx=5, pady=2)
        ttk.Button(io_frame, text="Browse...", command=self._deconv_browse_input).grid(
            row=0, column=2, pady=2)

        ttk.Label(io_frame, text="Output directory:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.deconv_out_dir = tk.StringVar()
        ttk.Entry(io_frame, textvariable=self.deconv_out_dir, width=55).grid(
            row=1, column=1, padx=5, pady=2)
        ttk.Button(io_frame, text="Browse...", command=self._deconv_browse_output).grid(
            row=1, column=2, pady=2)

        ch_frame = ttk.Frame(io_frame)
        ch_frame.grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=(5, 0))
        ttk.Label(ch_frame, text="Channel:").pack(side=tk.LEFT, padx=(0, 5))
        self.deconv_channel = tk.IntVar(value=0)
        self.deconv_channel_name = tk.StringVar(value="")
        self._deconv_ome_channel_names = []
        self.deconv_channel_combo = ttk.Combobox(
            ch_frame, textvariable=self.deconv_channel_name, width=22, state="readonly")
        self.deconv_channel_combo.pack(side=tk.LEFT, padx=(0, 5))
        self.deconv_channel_combo.bind(
            "<<ComboboxSelected>>",
            lambda e: self.deconv_channel.set(self.deconv_channel_combo.current()))
        ttk.Button(ch_frame, text="Detect",
                   command=self._deconv_detect_channels).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(ch_frame, text="or index:").pack(side=tk.LEFT, padx=(0, 3))
        ttk.Spinbox(ch_frame, from_=0, to=10, textvariable=self.deconv_channel, width=4).pack(
            side=tk.LEFT, padx=(0, 15))
        ttk.Label(ch_frame, text="Z-plane index:").pack(side=tk.LEFT, padx=(0, 5))
        self.deconv_z_index = tk.IntVar(value=0)
        ttk.Spinbox(ch_frame, from_=0, to=200, textvariable=self.deconv_z_index, width=4).pack(
            side=tk.LEFT)

        # -- Method selection --
        method_frame = ttk.LabelFrame(body, text="Method", padding=10)
        method_frame.pack(fill=tk.X, padx=10, pady=5)

        self.deconv_method = tk.StringVar(value="richardson_lucy")
        ttk.Radiobutton(method_frame, text="Richardson-Lucy (recommended)",
                         variable=self.deconv_method, value="richardson_lucy",
                         command=self._deconv_on_method_change).pack(
            side=tk.LEFT, padx=(0, 20))
        ttk.Radiobutton(method_frame, text="CARE / CSBDeep (deep-learning)",
                         variable=self.deconv_method, value="care",
                         command=self._deconv_on_method_change).pack(
            side=tk.LEFT)

        # -- Pre-denoising (shared, shown for RL) --
        self.deconv_denoise_frame = ttk.LabelFrame(
            body, text="Pre-denoising (stabilises RL on noisy data)", padding=10)
        self.deconv_denoise_frame.pack(fill=tk.X, padx=10, pady=5)

        dn_row0 = ttk.Frame(self.deconv_denoise_frame)
        dn_row0.pack(fill=tk.X, pady=2)
        self.deconv_denoise_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(dn_row0, text="Enable pre-denoising",
                         variable=self.deconv_denoise_enabled).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Label(dn_row0, text="Method:").pack(side=tk.LEFT, padx=(0, 5))
        self.deconv_denoise_method = tk.StringVar(value="nlm")
        ttk.Combobox(dn_row0, textvariable=self.deconv_denoise_method,
                      values=["nlm", "bilateral", "gaussian", "median"],
                      width=10, state="readonly").pack(side=tk.LEFT, padx=(0, 15))
        ttk.Label(dn_row0, text="Sigma / radius:").pack(side=tk.LEFT, padx=(0, 5))
        self.deconv_denoise_sigma = tk.DoubleVar(value=0.0)
        ttk.Entry(dn_row0, textvariable=self.deconv_denoise_sigma, width=6).pack(
            side=tk.LEFT, padx=(0, 5))
        ttk.Label(dn_row0, text="(0 = auto-estimate)", foreground="gray").pack(side=tk.LEFT)

        dn_hint = ttk.Label(
            self.deconv_denoise_frame,
            text="NLM (non-local means): best SNR preservation, recommended for most data.\n"
                 "Bilateral: edge-preserving, fast.  Gaussian: fast blur.  "
                 "Median: salt-and-pepper noise.",
            foreground="gray",
        )
        dn_hint.pack(anchor=tk.W, pady=(2, 0))

        # -- Richardson-Lucy parameters --
        self.deconv_rl_frame = ttk.LabelFrame(
            body, text="Richardson-Lucy Parameters", padding=10)
        self.deconv_rl_frame.pack(fill=tk.X, padx=10, pady=5)

        # Modality selector
        mod_row = ttk.Frame(self.deconv_rl_frame)
        mod_row.pack(fill=tk.X, pady=2)
        ttk.Label(mod_row, text="Modality:").pack(side=tk.LEFT, padx=(0, 5))
        self.deconv_modality = tk.StringVar(value="widefield")
        ttk.Radiobutton(mod_row, text="Widefield",
                         variable=self.deconv_modality, value="widefield",
                         command=self._deconv_on_modality_change).pack(
            side=tk.LEFT, padx=(0, 15))
        ttk.Radiobutton(mod_row, text="Spinning Disc",
                         variable=self.deconv_modality, value="spinning_disc",
                         command=self._deconv_on_modality_change).pack(
            side=tk.LEFT)

        # PSF source
        psf_row0 = ttk.Frame(self.deconv_rl_frame)
        psf_row0.pack(fill=tk.X, pady=2)
        ttk.Label(psf_row0, text="PSF:").pack(side=tk.LEFT, padx=(0, 5))
        self.deconv_psf_type = tk.StringVar(value="theoretical")
        ttk.Radiobutton(psf_row0, text="Theoretical",
                         variable=self.deconv_psf_type, value="theoretical",
                         command=self._deconv_on_psf_change).pack(side=tk.LEFT, padx=(0, 15))
        ttk.Radiobutton(psf_row0, text="Measured PSF file",
                         variable=self.deconv_psf_type, value="measured",
                         command=self._deconv_on_psf_change).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Button(psf_row0, text="Auto-detect from images",
                    command=self._deconv_autodetect_metadata).pack(side=tk.LEFT)

        # Theoretical PSF parameters — row 1: optical params
        self.deconv_psf_theo_frame = ttk.Frame(self.deconv_rl_frame)
        self.deconv_psf_theo_frame.pack(fill=tk.X, pady=2)
        ttk.Label(self.deconv_psf_theo_frame, text="Emission (nm):").pack(
            side=tk.LEFT, padx=(0, 5))
        self.deconv_wavelength_em = tk.DoubleVar(value=520.0)
        ttk.Entry(self.deconv_psf_theo_frame,
                   textvariable=self.deconv_wavelength_em, width=6).pack(
            side=tk.LEFT, padx=(0, 12))
        ttk.Label(self.deconv_psf_theo_frame, text="NA:").pack(side=tk.LEFT, padx=(0, 5))
        self.deconv_na = tk.DoubleVar(value=1.4)
        ttk.Entry(self.deconv_psf_theo_frame, textvariable=self.deconv_na, width=5).pack(
            side=tk.LEFT, padx=(0, 12))
        ttk.Label(self.deconv_psf_theo_frame, text="Pixel size (nm):").pack(
            side=tk.LEFT, padx=(0, 5))
        self.deconv_pixel_size = tk.DoubleVar(value=65.0)
        ttk.Entry(self.deconv_psf_theo_frame,
                   textvariable=self.deconv_pixel_size, width=6).pack(
            side=tk.LEFT, padx=(0, 12))
        ttk.Label(self.deconv_psf_theo_frame, text="PSF size (px, 0=auto):").pack(
            side=tk.LEFT, padx=(0, 5))
        self.deconv_psf_size = tk.IntVar(value=0)
        ttk.Spinbox(self.deconv_psf_theo_frame, from_=0, to=101,
                      textvariable=self.deconv_psf_size, width=4).pack(side=tk.LEFT)

        # Theoretical PSF parameters — row 2: immersion RI & magnification
        self.deconv_psf_theo_frame2 = ttk.Frame(self.deconv_rl_frame)
        self.deconv_psf_theo_frame2.pack(fill=tk.X, pady=2)
        ttk.Label(self.deconv_psf_theo_frame2, text="Immersion RI:").pack(
            side=tk.LEFT, padx=(0, 5))
        self.deconv_n_immersion = tk.DoubleVar(value=1.515)
        imm_combo = ttk.Combobox(
            self.deconv_psf_theo_frame2,
            textvariable=self.deconv_n_immersion, width=6,
            values=["1.515", "1.33", "1.0"],
        )
        imm_combo.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(self.deconv_psf_theo_frame2,
                   text="(oil=1.515, water=1.33, air=1.0)",
                   foreground="gray").pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(self.deconv_psf_theo_frame2, text="Magnification:").pack(
            side=tk.LEFT, padx=(0, 5))
        self.deconv_magnification = tk.DoubleVar(value=0.0)
        ttk.Entry(self.deconv_psf_theo_frame2,
                   textvariable=self.deconv_magnification, width=6).pack(
            side=tk.LEFT, padx=(0, 5))
        ttk.Label(self.deconv_psf_theo_frame2,
                   text="(0 = pixel size is already at sample plane)",
                   foreground="gray").pack(side=tk.LEFT)

        # Spinning disc parameters — row 3: excitation & pinhole
        self.deconv_psf_sd_frame = ttk.Frame(self.deconv_rl_frame)
        ttk.Label(self.deconv_psf_sd_frame, text="Excitation (nm):").pack(
            side=tk.LEFT, padx=(0, 5))
        self.deconv_wavelength_ex = tk.DoubleVar(value=488.0)
        ttk.Entry(self.deconv_psf_sd_frame,
                   textvariable=self.deconv_wavelength_ex, width=6).pack(
            side=tk.LEFT, padx=(0, 12))
        ttk.Label(self.deconv_psf_sd_frame, text="Pinhole (\u00b5m):").pack(
            side=tk.LEFT, padx=(0, 5))
        self.deconv_pinhole_um = tk.DoubleVar(value=50.0)
        pinhole_combo = ttk.Combobox(
            self.deconv_psf_sd_frame,
            textvariable=self.deconv_pinhole_um, width=5,
            values=["25", "50", "70"],
        )
        pinhole_combo.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(self.deconv_psf_sd_frame,
                   text="(Yokogawa CSU-X1: 50, CSU-W1: 25 or 50)",
                   foreground="gray").pack(side=tk.LEFT)

        # Measured PSF path
        self.deconv_psf_meas_frame = ttk.Frame(self.deconv_rl_frame)
        ttk.Label(self.deconv_psf_meas_frame, text="PSF file:").pack(
            side=tk.LEFT, padx=(0, 5))
        self.deconv_psf_path = tk.StringVar()
        ttk.Entry(self.deconv_psf_meas_frame,
                   textvariable=self.deconv_psf_path, width=45).pack(
            side=tk.LEFT, padx=(0, 5))
        ttk.Button(self.deconv_psf_meas_frame, text="Browse...",
                    command=self._deconv_browse_psf).pack(side=tk.LEFT)

        # RL iteration parameters
        rl_row = ttk.Frame(self.deconv_rl_frame)
        rl_row.pack(fill=tk.X, pady=2)
        ttk.Label(rl_row, text="Iterations:").pack(side=tk.LEFT, padx=(0, 5))
        self.deconv_iterations = tk.IntVar(value=30)
        ttk.Spinbox(rl_row, from_=1, to=200,
                      textvariable=self.deconv_iterations, width=5).pack(
            side=tk.LEFT, padx=(0, 15))
        ttk.Label(rl_row, text="TV lambda:").pack(side=tk.LEFT, padx=(0, 5))
        self.deconv_tv_lambda = tk.DoubleVar(value=0.0)
        ttk.Entry(rl_row, textvariable=self.deconv_tv_lambda, width=7).pack(
            side=tk.LEFT, padx=(0, 15))
        ttk.Label(rl_row, text="Early-stop delta:").pack(side=tk.LEFT, padx=(0, 5))
        self.deconv_early_stop = tk.DoubleVar(value=0.0)
        ttk.Entry(rl_row, textvariable=self.deconv_early_stop, width=7).pack(
            side=tk.LEFT, padx=(0, 5))
        ttk.Label(rl_row, text="(0 = disabled)", foreground="gray").pack(side=tk.LEFT)

        # Background subtraction
        rl_row2 = ttk.Frame(self.deconv_rl_frame)
        rl_row2.pack(fill=tk.X, pady=2)
        self.deconv_subtract_bg = tk.BooleanVar(value=True)
        ttk.Checkbutton(rl_row2, text="Subtract background before RL",
                         variable=self.deconv_subtract_bg).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(rl_row2,
                   text="(auto-detects and removes camera/fluorescence background)",
                   foreground="gray").pack(side=tk.LEFT)

        rl_hint = ttk.Label(
            self.deconv_rl_frame,
            text="10-50 iterations typical.  TV lambda 0.001-0.01 suppresses noise (0 = pure RL).\n"
                 "Background subtraction greatly improves RL convergence.",
            foreground="gray",
        )
        rl_hint.pack(anchor=tk.W, pady=(2, 0))

        # -- CARE / CSBDeep parameters --
        self.deconv_care_frame = ttk.LabelFrame(
            body, text="CARE / CSBDeep Parameters", padding=10)

        care_row0 = ttk.Frame(self.deconv_care_frame)
        care_row0.pack(fill=tk.X, pady=2)
        ttk.Label(care_row0, text="Model directory:").pack(side=tk.LEFT, padx=(0, 5))
        self.deconv_care_model_dir = tk.StringVar()
        ttk.Entry(care_row0, textvariable=self.deconv_care_model_dir, width=40).pack(
            side=tk.LEFT, padx=(0, 5))
        ttk.Button(care_row0, text="Browse...", command=self._deconv_browse_care_model).pack(
            side=tk.LEFT)

        care_row1 = ttk.Frame(self.deconv_care_frame)
        care_row1.pack(fill=tk.X, pady=2)
        ttk.Label(care_row1, text="Model name:").pack(side=tk.LEFT, padx=(0, 5))
        self.deconv_care_model_name = tk.StringVar(value="my_care_model")
        ttk.Entry(care_row1, textvariable=self.deconv_care_model_name, width=25).pack(
            side=tk.LEFT, padx=(0, 15))
        ttk.Label(care_row1, text="Axes:").pack(side=tk.LEFT, padx=(0, 5))
        self.deconv_care_axes = tk.StringVar(value="YX")
        ttk.Combobox(care_row1, textvariable=self.deconv_care_axes,
                      values=["YX", "ZYX", "CYX", "CZYX"], width=6).pack(
            side=tk.LEFT, padx=(0, 15))
        ttk.Label(care_row1, text="(subfolder name inside model directory)",
                  foreground="gray").pack(side=tk.LEFT)

        care_row2 = ttk.Frame(self.deconv_care_frame)
        care_row2.pack(fill=tk.X, pady=2)
        ttk.Label(care_row2, text="Tile Y:").pack(side=tk.LEFT, padx=(0, 5))
        self.deconv_care_tile_y = tk.IntVar(value=1)
        ttk.Spinbox(care_row2, from_=1, to=16,
                      textvariable=self.deconv_care_tile_y, width=4).pack(
            side=tk.LEFT, padx=(0, 10))
        ttk.Label(care_row2, text="Tile X:").pack(side=tk.LEFT, padx=(0, 5))
        self.deconv_care_tile_x = tk.IntVar(value=1)
        ttk.Spinbox(care_row2, from_=1, to=16,
                      textvariable=self.deconv_care_tile_x, width=4).pack(
            side=tk.LEFT, padx=(0, 15))
        self.deconv_care_normalise = tk.BooleanVar(value=True)
        ttk.Checkbutton(care_row2, text="Percentile-normalise input",
                         variable=self.deconv_care_normalise).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(care_row2, text="pmin:").pack(side=tk.LEFT, padx=(0, 3))
        self.deconv_care_pmin = tk.DoubleVar(value=1.0)
        ttk.Entry(care_row2, textvariable=self.deconv_care_pmin, width=5).pack(
            side=tk.LEFT, padx=(0, 8))
        ttk.Label(care_row2, text="pmax:").pack(side=tk.LEFT, padx=(0, 3))
        self.deconv_care_pmax = tk.DoubleVar(value=99.8)
        ttk.Entry(care_row2, textvariable=self.deconv_care_pmax, width=5).pack(side=tk.LEFT)

        # -- Run / progress --
        btn_frame = ttk.Frame(body)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        self.btn_deconv_run = ttk.Button(btn_frame, text="Run Deconvolution",
                                          command=self._deconv_run)
        self.btn_deconv_run.pack(side=tk.LEFT, padx=5)

        self.deconv_progress = ttk.Progressbar(btn_frame, length=300, mode="determinate")
        self.deconv_progress.pack(side=tk.LEFT, padx=10)

        self.deconv_status = tk.StringVar(value="Ready")
        ttk.Label(btn_frame, textvariable=self.deconv_status).pack(side=tk.LEFT, padx=5)

        # -- Results table --
        tree_frame = ttk.Frame(body)
        tree_frame.pack(fill=tk.BOTH, expand=False, padx=10, pady=(0, 5))
        self.deconv_tree = ttk.Treeview(
            tree_frame, columns=("file", "status", "message"), show="headings", height=6)
        self.deconv_tree.heading("file", text="File")
        self.deconv_tree.heading("status", text="Status")
        self.deconv_tree.heading("message", text="Message")
        self.deconv_tree.column("file", width=250)
        self.deconv_tree.column("status", width=80)
        self.deconv_tree.column("message", width=350)
        self.deconv_tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        sb = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.deconv_tree.yview)
        self.deconv_tree.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        # -- Log --
        log_frame = ttk.LabelFrame(body, text="Log", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.deconv_log = scrolledtext.ScrolledText(
            log_frame, wrap=tk.WORD, height=8, state=tk.DISABLED, font=("Courier", 9))
        self.deconv_log.pack(fill=tk.BOTH, expand=True)

        # Show the correct parameter panel
        self._deconv_on_method_change()
        self._deconv_on_psf_change()

    # ---- Deconvolution helpers ----

    def _deconv_on_method_change(self):
        method = self.deconv_method.get()
        if method == "richardson_lucy":
            self.deconv_denoise_frame.pack(fill=tk.X, padx=10, pady=5,
                                            before=self.deconv_rl_frame)
            self.deconv_rl_frame.pack(fill=tk.X, padx=10, pady=5)
            self.deconv_care_frame.pack_forget()
        else:
            self.deconv_denoise_frame.pack_forget()
            self.deconv_rl_frame.pack_forget()
            self.deconv_care_frame.pack(fill=tk.X, padx=10, pady=5)

    def _deconv_on_modality_change(self):
        self._deconv_on_psf_change()

    def _deconv_on_psf_change(self):
        if self.deconv_psf_type.get() == "theoretical":
            self.deconv_psf_theo_frame.pack(fill=tk.X, pady=2)
            self.deconv_psf_theo_frame2.pack(fill=tk.X, pady=2)
            if self.deconv_modality.get() == "spinning_disc":
                self.deconv_psf_sd_frame.pack(fill=tk.X, pady=2)
            else:
                self.deconv_psf_sd_frame.pack_forget()
            self.deconv_psf_meas_frame.pack_forget()
        else:
            self.deconv_psf_meas_frame.pack(fill=tk.X, pady=2)
            self.deconv_psf_theo_frame.pack_forget()
            self.deconv_psf_theo_frame2.pack_forget()
            self.deconv_psf_sd_frame.pack_forget()

    def _deconv_browse_input(self):
        d = filedialog.askdirectory(title="Select Image Directory")
        if d:
            self.deconv_input_dir.set(d)

    def _deconv_browse_output(self):
        d = filedialog.askdirectory(title="Select Output Directory")
        if d:
            self.deconv_out_dir.set(d)

    def _deconv_browse_psf(self):
        path = filedialog.askopenfilename(
            title="Select Measured PSF File",
            filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")],
        )
        if path:
            self.deconv_psf_path.set(path)

    def _deconv_detect_channels(self):
        """Detect channels from OME-TIFF for deconvolution tab."""
        names = self._detect_ome_channels(self.deconv_input_dir)
        if names:
            self._deconv_ome_channel_names = names
            display = [f"{i}: {n}" for i, n in enumerate(names)]
            self.deconv_channel_combo["values"] = display
            self.deconv_channel_combo.current(0)
            self.deconv_channel.set(0)
            self._deconv_log_append(f"Detected channels: {names}")

    def _deconv_autodetect_metadata(self):
        """Read OME-TIFF metadata from the first image in the input dir
        and populate emission wavelength, NA, pixel size fields."""
        img_dir = self.deconv_input_dir.get()
        if not img_dir:
            messagebox.showwarning(
                "No image directory",
                "Set the image directory first, then click Auto-detect.")
            return

        try:
            from preprocessing.deconvolution import collect_tiffs, read_ome_metadata
        except ImportError as exc:
            self._deconv_log_append(f"[ERROR] Import failed: {exc}")
            return

        paths = collect_tiffs(img_dir)
        if not paths:
            messagebox.showinfo("No images", "No TIFF files found in the directory.")
            return

        meta = read_ome_metadata(paths[0])
        populated = []

        if meta.get("emission_nm"):
            self.deconv_wavelength_em.set(meta["emission_nm"])
            populated.append(f"emission={meta['emission_nm']} nm")
        if meta.get("na"):
            self.deconv_na.set(meta["na"])
            populated.append(f"NA={meta['na']}")
        if meta.get("pixel_size_nm"):
            self.deconv_pixel_size.set(round(meta["pixel_size_nm"], 2))
            populated.append(f"pixel={meta['pixel_size_nm']:.1f} nm")
        elif meta.get("pixel_size_um"):
            nm = meta["pixel_size_um"] * 1000.0
            self.deconv_pixel_size.set(round(nm, 2))
            populated.append(f"pixel={nm:.1f} nm")

        if meta.get("immersion_ri"):
            self.deconv_n_immersion.set(meta["immersion_ri"])
            populated.append(f"immersion RI={meta['immersion_ri']}")
        if meta.get("excitation_nm"):
            self.deconv_wavelength_ex.set(meta["excitation_nm"])
            populated.append(f"excitation={meta['excitation_nm']} nm")
        if meta.get("pinhole_um"):
            self.deconv_pinhole_um.set(meta["pinhole_um"])
            populated.append(f"pinhole={meta['pinhole_um']} \u00b5m")
            # If a pinhole is present, auto-select spinning disc modality
            self.deconv_modality.set("spinning_disc")
            self._deconv_on_modality_change()
            populated.append("modality=Spinning Disc")
        # NOTE: Do NOT auto-fill magnification — OME-TIFF pixel sizes are
        # already at the sample plane (PhysicalSizeX already accounts for
        # magnification).  Setting magnification here would double-correct,
        # making the PSF enormous.
        if meta.get("magnification"):
            self.deconv_magnification.set(0.0)  # keep at 0 (sample-plane px)

        if populated:
            summary = ", ".join(populated)
            self._deconv_log_append(
                f"[AUTO] From {paths[0].name}: {summary}")
            extra = []
            if meta.get("magnification"):
                extra.append(f"magnification={meta['magnification']}x (pixel size already at sample plane)")
            if meta.get("objective_name"):
                extra.append(f"objective={meta['objective_name']}")
            if meta.get("channel_name"):
                extra.append(f"channel={meta['channel_name']}")
            if extra:
                self._deconv_log_append(f"       Also found: {', '.join(extra)}")
        else:
            self._deconv_log_append(
                f"[AUTO] No optical metadata found in {paths[0].name}. "
                "Ensure images were converted with ND2 Conversion tab.")

    def _deconv_browse_care_model(self):
        d = filedialog.askdirectory(title="Select CARE Model Directory")
        if d:
            self.deconv_care_model_dir.set(d)

    def _deconv_log_append(self, msg):
        self.deconv_log.config(state=tk.NORMAL)
        self.deconv_log.insert(tk.END, msg + "\n")
        self.deconv_log.see(tk.END)
        self.deconv_log.config(state=tk.DISABLED)

    def _deconv_run(self):
        img_dir = self.deconv_input_dir.get()
        out_dir = self.deconv_out_dir.get()
        if not img_dir or not out_dir:
            messagebox.showwarning("Missing input",
                                    "Select both image and output directories.")
            return

        method = self.deconv_method.get()
        channel = self.deconv_channel.get()
        # Resolve channel by OME name if detected
        if self._deconv_ome_channel_names:
            sel = self.deconv_channel_combo.current()
            if 0 <= sel < len(self._deconv_ome_channel_names):
                channel = sel  # use OME-resolved index
        z_idx = self.deconv_z_index.get()

        self.btn_deconv_run.config(state=tk.DISABLED)
        self.deconv_tree.delete(*self.deconv_tree.get_children())
        self.deconv_progress.config(mode="determinate", value=0)
        self.deconv_status.set(f"Running {method} deconvolution...")
        self._deconv_log_append(f"Deconvolution: method={method}")

        def task():
            try:
                from preprocessing.deconvolution import (
                    deconvolve_richardson_lucy, deconvolve_care,
                )

                def _on_progress(idx, total, fname):
                    pct = int(100 * idx / total)
                    self.log_queue.put(f"__DECONV_PROGRESS__{pct}")
                    self.log_queue.put(f"__DECONV_LOG__  [{idx}/{total}] {fname}")

                if method == "richardson_lucy":
                    results = deconvolve_richardson_lucy(
                        image_dir=img_dir,
                        out_dir=out_dir,
                        psf_type=self.deconv_psf_type.get(),
                        psf_path=self.deconv_psf_path.get(),
                        modality=self.deconv_modality.get(),
                        wavelength_em=self.deconv_wavelength_em.get(),
                        wavelength_ex=self.deconv_wavelength_ex.get(),
                        na=self.deconv_na.get(),
                        pixel_size_nm=self.deconv_pixel_size.get(),
                        n_immersion=self.deconv_n_immersion.get(),
                        magnification=self.deconv_magnification.get(),
                        pinhole_um=self.deconv_pinhole_um.get(),
                        psf_size=self.deconv_psf_size.get(),
                        iterations=self.deconv_iterations.get(),
                        tv_lambda=self.deconv_tv_lambda.get(),
                        early_stop_delta=self.deconv_early_stop.get(),
                        subtract_background=self.deconv_subtract_bg.get(),
                        predenoise_enabled=self.deconv_denoise_enabled.get(),
                        predenoise_method=self.deconv_denoise_method.get(),
                        predenoise_sigma=self.deconv_denoise_sigma.get(),
                        channel_index=channel,
                        z_index=z_idx,
                        progress_callback=_on_progress,
                    )
                else:
                    n_tiles = (self.deconv_care_tile_y.get(),
                               self.deconv_care_tile_x.get())
                    results = deconvolve_care(
                        image_dir=img_dir,
                        out_dir=out_dir,
                        model_dir=self.deconv_care_model_dir.get(),
                        model_name=self.deconv_care_model_name.get(),
                        axes=self.deconv_care_axes.get(),
                        n_tiles=n_tiles,
                        channel_index=channel,
                        z_index=z_idx,
                        normalise_input=self.deconv_care_normalise.get(),
                        norm_pmin=self.deconv_care_pmin.get(),
                        norm_pmax=self.deconv_care_pmax.get(),
                        progress_callback=_on_progress,
                    )

                for r in results:
                    self.log_queue.put(
                        f"__DECONV_RESULT__{r['file']}||{r['status']}||{r['message']}"
                    )

                self.log_queue.put("__DECONV_DONE__")
            except Exception as exc:
                import traceback
                self.log_queue.put(
                    f"__DECONV_ERROR__{exc}\n{traceback.format_exc()}"
                )

        threading.Thread(target=task, daemon=True).start()

    def _on_deconv_finished(self, error=None):
        self.btn_deconv_run.config(state=tk.NORMAL)
        self.deconv_progress.config(value=100)
        if error:
            self.deconv_status.set(f"Error: {str(error)[:80]}")
            self._deconv_log_append(f"[ERROR] {error}")
        else:
            self.deconv_status.set("Deconvolution complete")
            self._deconv_log_append("[DONE] Deconvolution complete.")

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

        # Detect channels button
        self._seg_ome_channel_names = []
        ttk.Button(norm_frame, text="Detect channels from image",
                   command=self._seg_detect_channels).grid(
            row=0, column=0, columnspan=4, sticky=tk.W, pady=(0, 5))

        # Segment channel with label
        ttk.Label(norm_frame, text="Segment Channel:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.seg_channel = tk.IntVar(value=0)
        seg_combo = ttk.Combobox(
            norm_frame, textvariable=self.seg_channel, values=[0, 1, 2, 3],
            width=5, state="readonly",
        )
        seg_combo.grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)
        self.seg_ch_label = tk.StringVar(value="DIC")
        ttk.Label(norm_frame, textvariable=self.seg_ch_label, foreground="gray").grid(
            row=1, column=2, sticky=tk.W, padx=5
        )
        seg_combo.bind("<<ComboboxSelected>>", self._seg_update_channel_labels)
        self._seg_combo = seg_combo  # keep reference for updating values

        # Nuclear channel (for dual-channel input)
        ttk.Label(norm_frame, text="Nuclear Channel:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.seg_nuc_channel = tk.IntVar(value=0)
        nuc_combo = ttk.Combobox(
            norm_frame, textvariable=self.seg_nuc_channel, values=[0, 1, 2, 3],
            width=5, state="readonly",
        )
        nuc_combo.grid(row=2, column=1, padx=5, pady=2, sticky=tk.W)
        self.seg_nuc_label = tk.StringVar(value="None (grayscale)")
        ttk.Label(norm_frame, textvariable=self.seg_nuc_label, foreground="gray").grid(
            row=2, column=2, sticky=tk.W, padx=5
        )
        self._seg_nuc_combo = nuc_combo  # keep reference for updating values
        nuc_combo.bind("<<ComboboxSelected>>", self._seg_update_channel_labels)

        # Percentile range
        ttk.Label(norm_frame, text="Normalization Percentile:").grid(row=3, column=0, sticky=tk.W, pady=2)
        pct_frame = ttk.Frame(norm_frame)
        pct_frame.grid(row=3, column=1, columnspan=2, sticky=tk.W, padx=5)
        ttk.Label(pct_frame, text="Low:").pack(side=tk.LEFT)
        self.seg_lower_pct = tk.DoubleVar(value=1.0)
        ttk.Entry(pct_frame, textvariable=self.seg_lower_pct, width=6).pack(side=tk.LEFT, padx=(2, 10))
        ttk.Label(pct_frame, text="High:").pack(side=tk.LEFT)
        self.seg_upper_pct = tk.DoubleVar(value=99.0)
        ttk.Entry(pct_frame, textvariable=self.seg_upper_pct, width=6).pack(side=tk.LEFT, padx=2)

        # DIC tile blocksize
        ttk.Label(norm_frame, text="DIC Tile Blocksize:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.seg_tile_bs = tk.IntVar(value=128)
        ttk.Entry(norm_frame, textvariable=self.seg_tile_bs, width=8).grid(
            row=4, column=1, padx=5, pady=2, sticky=tk.W
        )
        ttk.Label(norm_frame, text="(0 = global, >0 = tile-based for uneven illumination)",
                  foreground="gray").grid(row=4, column=2, sticky=tk.W, padx=5)

        # Z-slice
        ttk.Label(norm_frame, text="Z-Slice:").grid(row=5, column=0, sticky=tk.W, pady=2)
        z_frame = ttk.Frame(norm_frame)
        z_frame.grid(row=5, column=1, columnspan=2, sticky=tk.W, padx=5)
        self.seg_z_idx = tk.StringVar(value="0")
        ttk.Entry(z_frame, textvariable=self.seg_z_idx, width=6).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(z_frame, text="(0-indexed)", foreground="gray").pack(side=tk.LEFT)

        # Invert DIC + DIC normalization checkboxes
        chk_frame = ttk.Frame(norm_frame)
        chk_frame.grid(row=6, column=0, columnspan=3, sticky=tk.W, pady=2)

        self.seg_dic_norm_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(chk_frame, text="DIC normalization (CLAHE)",
                        variable=self.seg_dic_norm_var).pack(side=tk.LEFT, padx=(0, 15))

        self.seg_invert_dic = tk.BooleanVar(value=False)
        ttk.Checkbutton(chk_frame, text="Invert DIC (cells dark on bright background)",
                        variable=self.seg_invert_dic).pack(side=tk.LEFT)

        # CLAHE clip limit
        ttk.Label(norm_frame, text="CLAHE Clip Limit:").grid(row=7, column=0, sticky=tk.W, pady=2)
        self.seg_clahe_clip = tk.DoubleVar(value=0.02)
        clahe_frame = ttk.Frame(norm_frame)
        clahe_frame.grid(row=7, column=1, columnspan=2, sticky=tk.W, padx=5)
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

        # Row 3: multi-scale pass checkboxes & GPU count
        row3 = ttk.Frame(cp_frame)
        row3.pack(fill=tk.X, pady=2)

        ttk.Label(row3, text="Multi-scale passes:").pack(side=tk.LEFT, padx=(0, 5))
        self.seg_scale_05x = tk.BooleanVar(value=True)
        ttk.Checkbutton(row3, text="0.5x", variable=self.seg_scale_05x).pack(side=tk.LEFT, padx=2)
        self.seg_scale_1x = tk.BooleanVar(value=True)
        ttk.Checkbutton(row3, text="1x", variable=self.seg_scale_1x).pack(side=tk.LEFT, padx=2)
        self.seg_scale_2x = tk.BooleanVar(value=True)
        ttk.Checkbutton(row3, text="2x", variable=self.seg_scale_2x).pack(side=tk.LEFT, padx=2)
        self.seg_scale_4x = tk.BooleanVar(value=True)
        ttk.Checkbutton(row3, text="4x", variable=self.seg_scale_4x).pack(side=tk.LEFT, padx=2)
        ttk.Label(row3, text="(used with auto-multi diameter)",
                  foreground="gray").pack(side=tk.LEFT, padx=(5, 15))

        ttk.Label(row3, text="GPUs:").pack(side=tk.LEFT, padx=(0, 5))
        self.seg_num_gpus = tk.IntVar(value=1)
        ttk.Spinbox(row3, from_=1, to=8, textvariable=self.seg_num_gpus,
                    width=3).pack(side=tk.LEFT)

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

    def _seg_detect_channels(self):
        """Detect channels from OME-TIFF for cell segmentation tab."""
        names = self._detect_ome_channels(self.seg_input_dir)
        if names:
            self._seg_ome_channel_names = names
            # Update _CHANNEL_NAMES to reflect actual OME names
            self._CHANNEL_NAMES = {i: n for i, n in enumerate(names)}
            # Update combobox values
            values = list(range(len(names)))
            self._seg_combo["values"] = values
            self._seg_nuc_combo["values"] = values
            self._seg_update_channel_labels()
            self._seg_log_append(f"Detected channels: {names}")

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
            self.seg_diameter.set("auto-multi")
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
        diam_str = self.seg_diameter.get().strip()
        channel = self.seg_channel.get()
        # Resolve channel name from OME detection
        seg_ch_name = None
        if self._seg_ome_channel_names and channel < len(self._seg_ome_channel_names):
            seg_ch_name = self._seg_ome_channel_names[channel]
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
        num_gpus = self.seg_num_gpus.get()

        # Build scale factors tuple from checkboxes
        scale_factors = []
        if self.seg_scale_05x.get():
            scale_factors.append(0.5)
        if self.seg_scale_1x.get():
            scale_factors.append(1.0)
        if self.seg_scale_2x.get():
            scale_factors.append(2.0)
        if self.seg_scale_4x.get():
            scale_factors.append(4.0)
        if not scale_factors:
            scale_factors = [1.0]  # fallback
        scale_factors = tuple(scale_factors)

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
            f"Segmentation ({mode}): model={model_type}, diameter={diam_str}, "
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
                    load_cellpose_model, run_cellpose_multipass,
                    parse_diameters,
                    postprocess_mask,
                    save_mask, save_seg_npy, save_triptych,
                    save_cytoplasm_triptych,
                    collect_image_paths,
                    compute_cytoplasm_mask,
                )
                import numpy as np
                import tifffile as tiff_io
                from concurrent.futures import ThreadPoolExecutor, as_completed
                try:
                    import torch
                except ImportError:
                    torch = None

                image_paths = collect_image_paths(input_path)
                if not image_paths:
                    self.log_queue.put("__SEG_ERROR__No TIFF images found.")
                    return

                # Determine how many GPUs to use
                effective_gpus = 1
                if gpu and num_gpus > 1 and torch is not None:
                    available = torch.cuda.device_count() if torch.cuda.is_available() else 0
                    effective_gpus = min(num_gpus, available) if available > 1 else 1

                self._seg_log_append_q(
                    f"Found {len(image_paths)} image(s). "
                    f"Loading model ({model_type}) on {effective_gpus} GPU(s)..."
                )

                # Load one model per GPU
                models = []
                for gpu_idx in range(effective_gpus):
                    if effective_gpus > 1:
                        torch.cuda.set_device(gpu_idx)
                    m = load_cellpose_model(gpu=gpu, model_type=model_type)
                    models.append(m)

                outdir = Path(out_dir)
                outdir.mkdir(parents=True, exist_ok=True)
                trip_dir = outdir / "triptychs"
                trip_dir.mkdir(parents=True, exist_ok=True)
                total = len(image_paths)
                completed_count = [0]  # mutable counter for progress

                def process_image(img_path, gpu_model, gpu_idx):
                    """Process a single image on the assigned GPU."""
                    if effective_gpus > 1:
                        torch.cuda.set_device(gpu_idx)
                    n_objects = -1
                    try:
                        img2d = load_image_2d(img_path, channel_index=channel, z_index=z,
                                             channel_name=seg_ch_name)
                        if use_dic_norm:
                            img_norm = normalize_dic(img2d, clip_limit=clahe_clip)
                        else:
                            img_norm = auto_lut_clip(img2d)

                        diameters = parse_diameters(diam_str)
                        masks, flows = run_cellpose_multipass(
                            img_norm, model=gpu_model, diameters=diameters,
                            flow_threshold=flow_thr,
                            cellprob_threshold=cellprob_thr,
                            augment=augment,
                            auto_scale_factors=scale_factors,
                        )
                        edge_thr = 0.25 if use_dic_norm else 0.0
                        masks = postprocess_mask(
                            masks, min_size=min_sz, max_size=max_sz,
                            remove_edges=rm_edges, smooth_radius=smooth_r,
                            edge_thresh=edge_thr, min_solidity=min_sol,
                        )
                        n_objects = int(masks.max())
                        stem = img_path.stem
                        diam0 = diameters[0] if isinstance(diameters, list) else None

                        if is_cyto:
                            nuc_path = self._find_nuc_mask(nuc_mask_dir, stem)
                            if nuc_path is None:
                                self._seg_log_append_q(f"    [WARN] No nucleus mask for {stem}")
                                save_mask(masks, outdir / f"{stem}_cell_masks.tif")
                                save_seg_npy(img_norm, masks, flows, img_path.name, img_path.parent, diam0)
                                save_triptych(img_norm, masks, trip_dir / f"{stem}_cell_triptych.png")
                            else:
                                nuc_m = ensure_2d(tiff_io.imread(str(nuc_path)))
                                if nuc_m.shape != masks.shape:
                                    self._seg_log_append_q(f"    [WARN] Shape mismatch, skipping cyto")
                                    save_mask(masks, outdir / f"{stem}_cell_masks.tif")
                                    save_seg_npy(img_norm, masks, flows, img_path.name, img_path.parent, diam0)
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
                                    save_seg_npy(img_norm, masks, flows, img_path.name, img_path.parent, diam0)
                                    save_seg_npy(img_norm, cyto_mask, flows, f"{stem}_cyto", outdir, diam0)
                                    save_cytoplasm_triptych(
                                        img_norm, masks, nuc_m, cyto_mask,
                                        trip_dir / f"{stem}_cyto_triptych.png"
                                    )
                        else:
                            save_mask(masks, outdir / f"{stem}_cyto3_masks.tif")
                            save_seg_npy(img_norm, masks, flows, img_path.name, img_path.parent, diam0)
                            save_triptych(img_norm, masks, trip_dir / f"{stem}_triptych.png")

                        return img_path, n_objects, "OK" if n_objects > 0 else "No signal"
                    except Exception as e:
                        self._seg_log_append_q(f"    [ERROR] {e}")
                        return img_path, -1, "FAILED"

                if effective_gpus <= 1:
                    # Single GPU — sequential processing (preserves log order)
                    for i, img_path in enumerate(image_paths, 1):
                        self._seg_log_append_q(f"  [{i}/{total}] {img_path.name}")
                        self.log_queue.put(f"__SEG_PROGRESS__{int(100 * i / total)}")
                        img_path, n_objects, status = process_image(img_path, models[0], 0)
                        self.log_queue.put(f"__SEG_RESULT__{img_path.name}||{n_objects}||{status}")
                else:
                    # Multi-GPU — parallel processing
                    self._seg_log_append_q(f"Running parallel on {effective_gpus} GPUs...")
                    futures = {}
                    with ThreadPoolExecutor(max_workers=effective_gpus) as executor:
                        for i, img_path in enumerate(image_paths):
                            gpu_idx = i % effective_gpus
                            fut = executor.submit(process_image, img_path, models[gpu_idx], gpu_idx)
                            futures[fut] = (i, img_path)

                        for fut in as_completed(futures):
                            idx, orig_path = futures[fut]
                            completed_count[0] += 1
                            pct = int(100 * completed_count[0] / total)
                            self.log_queue.put(f"__SEG_PROGRESS__{pct}")
                            img_path, n_objects, status = fut.result()
                            self._seg_log_append_q(
                                f"  [{completed_count[0]}/{total}] {img_path.name} -> {status}"
                            )
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

        self._pseg_ome_channel_names = []
        ttk.Button(ch_frame, text="Detect channels from image",
                   command=self._pseg_detect_channels).grid(
            row=0, column=0, columnspan=4, sticky=tk.W, pady=(0, 5))

        ttk.Label(ch_frame, text="Puncta Channel:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.pseg_channel = tk.IntVar(value=1)
        self._pseg_ch_combo = ttk.Combobox(
            ch_frame, textvariable=self.pseg_channel, values=[0, 1, 2, 3],
            width=5, state="readonly")
        self._pseg_ch_combo.grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)
        self.pseg_ch_label = tk.StringVar(value="mEGFP")
        ttk.Label(ch_frame, textvariable=self.pseg_ch_label, foreground="gray").grid(
            row=1, column=2, sticky=tk.W, padx=5)
        self._pseg_ch_combo.bind(
            "<<ComboboxSelected>>",
            lambda e: self.pseg_ch_label.set(
                self._pseg_ome_channel_names[self.pseg_channel.get()]
                if self._pseg_ome_channel_names and self.pseg_channel.get() < len(self._pseg_ome_channel_names)
                else f"Channel {self.pseg_channel.get()}"))

        ttk.Label(ch_frame, text="Z-Slice:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.pseg_z_idx = tk.IntVar(value=0)
        ttk.Entry(ch_frame, textvariable=self.pseg_z_idx, width=6).grid(row=2, column=1, padx=5, pady=2, sticky=tk.W)

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

    def _pseg_detect_channels(self):
        """Detect channels from OME-TIFF for puncta segmentation tab."""
        names = self._detect_ome_channels(self.pseg_input_dir)
        if names:
            self._pseg_ome_channel_names = names
            self._pseg_ch_combo["values"] = list(range(len(names)))
            # Auto-select GFP if found
            for i, n in enumerate(names):
                if any(kw in n.lower() for kw in ("gfp", "egfp", "green")):
                    self.pseg_channel.set(i)
                    self.pseg_ch_label.set(n)
                    break
            else:
                self.pseg_ch_label.set(names[self.pseg_channel.get()]
                                       if self.pseg_channel.get() < len(names)
                                       else f"Channel {self.pseg_channel.get()}")
            self._pseg_log_append(f"Detected channels: {names}")

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
        # Resolve channel name from OME detection
        pseg_ch_name = None
        if self._pseg_ome_channel_names and channel < len(self._pseg_ome_channel_names):
            pseg_ch_name = self._pseg_ome_channel_names[channel]
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
                    channel_name=pseg_ch_name,
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
        bench_ch_name = None
        if self._pseg_ome_channel_names and channel < len(self._pseg_ome_channel_names):
            bench_ch_name = self._pseg_ome_channel_names[channel]
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
                        channel_name=bench_ch_name,
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
    # ==================================================================
    # Phase Separation Analysis tab
    # ==================================================================

    def _build_analysis_tab(self):
        tab = self.tab_analysis

        # Use a canvas+scrollbar so the tab is scrollable
        canvas = tk.Canvas(tab, highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, orient=tk.VERTICAL, command=canvas.yview)
        self._ana_scroll_frame = ttk.Frame(canvas)
        self._ana_scroll_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self._ana_scroll_frame, anchor=tk.NW)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(-1 * (e.delta // 120), "units"))
        body = self._ana_scroll_frame

        # ---- Description ----
        info = ttk.Label(
            body,
            text="Estimate the critical concentration (Csat) for protein phase separation\n"
                 "from mEGFP fluorescence images using pre-computed masks.\n"
                 "Provide directories of images and masks -- files are auto-matched by name.\n\n"
                 "Method 1 (Binary): P(puncta present) vs cytoplasmic intensity -> Csat at P=0.5\n"
                 "Method 2 (Intensity): Puncta sum intensity vs avg cytoplasm/nucleus intensity -> sigmoid midpoint",
            foreground="gray",
        )
        info.pack(anchor=tk.W, padx=10, pady=(10, 5))

        # ---- Input Directories ----
        io_frame = ttk.LabelFrame(body, text="Input Directories", padding=10)
        io_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(io_frame, text="Fluorescence Images (OME-TIFF):").grid(
            row=0, column=0, sticky=tk.W, pady=2)
        self.ana_image_dir = tk.StringVar()
        ttk.Entry(io_frame, textvariable=self.ana_image_dir, width=55).grid(
            row=0, column=1, padx=5, pady=2)
        ttk.Button(io_frame, text="Browse...",
                   command=lambda: self._browse_dir(self.ana_image_dir)).grid(
            row=0, column=2, pady=2)

        ttk.Label(io_frame, text="Cell Masks (DIC or mScarlet):").grid(
            row=1, column=0, sticky=tk.W, pady=2)
        self.ana_cell_mask_dir = tk.StringVar()
        ttk.Entry(io_frame, textvariable=self.ana_cell_mask_dir, width=55).grid(
            row=1, column=1, padx=5, pady=2)
        ttk.Button(io_frame, text="Browse...",
                   command=lambda: self._browse_dir(self.ana_cell_mask_dir)).grid(
            row=1, column=2, pady=2)

        ttk.Label(io_frame, text="Puncta Masks (mEGFP):").grid(
            row=2, column=0, sticky=tk.W, pady=2)
        self.ana_puncta_mask_dir = tk.StringVar()
        ttk.Entry(io_frame, textvariable=self.ana_puncta_mask_dir, width=55).grid(
            row=2, column=1, padx=5, pady=2)
        ttk.Button(io_frame, text="Browse...",
                   command=lambda: self._browse_dir(self.ana_puncta_mask_dir)).grid(
            row=2, column=2, pady=2)

        ttk.Label(io_frame, text="Nucleus Masks (optional):").grid(
            row=3, column=0, sticky=tk.W, pady=2)
        self.ana_nuc_mask_dir = tk.StringVar()
        ttk.Entry(io_frame, textvariable=self.ana_nuc_mask_dir, width=55).grid(
            row=3, column=1, padx=5, pady=2)
        ttk.Button(io_frame, text="Browse...",
                   command=lambda: self._browse_dir(self.ana_nuc_mask_dir)).grid(
            row=3, column=2, pady=2)

        # ---- Parameters ----
        param_frame = ttk.LabelFrame(body, text="Parameters", padding=10)
        param_frame.pack(fill=tk.X, padx=10, pady=5)

        p_row0 = ttk.Frame(param_frame)
        p_row0.pack(fill=tk.X, pady=2)

        ttk.Label(p_row0, text="Intensity channel:").pack(side=tk.LEFT, padx=(0, 5))
        self.ana_fluor_ch = tk.IntVar(value=1)
        self.ana_fluor_ch_name = tk.StringVar(value="")
        self.ana_channel_combo = ttk.Combobox(
            p_row0, textvariable=self.ana_fluor_ch_name, width=22, state="readonly")
        self.ana_channel_combo.pack(side=tk.LEFT, padx=(0, 5))
        self.ana_channel_combo.bind("<<ComboboxSelected>>", self._ana_on_channel_select)
        self._ana_ome_channel_names = []  # populated by detect
        ttk.Button(p_row0, text="Detect from image",
                   command=self._ana_detect_channels).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(p_row0, text="or index:").pack(side=tk.LEFT, padx=(0, 3))
        ttk.Entry(p_row0, textvariable=self.ana_fluor_ch, width=4).pack(side=tk.LEFT, padx=(0, 15))

        ttk.Label(p_row0, text="Outlier Z-threshold:").pack(side=tk.LEFT, padx=(0, 5))
        self.ana_z_thresh = tk.DoubleVar(value=3.0)
        ttk.Entry(p_row0, textvariable=self.ana_z_thresh, width=5).pack(side=tk.LEFT, padx=(0, 15))

        ttk.Label(p_row0, text="Bootstrap iterations:").pack(side=tk.LEFT, padx=(0, 5))
        self.ana_n_bootstrap = tk.IntVar(value=1000)
        ttk.Entry(p_row0, textvariable=self.ana_n_bootstrap, width=6).pack(side=tk.LEFT)

        # DropFit bins
        ttk.Label(p_row0, text="   DropFit bins:").pack(side=tk.LEFT, padx=(10, 5))
        self.ana_dropfit_bins = tk.IntVar(value=20)
        ttk.Entry(p_row0, textvariable=self.ana_dropfit_bins, width=4).pack(side=tk.LEFT)

        # Method 2 X-axis selection
        p_row1 = ttk.Frame(param_frame)
        p_row1.pack(fill=tk.X, pady=2)

        ttk.Label(p_row1, text="Method 2 X-axis:").pack(side=tk.LEFT, padx=(0, 5))
        self.ana_method2_xaxis = tk.StringVar(value="cytoplasm")
        ttk.Radiobutton(p_row1, text="Cytoplasm (Cell - Nucleus)",
                        variable=self.ana_method2_xaxis,
                        value="cytoplasm").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(p_row1, text="Nucleus",
                        variable=self.ana_method2_xaxis,
                        value="nucleus").pack(side=tk.LEFT)

        # ---- Action buttons ----
        btn_frame = ttk.LabelFrame(body, text="Actions", padding=10)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)

        self.btn_ana_measure = ttk.Button(
            btn_frame, text="1. Extract Measurements (Bulk)",
            command=self._ana_run_measurement)
        self.btn_ana_measure.pack(side=tk.LEFT, padx=5)

        self.btn_ana_csat = ttk.Button(
            btn_frame, text="2. Estimate Csat (Both Methods)",
            command=self._ana_run_csat, state=tk.DISABLED)
        self.btn_ana_csat.pack(side=tk.LEFT, padx=5)

        self.btn_ana_dropfit = ttk.Button(
            btn_frame, text="3. DropFit Csat Analysis",
            command=self._ana_run_dropfit, state=tk.DISABLED)
        self.btn_ana_dropfit.pack(side=tk.LEFT, padx=5)

        self.btn_ana_export = ttk.Button(
            btn_frame, text="4. Export CSV",
            command=self._ana_export_csv, state=tk.DISABLED)
        self.btn_ana_export.pack(side=tk.LEFT, padx=5)

        # ---- Progress ----
        self.ana_progress = ttk.Progressbar(body, mode="determinate")
        self.ana_progress.pack(fill=tk.X, padx=10, pady=5)
        self.ana_status = tk.StringVar(value="Ready")
        ttk.Label(body, textvariable=self.ana_status).pack(padx=10, anchor=tk.W)

        # ---- Results summary ----
        summary_frame = ttk.LabelFrame(body, text="Results Summary", padding=10)
        summary_frame.pack(fill=tk.X, padx=10, pady=5)
        self.ana_summary_text = tk.StringVar(
            value="Run measurement extraction and Csat estimation to see results here."
        )
        ttk.Label(summary_frame, textvariable=self.ana_summary_text,
                  justify=tk.LEFT, wraplength=700).pack(anchor=tk.W)

        # ---- Plot panels (3x2 grid) ----
        plot_frame = ttk.LabelFrame(body, text="Visualization", padding=5)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self._ana_plot_frames = {}
        for r, c, title in [
            (0, 0, "Overlay"),
            (0, 1, "Intensity Histogram"),
            (1, 0, "Intensity vs Puncta Count"),
            (1, 1, "Method 1: Binary Phase Transition"),
            (2, 0, "Method 2: Intensity Sigmoid"),
            (2, 1, "DropFit: Probability vs Intensity"),
            (3, 0, "DropFit: Size Distribution"),
        ]:
            f = ttk.LabelFrame(plot_frame, text=title, padding=2)
            f.grid(row=r, column=c, padx=4, pady=4, sticky="nsew")
            self._ana_plot_frames[title] = f
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.columnconfigure(1, weight=1)
        for r in range(4):
            plot_frame.rowconfigure(r, weight=1)

        # ---- Log ----
        log_frame = ttk.LabelFrame(body, text="Log", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))
        self.ana_log = scrolledtext.ScrolledText(
            log_frame, wrap=tk.WORD, height=6, state=tk.DISABLED, font=("Courier", 9)
        )
        self.ana_log.pack(fill=tk.BOTH, expand=True)

        # Internal state
        self._ana_df = None
        self._ana_df_clean = None
        self._ana_result_m1 = None
        self._ana_result_m2 = None
        self._ana_result_dropfit = None

    def _ana_browse_csv(self):
        path = filedialog.asksaveasfilename(
            title="Save CSV As",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            defaultextension=".csv",
        )
        if path:
            return path
        return None

    def _ana_log_append(self, msg):
        self.ana_log.config(state=tk.NORMAL)
        self.ana_log.insert(tk.END, msg + "\n")
        self.ana_log.see(tk.END)
        self.ana_log.config(state=tk.DISABLED)

    def _ana_log_append_q(self, msg):
        """Thread-safe log append via queue."""
        self.log_queue.put(f"__ANA_LOG__{msg}")

    # ---- Embed matplotlib figure in a tk frame ----
    def _ana_embed_figure(self, fig, parent_frame):
        """Embed a matplotlib Figure into a ttk frame, replacing old content."""
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        for w in parent_frame.winfo_children():
            w.destroy()
        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ---- Channel detection helpers ----
    def _ana_detect_channels(self):
        """Read OME-TIFF channel names from the first image in the image dir."""
        names = self._detect_ome_channels(self.ana_image_dir)
        if not names:
            return

        self._ana_ome_channel_names = names
        display_values = [f"{i}: {n}" for i, n in enumerate(names)]
        self.ana_channel_combo["values"] = display_values

        # Auto-select GFP/EGFP/mEGFP if found
        for i, n in enumerate(names):
            if any(kw in n.lower() for kw in ("gfp", "egfp", "green")):
                self.ana_channel_combo.current(i)
                self.ana_fluor_ch.set(i)
                self.ana_fluor_ch_name.set(display_values[i])
                break
        else:
            self.ana_channel_combo.current(0)
            self.ana_fluor_ch.set(0)

        self._ana_log_append(f"Detected channels: {names}")

    def _ana_on_channel_select(self, event=None):
        """Update channel index when user selects from dropdown."""
        sel = self.ana_channel_combo.current()
        if sel >= 0:
            self.ana_fluor_ch.set(sel)

    # ---- Run Bulk Measurement Extraction ----
    def _ana_run_measurement(self):
        img_dir = self.ana_image_dir.get()
        cell_dir = self.ana_cell_mask_dir.get()
        puncta_dir = self.ana_puncta_mask_dir.get()
        if not img_dir or not cell_dir or not puncta_dir:
            messagebox.showwarning(
                "Missing input",
                "Please provide all required directories:\n"
                "- Fluorescence images folder\n"
                "- Cell masks folder\n"
                "- Puncta masks folder")
            return

        fluor_ch = self.ana_fluor_ch.get()  # 0-based index (fallback)
        # Extract channel name from OME-detected list if available
        fluor_ch_name = None
        if self._ana_ome_channel_names:
            sel_idx = self.ana_channel_combo.current()
            if 0 <= sel_idx < len(self._ana_ome_channel_names):
                fluor_ch_name = self._ana_ome_channel_names[sel_idx]
        nuc_dir = self.ana_nuc_mask_dir.get() or None

        self.btn_ana_measure.config(state=tk.DISABLED)
        self.btn_ana_csat.config(state=tk.DISABLED)
        self.btn_ana_dropfit.config(state=tk.DISABLED)
        self.btn_ana_export.config(state=tk.DISABLED)
        self.ana_progress.config(value=0)
        self.ana_status.set("Matching files and extracting measurements...")
        self._ana_log_append("Starting bulk extraction...")

        def task():
            try:
                from phase_separation import (
                    match_files, extract_bulk, plot_overlay,
                )

                # Match files across directories
                matched, warnings = match_files(
                    img_dir, cell_dir, puncta_dir, nuc_dir)

                if warnings:
                    for w in warnings:
                        self._ana_log_append_q(f"  [WARN] Unmatched: {w}")

                if not matched:
                    self.log_queue.put(
                        "__ANA_ERROR__No matching files found across directories. "
                        "Check that file names match between folders.")
                    return

                self._ana_log_append_q(
                    f"Matched {len(matched)} image sets across directories")

                def _on_progress(current, total):
                    pct = int(100 * current / total) if total > 0 else 0
                    self.log_queue.put(f"__ANA_PROGRESS__{pct}")

                df_all, last_overlay = extract_bulk(
                    matched_files=matched,
                    channel_index=fluor_ch,
                    channel_name=fluor_ch_name,
                    log_callback=self._ana_log_append_q,
                    progress_callback=_on_progress,
                )

                self._ana_df = df_all

                n_total = len(df_all)
                n_images = df_all["image"].nunique() if n_total > 0 else 0
                n_with = int(df_all["puncta_present"].sum()) if n_total > 0 else 0
                self._ana_log_append_q(
                    f"Bulk extraction complete: {n_images} images, "
                    f"{n_total} cells, {n_with} with puncta "
                    f"({100*n_with/max(n_total,1):.1f}%)")

                # Overlay from last processed image
                if last_overlay is not None:
                    fluor, cmask, pmask = last_overlay
                    fig_overlay = plot_overlay(fluor, cmask, pmask)
                    self.log_queue.put(("__ANA_PLOT__", "Overlay", fig_overlay))

                self.log_queue.put("__ANA_MEASURE_DONE__")
            except Exception as exc:
                self.log_queue.put(f"__ANA_ERROR__{exc}")

        threading.Thread(target=task, daemon=True).start()

    # ---- Estimate Csat (both methods) ----
    def _ana_run_csat(self):
        if self._ana_df is None or len(self._ana_df) == 0:
            messagebox.showwarning("No data", "Run measurement extraction first.")
            return

        z_thresh = self.ana_z_thresh.get()
        n_boot = self.ana_n_bootstrap.get()
        m2_xaxis = self.ana_method2_xaxis.get()  # "cytoplasm" or "nucleus"

        # Determine Method 2 column
        if m2_xaxis == "nucleus":
            m2_col = "nucleus_mean_intensity"
            m2_label = "Nucleus"
        else:
            m2_col = "cytoplasm_mean_intensity"
            m2_label = "Cytoplasm"

        # Check nucleus data available if needed
        if m2_xaxis == "nucleus" and self._ana_df["nucleus_mean_intensity"].isna().all():
            messagebox.showwarning(
                "No nucleus data",
                "Nucleus masks were not provided during extraction.\n"
                "Please re-extract with nucleus masks, or select 'Cytoplasm' for Method 2.")
            return

        self.btn_ana_csat.config(state=tk.DISABLED)
        self.ana_status.set("Fitting Csat models (Methods 1 & 2)...")
        self._ana_log_append("Cleaning data and fitting both Csat methods...")

        def task():
            try:
                from phase_separation import (
                    clean_data, fit_logistic_binary, fit_sigmoid_intensity,
                    plot_intensity_histogram, plot_scatter_puncta_count,
                    plot_method1_phase_transition, plot_method2_sigmoid,
                )

                # Clean data (using cytoplasm col for Method 1)
                df_clean = clean_data(self._ana_df,
                                      intensity_col="cytoplasm_mean_intensity",
                                      z_threshold=z_thresh)
                self._ana_df_clean = df_clean
                n_removed = len(self._ana_df) - len(df_clean)
                self._ana_log_append_q(
                    f"Cleaned: {len(df_clean)} cells kept, {n_removed} removed")

                # --- Method 1: Binary logistic ---
                self._ana_log_append_q("Method 1: Fitting logistic P(puncta) ~ cyto intensity...")
                result_m1 = fit_logistic_binary(
                    df_clean, x_col="cytoplasm_mean_intensity",
                    n_bootstrap=n_boot)
                self._ana_result_m1 = result_m1

                if "error" in result_m1:
                    self._ana_log_append_q(f"  [WARN] Method 1: {result_m1['error']}")
                else:
                    self._ana_log_append_q(
                        f"  Method 1 Csat = {result_m1['csat']:.2f}  "
                        f"(95% CI: [{result_m1['ci_low']:.2f}, {result_m1['ci_high']:.2f}])")

                # --- Method 2: Sigmoid intensity ---
                self._ana_log_append_q(
                    f"Method 2: Fitting sigmoid puncta_sum ~ {m2_label} intensity...")
                result_m2 = fit_sigmoid_intensity(
                    df_clean, x_col=m2_col, n_bootstrap=n_boot)
                self._ana_result_m2 = result_m2

                if "error" in result_m2:
                    self._ana_log_append_q(f"  [WARN] Method 2: {result_m2['error']}")
                else:
                    self._ana_log_append_q(
                        f"  Method 2 Csat = {result_m2['csat']:.2f}  "
                        f"(95% CI: [{result_m2['ci_low']:.2f}, {result_m2['ci_high']:.2f}])")

                # --- Build summary ---
                lines = []
                n_images = df_clean["image"].nunique() if "image" in df_clean.columns else "?"
                lines.append("=" * 50)
                lines.append(f"Images processed: {n_images}  |  Cells analyzed: {len(df_clean)}")
                lines.append("")
                lines.append("METHOD 1: Binary Logistic Regression")
                lines.append(f"  P(puncta_present) vs Cytoplasmic Mean Intensity (mEGFP)")
                if "error" not in result_m1:
                    lines.append(f"  Csat = {result_m1['csat']:.2f}  "
                                 f"(95% CI: [{result_m1['ci_low']:.2f}, {result_m1['ci_high']:.2f}])")
                    lines.append(f"  Cells with puncta: {result_m1['n_with_puncta']}  |  "
                                 f"Without: {result_m1['n_no_puncta']}")
                else:
                    lines.append(f"  {result_m1['error']}")

                lines.append("")
                lines.append("METHOD 2: Sigmoid Intensity Fit")
                lines.append(f"  Puncta Sum Intensity vs {m2_label} Mean Intensity (mEGFP)")
                if "error" not in result_m2:
                    lines.append(f"  Csat = {result_m2['csat']:.2f}  "
                                 f"(95% CI: [{result_m2['ci_low']:.2f}, {result_m2['ci_high']:.2f}])")
                else:
                    lines.append(f"  {result_m2['error']}")
                lines.append("=" * 50)

                summary_text = "\n".join(lines)
                self.log_queue.put(f"__ANA_SUMMARY__{summary_text}")

                # --- Plots ---
                fig_hist = plot_intensity_histogram(df_clean)
                fig_scatter = plot_scatter_puncta_count(df_clean)
                fig_m1 = plot_method1_phase_transition(df_clean, result_m1)
                fig_m2 = plot_method2_sigmoid(df_clean, result_m2)

                self.log_queue.put(("__ANA_PLOT__", "Intensity Histogram", fig_hist))
                self.log_queue.put(("__ANA_PLOT__", "Intensity vs Puncta Count", fig_scatter))
                self.log_queue.put(("__ANA_PLOT__", "Method 1: Binary Phase Transition", fig_m1))
                self.log_queue.put(("__ANA_PLOT__", "Method 2: Intensity Sigmoid", fig_m2))

                self.log_queue.put("__ANA_CSTAR_DONE__")
            except Exception as exc:
                self.log_queue.put(f"__ANA_ERROR__{exc}")

        threading.Thread(target=task, daemon=True).start()

    # ---- DropFit Csat Analysis ----
    def _ana_run_dropfit(self):
        if self._ana_df is None or len(self._ana_df) == 0:
            messagebox.showwarning("No data", "Run measurement extraction first.")
            return

        z_thresh = self.ana_z_thresh.get()
        n_boot = self.ana_n_bootstrap.get()
        n_bins = self.ana_dropfit_bins.get()

        self.btn_ana_dropfit.config(state=tk.DISABLED)
        self.ana_status.set("Running DropFit Csat analysis...")
        self._ana_log_append("DropFit: Cleaning data and fitting logistic model...")

        def task():
            try:
                from phase_separation import (
                    clean_data, fit_dropfit_csat,
                    plot_dropfit_csat, plot_dropfit_size_distribution,
                )

                # Clean data
                df_clean = clean_data(self._ana_df,
                                      intensity_col="cytoplasm_mean_intensity",
                                      z_threshold=z_thresh)
                self._ana_df_clean = df_clean
                n_removed = len(self._ana_df) - len(df_clean)
                self._ana_log_append_q(
                    f"Cleaned: {len(df_clean)} cells kept, {n_removed} removed")

                # Fit DropFit Csat
                self._ana_log_append_q(
                    f"DropFit: Fitting logistic on {n_bins} bins, "
                    f"{n_boot} bootstrap iterations...")
                result_df = fit_dropfit_csat(
                    df_clean, x_col="cytoplasm_mean_intensity",
                    n_bins=n_bins, n_bootstrap=n_boot)
                self._ana_result_dropfit = result_df

                if "error" in result_df:
                    self._ana_log_append_q(
                        f"  [WARN] DropFit: {result_df['error']}")
                else:
                    self._ana_log_append_q(
                        f"  DropFit Csat = {result_df['csat']:.2f}  "
                        f"(95% CI: [{result_df['ci_low']:.2f}, "
                        f"{result_df['ci_high']:.2f}])")

                # Build summary
                lines = []
                n_images = (df_clean["image"].nunique()
                            if "image" in df_clean.columns else "?")
                lines.append("=" * 50)
                lines.append(f"DropFit Csat Analysis")
                lines.append(f"Images: {n_images}  |  "
                             f"Cells: {len(df_clean)}  |  "
                             f"Bins: {n_bins}")
                lines.append("")
                if "error" not in result_df:
                    lines.append(
                        f"Csat = {result_df['csat']:.2f}  "
                        f"(95% CI: [{result_df['ci_low']:.2f}, "
                        f"{result_df['ci_high']:.2f}])")
                    # Show bin summary
                    bin_df = result_df.get("bin_df")
                    if bin_df is not None and len(bin_df) > 0:
                        lines.append("")
                        lines.append("Per-bin summary (non-empty bins):")
                        for _, row in bin_df.iterrows():
                            lines.append(
                                f"  [{row['bin_low']:.0f}-{row['bin_high']:.0f}]  "
                                f"cells={row['total_cells']:.0f}  "
                                f"P={row['droplet_probability']:.2f}  "
                                f"mean_area={row['mean_droplet_area']:.1f}")
                else:
                    lines.append(f"  {result_df['error']}")
                lines.append("=" * 50)

                summary_text = "\n".join(lines)
                self.log_queue.put(f"__ANA_SUMMARY__{summary_text}")

                # Plots
                fig_prob = plot_dropfit_csat(df_clean, result_df)
                fig_size = plot_dropfit_size_distribution(df_clean, result_df)
                self.log_queue.put((
                    "__ANA_PLOT__",
                    "DropFit: Probability vs Intensity", fig_prob))
                self.log_queue.put((
                    "__ANA_PLOT__",
                    "DropFit: Size Distribution", fig_size))

                self.log_queue.put("__ANA_DROPFIT_DONE__")
            except Exception as exc:
                self.log_queue.put(f"__ANA_ERROR__{exc}")

        threading.Thread(target=task, daemon=True).start()

    # ---- Export CSV ----
    def _ana_export_csv(self):
        df = self._ana_df_clean if self._ana_df_clean is not None else self._ana_df
        if df is None or len(df) == 0:
            messagebox.showwarning("No data", "Run measurement extraction first.")
            return

        path = self._ana_browse_csv()
        if not path:
            return

        # --- Per-cell CSV ---
        cols = ["image", "cell_id", "cell_area", "cytoplasm_mean_intensity",
                "nucleus_mean_intensity", "cyto_dilute_mean_intensity",
                "nuc_dilute_mean_intensity", "total_cell_intensity",
                "puncta_present", "puncta_count", "puncta_total_area",
                "puncta_sum_intensity"]
        export_cols = [c for c in cols if c in df.columns]
        df[export_cols].to_csv(path, index=False)
        self._ana_log_append(f"Per-cell CSV exported: {path}  ({len(df)} rows)")

        # --- Per-image summary CSV ---
        summary_path = path.replace(".csv", "_summary.csv")
        if "image" in df.columns:
            summary_records = []
            for img_name, grp in df.groupby("image"):
                n_total = len(grp)
                n_with = int(grp["puncta_present"].sum())
                n_without = n_total - n_with
                summary_records.append({
                    "image": img_name,
                    "total_cells": n_total,
                    "cells_with_puncta": n_with,
                    "cells_without_puncta": n_without,
                })
            # Add grand total row
            grand_total = len(df)
            grand_with = int(df["puncta_present"].sum())
            summary_records.append({
                "image": "TOTAL",
                "total_cells": grand_total,
                "cells_with_puncta": grand_with,
                "cells_without_puncta": grand_total - grand_with,
            })
            import pandas as pd
            df_summary = pd.DataFrame(summary_records)
            df_summary.to_csv(summary_path, index=False)
            self._ana_log_append(f"Summary CSV exported: {summary_path}")

        # --- DropFit binned CSV (Prism-ready) ---
        if self._ana_result_dropfit is not None:
            bin_df = self._ana_result_dropfit.get("bin_df")
            if bin_df is not None and len(bin_df) > 0:
                import math
                dropfit_path = path.replace(".csv", "_dropfit_bins.csv")
                bin_df.to_csv(dropfit_path, index=False)
                self._ana_log_append(
                    f"DropFit binned CSV exported: {dropfit_path}  "
                    f"({len(bin_df)} bins)")
                csat = self._ana_result_dropfit.get("csat", float("nan"))
                ci_lo = self._ana_result_dropfit.get("ci_low", float("nan"))
                ci_hi = self._ana_result_dropfit.get("ci_high", float("nan"))
                if not math.isnan(csat):
                    self._ana_log_append(
                        f"  DropFit Csat = {csat:.2f}  "
                        f"(95% CI: [{ci_lo:.2f}, {ci_hi:.2f}])")

        self.ana_status.set(f"CSV exported to {Path(path).name}")

    # ---- Queue message handlers (called from _poll_log_queue) ----
    def _on_ana_finished(self, csv_path=None, error=None):
        """Handle analysis error messages."""
        self.btn_ana_measure.config(state=tk.NORMAL)
        self.btn_ana_csat.config(state=tk.NORMAL)
        self.btn_ana_dropfit.config(state=tk.NORMAL)
        self.btn_ana_export.config(state=tk.NORMAL)
        self.ana_progress.config(value=100)
        if error:
            self.ana_status.set(f"Error: {str(error)[:80]}")
            self._ana_log_append(f"[ERROR] {error}")
        else:
            self.ana_status.set("Complete")

    def _on_ana_measure_done(self):
        """Called when measurement extraction finishes."""
        self.btn_ana_measure.config(state=tk.NORMAL)
        self.btn_ana_csat.config(state=tk.NORMAL)
        self.btn_ana_dropfit.config(state=tk.NORMAL)
        self.btn_ana_export.config(state=tk.NORMAL)
        self.ana_progress.config(value=100)
        self.ana_status.set("Bulk extraction complete. Ready for Csat estimation.")

    def _on_ana_cstar_done(self):
        """Called when Csat estimation finishes."""
        self.btn_ana_csat.config(state=tk.NORMAL)
        self.btn_ana_dropfit.config(state=tk.NORMAL)
        self.btn_ana_export.config(state=tk.NORMAL)
        self.ana_progress.config(value=100)
        self.ana_status.set("Csat estimation complete (both methods).")

    def _on_ana_dropfit_done(self):
        """Called when DropFit Csat estimation finishes."""
        self.btn_ana_dropfit.config(state=tk.NORMAL)
        self.btn_ana_export.config(state=tk.NORMAL)
        self.ana_progress.config(value=100)
        self.ana_status.set("DropFit Csat estimation complete.")

    # ==================================================================
    # Helper: generic directory browse
    # ==================================================================
    def _browse_dir(self, string_var):
        d = filedialog.askdirectory()
        if d:
            string_var.set(d)

    def _detect_ome_channels(self, img_dir_var):
        """Read OME channel names from the first TIFF in a directory.

        Returns list of channel name strings, or empty list.
        """
        from pathlib import Path
        img_dir = img_dir_var.get() if hasattr(img_dir_var, 'get') else str(img_dir_var)
        if not img_dir:
            messagebox.showwarning("No image directory",
                                   "Set the image directory first.")
            return []

        # Import from segmentation_utils (shared across all tabs)
        try:
            from segmentation_utils import get_ome_channel_names
        except ImportError:
            from phase_separation import get_ome_channel_names

        tiff_exts = {".tif", ".tiff"}
        first_tiff = None
        for f in sorted(Path(img_dir).iterdir()):
            if f.suffix.lower() in tiff_exts and f.is_file():
                first_tiff = f
                break

        if first_tiff is None:
            messagebox.showinfo("No images",
                                "No TIFF files found in the image directory.")
            return []

        names = get_ome_channel_names(first_tiff)
        if not names:
            messagebox.showinfo(
                "No OME metadata",
                f"No channel names in OME metadata of:\n{first_tiff.name}\n\n"
                "Use the fallback index field instead.")
        return names

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

        # ---- Split mixed folder (images + _seg.npy in same dir) ----
        npy_frame = ttk.LabelFrame(
            tab,
            text="Split Mixed Folder (images + Cellpose _seg.npy in same directory)",
            padding=10,
        )
        npy_frame.pack(fill=tk.X, padx=10, pady=5)

        npy_hint = ttk.Label(
            npy_frame,
            text="For the common Cellpose workflow: images (.tif) and masks (_seg.npy) in the same folder.\n"
                 "Pairs each <name>.tif with <name>_seg.npy, converts masks to TIFF, and renames both\n"
                 "to pipeline convention (<prefix><NNN>_img.tif / <prefix><NNN>_masks.tif).",
            foreground="gray",
        )
        npy_hint.grid(row=0, column=0, columnspan=4, sticky=tk.W, pady=(0, 5))

        ttk.Label(npy_frame, text="Source folder:").grid(
            row=1, column=0, sticky=tk.W, pady=2)
        self.npy_input_dir = tk.StringVar()
        ttk.Entry(npy_frame, textvariable=self.npy_input_dir, width=50).grid(
            row=1, column=1, columnspan=2, padx=5, pady=2, sticky=tk.W)
        ttk.Button(npy_frame, text="Browse...",
                    command=lambda: self._browse_set(self.npy_input_dir)).grid(
            row=1, column=3, pady=2)

        ttk.Label(npy_frame, text="Image output dir:").grid(
            row=2, column=0, sticky=tk.W, pady=2)
        self.npy_img_out = tk.StringVar()
        ttk.Entry(npy_frame, textvariable=self.npy_img_out, width=50).grid(
            row=2, column=1, columnspan=2, padx=5, pady=2, sticky=tk.W)
        ttk.Button(npy_frame, text="Browse...",
                    command=lambda: self._browse_set(self.npy_img_out)).grid(
            row=2, column=3, pady=2)

        ttk.Label(npy_frame, text="Mask output dir:").grid(
            row=3, column=0, sticky=tk.W, pady=2)
        self.npy_mask_out = tk.StringVar()
        ttk.Entry(npy_frame, textvariable=self.npy_mask_out, width=50).grid(
            row=3, column=1, columnspan=2, padx=5, pady=2, sticky=tk.W)
        ttk.Button(npy_frame, text="Browse...",
                    command=lambda: self._browse_set(self.npy_mask_out)).grid(
            row=3, column=3, pady=2)

        npy_btn_row = ttk.Frame(npy_frame)
        npy_btn_row.grid(row=4, column=0, columnspan=4, sticky=tk.W, pady=(5, 0))
        ttk.Button(npy_btn_row, text="Preview",
                    command=self._split_mixed_preview).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(npy_btn_row, text="Split & Convert",
                    command=self._split_mixed_execute).pack(side=tk.LEFT, padx=(0, 15))
        ttk.Button(npy_btn_row, text="Convert _seg.npy only",
                    command=self._convert_seg_npy).pack(side=tk.LEFT)
        ttk.Label(npy_btn_row,
                   text="(convert in-place without renaming)",
                   foreground="gray").pack(side=tk.LEFT, padx=(5, 0))

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

    def _browse_set(self, var):
        d = filedialog.askdirectory(title="Select Directory")
        if d:
            var.set(d)

    def _split_mixed_preview(self):
        """Preview split of mixed folder (images + _seg.npy)."""
        self.rename_tree.delete(*self.rename_tree.get_children())
        src = self.npy_input_dir.get()
        if not src:
            messagebox.showwarning("Missing", "Select the source folder.")
            return
        try:
            from rename_files import split_mixed_folder
            prefix = self.rename_prefix.get()
            img_r, mask_r = split_mixed_folder(
                source_dir=src,
                image_output_dir=self.npy_img_out.get() or src,
                mask_output_dir=self.npy_mask_out.get() or src,
                prefix=prefix,
                dry_run=True,
            )
            for old, new in img_r:
                self.rename_tree.insert("", tk.END, values=("Image", old, new))
            for old, new in mask_r:
                self.rename_tree.insert("", tk.END, values=("Mask (.npy)", old, new))
            if not img_r:
                messagebox.showinfo("No pairs", "No image/_seg.npy pairs found.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _split_mixed_execute(self):
        """Split mixed folder: copy images + convert _seg.npy to TIFF."""
        src = self.npy_input_dir.get()
        img_out = self.npy_img_out.get()
        mask_out = self.npy_mask_out.get()
        if not src or not img_out or not mask_out:
            messagebox.showwarning(
                "Missing",
                "Set source folder and both output directories.")
            return
        try:
            from rename_files import split_mixed_folder
            prefix = self.rename_prefix.get()
            img_r, mask_r = split_mixed_folder(
                source_dir=src,
                image_output_dir=img_out,
                mask_output_dir=mask_out,
                prefix=prefix,
                dry_run=False,
            )
            if img_r:
                messagebox.showinfo(
                    "Done",
                    f"Split {len(img_r)} pairs:\n"
                    f"  Images -> {img_out}\n"
                    f"  Masks  -> {mask_out}",
                )
                self._split_mixed_preview()
            else:
                messagebox.showinfo("No pairs", "No image/_seg.npy pairs found.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _convert_seg_npy(self):
        """Convert _seg.npy files to TIFF masks in-place."""
        in_dir = self.npy_input_dir.get()
        if not in_dir:
            messagebox.showwarning("Missing", "Select the source folder.")
            return
        try:
            from data_preparation import convert_seg_npy_to_tif
            converted = convert_seg_npy_to_tif(in_dir)
            if converted:
                messagebox.showinfo(
                    "Done",
                    f"Converted {len(converted)} _seg.npy files to TIFF masks\n"
                    f"in {in_dir}",
                )
            else:
                messagebox.showinfo(
                    "No files",
                    "No _seg.npy files found in the selected directory.",
                )
        except Exception as e:
            messagebox.showerror("Error", str(e))

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
        ttk.Entry(top, textvariable=self.cfg_path_var, width=45).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(top, text="New", command=self._new_config).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Open...", command=self._open_config).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Save", command=self._save_config).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Save As...", command=self._save_config_as).pack(side=tk.LEFT, padx=2)

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

    def _new_config(self):
        """Create a new blank config with sensible defaults."""
        self.yaml_config = {
            "SYSTEM": {"NUM_GPUS": 1, "NUM_CPUS": 4, "SEED": 42},
            "PROBLEM": {"TYPE": "SEMANTIC_SEG", "NDIM": "2D", "DESCRIPTION": ""},
            "DATA": {
                "TRAIN": {
                    "PATH": "data/train/raw",
                    "MASK_PATH": "data/train/labels",
                    "IMAGE_FILTER": "_img",
                    "MASK_FILTER": "_masks",
                },
                "TEST": {
                    "PATH": "data/test/raw",
                    "MASK_PATH": "data/test/labels",
                    "IMAGE_FILTER": "_img",
                    "MASK_FILTER": "_masks",
                },
                "PATCH_SIZE": [256, 256],
                "CHANNELS": [0, 0],
                "Z_SLICE": None,
                "NORMALIZE": True,
            },
            "AUGMENTATION": {
                "ENABLE": True,
                "RANDOM_FLIP": True,
                "RANDOM_ROTATION": {"ENABLE": True, "DEGREES": 180},
                "ELASTIC_DEFORM": {"ENABLE": True, "ALPHA": [20, 40], "SIGMA": [5, 7]},
                "GAUSSIAN_NOISE": {"ENABLE": True, "MEAN": 0.0, "STD": 0.05},
                "BRIGHTNESS_CONTRAST": {
                    "ENABLE": True,
                    "BRIGHTNESS_RANGE": [-0.1, 0.1],
                    "CONTRAST_RANGE": [0.9, 1.1],
                },
            },
            "MODEL": {
                "BACKEND": "cellpose",
                "PRETRAINED_MODEL": None,
                "ARCHITECTURE": "cpsam",
            },
            "TRAIN": {
                "ENABLE": True,
                "EPOCHS": 100,
                "LEARNING_RATE": 1e-5,
                "WEIGHT_DECAY": 0.1,
                "BATCH_SIZE": 1,
                "OPTIMIZER": "adam",
                "SAVE_EVERY": 25,
                "MIN_TRAIN_MASKS": 5,
            },
            "INFERENCE": {
                "ENABLE": True,
                "DIAMETER": None,
                "FLOW_THRESHOLD": 0.4,
                "CELLPROB_THRESHOLD": 0.0,
                "NORMALIZE": {"TILE_NORM_BLOCKSIZE": 128},
                "AUGMENT": False,
                "RESAMPLE": True,
            },
            "PATHS": {
                "MODEL_DIR": "models",
                "RESULT_DIR": "results",
                "MODEL_NAME": "cellpose_custom",
            },
        }
        self.yaml_config_path = ""
        self.cfg_path_var.set("(new unsaved config)")
        self._populate_param_editor()
        self._sync_model_selector_from_config()
        logger.info("Created new config with defaults")

    def _save_config_as(self):
        """Save current config to a new file (always prompts for location)."""
        if not self.yaml_config:
            messagebox.showwarning("No Config", "Create or load a config first.")
            return
        self._read_params_into_config()
        path = filedialog.asksaveasfilename(
            title="Save Configuration As",
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

        # ---- DATA directories (nested under DATA.TRAIN / DATA.TEST) ----
        data_cfg = self.yaml_config.get("DATA", {})
        ttk.Label(
            self.param_inner,
            text="--- DATA DIRECTORIES ---",
            font=("TkDefaultFont", 10, "bold"),
        ).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(10, 2))
        row += 1
        for split_key, split_label in [("TRAIN", "Train"), ("TEST", "Test")]:
            split_cfg = data_cfg.get(split_key, {})
            for field_key, field_label in [
                ("PATH", f"{split_label} Image Dir"),
                ("MASK_PATH", f"{split_label} Mask Dir"),
                ("IMAGE_FILTER", f"{split_label} Image Filter"),
                ("MASK_FILTER", f"{split_label} Mask Filter"),
            ]:
                ttk.Label(self.param_inner, text=f"  {field_label}:").grid(
                    row=row, column=0, sticky=tk.W, padx=(10, 5), pady=1
                )
                full_key = f"DATA.{split_key}.{field_key}"
                var = tk.StringVar(value=str(split_cfg.get(field_key, "")))
                w = 40 if "Dir" in field_label else 15
                entry = ttk.Entry(self.param_inner, textvariable=var, width=w)
                entry.grid(row=row, column=1, sticky=tk.W, pady=1)
                if "Dir" in field_label:
                    ttk.Button(
                        self.param_inner, text="...",
                        command=lambda v=var: self._browse_dir(v),
                    ).grid(row=row, column=2, padx=2, pady=1)
                self.param_vars[full_key] = var
                row += 1

        row = self._add_section("DATA", row, [
            ("CHANNELS", "Channels [segment, nuclear]  (0=DIC, 1=mEGFP, 2=mScarlet, 3=miRFPnano3)", "str"),
            ("Z_SLICE", "Z-Slice (null=first, or 0-indexed integer)", "str"),
        ])

        row = self._add_section("TRAIN", row, [
            ("EPOCHS", "Epochs", "int"),
            ("LEARNING_RATE", "Learning Rate", "float"),
            ("WEIGHT_DECAY", "Weight Decay", "float"),
            ("BATCH_SIZE", "Batch Size", "int"),
            ("SAVE_EVERY", "Save Every N Epochs", "int"),
            ("MIN_TRAIN_MASKS", "Min Train Masks", "int"),
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
                try:
                    val = yaml.safe_load(raw)
                except yaml.YAMLError:
                    val = raw
            else:
                val = raw

            # Handle nested keys like DATA.TRAIN.PATH
            parts = full_key.split(".")
            d = self.yaml_config
            for p in parts[:-1]:
                if p not in d:
                    d[p] = {}
                d = d[p]
            d[parts[-1]] = val

    def _save_config(self):
        if not self.yaml_config:
            messagebox.showwarning("No Config", "Create or load a config first.")
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

        # Scrollable body
        canvas = tk.Canvas(tab, highlightthickness=0)
        sb = ttk.Scrollbar(tab, orient=tk.VERTICAL, command=canvas.yview)
        body = ttk.Frame(canvas)
        body.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=body, anchor=tk.NW)
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        def _on_scroll(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_scroll, add="+")

        # ---- Task Preset ----
        task_frame = ttk.LabelFrame(body, text="Task Preset", padding=10)
        task_frame.pack(fill=tk.X, padx=10, pady=(10, 5))

        ttk.Label(task_frame, text="Task:").grid(row=0, column=0, sticky=tk.W)
        self.train_task = tk.StringVar(value="custom")
        task_combo = ttk.Combobox(
            task_frame,
            textvariable=self.train_task,
            values=["dic", "fluor", "both", "custom"],
            width=12,
            state="readonly",
        )
        task_combo.grid(row=0, column=1, padx=5, sticky=tk.W)
        task_combo.bind("<<ComboboxSelected>>", self._train_on_task_change)

        ttk.Label(
            task_frame,
            text="'custom' uses the directories and config below; presets use built-in configs.",
            foreground="gray",
        ).grid(row=1, column=0, columnspan=4, sticky=tk.W, pady=(2, 0))

        # ---- Data Directories ----
        data_frame = ttk.LabelFrame(body, text="Training Data", padding=10)
        data_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(data_frame, text="Train Images:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.train_img_dir = tk.StringVar()
        ttk.Entry(data_frame, textvariable=self.train_img_dir, width=50).grid(
            row=0, column=1, padx=5, pady=2, sticky=tk.EW
        )
        ttk.Button(
            data_frame, text="Browse...",
            command=lambda: self._browse_dir(self.train_img_dir),
        ).grid(row=0, column=2, pady=2)

        ttk.Label(data_frame, text="Train Masks:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.train_mask_dir = tk.StringVar()
        ttk.Entry(data_frame, textvariable=self.train_mask_dir, width=50).grid(
            row=1, column=1, padx=5, pady=2, sticky=tk.EW
        )
        ttk.Button(
            data_frame, text="Browse...",
            command=lambda: self._browse_dir(self.train_mask_dir),
        ).grid(row=1, column=2, pady=2)

        ttk.Label(data_frame, text="Test Images:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.test_img_dir = tk.StringVar()
        ttk.Entry(data_frame, textvariable=self.test_img_dir, width=50).grid(
            row=2, column=1, padx=5, pady=2, sticky=tk.EW
        )
        ttk.Button(
            data_frame, text="Browse...",
            command=lambda: self._browse_dir(self.test_img_dir),
        ).grid(row=2, column=2, pady=2)

        ttk.Label(data_frame, text="Test Masks:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.test_mask_dir = tk.StringVar()
        ttk.Entry(data_frame, textvariable=self.test_mask_dir, width=50).grid(
            row=3, column=1, padx=5, pady=2, sticky=tk.EW
        )
        ttk.Button(
            data_frame, text="Browse...",
            command=lambda: self._browse_dir(self.test_mask_dir),
        ).grid(row=3, column=2, pady=2)

        # File filters
        filter_frame = ttk.Frame(data_frame)
        filter_frame.grid(row=4, column=0, columnspan=3, sticky=tk.W, pady=(5, 0))
        ttk.Label(filter_frame, text="Image Filter:").pack(side=tk.LEFT)
        self.train_img_filter = tk.StringVar(value="_img")
        ttk.Entry(filter_frame, textvariable=self.train_img_filter, width=10).pack(
            side=tk.LEFT, padx=(5, 15)
        )
        ttk.Label(filter_frame, text="Mask Filter:").pack(side=tk.LEFT)
        self.train_mask_filter = tk.StringVar(value="_masks")
        ttk.Entry(filter_frame, textvariable=self.train_mask_filter, width=10).pack(
            side=tk.LEFT, padx=5
        )

        data_frame.columnconfigure(1, weight=1)

        # ---- Training Parameters (quick settings) ----
        param_frame = ttk.LabelFrame(body, text="Training Parameters", padding=10)
        param_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(param_frame, text="Epochs:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.train_epochs = tk.StringVar(value="100")
        ttk.Entry(param_frame, textvariable=self.train_epochs, width=10).grid(
            row=0, column=1, padx=5, pady=2, sticky=tk.W
        )

        ttk.Label(param_frame, text="Learning Rate:").grid(row=0, column=2, sticky=tk.W, padx=(20, 0), pady=2)
        self.train_lr = tk.StringVar(value="0.00001")
        ttk.Entry(param_frame, textvariable=self.train_lr, width=12).grid(
            row=0, column=3, padx=5, pady=2, sticky=tk.W
        )

        ttk.Label(param_frame, text="Batch Size:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.train_batch = tk.StringVar(value="1")
        ttk.Entry(param_frame, textvariable=self.train_batch, width=10).grid(
            row=1, column=1, padx=5, pady=2, sticky=tk.W
        )

        ttk.Label(param_frame, text="Weight Decay:").grid(row=1, column=2, sticky=tk.W, padx=(20, 0), pady=2)
        self.train_wd = tk.StringVar(value="0.1")
        ttk.Entry(param_frame, textvariable=self.train_wd, width=12).grid(
            row=1, column=3, padx=5, pady=2, sticky=tk.W
        )

        ttk.Label(param_frame, text="Channels [seg, nuc]:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.train_channels = tk.StringVar(value="[0, 0]")
        ttk.Entry(param_frame, textvariable=self.train_channels, width=12).grid(
            row=2, column=1, padx=5, pady=2, sticky=tk.W
        )

        ttk.Label(param_frame, text="Diameter:").grid(row=2, column=2, sticky=tk.W, padx=(20, 0), pady=2)
        self.train_diameter = tk.StringVar(value="null")
        ttk.Entry(param_frame, textvariable=self.train_diameter, width=12).grid(
            row=2, column=3, padx=5, pady=2, sticky=tk.W
        )

        ttk.Label(param_frame, text="Save Every N Epochs:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.train_save_every = tk.StringVar(value="25")
        ttk.Entry(param_frame, textvariable=self.train_save_every, width=10).grid(
            row=3, column=1, padx=5, pady=2, sticky=tk.W
        )

        ttk.Label(param_frame, text="Min Train Masks:").grid(row=3, column=2, sticky=tk.W, padx=(20, 0), pady=2)
        self.train_min_masks = tk.StringVar(value="5")
        ttk.Entry(param_frame, textvariable=self.train_min_masks, width=10).grid(
            row=3, column=3, padx=5, pady=2, sticky=tk.W
        )

        self.train_augment = tk.BooleanVar(value=True)
        ttk.Checkbutton(param_frame, text="Enable Data Augmentation", variable=self.train_augment).grid(
            row=4, column=0, columnspan=2, sticky=tk.W, pady=2
        )

        self.train_use_gpu = tk.BooleanVar(value=True)
        ttk.Checkbutton(param_frame, text="Use GPU", variable=self.train_use_gpu).grid(
            row=4, column=2, columnspan=2, sticky=tk.W, pady=2
        )

        # ---- Model ----
        model_frame = ttk.LabelFrame(body, text="Model", padding=10)
        model_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(model_frame, text="Pretrained Model:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.train_model_var = tk.StringVar(value="cpsam (default)")
        ttk.Combobox(
            model_frame,
            textvariable=self.train_model_var,
            values=["cpsam (default)", "Custom model..."],
            width=20,
            state="readonly",
        ).grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)

        self.train_model_path = tk.StringVar()
        self.train_model_entry = ttk.Entry(
            model_frame, textvariable=self.train_model_path, width=40
        )
        self.train_model_entry.grid(row=0, column=2, padx=5, pady=2, sticky=tk.EW)
        self.train_model_entry.config(state=tk.DISABLED)
        self.train_model_btn = ttk.Button(
            model_frame, text="Browse...", command=self._train_browse_model
        )
        self.train_model_btn.grid(row=0, column=3, pady=2)
        self.train_model_btn.config(state=tk.DISABLED)
        self.train_model_var.trace_add("write", lambda *_: self._train_on_model_change())

        ttk.Label(model_frame, text="Output Model Name:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.train_model_name = tk.StringVar(value="cellpose_custom")
        ttk.Entry(model_frame, textvariable=self.train_model_name, width=30).grid(
            row=1, column=1, columnspan=2, padx=5, pady=2, sticky=tk.W
        )

        ttk.Label(model_frame, text="Model Save Dir:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.train_model_dir = tk.StringVar(value="models")
        ttk.Entry(model_frame, textvariable=self.train_model_dir, width=40).grid(
            row=2, column=1, columnspan=2, padx=5, pady=2, sticky=tk.EW
        )
        ttk.Button(
            model_frame, text="Browse...",
            command=lambda: self._browse_dir(self.train_model_dir),
        ).grid(row=2, column=3, pady=2)

        model_frame.columnconfigure(2, weight=1)

        # ---- Config link ----
        cfg_link = ttk.Frame(body)
        cfg_link.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(
            cfg_link,
            text="For advanced settings (augmentation details, inference params), use the Configuration tab.",
            foreground="gray",
        ).pack(anchor=tk.W)
        ttk.Button(
            cfg_link, text="Load Settings from Config File...",
            command=self._train_load_from_config,
        ).pack(anchor=tk.W, pady=(2, 0))

        # ---- Buttons ----
        btn_frame = ttk.Frame(body)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)

        self.btn_train = ttk.Button(
            btn_frame, text="Start Training", command=self._start_training
        )
        self.btn_train.pack(side=tk.LEFT, padx=5)

        self.btn_stop = ttk.Button(
            btn_frame, text="Stop", command=self._stop_training, state=tk.DISABLED
        )
        self.btn_stop.pack(side=tk.LEFT, padx=5)

        # Progress
        self.train_progress = ttk.Progressbar(body, mode="indeterminate")
        self.train_progress.pack(fill=tk.X, padx=10, pady=5)

        self.train_status = tk.StringVar(value="Ready")
        ttk.Label(body, textvariable=self.train_status).pack(padx=10, anchor=tk.W)

        # Log output
        log_frame = ttk.LabelFrame(body, text="Training Log", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))

        self.train_log = scrolledtext.ScrolledText(
            log_frame, wrap=tk.WORD, height=15, state=tk.DISABLED, font=("Courier", 9)
        )
        self.train_log.pack(fill=tk.BOTH, expand=True)

        self._training_thread = None
        self._stop_event = threading.Event()

    def _train_on_task_change(self, event=None):
        """When task preset changes, auto-fill directories from preset config."""
        task = self.train_task.get()
        if task == "custom":
            return
        presets = {
            "dic": PROJECT_ROOT / "configs" / "dic_wholecell.yaml",
            "fluor": PROJECT_ROOT / "configs" / "fluor_nucleus.yaml",
        }
        paths = []
        if task == "both":
            paths = [presets["dic"], presets["fluor"]]
        elif task in presets:
            paths = [presets[task]]

        if paths and paths[0].exists():
            with open(paths[0]) as f:
                cfg = yaml.safe_load(f)
            data_cfg = cfg.get("DATA", {})
            base = PROJECT_ROOT
            self.train_img_dir.set(str(base / data_cfg.get("TRAIN", {}).get("PATH", "")))
            self.train_mask_dir.set(str(base / data_cfg.get("TRAIN", {}).get("MASK_PATH", "")))
            self.test_img_dir.set(str(base / data_cfg.get("TEST", {}).get("PATH", "")))
            self.test_mask_dir.set(str(base / data_cfg.get("TEST", {}).get("MASK_PATH", "")))
            self.train_img_filter.set(data_cfg.get("TRAIN", {}).get("IMAGE_FILTER", "_img"))
            self.train_mask_filter.set(data_cfg.get("TRAIN", {}).get("MASK_FILTER", "_masks"))
            ch = data_cfg.get("CHANNELS", [0, 0])
            self.train_channels.set(str(ch))
            train_cfg = cfg.get("TRAIN", {})
            self.train_epochs.set(str(train_cfg.get("EPOCHS", 100)))
            self.train_lr.set(str(train_cfg.get("LEARNING_RATE", 1e-5)))
            self.train_batch.set(str(train_cfg.get("BATCH_SIZE", 1)))
            self.train_wd.set(str(train_cfg.get("WEIGHT_DECAY", 0.1)))
            self.train_save_every.set(str(train_cfg.get("SAVE_EVERY", 25)))
            self.train_min_masks.set(str(train_cfg.get("MIN_TRAIN_MASKS", 5)))
            paths_cfg = cfg.get("PATHS", {})
            self.train_model_name.set(paths_cfg.get("MODEL_NAME", "cellpose_model"))
            self.train_model_dir.set(str(base / paths_cfg.get("MODEL_DIR", "models")))

    def _train_on_model_change(self):
        choice = self.train_model_var.get()
        if "Custom" in choice:
            self.train_model_entry.config(state=tk.NORMAL)
            self.train_model_btn.config(state=tk.NORMAL)
        else:
            self.train_model_entry.config(state=tk.DISABLED)
            self.train_model_btn.config(state=tk.DISABLED)
            self.train_model_path.set("")

    def _train_browse_model(self):
        path = filedialog.askopenfilename(
            title="Select Pretrained Model",
            initialdir=str(PROJECT_ROOT / "models"),
        )
        if path:
            self.train_model_path.set(path)

    def _train_load_from_config(self):
        """Load all training tab fields from a YAML config file."""
        path = filedialog.askopenfilename(
            title="Load Training Settings from Config",
            filetypes=[("YAML", "*.yaml *.yml"), ("All", "*.*")],
            initialdir=str(PROJECT_ROOT / "configs"),
        )
        if not path:
            return
        try:
            with open(path) as f:
                cfg = yaml.safe_load(f)
            data_cfg = cfg.get("DATA", {})
            base = Path(path).parent.parent  # configs -> project root
            self.train_img_dir.set(str(base / data_cfg.get("TRAIN", {}).get("PATH", "")))
            self.train_mask_dir.set(str(base / data_cfg.get("TRAIN", {}).get("MASK_PATH", "")))
            self.test_img_dir.set(str(base / data_cfg.get("TEST", {}).get("PATH", "")))
            self.test_mask_dir.set(str(base / data_cfg.get("TEST", {}).get("MASK_PATH", "")))
            self.train_img_filter.set(data_cfg.get("TRAIN", {}).get("IMAGE_FILTER", "_img"))
            self.train_mask_filter.set(data_cfg.get("TRAIN", {}).get("MASK_FILTER", "_masks"))
            ch = data_cfg.get("CHANNELS", [0, 0])
            self.train_channels.set(str(ch))
            train_cfg = cfg.get("TRAIN", {})
            self.train_epochs.set(str(train_cfg.get("EPOCHS", 100)))
            self.train_lr.set(str(train_cfg.get("LEARNING_RATE", 1e-5)))
            self.train_batch.set(str(train_cfg.get("BATCH_SIZE", 1)))
            self.train_wd.set(str(train_cfg.get("WEIGHT_DECAY", 0.1)))
            self.train_save_every.set(str(train_cfg.get("SAVE_EVERY", 25)))
            self.train_min_masks.set(str(train_cfg.get("MIN_TRAIN_MASKS", 5)))
            aug_cfg = cfg.get("AUGMENTATION", {})
            self.train_augment.set(aug_cfg.get("ENABLE", True))
            sys_cfg = cfg.get("SYSTEM", {})
            self.train_use_gpu.set(sys_cfg.get("NUM_GPUS", 1) > 0)
            paths_cfg = cfg.get("PATHS", {})
            self.train_model_name.set(paths_cfg.get("MODEL_NAME", "cellpose_model"))
            self.train_model_dir.set(str(base / paths_cfg.get("MODEL_DIR", "models")))
            model_cfg = cfg.get("MODEL", {})
            pretrained = model_cfg.get("PRETRAINED_MODEL")
            if pretrained:
                self.train_model_var.set("Custom model...")
                self.train_model_path.set(str(pretrained))
            else:
                self.train_model_var.set("cpsam (default)")
            self.train_task.set("custom")
            logger.info(f"Loaded training settings from {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load config:\n{e}")

    def _build_train_config_from_gui(self) -> dict:
        """Build a full training config dict from the Training tab GUI fields."""
        channels = [0, 0]
        try:
            channels = yaml.safe_load(self.train_channels.get())
        except Exception:
            pass
        diameter = None
        d_str = self.train_diameter.get().strip().lower()
        if d_str not in ("null", "none", "auto", ""):
            try:
                diameter = float(d_str)
            except ValueError:
                pass
        pretrained = None
        if "Custom" in self.train_model_var.get() and self.train_model_path.get():
            pretrained = self.train_model_path.get()

        return {
            "SYSTEM": {
                "NUM_GPUS": 1 if self.train_use_gpu.get() else 0,
                "NUM_CPUS": 4,
                "SEED": 42,
            },
            "DATA": {
                "TRAIN": {
                    "PATH": self.train_img_dir.get(),
                    "MASK_PATH": self.train_mask_dir.get(),
                    "IMAGE_FILTER": self.train_img_filter.get(),
                    "MASK_FILTER": self.train_mask_filter.get(),
                },
                "TEST": {
                    "PATH": self.test_img_dir.get(),
                    "MASK_PATH": self.test_mask_dir.get(),
                    "IMAGE_FILTER": self.train_img_filter.get(),
                    "MASK_FILTER": self.train_mask_filter.get(),
                },
                "CHANNELS": channels,
                "NORMALIZE": True,
            },
            "AUGMENTATION": {
                "ENABLE": self.train_augment.get(),
                "RANDOM_FLIP": True,
                "RANDOM_ROTATION": {"ENABLE": True, "DEGREES": 180},
                "ELASTIC_DEFORM": {"ENABLE": True, "ALPHA": [20, 40], "SIGMA": [5, 7]},
                "GAUSSIAN_NOISE": {"ENABLE": True, "MEAN": 0.0, "STD": 0.05},
                "BRIGHTNESS_CONTRAST": {
                    "ENABLE": True,
                    "BRIGHTNESS_RANGE": [-0.1, 0.1],
                    "CONTRAST_RANGE": [0.9, 1.1],
                },
            },
            "MODEL": {
                "BACKEND": "cellpose",
                "PRETRAINED_MODEL": pretrained,
                "ARCHITECTURE": "cpsam",
            },
            "TRAIN": {
                "ENABLE": True,
                "EPOCHS": int(self.train_epochs.get()),
                "LEARNING_RATE": float(self.train_lr.get()),
                "WEIGHT_DECAY": float(self.train_wd.get()),
                "BATCH_SIZE": int(self.train_batch.get()),
                "OPTIMIZER": "adam",
                "SAVE_EVERY": int(self.train_save_every.get()),
                "MIN_TRAIN_MASKS": int(self.train_min_masks.get()),
            },
            "INFERENCE": {
                "ENABLE": True,
                "DIAMETER": diameter,
                "FLOW_THRESHOLD": 0.4,
                "CELLPROB_THRESHOLD": 0.0,
                "NORMALIZE": {"TILE_NORM_BLOCKSIZE": 128},
            },
            "PATHS": {
                "MODEL_DIR": self.train_model_dir.get(),
                "RESULT_DIR": str(Path(self.train_model_dir.get()).parent / "results"),
                "MODEL_NAME": self.train_model_name.get(),
            },
        }

    def _start_training(self):
        task = self.train_task.get()

        if task == "custom":
            # Validate that at least training image dir is set
            if not self.train_img_dir.get():
                messagebox.showwarning(
                    "Missing Data", "Set the Train Images directory."
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
                # Build config from GUI fields and run directly
                gui_cfg = self._build_train_config_from_gui()
                logger.info("Training with settings from Training tab")
                train_cellpose_model(gui_cfg)
                self.log_queue.put("__TRAINING_DONE__")
                return

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
    # TAB 5: MASK COMPARISON  (single pair + bulk directories)
    # ==================================================================
    def _build_compare_tab(self):
        tab = self.tab_compare

        # Sub-notebook: Single / Bulk
        self.cmp_notebook = ttk.Notebook(tab)
        self.cmp_notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self._cmp_single_frame = ttk.Frame(self.cmp_notebook)
        self._cmp_bulk_frame = ttk.Frame(self.cmp_notebook)
        self.cmp_notebook.add(self._cmp_single_frame, text="  Single Pair  ")
        self.cmp_notebook.add(self._cmp_bulk_frame, text="  Bulk (Directories)  ")

        self._build_cmp_single()
        self._build_cmp_bulk()

    # -- Single pair sub-tab ----
    def _build_cmp_single(self):
        tab = self._cmp_single_frame

        ttk.Label(
            tab,
            text="Compare one image against two masks (ground truth vs model).\n"
                 "Supports _seg.npy (Cellpose) and .tif mask formats.",
            foreground="gray",
        ).pack(anchor=tk.W, padx=10, pady=(10, 5))

        io_frame = ttk.LabelFrame(tab, text="Input Files", padding=10)
        io_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(io_frame, text="Original Image:").grid(row=0, column=0, sticky=tk.W, pady=3)
        self.cmp_image_var = tk.StringVar()
        ttk.Entry(io_frame, textvariable=self.cmp_image_var, width=55).grid(
            row=0, column=1, padx=5, pady=3, sticky=tk.EW)
        ttk.Button(
            io_frame, text="Browse...",
            command=lambda: self._cmp_browse_file(
                self.cmp_image_var, "Select Original Image",
                [("Image files", "*.tif *.tiff *.ome.tif *.png"), ("All", "*.*")]),
        ).grid(row=0, column=2, pady=3)

        ttk.Label(io_frame, text="Mask 1 (Ground Truth):").grid(row=1, column=0, sticky=tk.W, pady=3)
        self.cmp_mask1_var = tk.StringVar()
        ttk.Entry(io_frame, textvariable=self.cmp_mask1_var, width=55).grid(
            row=1, column=1, padx=5, pady=3, sticky=tk.EW)
        ttk.Button(
            io_frame, text="Browse...",
            command=lambda: self._cmp_browse_file(
                self.cmp_mask1_var, "Select Mask 1 (Ground Truth)",
                [("Mask files", "*.npy *.tif *.tiff"), ("All", "*.*")]),
        ).grid(row=1, column=2, pady=3)

        ttk.Label(io_frame, text="Mask 2 (Model Output):").grid(row=2, column=0, sticky=tk.W, pady=3)
        self.cmp_mask2_var = tk.StringVar()
        ttk.Entry(io_frame, textvariable=self.cmp_mask2_var, width=55).grid(
            row=2, column=1, padx=5, pady=3, sticky=tk.EW)
        ttk.Button(
            io_frame, text="Browse...",
            command=lambda: self._cmp_browse_file(
                self.cmp_mask2_var, "Select Mask 2 (Model Output)",
                [("Mask files", "*.npy *.tif *.tiff"), ("All", "*.*")]),
        ).grid(row=2, column=2, pady=3)
        io_frame.columnconfigure(1, weight=1)

        out_frame = ttk.LabelFrame(tab, text="Output", padding=10)
        out_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(out_frame, text="Report Directory:").grid(row=0, column=0, sticky=tk.W, pady=3)
        self.cmp_output_var = tk.StringVar(value="comparison_report")
        ttk.Entry(out_frame, textvariable=self.cmp_output_var, width=55).grid(
            row=0, column=1, padx=5, pady=3, sticky=tk.EW)
        ttk.Button(
            out_frame, text="Browse...",
            command=lambda: self._browse_dir(self.cmp_output_var),
        ).grid(row=0, column=2, pady=3)

        self.cmp_open_fig = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            out_frame, text="Open comparison figure when done",
            variable=self.cmp_open_fig,
        ).grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=2)
        out_frame.columnconfigure(1, weight=1)

        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)

        self.btn_cmp_run = ttk.Button(btn_frame, text="Run Comparison", command=self._cmp_run)
        self.btn_cmp_run.pack(side=tk.LEFT, padx=5)

        self.btn_cmp_quick = ttk.Button(
            btn_frame, text="Quick Summary (no figure)", command=self._cmp_run_quick)
        self.btn_cmp_quick.pack(side=tk.LEFT, padx=5)

        self.cmp_progress = ttk.Progressbar(tab, mode="indeterminate")
        self.cmp_progress.pack(fill=tk.X, padx=10, pady=5)

        self.cmp_status = tk.StringVar(value="Ready")
        ttk.Label(tab, textvariable=self.cmp_status).pack(padx=10, anchor=tk.W)

        results_frame = ttk.LabelFrame(tab, text="Results", padding=5)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))

        self.cmp_results = scrolledtext.ScrolledText(
            results_frame, wrap=tk.WORD, height=12, state=tk.DISABLED,
            font=("Courier", 9))
        self.cmp_results.pack(fill=tk.BOTH, expand=True)

    # -- Bulk sub-tab ----
    def _build_cmp_bulk(self):
        tab = self._cmp_bulk_frame

        ttk.Label(
            tab,
            text="Compare all matching image/mask pairs across directories.\n"
                 "Files are auto-matched by filename stem "
                 "(e.g. dic_001_img.tif  <->  dic_001_masks.tif  <->  dic_001_seg.npy).",
            foreground="gray",
        ).pack(anchor=tk.W, padx=10, pady=(10, 5))

        dir_frame = ttk.LabelFrame(tab, text="Input Directories", padding=10)
        dir_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(dir_frame, text="Images Directory:").grid(row=0, column=0, sticky=tk.W, pady=3)
        self.cmp_bulk_img_dir = tk.StringVar()
        ttk.Entry(dir_frame, textvariable=self.cmp_bulk_img_dir, width=55).grid(
            row=0, column=1, padx=5, pady=3, sticky=tk.EW)
        ttk.Button(
            dir_frame, text="Browse...",
            command=lambda: self._browse_dir(self.cmp_bulk_img_dir),
        ).grid(row=0, column=2, pady=3)

        ttk.Label(dir_frame, text="Mask 1 Dir (Ground Truth):").grid(row=1, column=0, sticky=tk.W, pady=3)
        self.cmp_bulk_mask1_dir = tk.StringVar()
        ttk.Entry(dir_frame, textvariable=self.cmp_bulk_mask1_dir, width=55).grid(
            row=1, column=1, padx=5, pady=3, sticky=tk.EW)
        ttk.Button(
            dir_frame, text="Browse...",
            command=lambda: self._browse_dir(self.cmp_bulk_mask1_dir),
        ).grid(row=1, column=2, pady=3)

        ttk.Label(dir_frame, text="Mask 2 Dir (Model Output):").grid(row=2, column=0, sticky=tk.W, pady=3)
        self.cmp_bulk_mask2_dir = tk.StringVar()
        ttk.Entry(dir_frame, textvariable=self.cmp_bulk_mask2_dir, width=55).grid(
            row=2, column=1, padx=5, pady=3, sticky=tk.EW)
        ttk.Button(
            dir_frame, text="Browse...",
            command=lambda: self._browse_dir(self.cmp_bulk_mask2_dir),
        ).grid(row=2, column=2, pady=3)
        dir_frame.columnconfigure(1, weight=1)

        out_frame = ttk.LabelFrame(tab, text="Output", padding=10)
        out_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(out_frame, text="Report Directory:").grid(row=0, column=0, sticky=tk.W, pady=3)
        self.cmp_bulk_output_var = tk.StringVar(value="batch_comparison")
        ttk.Entry(out_frame, textvariable=self.cmp_bulk_output_var, width=55).grid(
            row=0, column=1, padx=5, pady=3, sticky=tk.EW)
        ttk.Button(
            out_frame, text="Browse...",
            command=lambda: self._browse_dir(self.cmp_bulk_output_var),
        ).grid(row=0, column=2, pady=3)
        out_frame.columnconfigure(1, weight=1)

        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)

        self.btn_cmp_bulk_preview = ttk.Button(
            btn_frame, text="Preview Matches", command=self._cmp_bulk_preview)
        self.btn_cmp_bulk_preview.pack(side=tk.LEFT, padx=5)

        self.btn_cmp_bulk_run = ttk.Button(
            btn_frame, text="Run Batch Comparison", command=self._cmp_bulk_run)
        self.btn_cmp_bulk_run.pack(side=tk.LEFT, padx=5)

        self.cmp_bulk_progress = ttk.Progressbar(tab, mode="determinate", maximum=100)
        self.cmp_bulk_progress.pack(fill=tk.X, padx=10, pady=5)

        self.cmp_bulk_status = tk.StringVar(value="Ready")
        ttk.Label(tab, textvariable=self.cmp_bulk_status).pack(padx=10, anchor=tk.W)

        results_frame = ttk.LabelFrame(tab, text="Results", padding=5)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))

        self.cmp_bulk_results = scrolledtext.ScrolledText(
            results_frame, wrap=tk.WORD, height=12, state=tk.DISABLED,
            font=("Courier", 9))
        self.cmp_bulk_results.pack(fill=tk.BOTH, expand=True)

    # -- Shared helpers ----
    def _cmp_browse_file(self, var, title, filetypes):
        p = filedialog.askopenfilename(title=title, filetypes=filetypes)
        if p:
            var.set(p)

    def _cmp_validate_single(self) -> bool:
        for var, name in [
            (self.cmp_image_var, "original image"),
            (self.cmp_mask1_var, "Mask 1"),
            (self.cmp_mask2_var, "Mask 2"),
        ]:
            if not var.get() or not os.path.isfile(var.get()):
                messagebox.showwarning("Missing", f"Select a valid {name} file.")
                return False
        return True

    def _cmp_validate_bulk(self) -> bool:
        for var, name in [
            (self.cmp_bulk_img_dir, "Images Directory"),
            (self.cmp_bulk_mask1_dir, "Mask 1 Directory"),
            (self.cmp_bulk_mask2_dir, "Mask 2 Directory"),
        ]:
            if not var.get() or not os.path.isdir(var.get()):
                messagebox.showwarning("Missing", f"Select a valid {name}.")
                return False
        return True

    def _cmp_set_running(self, running: bool):
        state = tk.DISABLED if running else tk.NORMAL
        self.btn_cmp_run.config(state=state)
        self.btn_cmp_quick.config(state=state)
        if running:
            self.cmp_progress.start(10)
        else:
            self.cmp_progress.stop()

    def _cmp_bulk_set_running(self, running: bool):
        state = tk.DISABLED if running else tk.NORMAL
        self.btn_cmp_bulk_run.config(state=state)
        self.btn_cmp_bulk_preview.config(state=state)
        if not running:
            self.cmp_bulk_progress.config(value=0)

    @staticmethod
    def _cmp_format_single(results: dict) -> str:
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
            if len(sorted_m) > 10:
                lines.append("")
                lines.append("  BOTTOM 10 WORST MATCHED OBJECTS")
                lines.append(f"  {'M1 Label':<12}{'M2 Label':<12}{'IoU':<10}")
                lines.append("  " + "-" * 34)
                for l1, l2, iou in sorted_m[-10:]:
                    lines.append(f"  {l1:<12}{l2:<12}{iou:.4f}")

        lines.append("")
        lines.append("=" * 55)
        return "\n".join(lines)

    @staticmethod
    def _cmp_format_batch(batch: dict) -> str:
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

        lines.append("  PER-PAIR SUMMARY")
        lines.append(f"  {'Key':<20}{'BinIoU':<10}{'BinDice':<10}"
                     f"{'ObjM1':<8}{'ObjM2':<8}{'Match':<8}{'MeanIoU':<10}")
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

    # -- Single pair actions ----
    def _cmp_run(self):
        if not self._cmp_validate_single():
            return
        self._cmp_set_running(True)
        self.cmp_status.set("Running comparison...")
        self.cmp_results.config(state=tk.NORMAL)
        self.cmp_results.delete("1.0", tk.END)
        self.cmp_results.config(state=tk.DISABLED)

        def worker():
            try:
                from compare_masks import compare
                results = compare(
                    self.cmp_image_var.get(),
                    self.cmp_mask1_var.get(),
                    self.cmp_mask2_var.get(),
                    self.cmp_output_var.get(),
                )
                text = self._cmp_format_single(results)
                self.log_queue.put(f"__CMP_DONE__{text}")
                if self.cmp_open_fig.get():
                    fig_path = os.path.join(self.cmp_output_var.get(), "comparison.png")
                    if os.path.isfile(fig_path):
                        self.log_queue.put(f"__CMP_OPEN__{fig_path}")
            except Exception as e:
                logger.exception("Mask comparison failed")
                self.log_queue.put(f"__CMP_ERROR__{e}")

        threading.Thread(target=worker, daemon=True).start()

    def _cmp_run_quick(self):
        if not self._cmp_validate_single():
            return
        self._cmp_set_running(True)
        self.cmp_status.set("Computing metrics...")
        self.cmp_results.config(state=tk.NORMAL)
        self.cmp_results.delete("1.0", tk.END)
        self.cmp_results.config(state=tk.DISABLED)

        def worker():
            try:
                from compare_masks import (
                    load_image, load_mask, binary_metrics,
                    match_objects, intensity_stats, average_precision,
                )
                image = load_image(self.cmp_image_var.get())
                mask1 = load_mask(self.cmp_mask1_var.get())
                mask2 = load_mask(self.cmp_mask2_var.get())
                binary = binary_metrics(mask1, mask2)
                matches, _, _ = match_objects(mask1, mask2)
                ap = average_precision(mask1, mask2)
                stats1 = intensity_stats(image, mask1)
                stats2 = intensity_stats(image, mask2)
                results = {
                    "binary": binary, "matches": matches, "ap": ap,
                    "stats_mask1": stats1, "stats_mask2": stats2,
                }
                text = self._cmp_format_single(results)
                self.log_queue.put(f"__CMP_DONE__{text}")
            except Exception as e:
                logger.exception("Quick comparison failed")
                self.log_queue.put(f"__CMP_ERROR__{e}")

        threading.Thread(target=worker, daemon=True).start()

    def _on_cmp_finished(self, text: str | None = None, error: str | None = None):
        self._cmp_set_running(False)
        if error:
            self.cmp_status.set(f"Error: {error}")
            messagebox.showerror("Comparison Error", str(error))
        else:
            self.cmp_status.set("Comparison complete")
            if text:
                self.cmp_results.config(state=tk.NORMAL)
                self.cmp_results.insert(tk.END, text)
                self.cmp_results.see("1.0")
                self.cmp_results.config(state=tk.DISABLED)

    # -- Bulk actions ----
    def _cmp_bulk_preview(self):
        if not self._cmp_validate_bulk():
            return
        try:
            from compare_masks import auto_match
            matched, warnings = auto_match(
                self.cmp_bulk_img_dir.get(),
                self.cmp_bulk_mask1_dir.get(),
                self.cmp_bulk_mask2_dir.get(),
            )
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        lines = [f"Found {len(matched)} matched triplets:\n"]
        for img, m1, m2 in matched:
            lines.append(f"  {os.path.basename(img)}")
            lines.append(f"    M1: {os.path.basename(m1)}")
            lines.append(f"    M2: {os.path.basename(m2)}")
            lines.append("")
        if warnings:
            lines.append(f"\nWarnings ({len(warnings)}):")
            for w in warnings:
                lines.append(f"  {w}")

        self.cmp_bulk_results.config(state=tk.NORMAL)
        self.cmp_bulk_results.delete("1.0", tk.END)
        self.cmp_bulk_results.insert(tk.END, "\n".join(lines))
        self.cmp_bulk_results.config(state=tk.DISABLED)
        self.cmp_bulk_status.set(f"{len(matched)} pairs matched, {len(warnings)} warnings")

    def _cmp_bulk_run(self):
        if not self._cmp_validate_bulk():
            return
        self._cmp_bulk_set_running(True)
        self.cmp_bulk_status.set("Running batch comparison...")
        self.cmp_bulk_results.config(state=tk.NORMAL)
        self.cmp_bulk_results.delete("1.0", tk.END)
        self.cmp_bulk_results.config(state=tk.DISABLED)

        def progress_cb(current, total, key, _result):
            pct = int(100 * current / total) if total > 0 else 0
            self.log_queue.put(f"__BCMP_PROGRESS__{pct}||{current}||{total}||{key}")

        def worker():
            try:
                from compare_masks import compare_batch
                batch = compare_batch(
                    self.cmp_bulk_img_dir.get(),
                    self.cmp_bulk_mask1_dir.get(),
                    self.cmp_bulk_mask2_dir.get(),
                    self.cmp_bulk_output_var.get(),
                    progress_callback=progress_cb,
                )
                text = self._cmp_format_batch(batch)
                self.log_queue.put(f"__BCMP_DONE__{text}")
            except Exception as e:
                logger.exception("Batch comparison failed")
                self.log_queue.put(f"__BCMP_ERROR__{e}")

        threading.Thread(target=worker, daemon=True).start()

    def _on_bcmp_finished(self, text: str | None = None, error: str | None = None):
        self._cmp_bulk_set_running(False)
        if error:
            self.cmp_bulk_status.set(f"Error: {error}")
            messagebox.showerror("Batch Error", str(error))
        else:
            self.cmp_bulk_status.set("Batch comparison complete")
            if text:
                self.cmp_bulk_results.config(state=tk.NORMAL)
                self.cmp_bulk_results.insert(tk.END, text)
                self.cmp_bulk_results.see("1.0")
                self.cmp_bulk_results.config(state=tk.DISABLED)

    @staticmethod
    def _cmp_open_figure(path: str):
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

            # -- Tuple messages (e.g. plot embeds) --
            if isinstance(msg, tuple):
                if msg[0] == "__ANA_PLOT__":
                    _, plot_title, fig = msg
                    if plot_title in self._ana_plot_frames:
                        self._ana_embed_figure(fig, self._ana_plot_frames[plot_title])
                continue

            # -- ND2 Conversion --
            if msg == "__ND2_DONE__":
                self._on_nd2_finished()
                continue
            if msg.startswith("__ND2_ERROR__"):
                self._on_nd2_finished(error=msg[len("__ND2_ERROR__"):])
                continue

            # -- Deconvolution --
            if msg == "__DECONV_DONE__":
                self._on_deconv_finished()
                continue
            if msg.startswith("__DECONV_ERROR__"):
                self._on_deconv_finished(error=msg[len("__DECONV_ERROR__"):])
                continue
            if msg.startswith("__DECONV_LOG__"):
                self._deconv_log_append(msg[len("__DECONV_LOG__"):])
                continue
            if msg.startswith("__DECONV_PROGRESS__"):
                pct = int(msg[len("__DECONV_PROGRESS__"):])
                self.deconv_progress.config(value=pct)
                self.deconv_status.set(f"Deconvolving... {pct}%")
                continue
            if msg.startswith("__DECONV_RESULT__"):
                payload = msg[len("__DECONV_RESULT__"):]
                fname, status, message = payload.split("||", 2)
                self.deconv_tree.insert("", tk.END,
                                         values=(fname, status, message[:80]))
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

            # -- Phase Separation Analysis --
            if msg.startswith("__ANA_PROGRESS__"):
                pct = int(msg[len("__ANA_PROGRESS__"):])
                self.ana_progress.config(value=pct)
                self.ana_status.set(f"Processing... {pct}%")
                continue
            if msg.startswith("__ANA_LOG__"):
                self._ana_log_append(msg[len("__ANA_LOG__"):])
                continue
            if msg.startswith("__ANA_SUMMARY__"):
                self.ana_summary_text.set(msg[len("__ANA_SUMMARY__"):])
                continue
            if msg == "__ANA_MEASURE_DONE__":
                self._on_ana_measure_done()
                continue
            if msg == "__ANA_CSTAR_DONE__":
                self._on_ana_cstar_done()
                continue
            if msg == "__ANA_DROPFIT_DONE__":
                self._on_ana_dropfit_done()
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

            # -- Mask Comparison (single) --
            if msg.startswith("__CMP_DONE__"):
                self._on_cmp_finished(text=msg[len("__CMP_DONE__"):])
                continue
            if msg.startswith("__CMP_ERROR__"):
                self._on_cmp_finished(error=msg[len("__CMP_ERROR__"):])
                continue
            if msg.startswith("__CMP_OPEN__"):
                self._cmp_open_figure(msg[len("__CMP_OPEN__"):])
                continue

            # -- Mask Comparison (bulk) --
            if msg.startswith("__BCMP_PROGRESS__"):
                parts = msg[len("__BCMP_PROGRESS__"):].split("||")
                pct = int(parts[0])
                cur, tot, key = parts[1], parts[2], parts[3]
                self.cmp_bulk_progress.config(value=pct)
                self.cmp_bulk_status.set(f"Comparing {cur}/{tot}: {key}")
                continue
            if msg.startswith("__BCMP_DONE__"):
                self._on_bcmp_finished(text=msg[len("__BCMP_DONE__"):])
                continue
            if msg.startswith("__BCMP_ERROR__"):
                self._on_bcmp_finished(error=msg[len("__BCMP_ERROR__"):])
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
            "  - Deconvolution pre-processing (Richardson-Lucy / CARE-CSBDeep)\n"
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
