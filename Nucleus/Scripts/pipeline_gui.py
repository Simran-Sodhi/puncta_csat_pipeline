#!/usr/bin/env python3
"""
pipeline_gui.py

User-friendly tkinter GUI for the Puncta-CSAT segmentation pipeline.

Provides a step-by-step interface for non-coders to:
  1. Convert ND2 files to OME-TIFF
  2. Run Cellpose segmentation (nucleus / puncta / cytoplasm)
     - Cytoplasm mode: segments whole cells, then subtracts nucleus masks
  3. Run per-cell intensity & puncta analysis
  4. View results

Launch:
    python pipeline_gui.py
"""

import os
import sys
import threading
import queue
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

# ------------------------------------------------------------------ #
#  Locate the Scripts directory (where segmentation_utils.py lives)
# ------------------------------------------------------------------ #
# Works when pipeline_gui.py lives inside Nucleus/Scripts/ OR when
# it has been copied elsewhere.  In the latter case, the user is
# prompted to select the Scripts folder on first launch.

def _find_scripts_dir():
    """Return the path to Nucleus/Scripts, or None if not found."""
    candidate = Path(__file__).resolve().parent
    if (candidate / "segmentation_utils.py").exists():
        return candidate
    # Try one level up (e.g. user put gui in the repo root)
    for p in [candidate.parent, candidate.parent / "Nucleus" / "Scripts"]:
        if (p / "segmentation_utils.py").exists():
            return p
    return None

SCRIPT_DIR = _find_scripts_dir()

def _ensure_scripts_dir():
    """Make sure SCRIPT_DIR is set; prompt interactively if needed."""
    global SCRIPT_DIR
    if SCRIPT_DIR is not None:
        return SCRIPT_DIR
    # Ask the user via a simple dialog before the main window opens
    import tkinter as _tk
    from tkinter import filedialog as _fd, messagebox as _mb
    _root = _tk.Tk()
    _root.withdraw()
    _mb.showinfo(
        "Scripts folder required",
        "pipeline_gui.py was launched outside the Nucleus/Scripts directory.\n\n"
        "Please select the 'Nucleus/Scripts' folder that contains\n"
        "segmentation_utils.py, preprocessing/, etc."
    )
    chosen = _fd.askdirectory(title="Select Nucleus/Scripts folder")
    _root.destroy()
    if chosen and (Path(chosen) / "segmentation_utils.py").exists():
        SCRIPT_DIR = Path(chosen).resolve()
        return SCRIPT_DIR
    print("[ERROR] Could not locate segmentation_utils.py. "
          "Please run from Nucleus/Scripts/ or select the correct folder.")
    sys.exit(1)

SCRIPT_DIR = _ensure_scripts_dir()
sys.path.insert(0, str(SCRIPT_DIR))


# ------------------------------------------------------------------ #
#  Colour palette & style constants
# ------------------------------------------------------------------ #

BG_COLOR = "#f5f6fa"
HEADER_BG = "#2c3e50"
HEADER_FG = "#ecf0f1"
ACCENT = "#3498db"
ACCENT_HOVER = "#2980b9"
SUCCESS = "#27ae60"
WARNING = "#e67e22"
ERROR_COLOR = "#e74c3c"
CARD_BG = "#ffffff"
TEXT_COLOR = "#2c3e50"
SUBTLE_TEXT = "#7f8c8d"
FONT_FAMILY = "Segoe UI" if sys.platform == "win32" else "Helvetica"


# ------------------------------------------------------------------ #
#  Reusable widgets
# ------------------------------------------------------------------ #

class FolderPicker(ttk.Frame):
    """A labelled entry + Browse button for selecting a folder or file."""

    def __init__(self, parent, label, mode="directory", filetypes=None, **kw):
        super().__init__(parent, **kw)
        self.mode = mode
        self.filetypes = filetypes or [("All files", "*.*")]

        ttk.Label(self, text=label, font=(FONT_FAMILY, 10)).pack(
            anchor="w", padx=4, pady=(4, 0)
        )
        row = ttk.Frame(self)
        row.pack(fill="x", padx=4, pady=2)

        self.var = tk.StringVar()
        self.entry = ttk.Entry(row, textvariable=self.var, width=50)
        self.entry.pack(side="left", fill="x", expand=True)

        self.btn = ttk.Button(row, text="Browse...", command=self._browse, width=10)
        self.btn.pack(side="right", padx=(6, 0))

    def _browse(self):
        if self.mode == "directory":
            path = filedialog.askdirectory()
        elif self.mode == "savefile":
            path = filedialog.asksaveasfilename(filetypes=self.filetypes)
        else:
            path = filedialog.askopenfilename(filetypes=self.filetypes)
        if path:
            self.var.set(path)

    def get(self):
        return self.var.get().strip()


class NumberEntry(ttk.Frame):
    """A labelled entry for an integer or float value."""

    def __init__(self, parent, label, default=0, dtype=int, tooltip="", **kw):
        super().__init__(parent, **kw)
        self.dtype = dtype

        lbl = ttk.Label(self, text=label, font=(FONT_FAMILY, 10))
        lbl.pack(side="left", padx=(4, 8))
        if tooltip:
            lbl.bind("<Enter>", lambda e: self._show_tip(e, tooltip))
            lbl.bind("<Leave>", self._hide_tip)

        self.var = tk.StringVar(value=str(default))
        self.entry = ttk.Entry(self, textvariable=self.var, width=10)
        self.entry.pack(side="left")

        self._tip = None

    def _show_tip(self, event, text):
        self._tip = tw = tk.Toplevel(self)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{event.x_root + 15}+{event.y_root + 10}")
        lbl = tk.Label(tw, text=text, background="#ffffe0", relief="solid",
                       borderwidth=1, font=(FONT_FAMILY, 9), wraplength=300,
                       justify="left", padx=6, pady=4)
        lbl.pack()

    def _hide_tip(self, _event=None):
        if self._tip:
            self._tip.destroy()
            self._tip = None

    def get(self):
        raw = self.var.get().strip()
        if not raw:
            return self.dtype(0)
        return self.dtype(raw)


# ------------------------------------------------------------------ #
#  Step 1 – ND2 to OME-TIFF conversion
# ------------------------------------------------------------------ #

class Step1Frame(ttk.LabelFrame):
    def __init__(self, parent, log_callback):
        super().__init__(parent, text="  Step 1: Convert ND2 to OME-TIFF  ",
                         padding=12)
        self.log = log_callback

        self.nd2_picker = FolderPicker(
            self, "ND2 File:", mode="file",
            filetypes=[("ND2 files", "*.nd2"), ("All files", "*.*")]
        )
        self.nd2_picker.pack(fill="x", pady=2)

        self.out_picker = FolderPicker(self, "Output Directory:")
        self.out_picker.pack(fill="x", pady=2)

        params = ttk.Frame(self)
        params.pack(fill="x", pady=4)
        self.z_index = NumberEntry(
            params, "Z-plane index:", default=8,
            tooltip="Which Z-plane to extract from the stack (0-based)."
        )
        self.z_index.pack(side="left", padx=4)

        self.run_btn = ttk.Button(self, text="Convert", command=self._run,
                                  style="Accent.TButton")
        self.run_btn.pack(pady=(8, 4))

    def _run(self):
        nd2_path = self.nd2_picker.get()
        out_dir = self.out_picker.get()
        if not nd2_path or not out_dir:
            messagebox.showwarning("Missing input",
                                   "Please select both the ND2 file and output directory.")
            return
        z = self.z_index.get()
        self.log(f"\n{'='*60}\nStep 1: Converting ND2 -> OME-TIFF  (z={z})\n{'='*60}")
        self.run_btn.config(state="disabled")

        def task():
            try:
                from preprocessing.nd2_to_ome_tif import convert_nd2
                convert_nd2(nd2_path, out_dir, z_index=z)
                self.log("[DONE] ND2 conversion complete.")
            except ImportError as exc:
                self.log(f"[ERROR] Missing dependency: {exc}")
                self.log("  Install with: pip install aicsimageio")
            except Exception as exc:
                import traceback
                self.log(f"[ERROR] {exc}")
                self.log(traceback.format_exc())
            finally:
                self.run_btn.config(state="normal")

        threading.Thread(target=task, daemon=True).start()


# ------------------------------------------------------------------ #
#  Step 2 – Cellpose segmentation (nucleus / puncta / cytoplasm)
# ------------------------------------------------------------------ #

class Step2Frame(ttk.LabelFrame):
    def __init__(self, parent, log_callback):
        super().__init__(parent,
                         text="  Step 2: Cellpose Segmentation  ",
                         padding=12)
        self.log = log_callback

        self.input_picker = FolderPicker(
            self, "Input Images (folder or file):", mode="directory"
        )
        self.input_picker.pack(fill="x", pady=2)

        self.out_picker = FolderPicker(self, "Output Masks Directory:")
        self.out_picker.pack(fill="x", pady=2)

        # ---- Mode selector ----
        mode_frame = ttk.Frame(self)
        mode_frame.pack(fill="x", pady=4)
        ttk.Label(mode_frame, text="Mode:", font=(FONT_FAMILY, 10)).pack(
            side="left", padx=4
        )
        self.mode_var = tk.StringVar(value="nucleus")
        modes = ttk.Frame(mode_frame)
        modes.pack(side="left", padx=4)
        for label, val in [("Nucleus", "nucleus"), ("Puncta", "puncta"),
                           ("Cytoplasm", "cytoplasm")]:
            ttk.Radiobutton(modes, text=label, variable=self.mode_var,
                             value=val,
                             command=self._on_mode_change).pack(side="left", padx=8)

        # ---- Common parameters ----
        param_frame = ttk.Frame(self)
        param_frame.pack(fill="x", pady=4)

        self.diameter = NumberEntry(
            param_frame, "Diameter (px):", default=200, dtype=float,
            tooltip="Approximate object diameter in pixels.\n"
                    "Nucleus ~200, Puncta ~20,\n"
                    "Cytoplasm: 0 = auto-estimate."
        )
        self.diameter.pack(side="left", padx=4)

        self.channel_idx = NumberEntry(
            param_frame, "Channel:", default=2,
            tooltip="0-based channel index in the image.\n"
                    "Nucleus typically = 2, Puncta = 1,\n"
                    "Cytoplasm typically = 1."
        )
        self.channel_idx.pack(side="left", padx=4)

        self.z_idx = NumberEntry(
            param_frame, "Z-index:", default=5,
            tooltip="Z-plane to use (0-based)."
        )
        self.z_idx.pack(side="left", padx=4)

        param_frame2 = ttk.Frame(self)
        param_frame2.pack(fill="x", pady=4)

        self.min_size = NumberEntry(
            param_frame2, "Min object size (px):", default=10000,
            tooltip="Objects smaller than this will be removed.\n"
                    "Set 0 to disable."
        )
        self.min_size.pack(side="left", padx=4)

        self.gpu_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(param_frame2, text="Use GPU",
                         variable=self.gpu_var).pack(side="left", padx=16)

        self.edges_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(param_frame2, text="Remove edge objects",
                         variable=self.edges_var).pack(side="left", padx=8)

        # ---- Cytoplasm-specific parameters (shown/hidden) ----
        self.cyto_frame = ttk.LabelFrame(
            self,
            text="  Cytoplasm Options (cell - nucleus = cytoplasm)  ",
            padding=8,
        )
        # Initially hidden; _on_mode_change() manages visibility

        self.nuc_mask_picker = FolderPicker(
            self.cyto_frame,
            "Nucleus Masks Folder:",
            mode="directory",
        )
        self.nuc_mask_picker.pack(fill="x", pady=2)

        cyto_params = ttk.Frame(self.cyto_frame)
        cyto_params.pack(fill="x", pady=4)

        self.nuc_dilate = NumberEntry(
            cyto_params, "Nucleus dilation (px):", default=0,
            tooltip="Dilate the nucleus mask by this many pixels\n"
                    "before subtracting from the cell mask.\n"
                    "Useful to avoid including peri-nuclear signal.\n"
                    "Set 0 for no dilation."
        )
        self.nuc_dilate.pack(side="left", padx=4)

        self.min_nuc_px = NumberEntry(
            cyto_params, "Min nuc overlap (px):", default=10,
            tooltip="Minimum nucleus pixels that must overlap\n"
                    "a cell for the cell to be kept.\n"
                    "Cells without enough nuclear overlap\n"
                    "are removed as orphans."
        )
        self.min_nuc_px.pack(side="left", padx=4)

        self.min_overlap_frac = NumberEntry(
            cyto_params, "Min overlap fraction:", default=0.005,
            dtype=float,
            tooltip="Minimum fraction of cell area that must\n"
                    "overlap with nucleus to keep the cell.\n"
                    "Default: 0.005 (0.5%)."
        )
        self.min_overlap_frac.pack(side="left", padx=4)

        # ---- Run button ----
        self.run_btn = ttk.Button(self, text="Run Segmentation",
                                  command=self._run, style="Accent.TButton")
        self.run_btn.pack(pady=(8, 4))

        # Set initial preset values
        self._on_mode_change()

    def _on_mode_change(self):
        mode = self.mode_var.get()
        if mode == "nucleus":
            self.diameter.var.set("200")
            self.channel_idx.var.set("2")
            self.z_idx.var.set("5")
            self.min_size.var.set("10000")
            self.edges_var.set(True)
            self.cyto_frame.pack_forget()
        elif mode == "puncta":
            self.diameter.var.set("20")
            self.channel_idx.var.set("1")
            self.z_idx.var.set("8")
            self.min_size.var.set("0")
            self.edges_var.set(False)
            self.cyto_frame.pack_forget()
        elif mode == "cytoplasm":
            self.diameter.var.set("0")  # 0 = auto-estimate
            self.channel_idx.var.set("1")
            self.z_idx.var.set("5")
            self.min_size.var.set("80000")
            self.edges_var.set(True)
            # Show cytoplasm options (insert before run button)
            self.cyto_frame.pack(fill="x", pady=4,
                                 before=self.run_btn)

    def _run(self):
        input_path = self.input_picker.get()
        out_dir = self.out_picker.get()
        if not input_path or not out_dir:
            messagebox.showwarning("Missing input",
                                   "Please select input images and output directory.")
            return

        mode = self.mode_var.get()
        diameter = self.diameter.get()
        # Treat 0 as None (auto-estimate) for diameter
        if diameter == 0:
            diameter = None
        channel = self.channel_idx.get()
        z = self.z_idx.get()
        min_sz = self.min_size.get()
        gpu = self.gpu_var.get()
        rm_edges = self.edges_var.get()

        is_cyto = mode == "cytoplasm"
        nuc_mask_dir = None
        nuc_dilate_px = 0
        min_nuc_pixels = 10
        min_overlap_frac = 0.005

        if is_cyto:
            nuc_mask_dir = self.nuc_mask_picker.get()
            if not nuc_mask_dir:
                messagebox.showwarning(
                    "Missing input",
                    "Cytoplasm mode requires a Nucleus Masks Folder.\n"
                    "Please run nucleus segmentation first, then point\n"
                    "to the folder containing nucleus mask TIFFs."
                )
                return
            nuc_dilate_px = self.nuc_dilate.get()
            min_nuc_pixels = self.min_nuc_px.get()
            min_overlap_frac = self.min_overlap_frac.get()

        self.log(f"\n{'='*60}\n"
                 f"Step 2: Cellpose Segmentation ({mode})\n"
                 f"  diameter={diameter}, channel={channel}, z={z}\n"
                 f"  min_size={min_sz}, gpu={gpu}, remove_edges={rm_edges}")
        if is_cyto:
            self.log(f"  nuc_mask_dir={nuc_mask_dir}\n"
                     f"  nuc_dilate_px={nuc_dilate_px}, "
                     f"min_nuc_pixels={min_nuc_pixels}, "
                     f"min_overlap_frac={min_overlap_frac}")
        self.log(f"{'='*60}")
        self.run_btn.config(state="disabled")

        def task():
            try:
                from segmentation_utils import (
                    load_image_2d, auto_lut_clip, ensure_2d,
                    run_cellpose, postprocess_mask,
                    save_mask, save_triptych,
                    save_cytoplasm_triptych,
                    collect_image_paths,
                    compute_cytoplasm_mask,
                )
                from cellpose import models
                import numpy as np
                import tifffile as tiff
                import re

                image_paths = collect_image_paths(input_path)
                if not image_paths:
                    self.log("[ERROR] No TIFF images found.")
                    return

                self.log(f"Found {len(image_paths)} image(s). Loading model...")
                model = models.Cellpose(gpu=gpu, model_type="cyto3")
                outdir = Path(out_dir)
                outdir.mkdir(parents=True, exist_ok=True)
                trip_dir = outdir / "triptychs"
                trip_dir.mkdir(parents=True, exist_ok=True)

                for i, img_path in enumerate(image_paths, 1):
                    self.log(f"  [{i}/{len(image_paths)}] {img_path.name}")
                    try:
                        img2d = load_image_2d(img_path,
                                              channel_index=channel,
                                              z_index=z)
                        img_norm = auto_lut_clip(img2d)
                        masks = run_cellpose(img_norm, model=model,
                                             diameter=diameter)
                        masks = postprocess_mask(masks,
                                                 min_size=min_sz,
                                                 remove_edges=rm_edges)

                        stem = img_path.stem

                        if is_cyto:
                            # Find matching nucleus mask
                            nuc_path = self._find_nuc_mask(
                                nuc_mask_dir, stem)
                            if nuc_path is None:
                                self.log(
                                    f"    [WARN] No nucleus mask for "
                                    f"{stem}, saving whole-cell only.")
                                save_mask(masks,
                                          outdir / f"{stem}_cell_masks.tif")
                                save_triptych(
                                    img_norm, masks,
                                    trip_dir / f"{stem}_cell_triptych.png")
                                continue

                            nuc_m = ensure_2d(tiff.imread(nuc_path))
                            if nuc_m.shape != masks.shape:
                                self.log(
                                    f"    [WARN] Shape mismatch: "
                                    f"cell {masks.shape} vs "
                                    f"nucleus {nuc_m.shape}, skipping.")
                                save_mask(masks,
                                          outdir / f"{stem}_cell_masks.tif")
                                continue

                            cyto_mask, kept, orphans = \
                                compute_cytoplasm_mask(
                                    masks, nuc_m,
                                    nuc_dilate_px=nuc_dilate_px,
                                    min_nuc_pixels=min_nuc_pixels,
                                    min_overlap_frac=min_overlap_frac,
                                )
                            self.log(
                                f"    Kept {len(kept)} cells, "
                                f"removed {len(orphans)} orphans")

                            save_mask(masks,
                                      outdir / f"{stem}_cell_masks.tif")
                            save_mask(cyto_mask,
                                      outdir / f"{stem}_cyto_masks.tif")
                            save_cytoplasm_triptych(
                                img_norm, masks, nuc_m, cyto_mask,
                                trip_dir / f"{stem}_cyto_triptych.png")
                        else:
                            save_mask(masks,
                                      outdir / f"{stem}_cyto3_masks.tif")
                            save_triptych(
                                img_norm, masks,
                                trip_dir / f"{stem}_triptych.png")
                    except Exception as e:
                        self.log(f"    [ERROR] {e}")

                self.log("[DONE] Segmentation complete.")
            except Exception as exc:
                self.log(f"[ERROR] {exc}")
            finally:
                self.run_btn.config(state="normal")

        threading.Thread(target=task, daemon=True).start()

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


# ------------------------------------------------------------------ #
#  Step 3 – Intensity & puncta analysis
# ------------------------------------------------------------------ #

class Step3Frame(ttk.LabelFrame):
    def __init__(self, parent, log_callback):
        super().__init__(parent, text="  Step 3: Intensity & Puncta Analysis  ",
                         padding=12)
        self.log = log_callback

        self.nuc_picker = FolderPicker(self, "Nucleus Masks Folder:")
        self.nuc_picker.pack(fill="x", pady=2)

        self.puncta_picker = FolderPicker(self, "Puncta Masks Folder:")
        self.puncta_picker.pack(fill="x", pady=2)

        self.intensity_picker = FolderPicker(self, "Raw OME-TIFF Images Folder:")
        self.intensity_picker.pack(fill="x", pady=2)

        self.csv_picker = FolderPicker(
            self, "Output CSV File:", mode="savefile",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        self.csv_picker.pack(fill="x", pady=2)

        # Parameters
        params = ttk.Frame(self)
        params.pack(fill="x", pady=4)

        self.intensity_ch = NumberEntry(
            params, "Nucleus channel:", default=2,
            tooltip="Channel index for nucleus intensity measurement."
        )
        self.intensity_ch.pack(side="left", padx=4)

        self.puncta_ch = NumberEntry(
            params, "Puncta channel:", default=1,
            tooltip="Channel index for puncta intensity."
        )
        self.puncta_ch.pack(side="left", padx=4)

        self.min_area = NumberEntry(
            params, "Min puncta area:", default=5,
            tooltip="Minimum puncta pixel overlap to count as puncta-positive."
        )
        self.min_area.pack(side="left", padx=4)

        params2 = ttk.Frame(self)
        params2.pack(fill="x", pady=4)

        self.open_radius = NumberEntry(
            params2, "Puncta open radius:", default=1,
            tooltip="Morphological opening radius to clean puncta mask.\n"
                    "Set 0 to disable."
        )
        self.open_radius.pack(side="left", padx=4)

        self.trip_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(params2, text="Generate QC triptychs",
                         variable=self.trip_var).pack(side="left", padx=16)

        self.trip_dir_picker = FolderPicker(self, "Triptych Output Folder (optional):")
        self.trip_dir_picker.pack(fill="x", pady=2)

        self.run_btn = ttk.Button(self, text="Run Analysis",
                                  command=self._run, style="Accent.TButton")
        self.run_btn.pack(pady=(8, 4))

    def _run(self):
        nuc = self.nuc_picker.get()
        puncta = self.puncta_picker.get()
        intensity = self.intensity_picker.get()
        csv_path = self.csv_picker.get()

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

        int_ch = self.intensity_ch.get()
        pun_ch = self.puncta_ch.get()
        min_a = self.min_area.get()
        open_r = self.open_radius.get()
        trip = self.trip_var.get()
        trip_dir = self.trip_dir_picker.get() or None

        self.log(f"\n{'='*60}\n"
                 f"Step 3: Intensity & Puncta Analysis\n"
                 f"  nuc_ch={int_ch}, puncta_ch={pun_ch}, min_area={min_a}\n"
                 f"{'='*60}")
        self.run_btn.config(state="disabled")

        def task():
            try:
                from puncta_detection.mean_intensity_and_puncta import main as run_analysis
                run_analysis(
                    nuc_dir=nuc,
                    puncta_dir=puncta,
                    intensity_dir=intensity,
                    out_csv=csv_path,
                    min_puncta_area=min_a,
                    puncta_open_radius=open_r,
                    make_triptychs=trip,
                    triptych_out_dir=trip_dir,
                    intensity_channel=int_ch,
                    puncta_channel=pun_ch,
                )
                self.log(f"[DONE] Analysis complete. CSV saved to: {csv_path}")
            except Exception as exc:
                self.log(f"[ERROR] {exc}")
            finally:
                self.run_btn.config(state="normal")

        threading.Thread(target=task, daemon=True).start()


# ------------------------------------------------------------------ #
#  Thread-safe logging via a queue
# ------------------------------------------------------------------ #

class LogPanel(ttk.LabelFrame):
    """Scrolled text area that receives log messages via a queue."""

    def __init__(self, parent, **kw):
        super().__init__(parent, text="  Log Output  ", padding=8, **kw)
        self.text = scrolledtext.ScrolledText(
            self, height=12, state="disabled", wrap="word",
            font=("Consolas" if sys.platform == "win32" else "Monospace", 9),
            background="#1e272e", foreground="#d2dae2",
            insertbackground="#d2dae2",
        )
        self.text.pack(fill="both", expand=True)
        self._queue = queue.Queue()

        # Tag for different message types
        self.text.tag_config("error", foreground="#ff6b6b")
        self.text.tag_config("done", foreground="#7bed9f")
        self.text.tag_config("warn", foreground="#ffa502")

    def append(self, msg):
        """Thread-safe append (can be called from any thread)."""
        self._queue.put(msg)

    def poll(self):
        """Called from the main thread to flush queued messages."""
        while not self._queue.empty():
            msg = self._queue.get_nowait()
            self.text.config(state="normal")
            # Determine tag
            tag = None
            if "[ERROR]" in msg:
                tag = "error"
            elif "[DONE]" in msg:
                tag = "done"
            elif "[WARN]" in msg:
                tag = "warn"
            self.text.insert("end", msg + "\n", tag)
            self.text.see("end")
            self.text.config(state="disabled")


# ------------------------------------------------------------------ #
#  Main application window
# ------------------------------------------------------------------ #

class PipelineGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Puncta-CSAT Segmentation Pipeline")
        self.geometry("820x950")
        self.minsize(700, 700)
        self.configure(bg=BG_COLOR)

        self._apply_theme()
        self._build_ui()
        self._poll_log()

    # ---- theming ---- #

    def _apply_theme(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure("TFrame", background=BG_COLOR)
        style.configure("TLabel", background=BG_COLOR, foreground=TEXT_COLOR,
                         font=(FONT_FAMILY, 10))
        style.configure("TLabelframe", background=BG_COLOR,
                         foreground=TEXT_COLOR, font=(FONT_FAMILY, 11, "bold"))
        style.configure("TLabelframe.Label", background=BG_COLOR,
                         foreground=ACCENT, font=(FONT_FAMILY, 11, "bold"))
        style.configure("TButton", font=(FONT_FAMILY, 10), padding=6)
        style.configure("TEntry", padding=4)
        style.configure("TCheckbutton", background=BG_COLOR,
                         font=(FONT_FAMILY, 10))
        style.configure("TRadiobutton", background=BG_COLOR,
                         font=(FONT_FAMILY, 10))

        # Accent button style
        style.configure("Accent.TButton",
                         background=ACCENT, foreground="white",
                         font=(FONT_FAMILY, 10, "bold"), padding=8)
        style.map("Accent.TButton",
                   background=[("active", ACCENT_HOVER), ("disabled", SUBTLE_TEXT)])

    # ---- UI construction ---- #

    def _build_ui(self):
        # Header
        header = tk.Frame(self, bg=HEADER_BG, height=56)
        header.pack(fill="x")
        header.pack_propagate(False)
        tk.Label(
            header,
            text="Puncta-CSAT Segmentation Pipeline",
            bg=HEADER_BG, fg=HEADER_FG,
            font=(FONT_FAMILY, 16, "bold"),
        ).pack(side="left", padx=16, pady=12)
        tk.Label(
            header,
            text="Cellpose 3 + OME-TIFF",
            bg=HEADER_BG, fg=SUBTLE_TEXT,
            font=(FONT_FAMILY, 10),
        ).pack(side="right", padx=16)

        # Scrollable body
        canvas = tk.Canvas(self, bg=BG_COLOR, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.body = ttk.Frame(canvas)

        self.body.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.body, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _on_linux_scroll(event):
            if event.num == 4:
                canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                canvas.yview_scroll(1, "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        canvas.bind_all("<Button-4>", _on_linux_scroll)
        canvas.bind_all("<Button-5>", _on_linux_scroll)

        # Log panel (created first so steps can reference it)
        self.log_panel = LogPanel(self.body)

        # Steps
        self.step1 = Step1Frame(self.body, self.log_panel.append)
        self.step1.pack(fill="x", padx=12, pady=(12, 6))

        self.step2 = Step2Frame(self.body, self.log_panel.append)
        self.step2.pack(fill="x", padx=12, pady=6)

        self.step3 = Step3Frame(self.body, self.log_panel.append)
        self.step3.pack(fill="x", padx=12, pady=6)

        self.log_panel.pack(fill="both", expand=True, padx=12, pady=(6, 12))

    # ---- log polling ---- #

    def _poll_log(self):
        self.log_panel.poll()
        self.after(100, self._poll_log)


# ------------------------------------------------------------------ #
#  Redirect print() to the GUI log
# ------------------------------------------------------------------ #

class PrintRedirector:
    """Captures print() output and forwards it to the log panel."""

    def __init__(self, log_func):
        self._log = log_func
        self._buffer = ""

    def write(self, text):
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line:
                self._log(line)

    def flush(self):
        if self._buffer:
            self._log(self._buffer)
            self._buffer = ""


# ------------------------------------------------------------------ #
#  Entry point
# ------------------------------------------------------------------ #

def main():
    app = PipelineGUI()

    # Redirect stdout/stderr so print() from pipeline scripts appears in the log
    redirector = PrintRedirector(app.log_panel.append)
    sys.stdout = redirector
    sys.stderr = redirector

    app.mainloop()


if __name__ == "__main__":
    main()
