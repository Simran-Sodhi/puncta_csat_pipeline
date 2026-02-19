#!/usr/bin/env python3
"""
deconvolution.py — Deconvolution pre-processing for fluorescence microscopy.

Two back-ends:

1. **Huygens** (Commercial — requires a local HuCore installation)
   Generates a Tcl batch script and runs ``hucore`` on each image.
   Supports Classic Maximum Likelihood Estimation (CMLE) and
   Quick Maximum Likelihood Estimation (QMLE).

2. **CARE / CSBDeep** (Deep-learning — open-source, ``pip install csbdeep``)
   Applies a pre-trained CARE model for content-aware image restoration /
   deconvolution.  Users can supply their own model directory or use
   one of the bundled model names recognised by CSBDeep.

Both back-ends accept a directory of TIFF images, process each one, and
write the results into an output directory, calling an optional
``progress_callback(index, total, filename)`` after every image.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import tifffile as tiff


# ------------------------------------------------------------------ #
#  Shared helpers
# ------------------------------------------------------------------ #

_TIFF_GLOBS = ("*.tif", "*.tiff", "*.TIF", "*.TIFF")


def collect_tiffs(directory: str | Path) -> list[Path]:
    """Return sorted list of TIFF files in *directory*."""
    d = Path(directory)
    paths: list[Path] = []
    for g in _TIFF_GLOBS:
        paths.extend(d.glob(g))
    # deduplicate (case-insensitive systems) then sort
    seen: set[str] = set()
    unique: list[Path] = []
    for p in sorted(paths):
        key = str(p).lower()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


# ------------------------------------------------------------------ #
#  1.  Huygens deconvolution  (via HuCore CLI)
# ------------------------------------------------------------------ #

def _build_huygens_tcl(
    input_path: str,
    output_path: str,
    *,
    algorithm: str = "cmle",
    iterations: int = 40,
    snr: float = 20.0,
    microscope_type: str = "widefield",
    wavelength_ex: float = 488.0,
    wavelength_em: float = 520.0,
    na: float = 1.4,
    refractive_index_medium: float = 1.515,
    refractive_index_lens: float = 1.515,
    background_mode: str = "auto",
    quality_threshold: float = 0.05,
    brick_mode: str = "auto",
) -> str:
    """Return a HuCore Tcl batch script as a string."""
    alg_map = {"cmle": "cmle", "qmle": "qmle", "gmle": "gmle"}
    alg = alg_map.get(algorithm.lower(), "cmle")

    micro_map = {
        "widefield": "widefield",
        "confocal": "confocal",
        "spinning_disk": "nipkow",
        "multiphoton": "multiphoton",
    }
    micro = micro_map.get(microscope_type.lower(), "widefield")

    bg_map = {"auto": "auto", "lowest": "lowest", "manual": "manual"}
    bg = bg_map.get(background_mode.lower(), "auto")

    tcl = f"""\
# Auto-generated HuCore batch script
set img [img open "{input_path}"]

$img setp -micr {micro}
$img setp -ex {wavelength_ex}
$img setp -em {wavelength_em}
$img setp -na {na}
$img setp -ri {refractive_index_medium}
$img setp -ril {refractive_index_lens}

set result [$img decon -{alg} \\
    -it {iterations} \\
    -snr {snr} \\
    -bgMode {bg} \\
    -q {quality_threshold} \\
    -brMode {brick_mode}]

$result save "{output_path}" -type tiff
$img del
$result del
"""
    return tcl


def deconvolve_huygens(
    image_dir: str | Path,
    out_dir: str | Path,
    *,
    hucore_path: str = "hucore",
    algorithm: str = "cmle",
    iterations: int = 40,
    snr: float = 20.0,
    microscope_type: str = "widefield",
    wavelength_ex: float = 488.0,
    wavelength_em: float = 520.0,
    na: float = 1.4,
    refractive_index_medium: float = 1.515,
    refractive_index_lens: float = 1.515,
    background_mode: str = "auto",
    quality_threshold: float = 0.05,
    channel_index: int = 0,
    z_index: int = 0,
    progress_callback: Optional[Callable] = None,
) -> list[dict]:
    """Run Huygens deconvolution on every TIFF in *image_dir*.

    Parameters
    ----------
    hucore_path : str
        Path to the ``hucore`` executable.  If it is on ``$PATH`` the
        bare name ``"hucore"`` suffices.
    progress_callback : callable, optional
        ``callback(index, total, filename)`` called after each image.

    Returns
    -------
    list of dict
        Per-image summary: ``{"file": str, "status": str, "message": str}``.
    """
    image_dir = Path(image_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Verify hucore is available
    resolved = shutil.which(hucore_path)
    if resolved is None:
        raise FileNotFoundError(
            f"HuCore executable not found: {hucore_path!r}\n"
            "Install Huygens Professional / HuCore and ensure it is on your PATH,\n"
            "or provide the full path via the 'HuCore path' setting."
        )

    paths = collect_tiffs(image_dir)
    total = len(paths)
    results: list[dict] = []

    for idx, img_path in enumerate(paths, 1):
        stem = img_path.stem
        out_path = out_dir / f"{stem}_decon.tif"

        tcl = _build_huygens_tcl(
            str(img_path),
            str(out_path),
            algorithm=algorithm,
            iterations=iterations,
            snr=snr,
            microscope_type=microscope_type,
            wavelength_ex=wavelength_ex,
            wavelength_em=wavelength_em,
            na=na,
            refractive_index_medium=refractive_index_medium,
            refractive_index_lens=refractive_index_lens,
            background_mode=background_mode,
            quality_threshold=quality_threshold,
        )

        # Write Tcl script to a temp file and invoke hucore
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".tcl", delete=False
            ) as fp:
                fp.write(tcl)
                tcl_path = fp.name

            proc = subprocess.run(
                [resolved, "-noExecLog", "-script", tcl_path],
                capture_output=True,
                text=True,
                timeout=1800,  # 30 min per image
            )

            if proc.returncode != 0:
                results.append({
                    "file": stem, "status": "FAILED",
                    "message": proc.stderr[:500] or proc.stdout[:500],
                })
            else:
                results.append({
                    "file": stem, "status": "OK", "message": "",
                })

        except subprocess.TimeoutExpired:
            results.append({
                "file": stem, "status": "FAILED",
                "message": "Timeout (>30 min)",
            })
        except Exception as exc:
            results.append({
                "file": stem, "status": "FAILED", "message": str(exc),
            })
        finally:
            try:
                os.unlink(tcl_path)
            except OSError:
                pass

        if progress_callback is not None:
            progress_callback(idx, total, img_path.name)

    return results


# ------------------------------------------------------------------ #
#  2.  CARE / CSBDeep deep-learning deconvolution
# ------------------------------------------------------------------ #

def deconvolve_care(
    image_dir: str | Path,
    out_dir: str | Path,
    *,
    model_dir: str | Path,
    model_name: str = "my_care_model",
    axes: str = "YX",
    n_tiles: tuple[int, ...] = (1, 1),
    channel_index: int = 0,
    z_index: int = 0,
    normalise_input: bool = True,
    norm_pmin: float = 1.0,
    norm_pmax: float = 99.8,
    progress_callback: Optional[Callable] = None,
) -> list[dict]:
    """Apply a pre-trained CARE model to every TIFF in *image_dir*.

    Parameters
    ----------
    model_dir : path-like
        Parent directory containing the CARE model folder.
    model_name : str
        Name of the model subfolder inside *model_dir*.
    axes : str
        Axes string describing the input layout (e.g. ``"YX"``, ``"ZYX"``).
    n_tiles : tuple of int
        Tile counts per axis for memory-efficient prediction.
    normalise_input : bool
        Whether to percentile-normalise input before prediction.
    progress_callback : callable, optional
        ``callback(index, total, filename)`` called after each image.

    Returns
    -------
    list of dict
        Per-image summary.
    """
    try:
        from csbdeep.models import CARE
        from csbdeep.io import normalize as csb_normalize
    except ImportError:
        raise ImportError(
            "CSBDeep / CARE is not installed.\n"
            "Install with:  pip install csbdeep tensorflow\n"
            "  (GPU recommended: pip install tensorflow[and-cuda])"
        )

    image_dir = Path(image_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = CARE(config=None, name=model_name, basedir=str(model_dir))

    paths = collect_tiffs(image_dir)
    total = len(paths)
    results: list[dict] = []

    for idx, img_path in enumerate(paths, 1):
        stem = img_path.stem
        out_path = out_dir / f"{stem}_decon.tif"

        try:
            raw = tiff.imread(str(img_path))
            img2d = _extract_2d(raw, channel_index, z_index)
            img_in = img2d.astype(np.float32)

            if normalise_input:
                plo = np.percentile(img_in, norm_pmin)
                phi = np.percentile(img_in, norm_pmax)
                if phi > plo:
                    img_in = (img_in - plo) / (phi - plo)

            restored = model.predict(img_in, axes, n_tiles=n_tiles)
            restored = np.clip(restored, 0, None)

            # Save as 32-bit float TIFF
            tiff.imwrite(str(out_path), restored.astype(np.float32))

            results.append({
                "file": stem, "status": "OK", "message": "",
            })

        except Exception as exc:
            results.append({
                "file": stem, "status": "FAILED", "message": str(exc),
            })

        if progress_callback is not None:
            progress_callback(idx, total, img_path.name)

    return results


# ------------------------------------------------------------------ #
#  Private helpers
# ------------------------------------------------------------------ #

def _extract_2d(
    data: np.ndarray,
    channel_index: int = 0,
    z_index: int = 0,
) -> np.ndarray:
    """Best-effort extraction of a single 2-D plane from multi-dim data."""
    if data.ndim == 2:
        return data
    if data.ndim == 3:
        # (C, Y, X) or (Z, Y, X)
        if data.shape[0] <= 6:
            return data[channel_index]
        return data[z_index]
    if data.ndim == 4:
        # (Z, C, Y, X) or (C, Z, Y, X)
        if data.shape[1] <= 6:
            return data[z_index, channel_index]
        return data[channel_index, z_index]
    if data.ndim == 5:
        # (T, Z, C, Y, X) — take first time-point
        return data[0, z_index, channel_index]
    # Fallback: squeeze and hope for the best
    squeezed = np.squeeze(data)
    if squeezed.ndim == 2:
        return squeezed
    raise ValueError(
        f"Cannot extract 2-D plane from array with shape {data.shape}"
    )
