#!/usr/bin/env python3
"""
deconvolution.py — Deconvolution pre-processing for fluorescence microscopy.

Two back-ends:

1. **Richardson-Lucy** (classical, open-source, no external dependencies)
   Iterative RL deconvolution with a theoretical or measured PSF.
   Supports early-stopping and Total-Variation (TV) regularisation.
   Optional pre-denoising step for noisy data (NLM, bilateral,
   Gaussian, or median) to stabilise convergence.

2. **CARE / CSBDeep** (Deep-learning — open-source, ``pip install csbdeep``)
   Applies a pre-trained CARE model for content-aware image restoration /
   deconvolution.

Both back-ends accept a directory of TIFF images, process each one, and
write the results into an output directory, calling an optional
``progress_callback(index, total, filename)`` after every image.
"""

from __future__ import annotations

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
    seen: set[str] = set()
    unique: list[Path] = []
    for p in sorted(paths):
        key = str(p).lower()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


# ------------------------------------------------------------------ #
#  OME-TIFF metadata reader (for auto-populating optical parameters)
# ------------------------------------------------------------------ #

def read_ome_metadata(path: str | Path) -> dict:
    """Read optical metadata from an OME-TIFF file.

    Returns a dict with keys (any may be ``None`` if not present):
        ``emission_nm``, ``excitation_nm``, ``na``, ``pixel_size_nm``,
        ``pixel_size_um``, ``immersion_ri``, ``magnification``,
        ``objective_name``, ``channel_name``, ``pinhole_um``.

    The function inspects OME-XML first, then falls back to tifffile's
    ``imagej_metadata`` / ``description`` fields.
    """
    import xml.etree.ElementTree as ET

    result: dict = {
        "emission_nm": None,
        "excitation_nm": None,
        "na": None,
        "pixel_size_nm": None,
        "pixel_size_um": None,
        "immersion_ri": None,
        "magnification": None,
        "objective_name": None,
        "channel_name": None,
        "pinhole_um": None,
    }

    path = Path(path)
    if not path.exists():
        return result

    with tiff.TiffFile(str(path)) as tf:
        # --- Try OME-XML ---
        if tf.ome_metadata:
            _parse_ome_xml(tf.ome_metadata, result)

        # --- Fallback: tifffile metadata dict (imagej / description) ---
        if result["na"] is None:
            _parse_tiff_metadata(tf, result)

    # Derive pixel_size_nm from um if needed
    if result["pixel_size_nm"] is None and result["pixel_size_um"] is not None:
        result["pixel_size_nm"] = result["pixel_size_um"] * 1000.0
    if result["pixel_size_um"] is None and result["pixel_size_nm"] is not None:
        result["pixel_size_um"] = result["pixel_size_nm"] / 1000.0

    return result


def _parse_ome_xml(ome_xml: str, result: dict) -> None:
    """Extract metadata from OME-XML string."""
    import xml.etree.ElementTree as ET

    ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}

    try:
        root = ET.fromstring(ome_xml)
    except ET.ParseError:
        return

    # Try multiple namespace patterns
    for ns_uri in [
        "http://www.openmicroscopy.org/Schemas/OME/2016-06",
        "",
    ]:
        ns_prefix = f"{{{ns_uri}}}" if ns_uri else ""

        # Pixels element — physical sizes
        for pixels in root.iter(f"{ns_prefix}Pixels"):
            px = pixels.get("PhysicalSizeX")
            if px:
                try:
                    unit = pixels.get("PhysicalSizeXUnit", "µm")
                    val = float(px)
                    if "nm" in unit.lower():
                        result["pixel_size_nm"] = val
                    else:
                        result["pixel_size_um"] = val
                except (ValueError, TypeError):
                    pass

        # Channel element — wavelengths, name
        for channel in root.iter(f"{ns_prefix}Channel"):
            if result["channel_name"] is None:
                result["channel_name"] = channel.get("Name")
            em = channel.get("EmissionWavelength")
            if em and result["emission_nm"] is None:
                try:
                    result["emission_nm"] = float(em)
                except (ValueError, TypeError):
                    pass
            ex = channel.get("ExcitationWavelength")
            if ex and result["excitation_nm"] is None:
                try:
                    result["excitation_nm"] = float(ex)
                except (ValueError, TypeError):
                    pass
            ph = channel.get("PinholeSize")
            if ph and result["pinhole_um"] is None:
                try:
                    result["pinhole_um"] = float(ph)
                except (ValueError, TypeError):
                    pass

        # Objective element — NA, magnification, name, RI
        for obj in root.iter(f"{ns_prefix}Objective"):
            na = obj.get("LensNA")
            if na and result["na"] is None:
                try:
                    result["na"] = float(na)
                except (ValueError, TypeError):
                    pass
            mag = obj.get("NominalMagnification")
            if mag and result["magnification"] is None:
                try:
                    result["magnification"] = float(mag)
                except (ValueError, TypeError):
                    pass
            if result["objective_name"] is None:
                result["objective_name"] = obj.get("Model") or obj.get("ID")
            ri = obj.get("ImmersionRefractiveIndex") or obj.get("RefractiveIndex")
            if ri and result["immersion_ri"] is None:
                try:
                    result["immersion_ri"] = float(ri)
                except (ValueError, TypeError):
                    pass

        # If we found something, don't try other namespace
        if result["na"] is not None or result["emission_nm"] is not None:
            break


def _parse_tiff_metadata(tf, result: dict) -> None:
    """Fallback: parse metadata from tifffile's imagej or description dict."""
    # Try structured OME metadata dict (tifffile >= 2021)
    for page in tf.pages[:1]:
        desc = page.description
        if not desc:
            continue
        # Look for key=value patterns in description
        for line in desc.replace(";", "\n").split("\n"):
            line = line.strip()
            kv = line.split("=", 1)
            if len(kv) != 2:
                continue
            k, v = kv[0].strip().lower(), kv[1].strip()
            try:
                if "lensna" in k or k == "na":
                    result["na"] = result["na"] or float(v)
                elif "emissionwavelength" in k:
                    result["emission_nm"] = result["emission_nm"] or float(v)
                elif "excitationwavelength" in k:
                    result["excitation_nm"] = result["excitation_nm"] or float(v)
                elif "physicalsizex" in k:
                    result["pixel_size_um"] = result["pixel_size_um"] or float(v)
                elif "immersionrefractiveindex" in k:
                    result["immersion_ri"] = result["immersion_ri"] or float(v)
            except (ValueError, TypeError):
                continue


# ------------------------------------------------------------------ #
#  PSF generation
# ------------------------------------------------------------------ #

def generate_widefield_psf(
    wavelength_em: float = 520.0,
    na: float = 1.4,
    pixel_size_nm: float = 65.0,
    n_immersion: float = 1.515,
    magnification: float = 0.0,
    psf_size: int = 0,
) -> np.ndarray:
    """Generate a 2-D widefield PSF using the scalar diffraction Airy model.

    Uses the Born & Wolf scalar diffraction formula for a widefield
    fluorescence microscope.  The PSF is the squared Airy pattern:

        PSF(r) = [ 2 * J1(v) / v ]²

    where ``v = 2 * pi * NA * r / wavelength_em`` and *r* is the radial
    distance in the sample plane (nm).

    This is the same physical model used by commercial software (e.g.
    NIS Elements "Widefield" modality).

    Parameters
    ----------
    wavelength_em : float
        Emission wavelength in nm.
    na : float
        Numerical aperture.
    pixel_size_nm : float
        Pixel size in nm.  If *magnification* > 0 this is treated as the
        **camera** pixel size and divided by magnification to get the
        sample-plane pixel size.  Otherwise it is used directly as the
        sample-plane pixel size.
    n_immersion : float
        Refractive index of the immersion medium (oil ≈ 1.515,
        water ≈ 1.33, air ≈ 1.0).  NA is clamped to this value
        (physical limit).
    magnification : float
        Objective magnification (e.g. 60, 100).  When > 0, pixel_size_nm
        is divided by this value.  Set to 0 to use pixel_size_nm as-is
        (i.e. already at the sample plane).
    psf_size : int
        Kernel diameter in pixels.  0 = auto (covers ~4 Airy rings).
    """
    from scipy.special import j1

    # Clamp NA to the physical limit imposed by the immersion medium
    if na > n_immersion:
        na = n_immersion

    # Sample-plane pixel size
    sample_px_nm = pixel_size_nm
    if magnification > 0:
        sample_px_nm = pixel_size_nm / magnification

    # Airy radius (first zero of J1) = 0.61 * lambda / NA
    airy_radius_nm = 0.61 * wavelength_em / na
    airy_radius_px = airy_radius_nm / sample_px_nm

    if psf_size <= 0:
        # Cover ~4 Airy rings for accurate deconvolution
        psf_size = max(5, int(np.ceil(8 * airy_radius_px)) | 1)
    half = psf_size // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float64)

    # Radial distance in nm at the sample plane
    r_nm = np.sqrt(x ** 2 + y ** 2) * sample_px_nm

    # Normalised radial coordinate: v = 2*pi*NA*r / lambda
    v = 2.0 * np.pi * na * r_nm / wavelength_em

    # Airy pattern: [2*J1(v)/v]^2   (value at v=0 is 1 by L'Hôpital)
    psf = np.ones_like(v)
    mask = v > 0
    psf[mask] = (2.0 * j1(v[mask]) / v[mask]) ** 2

    psf /= psf.sum()
    return psf.astype(np.float32)


def generate_gaussian_psf(
    wavelength_em: float = 520.0,
    na: float = 1.4,
    pixel_size_nm: float = 65.0,
    psf_size: int = 0,
) -> np.ndarray:
    """Generate a 2-D Gaussian approximation of the PSF (legacy).

    Kept for backward compatibility.  Prefer :func:`generate_widefield_psf`.
    """
    sigma_nm = 0.21 * wavelength_em / na
    sigma_px = sigma_nm / pixel_size_nm

    if psf_size <= 0:
        psf_size = max(3, int(np.ceil(6 * sigma_px)) | 1)
    half = psf_size // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float64)
    psf = np.exp(-(x ** 2 + y ** 2) / (2 * sigma_px ** 2))
    psf /= psf.sum()
    return psf.astype(np.float32)


def load_measured_psf(path: str | Path) -> np.ndarray:
    """Load a measured PSF from a TIFF file and normalise to sum=1."""
    psf = tiff.imread(str(path)).astype(np.float32)
    if psf.ndim == 3:
        # Take the middle Z-plane of a 3-D PSF
        psf = psf[psf.shape[0] // 2]
    psf = np.clip(psf, 0, None)
    total = psf.sum()
    if total > 0:
        psf /= total
    return psf


# ------------------------------------------------------------------ #
#  Pre-denoising  (run before RL to stabilise noisy inputs)
# ------------------------------------------------------------------ #

def predenoise(
    image: np.ndarray,
    method: str = "nlm",
    sigma_est: float = 0.0,
    **kwargs,
) -> np.ndarray:
    """Denoise *image* prior to deconvolution.

    Parameters
    ----------
    method : {"nlm", "bilateral", "gaussian", "median"}
        * **nlm** — Non-local means (best SNR preservation; recommended).
        * **bilateral** — Edge-preserving bilateral filter.
        * **gaussian** — Gaussian blur (fast but blurs edges).
        * **median** — Median filter (good for salt-and-pepper noise).
    sigma_est : float
        Estimated noise standard deviation.
        - For *nlm*: used as ``sigma`` if > 0; otherwise auto-estimated.
        - For *gaussian*: used as Gaussian sigma in pixels.
        - For *bilateral*: used as ``sigma_spatial``.
        - For *median*: disk radius (rounded to int, min 1).
    """
    from skimage import restoration, filters, morphology

    img = image.astype(np.float32)

    if method == "nlm":
        if sigma_est <= 0:
            sigma_est = _estimate_sigma(img)
        # patch_size and patch_distance are standard defaults for 2-D
        return restoration.denoise_nl_means(
            img,
            h=1.15 * sigma_est,
            patch_size=5,
            patch_distance=6,
            fast_mode=True,
        ).astype(np.float32)

    if method == "bilateral":
        from skimage.restoration import denoise_bilateral
        sigma_sp = max(1.0, sigma_est) if sigma_est > 0 else 1.0
        return denoise_bilateral(
            img,
            sigma_color=None,  # auto
            sigma_spatial=sigma_sp,
        ).astype(np.float32)

    if method == "gaussian":
        sigma = max(0.5, sigma_est) if sigma_est > 0 else 0.7
        return filters.gaussian(img, sigma=sigma).astype(np.float32)

    if method == "median":
        r = max(1, int(round(sigma_est))) if sigma_est > 0 else 1
        se = morphology.disk(r)
        return filters.median(img, se).astype(np.float32)

    raise ValueError(f"Unknown pre-denoise method: {method!r}")


def _estimate_sigma(image: np.ndarray) -> float:
    """Robust noise estimate (MAD of Laplacian)."""
    from scipy.ndimage import laplace
    lap = laplace(image.astype(np.float64))
    mad = np.median(np.abs(lap - np.median(lap)))
    return float(mad * 1.4826 / np.sqrt(20))


# ------------------------------------------------------------------ #
#  Richardson-Lucy deconvolution
# ------------------------------------------------------------------ #

def _richardson_lucy_tv(
    image: np.ndarray,
    psf: np.ndarray,
    iterations: int = 30,
    tv_lambda: float = 0.0,
    early_stop_delta: float = 0.0,
) -> np.ndarray:
    """RL deconvolution with edge padding, optional TV and early stopping.

    Parameters
    ----------
    tv_lambda : float
        Strength of Total-Variation regularisation.  0 = disabled.
        Values around 0.001–0.01 are typical.
    early_stop_delta : float
        If the relative change between iterations drops below this
        threshold the loop terminates.  0 = run all iterations.
    """
    from scipy.signal import fftconvolve

    img = np.clip(image.astype(np.float64), 1e-12, None)
    psf64 = psf.astype(np.float64)
    psf_mirror = psf64[::-1, ::-1]

    # Pad image with reflected borders to reduce edge artifacts
    pad_w = psf.shape[0]
    img_padded = np.pad(img, pad_w, mode="reflect")

    estimate = img_padded.copy()
    prev_estimate = estimate.copy()

    for i in range(iterations):
        reblurred = fftconvolve(estimate, psf64, mode="same")
        reblurred = np.clip(reblurred, 1e-12, None)
        ratio = img_padded / reblurred
        correction = fftconvolve(ratio, psf_mirror, mode="same")

        if tv_lambda > 0:
            tv_grad = _tv_gradient(estimate)
            correction = correction / (1.0 + tv_lambda * tv_grad)

        estimate *= correction
        estimate = np.clip(estimate, 0, None)

        if early_stop_delta > 0:
            delta = np.mean(np.abs(estimate - prev_estimate)) / (np.mean(estimate) + 1e-12)
            if delta < early_stop_delta:
                break
            prev_estimate = estimate.copy()

    # Remove padding
    result = estimate[pad_w:-pad_w, pad_w:-pad_w]
    return result.astype(np.float32)


def _tv_gradient(image: np.ndarray) -> np.ndarray:
    """Gradient magnitude for isotropic TV regularisation."""
    gy = np.zeros_like(image)
    gx = np.zeros_like(image)
    gy[:-1, :] = np.diff(image, axis=0)
    gx[:, :-1] = np.diff(image, axis=1)
    return np.sqrt(gy ** 2 + gx ** 2 + 1e-12)


# ------------------------------------------------------------------ #
#  1.  Batch Richardson-Lucy deconvolution
# ------------------------------------------------------------------ #

def deconvolve_richardson_lucy(
    image_dir: str | Path,
    out_dir: str | Path,
    *,
    # PSF parameters
    psf_type: str = "theoretical",
    psf_path: str = "",
    wavelength_em: float = 520.0,
    na: float = 1.4,
    pixel_size_nm: float = 65.0,
    n_immersion: float = 1.515,
    magnification: float = 0.0,
    psf_size: int = 0,
    # RL parameters
    iterations: int = 30,
    tv_lambda: float = 0.0,
    early_stop_delta: float = 0.0,
    subtract_background: bool = True,
    # Pre-denoising
    predenoise_enabled: bool = False,
    predenoise_method: str = "nlm",
    predenoise_sigma: float = 0.0,
    # Image slicing
    channel_index: int = 0,
    z_index: int = 0,
    progress_callback: Optional[Callable] = None,
) -> list[dict]:
    """Run Richardson-Lucy deconvolution on every TIFF in *image_dir*.

    Parameters
    ----------
    psf_type : {"theoretical", "measured"}
        Use a widefield Airy PSF or load a measured PSF file.
    psf_path : str
        Path to a measured PSF TIFF (only when ``psf_type="measured"``).
    n_immersion : float
        Refractive index of the immersion medium (oil ≈ 1.515,
        water ≈ 1.33, air ≈ 1.0).
    magnification : float
        Objective magnification.  When > 0, *pixel_size_nm* is treated as
        camera pixel size and divided by magnification.  0 = pixel_size_nm
        is already the sample-plane pixel size.
    iterations : int
        Number of RL iterations (10–50 recommended).
    tv_lambda : float
        TV-regularisation weight.  0 disables it.
    early_stop_delta : float
        Relative convergence threshold for early stopping.  0 = disabled.
    subtract_background : bool
        Automatically subtract camera / fluorescence background before
        RL.  Greatly improves convergence and prevents RL from wasting
        iterations trying to explain uniform background.
    predenoise_enabled : bool
        Apply a denoising filter before RL.  Disabled by default — RL
        works best on unsmoothed data with a correct PSF.
    predenoise_method : {"nlm", "bilateral", "gaussian", "median"}
        Which denoiser to apply.  NLM is recommended if enabled.
    predenoise_sigma : float
        Noise estimate or filter parameter.  0 = auto-estimate.
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

    # Build / load PSF once
    if psf_type == "measured":
        if not psf_path:
            raise ValueError("psf_path must be set when psf_type='measured'")
        psf = load_measured_psf(psf_path)
    else:
        psf = generate_widefield_psf(
            wavelength_em=wavelength_em,
            na=na,
            pixel_size_nm=pixel_size_nm,
            n_immersion=n_immersion,
            magnification=magnification,
            psf_size=psf_size,
        )

    paths = collect_tiffs(image_dir)
    total = len(paths)
    results: list[dict] = []

    for idx, img_path in enumerate(paths, 1):
        stem = img_path.stem
        out_path = out_dir / f"{stem}_decon.tif"

        try:
            raw = tiff.imread(str(img_path))
            orig_dtype = raw.dtype
            img2d = _extract_2d(raw, channel_index, z_index)
            img_in = img2d.astype(np.float32)

            # Subtract background (critical for RL convergence)
            bg_level = 0.0
            if subtract_background:
                bg_level = float(np.percentile(img_in, 1.0))
                img_in = np.clip(img_in - bg_level, 0, None)

            # Optional pre-denoising
            if predenoise_enabled:
                img_in = predenoise(
                    img_in,
                    method=predenoise_method,
                    sigma_est=predenoise_sigma,
                )

            # Richardson-Lucy
            restored = _richardson_lucy_tv(
                img_in, psf,
                iterations=iterations,
                tv_lambda=tv_lambda,
                early_stop_delta=early_stop_delta,
            )

            # Add background back and convert to original bit depth
            restored = restored + bg_level
            if np.issubdtype(orig_dtype, np.integer):
                info = np.iinfo(orig_dtype)
                restored = np.clip(restored, info.min, info.max)
                restored = restored.astype(orig_dtype)

            tiff.imwrite(str(out_path), restored)
            results.append({"file": stem, "status": "OK", "message": ""})

        except Exception as exc:
            results.append({"file": stem, "status": "FAILED", "message": str(exc)})

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

            tiff.imwrite(str(out_path), restored.astype(np.float32))
            results.append({"file": stem, "status": "OK", "message": ""})

        except Exception as exc:
            results.append({"file": stem, "status": "FAILED", "message": str(exc)})

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
        if data.shape[0] <= 6:
            return data[channel_index]
        return data[z_index]
    if data.ndim == 4:
        if data.shape[1] <= 6:
            return data[z_index, channel_index]
        return data[channel_index, z_index]
    if data.ndim == 5:
        return data[0, z_index, channel_index]
    squeezed = np.squeeze(data)
    if squeezed.ndim == 2:
        return squeezed
    raise ValueError(
        f"Cannot extract 2-D plane from array with shape {data.shape}"
    )
