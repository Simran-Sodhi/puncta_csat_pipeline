"""
Preprocessing module for 16-bit multi-channel microscopy images.

Handles the full preprocessing workflow:
  1. Load multi-channel 16-bit TIFFs (DIC + fluorescence channels)
  2. Split and normalize each channel independently
  3. Run Cellpose cpsam (or custom model) to generate draft masks
  4. Save masks to a separate folder for manual curation

Channel mapping for this dataset:
  0 = DIC (brightfield)
  1 = mEGFP
  2 = mScarlet
  3 = miRFPnano3 (not always present)

Compatible with Cellpose 3 and 4.
"""

import os
import glob
import inspect
import logging
from pathlib import Path

import numpy as np
import tifffile
from skimage import io as skio

logger = logging.getLogger(__name__)

CHANNEL_NAMES = {
    0: "DIC",
    1: "mEGFP",
    2: "mScarlet",
    3: "miRFPnano3",
}


def load_multichannel_image(path, position=0, z_slice=None):
    """Load a multi-channel image. Returns (C, H, W) or (H, W)."""
    ext = Path(path).suffix.lower()
    if ext == ".nd2":
        import nd2
        with nd2.ND2File(path) as f:
            sizes = f.sizes
            img = f.asarray()
            axis_order = list(sizes.keys())
            for dim in ("P", "Z"):
                if dim in sizes:
                    ax = axis_order.index(dim)
                    n = sizes[dim]
                    if dim == "P":
                        idx = position
                        if idx >= n:
                            raise ValueError(f"Position {idx} out of range; file has {n} positions")
                    elif dim == "Z":
                        idx = z_slice if z_slice is not None else 0
                        if idx < 0 or idx >= n:
                            raise ValueError(f"Z-slice {idx} out of range; file has {n} Z-slices")
                    img = np.take(img, idx, axis=ax)
                    axis_order.pop(ax)
    elif ext in (".tif", ".tiff"):
        img = tifffile.imread(path)
    else:
        img = skio.imread(path)

    if img.ndim == 2:
        return img
    elif img.ndim == 3:
        if img.shape[-1] <= 4 and img.shape[0] > 4:
            img = np.moveaxis(img, -1, 0)
        return img
    elif img.ndim > 3:
        while img.ndim > 3:
            img = img[0]
        if img.ndim == 3 and img.shape[-1] <= 4 and img.shape[0] > 4:
            img = np.moveaxis(img, -1, 0)
        return img
    else:
        return img


def get_image_info(path, z_slice=None):
    """Get metadata about a multi-channel image."""
    img = load_multichannel_image(path, z_slice=z_slice)
    n_channels = img.shape[0] if img.ndim == 3 else 1
    info = {
        "path": path, "shape": img.shape, "dtype": str(img.dtype),
        "num_channels": n_channels, "is_16bit": img.dtype in (np.uint16, np.int16),
        "channels": {},
    }
    for ch_idx in range(n_channels):
        ch_name = CHANNEL_NAMES.get(ch_idx, f"Channel {ch_idx}")
        ch_data = img[ch_idx] if img.ndim == 3 else img
        info["channels"][ch_idx] = {
            "name": ch_name, "min": int(ch_data.min()), "max": int(ch_data.max()),
            "mean": float(ch_data.mean()), "has_signal": int(ch_data.max()) > int(ch_data.min()),
        }
    return info


def normalize_channel(channel, lower_percentile=1.0, upper_percentile=99.0, tile_blocksize=0):
    """Normalize a single channel to float32 using percentile-based normalization."""
    channel = channel.astype(np.float32)
    if tile_blocksize > 0:
        return _tile_normalize(channel, lower_percentile, upper_percentile, tile_blocksize)
    p_low = np.percentile(channel, lower_percentile)
    p_high = np.percentile(channel, upper_percentile)
    if p_high - p_low < 1e-3:
        logger.warning("Channel has no dynamic range, zeroing out")
        return np.zeros_like(channel)
    return (channel - p_low) / (p_high - p_low)


def _tile_normalize(channel, lower_percentile, upper_percentile, blocksize):
    """Tile-based normalization for uneven illumination (e.g., DIC)."""
    h, w = channel.shape
    result = np.zeros_like(channel)
    n_tiles_y = max(1, h // blocksize)
    n_tiles_x = max(1, w // blocksize)
    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            y0, y1 = ty * blocksize, min((ty + 1) * blocksize, h)
            x0, x1 = tx * blocksize, min((tx + 1) * blocksize, w)
            tile = channel[y0:y1, x0:x1]
            p_low = np.percentile(tile, lower_percentile)
            p_high = np.percentile(tile, upper_percentile)
            if p_high - p_low < 1e-3:
                result[y0:y1, x0:x1] = 0.0
            else:
                result[y0:y1, x0:x1] = (tile - p_low) / (p_high - p_low)
    return result


def preprocess_image(image, segment_channel=0, nuclear_channel=0,
                     lower_percentile=1.0, upper_percentile=99.0,
                     tile_blocksize_dic=128, invert_dic=False):
    """Preprocess a multi-channel 16-bit image for Cellpose."""
    if image.ndim == 2:
        ch = normalize_channel(image, lower_percentile, upper_percentile)
        if invert_dic and segment_channel == 0:
            ch = -ch
        return ch

    n_channels = image.shape[0]
    if segment_channel >= n_channels:
        raise ValueError(f"Segment channel {segment_channel} not available. Image has {n_channels} channels.")

    is_dic = (segment_channel == 0)
    tile_bs = tile_blocksize_dic if is_dic else 0

    if is_dic:
        seg_ch = normalize_channel(image[segment_channel], lower_percentile, upper_percentile, tile_bs)
        if invert_dic:
            seg_ch = -seg_ch
    else:
        fluor_low = max(0.1, lower_percentile * 0.5)
        fluor_high = min(99.99, upper_percentile + (100 - upper_percentile) * 0.9)
        seg_ch = normalize_channel(image[segment_channel], fluor_low, fluor_high)

    if nuclear_channel > 0 and nuclear_channel < n_channels:
        nuc_ch = normalize_channel(
            image[nuclear_channel],
            max(0.1, lower_percentile * 0.5),
            min(99.99, upper_percentile + (100 - upper_percentile) * 0.9),
        )
        return np.stack([seg_ch, nuc_ch], axis=-1)

    return seg_ch


def generate_masks(image_dir, output_dir, segment_channel=0, nuclear_channel=0,
                   model_path=None, diameter=None, flow_threshold=0.4,
                   cellprob_threshold=0.0, lower_percentile=1.0, upper_percentile=99.0,
                   tile_blocksize_dic=128, invert_dic=False, use_gpu=True,
                   z_slice=None, progress_callback=None):
    """Generate draft segmentation masks for all images in a directory."""
    from cellpose import models
    os.makedirs(output_dir, exist_ok=True)

    extensions = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg", "*.nd2")
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
    image_files = sorted(image_files)

    if not image_files:
        logger.warning(f"No images found in {image_dir}")
        return []

    # Auto-expand multi-position .nd2 files
    expanded_files = []
    for f in image_files:
        if Path(f).suffix.lower() == ".nd2":
            from data_preparation import expand_nd2_positions, get_nd2_info
            info = get_nd2_info(f)
            if info["n_positions"] > 1:
                logger.info(f"Expanding multi-position ND2: {Path(f).name}")
                tiff_dir = os.path.join(image_dir, "_nd2_expanded")
                tiff_paths = expand_nd2_positions(f, tiff_dir, z_slice=z_slice)
                expanded_files.extend(tiff_paths)
            else:
                expanded_files.append(f)
        else:
            expanded_files.append(f)
    image_files = sorted(expanded_files)

    logger.info(f"Found {len(image_files)} images in {image_dir}")

    from train_cellpose import resolve_pretrained_model
    resolved_model = resolve_pretrained_model(model_path)
    if resolved_model:
        logger.info(f"Loading model: {resolved_model}")
        try:
            model = models.CellposeModel(gpu=use_gpu, pretrained_model=resolved_model)
        except TypeError:
            model = models.CellposeModel(gpu=use_gpu, model_type=resolved_model)
    else:
        logger.info("Using default cpsam model")
        model = models.CellposeModel(gpu=use_gpu)

    if nuclear_channel > 0:
        cp_channels = [1, 2]
    else:
        cp_channels = [0, 0]

    # Check if model.eval() accepts channels (Cellpose 3 vs 4)
    eval_params = inspect.signature(model.eval).parameters
    pass_channels = "channels" in eval_params

    results = []
    for i, img_path in enumerate(image_files):
        filename = Path(img_path).name
        logger.info(f"[{i+1}/{len(image_files)}] Processing: {filename}")
        if progress_callback:
            progress_callback(i, len(image_files), filename)
        try:
            raw = load_multichannel_image(img_path, z_slice=z_slice)
            img_info = get_image_info(img_path)
            if segment_channel in img_info["channels"]:
                ch_info = img_info["channels"][segment_channel]
                if not ch_info["has_signal"]:
                    logger.warning(f"  Channel {segment_channel} has no signal, skipping")
                    results.append((filename, 0))
                    continue

            preprocessed = preprocess_image(
                raw, segment_channel=segment_channel, nuclear_channel=nuclear_channel,
                lower_percentile=lower_percentile, upper_percentile=upper_percentile,
                tile_blocksize_dic=tile_blocksize_dic, invert_dic=invert_dic,
            )

            eval_kwargs = dict(
                diameter=diameter,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
            )
            if pass_channels:
                eval_kwargs["channels"] = cp_channels

            result = model.eval(preprocessed, **eval_kwargs)
            mask = result[0]

            num_objects = int(mask.max())
            logger.info(f"  Found {num_objects} objects")

            stem = Path(filename).stem
            mask_path = os.path.join(output_dir, f"{stem}_masks.tif")
            tifffile.imwrite(mask_path, mask.astype(np.uint16))
            results.append((filename, num_objects))

        except Exception as e:
            logger.error(f"  Failed to process {filename}: {e}")
            results.append((filename, -1))

    if progress_callback:
        progress_callback(len(image_files), len(image_files), "Done")

    total_objects = sum(n for _, n in results if n > 0)
    successful = sum(1 for _, n in results if n >= 0)
    logger.info(f"Preprocessing complete: {successful}/{len(image_files)} images, {total_objects} total objects")

    return results


def save_channel_preview(image_path, output_dir, lower_percentile=1.0,
                         upper_percentile=99.0, tile_blocksize_dic=128, z_slice=None):
    """Save a side-by-side preview of all channels in an image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw = load_multichannel_image(image_path, z_slice=z_slice)
    info = get_image_info(image_path)
    n_channels = info["num_channels"]

    fig, axes = plt.subplots(1, n_channels, figsize=(5 * n_channels, 5))
    if n_channels == 1:
        axes = [axes]

    for ch_idx in range(n_channels):
        ax = axes[ch_idx]
        ch_data = raw[ch_idx] if raw.ndim == 3 else raw
        ch_info = info["channels"][ch_idx]
        is_dic = (ch_idx == 0)
        tile_bs = tile_blocksize_dic if is_dic else 0
        normalized = normalize_channel(ch_data, lower_percentile, upper_percentile, tile_bs)
        display = np.clip(normalized, 0, 1)
        ax.imshow(display, cmap="gray")
        ax.set_title(f"Ch {ch_idx}: {ch_info['name']}\n[{ch_info['min']}-{ch_info['max']}]")
        ax.axis("off")

    fig.suptitle(Path(image_path).name, fontsize=12)
    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    stem = Path(image_path).stem
    preview_path = os.path.join(output_dir, f"{stem}_channels.png")
    fig.savefig(preview_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return preview_path


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Preprocess multi-channel images and generate draft masks")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--segment-channel", type=int, default=0)
    parser.add_argument("--nuclear-channel", type=int, default=0)
    parser.add_argument("--model", default=None)
    parser.add_argument("--diameter", type=float, default=None)
    parser.add_argument("--lower-pct", type=float, default=1.0)
    parser.add_argument("--upper-pct", type=float, default=99.0)
    parser.add_argument("--tile-blocksize", type=int, default=128)
    parser.add_argument("--invert-dic", action="store_true")
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--z-slice", type=int, default=None)
    args = parser.parse_args()

    generate_masks(
        image_dir=args.image_dir, output_dir=args.output_dir,
        segment_channel=args.segment_channel, nuclear_channel=args.nuclear_channel,
        model_path=args.model, diameter=args.diameter,
        lower_percentile=args.lower_pct, upper_percentile=args.upper_pct,
        tile_blocksize_dic=args.tile_blocksize, invert_dic=args.invert_dic,
        use_gpu=not args.no_gpu, z_slice=args.z_slice,
    )
