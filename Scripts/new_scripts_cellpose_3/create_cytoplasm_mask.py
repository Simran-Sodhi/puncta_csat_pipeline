#!/usr/bin/env python3
"""
make_cytoplasm_masks_from_dirs.py

Given:
  - a directory of nucleus masks (label images)
  - a directory of whole-cell/cytoplasm masks (label images)
  - a directory of raw images (e.g. file.ome.tif)

with consistent naming, produce:
  - cytoplasm-only masks (cell minus nucleus)
  - triptychs (image, masks, overlay) for visual QA

Assumed filename pattern (can be tweaked via MASK_SUFFIX):
  image in data_dir:           file.ome.tif
  corresponding mask filename: file_cyto3_masks.ome.tif

Usage:
  python make_cytoplasm_masks_from_dirs.py \
      /path/to/nuc_masks \
      /path/to/cyto_masks \
      --data-dir /path/to/images \
      --outdir  /path/to/cyto_only_masks \
      --nuc-dilate-px 1 \
      --channel-index 1 \
      --z-index 0
"""

import argparse
from pathlib import Path

import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

try:
    from skimage.morphology import dilation, disk
    HAVE_SKIMAGE = True
except ImportError:
    HAVE_SKIMAGE = False

# Suffix in the mask filenames before the extension(s)
# e.g. image: "file.ome.tif" -> mask: "file_cyto3_masks.ome.tif"
MASK_SUFFIX = "_cyto3_masks"


# ---------- IO helpers ---------- #

def load_mask(path):
    """Load a label mask as int32."""
    arr = tiff.imread(str(path))
    return arr.astype(np.int32)


def save_mask(mask, out_path):
    """Save label mask as uint16 TIFF."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(str(out_path), mask.astype(np.uint16))


def collect_masks_by_name(mask_dir):
    """
    Return a dict: { filename (with extension) -> Path }
    for all .tif/.tiff/.ome.tif/.ome.tiff in mask_dir.
    """
    mask_dir = Path(mask_dir)
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"{mask_dir} is not a directory")

    files = []
    for ext in (".tif", ".tiff"):
        files.extend(mask_dir.glob(f"*{ext}"))
        files.extend(mask_dir.glob(f"*{ext.upper()}"))

    return {p.name: p for p in files}


# ---------- image loading (for triptychs) ---------- #

def load_image_plane(path, channel_index=0, z_index=0):
    """
    Load a single 2D image plane from an OME-TIFF or regular TIFF.

    Parameters
    ----------
    path : str or Path
        Path to the image file.
    channel_index : int
        Channel index to use (0-based).
    z_index : int
        Z-plane index to use if multiple Z planes exist.

    Returns
    -------
    img2d : np.ndarray (Y, X)
    """
    path = str(path)
    with tiff.TiffFile(path) as tf:
        series = tf.series[0]
        data = series.asarray()
        axes = getattr(series, "axes", None)

    # OME-TIFF case with axes string
    if axes is not None:
        sl = [slice(None)] * len(axes)

        if "T" in axes:
            sl[axes.index("T")] = 0

        if "C" in axes:
            c_idx = axes.index("C")
            if channel_index >= data.shape[c_idx]:
                raise ValueError(
                    f"Requested channel_index={channel_index}, "
                    f"but image has only {data.shape[c_idx]} channels."
                )
            sl[c_idx] = channel_index

        if "Z" in axes:
            z_idx_axis = axes.index("Z")
            if z_index >= data.shape[z_idx_axis]:
                raise ValueError(
                    f"Requested z_index={z_index}, "
                    f"but image has only {data.shape[z_idx_axis]} z-planes."
                )
            sl[z_idx_axis] = z_index

        img = data[tuple(sl)]
        img2d = np.squeeze(img)
        if img2d.ndim != 2:
            raise ValueError(
                f"Expected 2D image after slicing, got shape {img2d.shape} with axes '{axes}'."
            )
        return img2d

    # non-OME TIFF heuristics
    if data.ndim == 2:
        return data

    if data.ndim == 3:
        if data.shape[0] <= 4 and data.shape[0] <= data.shape[-1]:
            # (C, Y, X)
            if channel_index >= data.shape[0]:
                raise ValueError(
                    f"Requested channel_index={channel_index}, "
                    f"but image has only {data.shape[0]} channels (C, Y, X)."
                )
            return data[channel_index, :, :]

        if data.shape[-1] <= 4:
            # (Y, X, C)
            if channel_index >= data.shape[-1]:
                raise ValueError(
                    f"Requested channel_index={channel_index}, "
                    f"but image has only {data.shape[-1]} channels (Y, X, C)."
                )
            return data[:, :, channel_index]

        # assume (Z, Y, X)
        if z_index >= data.shape[0]:
            raise ValueError(
                f"Requested z_index={z_index}, but image has only {data.shape[0]} z-planes (Z, Y, X)."
            )
        return data[z_index, :, :]

    raise ValueError(f"Unsupported TIFF shape {data.shape} for file {path}")


# ---------- LUT / normalization ---------- #

def auto_lut_clip(img, low_percentile=2.0, high_percentile=99.8):
    """
    Viewer-style LUT clipping:
    - Compute low/high percentiles
    - Clip to [lo, hi]
    - Rescale to [0, 1]
    - Values below lo become ~0 (black), above hi become ~1 (white)
    """
    img = img.astype(np.float32)
    lo = np.percentile(img, low_percentile)
    hi = np.percentile(img, high_percentile)

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(img))
        hi = float(np.max(img) if np.max(img) != np.min(img) else lo + 1.0)

    img_clipped = np.clip(img, lo, hi)
    img_norm = (img_clipped - lo) / (hi - lo + 1e-8)
    return img_norm.astype(np.float32)


# ---------- mask logic ---------- #

# def compute_cytoplasm_mask(cell_mask, nuc_mask, nuc_dilate_px=0):
#     """
#     cell_mask: label image of whole cells (0 = background)
#     nuc_mask:  label image of nuclei (0 = background)
#     nuc_dilate_px: int, if > 0 dilate nuclei before subtraction

#     Returns:
#         cyto_mask: same shape, labels from cell_mask, nuclei removed.
#     """
#     assert cell_mask.shape == nuc_mask.shape, "Mask shapes must match"
#     nuc_binary = nuc_mask > 0

#     if nuc_dilate_px > 0:
#         if not HAVE_SKIMAGE:
#             raise RuntimeError(
#                 "skimage is required for dilation. Install with 'pip install scikit-image' "
#                 "or run with --nuc-dilate-px 0."
#             )
#         nuc_binary = dilation(nuc_binary, disk(nuc_dilate_px))

#     cyto_mask = cell_mask.copy()
#     cyto_mask[nuc_binary] = 0
#     return cyto_mask


import numpy as np
from skimage.morphology import dilation, disk

def compute_cytoplasm_mask_filtered(
    cell_mask,
    nuc_mask,
    nuc_dilate_px=0,
    min_nuc_pixels=10,
    min_overlap_frac=0.005,  # 0.5% of cell area
):
    """
    cell_mask: label image of whole cells (0 = background)
    nuc_mask:  label image of nuclei (0 = background)

    nuc_dilate_px:    dilate nuclei before subtraction (to be safe on boundaries)
    min_nuc_pixels:   minimum # of overlapping pixels to consider cell valid
    min_overlap_frac: minimum fraction of cell area that must overlap nuclei

    Returns:
        cyto_mask_filtered: cytoplasm-only mask with “orphan” cells removed
        cell_mask_filtered: (optional) filtered whole-cell mask
        orphan_labels:      list of cell labels that were removed
    """
    assert cell_mask.shape == nuc_mask.shape, "Mask shapes must match"

    # 1) binary nuclei (with optional dilation)
    nuc_binary = nuc_mask > 0
    if nuc_dilate_px > 0:
        nuc_binary = dilation(nuc_binary, disk(nuc_dilate_px))

    # 2) start from normal cyto = cell minus nuclei
    cyto_mask = cell_mask.copy()
    cyto_mask[nuc_binary] = 0

    # 3) figure out which cell labels to drop
    max_label = int(cell_mask.max())
    keep = np.zeros(max_label + 1, dtype=bool)
    keep[0] = False  # background

    orphan_labels = []

    for k in range(1, max_label + 1):
        cell_region = (cell_mask == k)
        if not cell_region.any():
            continue

        cell_area = int(cell_region.sum())
        overlap_area = int((cell_region & nuc_binary).sum())

        if overlap_area < min_nuc_pixels or overlap_area < min_overlap_frac * cell_area:
            # orphan or very weakly overlapping cell -> drop
            orphan_labels.append(k)
        else:
            keep[k] = True

    # 4) apply removal mask
    drop_mask = np.isin(cell_mask, orphan_labels)

    cyto_mask_filtered = cyto_mask.copy()
    cyto_mask_filtered[drop_mask] = 0

    cell_mask_filtered = cell_mask.copy()
    cell_mask_filtered[drop_mask] = 0

    return cyto_mask_filtered, cell_mask_filtered, orphan_labels


def make_combined_mask(cyto_mask, nuc_mask):
    """
    Combine cyto and nuc masks into one label image with unique labels.
    - cytoplasm labels keep their original cell IDs
    - nuclei labels are offset so they don't collide
    """
    combined = cyto_mask.copy()
    offset = int(cyto_mask.max())
    nuc = nuc_mask.copy()
    nuc_pixels = nuc > 0
    nuc[nuc_pixels] = nuc[nuc_pixels] + offset
    combined[nuc_pixels] = nuc[nuc_pixels]
    return combined


# ---------- triptych ---------- #

def save_triptych(img, cyto_mask, out_path,
                  lut_low=2.0, lut_high=99.8):
    """
    Save a triptych with:
      [0] LUT-normalized image
      [1] combined masks (nuc + cyto-only)
      [2] overlay of combined masks on image
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img_norm = auto_lut_clip(img, lut_low, lut_high)
    # combined = make_combined_mask(cyto_mask, nuc_mask)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    cmap = plt.cm.get_cmap("tab20").copy()
    cmap.set_bad(color="black") 

    # Image
    axes[0].imshow(img_norm, cmap="gray")
    axes[0].set_title("Image (LUT-normalized)")
    axes[0].axis("off")

    # Masks (combined)
    masked_2 = np.ma.masked_where(cyto_mask == 0, cyto_mask)
    axes[1].imshow(masked_2, cmap=cmap)
    axes[1].set_title("Masks (cyto)")
    axes[1].axis("off")

    # Overlay
    axes[2].imshow(img_norm, cmap="gray")
    masked = np.ma.masked_where(cyto_mask == 0, cyto_mask)
    axes[2].imshow(masked, cmap="tab20", alpha=0.5)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------- main ---------- #

def main():
    parser = argparse.ArgumentParser(
        description="Create cytoplasm-only masks from nucleus and whole-cell mask dirs, with optional triptychs."
    )
    parser.add_argument("--nuc_dir", help="Directory with nucleus masks (label TIFFs)")
    parser.add_argument("--cyto_dir", help="Directory with whole-cell/cytoplasm masks (label TIFFs)")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory with raw images (e.g. file.ome.tif) for triptychs. "
             "If omitted, triptychs are not generated.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="cytoplasm_masks",
        help="Output directory for cytoplasm-only masks (default: cytoplasm_masks)",
    )
    parser.add_argument(
        "--nuc-dilate-px",
        type=int,
        default=0,
        help="Dilate nuclei by this many pixels before subtraction (default: 0)",
    )
    parser.add_argument(
        "--channel-index",
        type=int,
        default=0,
        help="Channel index in the raw image for triptychs (default: 0)",
    )
    parser.add_argument(
        "--z-index",
        type=int,
        default=0,
        help="Z-plane index for triptychs (default: 0)",
    )
    parser.add_argument(
        "--lut-low",
        type=float,
        default=2.0,
        help="Low percentile for LUT clipping (default: 2.0)",
    )
    parser.add_argument(
        "--lut-high",
        type=float,
        default=99.8,
        help="High percentile for LUT clipping (default: 99.8)",
    )

    args = parser.parse_args()

    nuc_map = collect_masks_by_name(args.nuc_dir)
    cyto_map = collect_masks_by_name(args.cyto_dir)

    common_names = sorted(set(nuc_map.keys()) & set(cyto_map.keys()))
    if not common_names:
        raise RuntimeError(
            "No matching filenames found between nucleus and cyto directories.\n"
            f"nuc_dir:  {args.nuc_dir}\n"
            f"cyto_dir: {args.cyto_dir}"
        )

    missing_in_nuc = sorted(set(cyto_map.keys()) - set(nuc_map.keys()))
    missing_in_cyto = sorted(set(nuc_map.keys()) - set(cyto_map.keys()))

    if missing_in_nuc:
        print("Warning: these cyto masks have no matching nucleus mask:")
        for name in missing_in_nuc:
            print("  ", name)

    if missing_in_cyto:
        print("Warning: these nucleus masks have no matching cyto mask:")
        for name in missing_in_cyto:
            print("  ", name)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    triptych_dir = outdir / "triptychs"
    if args.data_dir is not None:
        triptych_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(common_names)} matching pairs. Writing masks to {outdir}")
    if args.data_dir is not None:
        print(f"Triptychs will be saved to {triptych_dir}")

    data_dir = Path(args.data_dir) if args.data_dir is not None else None

    for name in common_names:
        nuc_path = nuc_map[name]
        cyto_path = cyto_map[name]
        print(f"Processing {name}...")

        nuc_mask = load_mask(nuc_path)
        cell_mask = load_mask(cyto_path)

        # cyto_only = compute_cytoplasm_mask(
        #     cell_mask,
        #     nuc_mask,
        #     nuc_dilate_px=args.nuc_dilate_px,
        # )
        
        cyto_only, cell_filtered, orphan_labels = compute_cytoplasm_mask_filtered(
            cell_mask,
            nuc_mask,
            nuc_dilate_px=args.nuc_dilate_px,
            min_nuc_pixels=10,
            min_overlap_frac=0.005,
        )

        # save cytoplasm-only mask (same filename)
        out_mask_path = outdir / name
        save_mask(cyto_only, out_mask_path)

        # triptych (if data_dir provided)
        if data_dir is not None:
            # derive image filename by stripping MASK_SUFFIX
            # e.g. "file_cyto3_masks.ome.tif" -> "file.ome.tif"
            img_name = name.replace(MASK_SUFFIX, "")
            img_path = data_dir / img_name

            if not img_path.exists():
                print(f"  [WARN] No image found for {name} at {img_path}, skipping triptych.")
                continue

            try:
                img = load_image_plane(
                    img_path,
                    channel_index=args.channel_index,
                    z_index=args.z_index,
                )
                if img.shape != cyto_only.shape:
                    print(
                        f"  [WARN] Image and mask shapes differ for {name}: "
                        f"image {img.shape}, mask {cyto_only.shape}. Skipping triptych."
                    )
                    continue

                trip_path = triptych_dir / f"{Path(img_name).stem}_trigriph.png"
                save_triptych(
                    img,
                    cyto_only,
                    # nuc_mask,
                    trip_path,
                    lut_low=args.lut_low,
                    lut_high=args.lut_high,
                )
            except Exception as e:
                print(f"  [ERROR] Triptych failed for {name}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
