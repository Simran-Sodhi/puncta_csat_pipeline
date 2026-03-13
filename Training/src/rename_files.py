"""
File renaming utility for the Cellpose segmentation pipeline.

Renames image and mask files in a folder to match the naming convention
required by the pipeline:
  - Images: <prefix><NNN>_img.<ext>
  - Masks:  <prefix><NNN>_masks.<ext>
"""

import os
import shutil
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".nd2"}


def list_image_files(directory: str, include_npy: bool = False) -> list[str]:
    """List all supported image files in a directory, sorted.

    Parameters
    ----------
    include_npy : bool
        If True, also include ``_seg.npy`` files (Cellpose masks).
    """
    files = []
    for f in sorted(os.listdir(directory)):
        ext = Path(f).suffix.lower()
        if ext in SUPPORTED_EXTENSIONS:
            files.append(f)
        elif include_npy and f.endswith("_seg.npy"):
            files.append(f)
    return files


def generate_new_name(index, prefix, suffix, extension, zero_pad=3):
    """Generate a pipeline-compatible filename."""
    return f"{prefix}{str(index).zfill(zero_pad)}{suffix}{extension}"


def rename_files(
    directory, suffix="_img", prefix="", target_extension=None,
    dry_run=False, copy_mode=False, output_dir=None,
):
    """Rename (or copy) all image files in a directory to the pipeline convention."""
    files = list_image_files(directory)
    if not files:
        logger.warning(f"No image files found in {directory}")
        return []

    renames = []
    for i, filename in enumerate(files, start=1):
        old_path = os.path.join(directory, filename)
        ext = target_extension or Path(filename).suffix.lower()
        new_name = generate_new_name(i, prefix, suffix, ext)

        if copy_mode and output_dir:
            new_path = os.path.join(output_dir, new_name)
        else:
            new_path = os.path.join(directory, new_name)

        renames.append((old_path, new_path))

    if dry_run:
        logger.info(f"Dry run -- {len(renames)} files would be renamed:")
        for old, new in renames:
            logger.info(f"  {Path(old).name}  ->  {Path(new).name}")
        return [(Path(o).name, Path(n).name) for o, n in renames]

    if copy_mode and output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for old_path, new_path in renames:
        if old_path == new_path:
            continue
        if copy_mode:
            shutil.copy2(old_path, new_path)
            logger.info(f"Copied: {Path(old_path).name} -> {Path(new_path).name}")
        else:
            tmp_path = old_path + ".tmp_rename"
            os.rename(old_path, tmp_path)
            os.rename(tmp_path, new_path)
            logger.info(f"Renamed: {Path(old_path).name} -> {Path(new_path).name}")

    return [(Path(o).name, Path(n).name) for o, n in renames]


def rename_image_mask_pair(
    image_dir, mask_dir, prefix="", target_extension=None,
    dry_run=False, copy_mode=False, image_output_dir=None, mask_output_dir=None,
):
    """Rename both image and mask directories together, ensuring paired indices."""
    img_files = list_image_files(image_dir)
    mask_files = list_image_files(mask_dir)

    if len(img_files) != len(mask_files):
        logger.warning(
            f"Mismatch: {len(img_files)} images vs {len(mask_files)} masks. "
            "Files will be renamed independently -- verify pairing manually."
        )

    img_renames = rename_files(
        directory=image_dir, suffix="_img", prefix=prefix,
        target_extension=target_extension, dry_run=dry_run,
        copy_mode=copy_mode, output_dir=image_output_dir,
    )
    mask_renames = rename_files(
        directory=mask_dir, suffix="_masks", prefix=prefix,
        target_extension=target_extension, dry_run=dry_run,
        copy_mode=copy_mode, output_dir=mask_output_dir,
    )

    return img_renames, mask_renames


def auto_sort_and_rename(
    source_dir, task="dic", split="train", project_root=".", dry_run=False,
):
    """Sort files from a flat source directory into the pipeline's data structure."""
    img_candidates = ["images", "raw", "imgs", "input"]
    mask_candidates = ["masks", "labels", "annotations", "gt", "ground_truth"]

    img_subdir = None
    for name in img_candidates:
        candidate = os.path.join(source_dir, name)
        if os.path.isdir(candidate):
            img_subdir = candidate
            break

    mask_subdir = None
    for name in mask_candidates:
        candidate = os.path.join(source_dir, name)
        if os.path.isdir(candidate):
            mask_subdir = candidate
            break

    if img_subdir is None or mask_subdir is None:
        logger.error(f"Could not find image/mask subdirectories in {source_dir}.")
        return [], []

    modality = "dic" if task == "dic" else "fluor"
    img_output = os.path.join(project_root, "data", split, f"{modality}_raw")
    mask_output = os.path.join(project_root, "data", split, f"{modality}_labels")
    prefix = f"{modality}_"

    return rename_image_mask_pair(
        image_dir=img_subdir, mask_dir=mask_subdir, prefix=prefix,
        target_extension=".tif", dry_run=dry_run, copy_mode=True,
        image_output_dir=img_output, mask_output_dir=mask_output,
    )


def split_mixed_folder(
    source_dir,
    image_output_dir,
    mask_output_dir,
    prefix="",
    target_extension=".tif",
    dry_run=False,
):
    """Split a folder containing images + Cellpose ``_seg.npy`` masks.

    Handles the common Cellpose workflow where images (``*.tif``) and masks
    (``*_seg.npy``) live in the same directory.  Each ``<name>_seg.npy`` is
    paired with ``<name>.tif`` (or ``.tiff``, ``.png``, etc.).

    The function copies images and converts `_seg.npy` masks to TIFF,
    renaming both to the pipeline convention:
      - Images: ``<prefix><NNN>_img.tif``
      - Masks:  ``<prefix><NNN>_masks.tif``

    Parameters
    ----------
    source_dir : str
        Folder with mixed image + ``_seg.npy`` files.
    image_output_dir, mask_output_dir : str
        Where to write the separated & renamed files.
    prefix : str
        Naming prefix (e.g. ``"dic_"``).
    target_extension : str
        Output extension for both images and masks.
    dry_run : bool
        If True, only preview — don't write files.

    Returns
    -------
    img_renames : list of (old_name, new_name)
    mask_renames : list of (old_name, new_name)
    """
    import numpy as np
    import tifffile

    source = Path(source_dir)

    # Find all _seg.npy files
    npy_files = sorted(source.glob("*_seg.npy"))
    if not npy_files:
        logger.warning(f"No _seg.npy files found in {source_dir}")
        return [], []

    # Match each _seg.npy with its image
    pairs = []
    for npy_path in npy_files:
        # <name>_seg.npy → look for <name>.tif/tiff/png/...
        base = npy_path.name[:-len("_seg.npy")]
        img_path = None
        for ext in (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".nd2"):
            candidate = source / (base + ext)
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            logger.warning(f"No image found for {npy_path.name}, skipping")
            continue
        pairs.append((img_path, npy_path))

    logger.info(f"Found {len(pairs)} image/_seg.npy pairs in {source_dir}")

    img_renames = []
    mask_renames = []
    ext = target_extension or ".tif"

    for i, (img_path, npy_path) in enumerate(pairs, start=1):
        img_new = generate_new_name(i, prefix, "_img", ext)
        mask_new = generate_new_name(i, prefix, "_masks", ext)

        img_renames.append((img_path.name, img_new))
        mask_renames.append((npy_path.name, mask_new))

    if dry_run:
        return img_renames, mask_renames

    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(mask_output_dir, exist_ok=True)

    for i, (img_path, npy_path) in enumerate(pairs, start=1):
        img_new = generate_new_name(i, prefix, "_img", ext)
        mask_new = generate_new_name(i, prefix, "_masks", ext)

        # Copy image
        shutil.copy2(str(img_path), os.path.join(image_output_dir, img_new))
        logger.info(f"Copied image: {img_path.name} -> {img_new}")

        # Convert _seg.npy mask to TIFF
        dat = np.load(str(npy_path), allow_pickle=True).item()
        masks = np.asarray(dat["masks"]).astype(np.uint16)
        tifffile.imwrite(os.path.join(mask_output_dir, mask_new), masks)
        logger.info(f"Converted mask: {npy_path.name} -> {mask_new}")

    return img_renames, mask_renames


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Rename files to match pipeline naming conventions")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--ext", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--copy", action="store_true")
    parser.add_argument("--image-output", default=None)
    parser.add_argument("--mask-output", default=None)
    args = parser.parse_args()

    rename_image_mask_pair(
        image_dir=args.image_dir, mask_dir=args.mask_dir, prefix=args.prefix,
        target_extension=args.ext, dry_run=args.dry_run, copy_mode=args.copy,
        image_output_dir=args.image_output, mask_output_dir=args.mask_output,
    )
