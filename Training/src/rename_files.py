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


def list_image_files(directory: str) -> list[str]:
    """List all supported image files in a directory, sorted."""
    files = []
    for f in sorted(os.listdir(directory)):
        if Path(f).suffix.lower() in SUPPORTED_EXTENSIONS:
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
