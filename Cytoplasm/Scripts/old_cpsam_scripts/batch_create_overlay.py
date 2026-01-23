#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import tifffile as tiff
from skimage.segmentation import find_boundaries

def to_gray(img):
    # Pick one channel if needed (matches your logic)
    if img.ndim == 3 and img.shape[0] <= 4:          # CYX
        img_gray = img[1]
    elif img.ndim == 3 and img.shape[-1] <= 4:       # YXC
        img_gray = img[..., 1]
    else:
        img_gray = img
    # Normalize to [0,1] for display
    return (img_gray - img_gray.min()) / (np.ptp(img_gray) + 1e-8)

def make_overlay(img_gray, mask):
    boundaries = find_boundaries(mask, mode="outer")
    overlay = np.dstack([img_gray, img_gray, img_gray])
    overlay[boundaries] = [1.0, 0.0, 0.0]  # paint red outlines
    return overlay

def strip_ome_suffix(path: Path) -> str:
    # Return filename without the trailing ".ome.tif" (keeps other dots if any)
    name = path.name
    return name[:-8] if name.endswith(".ome.tif") else path.stem

def main(in_dir: Path, out_dir: Path, overwrite: bool):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Iterate over all *.ome.tif that are not masks themselves
    ome_files = sorted(f for f in in_dir.glob("*.ome.tif") if "_masks" not in f.name)
    if not ome_files:
        print(f"No .ome.tif images found in: {in_dir}")
        return

    for img_path in ome_files:
        # Derive mask path: replace ".ome.tif" with "_masks.tif"
        mask_path = img_path.with_name(img_path.name.replace(".ome.tif", "_masks.tif"))
        if not mask_path.exists():
            print(f"[skip] mask not found for image: {img_path.name}")
            continue

        # Output path
        base = strip_ome_suffix(img_path)
        out_png = out_dir / f"{base}_overlay.png"
        if out_png.exists() and not overwrite:
            print(f"[skip] exists: {out_png.name}")
            continue

        try:
            img = tiff.imread(img_path)
            mask = tiff.imread(mask_path)  # integer-labeled mask (Y,X)

            img_gray = to_gray(img)
            overlay = make_overlay(img_gray, mask)

            # Save as PNG (float in [0,1] is fine)
            # Using tifffile.imwrite also works, but PNG is convenient for viewing.
            import imageio.v3 as iio
            iio.imwrite(out_png.as_posix(), (overlay * 255).astype(np.uint8))
            print(f"[ok] {img_path.name} -> {out_png.name}")
        except Exception as e:
            print(f"[error] {img_path.name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch overlay red mask boundaries over OME-TIFF images.")
    parser.add_argument("input_dir", type=str, help="Path to Ome_tifs_2D_cleaned folder")
    parser.add_argument("-o", "--output_dir", type=str, default="overlays", help="Output folder for PNG overlays")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    main(in_dir, out_dir, args.overwrite)
