#!/usr/bin/env python3
import argparse, re, shutil
from pathlib import Path
import numpy as np
import tifffile as tiff


Z_KEEP_RE = re.compile(r'_Z00[5](?=[^0-9]|$)', re.IGNORECASE)

def is_target_z(fn: str) -> bool:
    return bool(Z_KEEP_RE.search(fn))


def read_ome(path: Path):
    with tiff.TiffFile(path) as tf:
        arr = tf.asarray()               # full ndarray
        # Prefer series[0].axes if present; else fall back to tags or guess
        axes = getattr(tf.series[0], 'axes', '')
        if not axes:
            # best-effort fallback
            axes = 'CYX' if arr.ndim == 3 else 'YX'
    return arr, axes

def process_folder(src: Path, dst: Path, dry_run: bool = False):
    if not src.exists():
        raise SystemExit(f"Source not found: {src}")

    # 1) Copy only Z005 files
    copied = []
    for p in src.glob('**/*'):
        if not p.is_file():
            continue
        name = p.name
        # Accept common OME-TIFF extensions
        if not name.lower().endswith(('.ome.tif', '.ome.tiff','.ome.tifs')):
            continue
        if is_target_z(name):
            rel = p.relative_to(src)
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if not dry_run:
                shutil.copy2(p, target)
            copied.append(target)

    print(f"Copied {len(copied)} files into {dst}")

def main():
    ap = argparse.ArgumentParser(description="Keep Z005, extract channel, save as .tif")
    ap.add_argument("src", type=Path, help="Source folder (contains the OME-TIF files)")
    ap.add_argument("dst", type=Path, help="Destination folder (new folder to hold filtered files and outputs)")
    ap.add_argument("--dry-run", action="store_true", help="Show what would happen without writing files")
    args = ap.parse_args()

    process_folder(args.src, args.dst, args.dry_run)

if __name__ == "__main__":
    main()