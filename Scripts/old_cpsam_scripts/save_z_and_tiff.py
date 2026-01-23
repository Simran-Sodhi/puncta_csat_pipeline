#!/usr/bin/env python3
import argparse, re, shutil
from pathlib import Path
import numpy as np
import tifffile as tiff

# --- Config: change if needed ---
CHANNEL_INDEX = 2  # 0-based (so 1 == second channel)

# Regex to match ..._Z004.ome.tif(s), ..._Z005..., ..._Z006...
# Z_KEEP_RE = re.compile(r'_Z00[456](?=[^0-9]|$)', re.IGNORECASE)
Z_KEEP_RE = re.compile(r'_Z00[5](?=[^0-9]|$)', re.IGNORECASE)

def is_target_z(fn: str) -> bool:
    return bool(Z_KEEP_RE.search(fn))

def select_channel(arr: np.ndarray, axes: str, c_index: int) -> np.ndarray:
    """
    Return array with a single channel (c_index) preserved.
    Handles common OME axis orders like CYX, CZYX, ZCYX, YXC, etc.
    If there's no 'C' in axes, returns arr unchanged.
    """
    axes = axes.upper()
    if 'C' not in axes:
        # no channel dimension
        return arr

    # Build slicer for all dims
    sl = [slice(None)] * arr.ndim
    c_pos = axes.index('C')
    sl[c_pos] = c_index
    arr_c = arr[tuple(sl)]

    # Optionally squeeze if channel dimension collapses to scalar
    return np.squeeze(arr_c)

def read_ome(path: Path):
    with tiff.TiffFile(path) as tf:
        arr = tf.asarray()               # full ndarray
        # Prefer series[0].axes if present; else fall back to tags or guess
        axes = getattr(tf.series[0], 'axes', '')
        if not axes:
            # best-effort fallback
            axes = 'CYX' if arr.ndim == 3 else 'YX'
    return arr, axes

def save_plain_tif(out_path: Path, img: np.ndarray):
    # Ensure contiguous & dtype preserved
    img = np.ascontiguousarray(img)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(out_path, img)

def process_folder(src: Path, dst: Path, dry_run: bool = False):
    if not src.exists():
        raise SystemExit(f"Source not found: {src}")

    # 1) Copy only Z004/5/6 files
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

    # 2) From each copied OME-TIF, keep channel and write .tif
    converted = []
    for q in copied:
        try:
            arr, axes = read_ome(q)
            ch = select_channel(arr, axes, CHANNEL_INDEX)
            # Build a clean .tif name (drop .ome.* and add _C{index+1}.tif)
            stem = q.name
            # strip .ome.tif(f)(s) patterns
            stem_noext = re.sub(r'\.ome\.tiff?s?$', '', stem, flags=re.IGNORECASE)
            stem_noext = re.sub(r'\.tiff?s?$', '', stem_noext, flags=re.IGNORECASE)
            out_name = f"{stem_noext}_C{CHANNEL_INDEX+1}.tif"
            out_path = q.with_name(out_name)

            if not dry_run:
                save_plain_tif(out_path, ch)
            converted.append(out_path)

            # Optionally remove the copied OME file after conversion:
            q.unlink()

        except Exception as e:
            print(f"[WARN] Failed to convert {q}: {e}")

    print(f"Copied {len(copied)} files into {dst}")
    # print(f"Converted {len(converted)} files to single-channel .tif")

def main():
    ap = argparse.ArgumentParser(description="Keep Z004/Z005/Z006, extract channel, save as .tif")
    ap.add_argument("src", type=Path, help="Source folder (contains the OME-TIF files)")
    ap.add_argument("dst", type=Path, help="Destination folder (new folder to hold filtered files and outputs)")
    ap.add_argument("--dry-run", action="store_true", help="Show what would happen without writing files")
    args = ap.parse_args()

    process_folder(args.src, args.dst, args.dry_run)

if __name__ == "__main__":
    main()
