# Scripts/ready_masks_v2.py
from pathlib import Path
import tifffile as tiff
import numpy as np
import re

DATA = Path("Ome_tifs_2D")             # <-- input folder
OUT  = Path("Ome_tifs_2D_cleaned_new")     # <-- output folder
OUT.mkdir(parents=True, exist_ok=True)

def rgb_to_labels(rgb):
    """Convert RGB(A) mask (Y,X,3/4) to 2D instance labels (heuristic)."""
    if rgb.ndim != 3 or rgb.shape[-1] not in (3,4):
        raise ValueError(f"Not RGB(A), shape={rgb.shape}")
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    rgb = rgb.astype(np.uint32)
    keys = (rgb[...,0] << 16) | (rgb[...,1] << 8) | rgb[...,2]
    uniq, inv = np.unique(keys, return_inverse=True)
    labels = inv.reshape(rgb.shape[0], rgb.shape[1]).astype(np.int32)
    # Most frequent color -> background 0
    counts = np.bincount(labels.ravel())
    bg = int(np.argmax(counts))
    remap = np.zeros_like(uniq, dtype=np.int32)
    nxt = 1
    for u in range(len(uniq)):
        remap[u] = 0 if u == bg else (nxt := nxt+1) - 1
    return remap[labels].astype(np.uint16)

def collapse_cyx_labels(arr):
    """Collapse (C,Y,X) label stack to a single (Y,X) label image."""
    assert arr.ndim == 3
    C = arr.shape[0]
    arr = arr.astype(np.int32)
    # If all slices identical, just take the first
    if all(np.array_equal(arr[0], arr[k]) for k in range(1, C)):
        return arr[0].astype(np.uint16)
    # Otherwise pick the slice with most labeled pixels
    counts = [int(np.count_nonzero(arr[k])) for k in range(C)]
    k = int(np.argmax(counts))
    return arr[k].astype(np.uint16)

converted_rgb = 0
collapsed_cyx  = 0
copied_imgs    = 0
saved_masks    = 0

for img in sorted(DATA.glob("*.ome.tif")):
    base = re.sub(r"\.ome\.tif$", "", img.name)
    msk_path = img.with_name(base + "_masks.tif")
    if not msk_path.exists():
        continue

    # copy the raw image as-is
    out_img = OUT / img.name
    if not out_img.exists():
        tiff.imwrite(out_img, tiff.imread(img))
        copied_imgs += 1

    Y = tiff.imread(msk_path)
    Y_fixed = None

    if Y.ndim == 2:
        # already a 2D label image
        if not np.issubdtype(Y.dtype, np.integer):
            Y = Y.astype(np.uint16)
        Y_fixed = Y.astype(np.uint16)

    elif Y.ndim == 3 and Y.shape[-1] in (3,4):
        # RGB(A) -> label ids
        Y_fixed = rgb_to_labels(Y)
        converted_rgb += 1

    elif Y.ndim == 3 and Y.shape[0] <= 8:
        # (C,Y,X) small channel stack -> collapse
        Y_fixed = collapse_cyx_labels(Y)
        collapsed_cyx += 1

    else:
        print("SKIP (unhandled mask shape):", msk_path, Y.shape)
        continue

    if Y_fixed.max() < 1:
        print("SKIP (empty mask after fix):", msk_path)
        continue

    out_mask = OUT / msk_path.name
    tiff.imwrite(out_mask, Y_fixed)
    saved_masks += 1
    print("saved:", out_mask.name, "shape", Y_fixed.shape, "max", int(Y_fixed.max()))

print("\nSummary:")
print("  Images copied:    ", copied_imgs)
print("  Masks saved:      ", saved_masks)
print("  RGB->labels:      ", converted_rgb)
print("  Collapsed CYX:    ", collapsed_cyx)
print("Output folder:", OUT.resolve())
