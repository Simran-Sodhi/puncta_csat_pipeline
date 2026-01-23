# check_pairs_and_fix.py
from pathlib import Path
import re

d = Path("../../Ome_tifs_2D_cleaned_new")  # change if needed
imgs = sorted([p for p in d.iterdir() if p.suffix.lower() in (".tif",".tiff",".png",".jpg",".jpeg") and "_masks" not in p.stem])

# collect existing masks
mask_map = {p.stem: p for p in d.iterdir() if p.suffix.lower() in (".tif",".tiff",".png") and "_masks" in p.stem}

missing = []
renamed = []

for img in imgs:
    stem = img.stem  # e.g., "2025...Z005.ome" if double suffix; that's okay
    # target mask name
    target_mask = d / f"{stem}_masks.tif"
    if target_mask.exists():
        continue

    # try to find mask with other suffixes to rename:
    # e.g., *_labels.tif, *_mask.tif, *_cp_masks.tif, *_masks.png
    candidates = list(d.glob(stem + "_*.tif")) + list(d.glob(stem + "_*.png"))
    picked = None
    # preference order
    prefer = ["_masks", "_cp_masks", "_labels", "_mask"]
    for c in candidates:
        for key in prefer:
            if c.stem.endswith(key):
                picked = c
                break
        if picked: break

    if picked and not target_mask.exists():
        picked.rename(target_mask)
        renamed.append((picked.name, target_mask.name))
    else:
        missing.append(img.name)

print(f"Total images: {len(imgs)}")
print(f"Renamed masks: {len(renamed)}")
for a,b in renamed[:10]:
    print("  ", a, "->", b)
if missing:
    print("\nImages missing masks (create or rename):")
    for m in missing:
        print("  ", m)
