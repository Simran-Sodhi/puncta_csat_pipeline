from pathlib import Path
import numpy as np, tifffile as tiff
import re

DATA = Path("../Ome_tifs_2D_cleaned")

def rgb_to_labels(rgb):
    # rgb: (Y,X,3) uint8/uint16
    rgb = np.asarray(rgb)
    H, W, C = rgb.shape
    assert C in (3,4), "Expected RGB/RGBA"
    if C == 4:  # drop alpha if present
        rgb = rgb[..., :3]
    # pack colors to 1D keys
    if rgb.dtype != np.uint32:
        rgb32 = rgb.astype(np.uint32)
    else:
        rgb32 = rgb
    keys = (rgb32[...,0] << 16) | (rgb32[...,1] << 8) | rgb32[...,2]
    # map unique colors to 0..N
    uniq, inv = np.unique(keys, return_inverse=True)
    labels = inv.reshape(H, W).astype(np.int32)

    # Optional: choose a background color → set it to 0
    # Here we assume the most frequent color is background:
    counts = np.bincount(labels.ravel())
    bg_label = int(np.argmax(counts))
    # remap so bg becomes 0 and others shift to 1..N
    remap = np.zeros_like(uniq, dtype=np.int32)
    # assign consecutive IDs skipping background
    next_id = 1
    for ulabel in range(len(uniq)):
        if ulabel == bg_label:
            remap[ulabel] = 0
        else:
            remap[ulabel] = next_id
            next_id += 1
    labels = remap[labels]
    return labels.astype(np.uint16)

converted = 0
for img in sorted(DATA.glob("*.ome.tif")):
    base = re.sub(r"\.ome\.tif$", "", img.name)
    mskp = img.with_name(base + "_masks.tif")
    if not mskp.exists():
        continue
    Y = tiff.imread(mskp)
    if Y.ndim == 3 and Y.shape[-1] in (3,4):
        lab = rgb_to_labels(Y)
        tiff.imwrite(mskp, lab)  # overwrite with proper label image
        print("fixed:", mskp.name, "->", lab.shape, lab.dtype, "max", lab.max())
        converted += 1

print("Converted", converted, "RGB masks to label images.")
