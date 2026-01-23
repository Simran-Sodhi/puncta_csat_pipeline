#!/usr/bin/env python3
from pathlib import Path
import tifffile as tiff
import numpy as np
src = Path("../../Ome_tifs_2D_cleaned_new_test")          # where your .ome.tif and *_masks.tif live
dst = Path("../../Ome_tifs_2D_cleaned_new_2_channel"); dst.mkdir(exist_ok=True)

for img in sorted(src.glob("*.ome.tif")):
    arr = tiff.imread(img)             # expect (C,Y,X) or (Z,C,Y,X)
    if arr.ndim == 3:                  # (C,Y,X)
        nuc, cyto = arr[0], arr[1]
    elif arr.ndim == 4:                # (Z,C,Y,X) -> max project
        nuc  = arr[:,0].max(axis=0)
        cyto = arr[:,1].max(axis=0)
    else:
        raise ValueError(f"Unexpected shape {arr.shape} for {img.name}")
    twoch = np.stack([cyto, nuc], axis=0).astype(np.float32)  # keep order consistent with --chan 0 --chan2 1 later
    out = dst / img.name
    tiff.imwrite(out.as_posix(), twoch)
    # copy/rename mask next to it
    m = src / img.name.replace(".tif", "_masks.tif")
    if m.exists():
        (dst / m.name).write_bytes(m.read_bytes())
    else:
        print("WARNING: missing mask for", img.name)