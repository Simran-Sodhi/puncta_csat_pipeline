from pathlib import Path
import tifffile as tiff
import numpy as np
import re

DATA = Path("../Ome_tifs_2D")

bad = []
ok  = []
for img in sorted(DATA.glob("*.ome.tif")):
    base = re.sub(r"\.ome\.tif$", "", img.name)
    mskp = img.with_name(base + "_masks.tif")
    if not mskp.exists():
        # print("MISSING MASK:", mskp.name)
        # bad.append((img, "missing"))
        continue
    Y = tiff.imread(mskp)
    print(f"{mskp.name}: shape={Y.shape}, dtype={Y.dtype}, min={Y.min()}, max={Y.max()}")
    # valid mask must be 2D integer-like with at least one labeled object (>0)
    if Y.ndim != 2:
        bad.append((mskp, f"ndim={Y.ndim} (expected 2)"))
        continue
    if not np.issubdtype(Y.dtype, np.integer):
        bad.append((mskp, f"dtype={Y.dtype} (expected integer)"))
        continue
    n_objs = int(Y.max())
    if n_objs < 1:
        bad.append((mskp, "no objects (max label < 1)"))
    else:
        ok.append((mskp, n_objs))

print(f"\nOK masks: {len(ok)}")
print(f"Bad masks: {len(bad)}")
for p, why in bad[:10]:
    print(" -", p, "->", why)