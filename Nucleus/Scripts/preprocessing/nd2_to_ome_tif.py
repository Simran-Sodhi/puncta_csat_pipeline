"""
nd2_to_ome_tif.py: This file converts the nd2 file to ome tif files for z = 8 (in this case) and creates such a file for each position/scene
"""

from aicsimageio import AICSImage
import tifffile, numpy as np, pathlib, time

# -------------------------
# INPUT / OUTPUT FOLDERS
# -------------------------

# Path to nd2 file
raw_path = pathlib.Path(
    "/Users/simransodhi/Desktop/CMU/Fall 2025/RA Work/image_analysis/Nucleus_Puncta/Raw_Data/2025.11.21_HeLa P9_p936 1ug p893 1ug_Nucleus/2025.11.21_HeLa P9_p936 1ug p893 1ug_Nucleus_001.nd2"
)

print("Raw path exists:", raw_path.exists())
print("Raw path:", raw_path)

inp = pathlib.Path(raw_path)
out_dir_2d  = pathlib.Path("../../Processed_Data/21_nov_data/Ome_tifs_2D_z8")   # one CYX file per Z plane
out_dir_2d.mkdir(parents=True, exist_ok=True)

img = AICSImage(inp)
t0 = time.perf_counter()

for i, s in enumerate(img.scenes):
    if i % 10 == 0:
        print("Current Point:", i)
    img.set_scene(s)

    # ---------- READ ----------
    data = img.get_image_data("CZYX")   # (C, Z, Y, X)
    data = np.ascontiguousarray(data)
    px = img.physical_pixel_sizes
    ch = list(map(str, img.channel_names))  # ensure strings

    C, Z, Y, X = data.shape

    # ---------- WRITE ONE FILE PER Z (axes CYX) ----------
    for z in range(Z):
        if z != 8:
            continue
        cyx = data[:, z, :, :]                   # shape (C, Y, X)
        # slice channel names to the actual C we’re writing
        ch_this = [{"Name": n} for n in ch[:cyx.shape[0]]]

        out_2d = out_dir_2d / f"{inp.stem}_{s}_Z{z:03d}.ome.tif"
        tifffile.imwrite(
            out_2d,
            cyx,
            ome=True,
            imagej=False,
            photometric="minisblack",
            compression="deflate",
            bigtiff=True,
            metadata={
                "axes": "CYX",                          # matches (C,Y,X)
                "Channel": ch_this,                     # length == C
                "PhysicalSizeX": float(px.X) if px.X else None,
                "PhysicalSizeY": float(px.Y) if px.Y else None,
                "PhysicalSizeXUnit": "micrometer",
                "PhysicalSizeYUnit": "micrometer",
            },
        )

elapsed = time.perf_counter() - t0
print(f"Exported {len(img.scenes)} scene(s) in {elapsed:.2f}s")

# ------------- QUICK VERIFICATION -------------
from tifffile import TiffFile
p_2d = out_dir_2d / f"{inp.stem}_{img.scenes[0]}_Z008.ome.tif"
with TiffFile(str(p_2d)) as tf:
    print("2D axes:", tf.series[0].axes)    # CYX
    print("2D shape:", tf.series[0].shape)  # (C, Y, X)
