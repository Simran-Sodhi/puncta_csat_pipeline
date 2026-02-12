from aicsimageio import AICSImage
import tifffile, numpy as np, pathlib, time

# -------------------------
# INPUT / OUTPUT FOLDERS
# -------------------------
inp = pathlib.Path("../../Raw Data/2025.09.11_H2B RFP P13_ID513 NES HOTag3 1ug.nd2")
out_dir_2d  = pathlib.Path("../Ome_tifs_2D_z5")   # one CYX file per Z plane
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
        if z != 5:
            continue
        cyx = data[:, z, :, :]                   # shape (C, Y, X)
        ch_this = ch[:cyx.shape[0]]

        # Build description with channel and resolution info
        desc_lines = [
            f"axes=CYX",
            f"channels={','.join(ch_this)}",
        ]
        if px.X:
            desc_lines.append(f"PhysicalSizeX={float(px.X):.6f}")
        if px.Y:
            desc_lines.append(f"PhysicalSizeY={float(px.Y):.6f}")
        desc_lines.append("PhysicalSizeUnit=micrometer")
        description = "\n".join(desc_lines)

        # Build resolution kwargs
        resolution_kwargs = {}
        if px.X and px.Y:
            # TIFF resolution unit 3 = CENTIMETER; store pixels per cm
            resolution_kwargs["resolution"] = (1e4 / float(px.X), 1e4 / float(px.Y))
            resolution_kwargs["resolutionunit"] = 3

        out_2d = out_dir_2d / f"{inp.stem}_{s}_Z{z:03d}.tif"
        tifffile.imwrite(
            out_2d,
            cyx,
            imagej=True,
            photometric="minisblack",
            compression="deflate",
            bigtiff=True,
            metadata={
                "axes": "CYX",
            },
            description=description,
            **resolution_kwargs,
        )

elapsed = time.perf_counter() - t0
print(f"Exported {len(img.scenes)} scene(s) in {elapsed:.2f}s")

# ------------- QUICK VERIFICATION -------------
from tifffile import TiffFile
p_2d = out_dir_2d / f"{inp.stem}_{img.scenes[0]}_Z005.tif"
with TiffFile(str(p_2d)) as tf:
    print("2D axes:", tf.series[0].axes)    # CYX
    print("2D shape:", tf.series[0].shape)  # (C, Y, X)
    page = tf.pages[0]
    if "XResolution" in page.tags:
        xr = page.tags["XResolution"].value
        print(f"XResolution tag: {xr[0]}/{xr[1]} pixels/cm")
