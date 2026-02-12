from aicsimageio import AICSImage
import tifffile, numpy as np, pathlib, time

# -------------------------
# FOCUS MEASURE
# -------------------------
def focus_score(img_2d: np.ndarray) -> float:
    """
    Compute a simple focus score for a 2D image using mean squared gradient magnitude.
    Higher value -> sharper image.
    """
    img = img_2d.astype(np.float32)
    gy, gx = np.gradient(img)
    return np.mean(gx**2 + gy**2)

# -------------------------
# INPUT / OUTPUT FOLDERS
# -------------------------
inp = pathlib.Path(
    "../../Raw Data/2025.11.14_HeLa H2B 670 P12_p899 1ug p599 1ug_Cytoplasm_006 - Denoised.nd2"
)
out_dir_2d  = pathlib.Path("../../Ome_tifs_DIC_2D_bestZ")
out_dir_2d.mkdir(parents=True, exist_ok=True)

img = AICSImage(inp)
t0 = time.perf_counter()

for i, s in enumerate(img.scenes):
    if i % 10 == 0:
        print("Current Point:", i)
    img.set_scene(s)

    # ---------- READ ----------
    # This will be (C, Z, Y, X) even if C=1
    data = img.get_image_data("CZYX")
    data = np.ascontiguousarray(data)
    px = img.physical_pixel_sizes
    ch = list(map(str, img.channel_names))  # ensure strings

    C, Z, Y, X = data.shape
    print(f"Scene {s}: data shape = (C={C}, Z={Z}, Y={Y}, X={X})")

    # ---------- FIND SHARPEST Z USING DIC (channel 0) ----------
    focus_scores = []
    for z in range(Z):
        dic_2d = data[0, z, :, :]   # channel 0 = DIC
        score = focus_score(dic_2d)
        focus_scores.append(score)

    best_z = int(np.argmax(focus_scores))
    print(f"Scene {s}: best focus at Z={best_z} (score={focus_scores[best_z]:.3e})")

    # ---------- WRITE ONLY THE BEST Z (axes CYX) ----------
    cyx = data[:, best_z, :, :]               # shape (C, Y, X)
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

    out_2d = out_dir_2d / f"{inp.stem}_{s}_Z{best_z:03d}.tif"
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
first_file = sorted(out_dir_2d.glob(f"{inp.stem}_*.tif"))[0]
print("Verifying:", first_file.name)
with TiffFile(str(first_file)) as tf:
    print("2D axes:", tf.series[0].axes)    # CYX
    print("2D shape:", tf.series[0].shape)  # (C, Y, X)
    page = tf.pages[0]
    if "XResolution" in page.tags:
        xr = page.tags["XResolution"].value
        print(f"XResolution tag: {xr[0]}/{xr[1]} pixels/cm")
