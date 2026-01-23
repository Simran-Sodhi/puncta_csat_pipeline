from aicsimageio import AICSImage
import tifffile, numpy as np, pathlib, time

# -------------------------
# FOCUS MEASURE
# -------------------------
def focus_score(img_2d: np.ndarray) -> float:
    """
    Compute a simple focus score for a 2D image using mean squared gradient magnitude.
    Higher value → sharper image.
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
    ch_this = [{"Name": n} for n in ch[:cyx.shape[0]]]

    out_2d = out_dir_2d / f"{inp.stem}_{s}_Z{best_z:03d}.ome.tif"
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
            # NOTE: no Plane[], no PhysicalSizeZ for CYX files
        },
    )

elapsed = time.perf_counter() - t0
print(f"Exported {len(img.scenes)} scene(s) in {elapsed:.2f}s")

# ------------- QUICK VERIFICATION -------------
from tifffile import TiffFile
first_file = sorted(out_dir_2d.glob(f"{inp.stem}_*.ome.tif"))[0]
print("Verifying:", first_file.name)
with TiffFile(str(first_file)) as tf:
    print("2D axes:", tf.series[0].axes)    # CYX
    print("2D shape:", tf.series[0].shape)  # (C, Y, X)