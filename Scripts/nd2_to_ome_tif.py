
from aicsimageio import AICSImage
import tifffile, numpy as np, pathlib
import time

inp = pathlib.Path("Raw Data/2025.09.11_H2B RFP P13_ID513 NES HOTag3 1ug.nd2")
img = AICSImage(inp)

start_time = time.perf_counter()

for i, s in enumerate(img.scenes):
    
    if i % 10 == 0:
        print("Current Point:", i)
    img.set_scene(s)
    data = img.get_image_data("CZYX")                 # (C, Z, Y, X)
    data = np.ascontiguousarray(data)                 # ensure memory layout
    px = img.physical_pixel_sizes
    ch = img.channel_names                   # safer API than .channel_names

    out = pathlib.Path("Ome_tifs") / f"{inp.stem}_{s}.ome.tif"
    tifffile.imwrite(
        out,
        data,
        ome=True,                                     # << force OME-XML
        imagej=False,                                 # << avoid ImageJ legacy tags
        photometric="minisblack",
        compression="deflate",
        bigtiff=True,
        metadata={
            "axes": "CZYX",
            "Channel": [{"Name": str(n)} for n in ch],
            "PhysicalSizeX": float(px.X), "PhysicalSizeY": float(px.Y), "PhysicalSizeZ": float(px.Z),
            "PhysicalSizeXUnit": "micrometer",
            "PhysicalSizeYUnit": "micrometer",
            "PhysicalSizeZUnit": "micrometer",
        },
    )

end_time = time.perf_counter()

elapsed_time = end_time - start_time

print(f"Code executed in {elapsed_time:.6f} seconds")


from tifffile import TiffFile
import re
p = "Ome_tifs/2025.09.11_H2B RFP P13_ID513 NES HOTag3 1ug_XYPos:0.ome.tif"  # note the underscore
with TiffFile(p) as tf:
    print("axes:", tf.series[0].axes)   # expect CZYX
    print("shape:", tf.series[0].shape) # expect (2, Z, Y, X)
    xml = tf.ome_metadata or ""
    print("SizeC:", re.search(r'SizeC="(\d+)"', xml).group(1))
    print("SizeZ:", re.search(r'SizeZ="(\d+)"', xml).group(1))
    print("SizeT:", re.search(r'SizeT="(\d+)"', xml).group(1))