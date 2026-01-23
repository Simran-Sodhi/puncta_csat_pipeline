# save as open_ome_in_napari.py and run: python open_ome_in_napari.py /path/to/file.ome.tif
import sys, re
import numpy as np
import tifffile as tiff
import napari

path = sys.argv[1]

with tiff.TiffFile(path) as tf:
    arr = tf.asarray()                  # expect CZYX per your writer
    axes = tf.series[0].axes
    xml  = tf.ome_metadata or ""

# Confirm shape is CZYX; if not, reorder here as needed.
assert axes == "CZYX", f"Expected CZYX, got {axes}"

# Parse physical sizes (micrometers)
def _get(tag):
    m = re.search(fr'{tag}="([^"]+)"', xml)
    return float(m.group(1)) if m else None

px_x = _get("PhysicalSizeX")
px_y = _get("PhysicalSizeY")
px_z = _get("PhysicalSizeZ")

# Scale for napari: (z, y, x) in *world units per pixel*
scale = (px_z or 1.0, px_y or 1.0, px_x or 1.0)

viewer = napari.Viewer()
# Split channels into separate layers -> no channel slider
nC = arr.shape[0]
for c in range(nC):
    viewer.add_image(
        arr[c],                    # ZYX for this channel
        name=f"ch{c}",
        scale=scale,               # sets anisotropy correctly
        blending="translucent_no_depth",
        colormap=("green" if c == 0 else "magenta")  # adjust as you like
    )

napari.run()

