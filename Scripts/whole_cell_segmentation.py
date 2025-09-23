# pip install cellpose tifffile numpy
import numpy as np, tifffile as tiff
from cellpose import models, io

def load_ome(path):
    with tiff.TiffFile(path) as tf:
        arr = tf.asarray()          # shape can be e.g. ZCYX, CZYX, TCZYX, etc.
        axes = tf.series[0].axes    # e.g. 'ZCYX'
    return arr, axes

def get_axis_map(axes):
    return {ax:i for i,ax in enumerate(axes)}

def to_ZCYX(arr, axes):
    ax = get_axis_map(axes)
    order = [ax.get('Z', None), ax.get('C', None), ax['Y'], ax['X']]
    # expand missing Z or C if absent
    if order[0] is None:
        arr = np.expand_dims(arr, 0); order[0] = 0
    if order[1] is None:
        arr = np.expand_dims(arr, 0); order[1] = 0
    return np.moveaxis(arr, order, (0,1,2,3))  # -> (Z, C, Y, X)

def pnorm(img2d, p1=1, p99=99):
    lo, hi = np.percentile(img2d, (p1, p99))
    if hi <= lo: return np.zeros_like(img2d, dtype=np.uint8)
    x = np.clip((img2d - lo) / (hi - lo), 0, 1)
    return (x * 255).astype(np.uint8)

# --- load ---
path = "Ome_tifs/2025.09.11_H2B RFP P13_ID513 NES HOTag3 1ug_XYPos:0.ome.tif"
arr, axes = load_ome(path)
ZCYX = to_ZCYX(arr, axes)            # (Z, C, Y, X)

# pick Cy3 channel (if you know the index, set it directly)
cy3_idx = 0  # <-- change this to your Cy3 channel index if needed
cy3_stack = ZCYX[:, cy3_idx]         # (Z, Y, X)

# 2D projection of Cy3
proj = np.max(cy3_stack, axis=0)     # (Y, X)
proj = pnorm(proj, 1, 99)            # uint8 normalization

# CP-SAM (v4) expects 3 channels: stack Cy3 + two blanks; tell channel_axis=0
img3 = np.stack([proj, np.zeros_like(proj), np.zeros_like(proj)], axis=0)  # (3, Y, X)

# ---- segment ----
model = models.CellposeModel(gpu=True, pretrained_model="cpsam")  # v4 default
masks, flows, styles = model.eval(
    img3,
    channel_axis=0,          # channels are first axis in img3
    normalize=True,          # v4 can normalize internally
    diameter=None,           # optional; you can still set pixels if you want
    cellprob_threshold=-5.5, # good start if Cy3 is dim; adjust as needed
    flow_threshold=0.4,
    do_3D=False
)

io.imsave("proj_cy3.tif", proj.astype(np.uint8))
io.imsave("masks_cy3.tif", masks.astype(np.uint16))
print("Saved masks_cy3.tif")
