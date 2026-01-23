import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries

# --- Load ---
img = tiff.imread("../Ome_tifs_2D_cleaned/2025.09.11_H2B RFP P13_ID513 NES HOTag3 1ug_XYPos:172_Z002.ome.tif")   # raw image
mask = tiff.imread("../Ome_tifs_2D_cleaned/2025.09.11_H2B RFP P13_ID513 NES HOTag3 1ug_XYPos:172_Z002_masks.tif")       # integer mask (Y,X)

# Pick one channel if needed
if img.ndim == 3 and img.shape[0] <= 4:      # CYX
    img_gray = img[1]                        # cytoplasm channel
elif img.ndim == 3 and img.shape[-1] <= 4:   # YXC
    img_gray = img[...,1]
else:
    img_gray = img

# Normalize grayscale for display
img_gray = (img_gray - img_gray.min()) / (np.ptp(img_gray) + 1e-8)

# --- Compute outlines ---
boundaries = find_boundaries(mask, mode="outer")

# Convert grayscale to RGB
overlay = np.dstack([img_gray, img_gray, img_gray])

# Paint red outlines where boundaries are True
overlay[boundaries] = [1, 0, 0]   # red

# --- Show ---
plt.figure(figsize=(8,8))
plt.imshow(overlay)
plt.axis("off")
plt.show()

# --- Save ---
plt.imsave("overlay_outlines.png", overlay)
