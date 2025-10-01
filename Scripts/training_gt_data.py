# train_cellpose.py
from pathlib import Path
import tifffile as tiff
import numpy as np
from cellpose import models
from cellpose import train

# Folder with paired files:   <name>.ome.tif   and   <name>_masks.tif
DATA = Path("../Ome_tifs_2D_cleaned")

# Pick your channels (edit to match your data)
# Example: cytoplasm = green channel index 1 ; nuclei = red channel index 0
CHAN  = 1   # cytoplasm channel
CHAN2 = 0   # optional nuclear channel (0 to disable, else integer index)

# Collect training pairs
X_list, Y_list, CH_list = [], [], []
for img in sorted(DATA.glob("*.ome.tif")):
    base = img.name.replace('.ome.tif', '')
    msk = img.with_name(base + "_masks.tif")
    if not msk.exists():
        continue
    X = tiff.imread(img)   # shape could be (C,Y,X) or (Y,X,C); we’ll normalize
    X = np.asarray(X)
    # Normalize to CYX for cellpose (channels-first)
    if X.ndim == 3 and X.shape[-1] in (2,3,4):  # YXC -> CYX
        X = np.moveaxis(X, -1, 0)
    elif X.ndim == 2:                            # single-channel image
        X = X[None, ...]
    assert X.ndim == 3 and X.shape[0] >= max(CHAN, CHAN2, 0)

    Y = tiff.imread(msk).astype(np.int32)        # labels image (H,W)
    X_list.append(X)
    Y_list.append(Y)
    CH_list.append([CHAN, CHAN2])                # one [chan,chan2] per image

print(f"Found {len(X_list)} training pairs")

# Create model: start from cyto2
model = models.CellposeModel(pretrained_model='cyto2', gpu=True)

# Train
save_dir = Path("cellpose_models/my_cyto2_finetune")
save_dir.mkdir(parents=True, exist_ok=True)

train.train_seg(
    net=model.net,
    train_data=X_list,
    train_labels=Y_list, 
    # channels=CH_list,                  # list of [chan, chan2]
    channel_axis=0, 
    n_epochs=30,
    learning_rate=0.2,
    batch_size=8,
    min_train_masks=1,
    save_path=str(save_dir)
)
# model.train(
#     train_data=X_list,                 # list of arrays (C,Y,X)
#     train_labels=Y_list,               # list of label images (Y,X)
#     channels=CH_list,                  # list of [chan, chan2]
#     channel_axis=0,                    # channels-first
#     # Common knobs (tune as you like)
#     n_epochs=100,
#     learning_rate=0.2,
#     batch_size=8,
#     min_train_masks=1,
#     save_path=str(save_dir)
# )
print("Training complete. Model saved in:", save_dir)
