# train_cellpose_cyto_only.py
from pathlib import Path
import numpy as np
import tifffile as tiff
from cellpose import models, train

DATA = Path("Ome_tifs_2D_cleaned_new")
CHAN  = 1   # cytoplasm index in your OME files

X_list, Y_list = [], []
for img in sorted(DATA.glob("*.ome.tif")):
    base = img.name.replace(".ome.tif", "")
    msk = img.with_name(base + "_masks.tif")
    if not msk.exists():
        continue

    X = tiff.imread(img)            # (C,Y,X) or (Y,X,C) or (Y,X)
    X = np.asarray(X)

    # normalize to (C,Y,X)
    if X.ndim == 3 and X.shape[-1] in (2,3,4):   # YXC -> CYX
        X = np.moveaxis(X, -1, 0)
    elif X.ndim == 2:                             # YX
        X = X[None, ...]
    assert X.ndim == 3 and CHAN < X.shape[0], f"Bad shape {X.shape} for {img}"

    # ---- take ONLY cytoplasm channel -> grayscale (Y,X)
    X_gray = X[CHAN].astype(np.float32)

    Y = tiff.imread(msk).astype(np.int32)        # (Y,X) integer labels
    X_list.append(X_gray)
    Y_list.append(Y)

print(f"Found {len(X_list)} training pairs")

# start from cyto2
model = models.CellposeModel(pretrained_model='cyto', gpu=True)

# save_dir = Path("cellpose_models/my_cyto2_finetune")
# save_dir.mkdir(parents=True, exist_ok=True)

# train.train_seg(
#     net=model.net,
#     train_data=X_list,         # list of (Y,X) arrays
#     train_labels=Y_list,
#     channel_axis=None,         # grayscale
#     n_epochs=30,
#     learning_rate=0.2,
#     batch_size=8,
#     min_train_masks=1,
#     save_path=str(save_dir),
# )
model_path = train.train_seg(
    net=model.net,
    train_data=X_list,         # list of (Y,X) arrays
    train_labels=Y_list,
    # channel_axis=None,         # grayscale
    n_epochs=35,
    # batch_size=8,
    # min_train_masks=1,
    # save_path=str(save_dir),
    weight_decay=0.1, 
    load_files=False,
    learning_rate=1e-5,
    min_train_masks=1,
    model_name="my_new_model_7"
)


print("Training complete:", model_path)
