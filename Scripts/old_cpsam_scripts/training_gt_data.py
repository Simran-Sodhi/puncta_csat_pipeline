# train_cellpose_cyto_only.py
from pathlib import Path
import numpy as np
from cellpose import models, train, io
import torch


print("torch:", torch.__version__)
print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())
torch.set_default_dtype(torch.float32)

# X, Y, _, _, _, _ = io.load_train_test_data("Drive data/Cellpose Train/mScarlet_Train1",
#                                      mask_filter="_seg.npy")
# X = [x.astype(np.float32, copy=False) for x in X]  # images
# Y = [y.astype(np.int32,   copy=False) for y in Y]  # labeled masks

# print(f"Found {len(X)} training pairs")

# train_cellpose_cyto_only.py

# ---- device setup ✨ ----
use_mps = torch.backends.mps.is_available()
device  = torch.device("mps") if use_mps else torch.device("cpu")
print("Using device:", device)

# (optional but helpful on Apple Silicon)
torch.set_default_dtype(torch.float32)
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

# ---- data ----
X, Y, *_ = io.load_train_test_data("Drive data/Cellpose Train/mScarlet_Train1", mask_filter="_seg.npy")
X = [x.astype(np.float32, copy=False) for x in X]
Y = [y.astype(np.int32,   copy=False) for y in Y]
print(f"Found {len(X)} training pairs")

# ---- model on MPS explicitly ✨ ----
model = models.CellposeModel(
    pretrained_model="models/my_new_model_5",
    gpu=use_mps,                 # tells cellpose to use non-CPU
    device=device                # explicitly pick mps
)

# confirm the net is on MPS ✨
print("Model param device:", next(model.net.parameters()).device)

save_dir = Path("cellpose_models/cpsam_double_finetune")
save_dir.mkdir(parents=True, exist_ok=True)

model_path = train.train_seg(
    net=model.net,
    train_data=X,
    train_labels=Y,
    #channels=[0,0],              # ✨ set channels explicitly
    n_epochs=35,
    save_path=str(save_dir),
    weight_decay=0.1,
    save_every=25,
    batch_size=1,                # ✨ MPS is happiest with 1
    learning_rate=1e-5,
    min_train_masks=1,
    model_name="cpsam_double_fine_tune_50_images_35_eps",
)
print("Training complete:", model_path)

# start from bioimage.io model
# model = models.CellposeModel(pretrained_model="Drive data/Cellpose Train/custom_model/loyal-parrot/cell_model_state_dict.pt",
#                              gpu=True) 

# save_dir = Path("cellpose_models/loyal_parrot_finetune")
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
# model_path = train.train_seg(
#     net=model.net,
#     train_data=X,         # list of (Y,X) arrays
#     train_labels=Y,
#     # channel_axis=None,         # grayscale
#     n_epochs=100,
#     # batch_size=8,
#     # min_train_masks=1,
#     save_path=str(save_dir),
#     weight_decay=0.1, 
#     save_every=25,
#     batch_size=4,
#     learning_rate=1e-5,
#     min_train_masks=1,
#     model_name="loyal_parrot_50_images"
# )


# print("Training complete:", model_path)
