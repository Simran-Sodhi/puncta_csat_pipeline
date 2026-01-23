#!/usr/bin/env python3
import os, argparse, numpy as np, tifffile as tiff, xml.etree.ElementTree as ET
from pathlib import Path
from cellpose import models, io
from scipy import ndimage as ndi
from skimage import filters, morphology, segmentation, measure,  exposure, transform
# ---------- OME helpers ----------
def read_ome(path):
    with tiff.TiffFile(path) as tf:
        arr = tf.asarray()
        axes = tf.series[0].axes  # e.g. 'TCZYX', 'CZYX', 'ZCYX', 'CYX', 'YX'
        meta_xml = tf.ome_metadata
    return arr, axes, meta_xml

def to_ZCYX(arr, axes):
    """
    Reorder to (Z, C, Y, X), adding singleton Z/C if missing.
    Works for most common OME axis strings.
    """
    ax = {a: i for i, a in enumerate(axes)}
    # add missing Z or C dims as singleton
    if 'Z' not in ax:
        arr = np.expand_dims(arr, 0)
        axes = 'Z' + axes
    if 'C' not in ax:
        # insert C after Z (or at start)
        arr = np.expand_dims(arr, 0)
        axes = axes[:1] + 'C' + axes[1:]
    ax = {a: i for i, a in enumerate(axes)}
    order = [ax['Z'], ax['C'], ax['Y'], ax['X']]
    out = np.moveaxis(arr, order, (0, 1, 2, 3))
    return out

def channel_names_from_ome(meta_xml):
    names = []
    if not meta_xml:
        return names
    root = ET.fromstring(meta_xml)
    ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
    for ch in root.findall('.//ome:Image/ome:Pixels/ome:Channel', ns):
        nm = ch.get('Name') or ch.get('ID') or ''
        names.append(nm)
    return names

def pick_cy3_idx(ch_names, hints):
    """Pick Cy3 channel by name (case-insensitive)."""
    if not ch_names:
        return 0
    lname = [str(n).lower() for n in ch_names]
    for h in hints:
        h = str(h).lower()
        for i, n in enumerate(lname):
            if h in n:
                return i
    return 0  # fallback

# ---------- simple viz helpers ----------
def percentile_norm(img2d, p1=1, p99=99):
    lo, hi = np.percentile(img2d, (p1, p99))
    if hi <= lo:
        return np.zeros_like(img2d, dtype=np.uint8)
    x = np.clip((img2d - lo) / (hi - lo), 0, 1)
    return (x * 255).astype(np.uint8)

def random_label_colors(mask, seed=0):
    """Return RGB image coloring each label (0 stays black)."""
    H, W = mask.shape
    K = int(mask.max()) + 1
    rng = np.random.default_rng(seed)
    palette = rng.integers(0, 255, size=(K, 3), dtype=np.uint8)
    palette[0] = 0
    rgb = palette[mask]
    return rgb

def mask_outlines(mask):
    """Boolean outline map on mask pixels (4-neighborhood)."""
    m = mask > 0
    edge = np.zeros_like(m, dtype=bool)
    edge[:-1, :] |= m[:-1, :] != m[1:, :]
    edge[1:,  :] |= m[1:,  :] != m[:-1, :]
    edge[:, :-1] |= m[:, :-1] != m[:, 1:]
    edge[:, 1:]  |= m[:, 1:]  != m[:, :-1]
    return edge & m

def overlay_outlines_on_gray(gray, mask, edge_color=(255, 0, 0), alpha_fill=0.0):
    """
    gray: uint8 (Y,X)
    mask: uint16/uint32 labels (Y,X)
    - draws red outlines
    - optionally alpha blends filled labels if alpha_fill>0
    """
    rgb = np.dstack([gray, gray, gray]).astype(np.uint8)
    if alpha_fill > 0:
        fill = random_label_colors(mask)
        rgb = (rgb*(1-alpha_fill) + fill*alpha_fill).astype(np.uint8)
    edges = mask_outlines(mask)
    rgb[edges] = np.array(edge_color, dtype=np.uint8)
    return rgb


def save_triptych(gray_u8, labels_u16, out_png, edge_color=(255, 0, 0), fill_alpha=0.35, dpi=150):
    """
    Save a 1x3 panel PNG:
      [0] raw grayscale image
      [1] colorized instance mask
      [2] filled instance colors alpha-blended on image + red outlines
    """
    import matplotlib.pyplot as plt

    # panel [1] — colorized labels
    rgb_labels = random_label_colors(labels_u16)

    # panel [2] — filled & outlined composite on top of grayscale
    base = np.dstack([gray_u8, gray_u8, gray_u8]).astype(np.uint8)
    filled = (base * (1 - fill_alpha) + rgb_labels * fill_alpha).astype(np.uint8)
    edges = mask_outlines(labels_u16)
    filled[edges] = np.array(edge_color, dtype=np.uint8)

    # draw figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=dpi)
    axes[0].imshow(gray_u8, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Image'); axes[0].axis('off')
    axes[1].imshow(rgb_labels)
    axes[1].set_title('Masks'); axes[1].axis('off')
    axes[2].imshow(filled)
    axes[2].set_title('Masks over image'); axes[2].axis('off')
    plt.tight_layout()
    fig.savefig(out_png, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def segment_nuclei_binary(nuc_slice_u8, min_area=50):
    """Simple per-slice nuclei mask: Otsu threshold + cleanup."""
    if nuc_slice_u8.max() == 0:
        return np.zeros_like(nuc_slice_u8, dtype=bool)
    thr = filters.threshold_otsu(nuc_slice_u8)
    m = nuc_slice_u8 > thr
    m = morphology.remove_small_objects(m, min_size=min_area)
    m = morphology.remove_small_holes(m, area_threshold=min_area)
    return m

def split_with_nuclei_on_slice(cyto_u8, mask_slice, nuc_u8):
    """
    Split fused cell masks on one Z-slice using nuclei as seeds.
    - cyto_u8 : uint8 cytoplasm slice
    - mask_slice : uint16 labels from Cellpose (Y,X)
    - nuc_u8 : uint8 nuclei slice
    Returns a new uint16 mask slice.
    """
    if mask_slice.max() == 0:
        return mask_slice

    # 1) nuclei → markers
    nuc_bin = segment_nuclei_binary(nuc_u8)
    markers = measure.label(nuc_bin)  # instance seeds

    if markers.max() <= 1:
        # not enough nuclei info; just return original
        return mask_slice

    # 2) boundary cue from cyto gradient
    grad = filters.sobel(cyto_u8.astype(np.float32))

    # 3) watershed per connected component of the original mask
    out = np.zeros_like(mask_slice, dtype=np.uint16)
    cur = 1
    for lab in range(1, mask_slice.max() + 1):
        region = (mask_slice == lab)
        if not region.any():
            continue
        # restrict markers to this region
        local_markers = markers * 0
        # keep only nuclei that fall inside this region
        inside = markers[region]
        if inside.max() <= 1:
            # nothing to split here
            out[region] = cur
            cur += 1
            continue
        # keep those marker ids
        keep_ids = np.unique(inside)
        keep_ids = keep_ids[keep_ids > 0]
        lm = np.zeros_like(markers, dtype=np.int32)
        for i, mid in enumerate(keep_ids, start=1):
            lm[markers == mid] = i
        ws = segmentation.watershed(grad, markers=lm, mask=region)
        for sid in range(1, ws.max() + 1):
            out[ws == sid] = cur
            cur += 1
    return out

def auto_lut_clip(img_list, low_percentile=2.0, high_percentile=99.8):
    """
    Apply viewer-style LUT clipping:
    - Values below low_percentile -> set to 0
    - Values above high_percentile -> set to 1
    - Everything else scaled linearly between 0 and 1
    (no global stretching of full dynamic range)
    """
    new_img_list = []
    for img in img_list:
        img = img.astype(np.float32)
        lo = np.percentile(img, low_percentile)
        hi = np.percentile(img, high_percentile)
        
        img_clipped = np.clip(img, lo, hi)
        img_clipped = (img_clipped - lo) / (hi - lo + 1e-8)
        img_clipped[img < lo] = 0.0  # fully black background
        new_img_list.append(img_clipped.astype(np.float32))
    return new_img_list

# ---------- light background subtraction for 2.5D ----------
def percentile_bg_sub(zimg, p=2):
    # subtract a low percentile per-slice to cut haze, keep >= 0
    bg = np.percentile(zimg, p)
    out = zimg.astype(np.float32) - bg
    out[out < 0] = 0
    return out.astype(zimg.dtype)

# ---------- MODE A: MIP → 2D segmentation (original approach) ----------
def process_one_mip2d(path, out_proj_dir, out_mask2d_dir, out_prev_dir,
                      cy3_idx=None, cy3_hints=('cy3', 'mscarlet', 'm-scarlet', '568'),
                      cellprob_threshold=-5.5, pretrained_model = "cyto2", flow_threshold=0.4, diameter=None, gpu=True):
    arr, axes, meta = read_ome(path)
    ZCYX = to_ZCYX(arr, axes)
    ch_names = channel_names_from_ome(meta)
    if cy3_idx is None:
        cy3_idx = pick_cy3_idx(ch_names, cy3_hints)

    cy3_stack = ZCYX[:, cy3_idx]             # (Z, Y, X)
    proj = np.max(cy3_stack, axis=0)         # 2D max projection
    proj_u8 = percentile_norm(proj, 1, 99)

    # CP-SAM expects 1–3 channels; we can give grayscale using channel_axis=None
    model = models.CellposeModel(gpu=gpu, pretrained_model=pretrained_model)
    masks, flows, styles = model.eval(
        proj_u8,
        channel_axis=None,        # grayscale
        do_3D=False,
        normalize=True,
        diameter=diameter,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold
    )
    masks = masks.astype(np.uint16)

    stem = Path(path).stem.replace('.ome', '')
    proj_tif = os.path.join(out_proj_dir, f"{stem}_proj.tif")
    mask2d_tif = os.path.join(out_mask2d_dir, f"{stem}_mask2d.tif")
    png_proj = os.path.join(out_prev_dir, f"{stem}_proj.png")
    png_mask = os.path.join(out_prev_dir, f"{stem}_mask.png")
    png_overlay = os.path.join(out_prev_dir, f"{stem}_overlay.png")

    io.imsave(proj_tif, proj_u8)
    io.imsave(mask2d_tif, masks)

    mask_rgb = random_label_colors(masks)
    overlay_rgb = overlay_outlines_on_gray(proj_u8, masks, edge_color=(255, 0, 0), alpha_fill=0.0)
    tiff.imwrite(png_proj, proj_u8, photometric='minisblack')
    tiff.imwrite(png_mask, mask_rgb)
    tiff.imwrite(png_overlay, overlay_rgb)

    return {
        "proj_tif": proj_tif,
        "mask2d_tif": mask2d_tif,
        "png_proj": png_proj,
        "png_mask": png_mask,
        "png_overlay": png_overlay,
        "cy3_idx": cy3_idx,
        "channel_names": ch_names
    }

# ---------- MODE B: 2.5D (2D per-slice on GPU + 3D linking) ----------


from scipy import ndimage as ndi
import numpy as np
import tifffile as tiff
from cellpose import models, io

def _lap_var(img):
    return ndi.variance(ndi.laplace(img.astype(np.float32)))

def _pick_focus_band(stack, half_width=2):
    # pick slice with max focus and keep ±half_width around it
    scores = np.array([_lap_var(z) for z in stack])
    z0 = int(scores.argmax())
    zlo = max(0, z0 - half_width)
    zhi = min(stack.shape[0], z0 + half_width + 1)
    return np.arange(zlo, zhi)

def _clip_and_smooth(stack, lo=1, hi=99.5, sigma=1.5):
    # global percentile clip, per-slice normalize, light Gaussian smooth
    lo_v, hi_v = np.percentile(stack, (lo, hi))
    stack = np.clip(stack, lo_v, hi_v)
    out = np.empty_like(stack, dtype=np.uint8)
    for i in range(stack.shape[0]):
        z = stack[i].astype(np.float32)
        if sigma and sigma > 0:
            z = ndi.gaussian_filter(z, sigma=sigma)
        mn, mx = np.percentile(z, (1, 99))
        if mx <= mn:
            out[i] = 0
        else:
            z = (np.clip(z, mn, mx) - mn) / (mx - mn) * 255.0
            out[i] = z.astype(np.uint8)
    return out

def to_png_like(img16, out_shape=None, p_lo=1.0, p_hi=99.8, gamma=0.45, to_uint8=True):
    img = img16.astype(np.float32)
    lo, hi = np.percentile(img, (p_lo, p_hi))
    img = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
    if gamma is not None:
        img = np.power(img, gamma)
    if out_shape is not None and out_shape != img.shape[:2]:
        img = transform.resize(img, out_shape, order=1, preserve_range=True, anti_aliasing=True)
    if to_uint8:
        img = (img * 255).round().astype(np.uint8)
    return img

def process_one_vol2p5d(path, out_mask3d_dir, out_prev_dir,
                        cy3_idx=None, cy3_hints=('cy3','mscarlet','m-scarlet','568'),
                        cellprob_threshold=-3.0, flow_threshold=0.5,
                        diameter=80, gpu=True, bg_percentile=None,
                        pretrained_model="cpsam", min_size=100,
                        focus_halfwidth=4, batch_size=8,
                        nuc_idx=None, nuc_hints=('cy5','h2b','dapi','hoechst','647','dna'),
                        use_nuclei_post=False, nuc_min_area=50, debug_raw = True):
    # --- load ---
    arr, axes, meta = read_ome(path)
    ZCYX = to_ZCYX(arr, axes)                 # (Z,C,Y,X)
    ch_names = channel_names_from_ome(meta)
    if cy3_idx is None:
        cy3_idx = pick_cy3_idx(ch_names, cy3_hints)
    cy3_stack = ZCYX[:, cy3_idx].astype(arr.dtype)  # (Z,Y,X)

    nuc_stack = None
    if use_nuclei_post:
        if nuc_idx is None:
            # pick by hints, default to "not cyto" if possible
            all_names = [s.lower() for s in ch_names] if ch_names else []
            fallback = 0
            if cy3_idx == 0 and ZCYX.shape[1] > 1:
                fallback = 1
            nuc_idx = pick_cy3_idx(ch_names, nuc_hints) if ch_names else fallback
        nuc_stack = ZCYX[:, nuc_idx].astype(arr.dtype)

    # light background subtraction
    if bg_percentile is not None:
        bg = np.percentile(cy3_stack, bg_percentile)
        cy3_stack = np.clip(cy3_stack.astype(np.float32) - bg, 0, None).astype(arr.dtype)

    # --- focus band + preprocess ---
    keep_z = _pick_focus_band(cy3_stack, half_width=focus_halfwidth)
    sub = cy3_stack[keep_z]
    sub_u8 = _clip_and_smooth(sub, lo=1, hi=99.5, sigma=1.0)  # uint8 stack

    sub_nuc_u8 = None
    if nuc_stack is not None:
        sub_nuc = nuc_stack[keep_z]
        sub_nuc_u8 = _clip_and_smooth(sub_nuc, lo=1, hi=99.5, sigma=1.0)

    # --- batch 2D eval (one call) ---
    # print(pretrained_model)
    model = models.CellposeModel(gpu=gpu, pretrained_model=pretrained_model)
    # print(model)
    # print(model.net)
    # total_params = sum(p.numel() for p in model.net.parameters())
    # trainable_params = sum(p.numel() for p in model.net.parameters() if p.requires_grad)
    # print("Total parameters:", total_params)
    # print("Trainable parameters:", trainable_params)
    img_list_temp = [sub_u8[i] for i in range(sub_u8.shape[0])]
   
    # img_list = [np.power(img, 1/2.2) for img in img_list_temp]
    H, W = sub.shape[-2:]
    out_shape = (H//2, W//2)
    img_list = [transform.resize(img, out_shape, order=1, preserve_range=True, anti_aliasing=True) for img in img_list_temp]
    img_norm = auto_lut_clip(
                img_list,
                low_percentile=2.0,
                high_percentile=99.8,
            )

    # out_hw = (H, W)
    # img_list = [to_png_like(img, out_shape=out_hw, p_lo=0.1, p_hi=99.9, gamma=0.4545) for img in sub]
    masks_list, flows, styles = model.eval(
        img_norm,
        channel_axis=None,     # grayscale
        do_3D=False,
        normalize=True,
        diameter=diameter,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
        min_size=min_size,     # remove tiny blobs (puncta)
        # net_avg=False,         # faster
        # augment=False,         # faster
        # batch_size=batch_size  # fewer forward passes
    )
    # raw_masks_stack = np.zeros_like(sub, dtype=np.uint16)
    # for i in range(sub_u8.shape[0]):
    #     raw_masks_stack[i] = masks_list[i].astype(np.uint16)

            # --- OPTIONAL nuclei-seeded splitting (per slice) ---
    # if use_nuclei_post and sub_nuc_u8 is not None:
    #     for i in range(sub_u8.shape[0]):
    #         m = masks_list[i].astype(np.uint16)
    #         # quick clean: zero tiny specks (same threshold you use in eval)
    #         m = morphology.remove_small_objects(m > 0, min_size=min_size).astype(np.uint16) * 1
    #         m = m.astype(np.uint16)
    #         masks_list[i] = split_with_nuclei_on_slice(
    #             sub_u8[i], m, sub_nuc_u8[i]
    #         )

    # --- rebuild full-Z label volume (zeros outside focus band) ---
    Z, H, W = cy3_stack.shape
    mH, mW = masks_list[0].shape
    masks_slices = np.zeros((Z, H, W), dtype=np.uint16)
    for i, z in enumerate(keep_z):
        m = masks_list[i].astype(np.uint16)
        if (mH, mW) != (H, W):
         m = transform.resize(
                m, (H, W),
                order=0, preserve_range=True, anti_aliasing=False
            ).astype(np.uint16)
        masks_slices[z] = m
        # masks_slices[z] = masks_list[i].astype(np.uint16)
        

    # --- 3D linking (6-connectivity) ---
    fg_3d = masks_slices > 0
    structure = ndi.generate_binary_structure(3, 1)
    mask3d, _ = ndi.label(fg_3d, structure=structure)
    mask3d = mask3d.astype(np.uint16)

    # --- save preview + mask ---
    stem = Path(path).stem.replace('.ome','')
    mask3d_tif = os.path.join(out_mask3d_dir, f"{stem}_mask3d.tif")
    io.imsave(mask3d_tif, mask3d)

    mip_signal = np.max(cy3_stack, axis=0)
    mip_u8 = percentile_norm(mip_signal, 1, 99)
    mip_labels = np.max(mask3d, axis=0).astype(np.uint16)

    png_triptych = os.path.join(out_prev_dir, f"{stem}_triptych.png")
    save_triptych(mip_u8, mip_labels, png_triptych, edge_color=(255, 0, 0), fill_alpha=0.35)
    # overlay = overlay_outlines_on_gray(mip_u8, mip_labels, edge_color=(255,0,0), alpha_fill=0.0)
    # png_overlay = os.path.join(out_prev_dir, f"{stem}_overlay.png")
    # tiff.imwrite(png_overlay, overlay)

    return {
        "mask3d_tif": mask3d_tif,
        "png_triptych": png_triptych,
        # "png_overlay": png_overlay,
        "n_objects": int(mask3d.max()),
        "cy3_idx": cy3_idx,
        "channel_names": ch_names
    }


# ---------- MODE C: true 3D Cellpose on CPU (optional) ----------
def process_one_vol3d_cpu(path, out_mask3d_dir, out_prev_dir,
                          cy3_idx=None, cy3_hints=('cy3','mscarlet','m-scarlet','568'),
                          cellprob_threshold=-5.5, flow_threshold=0.4,
                          diameter=None, gpu=False, pretrained_model = "cyto2", anisotropy=None):
    """
    True 3D eval. Works on CPU on macOS (MPS 3D is not supported by Cellpose).
    """
    arr, axes, meta = read_ome(path)
    ZCYX = to_ZCYX(arr, axes)
    ch_names = channel_names_from_ome(meta)
    if cy3_idx is None:
        cy3_idx = pick_cy3_idx(ch_names, cy3_hints)
    cy3_stack = ZCYX[:, cy3_idx]              # (Z,Y,X)

    model = models.CellposeModel(gpu=False, pretrained_model=pretrained_model)  # force CPU for 3D on macOS

    masks3d, flows, styles = model.eval(
        cy3_stack,                 # (Z,Y,X)
        channel_axis=None,         # grayscale
        z_axis=0,
        do_3D=True,
        normalize=True,
        diameter=diameter,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
        anisotropy=anisotropy
    )
    masks3d = masks3d.astype(np.uint16)

    stem = Path(path).stem.replace('.ome', '')
    mask3d_tif = os.path.join(out_mask3d_dir, f"{stem}_mask3d.tif")
    io.imsave(mask3d_tif, masks3d)

    proj_u8 = percentile_norm(np.max(cy3_stack, axis=0), 1, 99)
    mip_labels = np.max(masks3d, axis=0).astype(np.uint16)
    png_overlay = os.path.join(out_prev_dir, f"{stem}_overlay.png")
    overlay = overlay_outlines_on_gray(proj_u8, mip_labels, edge_color=(255, 0, 0), alpha_fill=0.0)
    tiff.imwrite(png_overlay, overlay)

    return {
        "mask3d_tif": mask3d_tif,
        "png_overlay": png_overlay,
        "n_objects": int(masks3d.max()),
        "cy3_idx": cy3_idx,
        "channel_names": ch_names
    }

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Folder with *.ome.tif / *.ome.tiff")
    ap.add_argument("--out_dir", default="cellpose_out", help="Output root folder")
    ap.add_argument("--mode", choices=["mip2d", "vol2p5d", "vol3d_cpu"], default="vol2p5d",
                    help="mip2d: MIP->2D seg; vol2p5d: 2D per-slice + 3D link (GPU OK); vol3d_cpu: true 3D on CPU")
    ap.add_argument("--cy3_idx", type=int, default=None, help="Force Cy3 channel index (overrides name hints)")
    ap.add_argument("--cy3_hint", nargs="*", default=["cy3", "mscarlet", "m-scarlet", "568"],
                    help="Name hints for Cy3 channel")
    ap.add_argument("--diameter", type=float, default=80, help="Cell diameter in pixels (None = estimate)")
    ap.add_argument("--cellprob", type=float, default=0.2)
    ap.add_argument("--flow", type=float, default=0.9)
    ap.add_argument("--cpu", action="store_true", help="Force CPU for mip2d/vol2p5d (GPU by default)")
    ap.add_argument("--bgp", type=float, default=None, help="Per-slice bg percentile for vol2p5d (None to disable)")
    ap.add_argument("--anisotropy", type=float, default=None,
                    help="Voxel z/y (or z/x) size ratio for vol3d_cpu (optional)")
    ap.add_argument("--model", default="cpsam",
                # choices=["cpsam","cyto2","cpcy","cpnc","cyto"],
                help="Cellpose pretrained model to use")
    ap.add_argument("--nuc_idx", type=int, default=None, help="Nuclei channel index for post-splitting")
    ap.add_argument("--nuc_hint", nargs="*", default=["cy5","h2b","dapi","hoechst","647","dna","nucleus","nuclei"],
                help="Name hints for nuclei channel (used if --nuc_idx not given)")
    ap.add_argument("--use_nuclei_post", action="store_true",
                help="Use nuclei-seeded watershed to split fused cells in vol2p5d")
    ap.add_argument("--nuc_min_area", type=int, default=50, help="Min nucleus area (pixels) for post-splitting")

    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Using model: {args.model}  "
      f"({'found' if Path(args.model).exists() else 'builtin name'})")
    
    # create subfolders based on mode
    out_prev = out_root / "previews"; out_prev.mkdir(parents=True, exist_ok=True)
    if args.mode == "mip2d":
        out_proj = out_root / "projections"; out_proj.mkdir(parents=True, exist_ok=True)
        out_mask2d = out_root / "masks_2d"; out_mask2d.mkdir(parents=True, exist_ok=True)
    else:
        out_mask3d = out_root / "masks_3d"; out_mask3d.mkdir(parents=True, exist_ok=True)

    paths = sorted(list(in_dir.glob("*.ome.tif")) + list(in_dir.glob("*.ome.tiff")))
    if not paths:
        print("No OME-TIFFs found. Supported suffixes: *.ome.tif, *.ome.tiff")
        return

    model_info_printed = False
    for p in paths:
        if args.mode == "mip2d":
            info = process_one_mip2d(
                str(p), str(out_proj), str(out_mask2d), str(out_prev),
                cy3_idx=args.cy3_idx,
                cy3_hints=args.cy3_hint,
                diameter=args.diameter,
                cellprob_threshold=args.cellprob,
                flow_threshold=args.flow,
                pretrained_model= args.model,
                gpu=not args.cpu
            )
            if not model_info_printed:
                print(f"[mip2d] Detected channels for {p.name}: {info['channel_names'] or '[none in OME XML]'}; using Cy3 idx = {info['cy3_idx']}")
                model_info_printed = True
            print(f"✓ {p.name} →")
            print(f"   {info['proj_tif']}")
            print(f"   {info['mask2d_tif']}")
            print(f"   {info['png_proj']}")
            print(f"   {info['png_mask']}")
            print(f"   {info['png_overlay']}")
        elif args.mode == "vol2p5d":
            info = process_one_vol2p5d(
                str(p), str(out_mask3d), str(out_prev),
                cy3_idx=args.cy3_idx,
                cy3_hints=args.cy3_hint,
                diameter=args.diameter,
                cellprob_threshold=args.cellprob,
                flow_threshold=args.flow,
                gpu=not args.cpu,
                pretrained_model=args.model,
                bg_percentile=args.bgp,
                nuc_idx=args.nuc_idx,
                nuc_hints=args.nuc_hint,
                use_nuclei_post=args.use_nuclei_post,
                nuc_min_area=args.nuc_min_area
            )

            if not model_info_printed:
                print(f"[vol2p5d] Detected channels for {p.name}: {info['channel_names'] or '[none in OME XML]'}; using Cy3 idx = {info['cy3_idx']}")
                model_info_printed = True
            print(f"✓ {p.name} →")
            print(f"   {info['mask3d_tif']}")
            print(f"   {info['png_triptych']}")
            # print(f"   {info['png_overlay']}")
            print(f"   objects: {info['n_objects']}")
        else:  # vol3d_cpu
            info = process_one_vol3d_cpu(
                str(p), str(out_mask3d), str(out_prev),
                cy3_idx=args.cy3_idx,
                cy3_hints=args.cy3_hint,
                diameter=args.diameter,
                cellprob_threshold=args.cellprob,
                flow_threshold=args.flow,
                gpu=False,
                pretrained_model= args.model,
                anisotropy=args.anisotropy
            )
            if not model_info_printed:
                print(f"[vol3d_cpu] Detected channels for {p.name}: {info['channel_names'] or '[none in OME XML]'}; using Cy3 idx = {info['cy3_idx']}")
                model_info_printed = True
            print(f"✓ {p.name} →")
            print(f"   {info['mask3d_tif']}")
            print(f"   {info['png_overlay']}")
            print(f"   objects: {info['n_objects']}")

if __name__ == "__main__":
    main()
