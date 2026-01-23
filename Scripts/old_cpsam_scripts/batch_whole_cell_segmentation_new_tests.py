#!/usr/bin/env python3
import os, argparse, numpy as np, tifffile as tiff, xml.etree.ElementTree as ET
from pathlib import Path
from cellpose import models, io
from scipy import ndimage as ndi
from skimage import filters, morphology, segmentation, measure
from skimage.measure import regionprops
from skimage.segmentation import relabel_sequential

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

# def channel_names_from_ome(meta_xml):
#     names = []
#     if not meta_xml:
#         return names
#     root = ET.fromstring(meta_xml)
#     ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
#     for ch in root.findall('.//ome:Image/ome:Pixels/ome:Channel', ns):
#         nm = ch.get('Name') or ch.get('ID') or ''
#         names.append(nm)
#     return names

def channel_names_from_ome(meta_xml):
    if not meta_xml: return []
    root = ET.fromstring(meta_xml)
    # find the OME ns dynamically
    ns = root.tag.split('}')[0].strip('{')
    NS = {'ome': ns}  # e.g., http://www.openmicroscopy.org/Schemas/OME/2016-06
    names = []
    for ch in root.findall('.//ome:Image/ome:Pixels/ome:Channel', NS):
        names.append(ch.get('Name') or ch.get('ID') or '')
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

def remove_small_instances(lbl, min_size):
    keep = np.zeros_like(lbl, dtype=bool)
    for r in regionprops(lbl):
        if r.area >= min_size:
            keep |= (lbl == r.label)
    lbl = lbl * keep
    lbl, _, _ = relabel_sequential(lbl)  # compact labels
    return lbl
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

def segment_nuclei_binary(nuc_slice_u8, min_area=50):
    """Simple per-slice nuclei mask: Otsu threshold + cleanup."""
    if nuc_slice_u8.max() == 0:
        return np.zeros_like(nuc_slice_u8, dtype=bool)
    thr = filters.threshold_otsu(nuc_slice_u8)
    m = nuc_slice_u8 > thr
    m = remove_small_instances(m.astype(np.int32), min_area).astype(np.uint16)
    m = morphology.remove_small_holes(m > 0, area_threshold=min_area)
    m = (m > 0) 
    # m = morphology.remove_small_holes(m, area_threshold=min_area)
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


# def _link_by_overlap(masks_list, min_pix_overlap=40, min_frac_overlap=0.05):
#     """
#     Link instance labels across consecutive slices by overlap, producing a 3D label volume.
#     - masks_list: list of 2D label arrays for kept Z slices (length K)
#     - min_pix_overlap: minimum pixel overlap to consider a match
#     - min_frac_overlap: min fraction of the smaller object that must overlap
#     Returns: vol_labels (K,H,W) with consistent instance IDs through Z
#     """
#     K, H, W = len(masks_list), masks_list[0].shape[0], masks_list[0].shape[1]
#     vol = np.zeros((K, H, W), dtype=np.int32)
#     next_id = 1

#     # map current slice label -> global id
#     prev_map = {}

#     for k in range(K):
#         lab = masks_list[k].astype(np.int32)
#         cur_map = {}

#         if k == 0:
#             # assign fresh IDs
#             for l in range(1, lab.max()+1):
#                 cur_map[l] = next_id; next_id += 1
#         else:
#             prev = masks_list[k-1].astype(np.int32)

#             # fast contingency table via hashing pairs
#             offset = (lab.max()+1)
#             pairs = prev.reshape(-1).astype(np.int64) * offset + lab.reshape(-1).astype(np.int64)
#             # ignore background (0)
#             valid = (prev.reshape(-1) > 0) & (lab.reshape(-1) > 0)
#             pairs = pairs[valid]
#             if pairs.size:
#                 uniq, counts = np.unique(pairs, return_counts=True)
#                 # decode pairs back to (p, c)
#                 p_lbl = (uniq // offset).astype(np.int32)
#                 c_lbl = (uniq %  offset).astype(np.int32)

#                 # compute areas per object
#                 prev_area = np.bincount(prev.reshape(-1), minlength=prev.max()+1)
#                 cur_area  = np.bincount(lab.reshape(-1),  minlength=lab.max()+1)

#                 # for each current label, find best previous match by overlap
#                 best_prev = {}  # cur -> (prev, overlap)
#                 for (p, c, ov) in zip(p_lbl, c_lbl, counts):
#                     if ov < min_pix_overlap: 
#                         continue
#                     small = min(prev_area[p], cur_area[c])
#                     if small == 0 or (ov / small) < min_frac_overlap:
#                         continue
#                     if (c not in best_prev) or (ov > best_prev[c][1]):
#                         best_prev[c] = (p, ov)

#                 # resolve: assign current to previous global id when unique,
#                 # otherwise allocate a fresh id (prevents merges of two prev → one cur)
#                 used_prev = set()
#                 for c in range(1, lab.max()+1):
#                     if c in best_prev:
#                         p = best_prev[c][0]
#                         if p not in used_prev:
#                             cur_map[c] = prev_map.get(p, None)
#                             used_prev.add(p)
#                     if c not in cur_map:
#                         cur_map[c] = next_id; next_id += 1
#             else:
#                 # no overlaps found; assign fresh IDs
#                 for l in range(1, lab.max()+1):
#                     cur_map[l] = next_id; next_id += 1

#         # write vol with global ids for slice k
#         out = np.zeros_like(lab, dtype=np.int32)
#         for l, gid in cur_map.items():
#             if gid is None:
#                 gid = next_id; next_id += 1
#             out[lab == l] = gid
#         vol[k] = out
#         prev_map = {l: cur_map[l] for l in cur_map}

#     return vol.astype(np.int32)

import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.measure import regionprops

def _props_from_labels(lbl):
    # returns {id: (area, cy, cx, bbox)} for nonzero labels
    out = {}
    for r in regionprops(lbl):
        out[r.label] = (r.area, r.centroid[0], r.centroid[1], r.bbox)  # (minr, minc, maxr, maxc)
    return out

def _iou_approx(lblA, lblB, idsA, idsB):
    """
    Fast IoU via pair contingency counts between two label images.
    Returns dict of ((a,b) -> iou, overlap_pixels).
    """
    maxA = lblA.max()
    maxB = lblB.max()
    base = maxB + 1
    a = lblA.reshape(-1).astype(np.int64)
    b = lblB.reshape(-1).astype(np.int64)
    valid = (a > 0) & (b > 0)
    if not valid.any():
        return {}
    pairs = a[valid] * base + b[valid]
    uniq, counts = np.unique(pairs, return_counts=True)
    res = {}
    # areas:
    areaA = np.bincount(a, minlength=maxA + 1)
    areaB = np.bincount(b, minlength=maxB + 1)
    for u, ov in zip(uniq, counts):
        aa = int(u // base)
        bb = int(u % base)
        if aa in idsA and bb in idsB:
            inter = ov
            union = areaA[aa] + areaB[bb] - inter
            res[(aa, bb)] = (inter / union if union > 0 else 0.0, inter)
    return res

def link_with_gaps(masks_list, max_gap=2,        # allow up to this many missing z-slices
                   min_iou=0.1, max_dist=40.0,   # hard gates
                   alpha=0.7,                    # weight for IoU vs distance in cost
                   sigma_xy=20.0,                # px; distance scale
                   z_spacing=1.0, yx_spacing=1.0 # for anisotropy-aware dist
                   ):
    """
    Link per-slice instance labels into a 3D label volume with:
    - gap closing up to 'max_gap'
    - Hungarian assignment using a blended IoU + centroid distance cost
    - split/merge handling with lineage
    Returns:
        vol (K,H,W) uint32   : global-track labels through z
        lineage (list[dict]) : [{id, parent_ids, start_z, end_z, slices}]
    """
    K, H, W = len(masks_list), masks_list[0].shape[0], masks_list[0].shape[1]
    vol = np.zeros((K, H, W), dtype=np.uint32)

    # Active tracks: id -> dict(last_z, last_label_id, last_centroid, last_slice_label_img_ref, missed)
    next_id = 1
    active = {}
    lineage = {}  # id -> dict(parent_ids, start_z, end_z, slices=list of (z, local_label_id))

    # store precomputed props
    props = [ _props_from_labels(m) for m in masks_list ]

    def end_track(tid, z):
        if tid in active:
            lineage[tid]["end_z"] = z
            active.pop(tid, None)

    def start_track(z, local_id, centroid, parents=None):
        nonlocal next_id
        tid = next_id; next_id += 1
        active[tid] = {
            "last_z": z,
            "last_local": local_id,
            "last_centroid": centroid,
            "missed": 0
        }
        lineage[tid] = {
            "id": tid,
            "parent_ids": [] if parents is None else parents,
            "start_z": z,
            "end_z": z,
            "slices": [(z, local_id)]
        }
        return tid

    # initialize at z=0
    z = 0
    lbl = masks_list[z]
    for lid, (area, cy, cx, _) in props[z].items():
        tid = start_track(z, lid, (cy, cx))
        vol[z][lbl == lid] = tid

    # helper: compute pair cost between previous set (track tips) and current labels
    def build_cost(prev_ids, cur_ids, prev_z, cur_z):
        cur_lbl = masks_list[cur_z]
        prev_lbl = masks_list[prev_z]
        # fast IoU where possible (only for prev_z == cur_z-1); for larger Δz IoU is less meaningful
        iou_map = _iou_approx(prev_lbl, cur_lbl, set([active[tid]["last_local"] for tid in prev_ids]), set(cur_ids)) \
                  if (cur_z - prev_z) == 1 else {}
        # assemble cost matrix
        P, C = len(prev_ids), len(cur_ids)
        cost = np.full((P, C), fill_value=1e3, dtype=np.float32)
        for i, tid in enumerate(prev_ids):
            pa, pcy, pcx, _ = props[prev_z][active[tid]["last_local"]]
            for j, cid in enumerate(cur_ids):
                ca, ccy, ccx, _ = props[cur_z][cid]
                # centroid distance (anisotropy-aware in z handled by look-ahead; here only XY)
                d = np.hypot((pcy - ccy) * (yx_spacing / yx_spacing),
                             (pcx - ccx) * (yx_spacing / yx_spacing))
                if d > max_dist:
                    continue
                iou = iou_map.get((active[tid]["last_local"], cid), (0.0, 0))[0]
                if (cur_z - prev_z) == 1 and iou < min_iou and d > max_dist * 0.5:
                    # if consecutive slice: require at least tiny IoU or be quite close
                    continue
                # blended score -> cost
                # score in [0,1+] where higher is better
                score = alpha * iou + (1 - alpha) * np.exp(-(d ** 2) / (2 * sigma_xy ** 2))
                cost[i, j] = 1.0 - score
        return cost

    # iterate z>0
    for z in range(1, K):
        cur_lbl = masks_list[z]
        cur_ids = list(props[z].keys())
        if not cur_ids:
            # nothing here; just age active tracks
            for tid in list(active.keys()):
                active[tid]["missed"] += 1
                lineage[tid]["end_z"] = z
                if active[tid]["missed"] > max_gap:
                    end_track(tid, z)
            continue

        # try to assign current labels to *some* previous slice within the lookback window
        assigned_cur = set()
        claimed_prev = set()  # tracks that already matched (to detect merges)
        matches = []          # (tid, cid)

        # look back Δz = 1..max_gap+1 (try closest first)
        for back in range(1, max_gap + 2):
            prev_z = z - back
            if prev_z < 0:
                break
            # candidate previous tracks that are still 'active' and last seen at prev_z
            prev_ids = [tid for tid, st in active.items() if st["last_z"] == prev_z and st["missed"] == back - 1]
            if not prev_ids:
                continue
            # cost between those prev tips and unassigned current cids
            cur_pool = [cid for cid in cur_ids if cid not in assigned_cur]
            if not cur_pool:
                break
            C = build_cost(prev_ids, cur_pool, prev_z, z)
            if np.all(C >= 1e2):
                continue
            r, c = linear_sum_assignment(C)
            for i, j in zip(r, c):
                if C[i, j] >= 1.0:  # too costly (no good match)
                    continue
                tid = prev_ids[i]
                cid = cur_pool[j]
                matches.append((tid, cid))
                assigned_cur.add(cid)
                claimed_prev.add(tid)

        # detect merges (multiple tids -> same cid) and splits (one tid -> multiple cids)
        from collections import defaultdict
        by_cur = defaultdict(list)
        by_prev = defaultdict(list)
        for tid, cid in matches:
            by_cur[cid].append(tid)
            by_prev[tid].append(cid)

        # APPLY matches to build volume & update tracks
        # 1) merges: multiple parents -> one current mask => end parents, start new child with parent_ids
        handled_tids = set()
        for cid, tids in by_cur.items():
            if len(tids) >= 2:
                # terminate all parents
                for tid in tids:
                    handled_tids.add(tid)
                    lineage[tid]["end_z"] = z
                    active.pop(tid, None)
                # create a new track with those parents
                ca, ccy, ccx, _ = props[z][cid]
                child = start_track(z, cid, (ccy, ccx), parents=tids)
                vol[z][cur_lbl == cid] = child
        # 2) simple one-to-one matches
        for tid, cids in by_prev.items():
            if len(cids) == 1 and tid not in handled_tids:
                cid = cids[0]
                # continue the same track
                ccy, ccx = props[z][cid][1], props[z][cid][2]
                active[tid]["last_z"] = z
                active[tid]["last_local"] = cid
                active[tid]["last_centroid"] = (ccy, ccx)
                active[tid]["missed"] = 0
                lineage[tid]["slices"].append((z, cid))
                lineage[tid]["end_z"] = z
                vol[z][cur_lbl == cid] = tid

        # 3) splits: one parent -> many children (remaining cids assigned from same tid)
        for tid, cids in by_prev.items():
            if len(cids) >= 2 and tid not in handled_tids:
                # parent ends, children start with parent_id = tid
                lineage[tid]["end_z"] = z
                active.pop(tid, None)
                for cid in cids:
                    ca, ccy, ccx, _ = props[z][cid]
                    child = start_track(z, cid, (ccy, ccx), parents=[tid])
                    vol[z][cur_lbl == cid] = child

        # 4) new objects (unassigned current)
        new_cids = [cid for cid in cur_ids if cid not in by_cur]
        for cid in new_cids:
            ca, ccy, ccx, _ = props[z][cid]
            tid = start_track(z, cid, (ccy, ccx))
            vol[z][cur_lbl == cid] = tid

        # 5) age unmatched active tracks
        for tid in list(active.keys()):
            if active[tid]["last_z"] < z:
                active[tid]["missed"] += 1
                lineage[tid]["end_z"] = z
                if active[tid]["missed"] > max_gap:
                    end_track(tid, z)

    # compact to uint32 IDs already stable; build lineage list
    lin = [dict(v) for v in lineage.values()]
    return vol.astype(np.uint32), lin

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

def _pick_focus_band(stack, half_width=4):
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

    # optional very light background subtraction
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
    img_list = [sub_u8[i] for i in range(sub_u8.shape[0])]
    masks_list, flows, styles = model.eval(
        img_list,
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
    if use_nuclei_post and sub_nuc_u8 is not None:
        for i in range(sub_u8.shape[0]):
            m = masks_list[i].astype(np.uint16)
            # quick clean: zero tiny specks (same threshold you use in eval)
            # m = morphology.remove_small_objects(m > 0, min_size=min_size).astype(np.uint16) * 1
            m = remove_small_instances(m.astype(np.int32), min_size).astype(np.uint16)
            m = m.astype(np.uint16)
            masks_list[i] = split_with_nuclei_on_slice(
                sub_u8[i], m, sub_nuc_u8[i]
            )

    # --- rebuild full-Z label volume (zeros outside focus band) ---
    Z, H, W = cy3_stack.shape
    masks_slices = np.zeros((Z, H, W), dtype=np.uint16)
    for i, z in enumerate(keep_z):
        masks_slices[z] = masks_list[i].astype(np.uint16)
    
    # --- 3D linking: replace ndi.label with overlap-tracking ---
    # sub_tracks = link_with_gaps([masks_slices[z] for z in keep_z])
                                
    #                             # ,
    #                             #   min_pix_overlap=40, min_frac_overlap=0.80)
    
    # # place back into full Z
    # track_vol = np.zeros_like(masks_slices, dtype=np.int32)
    # for i, z in enumerate(keep_z):
    #     track_vol[z] = sub_tracks[i]
    tracks_vol, lineage = link_with_gaps([masks_slices[z] for z in keep_z])
    # place back into full Z
    track_vol = np.zeros_like(masks_slices, dtype=np.int32)
    for i, z in enumerate(keep_z):
        track_vol[z] = tracks_vol[i]
    
    # compact to consecutive uint16 IDs
    uids = np.unique(track_vol)
    uids = uids[uids > 0]
    lut = np.zeros(int(uids.max())+1, dtype=np.uint16)
    lut[uids] = np.arange(1, len(uids)+1, dtype=np.uint16)
    mask3d = lut[track_vol]


    # # --- 3D linking (6-connectivity) ---
    # fg_3d = masks_slices > 0
    # structure = ndi.generate_binary_structure(3, 1)
    # mask3d, _ = ndi.label(fg_3d, structure=structure)
    # mask3d = mask3d.astype(np.uint16)

    # --- save preview + mask ---
    stem = Path(path).stem.replace('.ome','')
    mask3d_tif = os.path.join(out_mask3d_dir, f"{stem}_mask3d.tif")
    io.imsave(mask3d_tif, mask3d)

    mip_signal = np.max(cy3_stack, axis=0)
    mip_u8 = percentile_norm(mip_signal, 1, 99)
    mip_labels = np.max(mask3d, axis=0).astype(np.uint16)
    overlay = overlay_outlines_on_gray(mip_u8, mip_labels, edge_color=(255,0,0), alpha_fill=0.0)
    png_overlay = os.path.join(out_prev_dir, f"{stem}_overlay.png")
    tiff.imwrite(png_overlay, overlay)

    return {
        "mask3d_tif": mask3d_tif,
        "png_overlay": png_overlay,
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
    ap.add_argument("--bgp", type=float, default=2.0, help="Per-slice bg percentile for vol2p5d (None to disable)")
    ap.add_argument("--anisotropy", type=float, default=None,
                    help="Voxel z/y (or z/x) size ratio for vol3d_cpu (optional)")
    ap.add_argument("--model", default="cyto2",
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
            print(f"   {info['png_overlay']}")
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
