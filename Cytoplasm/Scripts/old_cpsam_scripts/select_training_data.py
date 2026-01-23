#!/usr/bin/env python3
import os, random
import numpy as np
import tifffile as tiff
import xml.etree.ElementTree as ET
from pathlib import Path
from scipy import ndimage as ndi

# --------- reuse helpers from your pipeline ----------
def read_ome(path):
    with tiff.TiffFile(path) as tf:
        arr = tf.asarray()
        axes = tf.series[0].axes
        meta_xml = tf.ome_metadata
    return arr, axes, meta_xml

def to_ZCYX(arr, axes):
    ax = {a: i for i, a in enumerate(axes)}
    if 'Z' not in ax:
        arr = np.expand_dims(arr, 0); axes = 'Z' + axes
    if 'C' not in ax:
        arr = np.expand_dims(arr, 0); axes = axes[:1] + 'C' + axes[1:]
    ax = {a: i for i, a in enumerate(axes)}
    order = [ax['Z'], ax['C'], ax['Y'], ax['X']]
    out = np.moveaxis(arr, order, (0,1,2,3))
    return out

def channel_names_from_ome(meta_xml):
    names = []
    if not meta_xml: return names
    root = ET.fromstring(meta_xml)
    ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
    for ch in root.findall('.//ome:Image/ome:Pixels/ome:Channel', ns):
        nm = ch.get('Name') or ch.get('ID') or ''
        names.append(nm)
    return names

def pick_channel_idx(ch_names, hints):
    lname = [str(n).lower() for n in ch_names]
    for h in hints:
        h = str(h).lower()
        for i, n in enumerate(lname):
            if h in n:
                return i
    return 0

def percentile_norm(img2d, p1=1, p99=99):
    lo, hi = np.percentile(img2d, (p1,p99))
    if hi <= lo: return np.zeros_like(img2d, dtype=np.uint8)
    x = np.clip((img2d-lo)/(hi-lo), 0, 1)
    return (x*255).astype(np.uint8)

def _lap_var(img):
    return ndi.variance(ndi.laplace(img.astype(np.float32)))

def _pick_focus_band(stack, half_width=4):
    scores = np.array([_lap_var(z) for z in stack])
    z0 = int(scores.argmax())
    zlo, zhi = max(0,z0-half_width), min(stack.shape[0], z0+half_width+1)
    return np.arange(zlo, zhi)

def _clip_and_smooth(stack, lo=1, hi=99.5, sigma=1.0):
    lo_v, hi_v = np.percentile(stack, (lo,hi))
    stack = np.clip(stack, lo_v, hi_v)
    out = np.empty_like(stack, dtype=np.uint8)
    for i in range(stack.shape[0]):
        z = ndi.gaussian_filter(stack[i].astype(np.float32), sigma=sigma)
        mn,mx = np.percentile(z,(1,99))
        if mx <= mn: out[i]=0
        else: out[i]=((np.clip(z,mn,mx)-mn)/(mx-mn)*255).astype(np.uint8)
    return out

# --------- main export ----------
def export_training_data(in_dir, out_dir, n_train=3, n_val=1,
                         cyto_hints=('cy3','mscarlet','568'),
                         nuc_hints=('cy5','h2b','dapi','dna')):
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    img_train = out_dir/'train'/'images'; lbl_train = out_dir/'train'/'labels'
    img_val = out_dir/'val'/'images'; lbl_val = out_dir/'val'/'labels'
    for d in [img_train,lbl_train,img_val,lbl_val]: d.mkdir(parents=True, exist_ok=True)

    files = sorted(list(in_dir.glob("*.ome.tif"))+list(in_dir.glob("*.ome.tiff")))
    if not files: 
        print("No OME-TIFFs found"); return
    chosen = random.sample(files, min(len(files), n_train+n_val))
    train_files = chosen[:n_train]; val_files = chosen[n_train:n_train+n_val]

    def process_files(files, img_dir):
        for f in files:
            arr,axes,meta = read_ome(f)
            ZCYX = to_ZCYX(arr,axes)
            ch_names = channel_names_from_ome(meta)
            cyto_idx = pick_channel_idx(ch_names, cyto_hints)
            nuc_idx  = pick_channel_idx(ch_names, nuc_hints)
            cyto = ZCYX[:,cyto_idx]; nuc = ZCYX[:,nuc_idx]

            keep_z = _pick_focus_band(cyto)
            cyto_u8 = _clip_and_smooth(cyto[keep_z])
            nuc_u8  = _clip_and_smooth(nuc[keep_z])

            for i,z in enumerate(keep_z):
                rgb = np.dstack([cyto_u8[i], nuc_u8[i], np.zeros_like(cyto_u8[i])])
                out_path = img_dir/f"{f.stem}_z{z:03d}.png"
                tiff.imwrite(out_path, rgb)
                print("Saved", out_path, "(for manual labeling)")
    
    process_files(train_files, img_train)
    process_files(val_files, img_val)

if __name__=="__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Folder with OME-TIFF stacks")
    ap.add_argument("--out_dir", default="training_data", help="Output dataset folder")
    ap.add_argument("--n_train", type=int, default=3)
    ap.add_argument("--n_val", type=int, default=1)
    args = ap.parse_args()
    export_training_data(args.in_dir, args.out_dir, args.n_train, args.n_val)
