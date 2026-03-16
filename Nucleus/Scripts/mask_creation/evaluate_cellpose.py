#!/usr/bin/env python3
"""
evaluate_cellpose.py

Unified Cellpose segmentation script with four modes:

  cell      – whole-cell segmentation on the DIC / bright-field channel (ch 0)
  nucleus   – nucleus segmentation on the mScarlet channel (ch 2)
  puncta    – puncta segmentation on the GFP channel (ch 1)
  cytoplasm – whole-cell on DIC, then subtract pre-existing nucleus masks

Channel layout (default):
    0 = DIC / bright-field
    1 = GFP  (puncta)
    2 = mScarlet (nucleus)

Usage examples
--------------
    # Whole-cell segmentation from DIC
    python evaluate_cellpose.py --mode cell --input /path/to/images --outdir masks --gpu

    # Nucleus segmentation from mScarlet
    python evaluate_cellpose.py --mode nucleus --input /path/to/images --outdir masks --gpu

    # Puncta segmentation from GFP
    python evaluate_cellpose.py --mode puncta --input /path/to/images --outdir masks --gpu

    # Cytoplasm: DIC cell seg then subtract nucleus masks
    python evaluate_cellpose.py --mode cytoplasm --input /path/to/images --outdir masks --gpu \\
        --nuc-mask-dir /path/to/nucleus_masks
"""

import sys
import argparse
from pathlib import Path

# Allow imports from parent directory (where segmentation_utils.py lives)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from segmentation_utils import (
    load_image_2d,
    auto_lut_clip,
    normalize_dic,
    ensure_2d,
    load_cellpose_model,
    run_cellpose_multipass,
    parse_diameters,
    postprocess_mask,
    save_mask,
    save_triptych,
    save_cytoplasm_triptych,
    collect_image_paths,
    compute_cytoplasm_mask,
)
import numpy as np
import tifffile as tiff
import re


# -------------------- presets -------------------- #
# Channel layout:  0 = DIC,  1 = GFP (puncta),  2 = mScarlet (nucleus)

PRESETS = {
    "cell": {
        "diameter": None,       # auto-estimate for whole cells on DIC
        "channel_index": 0,     # DIC / bright-field
        "z_index": 0,
        "min_size": 50000,
        "remove_edges": True,
        "use_dic_norm": True,
    },
    "nucleus": {
        "diameter": 200,
        "channel_index": 2,     # mScarlet
        "z_index": 0,
        "min_size": 10000,
        "remove_edges": True,
        "use_dic_norm": False,
    },
    "puncta": {
        "diameter": "auto-multi",  # auto-estimate + multi-scale passes
        "channel_index": 1,     # GFP
        "z_index": 0,
        "min_size": 0,
        "remove_edges": False,
        "use_dic_norm": False,
    },
    "cytoplasm": {
        "diameter": None,       # auto-estimate
        "channel_index": 0,     # DIC for whole-cell segmentation
        "z_index": 0,
        "min_size": 50000,
        "remove_edges": True,
        "use_dic_norm": True,
    },
}


# -------------------- helpers -------------------- #

def _find_matching_nuc_mask(nuc_mask_dir, image_stem):
    """
    Find the nucleus mask file that corresponds to *image_stem*.

    Tries exact stem match first, then falls back to a location-token
    match (e.g. '114_Z005').
    """
    nuc_dir = Path(nuc_mask_dir)
    # Exact stem match (image_stem may end with _cyto3_masks etc.)
    for p in nuc_dir.glob("*.tif"):
        if image_stem in p.stem or p.stem in image_stem:
            return p

    # Token-based fallback
    m = re.search(r"(\d+_Z\d+)", image_stem)
    if m:
        token = m.group(1)
        for p in nuc_dir.glob("*.tif"):
            if token in p.stem:
                return p

    return None


# -------------------- CLI -------------------- #

def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run Cellpose (cyto3) on OME-TIF / TIFF images. "
            "Use --mode to select a preset or override any parameter individually."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["cell", "nucleus", "puncta", "cytoplasm"],
        default=None,
        help="Load a preset (cell, nucleus, puncta, or cytoplasm). "
             "Individual flags override the preset.",
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to a single image or a directory of images.",
    )
    parser.add_argument(
        "--outdir", default="cellpose_masks",
        help="Directory to save mask TIFFs (default: cellpose_masks).",
    )
    parser.add_argument(
        "--gpu", action="store_true",
        help="Use GPU if available.",
    )
    parser.add_argument(
        "--diameter", type=str, default=None,
        help="Object diameter in pixels. '0' or 'auto' = auto-estimate. "
             "'auto-multi' = auto-estimate then run at 0.5x/1x/2x/4x scales "
             "(best for mixed-size objects). Comma-separated for explicit "
             "multi-pass: '20,100'.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Batch size for Cellpose (default: 1).",
    )
    parser.add_argument(
        "--channel-index", type=int, default=None,
        help="0-based channel index in the input image.",
    )
    parser.add_argument(
        "--z-index", type=int, default=None,
        help="Z-plane index to extract.",
    )
    parser.add_argument(
        "--lut-low", type=float, default=2.0,
        help="Low percentile for LUT normalization (default: 2.0).",
    )
    parser.add_argument(
        "--lut-high", type=float, default=99.8,
        help="High percentile for LUT normalization (default: 99.8).",
    )
    parser.add_argument(
        "--min-size", type=int, default=None,
        help="Minimum object size in pixels (objects below this are removed).",
    )
    parser.add_argument(
        "--remove-edges", action="store_true", default=None,
        help="Remove objects touching the image border.",
    )
    parser.add_argument(
        "--dic-norm", action="store_true", default=None,
        help="Use DIC/bright-field normalization (CLAHE) instead of "
             "percentile-based LUT normalization.",
    )

    # Cytoplasm-specific arguments
    cyto_group = parser.add_argument_group("Cytoplasm mode options")
    cyto_group.add_argument(
        "--nuc-mask-dir", default=None,
        help="Directory containing nucleus mask TIFFs (required for cytoplasm mode).",
    )
    cyto_group.add_argument(
        "--nuc-dilate-px", type=int, default=0,
        help="Pixels to dilate nucleus before subtracting from cell mask (default: 0).",
    )
    cyto_group.add_argument(
        "--min-nuc-pixels", type=int, default=10,
        help="Minimum nucleus pixels overlapping a cell to keep it (default: 10).",
    )
    cyto_group.add_argument(
        "--min-overlap-frac", type=float, default=0.005,
        help="Minimum fraction of cell area overlapping nucleus to keep (default: 0.005).",
    )
    return parser


def resolve_args(args):
    """Merge preset defaults with explicit CLI overrides."""
    preset = PRESETS.get(args.mode, {})
    args.diameter = args.diameter if args.diameter is not None else preset.get("diameter")
    args.channel_index = args.channel_index if args.channel_index is not None else preset.get("channel_index", 0)
    args.z_index = args.z_index if args.z_index is not None else preset.get("z_index", 0)
    args.min_size = args.min_size if args.min_size is not None else preset.get("min_size", 0)
    if args.remove_edges is None:
        args.remove_edges = preset.get("remove_edges", False)
    if args.dic_norm is None:
        args.dic_norm = preset.get("use_dic_norm", False)
    return args


def main():
    parser = build_parser()
    args = parser.parse_args()
    args = resolve_args(args)

    is_cyto = args.mode == "cytoplasm"
    if is_cyto and not args.nuc_mask_dir:
        parser.error("--nuc-mask-dir is required when using --mode cytoplasm")

    image_paths = collect_image_paths(args.input)
    if not image_paths:
        raise RuntimeError(f"No TIF/TIFF/OME-TIFF images found under {args.input}")

    mode_label = args.mode or "custom"
    norm_label = "DIC (CLAHE)" if args.dic_norm else "LUT"
    print(f"Found {len(image_paths)} image(s). Mode: {mode_label}")
    print(f"  diameter={args.diameter}, channel={args.channel_index}, "
          f"z={args.z_index}, min_size={args.min_size}, "
          f"remove_edges={args.remove_edges}, norm={norm_label}")
    if is_cyto:
        print(f"  nuc_mask_dir={args.nuc_mask_dir}, nuc_dilate_px={args.nuc_dilate_px}, "
              f"min_nuc_pixels={args.min_nuc_pixels}, min_overlap_frac={args.min_overlap_frac}")

    model = load_cellpose_model(gpu=args.gpu, model_type="cyto3")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    triptych_dir = outdir / "triptychs"
    triptych_dir.mkdir(parents=True, exist_ok=True)

    for i, img_path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] {img_path.name}")
        try:
            img2d = load_image_2d(
                img_path,
                channel_index=args.channel_index,
                z_index=args.z_index,
            )

            # Normalize: CLAHE for DIC, percentile LUT for fluorescence
            if args.dic_norm:
                img_norm = normalize_dic(img2d)
            else:
                img_norm = auto_lut_clip(
                    img2d,
                    low_percentile=args.lut_low,
                    high_percentile=args.lut_high,
                )

            diameters = parse_diameters(args.diameter)
            masks, _flows = run_cellpose_multipass(
                img_norm, model=model,
                diameters=diameters,
                batch_size=args.batch_size,
            )
            masks = postprocess_mask(
                masks,
                min_size=args.min_size,
                remove_edges=args.remove_edges,
            )

            stem = img_path.stem

            if is_cyto:
                # Find matching nucleus mask
                nuc_path = _find_matching_nuc_mask(args.nuc_mask_dir, stem)
                if nuc_path is None:
                    print(f"  [WARN] No nucleus mask found for {stem}, "
                          f"saving whole-cell mask only.")
                    save_mask(masks, outdir / f"{stem}_cell_masks.tif")
                    save_triptych(img_norm, masks,
                                  triptych_dir / f"{stem}_cell_triptych.png")
                    continue

                nuc_mask = ensure_2d(tiff.imread(nuc_path))
                if nuc_mask.shape != masks.shape:
                    print(f"  [WARN] Shape mismatch: cell {masks.shape} vs "
                          f"nucleus {nuc_mask.shape}, skipping subtraction.")
                    save_mask(masks, outdir / f"{stem}_cell_masks.tif")
                    continue

                cyto_mask, kept, orphans = compute_cytoplasm_mask(
                    masks, nuc_mask,
                    nuc_dilate_px=args.nuc_dilate_px,
                    min_nuc_pixels=args.min_nuc_pixels,
                    min_overlap_frac=args.min_overlap_frac,
                )
                print(f"  Kept {len(kept)} cells, removed {len(orphans)} orphans")

                save_mask(masks, outdir / f"{stem}_cell_masks.tif")
                save_mask(cyto_mask, outdir / f"{stem}_cyto_masks.tif")

                save_cytoplasm_triptych(
                    img_norm, masks, nuc_mask, cyto_mask,
                    triptych_dir / f"{stem}_cyto_triptych.png",
                )
                print(f"  -> cyto mask: {outdir / f'{stem}_cyto_masks.tif'}")
            else:
                # Standard mode (cell / nucleus / puncta / custom)
                mask_path = outdir / f"{stem}_cyto3_masks.tif"
                save_mask(masks, mask_path)
                print(f"  -> mask: {mask_path}")

                trip_path = triptych_dir / f"{stem}_triptych.png"
                save_triptych(img_norm, masks, trip_path)
                print(f"  -> triptych: {trip_path}")

        except Exception as e:
            print(f"  [ERROR] {e}")
            continue

    print("Done.")


if __name__ == "__main__":
    main()
