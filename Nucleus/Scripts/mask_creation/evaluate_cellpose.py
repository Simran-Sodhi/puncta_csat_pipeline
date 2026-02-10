#!/usr/bin/env python3
"""
evaluate_cellpose.py

Unified Cellpose segmentation script that replaces both evaluate_nucleus.py
and evaluate_puncta.py.  Supports configurable presets (--mode nucleus | puncta)
or fully custom parameters via CLI flags.

Usage examples
--------------
    # Nucleus segmentation with sensible defaults
    python evaluate_cellpose.py --mode nucleus --input /path/to/images --outdir masks --gpu

    # Puncta segmentation with sensible defaults
    python evaluate_cellpose.py --mode puncta --input /path/to/images --outdir masks --gpu

    # Fully custom run
    python evaluate_cellpose.py --input /path/to/images --outdir masks --gpu \\
        --diameter 150 --channel-index 0 --z-index 3 --min-size 5000
"""

import sys
import argparse
from pathlib import Path

# Allow imports from parent directory (where segmentation_utils.py lives)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from segmentation_utils import (
    load_image_2d,
    auto_lut_clip,
    run_cellpose,
    postprocess_mask,
    save_mask,
    save_triptych,
    collect_image_paths,
)
from cellpose import models


# -------------------- presets -------------------- #

PRESETS = {
    "nucleus": {
        "diameter": 200,
        "channel_index": 2,
        "z_index": 5,
        "min_size": 10000,
        "remove_edges": True,
    },
    "puncta": {
        "diameter": 20,
        "channel_index": 1,
        "z_index": 8,
        "min_size": 0,
        "remove_edges": False,
    },
}


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run Cellpose (cyto3) on OME-TIF / TIFF images. "
            "Use --mode to select a preset or override any parameter individually."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["nucleus", "puncta"],
        default=None,
        help="Load a preset (nucleus or puncta). Individual flags override the preset.",
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
        "--diameter", type=float, default=None,
        help="Approximate object diameter in pixels.",
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
    return parser


def resolve_args(args):
    """Merge preset defaults with explicit CLI overrides."""
    preset = PRESETS.get(args.mode, {})
    # For each tuneable parameter, use CLI value if given, else preset, else fallback
    args.diameter = args.diameter if args.diameter is not None else preset.get("diameter")
    args.channel_index = args.channel_index if args.channel_index is not None else preset.get("channel_index", 0)
    args.z_index = args.z_index if args.z_index is not None else preset.get("z_index", 0)
    args.min_size = args.min_size if args.min_size is not None else preset.get("min_size", 0)
    if args.remove_edges is None:
        args.remove_edges = preset.get("remove_edges", False)
    return args


def main():
    parser = build_parser()
    args = parser.parse_args()
    args = resolve_args(args)

    image_paths = collect_image_paths(args.input)
    if not image_paths:
        raise RuntimeError(f"No TIF/TIFF/OME-TIFF images found under {args.input}")

    mode_label = args.mode or "custom"
    print(f"Found {len(image_paths)} image(s). Mode: {mode_label}")
    print(f"  diameter={args.diameter}, channel={args.channel_index}, "
          f"z={args.z_index}, min_size={args.min_size}, remove_edges={args.remove_edges}")

    model = models.Cellpose(gpu=args.gpu, model_type="cyto3")

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
            img_norm = auto_lut_clip(
                img2d,
                low_percentile=args.lut_low,
                high_percentile=args.lut_high,
            )
            masks = run_cellpose(
                img_norm, model=model,
                diameter=args.diameter,
                batch_size=args.batch_size,
            )
            masks = postprocess_mask(
                masks,
                min_size=args.min_size,
                remove_edges=args.remove_edges,
            )
        except Exception as e:
            print(f"  [ERROR] {e}")
            continue

        stem = img_path.stem
        mask_path = outdir / f"{stem}_cyto3_masks.tif"
        save_mask(masks, mask_path)
        print(f"  -> mask: {mask_path}")

        trip_path = triptych_dir / f"{stem}_triptych.png"
        save_triptych(img_norm, masks, trip_path)
        print(f"  -> triptych: {trip_path}")

    print("Done.")


if __name__ == "__main__":
    main()
