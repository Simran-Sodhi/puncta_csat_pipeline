#!/usr/bin/env python3
"""
nd2_to_ome_tif.py

Convert an ND2 microscopy file to per-scene OME-TIFF files,
extracting a single Z-plane (configurable via --z-index).

Usage
-----
    python nd2_to_ome_tif.py --input /path/to/file.nd2 --outdir /path/to/output --z-index 8
"""

import argparse
import pathlib
import time

import numpy as np
import tifffile
from aicsimageio import AICSImage


def convert_nd2(input_path, output_dir, z_index=8):
    """
    Convert all scenes in an ND2 file to individual OME-TIFF files
    for a single Z-plane.

    Parameters
    ----------
    input_path : str or pathlib.Path
        Path to the .nd2 file.
    output_dir : str or pathlib.Path
        Directory for output OME-TIFF files.
    z_index : int
        Z-plane to extract (0-based).
    """
    inp = pathlib.Path(input_path)
    out_dir = pathlib.Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not inp.exists():
        raise FileNotFoundError(f"Input file not found: {inp}")

    img = AICSImage(inp)
    n_scenes = len(img.scenes)
    print(f"Opened {inp.name}: {n_scenes} scene(s)")

    t0 = time.perf_counter()
    exported = 0

    for i, scene in enumerate(img.scenes):
        img.set_scene(scene)

        data = img.get_image_data("CZYX")  # (C, Z, Y, X)
        data = np.ascontiguousarray(data)
        px = img.physical_pixel_sizes
        ch_names = list(map(str, img.channel_names))

        C, Z, Y, X = data.shape

        if z_index >= Z:
            print(f"  [WARN] Scene {i} ({scene}): only {Z} z-planes, "
                  f"skipping (requested z={z_index})")
            continue

        cyx = data[:, z_index, :, :]  # (C, Y, X)
        ch_meta = [{"Name": n} for n in ch_names[:cyx.shape[0]]]

        out_path = out_dir / f"{inp.stem}_{scene}_Z{z_index:03d}.ome.tif"
        tifffile.imwrite(
            str(out_path),
            cyx,
            ome=True,
            imagej=False,
            photometric="minisblack",
            compression="deflate",
            bigtiff=True,
            metadata={
                "axes": "CYX",
                "Channel": ch_meta,
                "PhysicalSizeX": float(px.X) if px.X else None,
                "PhysicalSizeY": float(px.Y) if px.Y else None,
                "PhysicalSizeXUnit": "micrometer",
                "PhysicalSizeYUnit": "micrometer",
            },
        )
        exported += 1
        if (i + 1) % 10 == 0 or (i + 1) == n_scenes:
            print(f"  Processed {i + 1}/{n_scenes} scenes")

    elapsed = time.perf_counter() - t0
    print(f"Exported {exported}/{n_scenes} scene(s) in {elapsed:.1f}s -> {out_dir}")

    # Quick verification on first exported file
    first = out_dir / f"{inp.stem}_{img.scenes[0]}_Z{z_index:03d}.ome.tif"
    if first.exists():
        with tifffile.TiffFile(str(first)) as tf:
            print(f"  Verification: axes={tf.series[0].axes}, shape={tf.series[0].shape}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ND2 file to per-scene OME-TIFF (single Z-plane)."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to the .nd2 file.",
    )
    parser.add_argument(
        "--outdir", required=True,
        help="Output directory for OME-TIFF files.",
    )
    parser.add_argument(
        "--z-index", type=int, default=8,
        help="Z-plane index to extract (default: 8).",
    )
    args = parser.parse_args()
    convert_nd2(args.input, args.outdir, args.z_index)


if __name__ == "__main__":
    main()
