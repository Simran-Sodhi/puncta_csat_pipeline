#!/usr/bin/env python3
"""
nd2_to_ome_tif.py

Convert an ND2 microscopy file to per-scene OME-TIFF files,
extracting a single Z-plane (configurable via --z-index).

Uses the lightweight ``nd2`` package (no aicsimageio / lxml needed).

Usage
-----
    python nd2_to_ome_tif.py --input /path/to/file.nd2 --outdir /path/to/output --z-index 8

Install
-------
    pip install nd2 tifffile numpy
"""

import argparse
import pathlib
import time

import numpy as np
import tifffile
import nd2


def convert_nd2(input_path, output_dir, z_index=8):
    """
    Convert all positions/scenes in an ND2 file to individual OME-TIFF
    files for a single Z-plane.

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

    f = nd2.ND2File(inp)
    print(f"Opened {inp.name}")

    # Get metadata
    sizes = f.sizes  # dict like {'P': 96, 'Z': 11, 'C': 3, 'Y': 2044, 'X': 2048}
    print(f"  Dimensions: {sizes}")

    # Determine axis order and counts
    n_positions = sizes.get("P", 1)
    n_z = sizes.get("Z", 1)
    n_channels = sizes.get("C", 1)

    if z_index >= n_z:
        raise ValueError(
            f"z_index={z_index} but file only has {n_z} z-planes (0-{n_z - 1})"
        )

    # Get channel names
    ch_names = []
    if f.metadata and f.metadata.channels:
        ch_names = [ch.channel.name for ch in f.metadata.channels]
    if not ch_names:
        ch_names = [f"Ch{i}" for i in range(n_channels)]

    # Get pixel sizes
    px_x = px_y = None
    if f.metadata and f.metadata.channels:
        vol = f.metadata.channels[0].volume
        if vol:
            px_x = vol.axesCalibration[0] if vol.axesCalibration else None
            px_y = vol.axesCalibration[1] if len(vol.axesCalibration) > 1 else None

    # Read the full data array
    # nd2 returns data in the order of the axes in f.sizes
    data = f.asarray()
    axes_order = list(sizes.keys())
    print(f"  Array shape: {data.shape}, axes: {''.join(axes_order)}")

    t0 = time.perf_counter()
    exported = 0

    for pos_idx in range(n_positions):
        # Build the indexing tuple based on axis order
        slicing = []
        for ax in axes_order:
            if ax == "P":
                slicing.append(pos_idx)
            elif ax == "Z":
                slicing.append(z_index)
            elif ax == "C":
                slicing.append(slice(None))  # keep all channels
            elif ax in ("Y", "X"):
                slicing.append(slice(None))  # keep full spatial dims
            elif ax == "T":
                slicing.append(0)  # first time point
            else:
                slicing.append(0)

        plane = data[tuple(slicing)]  # should be (C, Y, X) or (Y, X)

        # Ensure shape is (C, Y, X)
        if plane.ndim == 2:
            plane = plane[np.newaxis, :, :]

        plane = np.ascontiguousarray(plane)
        C = plane.shape[0]

        # Scene/position name
        if n_positions > 1:
            # Try to get position name from experiment metadata
            scene_name = f"P{pos_idx:04d}"
            if hasattr(f, 'experiment') and f.experiment:
                try:
                    loops = f.experiment
                    for loop in loops:
                        if hasattr(loop, 'parameters') and hasattr(loop.parameters, 'points'):
                            pts = loop.parameters.points
                            if pos_idx < len(pts):
                                scene_name = pts[pos_idx].name or scene_name
                except Exception:
                    pass
        else:
            scene_name = "single"

        ch_meta = [{"Name": n} for n in ch_names[:C]]

        out_path = out_dir / f"{inp.stem}_{scene_name}_Z{z_index:03d}.ome.tif"
        tifffile.imwrite(
            str(out_path),
            plane,
            ome=True,
            imagej=False,
            photometric="minisblack",
            compression="deflate",
            bigtiff=True,
            metadata={
                "axes": "CYX",
                "Channel": ch_meta,
                "PhysicalSizeX": float(px_x) if px_x else None,
                "PhysicalSizeY": float(px_y) if px_y else None,
                "PhysicalSizeXUnit": "micrometer",
                "PhysicalSizeYUnit": "micrometer",
            },
        )
        exported += 1
        if (pos_idx + 1) % 10 == 0 or (pos_idx + 1) == n_positions:
            print(f"  Processed {pos_idx + 1}/{n_positions} positions")

    f.close()

    elapsed = time.perf_counter() - t0
    print(f"Exported {exported}/{n_positions} position(s) in {elapsed:.1f}s -> {out_dir}")

    # Quick verification on first exported file
    first_files = sorted(out_dir.glob(f"{inp.stem}_*_Z{z_index:03d}.ome.tif"))
    if first_files:
        with tifffile.TiffFile(str(first_files[0])) as tf:
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
