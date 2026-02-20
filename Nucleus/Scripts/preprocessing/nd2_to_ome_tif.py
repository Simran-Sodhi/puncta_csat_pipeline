#!/usr/bin/env python3
"""
nd2_to_ome_tif.py

Convert an ND2 microscopy file to per-scene OME-TIFF files,
extracting a single Z-plane (configurable via --z-index).

Optionally splits each channel into a separate folder with
single-channel TIFFs that are compatible with Cellpose GUI
for training, manual curation, and evaluation.

Uses the lightweight ``nd2`` package (no aicsimageio / lxml needed).

Usage
-----
    python nd2_to_ome_tif.py --input /path/to/file.nd2 --outdir /path/to/output --z-index 8
    python nd2_to_ome_tif.py --input /path/to/file.nd2 --outdir /path/to/output --split-channels

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


def convert_nd2(input_path, output_dir, z_index=8, split_channels=False):
    """
    Convert all positions/scenes in an ND2 file to individual OME-TIFF
    files for a single Z-plane.

    When *split_channels* is True, each channel is additionally saved as
    a single-channel 16-bit TIFF inside a ``<ChannelName>/`` subfolder.
    These single-channel images retain the original intensity values and
    can be opened directly in the Cellpose GUI for training, manual
    curation, and evaluation.

    Parameters
    ----------
    input_path : str or pathlib.Path
        Path to the .nd2 file.
    output_dir : str or pathlib.Path
        Directory for output OME-TIFF files.
    z_index : int
        Z-plane to extract (0-based).
    split_channels : bool
        If True, also save per-channel single-channel TIFFs in
        ``<output_dir>/<ChannelName>/`` folders.
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

    # ------------------------------------------------------------------
    # Extract optical metadata from ND2 for OME-TIFF propagation
    # ------------------------------------------------------------------
    px_x = px_y = px_z = None
    obj_na = None
    obj_mag = None
    obj_name = None
    immersion_ri = None
    per_ch_ex = []   # excitation wavelength per channel (nm)
    per_ch_em = []   # emission wavelength per channel (nm)
    pinhole_um = None

    if f.metadata and f.metadata.channels:
        vol = f.metadata.channels[0].volume
        if vol:
            cal = getattr(vol, "axesCalibration", None)
            if cal:
                px_x = cal[0] if len(cal) > 0 else None
                px_y = cal[1] if len(cal) > 1 else None
                px_z = cal[2] if len(cal) > 2 else None
            obj_na = getattr(vol, "objectiveNumericalAperture", None)
            obj_mag = getattr(vol, "objectiveMagnification", None)
            obj_name = getattr(vol, "objectiveName", None)
            immersion_ri = getattr(vol, "immersionRefractiveIndex", None)

        for ch_meta in f.metadata.channels:
            ch_obj = getattr(ch_meta, "channel", None)
            if ch_obj:
                ex = getattr(ch_obj, "excitationWavelengthNm", None)
                em = getattr(ch_obj, "emissionWavelengthNm", None)
                per_ch_ex.append(float(ex) if ex else None)
                per_ch_em.append(float(em) if em else None)
                if pinhole_um is None:
                    ph = getattr(ch_obj, "pinholeDiameterUm", None)
                    if ph:
                        pinhole_um = float(ph)
            else:
                per_ch_ex.append(None)
                per_ch_em.append(None)

    # Pad lists to match channel count
    while len(per_ch_ex) < n_channels:
        per_ch_ex.append(None)
        per_ch_em.append(None)

    # Print extracted metadata
    if obj_na:
        print(f"  Objective: NA={obj_na}"
              f"{f', mag={obj_mag}x' if obj_mag else ''}"
              f"{f', {obj_name}' if obj_name else ''}")
    if immersion_ri:
        print(f"  Immersion RI: {immersion_ri}")
    if px_x:
        print(f"  Pixel size: {px_x:.4f} x {px_y:.4f} um"
              f"{f', Z-step={px_z:.4f} um' if px_z else ''}")
    for ci, (ex, em) in enumerate(zip(per_ch_ex[:n_channels],
                                       per_ch_em[:n_channels])):
        if ex or em:
            print(f"  Channel {ci} ({ch_names[ci]}): "
                  f"ex={ex} nm, em={em} nm")

    # Create per-channel subdirectories if splitting
    ch_dirs = {}
    if split_channels:
        for ch_idx, ch_name in enumerate(ch_names[:n_channels]):
            # Sanitise channel name for folder (replace spaces, slashes)
            safe_name = ch_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
            ch_dir = out_dir / safe_name
            ch_dir.mkdir(parents=True, exist_ok=True)
            ch_dirs[ch_idx] = ch_dir
        print(f"  Splitting {n_channels} channels: {ch_names[:n_channels]}")

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

        # ---- Build OME metadata for multi-channel image ----
        ch_ome_list = []
        for ci in range(C):
            ch_entry = {"Name": ch_names[ci] if ci < len(ch_names) else f"Ch{ci}"}
            if ci < len(per_ch_ex) and per_ch_ex[ci]:
                ch_entry["ExcitationWavelength"] = per_ch_ex[ci]
                ch_entry["ExcitationWavelengthUnit"] = "nm"
            if ci < len(per_ch_em) and per_ch_em[ci]:
                ch_entry["EmissionWavelength"] = per_ch_em[ci]
                ch_entry["EmissionWavelengthUnit"] = "nm"
            if pinhole_um:
                ch_entry["PinholeSize"] = pinhole_um
                ch_entry["PinholeSizeUnit"] = "um"
            ch_ome_list.append(ch_entry)

        ome_meta = {
            "axes": "CYX",
            "Channel": ch_ome_list,
            "PhysicalSizeX": float(px_x) if px_x else None,
            "PhysicalSizeY": float(px_y) if px_y else None,
            "PhysicalSizeXUnit": "micrometer",
            "PhysicalSizeYUnit": "micrometer",
        }
        # Objective metadata (stored in OME Instrument/Objective)
        if obj_na:
            ome_meta["NominalMagnification"] = float(obj_mag) if obj_mag else None
            ome_meta["LensNA"] = float(obj_na)
        if immersion_ri:
            ome_meta["ImmersionRefractiveIndex"] = float(immersion_ri)
        if obj_name:
            ome_meta["ObjectiveName"] = str(obj_name)

        # Save multi-channel OME-TIFF (always)
        out_path = out_dir / f"{inp.stem}_{scene_name}_Z{z_index:03d}.ome.tif"
        tifffile.imwrite(
            str(out_path),
            plane,
            ome=True,
            imagej=False,
            photometric="minisblack",
            compression="deflate",
            bigtiff=True,
            metadata=ome_meta,
        )

        # Save per-channel single-channel OME-TIFFs with full metadata
        if split_channels:
            for ch_idx in range(C):
                if ch_idx not in ch_dirs:
                    continue
                ch_data = plane[ch_idx]  # (Y, X), original dtype/intensity

                # Build per-channel OME metadata
                ch_single_meta = {
                    "axes": "YX",
                    "PhysicalSizeX": float(px_x) if px_x else None,
                    "PhysicalSizeY": float(px_y) if px_y else None,
                    "PhysicalSizeXUnit": "micrometer",
                    "PhysicalSizeYUnit": "micrometer",
                    "Channel": [ch_ome_list[ch_idx]],
                }
                if obj_na:
                    ch_single_meta["LensNA"] = float(obj_na)
                    ch_single_meta["NominalMagnification"] = (
                        float(obj_mag) if obj_mag else None
                    )
                if immersion_ri:
                    ch_single_meta["ImmersionRefractiveIndex"] = float(immersion_ri)
                if obj_name:
                    ch_single_meta["ObjectiveName"] = str(obj_name)

                ch_out = (
                    ch_dirs[ch_idx]
                    / f"{inp.stem}_{scene_name}_Z{z_index:03d}.ome.tif"
                )
                tifffile.imwrite(
                    str(ch_out),
                    np.ascontiguousarray(ch_data),
                    ome=True,
                    imagej=False,
                    photometric="minisblack",
                    compression="deflate",
                    metadata=ch_single_meta,
                )

        exported += 1
        if (pos_idx + 1) % 10 == 0 or (pos_idx + 1) == n_positions:
            print(f"  Processed {pos_idx + 1}/{n_positions} positions")

    f.close()

    elapsed = time.perf_counter() - t0
    print(f"Exported {exported}/{n_positions} position(s) in {elapsed:.1f}s -> {out_dir}")

    if split_channels:
        for ch_idx, ch_dir in ch_dirs.items():
            n_files = len(list(ch_dir.glob("*.ome.tif")))
            print(f"  Channel '{ch_names[ch_idx]}' -> {ch_dir.name}/ ({n_files} OME-TIFFs)")

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
    parser.add_argument(
        "--split-channels", action="store_true",
        help="Also save each channel as single-channel TIFFs in per-channel folders.",
    )
    args = parser.parse_args()
    convert_nd2(args.input, args.outdir, args.z_index, split_channels=args.split_channels)


if __name__ == "__main__":
    main()
