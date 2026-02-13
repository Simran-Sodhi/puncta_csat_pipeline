#!/usr/bin/env python3
"""
Main orchestration pipeline for Cellpose segmentation training.

Runs the full BiaPy-inspired workflow:
  1. Load YAML configuration
  2. Prepare and augment data
  3. Train Cellpose model (fine-tuning cpsam)
  4. Evaluate on test set
  5. Generate reports

Usage:
    python pipeline.py --task dic
    python pipeline.py --task fluor
    python pipeline.py --task both
    python pipeline.py --config path/to/config.yaml
    python pipeline.py --task dic --inference-only --model models/cellpose_dic_wholecell
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import yaml

from train_cellpose import (
    load_config,
    train_cellpose_model,
    train_dic_wholecell,
    train_fluor_nucleus,
)
from evaluate import evaluate_model

logger = logging.getLogger(__name__)

DEFAULT_CONFIGS = {
    "dic": "configs/dic_wholecell.yaml",
    "fluor": "configs/fluor_nucleus.yaml",
}


def validate_data_dirs(config: dict) -> bool:
    """Check that required data directories exist and contain images."""
    data_cfg = config["DATA"]
    required = [
        ("Training images", data_cfg["TRAIN"]["PATH"]),
        ("Training masks", data_cfg["TRAIN"]["MASK_PATH"]),
    ]

    all_ok = True
    for label, path in required:
        if not os.path.isdir(path):
            logger.error(f"{label} directory not found: {path}")
            all_ok = False
        elif not os.listdir(path):
            logger.warning(f"{label} directory is empty: {path}")

    test_path = data_cfg["TEST"]["PATH"]
    if os.path.isdir(test_path) and os.listdir(test_path):
        logger.info(f"Test data found at: {test_path}")
    else:
        logger.warning(f"No test data at {test_path} -- evaluation will be skipped.")

    return all_ok


def run_pipeline(
    config_path: str,
    inference_only: bool = False,
    model_path: str | None = None,
) -> dict:
    config = load_config(config_path)
    task_desc = config.get("PROBLEM", {}).get("DESCRIPTION", config_path)
    paths_cfg = config["PATHS"]

    logger.info("=" * 70)
    logger.info(f"Pipeline: {task_desc}")
    logger.info("=" * 70)

    if not inference_only:
        if not validate_data_dirs(config):
            logger.error(
                "Data directories are missing. Please populate the data/ "
                "directory following this structure:\n"
                "  data/train/<modality>_raw/   -- training images (*_img.tif)\n"
                "  data/train/<modality>_labels/ -- training masks (*_masks.tif)\n"
                "  data/test/<modality>_raw/    -- test images\n"
                "  data/test/<modality>_labels/  -- test masks"
            )
            return {"model_path": None, "metrics": {}}

    result = {"model_path": model_path, "metrics": {}}

    if not inference_only:
        start_time = time.time()
        result["model_path"] = train_cellpose_model(config)
        elapsed = time.time() - start_time
        logger.info(f"Training finished in {elapsed:.1f}s")

    inf_cfg = config.get("INFERENCE", {})
    if inf_cfg.get("ENABLE", True) and result["model_path"]:
        logger.info("Running evaluation on test set...")
        result["metrics"] = evaluate_model(config, result["model_path"])
    else:
        logger.info("Inference disabled or no model available, skipping evaluation.")

    logger.info("Pipeline complete.")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Cellpose Segmentation Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py --task dic          Train DIC whole-cell model
  python pipeline.py --task fluor        Train fluorescence nucleus model
  python pipeline.py --task both         Train both models
  python pipeline.py --config my.yaml    Train with custom config
  python pipeline.py --task dic --inference-only --model models/my_model
        """,
    )
    parser.add_argument("--task", choices=["dic", "fluor", "both"], default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--inference-only", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("pipeline.log"),
        ],
    )

    if args.config:
        configs = [args.config]
    elif args.task == "both":
        configs = [DEFAULT_CONFIGS["dic"], DEFAULT_CONFIGS["fluor"]]
    elif args.task:
        configs = [DEFAULT_CONFIGS[args.task]]
    else:
        parser.error("Specify --task or --config")

    all_results = {}
    for cfg_path in configs:
        logger.info(f"\n{'#' * 70}")
        logger.info(f"# Running config: {cfg_path}")
        logger.info(f"{'#' * 70}\n")
        result = run_pipeline(
            config_path=cfg_path,
            inference_only=args.inference_only,
            model_path=args.model,
        )
        all_results[cfg_path] = result

    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 70)
    for cfg, res in all_results.items():
        logger.info(f"\n  Config: {cfg}")
        logger.info(f"  Model:  {res['model_path']}")
        m = res.get("metrics", {})
        if m:
            logger.info(f"  mAP@0.5:  {m.get('mean_ap_50', 'N/A')}")
            logger.info(f"  mAP@0.75: {m.get('mean_ap_75', 'N/A')}")
            logger.info(f"  mIoU:     {m.get('mean_iou', 'N/A')}")
            logger.info(f"  mDice:    {m.get('mean_dice', 'N/A')}")
        else:
            logger.info("  Metrics: (no test data or evaluation skipped)")


if __name__ == "__main__":
    main()
