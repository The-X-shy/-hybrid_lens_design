#!/usr/bin/env python3
"""
Train GeoLens (Refractive Lens) using Curriculum Learning
使用课程学习训练折射透镜

Usage:
    python scripts/train_geolens.py --config configs/geolens.yaml
    python scripts/train_geolens.py --config configs/geolens.yaml --foclen 8.0 --fov 80.0
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.utils.config import (
    load_config,
    get_device,
    set_seed,
    set_nested,
    resolve_geolens_stage_configs,
    get_nested,
)
from src.geolens.trainer import GeoLensTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train GeoLens with curriculum learning"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/geolens.yaml",
        help="Path to config file",
    )

    # Override config options
    parser.add_argument("--foclen", type=float, help="Target focal length (mm)")
    parser.add_argument("--fov", type=float, help="Target diagonal FOV (degrees)")
    parser.add_argument("--fnum", type=float, help="Target F-number")
    parser.add_argument("--iterations", type=int, help="Number of training iterations")
    parser.add_argument(
        "--finetune_iterations",
        type=int,
        help="Number of fine-tune iterations",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["scratch_two_stage", "existing_two_stage", "existing_finetune_only"],
        help="Training mode for refractive lens stage",
    )
    parser.add_argument(
        "--existing_lens",
        type=str,
        help="Existing lens path used in existing_* modes",
    )
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument(
        "--device", type=str, choices=["cuda", "cpu", "mps"], help="Device"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Override config with command line arguments
    if args.foclen:
        config.lens.design.foclen = args.foclen
    if args.fov:
        config.lens.design.fov = args.fov
    if args.fnum:
        config.lens.design.fnum = args.fnum
    if args.iterations:
        if get_nested(config, "optimization.curriculum") is not None:
            set_nested(config, "optimization.curriculum.iterations", args.iterations)
        if get_nested(config, "curriculum") is not None:
            set_nested(config, "curriculum.iterations", args.iterations)
    if args.finetune_iterations:
        if get_nested(config, "optimization.finetune") is not None:
            set_nested(
                config,
                "optimization.finetune.iterations",
                args.finetune_iterations,
            )
        if get_nested(config, "finetune") is not None:
            set_nested(config, "finetune.iterations", args.finetune_iterations)
    if args.mode:
        set_nested(config, "training.mode", args.mode)
    if args.existing_lens:
        set_nested(config, "training.existing_lens_path", args.existing_lens)
    if args.output_dir:
        set_nested(config, "output.dir", args.output_dir)
    if args.seed:
        config.seed = args.seed

    # Set seed
    seed = getattr(config, "seed", 42)
    set_seed(seed)

    # Set device
    device = get_device(config) if not args.device else torch.device(args.device)

    print("=" * 60)
    print("GeoLens Training with Curriculum Learning")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print(f"Target focal length: {config.lens.design.foclen} mm")
    print(f"Target FOV: {config.lens.design.fov} deg")
    print(f"Target F-number: {config.lens.design.fnum}")
    print(f"Training mode: {get_nested(config, 'training.mode', 'scratch_two_stage')}")
    print(
        f"Existing lens path: "
        f"{get_nested(config, 'training.existing_lens_path', 'N/A')}"
    )

    # Create trainer
    trainer = GeoLensTrainer.from_config(config)

    # Training parameters (supports both optimization.* and legacy top-level fields)
    curriculum_config, finetune_config = resolve_geolens_stage_configs(config)
    early_stop_config = get_nested(config, "optimization.early_stop", None)

    # Train
    curriculum_history, finetune_history = trainer.train(
        curriculum_config=curriculum_config,
        finetune_config=finetune_config,
        early_stop_config=early_stop_config,
    )

    # Print final results
    lens = trainer.get_lens()
    print("\n" + "=" * 60)
    print("Training Results")
    print("=" * 60)
    print(f"Final focal length: {lens.foclen:.2f} mm")
    print(f"Final F-number: {lens.fnum:.2f}")
    if finetune_history:
        final_rms = finetune_history[-1]
    elif curriculum_history:
        final_rms = curriculum_history[-1]
    else:
        final_rms = float("nan")
    print(f"Final RMS: {final_rms:.6f} mm")
    print(f"Output saved to: {trainer.result_dir}")


if __name__ == "__main__":
    main()
