#!/usr/bin/env python3
"""
Train Metasurface HybridLens using Pixel2D DOE
使用 Pixel2D DOE 训练超表面混合透镜

Usage:
    python scripts/train_metalens.py --config configs/meta.yaml
    python scripts/train_metalens.py --geolens results/geolens/final_lens.json
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
    create_result_dir,
    get_device,
    set_seed,
    get_nested,
    set_nested,
)
from src.hybridlens.metasurface_trainer import MetasurfaceTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Metasurface HybridLens with Pixel2D DOE"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/meta.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--geolens",
        type=str,
        help="Path to optimized GeoLens JSON file (overrides config)",
    )

    # Override config options
    parser.add_argument("--iterations", type=int, help="Number of training iterations")
    parser.add_argument("--doe_lr", type=float, help="DOE learning rate")
    parser.add_argument("--spp", type=int, help="Samples per pixel for PSF")
    parser.add_argument(
        "--smoothness", type=float, help="Smoothness regularization weight"
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
    if args.geolens:
        config.lens.file_path = args.geolens
    if args.iterations:
        config.optimization.iterations = args.iterations
    if args.doe_lr:
        config.optimization.doe_lr = args.doe_lr
    if args.spp:
        config.optimization.spp = args.spp
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
    print("Metasurface HybridLens Training with Pixel2D DOE")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print(f"GeoLens source: {config.lens.file_path}")
    print(f"Metasurface type: {config.metasurface.type}")
    print(f"Metasurface resolution: {config.metasurface.res}")

    # Create result directory
    result_dir = create_result_dir(config)

    # Create trainer from GeoLens
    geolens_path = config.lens.file_path
    metasurface_config = {
        "type": "pixel2d",
        "res": config.metasurface.res,
        "fab_ps": config.metasurface.fab_ps,
        "is_square": config.metasurface.is_square,
    }

    trainer = MetasurfaceTrainer.from_geolens(
        geolens_path=geolens_path,
        result_dir=str(result_dir),
        metasurface_config=metasurface_config,
        device=device,
    )

    # Setup sensor
    trainer.setup_sensor(
        match_aperture=config.sensor.match_aperture,
        sensor_res=tuple(config.sensor.res),
    )

    # Get regularization weights
    smoothness_weight = getattr(config.regularization, "smoothness", 0.001)
    fabrication_weight = getattr(config.regularization, "fabrication_penalty", 0.01)

    if args.smoothness:
        smoothness_weight = args.smoothness

    # Train
    loss_history = trainer.train(
        doe_lr=config.optimization.doe_lr,
        lens_lr=config.optimization.lens_lr,
        lr_decay=config.optimization.lr_decay,
        iterations=config.optimization.iterations,
        test_per_iter=config.optimization.test_per_iter,
        spp=config.optimization.spp,
        psf_size=config.optimization.psf_size,
        smoothness_weight=smoothness_weight,
        fabrication_weight=fabrication_weight,
        early_stop_config=get_nested(config, "optimization.early_stop", None),
    )

    # Print final results
    print("\n" + "=" * 60)
    print("Training Results")
    print("=" * 60)
    print(f"Initial loss: {loss_history[0]:.6f}")
    print(f"Final loss: {loss_history[-1]:.6f}")
    print(f"Improvement: {(1 - loss_history[-1] / loss_history[0]) * 100:.1f}%")
    print(f"Output saved to: {result_dir}")


if __name__ == "__main__":
    main()
