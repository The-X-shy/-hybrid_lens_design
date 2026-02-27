#!/usr/bin/env python3
"""
Train HybridLens (Refractive + DOE) using Binary2 DOE
使用 Binary2 DOE 训练混合透镜

Usage:
    python scripts/train_hybridlens.py --config configs/hybrid.yaml
    python scripts/train_hybridlens.py --geolens results/geolens/final_lens.json
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
    set_nested,
    get_nested,
)
from src.hybridlens.trainer import HybridLensTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train HybridLens with Binary2 DOE")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/hybrid.yaml",
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
    parser.add_argument(
        "--wavelength",
        type=float,
        help="Single optimization wavelength in um (overrides config.optimization.wavelengths)",
    )
    parser.add_argument("--spp", type=int, help="Samples per pixel for PSF")
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
    if args.wavelength:
        set_nested(config, "optimization.wavelengths", [float(args.wavelength)])
        set_nested(config, "optimization.wavelength_weights", [1.0])
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
    print("HybridLens Training with Binary2 DOE")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print(f"GeoLens source: {config.lens.file_path}")
    print(f"DOE type: {config.doe.type}")
    print(f"DOE resolution: {config.doe.res}")
    print(
        f"Optimization wavelengths: "
        f"{get_nested(config, 'optimization.wavelengths', [0.55])}"
    )

    # Create result directory
    result_dir = create_result_dir(config)

    # Create trainer from GeoLens
    geolens_path = config.lens.file_path
    doe_config = {
        "type": config.doe.type,
        "res": config.doe.res,
        "fab_ps": config.doe.fab_ps,
        "is_square": config.doe.is_square,
        "param_model": getattr(config.doe, "param_model", "binary2"),
    }

    trainer = HybridLensTrainer.from_geolens(
        geolens_path=geolens_path,
        result_dir=str(result_dir),
        doe_config=doe_config,
        device=device,
    )

    # Setup sensor
    trainer.setup_sensor(
        match_aperture=config.sensor.match_aperture,
        sensor_res=tuple(config.sensor.res),
    )

    # Train
    loss_history = trainer.train(
        doe_lr=config.optimization.doe_lr,
        lens_lr=config.optimization.lens_lr,
        lr_decay=config.optimization.lr_decay,
        iterations=config.optimization.iterations,
        test_per_iter=config.optimization.test_per_iter,
        spp=config.optimization.spp,
        psf_size=config.optimization.psf_size,
        wavelengths=get_nested(config, "optimization.wavelengths", [0.55]),
        wavelength_weights=get_nested(config, "optimization.wavelength_weights", [1.0]),
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
