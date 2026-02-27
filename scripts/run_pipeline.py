#!/usr/bin/env python3
"""
Full Pipeline Script - Run complete hybrid lens design workflow
完整流程脚本 - 运行完整的混合透镜设计工作流

This script runs the complete pipeline:
1. GeoLens training (curriculum learning + fine-tuning)
2. HybridLens training (add DOE and joint optimization)
3. [Optional] Metasurface training (Pixel2D DOE)
4. [Optional] E2E training (joint HybridLens + Neural Network)
5. Evaluation and analysis

Usage:
    python scripts/run_pipeline.py --config configs/default.yaml
    python scripts/run_pipeline.py --stages geolens,hybridlens,eval
    python scripts/run_pipeline.py --full  # Run all stages
"""

import argparse
import sys
import time
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Run full hybrid lens design pipeline")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to default config file",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="geolens,hybridlens,eval",
        help="Comma-separated list of stages to run: geolens,hybridlens,metalens,e2e,eval",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run all stages including metasurface and E2E",
    )

    # Config overrides
    parser.add_argument("--output_dir", type=str, help="Base output directory")
    parser.add_argument("--foclen", type=float, help="Target focal length")
    parser.add_argument("--fov", type=float, help="Target FOV")
    parser.add_argument("--fnum", type=float, help="Target F-number")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device", type=str, choices=["cuda", "cpu", "mps"], help="Device"
    )

    # Stage-specific configs
    parser.add_argument("--geolens_config", type=str, default="configs/geolens.yaml")
    parser.add_argument("--hybrid_config", type=str, default="configs/hybrid.yaml")
    parser.add_argument("--meta_config", type=str, default="configs/meta.yaml")
    parser.add_argument("--e2e_config", type=str, default="configs/e2e.yaml")

    # Skip options
    parser.add_argument(
        "--skip_geolens", action="store_true", help="Skip GeoLens training"
    )
    parser.add_argument(
        "--geolens_path", type=str, help="Path to existing GeoLens (skip training)"
    )
    parser.add_argument(
        "--hybridlens_path", type=str, help="Path to existing HybridLens"
    )

    return parser.parse_args()


def run_geolens_stage(args, pipeline_dir):
    """Run GeoLens training stage."""
    from src.utils.config import (
        load_config,
        set_seed,
        set_nested,
        get_nested,
        resolve_geolens_stage_configs,
    )
    from src.geolens.trainer import GeoLensTrainer

    print("\n" + "=" * 60)
    print("STAGE 1: GeoLens Training (Curriculum Learning)")
    print("=" * 60)

    stage_start = time.time()
    result_dir = pipeline_dir / "01_geolens"
    result_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config = load_config(args.geolens_config)

    # Override with pipeline args
    if args.foclen:
        config.lens.design.foclen = args.foclen
    if args.fov:
        config.lens.design.fov = args.fov
    if args.fnum:
        config.lens.design.fnum = args.fnum
    set_nested(config, "output.dir", str(result_dir))
    if args.geolens_path:
        set_nested(config, "training.existing_lens_path", args.geolens_path)
        if get_nested(config, "training.mode", "scratch_two_stage") == "scratch_two_stage":
            set_nested(config, "training.mode", "existing_two_stage")
    if get_nested(config, "training.mode") is None:
        set_nested(config, "training.mode", "scratch_two_stage")

    set_seed(args.seed)

    # Create trainer
    trainer = GeoLensTrainer.from_config(config)

    # Training configs (compat with both optimization.* and legacy top-level)
    curriculum_config, finetune_config = resolve_geolens_stage_configs(config)

    # Train
    trainer.train(
        curriculum_config=curriculum_config,
        finetune_config=finetune_config,
        early_stop_config=get_nested(config, "optimization.early_stop", None),
    )

    # Save final lens path
    final_lens_path = result_dir / "final_lens.json"

    elapsed = time.time() - stage_start
    print(f"\nGeoLens stage completed in {elapsed / 60:.1f} minutes")
    print(f"Output: {final_lens_path}")

    return str(final_lens_path)


def run_hybridlens_stage(args, pipeline_dir, geolens_path):
    """Run HybridLens training stage."""
    from src.utils.config import load_config, set_seed, get_device, get_nested
    from src.hybridlens.trainer import HybridLensTrainer

    print("\n" + "=" * 60)
    print("STAGE 2: HybridLens Training (Binary2 DOE)")
    print("=" * 60)

    stage_start = time.time()
    result_dir = pipeline_dir / "02_hybridlens"
    result_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config = load_config(args.hybrid_config)

    set_seed(args.seed)
    device = get_device(config) if not args.device else torch.device(args.device)

    # DOE config
    doe_config = {
        "type": config.doe.type,
        "res": config.doe.res,
        "fab_ps": config.doe.fab_ps,
        "is_square": config.doe.is_square,
    }

    # Create trainer
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
    trainer.train(
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

    final_lens_path = result_dir / "hybridlens_final.json"

    elapsed = time.time() - stage_start
    print(f"\nHybridLens stage completed in {elapsed / 60:.1f} minutes")
    print(f"Output: {final_lens_path}")

    return str(final_lens_path)


def run_metalens_stage(args, pipeline_dir, geolens_path):
    """Run Metasurface (Pixel2D) training stage."""
    from src.utils.config import load_config, set_seed, get_device, get_nested
    from src.hybridlens.metasurface_trainer import MetasurfaceTrainer

    print("\n" + "=" * 60)
    print("STAGE 3: Metasurface Training (Pixel2D DOE)")
    print("=" * 60)

    stage_start = time.time()
    result_dir = pipeline_dir / "03_metalens"
    result_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config = load_config(args.meta_config)

    set_seed(args.seed)
    device = get_device(config) if not args.device else torch.device(args.device)

    # Metasurface config
    meta_config = {
        "type": "pixel2d",
        "res": config.metasurface.res,
        "fab_ps": config.metasurface.fab_ps,
        "is_square": config.metasurface.is_square,
    }

    # Create trainer
    trainer = MetasurfaceTrainer.from_geolens(
        geolens_path=geolens_path,
        result_dir=str(result_dir),
        metasurface_config=meta_config,
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

    # Train
    trainer.train(
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

    final_lens_path = result_dir / "metasurface_iterfinal.json"

    elapsed = time.time() - stage_start
    print(f"\nMetasurface stage completed in {elapsed / 60:.1f} minutes")
    print(f"Output: {final_lens_path}")

    return str(final_lens_path)


def run_e2e_stage(args, pipeline_dir, hybridlens_path):
    """Run E2E training stage."""
    from src.utils.config import load_config, set_seed, get_device, get_nested
    from src.e2e.trainer import E2ETrainer

    print("\n" + "=" * 60)
    print("STAGE 4: End-to-End Training (HybridLens + UNet)")
    print("=" * 60)

    stage_start = time.time()
    result_dir = pipeline_dir / "04_e2e"
    result_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config = load_config(args.e2e_config)

    set_seed(args.seed)
    device = get_device(config) if not args.device else torch.device(args.device)

    # Network config
    network_config = {
        "type": config.network.type,
        "in_channels": config.network.in_channels,
        "out_channels": config.network.out_channels,
    }
    if hasattr(config.network, "mha"):
        network_config["mha"] = (
            config.network.mha.to_dict()
            if hasattr(config.network.mha, "to_dict")
            else config.network.mha
        )

    # Create trainer
    trainer = E2ETrainer.from_hybridlens(
        hybridlens_path=hybridlens_path,
        result_dir=str(result_dir),
        network_config=network_config,
        freeze_doe=get_nested(config, "optimization.freeze_doe", True),
        wavelength=get_nested(config, "optimization.wavelength", 0.55),
        device=device,
    )

    # Loss weights
    loss_weights = {
        "mse": config.optimization.loss_weights.mse,
        "lpips": config.optimization.loss_weights.lpips,
        "psf": config.optimization.loss_weights.psf,
    }

    train_path = config.dataset.train_path
    val_path = getattr(config.dataset, "val_path", None)
    if not os.path.exists(train_path):
        if "BSDS300" in train_path:
            print(f"Dataset path not found: {train_path}")
            print("Attempting to download BSDS300...")
            from deeplens.network.dataset import download_bsd300

            output_image_dir = download_bsd300("./datasets")
            candidate_train = os.path.join(output_image_dir, "train", "images")
            candidate_test = os.path.join(output_image_dir, "test", "images")
            if not os.path.exists(candidate_train):
                candidate_train = os.path.join(output_image_dir, "train")
            if not os.path.exists(candidate_test):
                candidate_test = os.path.join(output_image_dir, "test")
            if os.path.exists(candidate_train):
                train_path = candidate_train
            if (val_path is None or not os.path.exists(val_path)) and os.path.exists(candidate_test):
                val_path = candidate_test
            print(f"BSDS300 ready. train={train_path}, val={val_path}")
        else:
            raise FileNotFoundError(f"Training dataset path not found: {train_path}")

    # Train
    trainer.train(
        train_dataset_path=train_path,
        val_dataset_path=val_path,
        doe_lr=config.optimization.doe_lr,
        lens_lr=config.optimization.lens_lr,
        network_lr=config.optimization.network_lr,
        lr_decay=config.optimization.lr_decay,
        epochs=config.optimization.epochs,
        batch_size=config.optimization.batch_size,
        test_per_epoch=config.optimization.test_per_epoch,
        spp=getattr(config.optimization, "spp", 100000),
        psf_size=getattr(config.optimization, "psf_size", 101),
        image_size=tuple(config.dataset.image_size),
        loss_weights=loss_weights,
        freeze_doe=get_nested(config, "optimization.freeze_doe", True),
        wavelength=get_nested(config, "optimization.wavelength", 0.55),
        early_stop_config=get_nested(config, "optimization.early_stop", None),
    )

    elapsed = time.time() - stage_start
    print(f"\nE2E stage completed in {elapsed / 60:.1f} minutes")

    return str(result_dir / "hybridlens_epochfinal.json")


def run_evaluation_stage(args, pipeline_dir, lens_paths):
    """Run evaluation stage for all generated lenses."""
    from src.hybridlens.evaluator import HybridLensEvaluator
    from src.geolens.evaluator import GeoLensEvaluator

    print("\n" + "=" * 60)
    print("STAGE 5: Evaluation and Analysis")
    print("=" * 60)

    result_dir = pipeline_dir / "05_evaluation"
    result_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate GeoLens
    if "geolens" in lens_paths and Path(lens_paths["geolens"]).exists():
        print("\nEvaluating GeoLens...")
        geolens_eval_dir = result_dir / "geolens"
        evaluator = GeoLensEvaluator.from_file(
            lens_paths["geolens"],
            str(geolens_eval_dir),
        )
        evaluator.full_analysis()

    # Evaluate HybridLens
    if "hybridlens" in lens_paths and Path(lens_paths["hybridlens"]).exists():
        print("\nEvaluating HybridLens...")
        hybrid_eval_dir = result_dir / "hybridlens"
        evaluator = HybridLensEvaluator.from_file(
            lens_paths["hybridlens"],
            str(hybrid_eval_dir),
        )
        evaluator.full_analysis()

    # Evaluate Metasurface lens
    if "metalens" in lens_paths and Path(lens_paths["metalens"]).exists():
        print("\nEvaluating Metasurface lens...")
        meta_eval_dir = result_dir / "metalens"
        evaluator = HybridLensEvaluator.from_file(
            lens_paths["metalens"],
            str(meta_eval_dir),
        )
        evaluator.full_analysis()

    print(f"\nEvaluation results saved to: {result_dir}")


def main():
    args = parse_args()

    # Determine which stages to run
    if args.full:
        stages = ["geolens", "hybridlens", "metalens", "e2e", "eval"]
    else:
        stages = [s.strip().lower() for s in args.stages.split(",")]

    # Create pipeline directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        pipeline_dir = Path(args.output_dir)
    else:
        pipeline_dir = Path(f"results/pipeline_{timestamp}")
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Hybrid Lens Design Pipeline")
    print("=" * 60)
    print(f"Output directory: {pipeline_dir}")
    print(f"Stages to run: {', '.join(stages)}")
    print(f"Device: {args.device or 'auto'}")
    print(f"Seed: {args.seed}")

    pipeline_start = time.time()

    # Track lens paths
    lens_paths = {}

    # Stage 1: GeoLens
    if "geolens" in stages:
        if args.geolens_path:
            print(f"\nSkipping GeoLens training, using: {args.geolens_path}")
            lens_paths["geolens"] = args.geolens_path
        elif args.skip_geolens:
            print("\nSkipping GeoLens training (--skip_geolens)")
        else:
            lens_paths["geolens"] = run_geolens_stage(args, pipeline_dir)

    # Stage 2: HybridLens
    if "hybridlens" in stages:
        geolens_path = lens_paths.get("geolens") or args.geolens_path
        if not geolens_path:
            print("\nError: GeoLens path required for HybridLens training")
            print("Either run geolens stage or provide --geolens_path")
            sys.exit(1)

        if args.hybridlens_path:
            print(f"\nSkipping HybridLens training, using: {args.hybridlens_path}")
            lens_paths["hybridlens"] = args.hybridlens_path
        else:
            lens_paths["hybridlens"] = run_hybridlens_stage(
                args, pipeline_dir, geolens_path
            )

    # Stage 3: Metasurface (optional)
    if "metalens" in stages:
        geolens_path = lens_paths.get("geolens") or args.geolens_path
        if not geolens_path:
            print("\nError: GeoLens path required for Metasurface training")
            sys.exit(1)
        lens_paths["metalens"] = run_metalens_stage(args, pipeline_dir, geolens_path)

    # Stage 4: E2E (optional)
    if "e2e" in stages:
        hybridlens_path = lens_paths.get("hybridlens") or args.hybridlens_path
        if not hybridlens_path:
            print("\nError: HybridLens path required for E2E training")
            sys.exit(1)
        lens_paths["e2e"] = run_e2e_stage(args, pipeline_dir, hybridlens_path)

    # Stage 5: Evaluation
    if "eval" in stages:
        run_evaluation_stage(args, pipeline_dir, lens_paths)

    # Summary
    total_time = time.time() - pipeline_start
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"Results saved to: {pipeline_dir}")

    print("\nGenerated files:")
    for name, path in lens_paths.items():
        if Path(path).exists():
            print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
