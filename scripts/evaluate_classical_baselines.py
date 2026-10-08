#!/usr/bin/env python3
"""
Evaluate classical restoration baselines on the current optical forward model.

Supported methods:
- Wiener deconvolution
- Richardson-Lucy deconvolution
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import richardson_lucy, wiener
from torch.utils.data import DataLoader, Dataset, Subset

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from deeplens.network.dataset import download_bsd300
from src.e2e.trainer import E2ETrainer
from src.utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate classical restoration baselines for optical systems"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--hybridlens", type=str, required=True, help="Path to HybridLens JSON file"
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["wiener", "richardson_lucy", "rl"],
        help="Classical restoration method",
    )
    parser.add_argument(
        "--dataset", type=str, help="Path to evaluation dataset directory"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory for results"
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=0.01,
        help="Regularization strength for Wiener deconvolution",
    )
    parser.add_argument(
        "--rl_iterations",
        type=int,
        default=30,
        help="Iteration count for Richardson-Lucy deconvolution",
    )
    parser.add_argument(
        "--clip",
        action="store_true",
        help="Enable clipping inside skimage restoration operators",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional limit on the number of evaluation images for smoke testing",
    )
    parser.add_argument(
        "--save_examples",
        type=int,
        default=5,
        help="Number of side-by-side examples to save",
    )
    parser.add_argument(
        "--psf_spp",
        type=int,
        help="Override PSF Monte Carlo sample count for faster smoke tests",
    )
    parser.add_argument(
        "--psf_size",
        type=int,
        help="Override PSF kernel size for faster smoke tests",
    )
    return parser.parse_args()


def to_jsonable(value):
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def calculate_metrics(clean_np, recon_np):
    b, c, _, _ = clean_np.shape
    batch_psnr = 0.0
    batch_ssim = 0.0

    for i in range(b):
        clean_img = np.transpose(clean_np[i], (1, 2, 0))
        recon_img = np.transpose(recon_np[i], (1, 2, 0))

        clean_img = np.clip(clean_img, 0, 1)
        recon_img = np.clip(recon_img, 0, 1)

        batch_psnr += psnr(clean_img, recon_img, data_range=1.0)
        batch_ssim += ssim(clean_img, recon_img, data_range=1.0, channel_axis=2)

    return batch_psnr / b, batch_ssim / b


class SimpleImageDataset(Dataset):
    def __init__(self, folder, size):
        self.files = sorted(
            glob.glob(os.path.join(folder, "*.[pjJ][pnN][gG]"))
            + glob.glob(os.path.join(folder, "*.bmp"))
        )
        self.transform = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = Image.open(self.files[index]).convert("RGB")
        return self.transform(img)


def resolve_dataset_path(config, dataset_override):
    val_path = dataset_override or getattr(config.dataset, "val_path", None)
    if val_path and os.path.exists(val_path):
        return val_path

    train_path = getattr(config.dataset, "train_path", "")
    if "BSDS300" in train_path:
        output_image_dir = download_bsd300("./datasets")
        candidate_test = os.path.join(output_image_dir, "test", "images")
        if not os.path.exists(candidate_test):
            candidate_test = os.path.join(output_image_dir, "test")
        if os.path.exists(candidate_test):
            return candidate_test

    raise ValueError("Validation dataset not found.")


def build_dataloader(config, dataset_override, limit):
    image_size = tuple(config.dataset.image_size)
    dataset_path = resolve_dataset_path(config, dataset_override)
    dataset = SimpleImageDataset(dataset_path, image_size)

    if len(dataset) == 0:
        raise ValueError(f"No images found in dataset: {dataset_path}")

    if limit is not None:
        dataset = Subset(dataset, list(range(min(limit, len(dataset)))))

    return dataset_path, DataLoader(dataset, batch_size=1, shuffle=False)


def compute_psf(trainer, config, psf_spp_override, psf_size_override):
    psf_size = psf_size_override or getattr(config.optimization, "psf_size", 101)
    wavelength = getattr(config.optimization, "wavelength", 0.55)
    requested_spp = psf_spp_override or getattr(config.optimization, "spp", 100000)
    spp = max(int(requested_spp), 1_000_000)

    with torch.no_grad():
        psf = trainer.hybrid_lens.psf(
            points=[0.0, 0.0, -10000.0],
            ks=psf_size,
            wvln=wavelength,
            spp=spp,
        )

    return psf, wavelength, psf_size, requested_spp, spp


def restore_rgb(image_np, psf_np, method, balance, rl_iterations, clip):
    restored = np.zeros_like(image_np, dtype=np.float32)
    psf_np = np.asarray(psf_np, dtype=np.float32)
    psf_np = psf_np / max(float(psf_np.sum()), 1e-12)

    for channel in range(image_np.shape[2]):
        channel_img = np.asarray(image_np[..., channel], dtype=np.float32)
        if method == "wiener":
            restored_channel = wiener(channel_img, psf_np, balance=balance, clip=clip)
        else:
            restored_channel = richardson_lucy(
                channel_img,
                psf_np,
                num_iter=rl_iterations,
                clip=clip,
            )

        restored[..., channel] = np.nan_to_num(
            restored_channel, nan=0.0, posinf=1.0, neginf=0.0
        )

    return np.clip(restored, 0.0, 1.0)


def save_side_by_side(clean_tensor, degraded_tensor, restored_tensor, path):
    ex_clean = transforms.ToPILImage()(clean_tensor.cpu())
    ex_degrad = transforms.ToPILImage()(degraded_tensor.cpu())
    ex_recon = transforms.ToPILImage()(restored_tensor.cpu())

    width, height = ex_clean.size
    combined = Image.new("RGB", (width * 3, height))
    combined.paste(ex_clean, (0, 0))
    combined.paste(ex_degrad, (width, 0))
    combined.paste(ex_recon, (width * 2, 0))
    combined.save(path)


def build_lpips(device):
    checkpoint = Path.home() / ".cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth"
    if not checkpoint.exists():
        print(
            "[classical-eval] LPIPS backbone weights not found in cache; "
            "skipping LPIPS to avoid implicit downloads."
        )
        return None

    try:
        import lpips

        return lpips.LPIPS(net="alex").to(device)
    except Exception as exc:
        print(f"[classical-eval] failed to initialize LPIPS, skipping: {exc}")
        return None


def main():
    args = parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    method_name = "richardson_lucy" if args.method == "rl" else args.method
    print(f"[classical-eval] method={method_name}")
    print(f"[classical-eval] hybridlens={args.hybridlens}")
    trainer = E2ETrainer.from_hybridlens(
        hybridlens_path=args.hybridlens,
        result_dir=str(output_dir),
        network_config={"type": "unet", "in_channels": 3, "out_channels": 3},
        freeze_doe=True,
        wavelength=getattr(config.optimization, "wavelength", 0.55),
        device=device,
    )

    dataset_path, val_loader = build_dataloader(config, args.dataset, args.limit)
    print(f"[classical-eval] dataset={dataset_path} num_images={len(val_loader)}")
    psf, wavelength, psf_size, requested_spp, effective_spp = compute_psf(
        trainer, config, args.psf_spp, args.psf_size
    )
    print(
        "[classical-eval] "
        f"psf_size={psf_size} requested_spp={requested_spp} "
        f"effective_spp={effective_spp} wavelength={wavelength}um"
    )

    psf_img = psf.squeeze().detach().cpu().numpy()
    psf_img_u8 = (psf_img / max(float(psf_img.max()), 1e-12) * 255).astype(np.uint8)
    Image.fromarray(psf_img_u8).save(output_dir / "eval_psf.png")

    mse_loss_fn = nn.MSELoss()
    lpips_fn = build_lpips(device)

    total_mse = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    num_batches = 0

    save_examples = min(args.save_examples, len(val_loader))
    psf_np = psf.squeeze().detach().cpu().numpy().astype(np.float32)

    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            print(f"[classical-eval] processing {idx + 1}/{len(val_loader)}")
            clean_images = batch["img"].to(device) if isinstance(batch, dict) else batch.to(device)
            clean_images = clean_images.float()

            degraded_images = trainer._simulate_image(clean_images, psf)
            degraded_np = degraded_images[0].detach().cpu().permute(1, 2, 0).numpy()
            restored_np = restore_rgb(
                image_np=degraded_np,
                psf_np=psf_np,
                method=method_name,
                balance=args.balance,
                rl_iterations=args.rl_iterations,
                clip=args.clip,
            )

            restored_tensor = (
                torch.from_numpy(restored_np.transpose(2, 0, 1))
                .unsqueeze(0)
                .to(device=device, dtype=torch.float32)
            )

            mse = mse_loss_fn(restored_tensor, clean_images).item()
            total_mse += mse

            if lpips_fn is not None:
                clean_norm = clean_images * 2.0 - 1.0
                recon_norm = restored_tensor * 2.0 - 1.0
                lpips_val = lpips_fn(recon_norm, clean_norm).mean().item()
                total_lpips += lpips_val

            clean_np = clean_images.detach().cpu().numpy()
            recon_np = restored_tensor.detach().cpu().numpy()
            batch_psnr, batch_ssim = calculate_metrics(clean_np, recon_np)
            total_psnr += batch_psnr
            total_ssim += batch_ssim
            num_batches += 1

            if idx < save_examples:
                save_side_by_side(
                    clean_tensor=clean_images[0],
                    degraded_tensor=degraded_images[0],
                    restored_tensor=restored_tensor[0],
                    path=output_dir / f"eval_example_{idx}.png",
                )

    results = {
        "method": method_name,
        "MSE": total_mse / max(num_batches, 1),
        "PSNR": total_psnr / max(num_batches, 1),
        "SSIM": total_ssim / max(num_batches, 1),
        "LPIPS": total_lpips / max(num_batches, 1) if lpips_fn else None,
        "num_images": num_batches,
        "dataset": dataset_path,
        "hybridlens": args.hybridlens,
        "wavelength_um": wavelength,
        "psf_size": psf_size,
        "psf_spp_requested": int(requested_spp),
        "psf_spp_effective": int(effective_spp),
        "wiener_balance": args.balance if method_name == "wiener" else None,
        "rl_iterations": args.rl_iterations
        if method_name == "richardson_lucy"
        else None,
        "clip": bool(args.clip),
    }

    print("\n" + "=" * 50)
    print(f"Classical Baseline Evaluation: {method_name}")
    print("=" * 50)
    for key in ["MSE", "PSNR", "SSIM", "LPIPS"]:
        value = results[key]
        if value is not None:
            print(f"{key}: {value:.4f}")

    with open(output_dir / "eval_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(results), handle, indent=4, ensure_ascii=False)

    with open(output_dir / "eval_config.json", "w", encoding="utf-8") as handle:
        json.dump(
            to_jsonable(
                {
                    "config": args.config,
                    "hybridlens": args.hybridlens,
                    "method": method_name,
                    "dataset": dataset_path,
                    "output_dir": str(output_dir),
                    "balance": args.balance,
                    "rl_iterations": args.rl_iterations,
                    "clip": bool(args.clip),
                    "limit": args.limit,
                    "save_examples": args.save_examples,
                    "psf_spp_requested": int(requested_spp),
                    "psf_spp_effective": int(effective_spp),
                    "psf_size": int(psf_size),
                }
            ),
            handle,
            indent=4,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
