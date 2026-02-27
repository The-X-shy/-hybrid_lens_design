#!/usr/bin/env python3
"""
Evaluate End-to-End System (HybridLens + Neural Network)
"""

import argparse
import sys
from pathlib import Path
import os
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import glob
from torch.utils.data import Dataset, DataLoader

# Import metrics
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from src.utils.config import load_config
from src.e2e.trainer import E2ETrainer
from deeplens.network.dataset import download_bsd300


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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate End-to-End E2E model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to network checkpoint (.pth)",
    )
    parser.add_argument(
        "--hybridlens", type=str, required=True, help="Path to HybridLens JSON"
    )
    parser.add_argument(
        "--dataset", type=str, help="Path to evaluation dataset directory"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory for results"
    )
    return parser.parse_args()


def calculate_metrics(clean_np, recon_np):
    """Calculate PSNR and SSIM for a batch of images"""
    b, c, h, w = clean_np.shape
    batch_psnr = 0
    batch_ssim = 0

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
        self.files = glob.glob(os.path.join(folder, "*.[pjJ][pnN][gG]")) + glob.glob(
            os.path.join(folder, "*.bmp")
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

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGB")
        return self.transform(img)


def main():
    args = parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    network_config = {
        "type": config.network.type,
        "in_channels": getattr(config.network, "in_channels", 3),
        "out_channels": getattr(config.network, "out_channels", 3),
    }
    if hasattr(config.network, "features"):
        network_config["features"] = config.network.features
    if hasattr(config.network, "mha"):
        network_config["mha"] = (
            config.network.mha.to_dict()
            if hasattr(config.network.mha, "to_dict")
            else config.network.mha
        )

    trainer = E2ETrainer.from_hybridlens(
        hybridlens_path=args.hybridlens,
        result_dir=args.output_dir,
        network_config=network_config,
        freeze_doe=True,
        wavelength=getattr(config.optimization, "wavelength", 0.55),
        device=device,
    )

    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    trainer.network.load_state_dict(state_dict)
    trainer.network.eval()

    val_path = args.dataset or getattr(config.dataset, "val_path", None)
    if not val_path or not os.path.exists(val_path):
        if "BSDS300" in getattr(config.dataset, "train_path", ""):
            output_image_dir = download_bsd300("./datasets")
            candidate_test = os.path.join(output_image_dir, "test", "images")
            if not os.path.exists(candidate_test):
                candidate_test = os.path.join(output_image_dir, "test")
            val_path = candidate_test
        else:
            raise ValueError("Validation dataset not found.")

    image_size = tuple(config.dataset.image_size)
    dataset = SimpleImageDataset(val_path, image_size)
    val_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    psf_size = getattr(config.optimization, "psf_size", 101)
    spp = getattr(config.optimization, "spp", 100000)
    wavelength = getattr(config.optimization, "wavelength", 0.55)

    with torch.no_grad():
        psf = trainer.hybrid_lens.psf(
            points=[0.0, 0.0, -10000.0],
            ks=psf_size,
            wvln=wavelength,
            spp=max(int(spp), 1_000_000),
        )

    psf_img = psf.squeeze().cpu().numpy()
    psf_img = (psf_img / psf_img.max() * 255).astype(np.uint8)
    Image.fromarray(psf_img).save(os.path.join(args.output_dir, "eval_psf.png"))

    mse_loss_fn = nn.MSELoss()
    try:
        import lpips

        lpips_fn = lpips.LPIPS(net="alex").to(device)
    except:
        lpips_fn = None

    total_mse = 0
    total_psnr = 0
    total_ssim = 0
    total_lpips = 0
    num_batches = 0

    save_examples = min(5, len(val_loader))

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if isinstance(batch, dict):
                clean_images = batch["img"].to(device)
            else:
                clean_images = batch.to(device)

            clean_images = clean_images.float()

            degraded_images = trainer._simulate_image(clean_images, psf)
            reconstructed_images = trainer.network(degraded_images)
            reconstructed_images = torch.clamp(reconstructed_images, 0.0, 1.0)

            mse = mse_loss_fn(reconstructed_images, clean_images).item()
            total_mse += mse

            if lpips_fn is not None:
                clean_norm = clean_images * 2.0 - 1.0
                recon_norm = reconstructed_images * 2.0 - 1.0
                lpips_val = lpips_fn(recon_norm, clean_norm).mean().item()
                total_lpips += lpips_val

            clean_np = clean_images.cpu().numpy()
            recon_np = reconstructed_images.cpu().numpy()
            b_psnr, b_ssim = calculate_metrics(clean_np, recon_np)

            total_psnr += b_psnr
            total_ssim += b_ssim
            num_batches += 1

            if i < save_examples:
                ex_clean = transforms.ToPILImage()(clean_images[0].cpu())
                ex_degrad = transforms.ToPILImage()(degraded_images[0].cpu())
                ex_recon = transforms.ToPILImage()(reconstructed_images[0].cpu())

                w, h = ex_clean.size
                comb = Image.new("RGB", (w * 3, h))
                comb.paste(ex_clean, (0, 0))
                comb.paste(ex_degrad, (w, 0))
                comb.paste(ex_recon, (w * 2, 0))
                comb.save(os.path.join(args.output_dir, f"eval_example_{i}.png"))

    results = {
        "MSE": total_mse / num_batches,
        "PSNR": total_psnr / num_batches,
        "SSIM": total_ssim / num_batches,
        "LPIPS": total_lpips / num_batches if lpips_fn else None,
    }

    print("\n" + "=" * 50)
    print("Final Evaluation Results on Test Set")
    print("=" * 50)
    for k, v in results.items():
        if v is not None:
            print(f"{k}: {v:.4f}")

    with open(os.path.join(args.output_dir, "eval_metrics.json"), "w") as f:
        json.dump(to_jsonable(results), f, indent=4)


if __name__ == "__main__":
    main()
