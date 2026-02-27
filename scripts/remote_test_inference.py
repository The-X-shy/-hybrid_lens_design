import argparse
import sys
import os
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.e2e.trainer import E2ETrainer

def parse_args():
    parser = argparse.ArgumentParser(description="Test inference on a single image")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to network checkpoint (.pth)")
    parser.add_argument("--hybridlens", type=str, required=True, help="Path to HybridLens JSON")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, required=True, help="Output path for the reconstructed image")
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    network_config = {
        "type": config.network.type,
        "in_channels": getattr(config.network, "in_channels", 3),
        "out_channels": getattr(config.network, "out_channels", 3),
    }
    if hasattr(config.network, "features"):
        network_config["features"] = config.network.features
    if hasattr(config.network, "mha"):
        network_config["mha"] = config.network.mha.to_dict() if hasattr(config.network.mha, "to_dict") else config.network.mha

    trainer = E2ETrainer.from_hybridlens(
        hybridlens_path=args.hybridlens,
        result_dir=os.path.dirname(args.output),
        network_config=network_config,
        freeze_doe=True,
        wavelength=getattr(config.optimization, "wavelength", 0.55),
        device=device,
    )
    
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    trainer.network.load_state_dict(state_dict)
    trainer.network.eval()
    
    img = Image.open(args.image).convert('RGB')
    image_size = tuple(config.dataset.image_size)
    
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor()
    ])
    
    clean_image = transform(img).unsqueeze(0).to(device)
    
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
        # Fix dynamic float/double casting for convolutions!
        clean_image = clean_image.to(psf.dtype)
        
        degraded_image = trainer._simulate_image(clean_image, psf)
        
        # Recon network expects float
        reconstructed_image = trainer.network(degraded_image.float())
        reconstructed_image = torch.clamp(reconstructed_image, 0.0, 1.0)
        
    ex_clean = transforms.ToPILImage()(clean_image[0].cpu())
    ex_degrad = transforms.ToPILImage()(degraded_image[0].cpu())
    ex_recon = transforms.ToPILImage()(reconstructed_image[0].cpu())
    
    w, h = ex_clean.size
    comb = Image.new('RGB', (w*3, h))
    comb.paste(ex_clean, (0, 0))
    comb.paste(ex_degrad, (w, 0))
    comb.paste(ex_recon, (w*2, 0))
    comb.save(args.output)
    print(f"Saved: {args.output}")

if __name__ == "__main__":
    main()
