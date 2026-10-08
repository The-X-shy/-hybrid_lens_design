"""
End-to-End Trainer - Joint Optimization of HybridLens and Neural Network
端到端训练器 - 混合透镜与神经网络联合优化

Based on: Xinge Yang et al., "End-to-End Hybrid Refractive-Diffractive Lens Design
with Differentiable Ray-Wave Model," SIGGRAPH Asia 2024.

This trainer jointly optimizes:
1. HybridLens (refractive lens + DOE)
2. Neural network (UNet for image reconstruction/deblurring)

The goal is to optimize the optical system and neural network together to achieve
better image quality than optimizing them separately.
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.utils import save_image

# DeepLens imports
from deeplens.hybridlens import HybridLens
from deeplens.optics.loss import PSFLoss
from deeplens.network.reconstruction.nafnet import NAFNet
from deeplens.network.reconstruction.restormer import Restormer
from deeplens.network.reconstruction.unet import UNet
from deeplens.network.dataset import PhotographicDataset

from src.e2e.networks import MHAUNet
from src.utils.stage_logger import StageLogger
from src.utils.early_stop import EarlyStopper


class E2ETrainer:
    """
    End-to-end trainer for joint optimization of HybridLens and neural network.
    混合透镜与神经网络联合优化的端到端训练器
    """

    def __init__(
        self,
        hybrid_lens: HybridLens,
        network: nn.Module,
        result_dir: str,
        source_hybridlens_path: Optional[str] = None,
        freeze_doe: bool = True,
        wavelength: float = 0.55,
        device: torch.device = None,
    ):
        """
        Initialize the E2E trainer.

        Args:
            hybrid_lens: HybridLens object to train
            network: Neural network for image reconstruction
            result_dir: Directory to save results
            device: PyTorch device
        """
        self.hybrid_lens = hybrid_lens
        self.network = network
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.source_hybridlens_path = source_hybridlens_path
        self.freeze_doe = freeze_doe
        self.wavelength = wavelength

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Keep reconstruction network in float32 for stable conv/attention ops,
        # while HybridLens stays in float64 for wave-optics simulation.
        self.network = self.network.to(device=self.device, dtype=torch.float32)
        self.logger = StageLogger(
            result_dir=str(self.result_dir),
            stage="e2e",
            mode="freeze_doe" if self.freeze_doe else "joint",
            wavelength=self.wavelength,
        )

    @classmethod
    def from_hybridlens(
        cls,
        hybridlens_path: str,
        result_dir: str,
        network_config: Dict[str, Any] = None,
        refocus_on_load: bool = True,
        freeze_doe: bool = True,
        wavelength: float = 0.55,
        device: torch.device = None,
    ) -> "E2ETrainer":
        """
        Create trainer from an optimized HybridLens file.
        从优化后的 HybridLens 文件创建训练器

        Args:
            hybridlens_path: Path to the optimized HybridLens JSON file
            result_dir: Directory to save results
            network_config: Network configuration dict
            device: PyTorch device
        """
        from src.utils.config import get_device

        if device is None:
            device = get_device(None)

        result_dir = Path(result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)

        # Set double precision for phase calculation
        torch.set_default_dtype(torch.float64)

        # Load HybridLens
        hybrid_lens = HybridLens(filename=hybridlens_path, dtype=torch.float64)

        # Training from a source hybrid lens still expects refocus to infinity,
        # but evaluating an already-saved E2E checkpoint lens must preserve the
        # serialized state instead of refocusing again.
        if refocus_on_load:
            hybrid_lens.refocus(foc_dist=-10000.0)

        # Default network config
        if network_config is None:
            network_config = {
                "type": "UNet",
                "in_channels": 3,
                "out_channels": 3,
            }

        # Create neural network
        network = cls._create_network(network_config)

        return cls(
            hybrid_lens=hybrid_lens,
            network=network,
            result_dir=str(result_dir),
            source_hybridlens_path=hybridlens_path,
            freeze_doe=freeze_doe,
            wavelength=wavelength,
            device=device,
        )

    @classmethod
    def from_config(cls, config) -> "E2ETrainer":
        """
        Create trainer from configuration.
        从配置创建训练器
        """
        from src.utils.config import create_result_dir, get_device

        result_dir = create_result_dir(config)
        device = get_device(config)

        # Load HybridLens path
        hybridlens_path = config.lens.file_path

        # Network config from yaml
        network_config = {
            "type": config.network.type,
            "in_channels": config.network.in_channels,
            "out_channels": config.network.out_channels,
        }
        if hasattr(config.network, "features"):
            network_config["features"] = config.network.features
        if hasattr(config.network, "mha"):
            network_config["mha"] = (
                config.network.mha.to_dict()
                if hasattr(config.network.mha, "to_dict")
                else config.network.mha
            )
        if hasattr(config.network, "params"):
            network_config["params"] = (
                config.network.params.to_dict()
                if hasattr(config.network.params, "to_dict")
                else config.network.params
            )
        if hasattr(config.network, "pretrained") and config.network.pretrained:
            network_config["pretrained"] = config.network.pretrained

        freeze_doe = getattr(config.optimization, "freeze_doe", True)
        wavelength = getattr(config.optimization, "wavelength", 0.55)

        return cls.from_hybridlens(
            hybridlens_path=hybridlens_path,
            result_dir=str(result_dir),
            network_config=network_config,
            freeze_doe=freeze_doe,
            wavelength=wavelength,
            device=device,
        )

    @staticmethod
    def _create_network(network_config: Dict[str, Any]) -> nn.Module:
        """
        Create neural network from configuration.
        从配置创建神经网络
        """
        network_type = network_config.get("type", "UNet").lower()
        in_channels = network_config.get("in_channels", 3)
        out_channels = network_config.get("out_channels", 3)
        params = network_config.get("params", {})
        if hasattr(params, "to_dict"):
            params = params.to_dict()
        if not isinstance(params, dict):
            params = {}

        if network_type == "unet":
            network = UNet(in_channels=in_channels, out_channels=out_channels)
        elif network_type == "mha_unet":
            mha_cfg = network_config.get("mha", {})
            network = MHAUNet(
                in_channels=in_channels,
                out_channels=out_channels,
                embed_dim=int(mha_cfg.get("embed_dim", 64)),
                num_heads=int(mha_cfg.get("num_heads", 4)),
                ff_dim=int(mha_cfg.get("ff_dim", 128))
                if mha_cfg.get("ff_dim", None) is not None
                else None,
                dropout=float(mha_cfg.get("dropout", 0.0)),
            )
        elif network_type == "nafnet":
            network = NAFNet(
                in_chan=in_channels,
                out_chan=out_channels,
                width=int(params.get("width", 32)),
                middle_blk_num=int(params.get("middle_blk_num", 1)),
                enc_blk_nums=list(params.get("enc_blk_nums", [1, 1, 1, 28])),
                dec_blk_nums=list(params.get("dec_blk_nums", [1, 1, 1, 1])),
            )
        elif network_type == "restormer":
            network = Restormer(
                inp_channels=in_channels,
                out_channels=out_channels,
                dim=int(params.get("dim", 48)),
                num_blocks=list(params.get("num_blocks", [4, 6, 6, 8])),
                num_refinement_blocks=int(params.get("num_refinement_blocks", 4)),
                heads=list(params.get("heads", [1, 2, 4, 8])),
                ffn_expansion_factor=float(params.get("ffn_expansion_factor", 2.66)),
                bias=bool(params.get("bias", False)),
                LayerNorm_type=str(params.get("LayerNorm_type", "WithBias")),
                dual_pixel_task=bool(params.get("dual_pixel_task", False)),
            )
        else:
            raise ValueError(f"Unsupported network type: {network_type}")

        # Load pretrained weights if specified
        if "pretrained" in network_config and network_config["pretrained"]:
            pretrained_path = network_config["pretrained"]
            if os.path.exists(pretrained_path):
                network.load_state_dict(torch.load(pretrained_path, weights_only=True))
                print(f"Loaded pretrained weights from: {pretrained_path}")

        return network

    def setup_sensor(
        self, match_aperture: bool = False, sensor_res: Tuple[int, int] = (2000, 2000)
    ):
        """
        Configure sensor geometry. By default preserve the loaded GeoLens sensor.
        默认保留输入折射镜头的传感器尺寸
        """
        if match_aperture:
            aper_r = self.hybrid_lens.geolens.surfaces[
                self.hybrid_lens.geolens.aper_idx
            ].r
            new_sensor_size = (2 * aper_r, 2 * aper_r)
            self.hybrid_lens.set_sensor(
                sensor_size=new_sensor_size, sensor_res=sensor_res
            )
            self.hybrid_lens.geolens.set_sensor(
                sensor_size=new_sensor_size, sensor_res=sensor_res
            )
        else:
            sensor_size = tuple(self.hybrid_lens.geolens.sensor_size)
            self.hybrid_lens.set_sensor(
                sensor_size=sensor_size, sensor_res=sensor_res
            )
            self.hybrid_lens.geolens.set_sensor(
                sensor_size=sensor_size, sensor_res=sensor_res
            )

    def _simulate_image(
        self,
        clean_image: torch.Tensor,
        psf: torch.Tensor,
    ) -> torch.Tensor:
        """
        Simulate degraded image by convolving with PSF.
        通过与 PSF 卷积模拟退化图像

        Args:
            clean_image: Clean image tensor [B, C, H, W]
            psf: PSF tensor [ks, ks]

        Returns:
            Degraded image tensor [B, C, H, W]
        """
        # Prepare PSF for convolution
        psf = psf.unsqueeze(0).unsqueeze(0)  # [1, 1, ks, ks]
        channels = clean_image.shape[1]
        psf = psf.repeat(channels, 1, 1, 1)
        psf = psf.float()

        # Pad image for convolution
        pad_size = psf.shape[-1] // 2
        padded_image = F.pad(clean_image, [pad_size] * 4, mode="reflect")

        # Convolve each channel
        degraded = F.conv2d(
            padded_image,
            psf,
            groups=channels,
            padding=0,
        )

        return degraded

    @staticmethod
    def _flatten_param_groups(param_groups: List[Dict[str, Any]]) -> List[torch.Tensor]:
        params: List[torch.Tensor] = []
        for group in param_groups:
            group_params = group.get("params", [])
            if isinstance(group_params, torch.Tensor):
                params.append(group_params)
            else:
                params.extend(list(group_params))
        return params

    def _get_doe_trainable_params(self, doe_lr: float) -> List[torch.Tensor]:
        try:
            param_groups = self.hybrid_lens.doe.get_optimizer_params(lr=doe_lr)
            return self._flatten_param_groups(param_groups)
        except Exception:
            return []

    def _get_geolens_param_groups(
        self,
        lens_lr: List[float],
        lr_decay: float,
    ) -> List[Dict[str, Any]]:
        return self.hybrid_lens.geolens.get_optimizer_params(
            lrs=lens_lr,
            decay=lr_decay,
            optim_mat=False,
        )

    @staticmethod
    def _set_param_requires_grad(
        param_groups: List[Dict[str, Any]],
        requires_grad: bool,
    ) -> None:
        params = E2ETrainer._flatten_param_groups(param_groups)
        for param in params:
            if isinstance(param, torch.Tensor):
                param.requires_grad = requires_grad

    @staticmethod
    def _resolve_training_mode(freeze_doe: bool, freeze_geolens: bool) -> str:
        if freeze_doe and freeze_geolens:
            return "network_only"
        if freeze_doe and not freeze_geolens:
            return "refractive_only"
        if not freeze_doe and freeze_geolens:
            return "doe_only"
        return "joint"

    def _capture_doe_state(self) -> Dict[str, torch.Tensor]:
        state = {}
        candidate_keys = [
            "alpha2",
            "alpha4",
            "alpha6",
            "alpha8",
            "alpha10",
            "phase_map",
        ]
        for key in candidate_keys:
            if hasattr(self.hybrid_lens.doe, key):
                value = getattr(self.hybrid_lens.doe, key)
                if isinstance(value, torch.Tensor):
                    state[key] = value.detach().cpu().clone()
        return state

    @staticmethod
    def _build_early_stopper(
        early_stop_config: Optional[Dict[str, Any]],
        default_mode: str = "min",
    ) -> EarlyStopper:
        cfg = early_stop_config or {}
        return EarlyStopper(
            enabled=bool(cfg.get("enabled", False)),
            patience=int(cfg.get("patience", 500)),
            min_delta=float(cfg.get("min_delta", 1e-6)),
            mode=str(cfg.get("mode", default_mode)),
        )

    def _save_input_hybridlens_snapshot(self) -> Optional[str]:
        """Persist the E2E input lens used as the evaluation reference."""
        if not self.source_hybridlens_path:
            return None

        src_path = Path(self.source_hybridlens_path)
        if not src_path.is_absolute():
            src_path = Path.cwd() / src_path
        src_path = src_path.resolve()
        if not src_path.exists():
            return None

        dst_path = self.result_dir / "hybridlens_input.json"
        shutil.copy2(src_path, dst_path)
        return str(dst_path)

    def train(
        self,
        train_dataset_path: str,
        val_dataset_path: str = None,
        doe_lr: float = 0.1,
        lens_lr: List[float] = [1e-4, 1e-4, 1e-2, 1e-5],
        network_lr: float = 1e-4,
        lr_decay: float = 0.01,
        epochs: int = 100,
        start_epoch: int = 0,
        batch_size: int = 4,
        test_per_epoch: int = 1,
        spp: int = 1000000,
        psf_size: int = 101,
        image_size: Tuple[int, int] = (256, 256),
        color_mode: str = "rgb",
        loss_weights: Dict[str, float] = None,
        freeze_doe: Optional[bool] = None,
        freeze_geolens: bool = False,
        freeze_network: bool = False,
        wavelength: Optional[float] = None,
        early_stop_config: Optional[Dict[str, Any]] = None,
        save_network_every: Optional[int] = None,
        optics_grad_batches: Optional[int] = None,
        save_validation_best: bool = False,
    ) -> Dict[str, List[float]]:
        """
        Train the E2E system by jointly optimizing HybridLens and neural network.
        通过联合优化 HybridLens 和神经网络来训练端到端系统

        Args:
            train_dataset_path: Path to training dataset
            val_dataset_path: Path to validation dataset (optional)
            doe_lr: Learning rate for DOE parameters
            lens_lr: Learning rates [d, c, k, ai] for lens parameters
            network_lr: Learning rate for neural network
            lr_decay: Learning rate decay
            epochs: Number of training epochs
            batch_size: Batch size
            test_per_epoch: Evaluation interval (epochs)
            spp: Samples per pixel for PSF calculation
            psf_size: PSF patch size
            image_size: Training image size
            loss_weights: Weights for different loss terms
            freeze_doe: Whether to freeze DOE parameters during E2E training
            freeze_geolens: Whether to freeze refractive lens parameters
            wavelength: Single wavelength used for PSF/rendering
            early_stop_config: Early stop settings
            optics_grad_batches: Number of batches per epoch that backpropagate
                through optics. If optics are trainable, defaults to 1.

        Returns:
            Dictionary of loss histories
        """
        hybrid_lens = self.hybrid_lens
        network = self.network
        freeze_doe = self.freeze_doe if freeze_doe is None else freeze_doe
        freeze_network = bool(freeze_network)
        wavelength = self.wavelength if wavelength is None else wavelength
        self.freeze_doe = freeze_doe
        self.wavelength = wavelength
        training_mode = self._resolve_training_mode(
            freeze_doe=freeze_doe, freeze_geolens=freeze_geolens
        )
        self.logger.mode = training_mode
        self.logger.wavelength = wavelength

        # Default loss weights
        if loss_weights is None:
            loss_weights = {
                "mse": 1.0,
                "lpips": 0.1,
                "psf": 0.01,
            }

        # Create dataset and dataloader
        if color_mode in {"grayscale", "grayscale_luminance", "luminance"}:
            from paper_exp.data.grayscale_dataset import GrayscalePhotographicDataset

            dataset_class = GrayscalePhotographicDataset
        elif color_mode == "rgb":
            dataset_class = PhotographicDataset
        else:
            raise ValueError(f"Unsupported color_mode: {color_mode}")
        train_dataset = dataset_class(
            img_dir=train_dataset_path,
            img_res=image_size,
            is_train=True,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        if val_dataset_path:
            val_dataset = dataset_class(
                img_dir=val_dataset_path,
                img_res=image_size,
                is_train=False,
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
            )
        else:
            val_loader = None

        # Create optimizers
        doe_state_before = self._capture_doe_state()
        doe_params = self._get_doe_trainable_params(doe_lr=doe_lr)
        for p in doe_params:
            p.requires_grad = not freeze_doe

        geolens_param_groups = self._get_geolens_param_groups(
            lens_lr=lens_lr, lr_decay=lr_decay
        )
        self._set_param_requires_grad(
            geolens_param_groups, requires_grad=not freeze_geolens
        )

        lens_param_groups: List[Dict[str, Any]] = []
        if not freeze_geolens:
            lens_param_groups.extend(geolens_param_groups)
        if not freeze_doe:
            lens_param_groups.extend(
                self.hybrid_lens.doe.get_optimizer_params(lr=doe_lr)
            )

        lens_optimizer = (
            torch.optim.Adam(lens_param_groups) if lens_param_groups else None
        )

        # Network optimizer
        for param in network.parameters():
            param.requires_grad = not freeze_network

        trainable_network_params = [
            param for param in network.parameters() if param.requires_grad
        ]
        network_optimizer = (
            torch.optim.Adam(trainable_network_params, lr=network_lr)
            if trainable_network_params
            else None
        )

        # Learning rate schedulers
        lens_scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(lens_optimizer, T_max=epochs)
            if lens_optimizer is not None
            else None
        )
        network_scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                network_optimizer, T_max=epochs
            )
            if network_optimizer is not None
            else None
        )

        # Loss functions
        mse_loss_fn = nn.MSELoss()
        psf_loss_fn = PSFLoss()

        # Try to load LPIPS if available
        try:
            import lpips

            lpips_loss_fn = lpips.LPIPS(net="alex").to(self.device)
            use_lpips = True
        except ImportError:
            print("Warning: LPIPS not available, using MSE only")
            lpips_loss_fn = None
            use_lpips = False

        # Training configuration
        print("=" * 60)
        print("Starting End-to-End Joint Optimization")
        print("=" * 60)
        print(f"Network type: {type(network).__name__}")
        print(f"Training images: {len(train_dataset)}")
        print(f"Color mode: {color_mode}")
        print(f"Batch size: {batch_size}")
        print(f"Epochs: {epochs}")
        print(f"PSF size: {psf_size}x{psf_size}")
        print(f"Wavelength: {wavelength} um")
        print(f"Freeze DOE: {freeze_doe}")
        print(f"Freeze GeoLens: {freeze_geolens}")
        print(f"Freeze Network: {freeze_network}")
        print(f"Training mode: {training_mode}")
        print(f"Loss weights: {loss_weights}")

        input_snapshot = self._save_input_hybridlens_snapshot()

        self.logger.log_event(
            event="train_start",
            payload={
                "train_dataset_path": train_dataset_path,
                "val_dataset_path": val_dataset_path,
                "epochs": epochs,
                "batch_size": batch_size,
                "color_mode": color_mode,
                "save_validation_best": bool(save_validation_best),
                "network_lr": network_lr,
                "doe_lr": doe_lr,
                "lens_lr": lens_lr,
                "lr_decay": lr_decay,
                "spp": spp,
                "psf_size": psf_size,
                "wavelength": wavelength,
                "freeze_doe": freeze_doe,
                "freeze_geolens": freeze_geolens,
                "freeze_network": freeze_network,
                "training_mode": training_mode,
                "input_hybridlens_snapshot": input_snapshot,
            },
        )
        early_stopper = self._build_early_stopper(early_stop_config, default_mode="min")
        monitor_key = str((early_stop_config or {}).get("monitor", "total"))
        if monitor_key.startswith("val_") and val_loader is None:
            raise ValueError(
                f"Validation loader is required when monitoring {monitor_key}"
            )
        if save_validation_best and val_loader is None:
            raise ValueError("Validation loader is required to save validation best")
        if early_stopper.enabled:
            self.logger.log_event(
                event="early_stop_enabled",
                payload={
                    "patience": early_stopper.patience,
                    "min_delta": early_stopper.min_delta,
                    "mode": early_stopper.mode,
                    "monitor": monitor_key,
                },
            )

        # Record loss history
        loss_history = {
            "total": [],
            "mse": [],
            "lpips": [],
            "psf": [],
            "val_mse": [],
        }
        best_validation_mse = None
        best_validation_epoch = None
        min_coherent_spp = 1_000_000
        effective_spp = max(int(spp), min_coherent_spp)
        optics_trainable = lens_optimizer is not None
        optics_grad_batches = (
            0
            if not optics_trainable
            else max(1, int(optics_grad_batches or 1))
        )
        psf_refresh_epochs = max(1, int(test_per_epoch))
        cached_psf = None
        cached_psf_loss = None

        # Training loop
        for epoch in range(start_epoch, epochs):
            network.train(not freeze_network)
            epoch_losses = {"total": 0, "mse": 0, "lpips": 0, "psf": 0}
            num_batches = 0
            if not optics_trainable and (
                cached_psf is None or (epoch % psf_refresh_epochs == 0)
            ):
                with torch.no_grad():
                    cached_psf = hybrid_lens.psf(
                        points=[0.0, 0.0, -10000.0],
                        ks=psf_size,
                        wvln=wavelength,
                        spp=effective_spp,
                    )
                    if loss_weights.get("psf", 0) > 0:
                        psf_input = cached_psf.unsqueeze(0).unsqueeze(0)
                        if psf_input.shape[1] == 1:
                            psf_input = psf_input.repeat(1, 3, 1, 1)
                        cached_psf_loss = psf_loss_fn(psf_input).detach()
                    else:
                        cached_psf_loss = None

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
            cached_psf = cached_psf if not optics_trainable else None
            cached_psf_loss = cached_psf_loss if not optics_trainable else None
            for batch_idx, batch in enumerate(pbar):
                # Get clean images
                if isinstance(batch, dict):
                    clean_images = batch["img"].to(self.device)
                else:
                    clean_images = batch.to(self.device)

                use_optics_grad = optics_trainable and batch_idx < optics_grad_batches
                if use_optics_grad:
                    psf = hybrid_lens.psf(
                        points=[0.0, 0.0, -10000.0],
                        ks=psf_size,
                        wvln=wavelength,
                        spp=effective_spp,
                    )
                    if loss_weights.get("psf", 0) > 0:
                        psf_input = psf.unsqueeze(0).unsqueeze(0)
                        if psf_input.shape[1] == 1:
                            psf_input = psf_input.repeat(1, 3, 1, 1)
                        loss_psf = psf_loss_fn(psf_input)
                    else:
                        loss_psf = torch.tensor(0.0, device=self.device)
                else:
                    if cached_psf is None:
                        with torch.no_grad():
                            cached_psf = hybrid_lens.psf(
                                points=[0.0, 0.0, -10000.0],
                                ks=psf_size,
                                wvln=wavelength,
                                spp=effective_spp,
                            )
                            if loss_weights.get("psf", 0) > 0:
                                psf_input = cached_psf.unsqueeze(0).unsqueeze(0)
                                if psf_input.shape[1] == 1:
                                    psf_input = psf_input.repeat(1, 3, 1, 1)
                                cached_psf_loss = psf_loss_fn(psf_input).detach()
                            else:
                                cached_psf_loss = None
                    psf = cached_psf

                # Simulate degraded images
                degraded_images = self._simulate_image(clean_images.float(), psf)

                # Reconstruct images with neural network
                # Note: Network needs float32
                reconstructed_images = network(degraded_images.float())

                # Calculate losses
                # MSE loss
                loss_mse = mse_loss_fn(reconstructed_images, clean_images.float())

                # LPIPS loss
                if use_lpips and loss_weights.get("lpips", 0) > 0:
                    lpips_reconstructed = reconstructed_images
                    lpips_clean = clean_images.float()
                    if lpips_reconstructed.shape[1] == 1:
                        lpips_reconstructed = lpips_reconstructed.repeat(1, 3, 1, 1)
                        lpips_clean = lpips_clean.repeat(1, 3, 1, 1)
                    loss_lpips = lpips_loss_fn(
                        lpips_reconstructed, lpips_clean
                    ).mean()
                else:
                    loss_lpips = torch.tensor(0.0, device=self.device)

                # PSF loss (encourage compact PSF)
                if not use_optics_grad:
                    if loss_weights.get("psf", 0) > 0:
                        loss_psf = (
                            cached_psf_loss.to(self.device)
                            if cached_psf_loss is not None
                            else torch.tensor(0.0, device=self.device)
                        )
                    else:
                        loss_psf = torch.tensor(0.0, device=self.device)

                # Total loss
                total_loss = (
                    loss_weights.get("mse", 1.0) * loss_mse
                    + loss_weights.get("lpips", 0.1) * loss_lpips
                    + loss_weights.get("psf", 0.01) * loss_psf
                )

                # Backward and optimize
                if lens_optimizer is not None:
                    lens_optimizer.zero_grad()
                if network_optimizer is not None:
                    network_optimizer.zero_grad()
                total_loss.backward()
                if lens_optimizer is not None and use_optics_grad:
                    lens_optimizer.step()
                if network_optimizer is not None:
                    network_optimizer.step()

                # Record losses
                epoch_losses["total"] += total_loss.item()
                epoch_losses["mse"] += loss_mse.item()
                epoch_losses["lpips"] += loss_lpips.item() if use_lpips else 0
                epoch_losses["psf"] += loss_psf.item()
                num_batches += 1

                pbar.set_postfix(
                    loss=f"{total_loss.item():.4f}",
                    mse=f"{loss_mse.item():.4f}",
                )

            # Update learning rates
            if lens_scheduler is not None:
                lens_scheduler.step()
            if network_scheduler is not None:
                network_scheduler.step()

            # Average epoch losses
            for key in epoch_losses:
                epoch_losses[key] /= num_batches
                loss_history[key].append(epoch_losses[key])

            print(
                f"Epoch {epoch + 1}: total={epoch_losses['total']:.4f}, "
                f"mse={epoch_losses['mse']:.4f}, lpips={epoch_losses['lpips']:.4f}, "
                f"psf={epoch_losses['psf']:.4f}"
            )
            self.logger.log_metric(
                epoch=epoch + 1,
                metrics={
                    "total": float(epoch_losses["total"]),
                    "mse": float(epoch_losses["mse"]),
                    "lpips": float(epoch_losses["lpips"]),
                    "psf": float(epoch_losses["psf"]),
                },
                lr={
                    "lens": float(lens_scheduler.get_last_lr()[0])
                    if lens_scheduler is not None
                    else 0.0,
                    "network": float(network_scheduler.get_last_lr()[0])
                    if network_scheduler is not None
                    else 0.0,
                },
                extra={
                    "freeze_doe": freeze_doe,
                    "freeze_geolens": freeze_geolens,
                    "freeze_network": freeze_network,
                    "wavelength": wavelength,
                    "training_mode": training_mode,
                    "optics_grad_batches": optics_grad_batches,
                },
            )
            stop_training = False
            if not monitor_key.startswith("val_"):
                monitor_val = float(
                    epoch_losses.get(monitor_key, epoch_losses["total"])
                )
                stop_training = early_stopper.update(
                    monitor_val, step=epoch + 1
                )

            # Periodic evaluation and saving
            if (epoch + 1) % test_per_epoch == 0:
                should_save_network = (save_network_every is None) or (
                    (epoch + 1) % save_network_every == 0
                )
                self._save_checkpoint(
                    epoch + 1,
                    psf_size,
                    spp,
                    wavelength,
                    save_network=should_save_network,
                )

                if val_loader:
                    val_loss = self._validate(val_loader, psf_size, spp, wavelength)
                    loss_history["val_mse"].append(float(val_loss))
                    print(f"Validation MSE: {val_loss:.4f}")
                    self.logger.log_metric(
                        epoch=epoch + 1,
                        metrics={"val_mse": float(val_loss)},
                        extra={"substage": "validation"},
                    )
                    is_validation_best = (
                        best_validation_mse is None
                        or val_loss < best_validation_mse
                    )
                    if is_validation_best:
                        best_validation_mse = float(val_loss)
                        best_validation_epoch = epoch + 1
                        if save_validation_best:
                            self._save_checkpoint(
                                "best", psf_size, spp, wavelength, save_network=True
                            )
                        self.logger.log_event(
                            event="validation_best",
                            payload={
                                "epoch": epoch + 1,
                                "val_mse": float(val_loss),
                                "checkpoint_saved": bool(save_validation_best),
                            },
                        )
                    if monitor_key == "val_mse":
                        stop_training = early_stopper.update(
                            float(val_loss), step=epoch + 1
                        )

            if stop_training:
                self.logger.log_event(
                    event="early_stop_triggered",
                    payload={
                        "stop_epoch": epoch + 1,
                        "best_metric": early_stopper.best,
                        "metric": monitor_key,
                    },
                )
                print(
                    f"Early stopping at epoch {epoch + 1}, "
                    f"best {monitor_key}={early_stopper.best:.6f}"
                )
                break

        # Save final model
        self._save_checkpoint("final", psf_size, spp, wavelength)

        print("\n" + "=" * 60)
        print("End-to-end optimization complete!")
        print(f"Initial total loss: {loss_history['total'][0]:.6f}")
        print(f"Final total loss: {loss_history['total'][-1]:.6f}")
        improvement = (1 - loss_history["total"][-1] / loss_history["total"][0]) * 100
        print(f"Improvement: {improvement:.1f}%")
        print("=" * 60)

        doe_state_after = self._capture_doe_state()
        doe_change_max = 0.0
        for key, before_tensor in doe_state_before.items():
            if key in doe_state_after:
                after_tensor = doe_state_after[key]
                if before_tensor.shape == after_tensor.shape:
                    diff = (after_tensor - before_tensor).abs().max().item()
                    doe_change_max = max(doe_change_max, float(diff))

        self.logger.flush_summary(
            {
                "initial_total_loss": float(loss_history["total"][0])
                if loss_history["total"]
                else None,
                "final_total_loss": float(loss_history["total"][-1])
                if loss_history["total"]
                else None,
                "improvement_percent": float(improvement),
                "freeze_doe": freeze_doe,
                "freeze_geolens": freeze_geolens,
                "freeze_network": freeze_network,
                "wavelength": float(wavelength),
                "doe_change_max_abs": float(doe_change_max),
                "training_mode": training_mode,
                "optics_grad_batches": int(optics_grad_batches),
                "network_type": type(network).__name__,
                "color_mode": color_mode,
                "best_validation_mse": best_validation_mse,
                "best_validation_epoch": best_validation_epoch,
                "validation_best_saved": bool(save_validation_best),
                "input_hybridlens_snapshot": input_snapshot,
            }
        )

        return loss_history

    def _validate(
        self,
        val_loader,
        psf_size: int,
        spp: int,
        wavelength: float,
    ) -> float:
        """
        Validate the model on validation set.
        在验证集上验证模型
        """
        self.network.eval()
        total_loss = 0
        num_batches = 0

        mse_loss_fn = nn.MSELoss()

        with torch.no_grad():
            # Calculate PSF once for validation
            psf = self.hybrid_lens.psf(
                points=[0.0, 0.0, -10000.0],
                ks=psf_size,
                wvln=wavelength,
                spp=max(int(spp), 1_000_000),
            )

            for batch in val_loader:
                if isinstance(batch, dict):
                    clean_images = batch["img"].to(self.device)
                else:
                    clean_images = batch.to(self.device)

                # Simulate and reconstruct
                degraded_images = self._simulate_image(clean_images.float(), psf)
                reconstructed_images = self.network(degraded_images.float())

                # Calculate MSE
                loss = mse_loss_fn(reconstructed_images, clean_images.float())
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def _save_checkpoint(
        self,
        epoch,
        psf_size: int,
        spp: int,
        wavelength: float,
        save_network: bool = True,
    ):
        """Save checkpoint including lens, network, and visualizations."""
        try:
            # Save network weights (conditionally)
            if save_network:
                torch.save(
                    self.network.state_dict(),
                    str(self.result_dir / f"network_epoch{epoch}.pth"),
                )

            # Save lens configuration
            self.hybrid_lens.write_lens_json(
                str(self.result_dir / f"hybridlens_epoch{epoch}.json")
            )

            # Save PSF visualization
            with torch.no_grad():
                psf = self.hybrid_lens.psf(
                    points=[0.0, 0.0, -10000.0],
                    ks=psf_size,
                    wvln=wavelength,
                    spp=max(int(spp), 1_000_000),
                )
                psf_vis = psf / psf.max()
                save_image(
                    psf_vis.unsqueeze(0).float(),
                    str(self.result_dir / f"psf_epoch{epoch}.png"),
                )

        except Exception as e:
            print(f"Warning: Failed to save checkpoint: {e}")

    def get_hybrid_lens(self) -> HybridLens:
        """Get the trained hybrid lens."""
        return self.hybrid_lens

    def get_network(self) -> nn.Module:
        """Get the trained neural network."""
        return self.network

    def save_lens(self, filename: str) -> None:
        """Save lens to file."""
        self.hybrid_lens.write_lens_json(filename)

    def save_network(self, filename: str) -> None:
        """Save network weights to file."""
        torch.save(self.network.state_dict(), filename)

    def load_network(self, filename: str) -> None:
        """Load network weights from file."""
        self.network.load_state_dict(
            torch.load(filename, map_location=self.device, weights_only=True)
        )
        print(f"Loaded network weights from: {filename}")


def create_e2e_trainer(
    geolens_path: str,
    result_dir: str,
    doe_type: str = "binary2",
    network_type: str = "unet",
    network_params: Optional[Dict[str, Any]] = None,
    freeze_doe: bool = True,
    wavelength: float = 0.55,
    device: torch.device = None,
) -> E2ETrainer:
    """
    Convenience function to create E2E trainer from GeoLens.
    从 GeoLens 创建端到端训练器的便捷函数

    Args:
        geolens_path: Path to optimized GeoLens JSON file
        result_dir: Directory to save results
        doe_type: DOE type ("binary2" or "pixel2d")
        network_type: Network type ("unet", "nafnet", "restormer")
        network_params: Optional constructor parameters for the network
        freeze_doe: Whether to freeze DOE in E2E stage
        wavelength: Wavelength for PSF/rendering in E2E stage
        device: PyTorch device

    Returns:
        E2ETrainer instance
    """
    from src.hybridlens.trainer import HybridLensTrainer

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    # First create HybridLens using HybridLensTrainer
    doe_config = {
        "type": doe_type,
        "res": [1000, 1000],
        "fab_ps": 0.003,
        "is_square": True,
    }

    hybrid_trainer = HybridLensTrainer.from_geolens(
        geolens_path=geolens_path,
        result_dir=str(result_dir / "hybridlens"),
        doe_config=doe_config,
        device=device,
    )

    # Get the HybridLens
    hybrid_lens = hybrid_trainer.get_hybrid_lens()

    # Create network
    network_config = {
        "type": network_type,
        "in_channels": 3,
        "out_channels": 3,
    }
    if network_params is not None:
        network_config["params"] = network_params

    network = E2ETrainer._create_network(network_config)

    return E2ETrainer(
        hybrid_lens=hybrid_lens,
        network=network,
        result_dir=str(result_dir),
        freeze_doe=freeze_doe,
        wavelength=wavelength,
        device=device,
    )
