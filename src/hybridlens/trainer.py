"""
HybridLens Trainer - DOE (Binary2) Refractive-Diffractive Lens Optimization
折衍混合透镜训练器 - DOE (Binary2) 折射-衍射透镜优化

Based on: Xinge Yang et al., "End-to-End Hybrid Refractive-Diffractive Lens Design
with Differentiable Ray-Wave Model," SIGGRAPH Asia 2024.
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.utils import save_image

# DeepLens imports
from deeplens.hybridlens import HybridLens
from deeplens.optics.loss import PSFLoss
from deeplens.basics import DEFAULT_WAVE

from src.utils.stage_logger import StageLogger
from src.utils.early_stop import EarlyStopper


class HybridLensTrainer:
    """
    Trainer for hybrid refractive-diffractive lens design using Binary2 DOE.
    使用 Binary2 DOE 的折衍混合透镜设计训练器
    """

    def __init__(
        self,
        hybrid_lens: HybridLens,
        result_dir: str,
        wavelengths: Optional[List[float]] = None,
        device: torch.device = None,
    ):
        """
        Initialize the HybridLens trainer.

        Args:
            hybrid_lens: HybridLens object to train
            result_dir: Directory to save results
            device: PyTorch device
        """
        self.hybrid_lens = hybrid_lens
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.wavelengths = wavelengths if wavelengths is not None else [0.55]

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.logger = StageLogger(
            result_dir=str(self.result_dir),
            stage="hybridlens",
            wavelength=self.wavelengths,
        )

    @classmethod
    def from_geolens(
        cls,
        geolens_path: str,
        result_dir: str,
        doe_config: Dict[str, Any] = None,
        wavelengths: Optional[List[float]] = None,
        device: torch.device = None,
    ) -> "HybridLensTrainer":
        """
        Create trainer from an optimized GeoLens file by adding DOE.
        从优化后的 GeoLens 文件创建训练器并添加 DOE

        Args:
            geolens_path: Path to the optimized GeoLens JSON file
            result_dir: Directory to save results
            doe_config: DOE configuration dict
            device: PyTorch device
        """
        from src.utils.config import get_device

        if device is None:
            device = get_device(None)

        result_dir = Path(result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)

        # Load GeoLens data
        with open(geolens_path, "r") as f:
            lens_data = json.load(f)

        # Get sensor distance
        d_sensor_key = "d_sensor" if "d_sensor" in lens_data else "(d_sensor)"
        d_sensor = lens_data.get(d_sensor_key, 10.0)

        # Default DOE config (merge user config with required fields)
        default_doe_config = {
            "type": "binary2",
            "d": d_sensor - 0.5,  # DOE position
            "res": [1000, 1000],
            "fab_ps": 0.003,
            "is_square": True,
            "param_model": "binary2",
            "order2": 0.0,
            "order4": 0.0,
            "order6": 0.0,
            "order8": 0.0,
        }
        if doe_config is None:
            doe_config = default_doe_config
        else:
            merged = dict(default_doe_config)
            merged.update(dict(doe_config))
            if merged.get("d") is None:
                merged["d"] = d_sensor - 0.5
            doe_config = merged

        # Add DOE to lens data
        lens_data["DOE"] = doe_config
        if "d_sensor" not in lens_data and "(d_sensor)" in lens_data:
            lens_data["d_sensor"] = lens_data["(d_sensor)"]

        # Save hybrid lens config
        hybridlens_path = str(result_dir / "hybridlens_init.json")
        with open(hybridlens_path, "w") as f:
            json.dump(lens_data, f, indent=4)

        # Set double precision for phase calculation
        torch.set_default_dtype(torch.float64)

        # Create HybridLens
        hybrid_lens = HybridLens(filename=hybridlens_path, dtype=torch.float64)

        # Refocus to infinity
        hybrid_lens.refocus(foc_dist=-10000.0)

        return cls(
            hybrid_lens=hybrid_lens,
            result_dir=str(result_dir),
            wavelengths=wavelengths,
            device=device,
        )

    @classmethod
    def from_config(cls, config) -> "HybridLensTrainer":
        """
        Create trainer from configuration.
        从配置创建训练器
        """
        from src.utils.config import create_result_dir, get_device

        result_dir = create_result_dir(config)
        device = get_device(config)

        # Load GeoLens path
        geolens_path = config.lens.file_path

        # DOE config from yaml
        doe_config = {
            "type": config.doe.type,
            "res": config.doe.res,
            "fab_ps": config.doe.fab_ps,
            "is_square": config.doe.is_square,
            "param_model": config.doe.param_model,
        }
        wavelengths = getattr(config.optimization, "wavelengths", [0.55])

        return cls.from_geolens(
            geolens_path=geolens_path,
            result_dir=str(result_dir),
            doe_config=doe_config,
            wavelengths=wavelengths,
            device=device,
        )

    def setup_sensor(
        self, match_aperture: bool = True, sensor_res: Tuple[int, int] = (2000, 2000)
    ):
        """
        Setup sensor size to match aperture.
        设置传感器大小以匹配光圈
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

    def train(
        self,
        doe_lr: float = 0.1,
        lens_lr: List[float] = [1e-4, 1e-4, 1e-2, 1e-5],
        lr_decay: float = 0.01,
        iterations: int = 2000,
        test_per_iter: int = 50,
        spp: int = 1000000,
        psf_size: int = 101,
        wavelengths: Optional[List[float]] = None,
        wavelength_weights: Optional[List[float]] = None,
        early_stop_config: Optional[Dict[str, Any]] = None,
    ) -> List[float]:
        """
        Train the hybrid lens by optimizing DOE and refractive lens jointly.
        通过联合优化 DOE 和折射透镜来训练混合透镜

        Args:
            doe_lr: Learning rate for DOE parameters
            lens_lr: Learning rates [d, c, k, ai] for lens parameters
            lr_decay: Learning rate decay
            iterations: Number of training iterations
            test_per_iter: Evaluation interval
            spp: Samples per pixel for PSF calculation
            psf_size: PSF patch size
            wavelengths: Optimization wavelengths, default [0.55]
            wavelength_weights: Weight for each wavelength, default equal weights
            early_stop_config: Early stop settings

        Returns:
            List of loss values during training
        """
        hybrid_lens = self.hybrid_lens
        wavelengths = list(wavelengths if wavelengths is not None else self.wavelengths)
        if not wavelengths:
            wavelengths = [0.55]
        if wavelength_weights is None:
            wavelength_weights = [1.0 for _ in wavelengths]
        if len(wavelength_weights) != len(wavelengths):
            raise ValueError("wavelength_weights length must match wavelengths length")
        weight_sum = float(sum(wavelength_weights))
        if weight_sum <= 0:
            raise ValueError("Sum of wavelength_weights must be > 0")
        wavelength_weights = [float(w) / weight_sum for w in wavelength_weights]
        self.wavelengths = wavelengths
        self.logger.wavelength = wavelengths

        # Create optimizer
        optimizer = hybrid_lens.get_optimizer(
            doe_lr=doe_lr,
            lens_lr=lens_lr,
            lr_decay=lr_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=iterations
        )
        all_params = []
        for group in optimizer.param_groups:
            params = group.get("params", [])
            if isinstance(params, torch.Tensor):
                params = [params]
            all_params.extend(params)

        # Loss function
        loss_fn = PSFLoss()
        early_stopper = self._build_early_stopper(early_stop_config, default_mode="min")
        if early_stopper.enabled:
            self.logger.log_event(
                event="early_stop_enabled",
                payload={
                    "patience": early_stopper.patience,
                    "min_delta": early_stopper.min_delta,
                    "mode": early_stopper.mode,
                    "monitor": (early_stop_config or {}).get("monitor", "loss"),
                },
            )

        self.logger.log_event(
            event="train_start",
            payload={
                "iterations": iterations,
                "test_per_iter": test_per_iter,
                "doe_lr": doe_lr,
                "lens_lr": lens_lr,
                "lr_decay": lr_decay,
                "spp": spp,
                "psf_size": psf_size,
                "wavelengths": wavelengths,
                "wavelength_weights": wavelength_weights,
            },
        )

        # Training configuration
        print("=" * 60)
        print("Starting Hybrid Refractive-Diffractive Lens Optimization")
        print("=" * 60)
        print(f"SPP: {spp}")
        print(f"PSF size: {psf_size}x{psf_size}")
        print(f"Wavelengths: {wavelengths}")
        print(f"Wavelength weights: {wavelength_weights}")
        print(f"Iterations: {iterations}")

        # Record loss history
        loss_history = []

        # Training loop
        pbar = tqdm(
            total=iterations + 1, desc="Hybrid Optimization", postfix={"loss": 0}
        )

        for i in range(iterations + 1):
            optimizer.zero_grad()

            total_loss_value = 0.0
            psf_for_save = None

            # Sequential wavelength optimization (memory efficient, notebook-aligned)
            for wvln, weight in zip(wavelengths, wavelength_weights):
                psf = hybrid_lens.psf(
                    points=[0.0, 0.0, -10000.0],
                    ks=psf_size,
                    wvln=wvln,
                    spp=spp,
                )

                psf_input = psf.unsqueeze(0).unsqueeze(0)
                # PSFLoss computes cross-channel consistency; for single-band
                # optimization we replicate to 3 channels to avoid zero-division.
                if psf_input.shape[1] == 1:
                    psf_input = psf_input.repeat(1, 3, 1, 1)
                loss = loss_fn(psf_input) * weight
                loss.backward()
                total_loss_value += float(loss.item())

                if abs(float(wvln) - 0.55) < 1e-8:
                    psf_for_save = psf.detach().cpu().clone()

                del psf, psf_input, loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if all_params:
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # Record loss
            loss_history.append(total_loss_value)
            pbar.set_postfix(loss=f"{total_loss_value:.4f}")
            pbar.update(1)

            # Periodic evaluation and saving
            if i % test_per_iter == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"\nIter {i}: loss={total_loss_value:.6f}, lr={current_lr:.2e}")

                # Save PSF visualization
                vis_wvln = 0.55 if 0.55 in wavelengths else float(wavelengths[0])
                self._save_psf_visualization(
                    i,
                    psf_size=psf_size,
                    spp=spp,
                    wavelength=vis_wvln,
                    psf_tensor=psf_for_save,
                )

                # Save lens configuration (both names for compatibility)
                hybrid_lens.write_lens_json(str(self.result_dir / f"hybrid_iter{i}.json"))
                hybrid_lens.write_lens_json(
                    str(self.result_dir / f"hybridlens_iter{i}.json")
                )
                self.logger.log_metric(
                    step=i,
                    metrics={"loss": float(total_loss_value)},
                    lr=float(current_lr),
                    extra={
                        "wavelengths": wavelengths,
                        "wavelength_weights": wavelength_weights,
                    },
                )
            if early_stopper.update(float(total_loss_value), step=i):
                self.logger.log_event(
                    event="early_stop_triggered",
                    payload={
                        "stop_step": i,
                        "best_metric": early_stopper.best,
                        "metric": "loss",
                    },
                )
                break

        pbar.close()

        # Save final lens (both names for compatibility)
        hybrid_lens.write_lens_json(str(self.result_dir / "hybrid_final.json"))
        hybrid_lens.write_lens_json(str(self.result_dir / "hybridlens_final.json"))
        try:
            hybrid_lens.analysis(save_name=str(self.result_dir / "hybrid_final_analysis"))
        except Exception as exc:
            self.logger.log_event(
                event="analysis_failed",
                payload={"substage": "final", "error": str(exc)},
            )
            print(f"Warning: hybrid final analysis failed: {exc}")
        self._save_loss_curve(loss_history)

        print("\n" + "=" * 60)
        print("Hybrid lens optimization complete!")
        print(f"Initial loss: {loss_history[0]:.6f}")
        print(f"Final loss: {loss_history[-1]:.6f}")
        print(f"Improvement: {(1 - loss_history[-1] / loss_history[0]) * 100:.1f}%")
        print("=" * 60)

        self.logger.flush_summary(
            {
                "initial_loss": float(loss_history[0]) if loss_history else None,
                "final_loss": float(loss_history[-1]) if loss_history else None,
                "improvement_percent": float(
                    (1 - loss_history[-1] / loss_history[0]) * 100
                )
                if len(loss_history) >= 2 and loss_history[0] != 0
                else None,
                "wavelengths": wavelengths,
                "wavelength_weights": wavelength_weights,
            }
        )

        return loss_history

    def _save_psf_visualization(
        self,
        iteration: int,
        psf_size: int,
        spp: int,
        wavelength: float = DEFAULT_WAVE,
        psf_tensor: Optional[torch.Tensor] = None,
    ):
        """Save PSF visualization for current iteration."""
        try:
            with torch.no_grad():
                if psf_tensor is None:
                    psf = self.hybrid_lens.psf(
                        points=[0.0, 0.0, -10000.0],
                        ks=psf_size,
                        wvln=wavelength,
                        spp=spp,
                    )
                    psf_tensor = psf.detach().cpu()

                psf_vis = psf_tensor / (psf_tensor.max() + 1e-12)
                save_image(
                    psf_vis.unsqueeze(0).float(),
                    str(self.result_dir / f"hybrid_psf_iter{iteration}.png"),
                    normalize=True,
                )
        except Exception as e:
            print(f"Warning: Failed to save PSF visualization: {e}")

    def _save_loss_curve(self, loss_history: List[float]) -> None:
        if not loss_history:
            return
        plt.figure(figsize=(10, 5), facecolor="white")
        plt.plot(loss_history, linewidth=1.5, color="green")
        plt.xlabel("Iteration", fontsize=12)
        plt.ylabel("PSF Loss", fontsize=12)
        plt.title("Hybrid Refractive-Diffractive Lens Optimization Loss Curve", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(str(self.result_dir / "hybrid_loss_curve.png"), dpi=150, facecolor="white")
        plt.close()

    @staticmethod
    def _build_early_stopper(
        early_stop_config: Optional[Dict[str, Any]],
        default_mode: str = "min",
    ) -> EarlyStopper:
        cfg = early_stop_config or {}
        mode = str(cfg.get("mode", default_mode))
        return EarlyStopper(
            enabled=bool(cfg.get("enabled", False)),
            patience=int(cfg.get("patience", 500)),
            min_delta=float(cfg.get("min_delta", 1e-6)),
            mode=mode,
        )

    def get_hybrid_lens(self) -> HybridLens:
        """Get the trained hybrid lens."""
        return self.hybrid_lens

    def save_lens(self, filename: str) -> None:
        """Save lens to file."""
        self.hybrid_lens.write_lens_json(filename)
