"""
Metasurface HybridLens Trainer - Pixel2D Refractive-Diffractive Lens Optimization
超表面混合透镜训练器 - Pixel2D 折射-衍射透镜优化

Based on: Xinge Yang et al., "End-to-End Hybrid Refractive-Diffractive Lens Design
with Differentiable Ray-Wave Model," SIGGRAPH Asia 2024.

Unlike Binary2 DOE which uses radial polynomial coefficients, Pixel2D represents
the phase map directly with each pixel as an independent parameter, enabling
more flexible metasurface designs.
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import torch
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image

# DeepLens imports
from deeplens.hybridlens import HybridLens
from deeplens.optics.loss import PSFLoss
from deeplens.basics import WAVE_RGB

from src.utils.stage_logger import StageLogger
from src.utils.early_stop import EarlyStopper


class MetasurfaceTrainer:
    """
    Trainer for hybrid refractive-diffractive lens design using Pixel2D metasurface.
    使用 Pixel2D 超表面的折衍混合透镜设计训练器

    Key differences from HybridLensTrainer (Binary2):
    - Uses Pixel2D DOE: direct phase map representation
    - Each pixel is an independent parameter (higher DoF)
    - Supports fabrication constraints and smoothness regularization
    """

    def __init__(
        self,
        hybrid_lens: HybridLens,
        result_dir: str,
        device: torch.device = None,
    ):
        """
        Initialize the Metasurface trainer.

        Args:
            hybrid_lens: HybridLens object with Pixel2D DOE to train
            result_dir: Directory to save results
            device: PyTorch device
        """
        self.hybrid_lens = hybrid_lens
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.logger = StageLogger(
            result_dir=str(self.result_dir),
            stage="metasurface",
            wavelength=WAVE_RGB,
        )

    @classmethod
    def from_geolens(
        cls,
        geolens_path: str,
        result_dir: str,
        metasurface_config: Dict[str, Any] = None,
        device: torch.device = None,
    ) -> "MetasurfaceTrainer":
        """
        Create trainer from an optimized GeoLens file by adding Pixel2D DOE.
        从优化后的 GeoLens 文件创建训练器并添加 Pixel2D DOE

        Args:
            geolens_path: Path to the optimized GeoLens JSON file
            result_dir: Directory to save results
            metasurface_config: Pixel2D metasurface configuration dict
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

        # Default metasurface config (Pixel2D) and merge with user-provided fields.
        default_meta_config = {
            "type": "pixel2d",
            "d": d_sensor - 0.5,  # DOE position (0.5mm before sensor)
            "res": [1000, 1000],
            "fab_ps": 0.003,  # 3um pixel size
            "is_square": True,
        }
        if metasurface_config is None:
            metasurface_config = default_meta_config
        else:
            merged = dict(default_meta_config)
            merged.update(dict(metasurface_config))
            if merged.get("d") is None:
                merged["d"] = d_sensor - 0.5
            metasurface_config = merged

        # Add DOE to lens data
        lens_data["DOE"] = metasurface_config
        if "d_sensor" not in lens_data and "(d_sensor)" in lens_data:
            lens_data["d_sensor"] = lens_data["(d_sensor)"]

        # Save hybrid lens config
        hybridlens_path = str(result_dir / "metasurface_init.json")
        with open(hybridlens_path, "w") as f:
            json.dump(lens_data, f, indent=4)

        # Set double precision for phase calculation
        torch.set_default_dtype(torch.float64)

        # Create HybridLens with Pixel2D DOE
        hybrid_lens = HybridLens(filename=hybridlens_path, dtype=torch.float64)

        # Refocus to infinity
        hybrid_lens.refocus(foc_dist=-10000.0)

        return cls(hybrid_lens=hybrid_lens, result_dir=str(result_dir), device=device)

    @classmethod
    def from_config(cls, config) -> "MetasurfaceTrainer":
        """
        Create trainer from configuration.
        从配置创建训练器
        """
        from src.utils.config import create_result_dir, get_device

        result_dir = create_result_dir(config)
        device = get_device(config)

        # Load GeoLens path
        geolens_path = config.lens.file_path

        # Metasurface config from yaml
        metasurface_config = {
            "type": "pixel2d",
            "res": config.metasurface.res,
            "fab_ps": config.metasurface.fab_ps,
            "is_square": config.metasurface.is_square,
        }

        return cls.from_geolens(
            geolens_path=geolens_path,
            result_dir=str(result_dir),
            metasurface_config=metasurface_config,
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
        smoothness_weight: float = 0.001,
        fabrication_weight: float = 0.01,
        early_stop_config: Optional[Dict[str, Any]] = None,
    ) -> List[float]:
        """
        Train the metasurface hybrid lens by optimizing Pixel2D DOE and refractive lens.
        通过优化 Pixel2D DOE 和折射透镜来训练超表面混合透镜

        Args:
            doe_lr: Learning rate for DOE phase map
            lens_lr: Learning rates [d, c, k, ai] for lens parameters
            lr_decay: Learning rate decay
            iterations: Number of training iterations
            test_per_iter: Evaluation interval
            spp: Samples per pixel for PSF calculation
            psf_size: PSF patch size
            smoothness_weight: Weight for phase map smoothness regularization
            fabrication_weight: Weight for fabrication constraint penalty
            early_stop_config: Early stop settings

        Returns:
            List of loss values during training
        """
        hybrid_lens = self.hybrid_lens

        # Create optimizer
        optimizer = hybrid_lens.get_optimizer(
            doe_lr=doe_lr,
            lens_lr=lens_lr,
            lr_decay=lr_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=iterations
        )

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
                "smoothness_weight": smoothness_weight,
                "fabrication_weight": fabrication_weight,
                "wavelengths": WAVE_RGB,
            },
        )

        # Training configuration
        print("=" * 60)
        print("Starting Metasurface (Pixel2D) Hybrid Lens Optimization")
        print("=" * 60)
        print(f"DOE type: Pixel2D (direct phase map)")
        print(f"DOE resolution: {hybrid_lens.doe.res}")
        print(f"SPP: {spp}")
        print(f"PSF size: {psf_size}x{psf_size}")
        print(f"Wavelengths: {WAVE_RGB}")
        print(f"Iterations: {iterations}")
        print(f"Smoothness weight: {smoothness_weight}")
        print(f"Fabrication weight: {fabrication_weight}")

        # Define field points for PSF evaluation
        field_points = [
            [0.0, 0.0, -10000.0],  # On-axis
            [0.5, 0.0, -10000.0],  # 50% field
            [0.0, 0.5, -10000.0],  # 50% field (y)
            [0.707, 0.0, -10000.0],  # 70.7% field
            [1.0, 0.0, -10000.0],  # Full field
        ]

        # Record loss history
        loss_history = []

        # Training loop
        pbar = tqdm(
            total=iterations + 1, desc="Metasurface Optimization", postfix={"loss": 0}
        )

        for i in range(iterations + 1):
            # Calculate PSF loss for each wavelength and field point
            total_loss = torch.tensor(0.0, device=self.device, dtype=torch.float64)
            valid_psf_count = 0

            for wvln in WAVE_RGB:
                for point in field_points:
                    # Calculate PSF
                    try:
                        psf = hybrid_lens.psf(
                            points=point,
                            ks=psf_size,
                            wvln=wvln,
                            spp=spp,
                        )
                    except Exception as exc:
                        self.logger.log_event(
                            event="psf_eval_skipped",
                            payload={
                                "step": i,
                                "wvln": float(wvln),
                                "point": [float(point[0]), float(point[1]), float(point[2])],
                                "error": str(exc),
                            },
                        )
                        continue

                    # Calculate loss (encourage compact PSF)
                    psf_input = psf.unsqueeze(0).unsqueeze(0)
                    # PSFLoss expects channel-wise comparison; when single-band
                    # input is used we replicate to three channels.
                    if psf_input.shape[1] == 1:
                        psf_input = psf_input.repeat(1, 3, 1, 1)
                    loss = loss_fn(psf_input)
                    total_loss += loss
                    valid_psf_count += 1

            # Average PSF loss
            if valid_psf_count == 0:
                self.logger.log_event(
                    event="psf_eval_failed_all",
                    payload={"step": i},
                )
                print(f"[WARN] No valid PSF samples at step {i}, stop metasurface stage.")
                break
            total_loss /= valid_psf_count

            # Add regularization terms for Pixel2D
            reg_loss = self._compute_regularization(
                smoothness_weight=smoothness_weight,
                fabrication_weight=fabrication_weight,
            )
            total_loss += reg_loss
            if not torch.isfinite(total_loss):
                self.logger.log_event(
                    event="nan_detected",
                    payload={"step": i, "loss": None},
                )
                print(f"[WARN] NaN/Inf detected in metasurface stage at step {i}, stop.")
                break

            # Backward and optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            # Record loss
            loss_history.append(total_loss.item())
            pbar.set_postfix(loss=f"{total_loss.item():.4f}")
            pbar.update(1)

            # Periodic evaluation and saving
            if i % test_per_iter == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"\nIter {i}: loss={total_loss.item():.6f}, lr={current_lr:.2e}")

                # Save PSF visualization
                self._save_psf_visualization(i, psf_size, spp)

                # Save phase map visualization
                self._save_phase_map(i)

                # Save lens configuration
                if i % (test_per_iter * 10) == 0:
                    self._save_lens(i)
                self.logger.log_metric(
                    step=i,
                    metrics={"loss": float(total_loss.item())},
                    lr=float(current_lr),
                    extra={
                        "smoothness_weight": float(smoothness_weight),
                        "fabrication_weight": float(fabrication_weight),
                    },
                )
            if early_stopper.update(float(total_loss.item()), step=i):
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

        # Save final lens and phase map
        self._save_lens("final")
        self._save_phase_map("final")
        try:
            hybrid_lens.analysis(save_name=str(self.result_dir / "metasurface_final"))
        except Exception as exc:
            self.logger.log_event(
                event="analysis_failed",
                payload={"substage": "final", "error": str(exc)},
            )
            print(f"Warning: metasurface final analysis failed: {exc}")

        print("\n" + "=" * 60)
        print("Metasurface hybrid lens optimization complete!")
        if loss_history:
            print(f"Initial loss: {loss_history[0]:.6f}")
            print(f"Final loss: {loss_history[-1]:.6f}")
            if loss_history[0] != 0:
                print(f"Improvement: {(1 - loss_history[-1] / loss_history[0]) * 100:.1f}%")
        else:
            print("No valid optimization steps were recorded.")
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
                "wavelengths": WAVE_RGB,
                "smoothness_weight": float(smoothness_weight),
                "fabrication_weight": float(fabrication_weight),
            }
        )

        return loss_history

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

    def _compute_regularization(
        self,
        smoothness_weight: float = 0.001,
        fabrication_weight: float = 0.01,
    ) -> torch.Tensor:
        """
        Compute regularization terms for Pixel2D phase map.
        计算 Pixel2D 相位图的正则化项

        Args:
            smoothness_weight: Weight for smoothness regularization (encourages smooth phase)
            fabrication_weight: Weight for fabrication penalty (phase within [0, 2π])

        Returns:
            Total regularization loss
        """
        doe = self.hybrid_lens.doe
        phase_map = doe.phase_func()

        reg_loss = torch.tensor(0.0, device=self.device, dtype=torch.float64)

        # Smoothness regularization (Total Variation)
        if smoothness_weight > 0:
            # Compute gradients
            dx = phase_map[:, 1:] - phase_map[:, :-1]
            dy = phase_map[1:, :] - phase_map[:-1, :]

            # Total variation loss
            tv_loss = (dx.abs().mean() + dy.abs().mean()) / 2
            reg_loss += smoothness_weight * tv_loss

        # Fabrication constraint (phase should be within [0, 2π])
        if fabrication_weight > 0:
            # Wrap phase to [0, 2π] and penalize deviation
            phase_wrapped = torch.remainder(phase_map, 2 * np.pi)
            # Penalize very rapid phase changes that are hard to fabricate
            phase_diff_x = (phase_wrapped[:, 1:] - phase_wrapped[:, :-1]).abs()
            phase_diff_y = (phase_wrapped[1:, :] - phase_wrapped[:-1, :]).abs()

            # Penalize jumps larger than π (wrapped phase discontinuities)
            jump_penalty_x = torch.relu(phase_diff_x - np.pi).mean()
            jump_penalty_y = torch.relu(phase_diff_y - np.pi).mean()

            reg_loss += fabrication_weight * (jump_penalty_x + jump_penalty_y)

        return reg_loss

    def _save_psf_visualization(self, iteration, psf_size: int, spp: int):
        """Save PSF visualization for current iteration."""
        try:
            with torch.no_grad():
                # Calculate on-axis PSF for each wavelength
                for wvln_idx, wvln in enumerate(WAVE_RGB):
                    psf = self.hybrid_lens.psf(
                        points=[0.0, 0.0, -10000.0],
                        ks=psf_size,
                        wvln=wvln,
                        spp=spp,
                    )
                    # Normalize for visualization
                    psf_vis = psf / psf.max()
                    save_image(
                        psf_vis.unsqueeze(0).float(),
                        str(
                            self.result_dir / f"psf_iter{iteration}_wvln{wvln_idx}.png"
                        ),
                    )
        except Exception as e:
            print(f"Warning: Failed to save PSF visualization: {e}")

    def _save_phase_map(self, iteration):
        """Save phase map visualization for current iteration."""
        try:
            with torch.no_grad():
                phase_map = self.hybrid_lens.doe.phase_func()

                # Normalize to [0, 1] for visualization
                phase_wrapped = torch.remainder(phase_map, 2 * np.pi)
                phase_vis = phase_wrapped / (2 * np.pi)

                save_image(
                    phase_vis.unsqueeze(0).unsqueeze(0).float(),
                    str(self.result_dir / f"phase_map_iter{iteration}.png"),
                )

                # Also save the raw phase map
                torch.save(
                    phase_map.clone().detach().cpu(),
                    str(self.result_dir / f"phase_map_iter{iteration}.pt"),
                )
        except Exception as e:
            print(f"Warning: Failed to save phase map: {e}")

    def _save_lens(self, iteration):
        """Save lens configuration for current iteration."""
        try:
            lens_path = str(self.result_dir / f"metasurface_iter{iteration}.json")
            phase_map_path = str(self.result_dir / f"phase_map_iter{iteration}.pt")
            self._write_hybrid_json_with_phase_map(lens_path, phase_map_path)
        except Exception as e:
            print(f"Warning: Failed to save lens: {e}")

    def _write_hybrid_json_with_phase_map(self, lens_path: str, phase_map_path: str) -> None:
        """Write hybrid lens JSON with explicit phase-map path for Pixel2D."""
        geolens = self.hybrid_lens.geolens
        data: Dict[str, Any] = {}
        data["info"] = geolens.lens_info if hasattr(geolens, "lens_info") else "None"
        data["foclen"] = round(float(geolens.foclen), 4)
        data["fnum"] = round(float(geolens.fnum), 4)
        data["r_sensor"] = round(float(geolens.r_sensor), 4)
        data["d_sensor"] = round(float(geolens.d_sensor.item()), 4)
        data["sensor_size"] = [round(float(i), 4) for i in geolens.sensor_size]
        data["sensor_res"] = geolens.sensor_res

        data["surfaces"] = []
        for idx, surf in enumerate(geolens.surfaces[:-1]):
            surf_dict = surf.surf_dict()
            if idx < len(geolens.surfaces) - 2:
                d_next = geolens.surfaces[idx + 1].d.item() - geolens.surfaces[idx].d.item()
            else:
                d_next = geolens.d_sensor.item() - geolens.surfaces[idx].d.item()
            surf_dict["d_next"] = round(float(d_next), 3)
            data["surfaces"].append(surf_dict)

        data["DOE"] = self.hybrid_lens.doe.surf_dict(phase_map_path)
        with open(lens_path, "w") as f:
            json.dump(data, f, indent=4)

    def get_hybrid_lens(self) -> HybridLens:
        """Get the trained hybrid lens."""
        return self.hybrid_lens

    def save_lens(self, filename: str) -> None:
        """Save lens to file."""
        self.hybrid_lens.write_lens_json(filename)

    def export_phase_map(self, filename: str) -> None:
        """Export the trained phase map to a file."""
        phase_map = self.hybrid_lens.doe.phase_func().clone().detach().cpu()
        torch.save(phase_map, filename)
        print(f"Phase map saved to: {filename}")
