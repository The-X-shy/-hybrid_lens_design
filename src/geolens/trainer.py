"""
GeoLens Trainer - Curriculum Learning + Fine-tuning for Refractive Lens Design
折射透镜训练器 - 课程学习 + 微调优化

Based on: Xinge Yang, Qiang Fu, Wolfgang Heidrich,
"Curriculum learning for ab initio deep learned refractive optics,"
Nature Communications 2024.
"""

import math
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import torch
import numpy as np
from tqdm import tqdm

# DeepLens imports
from deeplens import GeoLens
from deeplens.geolens_pkg.utils import create_lens
from deeplens.basics import DEPTH, EPSILON, WAVE_RGB

from src.utils.stage_logger import StageLogger
from src.utils.config import get_nested
from src.utils.early_stop import EarlyStopper


class GeoLensTrainer:
    """
    Trainer for end-to-end refractive lens design using curriculum learning.
    使用课程学习的端到端折射透镜设计训练器
    """

    def __init__(
        self,
        lens: GeoLens,
        result_dir: str,
        mode: str = "scratch_two_stage",
        target_fnum: Optional[float] = None,
        device: torch.device = None,
    ):
        """
        Initialize the GeoLens trainer.

        Args:
            lens: GeoLens object to train
            result_dir: Directory to save results
            device: PyTorch device
        """
        self.lens = lens
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.target_fnum = target_fnum

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Move lens to device
        self.lens.to(self.device)
        self.logger = StageLogger(
            result_dir=str(self.result_dir),
            stage="geolens",
            mode=self.mode,
        )
        self._last_valid_lens_path: Optional[str] = None

    @classmethod
    def from_config(cls, config) -> "GeoLensTrainer":
        """
        Create trainer from configuration.
        从配置创建训练器
        """
        from src.utils.config import create_result_dir, get_device

        result_dir = create_result_dir(config)
        device = get_device(config)

        mode = get_nested(config, "training.mode", "scratch_two_stage")
        default_existing_lens = "/Users/lilin/Desktop/deeplens backup/results/tutorial_0207-184619/final_lens.json"
        existing_lens_path = get_nested(
            config, "training.existing_lens_path", default_existing_lens
        )

        # Create or load lens based on mode
        if mode in {"existing_two_stage", "existing_finetune_only"}:
            lens_path = existing_lens_path or get_nested(config, "lens.file_path")
            if not lens_path:
                raise ValueError(
                    f"Mode '{mode}' requires training.existing_lens_path (or lens.file_path)."
                )
            lens = GeoLens(filename=lens_path)
        elif config.lens.source == "file" and config.lens.file_path:
            lens = GeoLens(filename=config.lens.file_path)
        else:
            # Create lens from design parameters
            design = config.lens.design
            lens = create_lens(
                foclen=design.foclen,
                fov=design.fov,
                fnum=design.fnum,
                flange=design.flange,
                thickness=design.thickness,
                surf_list=design.lens_type,
                save_dir=str(result_dir),
            )

        # Ensure consistent dtype
        lens.astype(torch.float32)

        # Set target FOV and F-number
        target_fov = get_nested(config, "lens.design.fov")
        target_fnum = get_nested(config, "lens.design.fnum")
        if target_fov is None:
            target_fov = float(np.rad2deg(lens.rfov) * 2.0)
        if target_fnum is None:
            target_fnum = float(lens.fnum)

        lens.set_target_fov_fnum(
            rfov=target_fov / 2 / 57.3,  # Convert to radians
            fnum=target_fnum,
        )

        return cls(
            lens=lens,
            result_dir=str(result_dir),
            mode=mode,
            target_fnum=float(target_fnum),
            device=device,
        )

    def curriculum_design(
        self,
        lrs: List[float] = [1e-4, 1e-4, 1e-2, 1e-4],
        decay: float = 0.01,
        iterations: int = 2000,
        test_per_iter: int = 100,
        optim_mat: bool = False,
        match_mat: bool = False,
        shape_control: bool = True,
        early_stop_config: Optional[Dict[str, Any]] = None,
    ) -> List[float]:
        """
        Stage 1: Curriculum learning optimization.
        阶段1：课程学习优化

        Gradually increases aperture size and field of view during training.
        在训练过程中逐渐增加光圈大小和视场角。

        Args:
            lrs: Learning rates [d, c, k, ai] for thickness, curvature, conic, aspheric
            decay: Learning rate decay for higher-order coefficients
            iterations: Number of training iterations
            test_per_iter: Evaluation interval
            optim_mat: Whether to optimize materials
            match_mat: Whether to match materials
            shape_control: Whether to enable shape control

        Returns:
            List of loss values during training
        """
        self.logger.log_event(
            event="curriculum_start",
            payload={
                "iterations": iterations,
                "test_per_iter": test_per_iter,
                "lrs": lrs,
                "decay": decay,
                "optim_mat": optim_mat,
                "match_mat": match_mat,
                "shape_control": shape_control,
            },
        )

        # Preparation
        depth = DEPTH
        num_ring = 8
        num_arm = 8
        spp = 2048

        aper_start = self.lens.surfaces[self.lens.aper_idx].r * 0.2
        aper_final = self.lens.surfaces[self.lens.aper_idx].r

        print(
            f"lr:{lrs}, iterations:{iterations}, spp:{spp}, num_ring:{num_ring}, num_arm:{num_arm}."
        )
        print(f"Aperture from {aper_start:.3f}mm to {aper_final:.3f}mm")

        # Optimizer
        optimizer = self.lens.get_optimizer(lrs, decay=decay, optim_mat=optim_mat)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=iterations
        )
        early_stopper = self._build_early_stopper(early_stop_config, default_mode="min")
        if early_stopper.enabled:
            self.logger.log_event(
                event="early_stop_enabled",
                payload={
                    "substage": "curriculum",
                    "patience": early_stopper.patience,
                    "min_delta": early_stopper.min_delta,
                    "mode": early_stopper.mode,
                },
            )

        # Record loss history
        loss_history = []

        # Training loop
        pbar = tqdm(
            total=iterations + 1,
            desc="Curriculum Learning",
            postfix={"loss_rms": 0, "loss_reg": 0},
        )

        for i in range(iterations + 1):
            # =======================================
            # Evaluate the lens
            # =======================================
            if i % test_per_iter == 0:
                with torch.no_grad():
                    # Curriculum learning: gradually increase aperture size
                    progress = 0.5 * (1 + math.cos(math.pi * (1 - i / iterations)))
                    aper_r = min(
                        aper_start + (aper_final - aper_start) * progress,
                        aper_final,
                    )
                    self.lens.surfaces[self.lens.aper_idx].update_r(aper_r)
                    self.lens.calc_pupil()

                    # Correct lens shape and evaluate
                    if i > 0:
                        if shape_control:
                            self.lens.correct_shape()

                        if optim_mat and match_mat:
                            self.lens.match_materials()

                    # Save lens snapshot and keep a pointer to the latest finite state.
                    iter_path = f"{self.result_dir}/iter{i}.json"
                    self.lens.write_lens_json(iter_path)
                    if self._is_lens_valid():
                        self._last_valid_lens_path = iter_path
                    try:
                        self.lens.analysis(f"{self.result_dir}/iter{i}")
                    except Exception as exc:
                        self.logger.log_event(
                            event="analysis_failed",
                            payload={
                                "substage": "curriculum",
                                "step": i,
                                "error": str(exc),
                            },
                        )

                    # Sample new rays
                    rays_backup = []
                    for wv in WAVE_RGB:
                        ray = self.lens.sample_ring_arm_rays(
                            num_ring=num_ring,
                            num_arm=num_arm,
                            depth=depth,
                            spp=spp,
                            wvln=wv,
                            scale_pupil=1.10,
                        )
                        rays_backup.append(ray)

                    center_ref = -self.lens.psf_center(
                        points_obj=ray.o[:, :, 0, :], method="pinhole"
                    )
                    center_ref = center_ref.unsqueeze(-2).repeat(1, 1, spp, 1)

            # =======================================
            # Optimize lens by minimizing RMS
            # =======================================
            loss_rms = []
            for wv_idx, wv in enumerate(WAVE_RGB):
                ray = rays_backup[wv_idx].clone()
                ray = self.lens.trace2sensor(ray)

                ray_xy = ray.o[..., :2]
                ray_valid = ray.is_valid
                ray_err = ray_xy - center_ref

                # Weight mask
                if wv_idx == 0:
                    with torch.no_grad():
                        weight_mask = ((ray_err**2).sum(-1) * ray_valid).sum(-1)
                        weight_mask /= ray_valid.sum(-1) + EPSILON
                        weight_mask /= weight_mask.mean()

                        # Dropout (20%)
                        dropout_mask = torch.rand_like(weight_mask) < 0.2
                        weight_mask = weight_mask * (~dropout_mask)

                # RMS loss
                l_rms = (((ray_err**2).sum(-1) + EPSILON).sqrt() * ray_valid).sum(-1)
                l_rms /= ray_valid.sum(-1) + EPSILON

                # Weighted loss
                l_rms_weighted = (l_rms * weight_mask).sum()
                l_rms_weighted /= weight_mask.sum() + EPSILON
                loss_rms.append(l_rms_weighted)

            loss_rms = sum(loss_rms) / len(loss_rms)

            # Add regularization
            loss_reg, loss_dict = self.lens.loss_reg()
            w_reg = 0.05
            L_total = loss_rms + w_reg * loss_reg
            if not torch.isfinite(loss_rms) or not torch.isfinite(L_total):
                self.logger.log_event(
                    event="nan_detected",
                    payload={
                        "substage": "curriculum",
                        "step": i,
                        "loss_rms": float(loss_rms.item())
                        if torch.isfinite(loss_rms)
                        else None,
                        "loss_total": float(L_total.item())
                        if torch.isfinite(L_total)
                        else None,
                    },
                )
                print(f"[WARN] NaN/Inf detected in curriculum at step {i}, stop stage.")
                break

            # Optimize
            optimizer.zero_grad()
            L_total.backward()
            optimizer.step()
            scheduler.step()

            loss_history.append(loss_rms.item())
            if i % test_per_iter == 0 or i == iterations:
                current_lr = scheduler.get_last_lr()[0]
                self.logger.log_metric(
                    step=i,
                    metrics={
                        "loss_rms": float(loss_rms.item()),
                        "loss_total": float(L_total.item()),
                        "loss_reg": float(loss_reg.item()),
                    },
                    lr=current_lr,
                )
            if early_stopper.update(float(loss_rms.item()), step=i):
                self.logger.log_event(
                    event="early_stop_triggered",
                    payload={
                        "substage": "curriculum",
                        "stop_step": i,
                        "best_metric": early_stopper.best,
                        "metric": "loss_rms",
                    },
                )
                break
            pbar.set_postfix(loss_rms=loss_rms.item(), **loss_dict)
            pbar.update(1)

        pbar.close()
        self.logger.log_event(
            event="curriculum_end",
            payload={
                "initial_rms": float(loss_history[0]) if loss_history else None,
                "final_rms": float(loss_history[-1]) if loss_history else None,
            },
        )
        return loss_history

    def finetune(
        self,
        lrs: List[float] = [1e-4, 1e-4, 1e-3, 1e-4],
        decay: float = 0.01,
        iterations: int = 1200,
        test_per_iter: int = 50,
        centroid: bool = False,
        optim_mat: bool = False,
        shape_control: bool = True,
        early_stop_config: Optional[Dict[str, Any]] = None,
    ) -> List[float]:
        """
        Stage 2: Fine-tuning optimization.
        阶段2：微调优化

        Notebook-aligned behavior:
        - Use native deeplens `lens.optimize(...)` for stage-2
        - Keep interface compatible with trainer config
        """
        self.logger.log_event(
            event="finetune_start",
            payload={
                "iterations": iterations,
                "test_per_iter": test_per_iter,
                "lrs": lrs,
                "decay": decay,
                "centroid": centroid,
                "optim_mat": optim_mat,
                "shape_control": shape_control,
                "implementation": "deeplens.optimize (notebook-aligned)",
            },
        )

        finetune_dir = self.result_dir / "fine-tune"
        finetune_dir.mkdir(exist_ok=True)

        # Keep notebook behavior exactly for stage-2 optimization.
        history = self.lens.optimize(
            lrs=lrs,
            decay=decay,
            iterations=iterations,
            test_per_iter=test_per_iter,
            centroid=centroid,
            optim_mat=optim_mat,
            shape_control=shape_control,
            result_dir=str(finetune_dir),
        )

        if history is None:
            history = []

        # Best-effort update of last valid snapshot pointer.
        final_iter_path = finetune_dir / f"iter{iterations}.json"
        if final_iter_path.exists() and self._is_lens_valid():
            self._last_valid_lens_path = str(final_iter_path)

        self.logger.log_event(
            event="finetune_end",
            payload={
                "initial_rms": float(history[0]) if history else None,
                "final_rms": float(history[-1]) if history else None,
            },
        )

        return history

    def train(
        self,
        curriculum_config: Dict[str, Any],
        finetune_config: Optional[Dict[str, Any]] = None,
        early_stop_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[float], List[float]]:
        """
        Run full training pipeline: curriculum learning + fine-tuning.
        运行完整训练流程：课程学习 + 微调

        Args:
            curriculum_config: Configuration for curriculum learning stage
            finetune_config: Configuration for fine-tuning stage (optional)

        Returns:
            Tuple of (curriculum_loss_history, finetune_loss_history)
        """
        self.logger.log_event(
            event="train_start",
            payload={"mode": self.mode},
        )

        curriculum_history = []
        if self.mode != "existing_finetune_only":
            print("=" * 60)
            print("Stage 1: Curriculum Learning Optimization")
            print("=" * 60)

            curriculum_kwargs = dict(curriculum_config)
            curriculum_es = (
                curriculum_kwargs.pop("early_stop", None) or early_stop_config
            )
            curriculum_history = self.curriculum_design(
                **curriculum_kwargs,
                early_stop_config=curriculum_es,
            )

            # Match materials and set fnum (align with notebook behavior)
            self.lens.match_materials()
            self.lens.set_fnum(
                float(self.target_fnum)
                if self.target_fnum is not None
                else float(self.lens.fnum)
            )

            print(f"\nStage 1 complete!")
            print(f"Initial RMS: {curriculum_history[0]:.6f} mm")
            print(f"Final RMS: {curriculum_history[-1]:.6f} mm")
            print(
                f"Improvement: {(1 - curriculum_history[-1] / curriculum_history[0]) * 100:.1f}%"
            )
        else:
            print("=" * 60)
            print("Stage 1 skipped (mode=existing_finetune_only)")
            print("=" * 60)
            self.logger.log_event(event="curriculum_skipped")

        finetune_history = []
        if finetune_config:
            print("\n" + "=" * 60)
            print("Stage 2: Fine-tuning Optimization")
            print("=" * 60)

            finetune_kwargs = dict(finetune_config)
            finetune_es = finetune_kwargs.pop("early_stop", None) or early_stop_config
            finetune_history = self.finetune(
                **finetune_kwargs,
                early_stop_config=finetune_es,
            )

            if finetune_history:
                print("\nStage 2 complete!")
                print(f"Final RMS: {finetune_history[-1]:.6f} mm")
        elif self.mode == "existing_finetune_only":
            raise ValueError(
                "Mode 'existing_finetune_only' requires a valid finetune_config."
            )

        # Final processing (restore last valid state first if needed)
        self._restore_last_valid_if_needed(context="before_final_processing")
        try:
            self.lens.prune_surf(expand_factor=0.05)
            self.lens.post_computation()
        except Exception as exc:
            self.logger.log_event(
                event="final_processing_failed",
                payload={"error": str(exc)},
            )
            restored = self._restore_last_valid_if_needed(
                context="after_final_processing_failure"
            )
            if restored:
                self.lens.prune_surf(expand_factor=0.05)
                self.lens.post_computation()

        # Save final lens
        self.lens.write_lens_json(f"{self.result_dir}/final_lens.json")
        self.lens.write_lens_zmx(f"{self.result_dir}/final_lens.zmx")
        if self._is_lens_valid():
            self._last_valid_lens_path = f"{self.result_dir}/final_lens.json"
            try:
                self.lens.analysis(f"{self.result_dir}/final_lens")
            except Exception as exc:
                self.logger.log_event(
                    event="analysis_failed",
                    payload={"substage": "final", "error": str(exc)},
                )
        else:
            self.logger.log_event(
                event="final_lens_invalid",
                payload={"path": f"{self.result_dir}/final_lens.json"},
            )

        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Focal length: {self.lens.foclen:.2f} mm")
        print(f"F-number: {self.lens.fnum:.2f}")
        print(f"Half FOV: {np.rad2deg(self.lens.rfov):.2f} deg")
        print(f"Results saved to: {self.result_dir}")

        final_rms = None
        if finetune_history:
            final_rms = float(finetune_history[-1])
        elif curriculum_history:
            final_rms = float(curriculum_history[-1])

        self.logger.flush_summary(
            {
                "mode": self.mode,
                "final_foclen": float(self.lens.foclen),
                "final_fnum": float(self.lens.fnum),
                "final_half_fov_deg": float(np.rad2deg(self.lens.rfov)),
                "curriculum_initial_rms": float(curriculum_history[0])
                if curriculum_history
                else None,
                "curriculum_final_rms": float(curriculum_history[-1])
                if curriculum_history
                else None,
                "finetune_initial_rms": float(finetune_history[0])
                if finetune_history
                else None,
                "finetune_final_rms": float(finetune_history[-1])
                if finetune_history
                else None,
                "final_rms": final_rms,
            }
        )

        return curriculum_history, finetune_history

    def _is_lens_valid(self) -> bool:
        try:
            scalar_vals = [
                float(self.lens.foclen),
                float(self.lens.fnum),
                float(self.lens.rfov),
            ]
            if not all(np.isfinite(v) for v in scalar_vals):
                return False

            for surf in getattr(self.lens, "surfaces", []):
                for attr in ("r", "d", "c", "k"):
                    if not hasattr(surf, attr):
                        continue
                    val = getattr(surf, attr)
                    if isinstance(val, torch.Tensor):
                        if not torch.isfinite(val).all():
                            return False
                    elif isinstance(val, (float, int, np.floating, np.integer)):
                        if not np.isfinite(float(val)):
                            return False

                ai = getattr(surf, "ai", None)
                if isinstance(ai, torch.Tensor) and not torch.isfinite(ai).all():
                    return False
        except Exception:
            return False
        return True

    def _restore_last_valid_if_needed(self, context: str) -> bool:
        if self._is_lens_valid():
            return False

        path = self._last_valid_lens_path
        if not path or not Path(path).exists():
            self.logger.log_event(
                event="restore_last_valid_failed",
                payload={"context": context, "path": path},
            )
            return False

        self.logger.log_event(
            event="restore_last_valid_lens",
            payload={"context": context, "path": path},
        )
        lens = GeoLens(filename=path)
        lens.astype(torch.float32)
        lens.to(self.device)
        self.lens = lens
        return True

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

    def get_lens(self) -> GeoLens:
        """Get the trained lens."""
        return self.lens

    def save_lens(self, filename: str) -> None:
        """Save lens to file."""
        self.lens.write_lens_json(filename)
