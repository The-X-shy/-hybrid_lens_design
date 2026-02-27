"""
HybridLens Evaluator - Optical Performance Evaluation for Hybrid Refractive-Diffractive Lens
折衍混合透镜评估器 - 光学性能评估

Provides analysis tools for PSF, MTF, DOE phase, and other optical metrics
specific to hybrid lenses with DOE elements.
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image

# DeepLens imports
from deeplens import GeoLens
from deeplens.hybridlens import HybridLens
from deeplens.basics import DEPTH, WAVE_RGB, DEFAULT_WAVE, SPP_COHERENT


class HybridLensEvaluator:
    """
    Evaluator for hybrid refractive-diffractive lens optical performance.
    折衍混合透镜光学性能评估器
    """

    def __init__(
        self,
        hybrid_lens: HybridLens,
        result_dir: str,
        device: torch.device = None,
    ):
        """
        Initialize the evaluator.

        Args:
            hybrid_lens: HybridLens object to evaluate
            result_dir: Directory to save results
            device: PyTorch device
        """
        self.hybrid_lens = hybrid_lens
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

    @classmethod
    def from_file(cls, lens_path: str, result_dir: str) -> "HybridLensEvaluator":
        """
        Create evaluator from hybrid lens file.
        从混合透镜文件创建评估器
        """
        # Set double precision for phase calculation
        torch.set_default_dtype(torch.float64)

        hybrid_lens = HybridLens(filename=lens_path, dtype=torch.float64)
        return cls(hybrid_lens=hybrid_lens, result_dir=result_dir)

    def analyze_basic_params(self) -> Dict[str, float]:
        """
        Analyze basic lens parameters.
        分析基本透镜参数

        Returns:
            Dictionary with basic lens parameters
        """
        geolens = self.hybrid_lens.geolens
        doe = self.hybrid_lens.doe

        results = {
            # GeoLens parameters
            "foclen": geolens.foclen,
            "fnum": geolens.fnum,
            "rfov_deg": np.rad2deg(geolens.rfov),
            "r_sensor": geolens.r_sensor,
            "d_sensor": geolens.d_sensor.item()
            if hasattr(geolens.d_sensor, "item")
            else geolens.d_sensor,
            "num_surfaces": len(geolens.surfaces),
            # DOE parameters
            "doe_type": type(doe).__name__,
            "doe_resolution": list(doe.res),
            "doe_pixel_size_um": doe.ps * 1000,  # Convert mm to um
            "doe_distance": doe.d.item() if hasattr(doe.d, "item") else doe.d,
        }

        return results

    def compute_psf(
        self,
        field_positions: List[List[float]] = None,
        wavelengths: List[float] = None,
        spp: int = SPP_COHERENT,
        psf_size: int = 101,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute PSF at different field positions for the hybrid lens.
        计算混合透镜在不同视场位置的 PSF

        Args:
            field_positions: List of [x, y, z] coordinates for each field point
            wavelengths: Wavelengths to compute PSF for
            spp: Samples per pixel
            psf_size: PSF kernel size

        Returns:
            Dictionary with PSF tensors for each field position and wavelength
        """
        if wavelengths is None:
            wavelengths = WAVE_RGB

        if field_positions is None:
            field_positions = [
                [0.0, 0.0, -10000.0],  # On-axis
                [0.5, 0.0, -10000.0],  # 50% field
                [0.707, 0.0, -10000.0],  # 70.7% field
                [1.0, 0.0, -10000.0],  # Full field
            ]

        psfs = {}

        print("Computing PSFs for hybrid lens...")
        for i, point in enumerate(field_positions):
            field_name = f"field_{i}_x{point[0]:.2f}_y{point[1]:.2f}"
            psfs[field_name] = {}

            for wv_idx, wv in enumerate(wavelengths):
                with torch.no_grad():
                    psf = self.hybrid_lens.psf(
                        points=point,
                        ks=psf_size,
                        wvln=wv,
                        spp=spp,
                    )
                    psfs[field_name][f"wv_{wv_idx}_{wv}um"] = psf

        return psfs

    def visualize_psf(
        self,
        psfs: Dict[str, Dict[str, torch.Tensor]] = None,
        save_name: str = "psf_analysis",
    ) -> None:
        """
        Visualize PSFs and save to files.
        可视化 PSF 并保存到文件

        Args:
            psfs: Dictionary of PSFs (if None, will compute)
            save_name: Base name for saved files
        """
        if psfs is None:
            psfs = self.compute_psf()

        num_fields = len(psfs)
        num_wavelengths = len(list(psfs.values())[0])

        # Create figure
        fig, axes = plt.subplots(
            num_fields,
            num_wavelengths,
            figsize=(4 * num_wavelengths, 4 * num_fields),
        )
        if num_fields == 1:
            axes = axes.reshape(1, -1)
        if num_wavelengths == 1:
            axes = axes.reshape(-1, 1)

        for i, (field_name, wv_psfs) in enumerate(psfs.items()):
            for j, (wv_name, psf) in enumerate(wv_psfs.items()):
                ax = axes[i, j]
                psf_np = psf.cpu().numpy()

                # Log scale for better visualization
                psf_log = np.log10(psf_np + 1e-10)

                im = ax.imshow(psf_log, cmap="hot")
                ax.set_title(f"{field_name}\n{wv_name}")
                ax.axis("off")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(self.result_dir / f"{save_name}.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Also save individual PSF images
        for field_name, wv_psfs in psfs.items():
            for wv_name, psf in wv_psfs.items():
                psf_vis = psf / psf.max()
                save_image(
                    psf_vis.unsqueeze(0).float(),
                    str(self.result_dir / f"psf_{field_name}_{wv_name}.png"),
                )

        print(f"PSF visualization saved to: {self.result_dir / save_name}.png")

    def analyze_doe_phase(self, save_name: str = "doe_phase") -> Dict[str, Any]:
        """
        Analyze the DOE phase map.
        分析 DOE 相位图

        Args:
            save_name: Base name for saved files

        Returns:
            Dictionary with DOE phase analysis results
        """
        doe = self.hybrid_lens.doe
        results = {}

        with torch.no_grad():
            # Get phase map
            phase_map = doe.phase_func()
            phase_np = phase_map.cpu().numpy()

            # Basic statistics
            results["phase_min"] = float(phase_np.min())
            results["phase_max"] = float(phase_np.max())
            results["phase_mean"] = float(phase_np.mean())
            results["phase_std"] = float(phase_np.std())
            results["phase_range_pi"] = float((phase_np.max() - phase_np.min()) / np.pi)

            # Wrapped phase statistics (0 to 2π)
            phase_wrapped = np.mod(phase_np, 2 * np.pi)
            results["wrapped_phase_mean"] = float(phase_wrapped.mean())
            results["wrapped_phase_std"] = float(phase_wrapped.std())

            # Visualize phase map
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Raw phase
            im0 = axes[0].imshow(phase_np, cmap="twilight")
            axes[0].set_title("Raw Phase")
            axes[0].axis("off")
            plt.colorbar(im0, ax=axes[0], label="Phase (rad)")

            # Wrapped phase
            im1 = axes[1].imshow(phase_wrapped, cmap="twilight", vmin=0, vmax=2 * np.pi)
            axes[1].set_title("Wrapped Phase (0 to 2π)")
            axes[1].axis("off")
            plt.colorbar(im1, ax=axes[1], label="Phase (rad)")

            # Phase gradient magnitude (for fabricability analysis)
            dy, dx = np.gradient(phase_np)
            gradient_mag = np.sqrt(dx**2 + dy**2)
            results["max_gradient"] = float(gradient_mag.max())
            results["mean_gradient"] = float(gradient_mag.mean())

            im2 = axes[2].imshow(gradient_mag, cmap="viridis")
            axes[2].set_title("Phase Gradient Magnitude")
            axes[2].axis("off")
            plt.colorbar(im2, ax=axes[2], label="Gradient (rad/pixel)")

            plt.tight_layout()
            plt.savefig(
                self.result_dir / f"{save_name}.png", dpi=150, bbox_inches="tight"
            )
            plt.close()

            # Save raw phase map
            save_image(
                torch.from_numpy(phase_wrapped / (2 * np.pi))
                .unsqueeze(0)
                .unsqueeze(0)
                .float(),
                str(self.result_dir / f"{save_name}_wrapped.png"),
            )

        print(f"DOE phase analysis saved to: {self.result_dir / save_name}.png")
        return results

    def analyze_chromatic_aberration(
        self,
        wavelengths: List[float] = None,
        spp: int = SPP_COHERENT // 10,
        psf_size: int = 101,
    ) -> Dict[str, float]:
        """
        Analyze chromatic aberration by comparing PSFs at different wavelengths.
        通过比较不同波长的 PSF 来分析色差

        Args:
            wavelengths: Wavelengths to analyze
            spp: Samples per pixel
            psf_size: PSF kernel size

        Returns:
            Dictionary with chromatic aberration metrics
        """
        if wavelengths is None:
            wavelengths = WAVE_RGB

        results = {}

        # Compute on-axis PSFs for each wavelength
        psf_centroids = []
        psf_sizes = []

        with torch.no_grad():
            for wv in wavelengths:
                psf = self.hybrid_lens.psf(
                    points=[0.0, 0.0, -10000.0],
                    ks=psf_size,
                    wvln=wv,
                    spp=spp,
                )

                # Compute centroid
                h, w = psf.shape
                y_coords = torch.arange(h, dtype=torch.float64, device=psf.device)
                x_coords = torch.arange(w, dtype=torch.float64, device=psf.device)
                y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")

                total = psf.sum()
                centroid_y = (y_grid * psf).sum() / total
                centroid_x = (x_grid * psf).sum() / total
                psf_centroids.append([centroid_x.item(), centroid_y.item()])

                # Compute RMS size
                dist_sq = (x_grid - centroid_x) ** 2 + (y_grid - centroid_y) ** 2
                rms_size = torch.sqrt((dist_sq * psf).sum() / total).item()
                psf_sizes.append(rms_size)

        # Calculate chromatic shifts
        ref_centroid = psf_centroids[
            len(psf_centroids) // 2
        ]  # Use middle wavelength as reference
        chromatic_shifts = []
        for i, centroid in enumerate(psf_centroids):
            shift = np.sqrt(
                (centroid[0] - ref_centroid[0]) ** 2
                + (centroid[1] - ref_centroid[1]) ** 2
            )
            chromatic_shifts.append(shift)
            results[f"shift_wv{i}_pixels"] = shift

        results["max_chromatic_shift_pixels"] = max(chromatic_shifts)
        results["psf_size_variation"] = max(psf_sizes) - min(psf_sizes)

        # Save wavelength-specific PSF sizes
        for i, (wv, size) in enumerate(zip(wavelengths, psf_sizes)):
            results[f"rms_size_wv{i}_{wv}um_pixels"] = size

        return results

    def compare_with_geolens(
        self,
        spp: int = SPP_COHERENT // 10,
        psf_size: int = 101,
    ) -> Dict[str, Any]:
        """
        Compare hybrid lens performance with the base GeoLens.
        将混合透镜性能与基础 GeoLens 进行比较

        Args:
            spp: Samples per pixel
            psf_size: PSF kernel size

        Returns:
            Dictionary with comparison metrics
        """
        results = {}

        geolens = self.hybrid_lens.geolens

        # Compare PSF at on-axis
        with torch.no_grad():
            # Hybrid lens PSF
            hybrid_psf = self.hybrid_lens.psf(
                points=[0.0, 0.0, -10000.0],
                ks=psf_size,
                wvln=DEFAULT_WAVE,
                spp=spp,
            )

            # GeoLens PSF (using ray-based method)
            # Note: This is an approximation as GeoLens uses different PSF calculation
            geo_psf = geolens.psf_rgb(
                point=None,
                ks=psf_size,
                spp=spp // 100,  # Ray-based needs fewer samples
            )
            if geo_psf.dim() == 3:
                geo_psf = geo_psf.mean(dim=0)  # Average over channels

        # Compute metrics for both
        def compute_psf_metrics(psf):
            h, w = psf.shape
            y_coords = torch.arange(h, dtype=torch.float64, device=psf.device)
            x_coords = torch.arange(w, dtype=torch.float64, device=psf.device)
            y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")

            total = psf.sum()
            centroid_y = (y_grid * psf).sum() / total
            centroid_x = (x_grid * psf).sum() / total

            dist_sq = (x_grid - centroid_x) ** 2 + (y_grid - centroid_y) ** 2
            rms_size = torch.sqrt((dist_sq * psf).sum() / total).item()

            # Strehl ratio approximation (max value / ideal max)
            strehl = psf.max().item() * (h * w)

            return {"rms_size": rms_size, "strehl_approx": strehl}

        hybrid_metrics = compute_psf_metrics(hybrid_psf)
        geo_metrics = compute_psf_metrics(geo_psf)

        results["hybrid_rms_size_pixels"] = hybrid_metrics["rms_size"]
        results["geolens_rms_size_pixels"] = geo_metrics["rms_size"]
        results["rms_improvement_percent"] = (
            (geo_metrics["rms_size"] - hybrid_metrics["rms_size"])
            / geo_metrics["rms_size"]
            * 100
        )

        results["hybrid_strehl_approx"] = hybrid_metrics["strehl_approx"]
        results["geolens_strehl_approx"] = geo_metrics["strehl_approx"]

        # Save comparison visualization
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(np.log10(geo_psf.cpu().numpy() + 1e-10), cmap="hot")
        axes[0].set_title(f"GeoLens PSF\nRMS: {geo_metrics['rms_size']:.2f} px")
        axes[0].axis("off")

        axes[1].imshow(np.log10(hybrid_psf.cpu().numpy() + 1e-10), cmap="hot")
        axes[1].set_title(f"HybridLens PSF\nRMS: {hybrid_metrics['rms_size']:.2f} px")
        axes[1].axis("off")

        plt.tight_layout()
        plt.savefig(self.result_dir / "comparison_geolens_vs_hybrid.png", dpi=150)
        plt.close()

        print(
            f"Comparison saved to: {self.result_dir / 'comparison_geolens_vs_hybrid.png'}"
        )
        return results

    def full_analysis(self, save_plots: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive optical analysis of the hybrid lens.
        对混合透镜执行全面的光学分析

        Args:
            save_plots: Whether to save analysis plots

        Returns:
            Dictionary with all analysis results
        """
        results = {}

        print("=" * 60)
        print("Hybrid Lens Full Analysis")
        print("=" * 60)

        # Basic parameters
        print("\n1. Analyzing basic parameters...")
        results["basic"] = self.analyze_basic_params()

        # DOE phase analysis
        print("\n2. Analyzing DOE phase map...")
        results["doe_phase"] = self.analyze_doe_phase()

        # PSF analysis
        print("\n3. Computing and visualizing PSFs...")
        psfs = self.compute_psf(spp=SPP_COHERENT // 10)  # Faster for analysis
        if save_plots:
            self.visualize_psf(psfs)
        results["psf_computed"] = True

        # Chromatic aberration
        print("\n4. Analyzing chromatic aberration...")
        results["chromatic"] = self.analyze_chromatic_aberration()

        # Comparison with GeoLens
        print("\n5. Comparing with base GeoLens...")
        results["comparison"] = self.compare_with_geolens()

        # Save summary
        if save_plots:
            self._save_summary(results)

        print("\n" + "=" * 60)
        print("Analysis complete!")
        print("=" * 60)

        return results

    def _save_summary(self, results: Dict[str, Any]) -> None:
        """Save analysis summary to file."""
        summary_path = self.result_dir / "analysis_summary.txt"

        with open(summary_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("Hybrid Lens Optical Performance Summary\n")
            f.write("=" * 60 + "\n\n")

            # Basic parameters
            if "basic" in results:
                basic = results["basic"]
                f.write("Basic Parameters:\n")
                f.write(f"  Focal length: {basic.get('foclen', 'N/A'):.2f} mm\n")
                f.write(f"  F-number: {basic.get('fnum', 'N/A'):.2f}\n")
                f.write(f"  Half FOV: {basic.get('rfov_deg', 'N/A'):.2f} deg\n")
                f.write(f"  Sensor radius: {basic.get('r_sensor', 'N/A'):.2f} mm\n")
                f.write(f"  Number of surfaces: {basic.get('num_surfaces', 'N/A')}\n")
                f.write("\n")

                f.write("DOE Parameters:\n")
                f.write(f"  Type: {basic.get('doe_type', 'N/A')}\n")
                f.write(f"  Resolution: {basic.get('doe_resolution', 'N/A')}\n")
                f.write(
                    f"  Pixel size: {basic.get('doe_pixel_size_um', 'N/A'):.2f} um\n"
                )
                f.write(
                    f"  Distance from sensor: {basic.get('doe_distance', 'N/A'):.3f} mm\n"
                )
                f.write("\n")

            # DOE phase analysis
            if "doe_phase" in results:
                phase = results["doe_phase"]
                f.write("DOE Phase Analysis:\n")
                f.write(f"  Phase range: {phase.get('phase_range_pi', 'N/A'):.2f} π\n")
                f.write(f"  Phase mean: {phase.get('phase_mean', 'N/A'):.4f} rad\n")
                f.write(f"  Phase std: {phase.get('phase_std', 'N/A'):.4f} rad\n")
                f.write(
                    f"  Max gradient: {phase.get('max_gradient', 'N/A'):.4f} rad/px\n"
                )
                f.write("\n")

            # Chromatic aberration
            if "chromatic" in results:
                chrom = results["chromatic"]
                f.write("Chromatic Aberration:\n")
                f.write(
                    f"  Max shift: {chrom.get('max_chromatic_shift_pixels', 'N/A'):.2f} pixels\n"
                )
                f.write(
                    f"  PSF size variation: {chrom.get('psf_size_variation', 'N/A'):.2f} pixels\n"
                )
                f.write("\n")

            # Comparison
            if "comparison" in results:
                comp = results["comparison"]
                f.write("Comparison with GeoLens:\n")
                f.write(
                    f"  GeoLens RMS: {comp.get('geolens_rms_size_pixels', 'N/A'):.2f} pixels\n"
                )
                f.write(
                    f"  HybridLens RMS: {comp.get('hybrid_rms_size_pixels', 'N/A'):.2f} pixels\n"
                )
                f.write(
                    f"  Improvement: {comp.get('rms_improvement_percent', 'N/A'):.1f}%\n"
                )
                f.write("\n")

        print(f"Summary saved to: {summary_path}")
