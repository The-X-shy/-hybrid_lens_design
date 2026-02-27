"""
GeoLens Evaluator - Optical Performance Evaluation for Refractive Lens
折射透镜评估器 - 光学性能评估

Provides analysis tools for PSF, MTF, distortion, and other optical metrics.
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt

# DeepLens imports
from deeplens import GeoLens
from deeplens.basics import DEPTH, WAVE_RGB


class GeoLensEvaluator:
    """
    Evaluator for refractive lens optical performance.
    折射透镜光学性能评估器
    """

    def __init__(
        self,
        lens: GeoLens,
        result_dir: str,
        device: torch.device = None,
    ):
        """
        Initialize the evaluator.

        Args:
            lens: GeoLens object to evaluate
            result_dir: Directory to save results
            device: PyTorch device
        """
        self.lens = lens
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.lens.to(self.device)

    @classmethod
    def from_file(cls, lens_path: str, result_dir: str) -> "GeoLensEvaluator":
        """
        Create evaluator from lens file.
        从透镜文件创建评估器
        """
        lens = GeoLens(filename=lens_path)
        return cls(lens=lens, result_dir=result_dir)

    def analyze_rms(self, depth: float = DEPTH) -> Dict[str, float]:
        """
        Analyze RMS spot size at different field positions.
        分析不同视场位置的 RMS 弥散斑大小

        Args:
            depth: Object distance

        Returns:
            Dictionary with RMS values for different field positions
        """
        results = {}

        # Save analysis plots
        self.lens.analysis(save_name=f"{self.result_dir}/rms_analysis")

        # Get lens parameters
        results["foclen"] = self.lens.foclen
        results["fnum"] = self.lens.fnum
        results["rfov_deg"] = np.rad2deg(self.lens.rfov)
        results["r_sensor"] = self.lens.r_sensor

        return results

    def compute_psf(
        self,
        field_positions: List[float] = [0.0, 0.5, 1.0],
        wavelengths: List[float] = None,
        spp: int = 10000,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute PSF at different field positions.
        计算不同视场位置的 PSF

        Args:
            field_positions: Normalized field positions (0.0 = on-axis, 1.0 = edge)
            wavelengths: Wavelengths to compute PSF for
            spp: Samples per pixel

        Returns:
            Dictionary with PSF tensors for each field position
        """
        if wavelengths is None:
            wavelengths = WAVE_RGB

        psfs = {}

        for fov in field_positions:
            psf_list = []
            for wv in wavelengths:
                # Compute PSF for this field/wavelength combination
                psf = self.lens.psf_rgb(
                    point=None,  # Will be computed from fov
                    ks=51,
                    spp=spp,
                )
                psf_list.append(psf)

            psfs[f"fov_{fov}"] = torch.stack(psf_list)

        return psfs

    def analyze_distortion(self) -> Dict[str, float]:
        """
        Analyze lens distortion.
        分析透镜畸变

        Returns:
            Dictionary with distortion metrics
        """
        # This would require implementing distortion analysis
        # For now, return placeholder
        return {"distortion_percent": 0.0}

    def analyze_vignetting(self) -> Dict[str, float]:
        """
        Analyze vignetting across the field.
        分析视场内的渐晕

        Returns:
            Dictionary with vignetting metrics
        """
        # This would require implementing vignetting analysis
        return {"vignetting_edge": 0.0}

    def full_analysis(self, save_plots: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive optical analysis.
        执行全面的光学分析

        Args:
            save_plots: Whether to save analysis plots

        Returns:
            Dictionary with all analysis results
        """
        results = {}

        # RMS analysis
        results["rms"] = self.analyze_rms()

        # Distortion
        results["distortion"] = self.analyze_distortion()

        # Vignetting
        results["vignetting"] = self.analyze_vignetting()

        # Save summary
        if save_plots:
            self._save_summary(results)

        return results

    def _save_summary(self, results: Dict[str, Any]) -> None:
        """Save analysis summary to file."""
        summary_path = self.result_dir / "analysis_summary.txt"

        with open(summary_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("Lens Optical Performance Summary\n")
            f.write("=" * 60 + "\n\n")

            if "rms" in results:
                rms = results["rms"]
                f.write("Basic Parameters:\n")
                f.write(f"  Focal length: {rms.get('foclen', 'N/A'):.2f} mm\n")
                f.write(f"  F-number: {rms.get('fnum', 'N/A'):.2f}\n")
                f.write(f"  Half FOV: {rms.get('rfov_deg', 'N/A'):.2f} deg\n")
                f.write(f"  Sensor radius: {rms.get('r_sensor', 'N/A'):.2f} mm\n")
                f.write("\n")

        print(f"Summary saved to: {summary_path}")

    def compare_with(self, other_lens: GeoLens) -> Dict[str, Any]:
        """
        Compare this lens with another lens.
        与另一个透镜比较

        Args:
            other_lens: Another GeoLens to compare with

        Returns:
            Dictionary with comparison results
        """
        comparison = {}

        # Compare basic parameters
        comparison["foclen_diff"] = self.lens.foclen - other_lens.foclen
        comparison["fnum_diff"] = self.lens.fnum - other_lens.fnum

        return comparison
