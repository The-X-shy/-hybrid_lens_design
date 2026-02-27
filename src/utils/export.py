"""
Zemax Export Utilities for GeoLens
用于 GeoLens 的 Zemax 导出工具

This module provides utilities to export trained lens designs to Zemax (.zmx)
and CODE V (.seq) formats for fabrication and further analysis.

DeepLens already has write_lens_zmx() and write_lens_seq() methods built-in.
This module provides a higher-level interface with additional features:
- Batch export with organized output directories
- Export metadata and training information
- Validation before export
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

import torch
import numpy as np

# DeepLens imports
from deeplens import GeoLens


class ZemaxExporter:
    """
    Export GeoLens designs to Zemax and CODE V formats.
    将 GeoLens 设计导出为 Zemax 和 CODE V 格式

    Note: Diffractive elements (DOE) in HybridLens cannot be directly
    represented in standard Zemax format. This exporter handles the
    refractive (GeoLens) portion only.
    """

    def __init__(
        self,
        lens: GeoLens,
        output_dir: Union[str, Path],
        lens_name: Optional[str] = None,
    ):
        """
        Initialize the exporter.

        Args:
            lens: GeoLens object to export
            output_dir: Directory for output files
            lens_name: Optional name for the lens design
        """
        self.lens = lens
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.lens_name = lens_name or "lens_design"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    @classmethod
    def from_file(
        cls,
        lens_path: Union[str, Path],
        output_dir: Union[str, Path],
        lens_name: Optional[str] = None,
    ) -> "ZemaxExporter":
        """
        Create exporter from lens file.
        从透镜文件创建导出器

        Args:
            lens_path: Path to lens JSON file
            output_dir: Directory for output files
            lens_name: Optional name for the lens design
        """
        lens = GeoLens(filename=str(lens_path))

        if lens_name is None:
            lens_name = Path(lens_path).stem

        return cls(lens=lens, output_dir=output_dir, lens_name=lens_name)

    def validate_lens(self) -> Dict[str, Any]:
        """
        Validate lens before export.
        导出前验证透镜

        Returns:
            Dictionary with validation results
        """
        results = {
            "valid": True,
            "warnings": [],
            "errors": [],
        }

        # Check basic parameters
        if not hasattr(self.lens, "surfaces") or len(self.lens.surfaces) == 0:
            results["valid"] = False
            results["errors"].append("No surfaces defined")

        if not hasattr(self.lens, "d_sensor"):
            results["valid"] = False
            results["errors"].append("Sensor distance not defined")

        if not hasattr(self.lens, "r_sensor"):
            results["valid"] = False
            results["errors"].append("Sensor radius not defined")

        # Check for valid focal length
        if hasattr(self.lens, "foclen"):
            if self.lens.foclen <= 0:
                results["warnings"].append(f"Unusual focal length: {self.lens.foclen}")
        else:
            results["warnings"].append("Focal length not computed")

        # Check for valid F-number
        if hasattr(self.lens, "fnum"):
            if self.lens.fnum < 0.5 or self.lens.fnum > 32:
                results["warnings"].append(f"Unusual F-number: {self.lens.fnum}")

        # Check entrance pupil
        if not hasattr(self.lens, "enpd") or self.lens.enpd is None:
            results["warnings"].append(
                "Entrance pupil diameter not set, will be computed"
            )

        return results

    def get_lens_info(self) -> Dict[str, Any]:
        """
        Get lens parameters for export metadata.
        获取用于导出元数据的透镜参数

        Returns:
            Dictionary with lens parameters
        """
        info = {
            "name": self.lens_name,
            "export_timestamp": self.timestamp,
            "num_surfaces": len(self.lens.surfaces),
        }

        # Add optical parameters if available
        if hasattr(self.lens, "foclen"):
            info["focal_length_mm"] = round(self.lens.foclen, 4)
        if hasattr(self.lens, "fnum"):
            info["f_number"] = round(self.lens.fnum, 2)
        if hasattr(self.lens, "rfov"):
            info["half_fov_deg"] = round(np.rad2deg(self.lens.rfov), 2)
        if hasattr(self.lens, "r_sensor"):
            info["sensor_radius_mm"] = round(self.lens.r_sensor, 4)
        if hasattr(self.lens, "d_sensor"):
            info["sensor_distance_mm"] = round(float(self.lens.d_sensor), 4)
        if hasattr(self.lens, "enpd") and self.lens.enpd is not None:
            info["entrance_pupil_diameter_mm"] = round(self.lens.enpd, 4)

        # Surface types
        surface_types = []
        for surf in self.lens.surfaces:
            surface_types.append(surf.__class__.__name__)
        info["surface_types"] = surface_types

        return info

    def export_zmx(
        self,
        filename: Optional[str] = None,
        include_metadata: bool = True,
    ) -> Path:
        """
        Export lens to Zemax .zmx format.
        将透镜导出为 Zemax .zmx 格式

        Args:
            filename: Output filename (without extension)
            include_metadata: Whether to save metadata JSON alongside

        Returns:
            Path to the exported .zmx file
        """
        if filename is None:
            filename = f"{self.lens_name}"

        zmx_path = self.output_dir / f"{filename}.zmx"

        # Ensure lens has required attributes for ZMX export
        self._prepare_for_export()

        # Use DeepLens built-in export
        self.lens.write_lens_zmx(str(zmx_path))

        # Save metadata
        if include_metadata:
            metadata_path = self.output_dir / f"{filename}_metadata.json"
            metadata = {
                "format": "Zemax",
                "file": zmx_path.name,
                "lens_info": self.get_lens_info(),
                "validation": self.validate_lens(),
            }
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            print(f"Metadata saved to: {metadata_path}")

        return zmx_path

    def export_seq(
        self,
        filename: Optional[str] = None,
        include_metadata: bool = True,
    ) -> Path:
        """
        Export lens to CODE V .seq format.
        将透镜导出为 CODE V .seq 格式

        Args:
            filename: Output filename (without extension)
            include_metadata: Whether to save metadata JSON alongside

        Returns:
            Path to the exported .seq file
        """
        if filename is None:
            filename = f"{self.lens_name}"

        seq_path = self.output_dir / f"{filename}.seq"

        # Ensure lens has required attributes for SEQ export
        self._prepare_for_export()

        # Use DeepLens built-in export
        self.lens.write_lens_seq(str(seq_path))

        # Save metadata
        if include_metadata:
            metadata_path = self.output_dir / f"{filename}_codev_metadata.json"
            metadata = {
                "format": "CODE V",
                "file": seq_path.name,
                "lens_info": self.get_lens_info(),
                "validation": self.validate_lens(),
            }
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            print(f"Metadata saved to: {metadata_path}")

        return seq_path

    def export_all(
        self,
        filename: Optional[str] = None,
        include_json: bool = True,
    ) -> Dict[str, Path]:
        """
        Export lens to all supported formats.
        将透镜导出为所有支持的格式

        Args:
            filename: Base filename (without extension)
            include_json: Whether to also export DeepLens JSON format

        Returns:
            Dictionary mapping format names to file paths
        """
        if filename is None:
            filename = f"{self.lens_name}"

        exports = {}

        # Export to Zemax
        exports["zmx"] = self.export_zmx(filename=filename, include_metadata=False)

        # Export to CODE V
        exports["seq"] = self.export_seq(filename=filename, include_metadata=False)

        # Export to DeepLens JSON
        if include_json:
            json_path = self.output_dir / f"{filename}.json"
            self.lens.write_lens_json(str(json_path))
            exports["json"] = json_path
            print(f"Lens JSON saved to: {json_path}")

        # Save combined metadata
        metadata_path = self.output_dir / f"{filename}_export_info.json"
        metadata = {
            "lens_info": self.get_lens_info(),
            "validation": self.validate_lens(),
            "exported_files": {k: str(v) for k, v in exports.items()},
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        exports["metadata"] = metadata_path

        print(f"\nExport complete! Files saved to: {self.output_dir}")
        print(f"  - Zemax: {exports['zmx'].name}")
        print(f"  - CODE V: {exports['seq'].name}")
        if include_json:
            print(f"  - JSON: {exports['json'].name}")
        print(f"  - Metadata: {exports['metadata'].name}")

        return exports

    def _prepare_for_export(self) -> None:
        """
        Prepare lens for export by ensuring required attributes exist.
        通过确保必需属性存在来准备透镜导出
        """
        # Ensure enpd is set
        if not hasattr(self.lens, "enpd") or self.lens.enpd is None:
            # Compute entrance pupil if not set
            if hasattr(self.lens, "foclen") and hasattr(self.lens, "fnum"):
                self.lens.enpd = self.lens.foclen / self.lens.fnum
            else:
                # Use aperture radius as fallback
                for surf in self.lens.surfaces:
                    if surf.__class__.__name__ == "Aperture":
                        self.lens.enpd = surf.r * 2
                        break
                else:
                    # Default fallback
                    self.lens.enpd = 5.0
            self.lens.float_enpd = False

        # Ensure rfov is set
        if not hasattr(self.lens, "rfov") or self.lens.rfov is None:
            # Default to 20 degrees half FOV
            self.lens.rfov = np.deg2rad(20)


def export_geolens_to_zemax(
    lens_path: Union[str, Path],
    output_dir: Union[str, Path],
    lens_name: Optional[str] = None,
    formats: List[str] = ["zmx"],
) -> Dict[str, Path]:
    """
    Convenience function to export a GeoLens to Zemax format.
    将 GeoLens 导出为 Zemax 格式的便捷函数

    Args:
        lens_path: Path to lens JSON file
        output_dir: Directory for output files
        lens_name: Optional name for the lens design
        formats: List of formats to export ("zmx", "seq", "json")

    Returns:
        Dictionary mapping format names to file paths
    """
    exporter = ZemaxExporter.from_file(
        lens_path=lens_path,
        output_dir=output_dir,
        lens_name=lens_name,
    )

    # Validate
    validation = exporter.validate_lens()
    if not validation["valid"]:
        print("Validation errors:")
        for error in validation["errors"]:
            print(f"  - {error}")
        raise ValueError("Lens validation failed")

    if validation["warnings"]:
        print("Validation warnings:")
        for warning in validation["warnings"]:
            print(f"  - {warning}")

    # Export requested formats
    exports = {}

    if "zmx" in formats:
        exports["zmx"] = exporter.export_zmx(include_metadata=False)

    if "seq" in formats:
        exports["seq"] = exporter.export_seq(include_metadata=False)

    if "json" in formats:
        json_path = exporter.output_dir / f"{exporter.lens_name}.json"
        exporter.lens.write_lens_json(str(json_path))
        exports["json"] = json_path

    # Always save metadata
    metadata_path = exporter.output_dir / f"{exporter.lens_name}_export_info.json"
    metadata = {
        "lens_info": exporter.get_lens_info(),
        "validation": validation,
        "exported_files": {k: str(v) for k, v in exports.items()},
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    exports["metadata"] = metadata_path

    return exports
