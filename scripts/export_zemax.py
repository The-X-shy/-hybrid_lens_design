#!/usr/bin/env python
"""
Export GeoLens to Zemax and CODE V formats
导出 GeoLens 为 Zemax 和 CODE V 格式

Usage:
    python scripts/export_zemax.py --lens results/geolens/final_lens.json --output exports/
    python scripts/export_zemax.py --lens results/geolens/final_lens.json --formats zmx seq json
"""

import argparse
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.export import ZemaxExporter, export_geolens_to_zemax


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export GeoLens to Zemax and CODE V formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export to Zemax only (default)
  python scripts/export_zemax.py --lens results/geolens/final_lens.json

  # Export to all formats
  python scripts/export_zemax.py --lens results/geolens/final_lens.json --formats zmx seq json

  # Specify output directory and lens name
  python scripts/export_zemax.py --lens results/geolens/final_lens.json \\
      --output exports/ --name my_lens_design
        """,
    )

    parser.add_argument(
        "--lens", "-l", type=str, required=True, help="Path to GeoLens JSON file"
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="exports",
        help="Output directory (default: exports/)",
    )

    parser.add_argument(
        "--name",
        "-n",
        type=str,
        default=None,
        help="Lens name for output files (default: use input filename)",
    )

    parser.add_argument(
        "--formats",
        "-f",
        nargs="+",
        choices=["zmx", "seq", "json"],
        default=["zmx"],
        help="Output formats (default: zmx)",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Export to all formats (equivalent to --formats zmx seq json)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Validate input file
    lens_path = Path(args.lens)
    if not lens_path.exists():
        print(f"Error: Lens file not found: {lens_path}")
        sys.exit(1)

    if not lens_path.suffix == ".json":
        print(f"Warning: Expected .json file, got {lens_path.suffix}")

    # Set output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine lens name
    lens_name = args.name if args.name else lens_path.stem

    # Determine formats
    formats = ["zmx", "seq", "json"] if args.all else args.formats

    print("=" * 60)
    print("GeoLens Export to Zemax/CODE V")
    print("=" * 60)
    print(f"Input lens: {lens_path}")
    print(f"Output directory: {output_dir}")
    print(f"Lens name: {lens_name}")
    print(f"Formats: {', '.join(formats)}")
    print("=" * 60)

    try:
        # Export
        exports = export_geolens_to_zemax(
            lens_path=lens_path,
            output_dir=output_dir,
            lens_name=lens_name,
            formats=formats,
        )

        print("\nExport successful!")
        print("\nExported files:")
        for fmt, path in exports.items():
            print(f"  {fmt}: {path}")

    except Exception as e:
        print(f"\nError during export: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
