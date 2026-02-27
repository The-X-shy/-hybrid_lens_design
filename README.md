# Hybrid Lens Design

End-to-end hybrid refractive-diffractive lens design framework based on DeepLens.

## Overview

This project implements a modular pipeline for designing and optimizing hybrid optical systems combining:
- **Refractive lenses** (GeoLens) - Traditional glass/plastic lens elements
- **Diffractive Optical Elements (DOE)** - Binary2 or Pixel2D metasurfaces
- **Neural Networks** - UNet for joint end-to-end optimization

Based on:
- Yang et al., "Curriculum learning for ab initio deep learned refractive optics," Nature Communications 2024
- Yang et al., "End-to-End Hybrid Refractive-Diffractive Lens Design with Differentiable Ray-Wave Model," SIGGRAPH Asia 2024

## Project Structure

```
hybrid_lens_design/
├── configs/                    # YAML configuration files
│   ├── default.yaml           # General defaults
│   ├── geolens.yaml           # Refractive lens training config
│   ├── hybrid.yaml            # HybridLens (Binary2 DOE) config
│   ├── meta.yaml              # Metasurface (Pixel2D) config
│   └── e2e.yaml               # End-to-end training config
├── src/
│   ├── geolens/               # Refractive lens modules
│   │   ├── trainer.py         # Curriculum learning trainer
│   │   └── evaluator.py       # Lens evaluation
│   ├── hybridlens/            # Hybrid lens modules
│   │   ├── trainer.py         # Binary2 DOE trainer
│   │   ├── metasurface_trainer.py  # Pixel2D trainer
│   │   └── evaluator.py       # Hybrid lens evaluation
│   ├── e2e/                   # End-to-end modules
│   │   └── trainer.py         # HybridLens + UNet trainer
│   └── utils/
│       ├── config.py          # Configuration loader
│       └── export.py          # Zemax/CODE V export utilities
├── scripts/                   # Training scripts
│   ├── train_geolens.py       # Train refractive lens
│   ├── train_hybridlens.py    # Train Binary2 hybrid lens
│   ├── train_metalens.py      # Train Pixel2D metasurface lens
│   ├── train_e2e.py           # Train end-to-end system
│   ├── run_pipeline.py        # Full pipeline script
│   └── export_zemax.py        # Export to Zemax/CODE V
├── deeplens/                  # DeepLens library (local copy)
├── datasets/                  # Datasets (symlink to DeepLens)
├── results/                   # Training outputs
├── environment.yml            # Conda environment
├── pyproject.toml             # Python package config
└── README.md
```

## Installation

### 1. Create Conda Environment

```bash
cd hybrid_lens_design
conda env create -f environment.yml
conda activate hybrid_lens
```

### 2. Install DeepLens (Local Editable)

```bash
pip install -e .
```

### 3. Verify Installation

```python
from deeplens import GeoLens
from deeplens.hybridlens import HybridLens
print("DeepLens installed successfully!")
```

## Quick Start

### Train a Complete Hybrid Lens System

```bash
# Run full pipeline: GeoLens -> HybridLens -> Evaluation
python scripts/run_pipeline.py --stages geolens,hybridlens,eval

# Or run with all stages including metasurface and E2E
python scripts/run_pipeline.py --full
```

### Individual Training Stages

```bash
# 1. Train refractive lens (GeoLens)
python scripts/train_geolens.py --config configs/geolens.yaml

# 2. Train hybrid lens with Binary2 DOE
python scripts/train_hybridlens.py --geolens results/geolens/final_lens.json

# 3. Train metasurface lens with Pixel2D DOE
python scripts/train_metalens.py --geolens results/geolens/final_lens.json

# 4. Train end-to-end system (HybridLens + UNet)
python scripts/train_e2e.py --hybridlens results/hybridlens/final.json
```

## Configuration

### GeoLens Configuration (configs/geolens.yaml)

```yaml
lens:
  design:
    foclen: 8.0      # Focal length (mm)
    fov: 80.0        # Field of view (degrees)
    fnum: 2.0        # F-number
    
optimization:
  curriculum:
    iterations: 2000
    lrs: [0.0001, 0.0001, 0.01, 0.0001]  # [d, c, k, ai]
```

### HybridLens Configuration (configs/hybrid.yaml)

```yaml
doe:
  type: "binary2"    # DOE type
  res: [1000, 1000]  # Resolution
  fab_ps: 0.003      # Pixel size (mm)
  
optimization:
  doe_lr: 0.1
  iterations: 2000
  spp: 1000000       # Samples per pixel
```

## API Usage

### GeoLens Training

```python
from src.geolens.trainer import GeoLensTrainer
from src.utils.config import load_config

config = load_config("configs/geolens.yaml")
trainer = GeoLensTrainer.from_config(config)

# Train with curriculum learning + fine-tuning
curriculum_history, finetune_history = trainer.train(
    curriculum_config={"iterations": 2000},
    finetune_config={"iterations": 1200},
)

# Get trained lens
lens = trainer.get_lens()
```

### HybridLens Training

```python
from src.hybridlens.trainer import HybridLensTrainer

trainer = HybridLensTrainer.from_geolens(
    geolens_path="results/geolens/final_lens.json",
    result_dir="results/hybridlens",
)

# Setup sensor
trainer.setup_sensor(match_aperture=True)

# Train
loss_history = trainer.train(
    doe_lr=0.1,
    iterations=2000,
    spp=1000000,
)
```

### Metasurface (Pixel2D) Training

```python
from src.hybridlens.metasurface_trainer import MetasurfaceTrainer

trainer = MetasurfaceTrainer.from_geolens(
    geolens_path="results/geolens/final_lens.json",
    result_dir="results/metalens",
)

loss_history = trainer.train(
    doe_lr=0.1,
    smoothness_weight=0.001,  # Regularization
    fabrication_weight=0.01,
)
```

### End-to-End Training

```python
from src.e2e.trainer import E2ETrainer

trainer = E2ETrainer.from_hybridlens(
    hybridlens_path="results/hybridlens/final.json",
    result_dir="results/e2e",
)

loss_history = trainer.train(
    train_dataset_path="./datasets/DIV2K/train",
    epochs=100,
    batch_size=4,
)
```

### Evaluation

```python
from src.hybridlens.evaluator import HybridLensEvaluator

evaluator = HybridLensEvaluator.from_file(
    "results/hybridlens/final.json",
    "results/evaluation",
)

# Full analysis
results = evaluator.full_analysis()

# Specific analyses
psfs = evaluator.compute_psf()
doe_analysis = evaluator.analyze_doe_phase()
chromatic = evaluator.analyze_chromatic_aberration()
```

## Export to Zemax/CODE V

Export trained lens designs for fabrication or further analysis in professional optical design software.

### Command Line Export

```bash
# Export to Zemax (.zmx) format only
python scripts/export_zemax.py --lens results/geolens/final_lens.json

# Export to all formats (Zemax, CODE V, JSON)
python scripts/export_zemax.py --lens results/geolens/final_lens.json --all

# Specify output directory and lens name
python scripts/export_zemax.py --lens results/geolens/final_lens.json \
    --output exports/ --name my_lens_design

# Export specific formats
python scripts/export_zemax.py --lens results/geolens/final_lens.json \
    --formats zmx seq
```

### Programmatic Export

```python
from src.utils.export import ZemaxExporter, export_geolens_to_zemax

# Simple export
exports = export_geolens_to_zemax(
    lens_path="results/geolens/final_lens.json",
    output_dir="exports/",
    formats=["zmx", "seq", "json"],
)

# Full control with ZemaxExporter
exporter = ZemaxExporter.from_file(
    lens_path="results/geolens/final_lens.json",
    output_dir="exports/",
    lens_name="my_design",
)

# Validate before export
validation = exporter.validate_lens()
if validation["warnings"]:
    print("Warnings:", validation["warnings"])

# Export to specific format
zmx_path = exporter.export_zmx()

# Export to all formats at once
all_exports = exporter.export_all()
```

### Supported Formats

| Format | Extension | Software | Notes |
|--------|-----------|----------|-------|
| Zemax | `.zmx` | Zemax OpticStudio | Standard, Even Asph surfaces |
| CODE V | `.seq` | Synopsys CODE V | Standard, Aspheric surfaces |
| DeepLens | `.json` | DeepLens | Native format, includes all parameters |

### Export Notes

- **GeoLens Only**: Export handles refractive lens elements. DOE/metasurface elements in HybridLens are not directly exportable to Zemax (these require separate fabrication workflows).
- **Metadata**: Each export includes a JSON metadata file with lens parameters and validation results.
- **Validation**: The exporter validates lens parameters before export and warns about unusual values.

## DOE Types

### Binary2 (Radial Polynomial)

Parameterized by radial polynomial coefficients (α₂, α₄, α₆, α₈, α₁₀):
```
φ(r) = π(α₂r² + α₄r⁴ + α₆r⁶ + α₈r⁸ + α₁₀r¹⁰)
```

Advantages:
- Compact representation (5 parameters)
- Smooth phase profile
- Easier to fabricate

### Pixel2D (Metasurface)

Direct phase map representation where each pixel is an independent parameter:
```
φ(x,y) = phase_map[x,y]
```

Advantages:
- Maximum design freedom
- Can achieve complex phase profiles
- Supports arbitrary patterns

## Technical Details

### Ray-Wave Hybrid Model

The HybridLens uses a differentiable ray-wave model:
1. **Ray tracing** through refractive elements (GeoLens)
2. **Coherent ray tracing** to compute complex field at DOE plane
3. **DOE phase modulation** applied to wavefield
4. **Angular Spectrum Method** propagation to sensor

### Loss Functions

- **PSFLoss**: Encourages compact PSF (spot size minimization)
- **MSE Loss**: Image reconstruction quality
- **LPIPS Loss**: Perceptual similarity
- **Regularization**: Smoothness and fabrication constraints for DOE

## Hardware Requirements

- **CPU**: Multi-core processor (training is parallelizable)
- **GPU**: CUDA-capable GPU recommended (10x+ speedup)
- **Memory**: 16GB+ RAM recommended
- **Storage**: ~10GB for datasets and results

## Citation

If you use this code, please cite:

```bibtex
@article{yang2024curriculum,
  title={Curriculum learning for ab initio deep learned refractive optics},
  author={Yang, Xinge and Fu, Qiang and Heidrich, Wolfgang},
  journal={Nature Communications},
  year={2024}
}

@inproceedings{yang2024endtoend,
  title={End-to-End Hybrid Refractive-Diffractive Lens Design with Differentiable Ray-Wave Model},
  author={Yang, Xinge and others},
  booktitle={SIGGRAPH Asia},
  year={2024}
}
```

## License

Apache 2.0 License - See LICENSE file for details.
