#!/usr/bin/env python3
"""Capture environment fingerprint for reproducibility.
采集环境指纹（Python/Torch/CUDA/GPU/关键依赖版本）。
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def get_cmd_output(cmd: list[str]) -> str | None:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=20)
        return out.strip()
    except Exception:
        return None


def package_version(name: str) -> str | None:
    try:
        import importlib.metadata as im

        return im.version(name)
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture env fingerprint")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument(
        "--pip-freeze-out",
        default="repro/environment/pip-freeze.txt",
        help="Write pip freeze to this path",
    )
    args = parser.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = Path(args.output or f"repro/reports/env_fingerprint_{ts}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fp: dict[str, object] = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "env": {
            "CONDA_DEFAULT_ENV": os.environ.get("CONDA_DEFAULT_ENV"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }

    # Torch stack
    torch_info: dict[str, object] = {}
    try:
        import torch

        torch_info["torch_version"] = torch.__version__
        torch_info["cuda_is_available"] = bool(torch.cuda.is_available())
        torch_info["cuda_version"] = torch.version.cuda
        torch_info["cudnn_version"] = torch.backends.cudnn.version()
        torch_info["device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            torch_info["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception as exc:
        torch_info["error"] = str(exc)
    fp["torch"] = torch_info

    # Key packages
    key_pkgs = [
        "deeplens-core",
        "hybrid-lens-design",
        "torch",
        "torchvision",
        "torchaudio",
        "numpy",
        "matplotlib",
        "scikit-image",
        "lpips",
        "timm",
        "transformers",
        "pyyaml",
    ]
    fp["packages"] = {k: package_version(k) for k in key_pkgs}

    # GPU command output
    fp["nvidia_smi"] = get_cmd_output(["nvidia-smi", "--query-gpu=name,driver_version,cuda_version,memory.total", "--format=csv,noheader"])

    # Save pip freeze
    pip_freeze_path = Path(args.pip_freeze_out)
    pip_freeze_path.parent.mkdir(parents=True, exist_ok=True)
    freeze = get_cmd_output([sys.executable, "-m", "pip", "freeze"])
    if freeze is not None:
        pip_freeze_path.write_text(freeze + "\n", encoding="utf-8")
        fp["pip_freeze_path"] = str(pip_freeze_path)
    else:
        fp["pip_freeze_path"] = None

    out_path.write_text(json.dumps(fp, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] env fingerprint written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
