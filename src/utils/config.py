"""
Configuration loading and management utilities.
配置加载和管理工具模块
"""

import os
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union


class Config:
    """Configuration object with attribute-style access."""

    def __init__(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert Config object back to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def __repr__(self) -> str:
        return f"Config({self.to_dict()})"


def load_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML file and return as dictionary."""
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries. Override values take precedence.
    深度合并两个字典，覆盖值优先
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(
    config_path: Union[str, Path], base_dir: Optional[Union[str, Path]] = None
) -> Config:
    """
    Load configuration from YAML file with inheritance support.
    从 YAML 文件加载配置，支持继承

    Args:
        config_path: Path to the config file
        base_dir: Base directory for resolving relative paths

    Returns:
        Config object with all settings
    """
    config_path = Path(config_path)

    if base_dir is None:
        base_dir = config_path.parent
    else:
        base_dir = Path(base_dir)

    config_dict = load_yaml(config_path)

    # Handle config inheritance
    if "_extends" in config_dict:
        extends_path = base_dir / config_dict.pop("_extends")
        base_config = load_yaml(extends_path)
        config_dict = deep_merge(base_config, config_dict)

    return Config(config_dict)


def create_result_dir(config: Config) -> Path:
    """
    Create result directory based on config.
    根据配置创建结果目录

    Args:
        config: Configuration object

    Returns:
        Path to the result directory
    """
    # Priority 1: explicit output.dir
    output_dir = None
    if hasattr(config, "output") and hasattr(config.output, "dir"):
        output_dir = getattr(config.output, "dir")

    if output_dir:
        result_dir = Path(output_dir)
    else:
        # Priority 2: output.base_dir (+ optional timestamp)
        base_dir = Path(
            getattr(config.output, "base_dir", "./results")
            if hasattr(config, "output")
            else "./results"
        )

        if hasattr(config.output, "timestamp") and config.output.timestamp:
            timestamp = datetime.now().strftime("%m%d-%H%M%S")
            result_dir = base_dir / f"run_{timestamp}"
        else:
            result_dir = base_dir

    result_dir.mkdir(parents=True, exist_ok=True)

    return result_dir


def get_nested(config_or_dict: Any, path: str, default: Any = None) -> Any:
    """
    Safely read nested value from Config/dict with dot path.
    从 Config/dict 中安全读取点路径字段
    """
    current = config_or_dict
    for key in path.split("."):
        if isinstance(current, Config):
            if not hasattr(current, key):
                return default
            current = getattr(current, key)
        elif isinstance(current, dict):
            if key not in current:
                return default
            current = current[key]
        else:
            if not hasattr(current, key):
                return default
            current = getattr(current, key)
    return current


def set_nested(config_or_dict: Any, path: str, value: Any) -> None:
    """
    Set nested value on Config/dict with dot path, creating intermediate Config nodes.
    通过点路径写入 Config/dict（必要时创建中间节点）
    """
    keys = path.split(".")
    current = config_or_dict
    for key in keys[:-1]:
        if isinstance(current, Config):
            if not hasattr(current, key):
                setattr(current, key, Config({}))
            current = getattr(current, key)
        elif isinstance(current, dict):
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]
        else:
            if not hasattr(current, key):
                setattr(current, key, Config({}))
            current = getattr(current, key)

    last_key = keys[-1]
    if isinstance(current, Config):
        setattr(current, last_key, value)
    elif isinstance(current, dict):
        current[last_key] = value
    else:
        setattr(current, last_key, value)


def resolve_geolens_stage_configs(config: Config) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Resolve GeoLens stage configs with compatibility for both:
    1) optimization.curriculum / optimization.finetune
    2) curriculum / finetune (legacy)
    解析 GeoLens 阶段配置，兼容新旧两种配置层级
    """
    curriculum_src = get_nested(config, "optimization.curriculum")
    if curriculum_src is None:
        curriculum_src = get_nested(config, "curriculum")

    if curriculum_src is None:
        raise ValueError(
            "Missing curriculum config. Expected optimization.curriculum or curriculum."
        )

    curriculum_config = {
        "lrs": get_nested(curriculum_src, "lrs", [1e-4, 1e-4, 1e-2, 1e-4]),
        "decay": float(get_nested(curriculum_src, "decay", 0.01)),
        "iterations": int(get_nested(curriculum_src, "iterations", 2000)),
        "test_per_iter": int(get_nested(curriculum_src, "test_per_iter", 50)),
        "optim_mat": bool(get_nested(curriculum_src, "optim_mat", False)),
        "match_mat": bool(get_nested(curriculum_src, "match_mat", False)),
        "shape_control": bool(get_nested(curriculum_src, "shape_control", True)),
    }

    finetune_src = get_nested(config, "optimization.finetune")
    if finetune_src is None:
        finetune_src = get_nested(config, "finetune")

    if finetune_src is None:
        return curriculum_config, None

    finetune_lrs = get_nested(finetune_src, "lrs")
    if finetune_lrs is None:
        multiplier = float(get_nested(finetune_src, "lr_multiplier", 0.1))
        finetune_lrs = [float(lr) * multiplier for lr in curriculum_config["lrs"]]

    finetune_config = {
        "lrs": finetune_lrs,
        "decay": float(get_nested(finetune_src, "decay", curriculum_config["decay"])),
        "iterations": int(get_nested(finetune_src, "iterations", 1200)),
        "test_per_iter": int(get_nested(finetune_src, "test_per_iter", 50)),
        "centroid": bool(get_nested(finetune_src, "centroid", False)),
        "optim_mat": bool(get_nested(finetune_src, "optim_mat", False)),
        "shape_control": bool(get_nested(finetune_src, "shape_control", True)),
    }
    return curriculum_config, finetune_config


def save_config(config: Config, save_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.
    将配置保存到 YAML 文件

    Args:
        config: Configuration object
        save_path: Path to save the config
    """
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, allow_unicode=True)


def get_device(config: Config):
    """
    Get PyTorch device based on config.
    根据配置获取 PyTorch 设备
    """
    import torch

    device_str = config.get("device", "cuda")

    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif (
        device_str == "mps"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    else:
        return torch.device("cpu")


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    设置随机种子以确保可复现性
    """
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_environment(config: Config) -> None:
    """
    Setup training environment based on config.
    根据配置设置训练环境
    """
    import torch

    # Set seed
    if hasattr(config, "seed"):
        set_seed(config.seed)

    # Set default dtype
    if hasattr(config, "dtype"):
        if config.dtype == "float64":
            torch.set_default_dtype(torch.float64)
        else:
            torch.set_default_dtype(torch.float32)

    # Set matplotlib style
    try:
        import matplotlib.pyplot as plt

        plt.rcParams["figure.facecolor"] = "white"
        plt.rcParams["axes.facecolor"] = "white"
        plt.rcParams["savefig.facecolor"] = "white"
        plt.rcParams["text.color"] = "black"
        plt.rcParams["axes.labelcolor"] = "black"
        plt.rcParams["xtick.color"] = "black"
        plt.rcParams["ytick.color"] = "black"
        plt.rcParams["axes.edgecolor"] = "black"
        plt.rcParams["axes.titlecolor"] = "black"
    except ImportError:
        pass
