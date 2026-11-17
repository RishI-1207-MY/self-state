"""Centralized logging and profile-aware configuration loader for SafeStep.

Supports separate configurations for Windows laptop development (`config.windows.yaml`)
and Raspberry Pi 4 Model B deployment (`config.rpi.yaml`), while preserving `config.yaml`.
"""

import logging
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict
import yaml


def get_project_root() -> Path:
    """Return the root directory of the SafeStep project."""
    return Path(__file__).resolve().parent.parent


def detect_active_profile() -> str:
    """Detect whether we are running on Raspberry Pi ('rpi') or development laptop ('windows')."""
    env_override = os.environ.get("SAFESTEP_ENV", "").strip().lower()
    if env_override in ("rpi", "raspberry_pi", "pi"):
        return "rpi"
    if env_override in ("windows", "win", "dev", "laptop"):
        return "windows"

    try:
        machine = platform.machine().lower()
        if "arm" in machine or "aarch64" in machine:
            model_file = Path("/proc/device-tree/model")
            if model_file.exists() and "raspberry pi" in model_file.read_text(encoding="utf-8", errors="ignore").lower():
                return "rpi"
    except Exception:
        pass
    return "windows"


def resolve_config_path(config_path: str | Path | None = None, profile: str | None = None) -> Path:
    """Resolve the configuration file path for the given or auto-detected profile."""
    root = get_project_root()
    if config_path is not None:
        return Path(config_path)

    active = (profile or os.environ.get("SAFESTEP_ENV", "")).strip().lower()
    if not active:
        # If running on actual Raspberry Pi without explicit override, use config.rpi.yaml
        if detect_active_profile() == "rpi" and (root / "config.rpi.yaml").exists():
            return root / "config.rpi.yaml"
        # Otherwise default to config.yaml (or config.windows.yaml)
        if (root / "config.yaml").exists():
            return root / "config.yaml"
        return root / "config.windows.yaml"

    if active in ("rpi", "raspberry_pi", "pi") and (root / "config.rpi.yaml").exists():
        return root / "config.rpi.yaml"
    if active in ("windows", "win", "dev", "laptop") and (root / "config.windows.yaml").exists():
        return root / "config.windows.yaml"

    return root / "config.yaml"


def load_config(config_path: str | Path | None = None, profile: str | None = None) -> Dict[str, Any]:
    """Load the YAML configuration file for the active or requested environment profile."""
    path = resolve_config_path(config_path=config_path, profile=profile)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data["_loaded_config_path"] = str(path)
    data["_active_profile"] = profile or detect_active_profile()
    return data


def save_config(config_data: Dict[str, Any], config_path: str | Path | None = None) -> None:
    """Save updated configuration back to the target YAML file (cleaning runtime keys)."""
    clean_copy = {k: v for k, v in config_data.items() if not str(k).startswith("_")}
    path = resolve_config_path(config_path=config_path)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(clean_copy, f, sort_keys=False, allow_unicode=True)

    # Also keep config.windows.yaml in sync when saving on Windows development machine
    root = get_project_root()
    if detect_active_profile() == "windows" and path.name == "config.yaml":
        win_path = root / "config.windows.yaml"
        if win_path.exists():
            win_copy = dict(clean_copy)
            win_copy["profile"] = "windows_development"
            with open(win_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(win_copy, f, sort_keys=False, allow_unicode=True)


def get_logger(name: str = "SafeStep", level: int = logging.INFO) -> logging.Logger:
    """Create or retrieve a configured logger that logs to both console and file."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    try:
        log_dir = get_project_root() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "safestep.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception:
        pass

    return logger
