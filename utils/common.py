"""
Shared utilities used across data/, model/, eval/, deploy/.
Keep this dependency-light — it must import cleanly even before
torch/cv2 are guaranteed to be installed (e.g. during data prep on
a bare machine).
"""
from __future__ import annotations

import os
import random
import json
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: str | Path = None) -> dict[str, Any]:
    """Load config.yaml. All scripts should call this instead of
    hardcoding hyperparameters."""
    if config_path is None:
        config_path = REPO_ROOT / "config.yaml"
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_path(relative_path: str) -> Path:
    """Resolve a path from config.yaml (which are given relative to
    repo root) into an absolute Path, creating parent dirs as needed."""
    p = REPO_ROOT / relative_path
    return p


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str | Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def get_device():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    except ImportError:
        return None


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    """Render a simple markdown table from headers + rows."""
    def fmt_row(r):
        return "| " + " | ".join(str(x) for x in r) + " |"

    lines = [fmt_row(headers), "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append(fmt_row(r))
    return "\n".join(lines)
