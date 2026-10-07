"""Configuration loading and path resolution."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(os.environ.get("SECOP_CONFIG_DIR", PROJECT_ROOT / "configs"))


def load_yaml(name: str) -> dict[str, Any]:
    """Load ``configs/<name>.yaml``."""
    with open(CONFIG_DIR / f"{name}.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def config_hash(*configs: dict[str, Any]) -> str:
    """Short, stable hash of one or more configuration dictionaries."""
    payload = json.dumps(configs, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def _resolve(value: str | os.PathLike, env_var: str) -> Path:
    path = Path(os.environ.get(env_var, value))
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Paths:
    """Locations of raw inputs and generated outputs."""

    data_dir: Path
    terridata_dir: Path
    output_dir: Path

    @classmethod
    def from_config(cls, data_config: dict[str, Any]) -> Paths:
        paths = data_config["paths"]
        return cls(
            data_dir=_resolve(paths["data_dir"], "SECOP_DATA_DIR"),
            terridata_dir=_resolve(paths["terridata_dir"], "SECOP_TERRIDATA_DIR"),
            output_dir=_resolve(paths["output_dir"], "SECOP_OUTPUT_DIR"),
        )

    @property
    def interim_dir(self) -> Path:
        return self.output_dir / "interim"

    @property
    def processed_dir(self) -> Path:
        return self.output_dir / "processed"

    @property
    def runs_dir(self) -> Path:
        return self.output_dir / "runs"

    def ensure(self) -> None:
        for directory in (self.interim_dir, self.processed_dir, self.runs_dir):
            directory.mkdir(parents=True, exist_ok=True)
