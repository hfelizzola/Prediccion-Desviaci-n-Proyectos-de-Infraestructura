"""Logging, text normalisation and run-metadata helpers."""

from __future__ import annotations

import logging
import platform
import subprocess
import sys
import unicodedata
from datetime import UTC, datetime
from importlib import metadata
from typing import Any

from secop_dev.config import PROJECT_ROOT

TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "imbalanced-learn",
    "xgboost",
    "optuna",
    "shap",
    "statsmodels",
    "scipy",
)


def get_logger(name: str) -> logging.Logger:
    """Module logger; configures a single stream handler for the package."""
    root = logging.getLogger("secop_dev")
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root.addHandler(handler)
        root.setLevel(logging.INFO)
    return logging.getLogger(name)


def strip_accents_upper(text: Any) -> Any:
    """Upper-case a string and remove combining accents (NFD); non-strings pass through."""
    if isinstance(text, str):
        text = text.upper()
        return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    return text


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def run_metadata(**extra: Any) -> dict[str, Any]:
    """Environment information stored next to every generated artifact."""
    versions = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        **extra,
    }
