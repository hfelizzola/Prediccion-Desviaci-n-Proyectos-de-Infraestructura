"""Comparison of a new run with the metrics reported in the thesis (v1.0.0)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from secop_dev.models.tuning import METRICS

V1_FILES = {"cost": "metrics_cost_deviation.xlsx", "time": "metrics_time_deviation.xlsx"}


def compare_with_v1(metrics: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Side-by-side metrics (in %) and differences ``new - v1`` per target, model and scenario."""
    frames = []
    for target, filename in V1_FILES.items():
        path = data_dir / filename
        if not path.exists():
            continue
        v1 = pd.read_excel(path).assign(target=target)
        frames.append(v1)
    if not frames:
        raise FileNotFoundError(f"No thesis metric files found in {data_dir}")
    v1 = pd.concat(frames, ignore_index=True)

    keys = ["target", "model", "stage"]
    merged = metrics.merge(v1[keys + METRICS], on=keys, how="left", suffixes=("", "_v1"))
    for metric in METRICS:
        merged[f"{metric}_diff"] = merged[metric] - merged[f"{metric}_v1"]
    ordered = keys + [c for m in METRICS for c in (m, f"{m}_v1", f"{m}_diff")]
    return merged[ordered].sort_values(keys).reset_index(drop=True)
