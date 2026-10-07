"""Scenario x model comparison with cross-validation and two-way ANOVA.

Port of the "Análisis Inferencial" section of ``notebooks/tesis_v1/model_analysis_V2.ipynb``.

Note (ROADMAP M6): the folds reuse the whole sample, including the test set used to
report hold-out metrics, and CV folds are not independent observations. Kept to
reproduce the thesis; replaced by corrected resampled tests in v2.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from secop_dev.models.pipelines import make_pipeline
from secop_dev.models.search_spaces import MODELS
from secop_dev.models.train import scenario_columns, select_sample
from secop_dev.models.tuning import apply_params
from secop_dev.utils import get_logger

logger = get_logger(__name__)


def cross_validate_scenarios(
    table: pd.DataFrame,
    run_dir: Path,
    target: str,
    features_config: dict[str, Any],
    experiments_config: dict[str, Any],
) -> pd.DataFrame:
    """Per-fold scores of every model x scenario with the tuned hyper-parameters of ``run_dir``."""
    cfg = experiments_config
    seed = cfg["seed"]
    sample = select_sample(table, cfg["sample_filters"])
    columns = scenario_columns(features_config, cfg)
    y = sample[cfg["targets"][target]]
    cv = StratifiedKFold(n_splits=cfg["doe"]["cv_folds"], shuffle=True, random_state=seed)

    frames = []
    for model_key in cfg["models"]:
        spec = MODELS[model_key]
        for scenario in cfg["scenarios"]:
            params_file = run_dir / target / model_key / scenario / "best_params.json"
            if not params_file.exists():
                logger.warning("Missing %s; skipped", params_file)
                continue
            X = sample[columns[scenario]]
            pipeline = make_pipeline(
                X, spec.build(seed), resampling=cfg["tuning"]["resampling"], random_state=seed
            )
            apply_params(pipeline, json.loads(params_file.read_text(encoding="utf-8")))
            scores = pd.DataFrame(cross_validate(pipeline, X, y, cv=cv, scoring=cfg["doe"]["scoring"]))
            scores["model"] = spec.name
            scores["setting"] = scenario
            frames.append(scores)
            logger.info(
                "DOE %s / %s / %s: roc_auc %.4f", target, spec.name, scenario, scores["test_roc_auc"].mean()
            )
    return pd.concat(frames, ignore_index=True)


def two_way_anova(cv_scores: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """``metric ~ C(setting) + C(model)`` (type II) for every metric, scores in %."""
    data = cv_scores.copy()
    rows = []
    for metric in metrics:
        column = f"test_{metric}"
        data[column] = (data[column] * 100).round(2)
        table = anova_lm(ols(f"{column} ~ C(setting) + C(model)", data=data).fit(), typ=2)
        result = table[["F", "PR(>F)"]].iloc[:-1].reset_index(names="Factor")
        result.insert(0, "Metric", metric)
        rows.append(result)
    return pd.concat(rows, ignore_index=True)
