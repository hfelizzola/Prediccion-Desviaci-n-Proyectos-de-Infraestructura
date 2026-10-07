"""SHAP values of the full-sample models of the thesis.

Port of the "Interpretación de Variables" sections of
``notebooks/tesis_v1/model_analysis_V2.ipynb``.

Note (ROADMAP M9): as in the thesis, the explained models are refitted on the whole
sample with their own preprocessing (no resampling, unscaled numbers for trees) and
the logistic regression uses default hyper-parameters. They are not the pipelines
evaluated on the test set; v2 explains the evaluated pipelines instead.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from secop_dev.models.train import scenario_columns, select_sample
from secop_dev.utils import get_logger

logger = get_logger(__name__)


def design_matrices(X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(numbers imputed + one-hot, numbers imputed and scaled + one-hot)."""
    X_num = X.select_dtypes(include=["int64", "float64"])
    X_cat = X.select_dtypes(include=["object"])
    X_num = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X_num), columns=X_num.columns)
    X_num_scaled = pd.DataFrame(StandardScaler().fit_transform(X_num), columns=X_num.columns)
    encoder = OneHotEncoder(handle_unknown="ignore", drop="if_binary")
    X_cat_enc = pd.DataFrame(encoder.fit_transform(X_cat).toarray(), columns=encoder.get_feature_names_out())
    return pd.concat([X_num, X_cat_enc], axis=1), pd.concat([X_num_scaled, X_cat_enc], axis=1)


def _rf_params(params: dict[str, Any]) -> dict[str, Any]:
    params = dict(params)
    if "max_features_float" in params:
        params["max_features"] = params.pop("max_features_float")
    params.pop("max_features_type", None)
    return params


def compute_shap(
    table: pd.DataFrame,
    run_dir: Path,
    target: str,
    features_config: dict[str, Any],
    experiments_config: dict[str, Any],
    out_dir: Path,
) -> dict[str, Path]:
    """Fit the SHAP models of ``experiments.yaml: shap`` and store values + design matrix."""
    cfg = experiments_config
    seed = cfg["seed"]
    scenario = cfg["shap"]["scenario"]
    sample = select_sample(table, cfg["sample_filters"])
    X = sample[scenario_columns(features_config, cfg)[scenario]].copy()
    y = sample[cfg["targets"][target]].copy()
    X_tree, X_std = design_matrices(X)

    def best_params(model_key: str) -> dict[str, Any]:
        path = run_dir / target / model_key / scenario / "best_params.json"
        return json.loads(path.read_text(encoding="utf-8"))

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for model_key in cfg["shap"]["models"]:
        logger.info("SHAP %s / %s / %s", target, model_key, scenario)
        if model_key == "xgb":
            model = XGBClassifier(**best_params("xgb"), random_state=seed).fit(X_tree, y)
            explainer, data = shap.TreeExplainer(model), X_tree
            values = explainer.shap_values(data)
        elif model_key == "rf":
            model = RandomForestClassifier(random_state=seed).set_params(**_rf_params(best_params("rf")))
            model.fit(X_tree, y)
            explainer, data = shap.TreeExplainer(model), X_tree
            values = explainer.shap_values(data)[:, :, 1]  # positive class
        elif model_key == "lr":
            model = LogisticRegression().fit(X_std, y)
            explainer, data = shap.LinearExplainer(model, X_std), X_std
            values = explainer.shap_values(data)
        else:
            raise ValueError(f"SHAP not implemented for model '{model_key}'")

        expected = np.atleast_1d(explainer.expected_value)[-1]
        path = out_dir / f"shap_{target}_{model_key}.joblib"
        joblib.dump({"values": values, "expected_value": expected, "data": data, "model": model}, path)
        importance = pd.Series(np.abs(values).mean(axis=0), index=data.columns).sort_values(ascending=False)
        importance.rename("mean_abs_shap").to_csv(out_dir / f"shap_importance_{target}_{model_key}.csv")
        outputs[model_key] = path
    return outputs
