import numpy as np
import pandas as pd
import pytest

from secop_dev.models.pipelines import make_pipeline, split_feature_types
from secop_dev.models.search_spaces import MODELS
from secop_dev.models.train import scenario_columns, select_sample
from secop_dev.models.tuning import apply_params, tune_and_evaluate


@pytest.fixture
def synthetic():
    rng = np.random.default_rng(42)
    n = 240
    X = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "cat": rng.choice(["A", "B", "C"], size=n).astype(object),
        }
    )
    y = pd.Series(((X["x1"] + (X["cat"] == "A") + rng.normal(scale=0.5, size=n)) > 0.5).astype(int))
    return X, y


def test_scenario_columns_drop_tournament_losers(features_config, experiments_config):
    columns = scenario_columns(features_config, experiments_config)
    assert set(columns) == {"S1", "S2", "S3", "S4", "S5"}
    assert "contractValueMw" not in columns["S1"] and "projectIntensityMw" in columns["S1"]
    assert columns["S5"] == columns["S1"] + [c for c in columns["S2"] if c not in columns["S1"]] + [
        c for c in columns["S3"] if c not in columns["S1"]
    ] + [c for c in columns["S4"] if c not in columns["S1"]]
    assert len(columns["S1"]) == 10


def test_select_sample(experiments_config):
    df = pd.DataFrame({"contractValueMw": [999, 1000, 5000], "totalCumContracts": [9, 9, 4]})
    assert len(select_sample(df, experiments_config["sample_filters"])) == 1


def test_split_feature_types_warns_on_unsupported(caplog):
    X = pd.DataFrame({"a": [1.0], "b": ["x"], "flag": [True]})
    numeric, categorical = split_feature_types(X)
    assert numeric == ["a"] and categorical == ["b"]
    assert "flag" in caplog.text


def test_rf_float_branch_is_ignored_on_refit_as_in_v1(synthetic):
    """Documents ROADMAP M8: the refit keeps the default max_features."""
    X, _ = synthetic
    pipeline = make_pipeline(X, MODELS["rf"].build(42))
    apply_params(pipeline, {"max_features_type": "float", "max_features_float": 0.2})
    assert pipeline.named_steps["classifier"].max_features == "sqrt"


@pytest.mark.parametrize("model_key", ["lr", "rf", "xgb", "knn"])
def test_tune_and_evaluate_smoke(synthetic, model_key):
    X, y = synthetic
    spec = MODELS[model_key]
    half = len(X) // 2
    result = tune_and_evaluate(
        X.iloc[:half],
        y.iloc[:half],
        X.iloc[half:],
        y.iloc[half:],
        spec.build(42),
        spec.search_space,
        n_trials=2,
        cv_folds=3,
        n_jobs=1,
    )
    assert set(result.metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert 0.5 < result.metrics["roc_auc"] <= 1.0
    assert len(result.trials) == 2
