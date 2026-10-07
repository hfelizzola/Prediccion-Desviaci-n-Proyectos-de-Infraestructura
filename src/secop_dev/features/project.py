"""Sample filters and project-level variables (group G1)."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from secop_dev.config import PROJECT_ROOT
from secop_dev.utils import get_logger

logger = get_logger(__name__)


def _log_filter(name: str, before: int, after: int) -> None:
    logger.info("Filter %-22s %6d -> %6d (removed %d)", name, before, after, before - after)


def apply_sample_filters(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Bid ratio, cost-deviation and duration filters of the thesis."""
    df = df.copy()
    df["bidContractRatio"] = df["contractValueMw"] / df["tenderValueMw"]

    n = len(df)
    bid = cfg["bid_ratio"]
    df = df.loc[(df["bidContractRatio"] > bid["min"]) & (df["bidContractRatio"] < bid["max"])]
    _log_filter("bid/contract ratio", n, len(df))

    n = len(df)
    cost = cfg["cost_deviation"]
    df = df.loc[(df["costDeviationPerc"] >= cost["min"]) & (df["costDeviationPerc"] <= cost["max"])]
    _log_filter("cost deviation", n, len(df))

    levels = cfg["cost_deviation_levels"]
    df["costDeviationLevel"] = "low"
    df.loc[
        (df["costDeviationPerc"] > levels["low_max"]) & (df["costDeviationPerc"] <= levels["medium_max"]),
        "costDeviationLevel",
    ] = "medium"
    df.loc[df["costDeviationPerc"] > levels["medium_max"], "costDeviationLevel"] = "high"
    df["costDeviationLevel"] = df["costDeviationLevel"].astype("category")

    n = len(df)
    duration = cfg["duration_days"]
    df = df.loc[
        (df["contractDurationDays"] >= duration["min"]) & (df["contractDurationDays"] < duration["max"])
    ]
    _log_filter("contract duration", n, len(df))
    return df


def outlier_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Preprocessing used by the thesis outlier detector (``create_preprocessing_pipeline``)."""
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent", add_indicator=True)),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="passthrough",
    )


def detect_outliers(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.Series:
    """Boolean mask of outlier contracts.

    Note (ROADMAP M1): the detector uses outcome variables, as in the thesis.
    """
    method = cfg["method"]
    if method == "frozen_v1":
        frozen = pd.read_csv(PROJECT_ROOT / cfg["frozen_uids_file"], dtype={"uid": str})
        return df["uid"].isin(set(frozen["uid"]))
    if method == "isolation_forest":
        X = df[cfg["columns"]].copy()
        numeric = X.select_dtypes(include=["int64", "float64"]).columns
        categorical = X.select_dtypes(include=["object", "string"]).columns
        model = Pipeline(
            [
                ("preprocesor", outlier_preprocessor(list(numeric), list(categorical))),
                (
                    "classifier",
                    IsolationForest(contamination=cfg["contamination"], random_state=cfg["random_state"]),
                ),
            ]
        )
        model.fit(X)
        return pd.Series(model.predict(X) == -1, index=df.index)
    raise ValueError(f"Unknown outlier method: {method}")


def remove_outliers(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    outliers = detect_outliers(df, cfg)
    _log_filter(f"outliers ({cfg['method']})", len(df), int((~outliers).sum()))
    return df.loc[~outliers].copy()


def frozen_outliers_from_v1(pre_outlier_uids: pd.Series, v1_training_uids: pd.Series) -> pd.DataFrame:
    """Recover the contracts the thesis Isolation Forest removed (it had no seed)."""
    removed = sorted(set(pre_outlier_uids) - set(v1_training_uids))
    return pd.DataFrame({"uid": removed})


def assign_work_category(description: pd.Series, patterns: dict[str, str], default: str) -> pd.Series:
    """Work type from keywords; a later category overrides an earlier one (as in v1)."""
    category = pd.Series("", index=description.index, dtype=object)
    matched_any = pd.Series(0, index=description.index)
    for name, pattern in patterns.items():
        matches = description.str.contains(pattern)
        category[matches == True] = name  # noqa: E712  (NaN descriptions do not match)
        matched_any = matched_any + matches
    category[matched_any == 0] = default
    return category


def add_project_features(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Location fixes, electoral cycle and work category."""
    df = df.copy()
    df["contractEndDate"] = pd.to_datetime(df["contractEndDate"])
    df["contractYearEnd"] = df["contractEndDate"].dt.year
    df["buyerDepartment"] = df["buyerDepartment"].replace(cfg["buyer_department_aliases"])
    df["buyerLocation"] = df["buyerLocation"].replace(cfg["buyer_location_aliases"])
    df["electoralYear"] = df["contractYearSigned"].isin(cfg["electoral_years"]).astype(int)
    df["preelectoralYear"] = df["contractYearSigned"].isin(cfg["preelectoral_years"]).astype(int)
    df["workCategory"] = assign_work_category(
        df["contractDescription"], cfg["work_categories"], cfg["work_category_default"]
    )
    return df
