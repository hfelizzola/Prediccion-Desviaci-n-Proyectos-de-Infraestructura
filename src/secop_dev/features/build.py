"""Assembly of the training table (thesis file ``trainData.csv``)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from secop_dev.data.terridata import complete_years, select_indicators
from secop_dev.features.owner import add_owner_features
from secop_dev.features.project import add_project_features, apply_sample_filters, remove_outliers
from secop_dev.utils import get_logger

logger = get_logger(__name__)


def add_territorial_features(df: pd.DataFrame, terridata: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Municipal management (previous year) and departmental economic indicators."""
    management = complete_years(
        select_indicators(terridata, cfg["management_indicators"]), cfg["management_added_years"]
    )
    df = pd.merge(
        df,
        management,
        left_on=["buyerDepartment", "buyerLocation", "lastYear"],
        right_on=["Departamento", "Entidad", "Año"],
        how="left",
    )
    years = range(cfg["economic_years"]["start"], cfg["economic_years"]["end"] + 1)
    economy = complete_years(
        select_indicators(terridata, cfg["economic_indicators"], years), cfg["economic_added_years"]
    )
    df = pd.merge(
        df, economy, left_on=["buyerDepartment", "lastYear"], right_on=["Departamento", "Año"], how="left"
    )
    return df.rename(columns=cfg["rename"])


def feature_columns(features_config: dict[str, Any]) -> list[str]:
    """Identifier, G1-G4 and outcome columns, in training-table order."""
    columns = list(features_config["id_columns"])
    for group in features_config["groups"].values():
        columns += group["columns"]
    return columns


def build_training_table(
    contracts: pd.DataFrame,
    penalties: pd.DataFrame,
    terridata: pd.DataFrame,
    data_config: dict[str, Any],
    features_config: dict[str, Any],
) -> pd.DataFrame:
    """Clean contracts -> filtered sample with G1-G4 features and outcomes."""
    cfg = data_config["features"]
    df = apply_sample_filters(contracts, cfg)
    df = remove_outliers(df, cfg["outliers"])
    df = add_project_features(df, cfg)
    df = add_owner_features(df, penalties)
    df = add_territorial_features(df, terridata, cfg["terridata"])
    df = df.rename(columns=cfg["rename"])
    table = df[feature_columns(features_config)].copy()
    logger.info("Training table: %d rows, %d columns", *table.shape)
    return table


def to_v1_format(table: pd.DataFrame, features_config: dict[str, Any]) -> pd.DataFrame:
    """Rename columns with the thesis group suffix, e.g. ``contractValueMw(G1)``."""
    suffixes = {
        column: f"{column}({group})"
        for group, spec in features_config["groups"].items()
        for column in spec["columns"]
    }
    return table.rename(columns=suffixes)


def from_v1_format(table: pd.DataFrame) -> pd.DataFrame:
    """Drop the thesis group suffix from column names."""
    return table.rename(columns=lambda c: c.split("(")[0] if c.endswith(")") else c)
