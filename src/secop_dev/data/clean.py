"""Cleaning of the unified SECOP contract table.

Port of ``notebooks/tesis_v1/ETL.ipynb`` (section "Procesar datos unificados"). It
produces the thesis file ``contratosSECOPCleaned.csv``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from secop_dev.utils import get_logger, strip_accents_upper

logger = get_logger(__name__)

VALUE_COLUMNS = ["tenderValue", "contractValue", "contractAditionalValue", "contractTotalValue"]
DATE_COLUMNS = ["contractDateSigned", "contractStartDate", "contractEndDate"]
MONTH_COLUMNS = ["contractMonthSigned", "contractMonthStart", "contractMonthEnd"]


def apply_corrections(df: pd.DataFrame, corrections: pd.DataFrame) -> pd.DataFrame:
    """Overwrite values of specific contracts. ``corrections`` has ``uid`` + target columns."""
    for _, row in corrections.iterrows():
        df.loc[df["uid"] == row["uid"], row.index[1:]] = row.values[1:]
    return df


def assign_region(departments: pd.Series, regions: dict[str, list[str]], default: str) -> pd.Series:
    lookup = {dept: region for region, depts in regions.items() for dept in depts}
    return departments.map(lambda d: lookup.get(d, default))


def classify_contract_method(methods: pd.Series, cfg: dict[str, Any]) -> pd.Series:
    """Map the legal procurement modality to OPEN / SIMPLIFIED / LIMITED / CLOSED / ..."""
    result = pd.Series(np.nan, index=methods.index, dtype=object)
    for label, pattern in cfg["contract_method_rules"]:
        result[methods.str.contains(pattern)] = label  # later rules override earlier ones
    result[result.isnull()] = cfg["contract_method_default"]
    return pd.Categorical(result, categories=cfg["contract_method_order"], ordered=True)


def duration_in_days(duration: pd.Series, unit: pd.Series, days_per_unit: dict[str, int]) -> pd.Series:
    """Contract term in days; unknown units give NaN."""
    return duration * unit.map(days_per_unit)


def clean_contracts(
    contracts: pd.DataFrame,
    corrections: dict[str, pd.DataFrame],
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Clean the unified contract table.

    Args:
        contracts: output of :func:`secop_dev.data.integrate.integrate_secop`
            (thesis file ``contratosSECOP.csv``).
        corrections: sheets of ``dataCorrections.xlsx`` keyed by name
            (``contractValues`` and ``contractDurations``).
        cfg: ``cleaning`` section of ``configs/data.yaml``.
    """
    df = contracts.rename(columns=cfg["column_names"])
    df = df.astype(
        {
            "contractYearSigned": "Int64",
            "contractValue": "float",
            "tenderValue": "float",
            "contractTotalValue": "float",
            "contractAditionalValue": "float",
            "additionalTimeDays": "Int64",
            "additionalTimeMonths": "Int64",
        }
    )
    for column in DATE_COLUMNS:
        df[column] = pd.to_datetime(df[column], errors="coerce")

    text_columns = df.select_dtypes(include=["object"]).columns
    df[text_columns] = df[text_columns].map(strip_accents_upper)

    df = apply_corrections(df, corrections["contractValues"])
    df = df.loc[~df["uid"].isin(cfg["removed_uids"])]

    # Values expressed in monthly minimum wages of the signing year.
    minimum_wage = cfg["minimum_wage"]
    df = df[df["contractYearSigned"].isin(minimum_wage)]
    df[MONTH_COLUMNS] = df[DATE_COLUMNS].apply(lambda column: column.dt.month)
    df[[f"{c}Mw" for c in VALUE_COLUMNS]] = df[VALUE_COLUMNS].div(
        df["contractYearSigned"].map(minimum_wage), axis=0
    )

    # Durations in days.
    df["contractDurationRange"] = df["contractDurationRange"].replace(cfg["duration_units"])
    df["contractDurationDays"] = duration_in_days(
        df["contractDuration"], df["contractDurationRange"], cfg["days_per_unit"]
    )
    df = df.drop(columns=["contractDuration", "contractDurationRange"])
    df["contractAditionalDuration"] = (
        df["additionalTimeDays"] + df["additionalTimeMonths"] * cfg["days_per_month"]
    )
    df = df.drop(columns=["additionalTimeMonths", "additionalTimeDays"])
    df["contractTotalDuration"] = df["contractDurationDays"] + df["contractAditionalDuration"]
    df = apply_corrections(df, corrections["contractDurations"])

    # Outcomes and derived ratios.
    df["haveCostDeviation"] = (df["contractAditionalValue"] > 0).astype(int)
    df["haveTimeDeviation"] = (df["contractAditionalDuration"] > 0).astype(int)
    df["haveTimeAndCostDeviation"] = (
        (df["contractAditionalValue"] > 0) & (df["contractAditionalDuration"] > 0)
    ).astype(int)
    df["projectIntensity"] = df["contractValue"] / df["contractDurationDays"]
    df["projectIntensityMw"] = df["contractValueMw"] / df["contractDurationDays"]
    df["awardGrowth"] = (df["contractValue"] - df["tenderValue"]) / df["tenderValue"]
    df["costDeviationPerc"] = (df["contractTotalValue"] - df["contractValue"]) / df["contractValue"]
    df["timeDeviationPerc"] = (df["contractTotalDuration"] - df["contractDurationDays"]) / df[
        "contractDurationDays"
    ]

    # Contracting entity.
    df["buyerLevel"] = df["buyerLevel"].replace(cfg["buyer_level"])
    df["buyerDepartment"] = df["buyerDepartment"].replace(cfg["buyer_department_aliases"])
    df["buyerRegion"] = assign_region(df["buyerDepartment"], cfg["regions"], cfg["default_region"])
    df["contractMethodType"] = classify_contract_method(df["contractMethodCountryLaw"], cfg)

    # Contractor.
    df["isBusinessGroup"] = df["isBusinessGroup"].str.contains(cfg["business_group_true_pattern"]).astype(int)
    df["isSME"] = df["isSME"].replace(cfg["sme_values"])
    df["isPostConflictContract"] = (
        df["isPostConflictContract"].str.contains(cfg["post_conflict_true_pattern"]).astype(int)
    )
    df["supplierDepartment"] = df["supplierDepartment"].replace(cfg["supplier_department_aliases"])
    df["supplierRegion"] = assign_region(df["supplierDepartment"], cfg["regions"], cfg["default_region"])

    logger.info("Clean contracts: %d rows, %d columns", *df.shape)
    return df
