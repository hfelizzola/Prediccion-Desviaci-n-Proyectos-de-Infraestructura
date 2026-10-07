"""Integration of SECOP I processes and SECOP II contracts into one contract table.

Port of ``notebooks/tesis_v1/ETL.ipynb`` (sections "Extracción SECOP I/II" and
"Unificar SECOP I y SECOP II").
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def _parse_dates(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        df[column] = pd.to_datetime(df[column])
    return df


def prepare_secop1(processes: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    df = _parse_dates(processes.copy(), cfg["date_columns"])
    df["anno_firma"] = df["fecha_firma"].dt.year
    df["es_grupo"] = (
        df["nom_razon_social_contratista"].str.upper().str.contains(cfg["business_group_pattern"])
    )
    return df.rename(columns={"orden": "orden_entidad", "dpto_y_muni_contratista": "departamento_proveedor"})


def prepare_secop2(contracts: pd.DataFrame, processes: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Join SECOP II contracts with their (single-contract) processes."""
    df = _parse_dates(contracts.copy(), cfg["date_columns"])
    df["anno_firma"] = df["fecha_firma"].dt.year

    # Keep processes that appear once after removing exact duplicates.
    process_counts = processes.drop_duplicates()["id_proceso"].value_counts().reset_index(name="total")
    unique_ids = list(process_counts[process_counts["total"] == 1]["id_proceso"])
    df = df.merge(processes[processes["id_proceso"].isin(unique_ids)], on="id_proceso", how="inner")

    df["diferencia_porcentual"] = (df["cuantia_contrato"] - df["cuantia_proceso"]) / df["cuantia_proceso"]
    df["valor_total_de_adiciones"] = df["valor_contrato_con_adiciones"] - df["cuantia_contrato"]
    df["tiene_adiciones"] = df["valor_total_de_adiciones"] > 0
    df["porcentaje_adicion"] = df["valor_total_de_adiciones"] / df["cuantia_contrato"]

    # Contract duration "<n> <unit>" from the contract, falling back to the process.
    duration_columns = ["plazo_de_ejec_del_contrato", "rango_de_ejec_del_contrato"]
    df[duration_columns] = df["plazo_de_ejec_del_contrato_x"].str.extract(cfg["duration_pattern"])
    df["plazo_de_ejec_del_contrato"] = pd.to_numeric(df["plazo_de_ejec_del_contrato"])
    missing_unit = df["rango_de_ejec_del_contrato"].isnull()
    df.loc[missing_unit, "rango_de_ejec_del_contrato"] = df.loc[missing_unit, "rango_de_ejec_del_contrato_y"]
    missing_term = df["plazo_de_ejec_del_contrato"].isnull()
    df.loc[missing_term, "plazo_de_ejec_del_contrato"] = df.loc[missing_term, "plazo_de_ejec_del_contrato_y"]
    df["tiempo_adiciones_en_meses"] = 0

    df["orden_entidad_SECOPII"] = df["orden_entidad_SECOPII"].str.upper()
    return df.rename(columns={"orden_entidad_SECOPII": "orden_entidad"})


def integrate_secop(
    secop1_processes: pd.DataFrame,
    secop2_contracts: pd.DataFrame,
    secop2_processes: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Unified contract table with the columns shared by SECOP I and SECOP II."""
    secop2 = prepare_secop2(secop2_contracts, secop2_processes, cfg)
    secop1 = prepare_secop1(secop1_processes, cfg)
    secop2["base_de_datos"] = "SECOPII"
    secop1["base_de_datos"] = "SECOPI"
    return pd.concat([secop2, secop1], axis=0, join="inner")
