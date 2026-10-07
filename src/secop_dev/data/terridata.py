"""TerriData (DNP) territorial indicators.

Port of the "Terridata" section of ``notebooks/tesis_v1/featureEngineering_V2.ipynb``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

KEY_COLUMNS = ["Departamento", "Entidad", "Indicador"]
VALUE_COLUMN = "Dato Numérico"

DTYPES = {
    "Código Departamento": "int32",
    "Departamento": "object",
    "Código Entidad": "int32",
    "Entidad": "object",
    "Dimensión": "object",
    "Subcategoría": "object",
    "Indicador": "object",
    VALUE_COLUMN: "object",  # converted below (Spanish number format)
    "Dato Cualitativo": "object",
    "Año": "int32",
    "Mes": "int32",
    "Fuente": "object",
    "Unidad de Medida": "object",
}


def load_terridata(files: Sequence[Path], separator: str = "|") -> pd.DataFrame:
    """Load TerriData exports and normalise numbers and text."""
    data = pd.concat([pd.read_csv(f, sep=separator, encoding="utf-8", dtype=DTYPES) for f in files])
    # "1.234,5" -> 1234.5
    data[VALUE_COLUMN] = (
        data[VALUE_COLUMN].str.replace(".", "", regex=False).str.replace(",", ".", regex=False).astype(float)
    )
    for column in data.select_dtypes(include="object").columns:
        data[column] = (
            data[column]
            .str.upper()
            .str.normalize("NFKD")
            .str.encode("ascii", errors="ignore")
            .str.decode("utf-8")
        )
    return data


def select_indicators(
    data: pd.DataFrame, indicators: Sequence[str], years: Sequence[int] | None = None
) -> pd.DataFrame:
    mask = data["Indicador"].isin(indicators)
    if years is not None:
        mask &= data["Año"].isin(years)
    return data.loc[mask, ["Departamento", "Entidad", "Año", "Indicador", VALUE_COLUMN]].copy()


def complete_years(data: pd.DataFrame, added_years: Sequence[int]) -> pd.DataFrame:
    """Add ``added_years`` and impute every gap with the territory-indicator mean.

    Returns one row per (Departamento, Entidad, Año) and one column per indicator.

    Note (ROADMAP M3): the mean uses every available year, including later ones.
    Rows added for a year that already exists are averaged with the observed value
    by ``pivot_table``. Both behaviours are kept to reproduce the thesis results.
    """
    new_rows = data[KEY_COLUMNS].drop_duplicates()
    for year in added_years:
        new_rows[year] = np.nan
    new_rows = new_rows.melt(id_vars=KEY_COLUMNS, var_name="Año", value_name=VALUE_COLUMN)
    new_rows["Año"] = new_rows["Año"].astype(int)
    new_rows = new_rows.drop(columns=VALUE_COLUMN)

    completed = pd.concat([data, new_rows], ignore_index=True)
    completed = completed.sort_values(KEY_COLUMNS + ["Año"])
    group_means = (
        completed.groupby(KEY_COLUMNS)[VALUE_COLUMN]
        .mean()
        .reset_index()
        .rename(columns={VALUE_COLUMN: "promedio"})
    )
    completed = completed.merge(group_means, on=KEY_COLUMNS, how="left")
    completed[VALUE_COLUMN] = completed[VALUE_COLUMN].fillna(completed["promedio"])
    completed = completed.drop(columns="promedio")

    return completed.pivot_table(
        index=["Departamento", "Entidad", "Año"], columns="Indicador", values=VALUE_COLUMN
    ).reset_index()
