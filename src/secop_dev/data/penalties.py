"""Fines and sanctions per contracting entity and year.

Port of ``analizar_multas`` in ``notebooks/tesis_v1/ETL.ipynb``. The thesis version
looped over every entity-year pair and scanned all penalties for each one; this
version sorts the penalty years once per entity and counts with binary search,
producing the same table.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


def penalties_by_buyer_year(
    penalties: pd.DataFrame,
    buyer_ids: Sequence[Any],
    start_year: int,
    end_year: int,
    recent_window: int = 2,
) -> pd.DataFrame:
    """Accumulated and recent penalties of each entity, for every year in the range.

    For entity *e* and year *t*:

    * ``MULTAS_ACUMULADAS``: number of penalties of *e* dated before year *t*.
    * ``MULTAS_RECIENTES``: 1 if *e* had a penalty in ``[t - recent_window, t)``.

    Args:
        penalties: columns ``NIT_ENTIDAD``, ``PENALTY_DATE`` and optionally ``buyerName``.
        buyer_ids: entity identifiers (NIT), in output order.
    """
    years = np.arange(start_year, end_year + 1)
    penalty_years = pd.to_datetime(penalties["PENALTY_DATE"]).dt.year
    valid = penalty_years.notna()
    years_by_buyer = {
        nit: np.sort(group.to_numpy())
        for nit, group in penalty_years[valid].groupby(penalties.loc[valid, "NIT_ENTIDAD"])
    }

    buyers = pd.unique(pd.Series(list(buyer_ids)))
    accumulated, recent = [], []
    empty = np.array([])
    for nit in buyers:
        sorted_years = years_by_buyer.get(nit, empty)
        before = np.searchsorted(sorted_years, years, side="left")
        before_window = np.searchsorted(sorted_years, years - recent_window, side="left")
        accumulated.append(before)
        recent.append((before - before_window > 0).astype(int))

    result = pd.DataFrame(
        {
            "NIT_ENTIDAD": np.repeat(buyers, len(years)),
            "AÑO": np.tile(years, len(buyers)),
            "MULTAS_ACUMULADAS": np.concatenate(accumulated) if accumulated else [],
            "MULTAS_RECIENTES": np.concatenate(recent) if recent else [],
        }
    )
    if "buyerName" in penalties.columns:
        names = (
            penalties[["NIT_ENTIDAD", "buyerName"]]
            .drop_duplicates()
            .set_index("NIT_ENTIDAD")["buyerName"]
            .to_dict()
        )
        result["NOMBRE_ENTIDAD"] = result["NIT_ENTIDAD"].map(names)
    return result
