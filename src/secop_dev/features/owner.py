"""Contracting-entity (owner) indicators, group G2.

Port of the "Indicadores Históricos de la entidad" section of
``notebooks/tesis_v1/featureEngineering_V2.ipynb``. Entities are identified by
``buyerName`` and indicators are taken from the previous year, as in the thesis
(see ROADMAP M2, M3 and M11 for the known limitations).
"""

from __future__ import annotations

import pandas as pd


def _hhi(shares: pd.DataFrame, value: str, keys: list[str]) -> pd.Series:
    totals = shares.groupby(keys)[value].transform("sum")
    return ((shares[value] / totals) * 100) ** 2


def herfindahl_index(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Supplier concentration (HHI, 0-10000) per entity and year.

    Returns two tables with columns ``year, buyerName`` and ``hhi_num`` (share of
    number of contracts) or ``hhi_valor`` (share of contract value).
    """
    keys = ["year", "buyerName"]
    data = df.assign(year=df["contractYearSigned"])

    by_count = data.groupby(keys + ["supplierName"])["uid"].count().reset_index(name="num_contratos")
    by_count["sq"] = _hhi(by_count, "num_contratos", keys)
    hhi_num = by_count.groupby(keys)["sq"].sum().reset_index(name="hhi_num")

    by_value = data.groupby(keys + ["supplierName"])["contractValue"].sum().reset_index()
    by_value["sq"] = _hhi(by_value, "contractValue", keys)
    hhi_value = by_value.groupby(keys)["sq"].sum().reset_index(name="hhi_valor")
    return hhi_num, hhi_value


def previous_year_hhi(df: pd.DataFrame) -> pd.DataFrame:
    """HHI of the previous year for each contract (``uid``).

    Missing values are imputed with the entity median over all years and then with
    the global median (ROADMAP M3: includes later years).
    """
    hhi_num, hhi_value = herfindahl_index(df)
    hhi_num = hhi_num.rename(columns={"year": "anio_calculo", "hhi_num": "hhi_num_calculado"})
    hhi_value = hhi_value.rename(columns={"year": "anio_calculo", "hhi_valor": "hhi_valor_calculado"})
    combined = pd.merge(hhi_num, hhi_value, on=["anio_calculo", "buyerName"], how="outer")

    result = df[["uid", "buyerName", "contractYearSigned"]].copy()
    result["anio_anterior"] = result["contractYearSigned"] - 1
    result = pd.merge(
        result,
        combined,
        left_on=["anio_anterior", "buyerName"],
        right_on=["anio_calculo", "buyerName"],
        how="left",
    ).rename(
        columns={
            "hhi_num_calculado": "hhi_num_anio_anterior",
            "hhi_valor_calculado": "hhi_valor_anio_anterior",
        }
    )

    for column, table, calc in (
        ("hhi_num_anio_anterior", hhi_num, "hhi_num_calculado"),
        ("hhi_valor_anio_anterior", hhi_value, "hhi_valor_calculado"),
    ):
        entity_median = result["buyerName"].map(table.groupby("buyerName")[calc].median())
        result[column] = result[column].fillna(entity_median).fillna(table[calc].median())
    return result[["uid", "hhi_num_anio_anterior", "hhi_valor_anio_anterior"]]


HISTORY_COLUMNS = [
    "totalCumContracts",
    "histPctDesvTiempo",
    "histPctDesvCost",
    "histPctDesvTimeAndCost",
    "histPctDirectContracts",
    "histPctTenderContracts",
    "histAvgProjectIntensity",
    "histAvgBidRatio",
    "histAvgAwardGrowth",
    "histAvgTimeDeviation",
    "histAvgCostDeviation",
    "histAvgContractValue",
    "histAvgContractDuration",
]


def cumulative_history(df: pd.DataFrame) -> pd.DataFrame:
    """Cumulative indicators of each entity up to (and including) every year.

    One row per ``buyerName`` x ``year`` (all years present in ``df``). Entities
    without contracts up to a year get zeros. Requires ``closedMethod`` and
    ``tenderMethod`` indicator columns.
    """
    years = sorted(df["contractYearSigned"].unique())
    rows = []
    for buyer in df["buyerName"].unique():
        contracts = df[df["buyerName"] == buyer]
        for year in years:
            cum = contracts[contracts["contractYearSigned"] <= year]
            n = len(cum)
            if n == 0:
                rows.append({"buyerName": buyer, "year": year, **dict.fromkeys(HISTORY_COLUMNS, 0)})
                continue
            rows.append(
                {
                    "buyerName": buyer,
                    "year": year,
                    "totalCumContracts": n,
                    "histPctDesvTiempo": cum["haveTimeDeviation"].sum() / n * 100,
                    "histPctDesvCost": cum["haveCostDeviation"].sum() / n * 100,
                    "histPctDesvTimeAndCost": cum["haveTimeAndCostDeviation"].sum() / n * 100,
                    "histPctDirectContracts": cum["closedMethod"].sum() / n * 100,
                    "histPctTenderContracts": cum["tenderMethod"].sum() / n * 100,
                    "histAvgProjectIntensity": cum["projectIntensityMw"].mean(),
                    "histAvgBidRatio": cum["bidContractRatio"].mean(),
                    "histAvgAwardGrowth": cum["awardGrowth"].mean(),
                    "histAvgTimeDeviation": cum["timeDeviationPerc"].mean(),
                    "histAvgCostDeviation": cum["costDeviationPerc"].mean(),
                    "histAvgContractValue": cum["contractValueMw"].mean(),
                    "histAvgContractDuration": cum["contractDurationDays"].mean(),
                }
            )
    history = pd.DataFrame(rows)
    history["year"] = history["year"].astype(int)
    return history.sort_values(["buyerName", "year"])


def add_owner_features(df: pd.DataFrame, penalties: pd.DataFrame) -> pd.DataFrame:
    """Previous-year HHI, cumulative history and penalties of the contracting entity."""
    df = pd.merge(df, previous_year_hhi(df), on="uid", how="left")

    df["closedMethod"] = (df["contractMethodType"] == "CLOSED").astype(int)
    df["tenderMethod"] = (df["contractMethodType"] == "OPEN").astype(int)
    df["lastYear"] = df["contractYearSigned"] - 1
    history = cumulative_history(df)
    df = pd.merge(df, history, left_on=["lastYear", "buyerName"], right_on=["year", "buyerName"], how="left")

    penalties = penalties.drop(columns=["NOMBRE_ENTIDAD"], errors="ignore")
    df = pd.merge(
        df, penalties, left_on=["buyerId", "contractYearSigned"], right_on=["NIT_ENTIDAD", "AÑO"], how="left"
    )
    df = df.rename(
        columns={"MULTAS_ACUMULADAS": "penaltiesAccumulated", "MULTAS_RECIENTES": "penaltiesRecent"}
    )
    return df.drop(columns=["NIT_ENTIDAD", "AÑO"])
