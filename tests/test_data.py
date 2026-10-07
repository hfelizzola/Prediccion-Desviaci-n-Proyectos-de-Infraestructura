import numpy as np
import pandas as pd

from secop_dev.data.clean import apply_corrections, assign_region, classify_contract_method, duration_in_days
from secop_dev.data.penalties import penalties_by_buyer_year
from secop_dev.data.terridata import complete_years


def test_duration_in_days_converts_units_and_flags_unknown():
    days = duration_in_days(
        pd.Series([10.0, 2.0, 3.0, 5.0]),
        pd.Series(["DIAS", "MESES", "SEMANAS", "AÑOS"]),
        {"DIAS": 1, "MESES": 30, "SEMANAS": 7},
    )
    assert days.iloc[:3].tolist() == [10.0, 60.0, 21.0]
    assert np.isnan(days.iloc[3])


def test_contract_method_later_rule_wins(data_config):
    cfg = data_config["cleaning"]
    methods = pd.Series(
        [
            "LICITACION PUBLICA",
            "CONTRATACION DIRECTA",
            "SELECCION ABREVIADA DE MENOR CUANTIA",
            "CONTRATACION MINIMA CUANTIA",
            "CONCURSO DE MERITOS",
            "REGIMEN ESPECIAL - LICITACION",
        ]
    )
    result = classify_contract_method(methods, cfg)
    # The last value matches OPEN and SPECIAL REGIME; the later rule wins (as in v1).
    assert list(result) == ["OPEN", "CLOSED", "SIMPLIFIED", "LIMITED", "OTHER", "SPECIAL REGIME"]
    assert result.ordered


def test_assign_region_uses_default(data_config):
    cfg = data_config["cleaning"]
    regions = assign_region(pd.Series(["META", "BOGOTA D.C.", "NO DEFINIDO", None]), cfg["regions"], "OTHER")
    assert regions.tolist() == ["ORINOQUIA", "ANDINA", "OTHER", "OTHER"]


def test_apply_corrections_overwrites_only_listed_contracts():
    df = pd.DataFrame({"uid": ["a", "b"], "contractValue": [1.0, 2.0]})
    corrections = pd.DataFrame({"uid": ["b"], "contractValue": [20.0]})
    assert apply_corrections(df, corrections)["contractValue"].tolist() == [1.0, 20.0]


def _penalties_reference(penalties, buyers, start, end):
    """Loop implementation of the thesis (analizar_multas), used as the oracle."""
    penalties = penalties.assign(YEAR=pd.to_datetime(penalties["PENALTY_DATE"]).dt.year)
    rows = []
    for nit in pd.unique(pd.Series(buyers)):
        for year in range(start, end + 1):
            own = penalties[penalties["NIT_ENTIDAD"] == nit]
            rows.append(
                {
                    "NIT_ENTIDAD": nit,
                    "AÑO": year,
                    "MULTAS_ACUMULADAS": int((own["YEAR"] < year).sum()),
                    "MULTAS_RECIENTES": int(((own["YEAR"] >= year - 2) & (own["YEAR"] < year)).any()),
                }
            )
    return pd.DataFrame(rows)


def test_penalties_match_thesis_loop():
    rng = np.random.default_rng(0)
    penalties = pd.DataFrame(
        {
            "NIT_ENTIDAD": rng.choice([1, 2, 3, 4], size=60),
            "PENALTY_DATE": pd.to_datetime("2010-01-01")
            + pd.to_timedelta(rng.integers(0, 5000, 60), unit="D"),
        }
    )
    buyers = [3, 1, 5, 3, 2]
    result = penalties_by_buyer_year(penalties, buyers, 2014, 2024, recent_window=2)
    expected = _penalties_reference(penalties, buyers, 2014, 2024)
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_complete_years_imputes_with_group_mean():
    data = pd.DataFrame(
        {
            "Departamento": ["A", "A"],
            "Entidad": ["X", "X"],
            "Año": [2016, 2018],
            "Indicador": ["PIB", "PIB"],
            "Dato Numérico": [10.0, 20.0],
        }
    )
    result = complete_years(data, [2024]).set_index("Año")["PIB"]
    assert result.loc[2016] == 10.0 and result.loc[2018] == 20.0
    assert result.loc[2024] == 15.0
