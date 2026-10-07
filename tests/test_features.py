import pandas as pd
import pytest

from secop_dev.features.build import from_v1_format, to_v1_format
from secop_dev.features.owner import cumulative_history, herfindahl_index, previous_year_hhi
from secop_dev.features.project import apply_sample_filters, assign_work_category


def _contracts(**columns):
    n = len(next(iter(columns.values())))
    base = {
        "uid": [f"u{i}" for i in range(n)],
        "contractValueMw": [100.0] * n,
        "tenderValueMw": [100.0] * n,
        "costDeviationPerc": [0.1] * n,
        "contractDurationDays": [100.0] * n,
    }
    base.update(columns)
    return pd.DataFrame(base)


def test_sample_filter_bounds(data_config):
    cfg = data_config["features"]
    df = _contracts(
        tenderValueMw=[100.0, 200.0, 100.0, 100.0, 100.0, 100.0],
        costDeviationPerc=[0.0, 0.1, 0.5, 0.51, 0.1, 0.1],
        contractDurationDays=[30.0, 100.0, 100.0, 100.0, 29.0, 1800.0],
    )
    kept = apply_sample_filters(df, cfg)["uid"].tolist()
    # bid ratio 0.5 excluded, cost deviation 0.5 kept, 30 days kept, 1800 days excluded
    assert kept == ["u0", "u2"]


def test_work_category_priority(data_config):
    cfg = data_config["features"]
    descriptions = pd.Series(
        [
            "CONSTRUCCION DE PLACA HUELLA",
            "CONSTRUCCION Y MEJORAMIENTO DE VIA",
            "MANTENIMIENTO Y REHABILITACION",
            "SUMINISTRO DE MATERIALES",
        ]
    )
    result = assign_work_category(descriptions, cfg["work_categories"], cfg["work_category_default"])
    assert result.tolist() == ["Construction", "Improvement", "Rehabilitation", "Others"]


@pytest.fixture
def owner_contracts():
    return pd.DataFrame(
        {
            "uid": ["a", "b", "c", "d", "e"],
            "buyerName": ["E1", "E1", "E1", "E2", "E1"],
            "supplierName": ["S1", "S2", "S1", "S1", "S1"],
            "contractYearSigned": [2015, 2015, 2016, 2016, 2017],
            "contractValue": [10.0, 30.0, 10.0, 5.0, 8.0],
            "haveTimeDeviation": [1, 0, 1, 0, 0],
            "haveCostDeviation": [0, 0, 1, 1, 0],
            "haveTimeAndCostDeviation": [0, 0, 1, 0, 0],
            "closedMethod": [1, 0, 0, 0, 1],
            "tenderMethod": [0, 1, 1, 1, 0],
            "projectIntensityMw": [1.0, 2.0, 3.0, 4.0, 5.0],
            "bidContractRatio": [1.0] * 5,
            "awardGrowth": [0.0] * 5,
            "timeDeviationPerc": [0.2, 0.0, 0.4, 0.0, 0.0],
            "costDeviationPerc": [0.0, 0.0, 0.3, 0.1, 0.0],
            "contractValueMw": [1.0] * 5,
            "contractDurationDays": [100.0] * 5,
        }
    )


def test_herfindahl_index(owner_contracts):
    hhi_num, hhi_value = herfindahl_index(owner_contracts)
    e1_2015 = hhi_num.query("buyerName == 'E1' and year == 2015")["hhi_num"].item()
    assert e1_2015 == pytest.approx(50**2 + 50**2)  # two suppliers, one contract each
    e1_2015_value = hhi_value.query("buyerName == 'E1' and year == 2015")["hhi_valor"].item()
    assert e1_2015_value == pytest.approx(25**2 + 75**2)


def test_previous_year_hhi_imputes_first_year(owner_contracts):
    result = previous_year_hhi(owner_contracts).set_index("uid")
    # 2016 contract of E1 takes the 2015 HHI; 2015 contracts have no previous year and get
    # the median of E1 over all years (5000, 10000, 10000), as in the thesis
    assert result.loc["c", "hhi_num_anio_anterior"] == pytest.approx(5000.0)
    assert result.loc["a", "hhi_num_anio_anterior"] == pytest.approx(10000.0)


def test_cumulative_history_counts_up_to_year(owner_contracts):
    history = cumulative_history(owner_contracts).set_index(["buyerName", "year"])
    assert history.loc[("E1", 2015), "totalCumContracts"] == 2
    assert history.loc[("E1", 2016), "histPctDesvTiempo"] == pytest.approx(2 / 3 * 100)
    assert history.loc[("E2", 2015), "totalCumContracts"] == 0  # no contracts yet -> zeros


def test_v1_column_format_roundtrip(features_config):
    table = pd.DataFrame(columns=["uid", "contractValueMw", "ownerLevel", "haveCostDeviation"])
    v1 = to_v1_format(table, features_config)
    assert list(v1.columns) == ["uid", "contractValueMw(G1)", "ownerLevel(G2)", "haveCostDeviation(G5)"]
    assert list(from_v1_format(v1).columns) == list(table.columns)
