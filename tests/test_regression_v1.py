"""The refactored pipeline must reproduce the thesis files (v1.0.0) exactly.

Needs the data folders (``Data/`` and ``terridata/``, or ``SECOP_DATA_DIR`` /
``SECOP_TERRIDATA_DIR``); skipped otherwise. Run with ``pytest -m regression``.
"""

from __future__ import annotations

import io

import pandas as pd
import pytest

from secop_dev.data.clean import clean_contracts
from secop_dev.data.integrate import integrate_secop
from secop_dev.data.penalties import penalties_by_buyer_year
from secop_dev.data.terridata import load_terridata
from secop_dev.features.build import build_training_table, to_v1_format

pytestmark = pytest.mark.regression


def _roundtrip(df: pd.DataFrame) -> pd.DataFrame:
    """Write and read back as CSV, as every pipeline step does."""
    return pd.read_csv(io.StringIO(df.to_csv(index=False)), low_memory=False)


def _read(paths, data_config, key: str) -> pd.DataFrame:
    path = paths.data_dir / data_config["files"][key]
    if not path.exists():
        pytest.skip(f"{path} not available")
    return pd.read_csv(path, low_memory=False)


def test_integration_reproduces_v1(paths, data_config):
    result = integrate_secop(
        _read(paths, data_config, "secop1_processes"),
        _read(paths, data_config, "secop2_contracts"),
        _read(paths, data_config, "secop2_processes"),
        data_config["integration"],
    )
    pd.testing.assert_frame_equal(_roundtrip(result), _read(paths, data_config, "contracts_integrated"))


def test_cleaning_reproduces_v1(paths, data_config):
    corrections = pd.read_excel(paths.data_dir / data_config["files"]["corrections"], sheet_name=None)
    result = clean_contracts(
        _read(paths, data_config, "contracts_integrated"), corrections, data_config["cleaning"]
    )
    pd.testing.assert_frame_equal(_roundtrip(result), _read(paths, data_config, "contracts_clean"))


def test_penalties_reproduce_v1(paths, data_config):
    cfg = data_config["penalties"]
    buyers = _read(paths, data_config, "contracts_clean")["buyerId"]
    result = penalties_by_buyer_year(
        _read(paths, data_config, "penalties_raw"),
        buyers,
        cfg["start_year"],
        cfg["end_year"],
        cfg["recent_window_years"],
    )
    pd.testing.assert_frame_equal(_roundtrip(result), _read(paths, data_config, "penalties_by_buyer"))


def test_training_table_reproduces_v1(paths, data_config, features_config):
    files = [paths.terridata_dir / f for f in data_config["files"]["terridata"]]
    if not all(f.exists() for f in files):
        pytest.skip("TerriData files not available")
    table = build_training_table(
        _read(paths, data_config, "contracts_clean"),
        _read(paths, data_config, "penalties_by_buyer"),
        load_terridata(files),
        data_config,
        features_config,
    )
    result = _roundtrip(to_v1_format(table, features_config))
    pd.testing.assert_frame_equal(result, _read(paths, data_config, "training_table"))
