from __future__ import annotations

import pytest

from secop_dev.config import Paths, load_yaml


@pytest.fixture(scope="session")
def data_config():
    return load_yaml("data")


@pytest.fixture(scope="session")
def features_config():
    return load_yaml("features")


@pytest.fixture(scope="session")
def experiments_config():
    return load_yaml("experiments")


@pytest.fixture(scope="session")
def paths(data_config):
    return Paths.from_config(data_config)
