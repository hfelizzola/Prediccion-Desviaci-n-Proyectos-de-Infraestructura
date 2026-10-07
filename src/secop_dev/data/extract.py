"""Extraction from the Socrata API of datos.gov.co (SECOP I, SECOP II and penalties).

Credentials are read from the environment (or a ``.env`` file):
``SOCRATA_APP_TOKEN``, ``SOCRATA_USERNAME`` and ``SOCRATA_PASSWORD``.

SECOP records change over time, so every extraction is written to a dated folder
(``<output_dir>/raw/<YYYY-MM-DD>/``) and never overwrites the thesis inputs.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from secop_dev.config import PROJECT_ROOT, Paths
from secop_dev.utils import get_logger

SQL_DIR = PROJECT_ROOT / "sql"
logger = get_logger(__name__)

# Socrata returns the aliases of queryMultasSanciones.sql in lower case; the thesis
# pipeline expects these column names.
PENALTY_COLUMNS = {
    "penaltyid": "PENALTY_ID",
    "buyername": "buyerName",
    "buyerid": "NIT_ENTIDAD",
    "penaltyvalue": "PENALTY_VALUE",
    "penaltydate": "PENALTY_DATE",
}


def read_query(name: str, **params: str) -> str:
    """Read ``sql/<name>.sql`` and fill ``{placeholders}`` with ``params``."""
    query = (SQL_DIR / f"{name}.sql").read_text(encoding="utf-8")
    return query.format(**params) if params else query


def quote_list(values: Iterable[Any]) -> str:
    """Render values as a SoQL ``IN`` list: ``'a','b','c'``."""
    return "'" + "','".join(str(v) for v in values) + "'"


def socrata_client(domain: str, timeout: int):
    """Authenticated Socrata client built from environment variables."""
    from sodapy import Socrata

    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass
    return Socrata(
        domain,
        os.environ.get("SOCRATA_APP_TOKEN"),
        username=os.environ.get("SOCRATA_USERNAME"),
        password=os.environ.get("SOCRATA_PASSWORD"),
        timeout=timeout,
    )


def run_query(client, dataset_id: str, query: str) -> pd.DataFrame:
    result = pd.DataFrame.from_dict(client.get(dataset_id, query=query))
    logger.info("Dataset %s: %d records", dataset_id, len(result))
    return result


def run_query_in_batches(
    client, dataset_id: str, query_name: str, placeholder: str, ids: Sequence[Any], batch_size: int
) -> pd.DataFrame:
    """Run a query whose ``WHERE ... IN ({placeholder})`` list is split into batches."""
    frames = []
    for start in range(0, len(ids), batch_size):
        batch = ids[start : start + batch_size]
        logger.info("Batch %d-%d of %d", start, start + len(batch), len(ids))
        query = read_query(query_name, **{placeholder: quote_list(batch)})
        frames.append(run_query(client, dataset_id, query))
    return pd.concat(frames) if frames else pd.DataFrame()


def extract_contracts(data_config: dict[str, Any], paths: Paths) -> Path:
    """Download SECOP I processes, SECOP II contracts and their SECOP II processes."""
    socrata, files = data_config["socrata"], data_config["files"]
    datasets = socrata["datasets"]
    out_dir = paths.output_dir / "raw" / date.today().isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)
    client = socrata_client(socrata["domain"], socrata["timeout"])

    secop1 = run_query(client, datasets["secop1_processes"], read_query("querySECOPIProcesos"))
    secop1.to_csv(out_dir / files["secop1_processes"], index=False)

    secop2 = run_query(client, datasets["secop2_contracts"], read_query("querySECOPIIContratos"))
    secop2.to_csv(out_dir / files["secop2_contracts"], index=False)

    # Only processes with a single contract give an unambiguous award value.
    counts = secop2["id_proceso"].value_counts()
    single_contract_ids = list(counts[counts <= 1].index)
    processes = run_query_in_batches(
        client,
        datasets["secop2_processes"],
        "querySECOPIIProcesos",
        "list_id_proceso",
        single_contract_ids,
        socrata["batch_size"],
    )
    processes.to_csv(out_dir / files["secop2_processes"], index=False)
    logger.info("Raw extracts written to %s", out_dir)
    return out_dir


def extract_penalties(data_config: dict[str, Any], buyer_ids: Sequence[Any], out_dir: Path) -> Path:
    """Download fines and sanctions of the given contracting entities (NIT)."""
    socrata = data_config["socrata"]
    client = socrata_client(socrata["domain"], socrata["timeout"])
    penalties = run_query_in_batches(
        client,
        socrata["datasets"]["penalties"],
        "queryMultasSanciones",
        "NIT_ENTIDAD_LIST",
        list(buyer_ids),
        socrata["batch_size"],
    )
    penalties = penalties.rename(columns=lambda c: PENALTY_COLUMNS.get(c.lower(), c))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / data_config["files"]["penalties_raw"]
    penalties.to_csv(path, index=False)
    return path
