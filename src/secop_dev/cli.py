"""Command-line interface: ``secop-dev <command>`` (or ``python -m secop_dev``).

Raw inputs are read from ``paths.data_dir`` / ``paths.terridata_dir``; everything that
is generated goes to ``paths.output_dir`` (see ``configs/data.yaml``):

    integrate   SECOP I + II extracts            -> interim/contratosSECOP.csv
    clean       unified contracts                -> interim/contratosSECOPCleaned.csv
    penalties   fines per entity and year        -> interim/analisis_multas_por_entidad.csv
    features    training table                   -> processed/trainData.csv
    data        integrate + clean + penalties + features
    train       Optuna tuning + hold-out metrics -> runs/<run-id>/
    doe         CV of every model x scenario + two-way ANOVA
    shap        SHAP values of the full-sample models
    compare-v1  metrics of a run vs. the thesis tables
    extract     download fresh extracts from datos.gov.co -> raw/<date>/
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from secop_dev.config import Paths, load_yaml
from secop_dev.utils import get_logger

logger = get_logger("secop_dev.cli")


def _context():
    data_config = load_yaml("data")
    paths = Paths.from_config(data_config)
    paths.ensure()
    return data_config, paths


def cmd_integrate(args) -> None:
    from secop_dev.data.integrate import integrate_secop

    cfg, paths = _context()
    files, source = cfg["files"], Path(args.raw_dir) if args.raw_dir else paths.data_dir
    table = integrate_secop(
        pd.read_csv(source / files["secop1_processes"]),
        pd.read_csv(source / files["secop2_contracts"]),
        pd.read_csv(source / files["secop2_processes"]),
        cfg["integration"],
    )
    out = paths.interim_dir / files["contracts_integrated"]
    table.to_csv(out, index=False)
    logger.info("Wrote %s (%d rows)", out, len(table))


def cmd_clean(args) -> None:
    from secop_dev.data.clean import clean_contracts

    cfg, paths = _context()
    files = cfg["files"]
    contracts = pd.read_csv(paths.interim_dir / files["contracts_integrated"], low_memory=False)
    corrections = pd.read_excel(paths.data_dir / files["corrections"], sheet_name=None)
    table = clean_contracts(contracts, corrections, cfg["cleaning"])
    out = paths.interim_dir / files["contracts_clean"]
    table.to_csv(out, index=False)
    logger.info("Wrote %s (%d rows)", out, len(table))


def cmd_penalties(args) -> None:
    from secop_dev.data.penalties import penalties_by_buyer_year

    cfg, paths = _context()
    files, pcfg = cfg["files"], cfg["penalties"]
    contracts = pd.read_csv(paths.interim_dir / files["contracts_clean"], usecols=["buyerName", "buyerId"])
    penalties = pd.read_csv(paths.data_dir / files["penalties_raw"])
    table = penalties_by_buyer_year(
        penalties, contracts["buyerId"], pcfg["start_year"], pcfg["end_year"], pcfg["recent_window_years"]
    )
    out = paths.interim_dir / files["penalties_by_buyer"]
    table.to_csv(out, index=False)
    logger.info("Wrote %s (%d rows)", out, len(table))


def cmd_features(args) -> None:
    from secop_dev.data.terridata import load_terridata
    from secop_dev.features.build import build_training_table, to_v1_format

    cfg, paths = _context()
    features_config = load_yaml("features")
    files = cfg["files"]
    contracts = pd.read_csv(paths.interim_dir / files["contracts_clean"], low_memory=False)
    penalties = pd.read_csv(paths.interim_dir / files["penalties_by_buyer"])
    terridata = load_terridata(
        [paths.terridata_dir / f for f in files["terridata"]], cfg["features"]["terridata"]["separator"]
    )
    table = build_training_table(contracts, penalties, terridata, cfg, features_config)
    out = paths.processed_dir / files["training_table"]
    # Same format as the thesis file so both can be compared directly.
    to_v1_format(table, features_config).to_csv(out, index=False)
    logger.info("Wrote %s (%d rows)", out, len(table))


def cmd_data(args) -> None:
    for step in (cmd_integrate, cmd_clean, cmd_penalties, cmd_features):
        step(args)


def _training_table(args, cfg, paths) -> Path:
    if args.train_data:
        return Path(args.train_data)
    return paths.processed_dir / cfg["files"]["training_table"]


def _run_dir(args, paths) -> Path:
    return paths.runs_dir / args.run_id


def cmd_train(args) -> None:
    from secop_dev.models.train import load_training_table, run_experiments

    cfg, paths = _context()
    table = load_training_table(_training_table(args, cfg, paths))
    metrics = run_experiments(
        table,
        load_yaml("features"),
        load_yaml("experiments"),
        _run_dir(args, paths),
        targets=args.target,
        models=args.model,
        scenarios=args.scenario,
        n_trials=args.n_trials,
        force=args.force,
    )
    logger.info("Metrics:\n%s", metrics.to_string(index=False))


def cmd_doe(args) -> None:
    from secop_dev.analysis.doe import cross_validate_scenarios, two_way_anova
    from secop_dev.models.train import load_training_table

    cfg, paths = _context()
    experiments = load_yaml("experiments")
    table = load_training_table(_training_table(args, cfg, paths))
    run_dir = _run_dir(args, paths)
    for target in args.target or experiments["targets"]:
        scores = cross_validate_scenarios(table, run_dir, target, load_yaml("features"), experiments)
        scores.to_csv(run_dir / f"doe_cv_{target}.csv", index=False)
        anova = two_way_anova(scores, experiments["doe"]["scoring"])
        anova.to_csv(run_dir / f"doe_anova_{target}.csv", index=False)
        logger.info("ANOVA %s:\n%s", target, anova.to_string(index=False))


def cmd_shap(args) -> None:
    from secop_dev.analysis.shap_values import compute_shap
    from secop_dev.models.train import load_training_table

    cfg, paths = _context()
    experiments = load_yaml("experiments")
    table = load_training_table(_training_table(args, cfg, paths))
    run_dir = _run_dir(args, paths)
    for target in args.target or experiments["targets"]:
        compute_shap(table, run_dir, target, load_yaml("features"), experiments, run_dir / "shap")


def cmd_compare_v1(args) -> None:
    from secop_dev.analysis.compare_v1 import compare_with_v1
    from secop_dev.models.train import collect_metrics

    _, paths = _context()
    run_dir = _run_dir(args, paths)
    comparison = compare_with_v1(collect_metrics(run_dir), paths.data_dir)
    comparison.to_csv(run_dir / "comparison_v1.csv", index=False)
    cols = ["target", "model", "stage", "roc_auc", "roc_auc_v1", "roc_auc_diff", "f1_diff", "accuracy_diff"]
    with pd.option_context("display.float_format", "{:.2f}".format, "display.width", 200):
        print(comparison[cols].to_string(index=False))
    print(f"\nMax |roc_auc_diff| = {comparison['roc_auc_diff'].abs().max():.2f} points")


def cmd_extract(args) -> None:
    from secop_dev.data.extract import extract_contracts

    cfg, paths = _context()
    extract_contracts(cfg, paths)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="secop-dev", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    integrate = sub.add_parser("integrate", help="SECOP I + II -> unified contracts")
    integrate.add_argument("--raw-dir", help="folder with the API extracts (default: data_dir)")
    integrate.set_defaults(func=cmd_integrate)
    sub.add_parser("clean", help="clean the unified contracts").set_defaults(func=cmd_clean)
    sub.add_parser("penalties", help="penalties per entity and year").set_defaults(func=cmd_penalties)
    sub.add_parser("features", help="build the training table").set_defaults(func=cmd_features)
    data = sub.add_parser("data", help="integrate + clean + penalties + features")
    data.add_argument("--raw-dir")
    data.set_defaults(func=cmd_data)
    sub.add_parser("extract", help="download new extracts from datos.gov.co").set_defaults(func=cmd_extract)

    default_run = datetime.now().strftime("%Y%m%d")
    for name, func, help_text in (
        ("train", cmd_train, "tune and evaluate models"),
        ("doe", cmd_doe, "CV of every model x scenario + ANOVA"),
        ("shap", cmd_shap, "SHAP values of the full-sample models"),
        ("compare-v1", cmd_compare_v1, "compare a run with the thesis metrics"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--run-id", default=default_run, help="run folder name (default: today)")
        p.set_defaults(func=func)
        if name == "compare-v1":
            continue
        p.add_argument("--train-data", help="training table (default: processed/trainData.csv)")
        p.add_argument("--target", action="append", choices=["cost", "time"])
        if name == "train":
            p.add_argument("--model", action="append", choices=["lr", "rf", "xgb", "knn"])
            p.add_argument("--scenario", action="append", choices=["S1", "S2", "S3", "S4", "S5"])
            p.add_argument("--n-trials", type=int, help="override tuning.n_trials (e.g. 2 for a smoke test)")
            p.add_argument("--force", action="store_true", help="retrain combinations already done")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
