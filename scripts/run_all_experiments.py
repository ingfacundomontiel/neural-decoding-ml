#!/usr/bin/env python3
"""Batch runner: iterate over every preprocessed bins200ms dataset, run one experiment
per (dataset × target) combination.

For each dataset under `data/processed/<region>/`, launches `experiments/run_experiment.py`
three times (one per target: pos, vel, dual). Experiments run as isolated subprocesses so
one crash does not take down the batch; each experiment's stdout/stderr goes to
`results/logs/<study-name>.log`.

Usage:
    # Full batch (all datasets × 3 targets), full-config runs on the GPU host:
    uv run python scripts/run_all_experiments.py

    # Smoke: reduced params via --smoke, useful for validating the whole batch flow:
    uv run python scripts/run_all_experiments.py --smoke

    # Subset by region and/or dataset name:
    uv run python scripts/run_all_experiments.py --regions piriform --only Pir_L5

    # Only one target:
    uv run python scripts/run_all_experiments.py --targets dual

    # Dry-run: print planned experiments without launching:
    uv run python scripts/run_all_experiments.py --dry-run

Idempotent: an experiment whose `results/<study-name>/final_results.json` already exists
is skipped (pass `--force` to re-run). Missing pickles are reported and counted as
failures, so if a dataset was not preprocessed the batch tells you which one.
"""

from __future__ import annotations

import argparse
import datetime
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
LOGS = RESULTS / "logs"
RUNNER = ROOT / "experiments" / "run_experiment.py"

# Target → suffix inside the pickle filename produced by preprocess_all.py.
TARGETS = ("pos", "vel", "dual")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--targets", default=",".join(TARGETS),
        help=f"Comma-separated targets to train. Any of {TARGETS}. Default: all three.",
    )
    p.add_argument(
        "--regions", default="",
        help="Comma-separated region filter (e.g. `piriform,ca3`). Default: every region.",
    )
    p.add_argument(
        "--only", default="",
        help="Filter datasets whose name contains this substring.",
    )
    p.add_argument(
        "--with-ctxt", action=argparse.BooleanOptionalAction, default=True,
        help="Use *_withCtxt_*.pickle variants (default: on). --no-with-ctxt to disable.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-run even if the experiment's final_results.json already exists.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print the planned experiments without launching any subprocess.",
    )
    # Passthrough flags to experiments/run_experiment.py
    p.add_argument("--smoke", action="store_true", help="Pass --smoke to the runner.")
    p.add_argument("--num-trials", type=int)
    p.add_argument("--num-epochs", type=int)
    p.add_argument("--outer-folds", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--es-patience", type=int)
    p.add_argument("--device", choices=["cpu", "cuda", "auto"])
    return p.parse_args()


def parse_target_list(csv: str) -> list[str]:
    tgts = [t.strip() for t in csv.split(",") if t.strip()]
    unknown = [t for t in tgts if t not in TARGETS]
    if unknown:
        raise SystemExit(f"Unknown target(s): {unknown}. Must be a subset of {list(TARGETS)}.")
    return tgts


def discover_datasets(regions_filter: set[str] | None, name_filter: str, with_ctxt: bool):
    """Yield (region, dataset_stem) for every preprocessed bins200ms dataset.

    Detects datasets by their `_flat.pickle` canonical output (the smallest common
    variant produced by preprocess_all.py). Whether we later load the with/without
    context variant is a separate axis controlled by --with-ctxt.
    """
    if not DATA_PROC.exists():
        return
    marker = "_withCtxt_flat.pickle" if with_ctxt else "_flat.pickle"
    marker_excl = "_flat.pickle" if with_ctxt else None  # avoid double-matching the ctxt one
    for region_dir in sorted(DATA_PROC.iterdir()):
        if not region_dir.is_dir() or region_dir.name == "logs":
            continue
        if regions_filter and region_dir.name not in regions_filter:
            continue
        for pkl in sorted(region_dir.glob("*_bins200ms" + marker)):
            # When with_ctxt is False, `marker` == "_flat.pickle" also matches
            # "..._withCtxt_flat.pickle"; filter those out.
            if not with_ctxt and "_withCtxt_" in pkl.name:
                continue
            stem = pkl.name[: -len(marker)]  # e.g. "CA3_S17_expert_bins200ms"
            if name_filter and name_filter not in stem:
                continue
            yield region_dir.name, stem


def pickle_for(region: str, stem: str, target: str, with_ctxt: bool) -> Path:
    """Resolve the pickle path for one (dataset, target) combination."""
    ctxt = "withCtxt_" if with_ctxt else ""
    if target == "pos":
        name = f"{stem}_{ctxt}onlyPosition_flat.pickle"
    elif target == "vel":
        name = f"{stem}_{ctxt}onlyVelocity_flat.pickle"
    elif target == "dual":
        name = f"{stem}_{ctxt}flat.pickle"
    else:
        raise ValueError(f"Unknown target: {target}")
    return DATA_PROC / region / name


def study_name(stem: str, target: str) -> str:
    """Convention: `mlp_<dataset-stem>_<target>` — human-readable + sortable."""
    return f"mlp_{stem}_{target}"


def already_done(study: str) -> bool:
    return (RESULTS / study / "final_results.json").exists()


def build_runner_cmd(pkl: Path, study: str, args: argparse.Namespace) -> list[str]:
    cmd = [
        "uv", "run", "python", str(RUNNER),
        "--pickle", str(pkl),
        "--study-name", study,
    ]
    if args.smoke:
        cmd.append("--smoke")
    for flag, value in [
        ("--num-trials", args.num_trials),
        ("--num-epochs", args.num_epochs),
        ("--outer-folds", args.outer_folds),
        ("--batch-size", args.batch_size),
        ("--es-patience", args.es_patience),
        ("--device", args.device),
    ]:
        if value is not None:
            cmd.extend([flag, str(value)])
    return cmd


def run_one(pkl: Path, study: str, args: argparse.Namespace) -> tuple[int, float, Path]:
    """Run one experiment as a subprocess. Log stdout+stderr to results/logs/<study>.log."""
    log_path = LOGS / f"{study}.log"
    cmd = build_runner_cmd(pkl, study, args)
    t0 = datetime.datetime.now()
    with log_path.open("w") as f:
        f.write(f"# Command: {' '.join(cmd)}\n")
        f.write(f"# Started: {t0.isoformat()}\n")
        f.write("#" + "-" * 60 + "\n\n")
        f.flush()
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    dt = (datetime.datetime.now() - t0).total_seconds()
    return proc.returncode, dt, log_path


def main() -> int:
    args = parse_args()
    targets = parse_target_list(args.targets)
    regions_filter = {r for r in args.regions.split(",") if r} if args.regions else None

    LOGS.mkdir(parents=True, exist_ok=True)

    datasets = list(discover_datasets(regions_filter, args.only, args.with_ctxt))
    if not datasets:
        print(f"No datasets found under {DATA_PROC.relative_to(ROOT)}/. "
              "Have you run scripts/preprocess_all.py?")
        return 1

    experiments = [(r, s, t) for (r, s) in datasets for t in targets]

    print("=" * 70)
    print(f"Planned: {len(experiments)} experiments "
          f"({len(datasets)} datasets × {len(targets)} targets)")
    print(f"Targets:  {targets}")
    print(f"With ctxt: {args.with_ctxt}")
    if args.smoke:
        print("Smoke mode: ON (reduced params)")
    print("=" * 70)

    if args.dry_run:
        for i, (region, stem, target) in enumerate(experiments, 1):
            pkl = pickle_for(region, stem, target, args.with_ctxt)
            exists = "✓" if pkl.exists() else "✗ MISSING"
            done = "  DONE" if already_done(study_name(stem, target)) else ""
            print(f"  [{i:>3}] {study_name(stem, target):<55}  {exists}{done}")
        return 0

    passed = failed = skipped = 0
    total_secs = 0.0
    batch_t0 = datetime.datetime.now()

    for i, (region, stem, target) in enumerate(experiments, 1):
        study = study_name(stem, target)
        pkl = pickle_for(region, stem, target, args.with_ctxt)
        prefix = f"[{i:>3}/{len(experiments)}]"

        if not pkl.exists():
            print(f"{prefix} MISS  {study}  (pickle not found: "
                  f"{pkl.relative_to(ROOT)})")
            failed += 1
            continue

        if already_done(study) and not args.force:
            print(f"{prefix} SKIP  {study}  (final_results.json exists — use --force)")
            skipped += 1
            continue

        print(f"{prefix} RUN   {study} ...", flush=True, end="")
        rc, secs, log = run_one(pkl, study, args)
        total_secs += secs
        status = "OK  " if rc == 0 else f"FAIL rc={rc}"
        print(f" {status}  ({secs:>6.0f}s)  log={log.relative_to(ROOT)}")
        if rc == 0:
            passed += 1
        else:
            failed += 1

    wall = (datetime.datetime.now() - batch_t0).total_seconds()
    print("=" * 70)
    print(f"Passed: {passed}   Failed: {failed}   Skipped: {skipped}")
    print(f"Compute time: {total_secs / 60:.1f} min · Wall time: {wall / 60:.1f} min")
    print(f"Logs: {LOGS.relative_to(ROOT)}/")
    print(f"Results: {RESULTS.relative_to(ROOT)}/")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
