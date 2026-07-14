#!/usr/bin/env python3
"""Batch preprocess datasets under data/raw/ using ibioml.preprocess_data.

Discovers every `*_bins200ms.mat` under `data/raw/<region>/`, derives the
per-dataset threshDPrime from the `expert`/`naive` tag in the filename, runs
`preprocess_data` from the lab library, and writes machine- and human-readable
reports summarizing what was filtered out.

Usage:
    python scripts/preprocess_all.py                      # process every dataset
    python scripts/preprocess_all.py --only Pir_L5        # single dataset (smoke)
    python scripts/preprocess_all.py --force              # reprocess even if outputs exist

Outputs (all under `data/processed/`):
- `<region>/<dataset>_*.pickle` — the 12 pickle variants preprocess_data emits.
- `logs/<dataset>.log` — captured stdout for that dataset's preprocess run.
- `preprocess_report.csv` — one row per dataset, machine-readable.
- `preprocess_report.md` — human-readable summary with config + per-dataset table.
- `README.md` — records the config, thesis commit, and ibioml commit used.
"""

from __future__ import annotations

import argparse
import datetime
import io
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

from ibioml.preprocess_data import preprocess_data

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_OUT = ROOT / "data" / "processed"

DEFAULT_CONFIG = {
    "bins_before": 2,
    "bins_after": 2,
    "bins_current": 1,
    "firingMinimo": 0.1,  # Hz
    "binLength": 200,     # ms
    "thVel": 1,           # cm/s
    "thDur": 4,           # s
    "thPosition": 1500,   # deg — drop out-of-corridor trials (peak position above this)
}

THRESH_DPRIME_EXPERT = 2
THRESH_DPRIME_NAIVE = -1.1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--only",
        metavar="SUBSTR",
        help="Only process datasets whose name contains this substring "
             "(e.g. --only Pir_L5 for a smoke run on one dataset).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Reprocess even if the canonical output pickle already exists.",
    )
    return p.parse_args()


def discover_datasets(only: str | None = None):
    """Yield (region, dataset_name, mat_path) for every raw *_bins200ms.mat."""
    for region_dir in sorted(DATA_RAW.iterdir()):
        if not region_dir.is_dir():
            continue
        region = region_dir.name
        for mat_path in sorted(region_dir.glob("*_bins200ms.mat")):
            name = mat_path.stem  # e.g. "CA3_S17_expert_bins200ms"
            if only and only not in name:
                continue
            yield region, name, mat_path


def parse_cohort(name: str) -> tuple[str, float]:
    """Extract cohort tag ('expert' | 'naive') and the matching threshDPrime."""
    parts = name.split("_")
    for tag in ("expert", "naive"):
        if tag in parts:
            thresh = THRESH_DPRIME_EXPERT if tag == "expert" else THRESH_DPRIME_NAIVE
            return tag, thresh
    raise ValueError(
        f"Cannot detect 'expert' or 'naive' tag in dataset name: {name!r}"
    )


def get_commit(cwd: Path) -> str:
    """Short SHA of HEAD in the given directory; 'unknown' if not a git repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, cwd=cwd,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def process_one(region: str, name: str, mat_path: Path, force: bool) -> dict | None:
    """Run preprocess_data for a single dataset. Returns a report row or None if skipped."""
    region_out = DATA_OUT / region
    region_out.mkdir(parents=True, exist_ok=True)
    logs_dir = DATA_OUT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    cohort, thresh = parse_cohort(name)

    # Idempotency check: preprocess_data writes many pickles; the plain flat one
    # is the "canonical" marker. Skip if it exists and --force wasn't passed.
    canonical = region_out / f"{name}_flat.pickle"
    if canonical.exists() and not force:
        print(f"[SKIP] {name} — output exists (use --force to reprocess)")
        return None

    print(f"[RUN ] {name}  region={region}  cohort={cohort}  threshDPrime={thresh}")

    log_path = logs_dir / f"{name}.log"
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            result = preprocess_data(
                file_path=str(mat_path),
                file_name_to_save=name,
                bins_before=DEFAULT_CONFIG["bins_before"],
                bins_after=DEFAULT_CONFIG["bins_after"],
                bins_current=DEFAULT_CONFIG["bins_current"],
                threshDPrime=thresh,
                firingMinimo=DEFAULT_CONFIG["firingMinimo"],
                binLength=DEFAULT_CONFIG["binLength"],
                thVel=DEFAULT_CONFIG["thVel"],
                thDur=DEFAULT_CONFIG["thDur"],
                thPosition=DEFAULT_CONFIG["thPosition"],
                data_dir=str(region_out),
            )
    finally:
        log_path.write_text(buf.getvalue())

    if not result or not result.get("success"):
        print(f"[FAIL] {name}: no data survived filtering — see {log_path}")
        return {
            "dataset": name, "region": region, "cohort": cohort,
            "threshDPrime": thresh, "success": False,
        }

    row = {
        "dataset": name,
        "region": region,
        "cohort": cohort,
        "threshDPrime": thresh,
        "success": True,
        "kept_bins": result["time_bins_kept"],
        "pct_removed_total": result["proportion_total_removed"],
        "pct_removed_immobility": result["proportion_immobility"],
        "pct_removed_low_perf": result["proportion_low_performance"],
        "pct_removed_nan_pos": result["proportion_nan_position"],
        "kept_neurons": result["neurons_Kept"],
        "low_firing_neurons": result["low_firing_neurons"],
        "unstable_neurons": result["unstable_neurons"],
    }
    print(f"       kept {int(row['kept_bins']):,} bins ({100 - row['pct_removed_total']:.1f}% survived), "
          f"{int(row['kept_neurons'])} neurons")
    return row


def write_csv(rows: list[dict]) -> Path:
    csv_path = DATA_OUT / "preprocess_report.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"[OUT ] {csv_path.relative_to(ROOT)}")
    return csv_path


def write_md_report(rows: list[dict], thesis_sha: str, ibioml_sha: str) -> Path:
    md_path = DATA_OUT / "preprocess_report.md"
    now = datetime.datetime.now().isoformat(timespec="seconds")
    df = pd.DataFrame(rows)
    ok = df[df["success"]] if "success" in df.columns else df

    lines = [
        f"# Preprocess report — {now}",
        "",
        f"Thesis commit: `{thesis_sha}` · ibioml commit: `{ibioml_sha}`",
        "",
        "## Config",
        "",
        *[f"- `{k}` = {v}" for k, v in DEFAULT_CONFIG.items()],
        f"- `threshDPrime` = {THRESH_DPRIME_EXPERT} (expert) / {THRESH_DPRIME_NAIVE} (naive) — derived from filename",
        "",
        "## Summary",
        "",
        f"- Datasets processed: **{len(df)}** ({len(ok)} succeeded, {len(df) - len(ok)} failed)",
    ]
    if len(ok):
        lines.append(f"- Median % time bins removed: **{ok['pct_removed_total'].median():.1f}%**")
        lines.append(f"- Median neurons kept: **{ok['kept_neurons'].median():.0f}**")
    lines.append("")
    lines.append("## Per-dataset detail")
    lines.append("")
    lines.append("| Dataset | Cohort | Bins kept | % removed (immob / lowperf / nan) | Neurons kept | Dropped (lowfire / unstable) |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for _, r in df.iterrows():
        if not r.get("success"):
            lines.append(f"| {r['dataset']} | {r['cohort']} | (failed) | — | — | — |")
            continue
        removed = (
            f"{r['pct_removed_immobility']:.1f} / "
            f"{r['pct_removed_low_perf']:.1f} / "
            f"{r['pct_removed_nan_pos']:.1f}"
        )
        dropped = f"{int(r['low_firing_neurons'])} / {int(r['unstable_neurons'])}"
        lines.append(
            f"| {r['dataset']} | {r['cohort']} | {int(r['kept_bins']):,} | "
            f"{removed} | {int(r['kept_neurons'])} | {dropped} |"
        )

    md_path.write_text("\n".join(lines) + "\n")
    print(f"[OUT ] {md_path.relative_to(ROOT)}")
    return md_path


def write_readme(thesis_sha: str, ibioml_sha: str) -> Path:
    readme = DATA_OUT / "README.md"
    now = datetime.datetime.now().isoformat(timespec="seconds")
    lines = [
        "# data/processed/",
        "",
        f"Autogenerated by `scripts/preprocess_all.py` on {now}.",
        "",
        f"Thesis commit: `{thesis_sha}`  ",
        f"ibioml commit: `{ibioml_sha}`",
        "",
        "## Config used",
        "",
        *[f"- `{k}` = {v}" for k, v in DEFAULT_CONFIG.items()],
        f"- `threshDPrime` = {THRESH_DPRIME_EXPERT} (expert) / {THRESH_DPRIME_NAIVE} (naive)",
        "",
        "## Files",
        "",
        "- `<region>/<dataset>_*.pickle` — 12 variants per dataset produced by `preprocess_data`.",
        "- `logs/<dataset>.log` — captured stdout for that dataset's preprocess run.",
        "- `preprocess_report.csv` — machine-readable summary (one row per dataset).",
        "- `preprocess_report.md` — human-readable summary with per-dataset table.",
        "",
        "Nothing here is tracked in git (see `specs/decisions.md` D-014). Regenerate with",
        "`python scripts/preprocess_all.py`.",
    ]
    readme.write_text("\n".join(lines) + "\n")
    print(f"[OUT ] {readme.relative_to(ROOT)}")
    return readme


def main() -> int:
    args = parse_args()

    datasets = list(discover_datasets(only=args.only))
    if not datasets:
        target = f" matching {args.only!r}" if args.only else ""
        print(f"No *_bins200ms.mat found under {DATA_RAW}{target}.")
        return 1

    print(f"Processing {len(datasets)} dataset(s)")
    print(f"Config: {DEFAULT_CONFIG}")
    print(f"Output: {DATA_OUT.relative_to(ROOT)}/")
    print("-" * 60)

    rows: list[dict] = []
    for region, name, mat_path in datasets:
        row = process_one(region, name, mat_path, force=args.force)
        if row is not None:
            rows.append(row)

    if not rows:
        print("\nNothing new was processed. Existing outputs left untouched.")
        return 0

    print("-" * 60)
    thesis_sha = get_commit(ROOT)
    ibioml_sha = get_commit(ROOT / "external" / "ibioml")
    write_csv(rows)
    write_md_report(rows, thesis_sha, ibioml_sha)
    write_readme(thesis_sha, ibioml_sha)

    return 0


if __name__ == "__main__":
    sys.exit(main())
