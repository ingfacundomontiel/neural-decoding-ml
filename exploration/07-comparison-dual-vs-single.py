# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Dual vs single-output — side-by-side comparison
#
# Cross-batch comparison between the two "same-preprocess" baselines:
#
# - **Dual model**: one MLP with `output_size=2`, predicts (position, velocity)
#   jointly. Studies at `results/mlp_<dataset>_bins200ms_dual/`.
# - **Single models**: two separate MLPs with `output_size=1`, one per target.
#   Studies at `results/mlp_<dataset>_bins200ms_pos/` and `_vel/`.
#
# For each variable (position, velocity), we compare the R² distribution across
# 5 outer folds × 17 datasets between the dual and single-output regimes.
#
# Emits under `results/plots/comparison/`:
#
# - `position_dual_vs_single_boxplot.svg` — two boxes per dataset (dual + single), position.
# - `velocity_dual_vs_single_boxplot.svg` — same for velocity.
# - `position_scatter_dual_vs_single.svg` — X = median R² single, Y = median R² dual, one point per dataset.
# - `velocity_scatter_dual_vs_single.svg` — same for velocity.
# - `comparison_table.csv` — per-dataset numeric summary (medians + deltas).
# - `comparison_table.md` — human-readable version of the same table.
#
# **No exclusions** — all 17 datasets are shown, including the ones (S38 / S41 /
# N59) usually filtered from the dual-only plots in `03-decoding-results.py`.

# %%
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

# %%
results_base_dir = 'results'
plot_path = 'results/plots/comparison'
Path(plot_path).mkdir(parents=True, exist_ok=True)

REGION_DISPLAY = {"Pir": "Pir", "CA3": "CA3", "DG": "DG"}
COHORT_DISPLAY = {"expert": "Expert", "naive": "Naive"}
COHORT_TYPE_SUFFIX = {"expert": "_experto", "naive": "_naive"}

COLOR_SCHEME = {
    'pir_naive':   "#58AAE9",
    'pir_experto': "#004E89",
    'CA3_naive':   "#69DA85",
    'CA3_experto': "#146D02",
    'DG_naive':    "#EBA447",
    'DG_experto':  "#9B4615",
}
DATASET_TYPE_ORDER = ['pir_naive', 'pir_experto', 'CA3_naive', 'CA3_experto', 'DG_naive', 'DG_experto']
TYPE_DISPLAY = {
    'pir_naive':   'Piriform-Naive',
    'pir_experto': 'Piriform-Expert',
    'CA3_naive':   'CA3-Naive',
    'CA3_experto': 'CA3-Expert',
    'DG_naive':    'DG-Naive',
    'DG_experto':  'DG-Expert',
}

# Model-type styling for the dual-vs-single grouping.
MODEL_COLOR = {'dual': "#7B848F", 'single': "#F1C40F"}  # gris para dual, amarillo para single

# All 17 datasets, ordered by type then animal (no exclusions).
FIXED_ORDER = [
    'Pir-Naive-N53', 'Pir-Naive-L7',
    'Pir-Expert-N31', 'Pir-Expert-L3', 'Pir-Expert-L5', 'Pir-Expert-N59',
    'CA3-Naive-S30', 'CA3-Naive-S31',
    'CA3-Expert-S17', 'CA3-Expert-S18',
    'DG-Naive-S24', 'DG-Naive-S38', 'DG-Naive-S39', 'DG-Naive-S41',
    'DG-Expert-S17', 'DG-Expert-S18', 'DG-Expert-S19',
]


def parse_study_dir_name(name, expected_suffix):
    prefix = "mlp_"
    suffix = f"_{expected_suffix}"
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None, None
    core = name[len(prefix):-len(suffix)]
    if not core.endswith("_bins200ms"):
        return None, None
    core = core[: -len("_bins200ms")]
    parts = core.split("_")
    if len(parts) < 3:
        return None, None
    region, cohort, dataset_id = parts[0], parts[-1], "_".join(parts[1:-1])
    if region not in REGION_DISPLAY or cohort not in COHORT_DISPLAY:
        return None, None
    display = f"{REGION_DISPLAY[region]}-{COHORT_DISPLAY[cohort]}-{dataset_id}"
    dtype = f"{region}{COHORT_TYPE_SUFFIX[cohort]}"
    if dtype.startswith("Pir"):
        dtype = "pir" + dtype[len("Pir"):]
    return display, dtype


def collect_dual(target):
    """For each `mlp_*_dual/`, extract the per-fold R² of the requested target.

    target ∈ {'pos', 'vel'}. Reads final_results.json.
    """
    key = 'test_r2_scores_pos' if target == 'pos' else 'test_r2_scores_vel'
    rows = []
    for entry in sorted(os.listdir(results_base_dir)):
        study_path = Path(results_base_dir) / entry
        if not study_path.is_dir():
            continue
        display, dtype = parse_study_dir_name(entry, "dual")
        if display is None:
            continue
        fr_path = study_path / "final_results.json"
        if not fr_path.exists():
            continue
        with fr_path.open() as f:
            d = json.load(f)
        r2s = d.get(key, [])
        for fold_idx, r2 in enumerate(r2s, start=1):
            rows.append({"Dataset": display, "Type": dtype, "Fold": fold_idx,
                         "Model": "dual", "R2_Score": r2})
    return pd.DataFrame(rows)


def collect_single(target):
    """For each `mlp_*_<target>/`, extract the per-fold R² from test_r2_scores."""
    suffix = target  # 'pos' or 'vel'
    rows = []
    for entry in sorted(os.listdir(results_base_dir)):
        study_path = Path(results_base_dir) / entry
        if not study_path.is_dir():
            continue
        display, dtype = parse_study_dir_name(entry, suffix)
        if display is None:
            continue
        fr_path = study_path / "final_results.json"
        if not fr_path.exists():
            continue
        with fr_path.open() as f:
            d = json.load(f)
        r2s = d.get("test_r2_scores", [])
        for fold_idx, r2 in enumerate(r2s, start=1):
            rows.append({"Dataset": display, "Type": dtype, "Fold": fold_idx,
                         "Model": "single", "R2_Score": r2})
    return pd.DataFrame(rows)


# %%
print("=" * 100)
print("DUAL vs SINGLE — comparison batch")
print("=" * 100)

pos_dual   = collect_dual('pos')
pos_single = collect_single('pos')
vel_dual   = collect_dual('vel')
vel_single = collect_single('vel')

print(f"Position dual   → {pos_dual['Dataset'].nunique()} datasets, {len(pos_dual)} folds")
print(f"Position single → {pos_single['Dataset'].nunique()} datasets, {len(pos_single)} folds")
print(f"Velocity dual   → {vel_dual['Dataset'].nunique()} datasets, {len(vel_dual)} folds")
print(f"Velocity single → {vel_single['Dataset'].nunique()} datasets, {len(vel_single)} folds")


# %%
# --- comparative boxplots (dual + single side-by-side per dataset) ---------

def create_comparison_boxplot(dual_df, single_df, target_label, save_path,
                              figsize=(16, 8)):
    df = pd.concat([dual_df, single_df], ignore_index=True)
    if df.empty:
        print(f"❌ No data for {target_label}")
        return None, None

    available = set(df['Dataset'].unique())
    dataset_order = [d for d in FIXED_ORDER if d in available]

    plt.style.use('default')
    fig, ax = plt.subplots(figsize=figsize, dpi=200)

    sns.boxplot(
        data=df, x='Dataset', y='R2_Score', hue='Model',
        order=dataset_order, hue_order=['dual', 'single'],
        palette=MODEL_COLOR, ax=ax,
        showfliers=False, linewidth=1.5,
    )
    sns.stripplot(
        data=df, x='Dataset', y='R2_Score', hue='Model',
        order=dataset_order, hue_order=['dual', 'single'],
        dodge=True, color='0.25', size=3, alpha=0.5, ax=ax,
        legend=False,
    )

    ax.set_xlabel('Dataset', fontsize=18, fontweight='bold')
    ax.set_ylabel(f'Accuracy (R²) - {target_label}', fontsize=18, fontweight='bold')
    ax.set_title(
        f'{target_label} — Dual vs Single output\n(5-Fold CV, R² per outer fold)',
        fontsize=20, fontweight='bold', pad=20,
    )
    plt.xticks(rotation=45, ha='right', fontsize=13)
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 0.8)

    ax.legend(
        title='Model output', title_fontsize=14, fontsize=13,
        loc='upper left', bbox_to_anchor=(1.02, 1),
        frameon=True, fancybox=True, shadow=True,
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, format='svg', bbox_inches='tight')
    print(f"📁 {target_label} boxplot saved: {save_path}")
    plt.close(fig)
    return fig, ax


create_comparison_boxplot(pos_dual, pos_single, 'Position',
                          save_path=f'{plot_path}/position_dual_vs_single_boxplot.svg')
create_comparison_boxplot(vel_dual, vel_single, 'Velocity',
                          save_path=f'{plot_path}/velocity_dual_vs_single_boxplot.svg')


# %%
# --- scatters: R² single (X) vs R² dual (Y), median per dataset ------------

def create_dual_vs_single_scatter(dual_df, single_df, target_label, save_path,
                                  figsize=(10, 8)):
    dual_med = dual_df.groupby(['Dataset', 'Type'])['R2_Score'].median().reset_index()
    dual_med = dual_med.rename(columns={'R2_Score': 'dual_r2'})
    single_med = single_df.groupby(['Dataset', 'Type'])['R2_Score'].median().reset_index()
    single_med = single_med.rename(columns={'R2_Score': 'single_r2'})
    merged = dual_med.merge(single_med, on=['Dataset', 'Type'])

    if merged.empty:
        print(f"❌ No overlap between dual and single for {target_label}")
        return None, None

    plt.style.use('default')
    fig, ax = plt.subplots(figsize=figsize, dpi=200)

    # Identity line — if single ≈ dual, points land on it.
    lo = min(merged['single_r2'].min(), merged['dual_r2'].min()) - 0.05
    hi = max(merged['single_r2'].max(), merged['dual_r2'].max()) + 0.05
    lo = max(lo, -1.0)
    ax.plot([lo, hi], [lo, hi], linestyle='--', color='#888', linewidth=1.2,
            label='Identity (dual == single)')

    for dtype in DATASET_TYPE_ORDER:
        sub = merged[merged['Type'] == dtype]
        if sub.empty:
            continue
        ax.scatter(
            sub['single_r2'], sub['dual_r2'],
            marker='X', s=200, color=COLOR_SCHEME[dtype], alpha=0.8,
            label=TYPE_DISPLAY[dtype], edgecolors='black', linewidth=1.5,
        )

    ax.set_xlabel(f'{target_label} R² — single-output model', fontsize=16, fontweight='bold')
    ax.set_ylabel(f'{target_label} R² — dual-output model', fontsize=16, fontweight='bold')
    ax.set_title(
        f'{target_label}: dual vs single — median R² per dataset',
        fontsize=18, fontweight='bold', pad=15,
    )
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(title='Dataset Type', title_fontsize=13, fontsize=11,
              loc='upper left', bbox_to_anchor=(1.02, 1),
              frameon=True, fancybox=True, shadow=True)

    for _, row in merged.iterrows():
        short = row['Dataset'].replace(' - ', '\n')
        ax.annotate(
            short, (row['single_r2'], row['dual_r2']),
            xytext=(6, 6), textcoords='offset points',
            fontsize=10, ha='left', va='bottom',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7),
        )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(save_path, format='svg', bbox_inches='tight')
    print(f"📁 {target_label} scatter saved: {save_path}")
    plt.close(fig)
    return fig, ax


create_dual_vs_single_scatter(pos_dual, pos_single, 'Position',
                              save_path=f'{plot_path}/position_scatter_dual_vs_single.svg')
create_dual_vs_single_scatter(vel_dual, vel_single, 'Velocity',
                              save_path=f'{plot_path}/velocity_scatter_dual_vs_single.svg')


# %%
# --- comparison table: medians + deltas per dataset ------------------------

def build_comparison_table(pos_dual, pos_single, vel_dual, vel_single):
    def med_by_dataset(df):
        return df.groupby(['Dataset', 'Type'])['R2_Score'].median().reset_index()

    pos_d = med_by_dataset(pos_dual).rename(columns={'R2_Score': 'r2_pos_dual'})
    pos_s = med_by_dataset(pos_single).rename(columns={'R2_Score': 'r2_pos_single'})
    vel_d = med_by_dataset(vel_dual).rename(columns={'R2_Score': 'r2_vel_dual'})
    vel_s = med_by_dataset(vel_single).rename(columns={'R2_Score': 'r2_vel_single'})

    t = (
        pos_d.merge(pos_s, on=['Dataset', 'Type'], how='outer')
             .merge(vel_d, on=['Dataset', 'Type'], how='outer')
             .merge(vel_s, on=['Dataset', 'Type'], how='outer')
    )
    t['delta_pos'] = t['r2_pos_dual'] - t['r2_pos_single']
    t['delta_vel'] = t['r2_vel_dual'] - t['r2_vel_single']

    # Order by FIXED_ORDER.
    order_map = {d: i for i, d in enumerate(FIXED_ORDER)}
    t = t.sort_values(by='Dataset', key=lambda s: s.map(order_map).fillna(len(FIXED_ORDER)))
    t = t.reset_index(drop=True)
    return t


table = build_comparison_table(pos_dual, pos_single, vel_dual, vel_single)
csv_path = f'{plot_path}/comparison_table.csv'
table.to_csv(csv_path, index=False, float_format='%.4f')
print(f"📁 CSV saved: {csv_path}")


# Human-readable Markdown twin.
def write_md_table(t, md_path):
    lines = [
        "# Dual vs single — median R² per dataset (comparison)",
        "",
        f"Generated by `exploration/07-comparison-dual-vs-single.py`.",
        "",
        "**delta = R² dual − R² single**. Positive delta means the dual-output "
        "model does better at that target than the single-output model. "
        "Negative delta means single is doing better.",
        "",
        "| Dataset | Type | R² pos (dual) | R² pos (single) | Δ pos | R² vel (dual) | R² vel (single) | Δ vel |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    def fmt(x):
        if pd.isna(x):
            return "—"
        return f"{x:+.3f}"

    for _, row in t.iterrows():
        lines.append(
            f"| {row['Dataset']} | {TYPE_DISPLAY.get(row['Type'], row['Type'])} "
            f"| {fmt(row['r2_pos_dual'])} | {fmt(row['r2_pos_single'])} | {fmt(row['delta_pos'])} "
            f"| {fmt(row['r2_vel_dual'])} | {fmt(row['r2_vel_single'])} | {fmt(row['delta_vel'])} |"
        )

    # Aggregate rows per type.
    lines.append("")
    lines.append("## Aggregate per type (median of dataset medians)")
    lines.append("")
    lines.append("| Type | R² pos (dual) | R² pos (single) | Δ pos | R² vel (dual) | R² vel (single) | Δ vel |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for dtype in DATASET_TYPE_ORDER:
        sub = t[t['Type'] == dtype]
        if sub.empty:
            continue
        lines.append(
            f"| {TYPE_DISPLAY[dtype]} "
            f"| {fmt(sub['r2_pos_dual'].median())} | {fmt(sub['r2_pos_single'].median())} | {fmt(sub['delta_pos'].median())} "
            f"| {fmt(sub['r2_vel_dual'].median())} | {fmt(sub['r2_vel_single'].median())} | {fmt(sub['delta_vel'].median())} |"
        )

    # Global aggregate.
    lines.append("")
    lines.append("## Global (median across the 17 datasets)")
    lines.append("")
    lines.append(f"- Δ pos median: **{fmt(t['delta_pos'].median())}**")
    lines.append(f"- Δ vel median: **{fmt(t['delta_vel'].median())}**")

    Path(md_path).write_text("\n".join(lines) + "\n")


md_path = f'{plot_path}/comparison_table.md'
write_md_table(table, md_path)
print(f"📁 MD saved: {md_path}")


# %%
# --- headline stats to stdout ----------------------------------------------

print()
print("=" * 100)
print("HEADLINE STATS")
print("=" * 100)
print(f"Position — median Δ (dual − single) across 17 datasets: {table['delta_pos'].median():+.3f}")
print(f"Velocity — median Δ (dual − single) across 17 datasets: {table['delta_vel'].median():+.3f}")
print()
print("Per type:")
for dtype in DATASET_TYPE_ORDER:
    sub = table[table['Type'] == dtype]
    if sub.empty:
        continue
    dp = sub['delta_pos'].median()
    dv = sub['delta_vel'].median()
    print(f"  {TYPE_DISPLAY[dtype]:<20} (n={len(sub)})   Δ pos={dp:+.3f}   Δ vel={dv:+.3f}")
