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
# # Decoding results — single-output experiments (pos-only and vel-only)
#
# Mirror of `03-decoding-results.py` for the single-output batch (Item 1 of
# Phase 5). Reads:
#
# - `results/mlp_<dataset>_bins200ms_pos/` — 17 studies, `output_size=1`, position target.
# - `results/mlp_<dataset>_bins200ms_vel/` — 17 studies, `output_size=1`, velocity target.
#
# Emits under `results/plots/single/performance/`:
#
# - `position_performance_boxplot.svg` — R² by dataset (position-only model).
# - `velocity_performance_boxplot.svg` — R² by dataset (velocity-only model).
# - `position_vs_velocity_scatter.svg` — median R² of both single-target models, one point per dataset.
#
# **No exclusions** — all 17 datasets are plotted (unlike `03-decoding-results.py`
# which drops S38 / S41 / N59).

# %%
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — no GUI window per figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %%
# --- config -----------------------------------------------------------------

results_base_dir = 'results'
plot_path = 'results/plots/single/performance'
Path(plot_path).mkdir(parents=True, exist_ok=True)

REGION_DISPLAY = {"Pir": "Pir", "CA3": "CA3", "DG": "DG"}
COHORT_DISPLAY = {"expert": "Expert", "naive": "Naive"}
COHORT_TYPE_SUFFIX = {"expert": "_experto", "naive": "_naive"}

# Type-level color scheme (kept consistent with 03-decoding-results.py).
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

# All 17 datasets, sorted by type then animal — no exclusions this time.
FIXED_ORDER = [
    'Pir-Naive-N53', 'Pir-Naive-L7',
    'Pir-Expert-N31', 'Pir-Expert-L3', 'Pir-Expert-L5', 'Pir-Expert-N59',
    'CA3-Naive-S30', 'CA3-Naive-S31',
    'CA3-Expert-S17', 'CA3-Expert-S18',
    'DG-Naive-S24', 'DG-Naive-S38', 'DG-Naive-S39', 'DG-Naive-S41',
    'DG-Expert-S17', 'DG-Expert-S18', 'DG-Expert-S19',
]


def parse_study_dir_name(study_dir_name, expected_suffix):
    """Parse `mlp_<REGION>_<ID>_<cohort>_bins200ms_<suffix>` → (display, dtype)."""
    prefix = "mlp_"
    suffix = f"_{expected_suffix}"
    if not study_dir_name.startswith(prefix) or not study_dir_name.endswith(suffix):
        return None, None
    core = study_dir_name[len(prefix):-len(suffix)]
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


def load_fold_r2_scores(study_path):
    """Return the list of 5 test R² scores from a single-output study."""
    p = Path(study_path) / "final_results.json"
    if not p.exists():
        return None
    with p.open() as f:
        d = json.load(f)
    return list(d.get("test_r2_scores", []))


def collect_single_target_studies(suffix):
    """Walk results/ and return a DataFrame with per-fold R² for every mlp_*_<suffix>/ study."""
    rows = []
    for entry in sorted(os.listdir(results_base_dir)):
        study_path = Path(results_base_dir) / entry
        if not study_path.is_dir():
            continue
        display, dtype = parse_study_dir_name(entry, suffix)
        if display is None:
            continue
        r2s = load_fold_r2_scores(study_path)
        if not r2s:
            print(f"   ⚠️  {entry}: no fold R² scores, skip")
            continue
        for fold_idx, r2 in enumerate(r2s, start=1):
            rows.append({
                "Dataset": display,
                "Type": dtype,
                "Fold": fold_idx,
                "R2_Score": r2,
            })
    return pd.DataFrame(rows)


# %%
# --- discover studies -------------------------------------------------------

print("=" * 100)
print("SINGLE-OUTPUT RESULTS (Item 1 of Phase 5)")
print("=" * 100)

pos_df = collect_single_target_studies("pos")
vel_df = collect_single_target_studies("vel")
print(f"Position studies found: {pos_df['Dataset'].nunique()} datasets, {len(pos_df)} folds total")
print(f"Velocity studies found: {vel_df['Dataset'].nunique()} datasets, {len(vel_df)} folds total")


# %%
# --- boxplot helper (uses seaborn, matches 03-decoding-results.py style) ----

def create_single_boxplot(df, target_label, save_path, figsize=(14, 8)):
    """Boxplot of per-fold R² by dataset, colored by dataset type."""
    if df.empty:
        print(f"❌ No data for {target_label}")
        return None, None

    available = set(df['Dataset'].unique())
    dataset_order = [d for d in FIXED_ORDER if d in available]

    # Per-dataset color, driven by its type.
    dataset_colors = {}
    for dataset in dataset_order:
        dataset_type = df[df['Dataset'] == dataset]['Type'].iloc[0]
        dataset_colors[dataset] = COLOR_SCHEME.get(dataset_type, '#333333')

    plt.style.use('default')
    fig, ax = plt.subplots(figsize=figsize, dpi=200)

    sns.boxplot(
        data=df, x='Dataset', y='R2_Score', order=dataset_order,
        palette=dataset_colors, ax=ax,
        showfliers=False, linewidth=1.5,
    )
    sns.stripplot(
        data=df, x='Dataset', y='R2_Score', order=dataset_order,
        color='0.25', size=3, alpha=0.5, ax=ax,
    )

    ax.set_xlabel('Dataset', fontsize=18, fontweight='bold')
    ax.set_ylabel(f'Accuracy (R²) - {target_label}', fontsize=18, fontweight='bold')
    ax.set_title(
        f'{target_label} Decoding Performance by Dataset — single-output model\n'
        '(5-Fold Cross-Validation)',
        fontsize=20, fontweight='bold', pad=20,
    )
    plt.xticks(rotation=45, ha='right', fontsize=15)
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 0.8)

    # Legend with per-dataset medians.
    medians = df.groupby('Dataset')['R2_Score'].median()
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=dataset_colors[d], edgecolor='black',
              label=f'{d}: {medians[d]:.3f}')
        for d in dataset_order
    ]
    ax.legend(
        handles=handles,
        title=f"{target_label} Accuracy (R²) - Median",
        title_fontsize=16, fontsize=15,
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


# %%
create_single_boxplot(pos_df, 'Position',
                      save_path=f'{plot_path}/position_performance_boxplot.svg')
create_single_boxplot(vel_df, 'Velocity',
                      save_path=f'{plot_path}/velocity_performance_boxplot.svg')


# %%
# --- scatter: median R² position (single) vs median R² velocity (single) ----

def create_pos_vs_vel_scatter(pos_df, vel_df, save_path, figsize=(12, 8)):
    """Scatter of median R² per dataset — position vs velocity single-output."""
    pos_med = pos_df.groupby(['Dataset', 'Type'])['R2_Score'].median().reset_index()
    pos_med = pos_med.rename(columns={'R2_Score': 'position_r2'})
    vel_med = vel_df.groupby(['Dataset', 'Type'])['R2_Score'].median().reset_index()
    vel_med = vel_med.rename(columns={'R2_Score': 'velocity_r2'})
    scatter_df = pos_med.merge(vel_med, on=['Dataset', 'Type'])

    if scatter_df.empty:
        print("❌ No overlap between pos and vel studies")
        return None, None

    plt.style.use('default')
    fig, ax = plt.subplots(figsize=figsize, dpi=200)

    for dtype in DATASET_TYPE_ORDER:
        sub = scatter_df[scatter_df['Type'] == dtype]
        if sub.empty:
            continue
        ax.scatter(
            sub['position_r2'], sub['velocity_r2'],
            marker='X', s=200, color=COLOR_SCHEME[dtype], alpha=0.8,
            label=TYPE_DISPLAY[dtype], edgecolors='black', linewidth=1.5,
        )

    ax.set_xlabel('Position Accuracy (R²) - Median', fontsize=18, fontweight='bold')
    ax.set_ylabel('Speed Accuracy (R²) - Median', fontsize=18, fontweight='bold')
    ax.set_title(
        'Performance Comparison: Position vs Speed — single-output models\n'
        '(Median R² per Dataset)',
        fontsize=20, fontweight='bold', pad=20,
    )
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(title='Dataset Type', title_fontsize=16, fontsize=15,
              loc='upper left', frameon=True, fancybox=True, shadow=True)

    for _, row in scatter_df.iterrows():
        short = row['Dataset'].replace(' - ', '\n')
        ax.annotate(
            short, (row['position_r2'], row['velocity_r2']),
            xytext=(8, 8), textcoords='offset points',
            fontsize=13, ha='left', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
        )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Autoscale with margin, starting at 0 (consistent with 03's scatter).
    x_margin = scatter_df['position_r2'].max() * 0.1
    y_margin = scatter_df['velocity_r2'].max() * 0.1
    ax.set_xlim(0.0, scatter_df['position_r2'].max() + x_margin)
    ax.set_ylim(0.0, scatter_df['velocity_r2'].max() + y_margin)

    plt.tight_layout()
    plt.savefig(save_path, format='svg', bbox_inches='tight')
    print(f"📁 scatter saved: {save_path}")
    plt.close(fig)
    return fig, ax


create_pos_vs_vel_scatter(
    pos_df, vel_df,
    save_path=f'{plot_path}/position_vs_velocity_scatter.svg',
)


# %%
# --- quick per-type summary printed to stdout -------------------------------

print()
print("=" * 100)
print("SUMMARY — median R² per dataset type (single-output)")
print("=" * 100)
for dtype in DATASET_TYPE_ORDER:
    pos_sub = pos_df[pos_df['Type'] == dtype]
    vel_sub = vel_df[vel_df['Type'] == dtype]
    if pos_sub.empty and vel_sub.empty:
        continue
    pos_med = pos_sub['R2_Score'].median() if not pos_sub.empty else float('nan')
    vel_med = vel_sub['R2_Score'].median() if not vel_sub.empty else float('nan')
    n = pos_sub['Dataset'].nunique()
    print(f"  {TYPE_DISPLAY[dtype]:<20} (n={n})   "
          f"pos median={pos_med:+.3f}   vel median={vel_med:+.3f}")
