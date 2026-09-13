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
# # Position/velocity decoupling — post-hoc analyses (Phase 0: A1, A2, A3)
#
# Is the position decoder reading position, or reconstructing it from the
# animal's speed via the stereotyped running profile? Three post-hoc tests on
# the stored test predictions (baseline single-output studies, Item 1):
#
# - **A1 — within-position coupling.** Inside each 2 cm stratum of *true*
#   position, correlate the animal's speed with the signed prediction error.
#   A decoder that ignores speed gives ≈ 0; a positive value ("faster ⇒
#   predicted further ahead") is the fingerprint of speed-mediated decoding.
#   Speed and error are z-scored *within stratum* (robust median/MAD), so the
#   pooled correlation is a partial correlation given position.
# - **A2 — broken-stereotypy bins** (plan Item 5). The stereotypy profile is
#   the median speed per position stratum; a bin is **atypical** when its
#   speed deviates |z| > 2 from its stratum (slow-atypical: pausing where the
#   animal usually runs; fast-atypical: running where it usually slows). If
#   the decoder reads position, its error on atypical bins at matched
#   positions should hold; if it reads speed, it should blow up. Δ MAE is
#   position-matched: per-stratum deltas weighted by the atypical counts,
#   with a bin-bootstrap CI.
# - **A3 — error correlation between decoders.** The pos-only and vel-only
#   studies share the fold partition, so every test bin has both predictions.
#   Within-position correlation of their |errors|: high ⇒ both models fail on
#   the same bins (shared underlying signal); ≈ 0 ⇒ independent codes.
#
# Outputs under `results/plots/decoupling/posthoc/`: per dataset a didactic
# stereotypy-profile figure and a coupling figure, plus `summary_table.md/csv`.

# %%
import os
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent if "__file__" in dir() else Path.cwd().parent
sys.path.insert(0, str(ROOT))

from analysis.loaders import CM_PER_DEGREE, CONTEXT_END_CM, CONTEXT_START_CM  # noqa: E402
from analysis.results import (  # noqa: E402
    fold_predictions,
    fold_test_trials,
    load_final_results,
    parse_study_name,
    pickle_path,
    pool_folds,
    pooled_test_rows,
)

# %%
# --- config -----------------------------------------------------------------

DATASETS = os.environ.get(
    "DATASETS",
    "Pir_L3_expert,Pir_L5_expert,Pir_N31_expert,Pir_N59_expert,CA3_S18_expert,DG_S17_expert",
).split(",")
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data" / "processed"))
PLOT_DIR = Path(os.environ.get("PLOT_DIR", ROOT / "results" / "plots" / "decoupling" / "posthoc"))
PLOT_DIR.mkdir(parents=True, exist_ok=True)

BIN_CM = 2.0
MIN_STRATUM = 30      # bins a stratum needs before it enters any statistic
MIN_ATYP = 20         # atypical bins per stratum for the per-stratum curves
Z_ATYPICAL = 2.0      # |z| beyond this = broken stereotypy (agreed 2026-09-13)
Z_TYPICAL = 1.0       # |z| within this = typical
POS_MAX_CM = CONTEXT_END_CM  # analyse inside the corridor proper; the tail has its own story
BOOT = 1000
INK = "#333333"
GRID = dict(color="0.85", linewidth=0.6)
COLORS = {"typical": "#7B848F", "slow": "#7E57C2", "fast": "#E76F51"}


# %%
# --- load one dataset: aligned pos + vel predictions, speed, folds -----------

def load_pair(stem_short):
    stem = f"{stem_short}_bins200ms"
    info_pos = parse_study_name(f"mlp_{stem}_pos", "pos")
    info_vel = parse_study_name(f"mlp_{stem}_vel", "vel")
    with pickle_path(DATA_DIR, info_pos).open("rb") as f:
        _, y, trial_markers = pickle.load(f)  # noqa: S301 — our own artefact
    rows_p = pooled_test_rows(RESULTS_DIR, info_pos.study, y[:, 0], trial_markers)
    rows_v = pooled_test_rows(RESULTS_DIR, info_vel.study, y[:, 1], trial_markers)
    if not np.array_equal(rows_p, rows_v):
        raise ValueError(f"{stem}: pos and vel studies do not share the fold partition")
    trials_pos = fold_test_trials(RESULTS_DIR, info_pos.study)
    trials_vel = fold_test_trials(RESULTS_DIR, info_vel.study)
    if trials_pos != trials_vel:
        raise ValueError(f"{stem}: fold trial lists differ between pos and vel studies")
    fp = load_final_results(RESULTS_DIR, info_pos.study)
    fv = load_final_results(RESULTS_DIR, info_vel.study)
    pos_t, pos_p, fold_id = pool_folds(
        [(t * CM_PER_DEGREE, p * CM_PER_DEGREE) for t, p in fold_predictions(fp, "position")])
    # vel study is single-output: fold_predictions returns its flat lists as-is
    vel_t, vel_p, _ = pool_folds(list(fold_predictions(fv, "position")))
    keep = pos_t <= POS_MAX_CM
    return {"display": info_pos.display,
            "pos_true": pos_t[keep], "pos_pred": pos_p[keep],
            "vel_true": vel_t[keep], "vel_pred": vel_p[keep],
            "fold_id": fold_id[keep]}


def robust_z(values, groups):
    """z-score within group using median and MAD (1.4826·MAD ≈ σ); NaN for thin groups."""
    z = np.full(len(values), np.nan)
    for g in np.unique(groups):
        m = groups == g
        if m.sum() < MIN_STRATUM:
            continue
        med = np.median(values[m])
        mad = 1.4826 * np.median(np.abs(values[m] - med))
        if mad > 0:
            z[m] = (values[m] - med) / mad
    return z


def prepare(d):
    """Attach strata and within-position z-scores to a loaded dataset."""
    edges = np.arange(0, POS_MAX_CM + BIN_CM, BIN_CM)
    d["stratum"] = np.digitize(d["pos_true"], edges) - 1
    d["edges"] = edges
    d["err"] = d["pos_pred"] - d["pos_true"]
    d["z_speed"] = robust_z(d["vel_true"], d["stratum"])
    d["z_err"] = robust_z(d["err"], d["stratum"])
    d["z_abs_pos"] = robust_z(np.abs(d["err"]), d["stratum"])
    d["z_abs_vel"] = robust_z(np.abs(d["vel_pred"] - d["vel_true"]), d["stratum"])
    return d


# %%
# --- the three statistics -----------------------------------------------------

def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 10 else np.nan


def per_fold_stat(d, a_key, b_key):
    return [corr(d[a_key][d["fold_id"] == k], d[b_key][d["fold_id"] == k])
            for k in np.unique(d["fold_id"])]


def matched_delta_mae(d, atyp_mask):
    """Position-matched ΔMAE (atypical − typical), weighted by atypical counts, + bootstrap CI."""
    typ = np.abs(d["z_speed"]) <= Z_TYPICAL

    def delta(idx_a, idx_t):
        out, weights = [], []
        for s in np.unique(d["stratum"]):
            ma = idx_a & (d["stratum"] == s)
            mt = idx_t & (d["stratum"] == s)
            if ma.sum() >= MIN_ATYP and mt.sum() >= MIN_STRATUM:
                out.append(np.mean(np.abs(d["err"][ma])) - np.mean(np.abs(d["err"][mt])))
                weights.append(ma.sum())
        return np.average(out, weights=weights) if out else np.nan

    point = delta(atyp_mask, typ)
    rng = np.random.default_rng(0)
    n = len(d["err"])
    boots = []
    for _ in range(BOOT):
        take = rng.integers(0, n, n)  # bootstrap over bins: resample, recompute masks
        da = {k: d[k][take] for k in ("err", "stratum", "z_speed")}
        typ_b = np.abs(da["z_speed"]) <= Z_TYPICAL
        atyp_b = atyp_mask[take]
        out, weights = [], []
        for s in np.unique(da["stratum"]):
            ma = atyp_b & (da["stratum"] == s)
            mt = typ_b & (da["stratum"] == s)
            if ma.sum() >= MIN_ATYP and mt.sum() >= MIN_STRATUM:
                out.append(np.mean(np.abs(da["err"][ma])) - np.mean(np.abs(da["err"][mt])))
                weights.append(ma.sum())
        boots.append(np.average(out, weights=weights) if out else np.nan)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return point, lo, hi, int(atyp_mask.sum())


def analyse(d):
    coupling = per_fold_stat(d, "z_speed", "z_err")
    errcorr = per_fold_stat(d, "z_abs_pos", "z_abs_vel")
    slow = d["z_speed"] < -Z_ATYPICAL
    fast = d["z_speed"] > Z_ATYPICAL
    d_slow = matched_delta_mae(d, slow)
    d_fast = matched_delta_mae(d, fast)
    return {"coupling_folds": coupling, "errcorr_folds": errcorr,
            "slow": d_slow, "fast": d_fast}


# %%
# --- figures -------------------------------------------------------------------

def style(ax):
    ax.grid(True, **GRID)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9, colors=INK)


def shade_context(ax):
    ax.axvspan(CONTEXT_START_CM, CONTEXT_END_CM, color="gold", alpha=0.15, lw=0)


def stereotypy_figure(d, name):
    """Didactic view: the speed-vs-position profile and what 'atypical' means."""
    centers = 0.5 * (d["edges"][:-1] + d["edges"][1:])
    med, p10, p90 = [np.full(len(centers), np.nan) for _ in range(3)]
    for s in range(len(centers)):
        v = d["vel_true"][d["stratum"] == s]
        if len(v) >= MIN_STRATUM:
            p10[s], med[s], p90[s] = np.percentile(v, [10, 50, 90])
    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    ax.fill_between(centers, p10, p90, color="0.75", alpha=0.4, lw=0,
                    label="p10–p90 (running stereotypy)")
    ax.plot(centers, med, color="0.2", linewidth=1.8, label="median speed")
    rng = np.random.default_rng(1)
    for kind, mask in (("slow", d["z_speed"] < -Z_ATYPICAL), ("fast", d["z_speed"] > Z_ATYPICAL)):
        idx = np.where(mask)[0]
        show = rng.choice(idx, min(400, len(idx)), replace=False) if len(idx) else idx
        ax.scatter(d["pos_true"][show], d["vel_true"][show], s=6, alpha=0.5,
                   color=COLORS[kind],
                   label=f"atypical {kind} (|z| > {Z_ATYPICAL:g}; n = {len(idx):,})")
    shade_context(ax)
    ax.set_xlabel("True position (cm)", fontsize=10, color=INK)
    ax.set_ylabel("Speed (cm/s)", fontsize=10, color=INK)
    ax.set_title(f"{name} — running stereotypy and the broken-stereotypy bins",
                 fontsize=10.5, color=INK, loc="left")
    ax.legend(fontsize=8.5, frameon=False)
    style(ax)
    fig.tight_layout()
    out = PLOT_DIR / f"{name}_stereotypy_profile.svg"
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"📁 {out.relative_to(ROOT)}")


def coupling_figure(d, name):
    centers = 0.5 * (d["edges"][:-1] + d["edges"][1:])
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), dpi=200, sharex=True)
    # top: per-stratum corr(speed, signed error)
    cs = np.full(len(centers), np.nan)
    for s in range(len(centers)):
        m = d["stratum"] == s
        if m.sum() >= MIN_STRATUM:
            cs[s] = corr(d["vel_true"][m], d["err"][m])
    axes[0].plot(centers, cs, color="#004E89", linewidth=1.7)
    axes[0].axhline(0, color="0.25", linewidth=0.9)
    axes[0].set_ylabel("corr(speed, signed error)\nwithin position stratum",
                       fontsize=9.5, color=INK)
    axes[0].set_ylim(-1, 1)
    # bottom: MAE by position for typical vs atypical bins
    groups = {"typical": np.abs(d["z_speed"]) <= Z_TYPICAL,
              "slow": d["z_speed"] < -Z_ATYPICAL,
              "fast": d["z_speed"] > Z_ATYPICAL}
    for kind, mask in groups.items():
        mae = np.full(len(centers), np.nan)
        need = MIN_STRATUM if kind == "typical" else MIN_ATYP
        for s in range(len(centers)):
            m = mask & (d["stratum"] == s)
            if m.sum() >= need:
                mae[s] = np.mean(np.abs(d["err"][m]))
        axes[1].plot(centers, mae, color=COLORS[kind], linewidth=1.7,
                     label=f"{kind} (n = {int(mask.sum()):,})")
    axes[1].set_ylabel("MAE (cm)", fontsize=10, color=INK)
    axes[1].set_xlabel("True position (cm)", fontsize=10, color=INK)
    axes[1].legend(fontsize=8.5, frameon=False)
    for ax in axes:
        shade_context(ax)
        style(ax)
    axes[0].set_title(f"{name} — is the position error coupled to speed?",
                      fontsize=10.5, color=INK, loc="left")
    fig.tight_layout()
    out = PLOT_DIR / f"{name}_coupling.svg"
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"📁 {out.relative_to(ROOT)}")


# %%
# --- run ------------------------------------------------------------------------

print("=" * 100)
print("DECOUPLING PHASE 0 — post-hoc (A1 coupling · A2 broken stereotypy · A3 error corr)")
print(f"analysed range: true position ≤ {POS_MAX_CM} cm")
print("=" * 100)

rows = []
for stem_short in DATASETS:
    d = prepare(load_pair(stem_short))
    name = d["display"]
    r = analyse(d)
    stereotypy_figure(d, name)
    coupling_figure(d, name)
    c, e = r["coupling_folds"], r["errcorr_folds"]
    ds_, lo_s, hi_s, n_s = r["slow"]
    df_, lo_f, hi_f, n_f = r["fast"]
    rows.append({
        "Dataset": name,
        "coupling_median": np.nanmedian(c),
        "coupling_min": np.nanmin(c), "coupling_max": np.nanmax(c),
        "errcorr_median": np.nanmedian(e),
        "n_atyp_slow": n_s, "dmae_slow": ds_, "dmae_slow_lo": lo_s, "dmae_slow_hi": hi_s,
        "n_atyp_fast": n_f, "dmae_fast": df_, "dmae_fast_lo": lo_f, "dmae_fast_hi": hi_f,
    })
    print(f"  {name:<16} coupling {np.nanmedian(c):+.2f} "
          f"[{np.nanmin(c):+.2f},{np.nanmax(c):+.2f}] · "
          f"err-corr {np.nanmedian(e):+.2f} · ΔMAE slow {ds_:+.1f} [{lo_s:+.1f},{hi_s:+.1f}] "
          f"(n={n_s:,}) · fast {df_:+.1f} [{lo_f:+.1f},{hi_f:+.1f}] (n={n_f:,})")

summary = pd.DataFrame(rows)
summary.to_csv(PLOT_DIR / "summary_table.csv", index=False, float_format="%.4f")


def write_md(t, path):
    lines = [
        "# Decoupling phase 0 — post-hoc summary",
        "",
        f"Baseline single-output studies, test bins with true position ≤ {POS_MAX_CM} cm, "
        f"strata of {BIN_CM:g} cm. **coupling** = within-position correlation between speed "
        "and the signed position error (median across 5 folds, [min, max]); 0 = the decoder "
        "ignores speed given position. **err-corr** = within-position correlation between the "
        "|errors| of the position and velocity decoders. **Δ MAE** = position-matched "
        "error "
        "difference, atypical (|z-speed| > 2) − typical (|z| ≤ 1), with 95 % bootstrap CI; "
        "positive = the decoder does worse when the stereotypy breaks.",
        "",
        "| Dataset | coupling | err-corr | Δ MAE slow-atypical [CI] (n) "
        "| Δ MAE fast-atypical [CI] (n) |",
        "|---|---:|---:|---:|---:|",
    ]
    def fnum(x, nd=1):
        return "—" if pd.isna(x) else f"{x:+.{nd}f}"

    for _, r in t.iterrows():
        lines.append(
            f"| {r['Dataset']} | {r['coupling_median']:+.2f} "
            f"[{r['coupling_min']:+.2f}, {r['coupling_max']:+.2f}] | {r['errcorr_median']:+.2f} "
            f"| {fnum(r['dmae_slow'])} [{fnum(r['dmae_slow_lo'])}, {fnum(r['dmae_slow_hi'])}] "
            f"({int(r['n_atyp_slow']):,}) "
            f"| {fnum(r['dmae_fast'])} [{fnum(r['dmae_fast_lo'])}, {fnum(r['dmae_fast_hi'])}] "
            f"({int(r['n_atyp_fast']):,}) |")
    Path(path).write_text("\n".join(lines) + "\n")


write_md(summary, PLOT_DIR / "summary_table.md")
print(f"📁 {(PLOT_DIR / 'summary_table.md').relative_to(ROOT)}")
