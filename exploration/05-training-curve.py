# %% [markdown]
# # Curva de entrenamiento (train vs. validación) — figura para la presentación
#
# Regenera la figura de la curva de pérdida (MSE) por época de una corrida real,
# mostrando entrenamiento vs. validación, el punto de mejor época (donde el early
# stopping conserva los pesos) y la ventana de `patience`.
#
# Los datos salen de los CSV de métricas que guarda `ibioml.trainer` por cada trial:
# `results/<study>/training_results/fold_<F>/trial_<T>/<run_id>_metrics.csv`
#
# Para regenerar con otra corrida, editá el bloque CONFIG y volvé a correr:
#     uv run python exploration/05-training-curve.py

# %%
import glob
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend no interactivo: guarda archivos sin abrir ventanas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %% [markdown]
# ## CONFIG — editá esto para regenerar con otra corrida

# %%
RESULTS_DIR = "results"
OUT_DIR = "results/plots/presentacion"

STUDY = "mlp_Pir_L3_expert_bins200ms_dual"  # carpeta bajo results/
FOLD = 5          # fold externo (1..outer_folds)
TRIAL = 3         # trial de Optuna; None = elige automáticamente el mejor del fold
PATIENCE = 20     # solo para rotular; debe coincidir con es_patience del experimento

REGION_NAMES = {"Pir": "Corteza piriforme", "CA3": "Hipocampo CA3", "DG": "Giro dentado (DG)"}
COHORT_NAMES = {"expert": "experto", "naive": "naïve"}
TARGET_NAMES = {"dual": "posición y velocidad", "pos": "posición", "vel": "velocidad"}

# Paleta (continúa la del deck)
INK, GREY, VAL, ACC = "#1a1a1a", "#a49b90", "#2f4f80", "#d98324"

# %% [markdown]
# ## Helpers

# %%
def describe_study(study: str):
    """`mlp_Pir_L3_expert_bins200ms_dual` -> ('Corteza piriforme · experto · L3', 'posición y velocidad')."""
    name = study[4:] if study.startswith("mlp_") else study
    left, _, target = name.partition("_bins200ms_")
    parts = left.split("_")
    region, cohort = parts[0], parts[-1]
    animal = "_".join(parts[1:-1])
    label = f"{REGION_NAMES.get(region, region)} · {COHORT_NAMES.get(cohort, cohort)} · {animal}"
    return label, TARGET_NAMES.get(target, target)


def pick_best_trial(fold_dir: str) -> int:
    """Trial con menor val_loss mínimo del fold (≈ el que selecciona Optuna)."""
    best_trial, best_val = None, np.inf
    for td in glob.glob(os.path.join(fold_dir, "trial_*")):
        csv = glob.glob(os.path.join(td, "*_metrics.csv"))
        if not csv:
            continue
        vmin = pd.read_csv(csv[0])["val_loss"].min()
        if vmin < best_val:
            best_val, best_trial = vmin, int(re.search(r"trial_(\d+)", td).group(1))
    if best_trial is None:
        raise FileNotFoundError(f"No hay metrics CSV en {fold_dir}")
    return best_trial


def load_metrics(study: str, fold: int, trial: int | None):
    fold_dir = os.path.join(RESULTS_DIR, study, "training_results", f"fold_{fold}")
    if trial is None:
        trial = pick_best_trial(fold_dir)
    csv = glob.glob(os.path.join(fold_dir, f"trial_{trial}", "*_metrics.csv"))
    if not csv:
        raise FileNotFoundError(f"No se encontró metrics CSV para fold {fold}, trial {trial}")
    return pd.read_csv(csv[0]), trial

# %% [markdown]
# ## Cargar la corrida

# %%
df, trial = load_metrics(STUDY, FOLD, TRIAL)
ep = df["epoch"].to_numpy()
tr = df["train_loss"].to_numpy()
val = df["val_loss"].to_numpy()
r2 = df["r2_score"].to_numpy() if "r2_score" in df else None

best = int(np.argmin(val))
best_ep, last_ep = int(ep[best]), int(ep[-1])
region_label, target_label = describe_study(STUDY)
r2_best = float(r2[best]) if r2 is not None else float("nan")

print(f"{STUDY} | fold {FOLD} | trial {trial} | épocas={len(ep)} | "
      f"mejor época={best_ep} | R²={r2_best:.3f}")

# %% [markdown]
# ## Figura

# %%
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
})

fig, ax = plt.subplots(figsize=(10, 5.6), dpi=200)

# ventana de patience (desde la mejor época hasta el corte)
ax.axvspan(best_ep, last_ep, color=ACC, alpha=0.09, zorder=0)

ax.plot(ep, tr, color=GREY, lw=2.2, label="Pérdida de entrenamiento", zorder=3)
ax.plot(ep, val, color=VAL, lw=2.5, label="Pérdida de validación", zorder=4)
ax.axvline(best_ep, color=ACC, ls="--", lw=1.8, zorder=2)
ax.plot([best_ep], [val[best]], "o", color=ACC, ms=8, zorder=6)

ax.set_yscale("log")
ax.set_ylim(min(0.33, float(val.min()) * 0.7), float(val.max()) * 1.5)
ax.set_xlim(1, last_ep + 1)
ax.set_xlabel("Época", fontsize=13)
ax.set_ylabel("Pérdida — MSE  (escala log)", fontsize=13)

# mejor época (early stopping)
ax.annotate(f"mejor época ({best_ep})\nse conservan estos pesos",
            xy=(best_ep, val[best]), xytext=(best_ep - 4, 9),
            fontsize=10.5, color=INK, ha="right", va="bottom",
            arrowprops=dict(arrowstyle="->", color=ACC, lw=1.5))

# bracket de patience
ytop = float(val.max()) * 0.55
ax.annotate("", xy=(best_ep, ytop), xytext=(last_ep, ytop),
            arrowprops=dict(arrowstyle="<->", color=ACC, lw=1.4))
ax.text((best_ep + last_ep) / 2, ytop * 1.3,
        f"{last_ep - best_ep} épocas sin mejora\n$\\rightarrow$ se detiene (paciencia = {PATIENCE})",
        ha="center", va="bottom", fontsize=10, color=ACC)

# nota de sobreajuste
ax.text(0.16, 0.04,
        "entrenamiento sigue bajando, validación se estanca  $\\rightarrow$  sobreajuste",
        transform=ax.transAxes, fontsize=9.5, color=GREY, style="italic", ha="left")

# título + caption (incluye el FOLD de donde sale el gráfico)
ax.set_title("Curva de entrenamiento (datos reales)", fontsize=15, fontweight="bold",
             loc="left", pad=16)
caption = (f"{region_label}   —   decodificación de {target_label}"
           f"   ·   fold {FOLD}   ·   R² = {r2_best:.2f}")
ax.text(0, 1.015, caption, transform=ax.transAxes, fontsize=10.5, color=GREY)

ax.legend(frameon=False, loc="center right", fontsize=11)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.spines["left"].set_color("#cfc8bd")
ax.spines["bottom"].set_color("#cfc8bd")
fig.tight_layout()

# %%
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
stem = f"curva_entrenamiento_{STUDY.replace('mlp_', '')}_fold{FOLD}"
for ext in ("png", "svg", "pdf"):
    out = os.path.join(OUT_DIR, f"{stem}.{ext}")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print("guardado:", out)
