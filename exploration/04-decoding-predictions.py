# %% [markdown]
# # Visualización de Predicciones vs Datos Reales
# 
# Este script visualiza las predicciones de los modelos MLP comparadas con los datos reales
# para posición y velocidad, usando el fold más cercano a la mediana de cada dataset.
# 
# Inspirado en las funciones de ibioml.plots: extract_predictions, get_fold_closest_to_median, plot_predictions

# %%
# Celda 1: Carga de datos y funciones base

import json
import os
from scipy import io
from scipy.io import loadmat
import pandas as pd
import numpy as np
import matplotlib
# Use a non-interactive backend so `plt.show()` doesn't pop up a window for
# every figure when this script is run from the CLI. Files still get saved.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import glob
from matplotlib.patches import Patch

print("=" * 100)
print("VISUALIZACIÓN DE PREDICCIONES vs DATOS REALES")
print("=" * 100)

# Directorio base de resultados
# Directorio base de resultados y raws.
# Layout: results/mlp_<dataset>_bins200ms_dual/, data/raw/<region>/<stem>.mat.
results_base_dir = 'results'
base_dir = 'data/raw'

# # Mapeo de datasets
# dataset_mapping = {
#     'pir_experto': {
#         'pir experto L5': 'Pir-Expert-L5',
#         'pir experto L3': 'Pir-Expert-L3',
#         'pir experto N31': 'Pir-Expert-N31'
#     },
#     'pir_naive': {
#         'pir naive L7': 'Pir-Naive-L7',
#         'pir naive N53': 'Pir-Naive-N53'
#     },
#     'CA3_naive': {
#         'CA3 naive S30': 'CA3-Naive-S30',
#         'CA3 naive S31': 'CA3-Naive-S31'
#     },
#     'CA3_experto': {
#         'CA3 experto S18': 'CA3-Expert-S18',
#         'CA3 experto S17': 'CA3-Expert-S17'
#     },
#     'DG_naive': {
#         'DG naive S24': 'DG-Naive-S24',
#         #'DG naive S38': 'DG-Naive-S38',
#         'DG naive S39': 'DG-Naive-S39',
#     },
#     'DG_experto': {
#         'DG experto S19': 'DG-Expert-S19',
#         'DG experto S17': 'DG-Expert-S17',
#         'DG experto S18': 'DG-Expert-S18',
#     }
# }

# Mapeo automático de study dirs → display names / tipos / paths .mat.
#
# Naming convention (from scripts/run_all_experiments.py):
#     mlp_<REGION>_<ID>_<expert|naive>_bins200ms_dual
#   e.g.
#     mlp_Pir_L5_expert_bins200ms_dual  → label "Pir-Expert-L5", type "pir_experto"
#     mlp_CA3_S17_expert_bins200ms_dual → label "CA3-Expert-S17", type "CA3_experto"
#
# For every study dir we build:
#   dataset_mapping[type][study_dir_name] = {"label": display, "mat_File": path}
# so downstream code (e.g. dataset_mapping[type][name]["mat_File"]) keeps
# working without further changes.

REGION_DISPLAY = {"Pir": "Pir", "CA3": "CA3", "DG": "DG"}
REGION_TO_DIR = {"Pir": "piriform", "CA3": "ca3", "DG": "dg"}
COHORT_DISPLAY = {"expert": "Expert", "naive": "Naive"}
COHORT_TYPE_SUFFIX = {"expert": "_experto", "naive": "_naive"}

# Datasets excluidos de todos los plots — mismos que en 03-decoding-results.py.
EXCLUDED_DATASETS_FROM_PLOTS = {
    "DG-Naive-S38",
    "DG-Naive-S41",
    "Pir-Expert-N59",
}


def parse_study_dir_name(study_dir_name):
    """Extract (display_name, dataset_type, region_dir) from `mlp_<...>_dual`."""
    if not study_dir_name.startswith("mlp_") or not study_dir_name.endswith("_dual"):
        return None, None, None
    core = study_dir_name[len("mlp_"):-len("_dual")]
    if not core.endswith("_bins200ms"):
        return None, None, None
    core = core[: -len("_bins200ms")]
    parts = core.split("_")
    if len(parts) < 3:
        return None, None, None
    region = parts[0]
    cohort = parts[-1]
    dataset_id = "_".join(parts[1:-1])
    if region not in REGION_DISPLAY or cohort not in COHORT_DISPLAY:
        return None, None, None
    display = f"{REGION_DISPLAY[region]}-{COHORT_DISPLAY[cohort]}-{dataset_id}"
    dtype = f"{region}{COHORT_TYPE_SUFFIX[cohort]}"
    if dtype.startswith("Pir"):
        dtype = "pir" + dtype[len("Pir"):]
    region_dir = REGION_TO_DIR[region]
    return display, dtype, region_dir


dataset_mapping = {}
for _entry in sorted(os.listdir(results_base_dir)) if os.path.exists(results_base_dir) else []:
    if not os.path.isdir(os.path.join(results_base_dir, _entry)):
        continue
    _display, _dtype, _region_dir = parse_study_dir_name(_entry)
    if _display is None:
        continue
    if _display in EXCLUDED_DATASETS_FROM_PLOTS:
        print(f"   ⏭️  Excluded from plots: {_display}")
        continue
    # mat_File is a path relative to base_dir (== data/raw/), e.g. "piriform/Pir_L5_expert_bins200ms.mat".
    _stem = _entry[len("mlp_"):-len("_dual")]
    dataset_mapping.setdefault(_dtype, {})[_entry] = {
        "label": _display,
        "mat_File": f"{_region_dir}/{_stem}.mat",
    }

bin_duration = 200  # ms

timeBinsPlot = 2000  # 2000 bins × 200ms = ~6:40 min of recording shown per plot

plot_path = 'results/plots/predictions'
if not os.path.exists(plot_path):
    os.makedirs(plot_path)
    print(f"📁 Directorio '{plot_path}' creado")


def find_study_directories(base_path, dataset_mapping):
    """Encuentra los study dirs para cada dataset del mapping.

    Layout esperado (this project):
        <base_path>/<study_dir_name>/training_results/fold_N/results.json
    (donde study_dir_name es la key del sub-mapping, e.g. `mlp_..._dual`).
    """
    study_dirs = {}
    for dataset_type, datasets_dict in dataset_mapping.items():
        study_dirs[dataset_type] = {}
        for dataset_name in datasets_dict.keys():
            study_path = os.path.join(base_path, dataset_name)
            if os.path.isdir(os.path.join(study_path, "training_results")):
                study_dirs[dataset_type][dataset_name] = study_path
                print(f"   ✅ Found: {dataset_type}/{dataset_name}")
            else:
                print(f"   ⚠️  Missing training_results/: {dataset_type}/{dataset_name}")
    return study_dirs

def load_fold_results(study_path):
    """Carga los resultados de todos los folds de un estudio"""
    training_results_path = os.path.join(study_path, 'training_results')
    fold_results = {}
    
    if os.path.exists(training_results_path):
        for fold_dir in os.listdir(training_results_path):
            if fold_dir.startswith('fold_'):
                fold_num = int(fold_dir.split('_')[1])
                results_file = os.path.join(training_results_path, fold_dir, 'results.json')
                
                if os.path.exists(results_file):
                    with open(results_file, 'r') as f:
                        fold_data = json.load(f)
                    fold_results[fold_num] = fold_data
    
    return fold_results

def load_final_results(study_path):
    """Carga el archivo final_results.json que contiene predicciones y valores reales"""
    final_results_path = os.path.join(study_path, 'final_results.json')
    
    if os.path.exists(final_results_path):
        try:
            with open(final_results_path, 'r') as f:
                final_data = json.load(f)
            return final_data
        except Exception as e:
            print(f"   ⚠️  Error cargando final_results.json: {e}")
            return None
    else:
        print(f"   ❌ No se encontró final_results.json en {study_path}")
        return None

# def load_fold_raw_data(study_path,fold_num):
#     """Carga el archivo final_results.json que contiene predicciones y valores reales"""
#     fold_path = os.path.join(study_path, f"/training_results/fold_{fold_num}/results.json")
#     print(fold_path)
#     # print(study_path)
#     # print(f"/training_results/fold_{int(fold_num)}/results.json")
#     # print(os.path.join(study_path, f"/training_results/fold_{fold_num}/results.json"))

#     if os.path.exists(fold_path):
#         try:
#             with open(fold_path, 'r') as f:
#                 fold_data = json.load(f)
#             return fold_data
#         except Exception as e:
#             print(f"   ⚠️  Error cargando fold results: {e}")
#             return None
#     else:
#         print(f"   ❌ No se encontró fold results en {study_path}")
#         return None

def extract_predictions_and_true_values(final_results):
    """Extrae predicciones y valores reales del archivo final_results.json"""
    
    if not final_results:
        return None, None
    
    predictions = {}
    true_values = {}
    
    try:
        # Verificar estructura de datos
        if 'predictions_per_fold' in final_results and 'true_values_per_fold' in final_results:
            predictions_per_fold = final_results['predictions_per_fold']
            true_values_per_fold = final_results['true_values_per_fold']
            
            # Extraer datos por fold
            for fold_idx, (pred_fold, true_fold) in enumerate(zip(predictions_per_fold, true_values_per_fold)):
                fold_key = fold_idx + 1  # Los folds empiezan en 1
                
                predictions[fold_key] = {
                    'position': np.array(pred_fold.get('position', [])),
                    'velocity': np.array(pred_fold.get('velocity', []))
                }
                
                true_values[fold_key] = {
                    'position': np.array(true_fold.get('position', [])),
                    'velocity': np.array(true_fold.get('velocity', []))
                }
        
        return predictions, true_values
        
    except Exception as e:
        print(f"   ❌ Error extrayendo predicciones: {e}")
        return None, None

def get_fold_closest_to_median(fold_results, target='position'):
    """Encuentra el fold más cercano a la mediana de R² scores"""
    
    if target == 'position':
        # Usar additional_test_r2_scores[0] para posición
        r2_scores = [fold_data.get('additional_test_r2_scores', [0, 0])[0] 
                    for fold_data in fold_results.values() 
                    if 'additional_test_r2_scores' in fold_data and 
                    len(fold_data.get('additional_test_r2_scores', [])) >= 2]
    elif target == 'velocity':
        # Usar additional_test_r2_scores[1] para velocidad
        r2_scores = [fold_data.get('additional_test_r2_scores', [0, 0])[1] 
                    for fold_data in fold_results.values() 
                    if 'additional_test_r2_scores' in fold_data and 
                    len(fold_data.get('additional_test_r2_scores', [])) >= 2]
    else:
        # Usar test_r2_score combinado como fallback
        r2_scores = [fold_data.get('test_r2_score', 0) 
                    for fold_data in fold_results.values()]
    
    if not r2_scores:
        print(f"   ⚠️  No R² scores found for {target}")
        return 1  # Devolver fold 1 como fallback
    
    median_r2 = np.median(r2_scores)
    fold_keys = list(fold_results.keys())
    
    # Encontrar el fold más cercano a la mediana
    closest_fold_idx = np.argmin(np.abs(np.array(r2_scores) - median_r2))
    closest_fold = fold_keys[closest_fold_idx]
    closest_r2 = r2_scores[closest_fold_idx]
    
    # Crear mapeo explícito para debugging
    fold_score_mapping = {fold_keys[i]: r2_scores[i] for i in range(len(fold_keys))}
    
    # Imprimir información de la mediana y el fold seleccionado
    print(f"      📊 Fold-R² mapping ({target}): {[(f'Fold {k}', f'{v:.3f}') for k, v in sorted(fold_score_mapping.items())]}")
    print(f"      📊 R² scores in order: {[f'{r:.3f}' for r in r2_scores]}")
    print(f"      📊 Fold keys in order: {fold_keys}")
    print(f"      📈 Median R²: {median_r2:.4f}")
    print(f"      🎯 Selected fold {closest_fold} with R²: {closest_r2:.4f} (diff: {abs(closest_r2 - median_r2):.4f})")
    
    # Verificación adicional
    actual_r2_from_fold = fold_score_mapping[closest_fold]
    if abs(actual_r2_from_fold - closest_r2) > 0.0001:
        print(f"      ⚠️  WARNING: R² mismatch! Expected {closest_r2:.4f}, got {actual_r2_from_fold:.4f}")
    
    return closest_fold

print("🔍 Searching for study directories...")
print("📋 Datasets to process:")
for dataset_type, datasets_dict in dataset_mapping.items():
    print(f"   {dataset_type}: {list(datasets_dict.keys())}")

study_dirs = find_study_directories(results_base_dir, dataset_mapping)

print(f"📁 Studies found:")
total_studies = 0
for dataset_type, datasets in study_dirs.items():
    print(f"   {dataset_type}: {len(datasets)} datasets")
    total_studies += len(datasets)
print(f"   Total: {total_studies} studies")

# %%
# Celda 2: Funciones de visualización inspiradas en ibioml.plots

def plot_predictions_comparison(predictions, true_values, fold_to_plot, testTrials=None, target='position', 
                              dataset_name='Dataset', save_path=None, time_limit=1000, 
                              bin_duration_ms=200, figsize=(12, 6), fold_r2=None,
                              dataset_key=None):
    """
    Plotea predicciones vs datos reales para un target específico
    Inspirado en plot_predictions de ibioml.plots
    
    Args:
        time_limit: Número de time bins a mostrar (default: 1000 = ~3.3 segundos con bins de 200ms)
        bin_duration_ms: Duración de cada time bin en milisegundos (default: 200ms)
    """
    
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=figsize, dpi=200)
    
    print(fold_to_plot)
    # Extraer datos del fold específico
    if fold_to_plot not in predictions or fold_to_plot not in true_values:
        print(f"   ❌ Fold {fold_to_plot} not found in data")
        return None, None
    
    y_true = true_values[fold_to_plot][target]
    y_pred = predictions[fold_to_plot][target]
    
    
    # Verificar que tenemos datos
    if len(y_true) == 0 or len(y_pred) == 0:
        print(f"   ❌ No data for {target} in fold {fold_to_plot}")
        return None, None
    
    # Aplicar límite temporal (número de bins)
    if time_limit is not None and time_limit > 0:
        end_idx = min(time_limit, len(y_true), len(y_pred))
        y_true = y_true[:end_idx]
        y_pred = y_pred[:end_idx]
    
    # Convertir time bins a tiempo real en segundos
    time_bins = np.arange(len(y_true))
    time_seconds = time_bins * (bin_duration_ms / 1000.0)  # Convertir ms a segundos
    
    # Convertir posición de mm a cm (solo para target='position')
    if target == 'position':
        y_true = y_true / 10.0  # Convertir mm a cm
        y_pred = y_pred / 10.0  # Convertir mm a cm
    
    # Colores específicos por target
    colors = {
        'position': '#63adf2',  # Azul claro
        'velocity': '#ad84ce'   # Violeta
    }
    color = colors.get(target, '#1f77b4')
    
    # Añadir parches de contexto si encontramos el pickle correspondiente
    try:
        if dataset_key is not None:
            # with open(dataset_key, 'rb') as f:
            #     loaded = pickle.load(f)
            
            mat_contents = io.loadmat(dataset_key)
            #if isinstance(loaded, tuple) and len(loaded) >= 3:
            if 'rewCtxt' in mat_contents:

                #ctxt_per_time= mat_contents['rewCtxt'].copy()
                #print(f'Acá!! {testTrials}')
                rewCtxtOrNot=np.asarray(mat_contents['rewCtxt'][mat_contents['trialFinalBin'][testTrials]-1])
                #a= mat_contents['trialFinalBin'][testTrials]-1
                #print(f'Acá 2!! {rewCtxtOrNot}')
                trialDurations=mat_contents['trialDurationInBins'].copy()
                trialLimits=np.append(0,np.cumsum(trialDurations[testTrials])-1).astype(int)
                indExceed=np.where(trialLimits>len(time_seconds))[0]
                #print(f"      ⏱️  Trial limits (in bins): {indExceed}")
                if len(indExceed)>0:
                    testTrialsCut=testTrials[:indExceed[0]-1].copy()
                    patchLimits=np.append(0,np.cumsum(trialDurations[testTrialsCut])-1).astype(int)
                    patchLimits=np.append(patchLimits,len(time_seconds)-1) # asegurar que el último límite sea el final del tiempo mostrado
                    testTrialsCut=testTrials[:indExceed[0]].copy()


                else:
                    testTrialsCut=testTrials.copy()
                    patchLimits=np.append(0,np.cumsum(trialDurations[testTrialsCut])-1).astype(int)

                #print(f'limites {time_seconds[patchLimits]}')
                #print(rewCtxtOrNot)
                patches = []
                #for ut in testTrials:
                #for i in range(0,len(testTrials)):
                for ind,testtrial in enumerate(testTrialsCut):
                    # inds = np.where(trial_markers == ut)[0]
                    # if inds.size == 0:
                    #     continue
                    # start_idx = inds[0]
                    # end_idx = inds[-1]
                    # # limitar al rango mostrado
                    # if start_idx >= len(time_seconds):
                    #     continue
                    # end_idx = min(end_idx, len(time_seconds)-1)
                    # is_rew = int(np.round(np.mean(rewCtxtOrNot[start_idx:end_idx+1]))) == 1
                    is_rew= rewCtxtOrNot[ind]==1
                    start_sec = time_seconds[patchLimits[ind]]
                    #endSample=min(trialLimits[ind + 1] -1, len(time_seconds)-1)
                    endSample=patchLimits[ind + 1]
                    #endSample=patchLimits[ind + 1] if (ind + 1) < len(patchLimits) else len(time_seconds)-1
                    end_sec = time_seconds[endSample]
                    #end_sec = min(time_seconds[trialLimits[ind + 1] -1],len(time_seconds)-1)
                    

                    # start_sec = time_seconds[start_idx]
                    # end_sec = time_seconds[end_idx]
                    color_green='#8EEB23'
                    color_gray='#747272'
                    #color_patches={True: '#8EEB23', False: '#747272'}
                    color_patch =color_green if is_rew else color_gray
                    #color_patch = color_patches[is_rew]
                    alpha_patch = 0.4
                    ax.axvspan(start_sec, end_sec, facecolor=color_patch, alpha=alpha_patch, zorder=0)
                    patches.append((is_rew, color_patch))
                
                # añadir leyenda para patches si se detectó al menos uno de cada tipo
                if patches:
                    has_rew = any(p for p, _ in patches)
                    has_non = any(not p for p, _ in patches)
                    legend_handles = []
                    if has_rew:
                        #legend_handles.append(Patch(facecolor=color_patches[True], alpha=0.9, label='rewCtxt'))
                        legend_handles.append(Patch(facecolor=color_green, alpha=alpha_patch, label='rewCtxt'))
                    if has_non:
                        legend_handles.append(Patch(facecolor=color_gray, alpha=alpha_patch, label='no rewCtxt'))
                    if legend_handles:
                        ax.legend(handles=ax.get_legend_handles_labels()[0] + legend_handles, loc='upper right', fontsize=11)
    except Exception as e:
        # en caso de cualquier fallo, no bloquear la generación del plot
        print(f"   ⚠️  Warning: Could not add context patches due to error: {e}")
        pass    
        

    

    # Plot estilo ibioml: datos reales en negro, predicciones en color
    ax.plot(time_seconds, y_true, label='Real Data', color='black', linewidth=2, alpha=0.8)
    ax.plot(time_seconds, y_pred, label='Predictions', color=color, linewidth=2, alpha=0.9)
    
    # Configuración de ejes y etiquetas
    ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    
    if target == 'position':
        ax.set_ylabel('Position (cm)', fontsize=12, fontweight='bold')
        title = f'Position Decoding: {dataset_name}\n(Fold {fold_to_plot} - Representative)'
    elif target == 'velocity':
        ax.set_ylabel('Speed (cm/s)', fontsize=12, fontweight='bold')
        title = f'Speed Decoding: {dataset_name}\n(Fold {fold_to_plot} - Representative)'
    else:
        ax.set_ylabel('Value', fontsize=12, fontweight='bold')
        title = f'Decoding - {dataset_name}\n(Fold {fold_to_plot})'
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Leyenda y grid
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Estilo limpio (sin bordes superiores/derechos)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Mostrar R² score del fold completo (si se proporciona) o calcular para datos mostrados
    if fold_r2 is not None:
        # Usar R² del fold completo
        r2_to_show = fold_r2
    elif len(y_true) == len(y_pred) and len(y_true) > 0:
        # Calcular R² para datos mostrados como fallback
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2_to_show = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    else:
        r2_to_show = 0
    
    # Añadir texto con R² score
    ax.text(0.02, 0.98, f'R² = {r2_to_show:.4f}', transform=ax.transAxes, 
            fontsize=12, fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Guardar en formato SVG si se especifica
    if save_path:
        # Cambiar extensión a .svg
        if save_path.endswith('.png'):
            save_path = save_path.replace('.png', '.svg')
        elif not save_path.endswith('.svg'):
            save_path += '.svg'
            
        plt.savefig(save_path, format='svg', bbox_inches='tight')
        print(f"📁 Graph saved to: {save_path}")
    
    plt.show()
    
    return fig, ax

def create_subplot_predictions(predictions, true_values, fold_to_plot, dataset_name='Dataset',
                             save_path=None, time_limit=1000, bin_duration_ms=200, figsize=(16, 8),
                             fold_r2_position=None, fold_r2_velocity=None):
    """Crea subplot con posición y velocidad lado a lado"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=200)
    
    # Verificar que tenemos datos
    if fold_to_plot not in predictions or fold_to_plot not in true_values:
        print(f"   ❌ Fold {fold_to_plot} not found")
        return None, None
    
    # Posición (izquierda)
    y_true_pos = true_values[fold_to_plot]['position']
    y_pred_pos = predictions[fold_to_plot]['position']
    
    if len(y_true_pos) > 0 and len(y_pred_pos) > 0:
        # Aplicar límite temporal
        if time_limit is not None and time_limit > 0:
            end_idx = min(time_limit, len(y_true_pos), len(y_pred_pos))
            y_true_pos = y_true_pos[:end_idx]
            y_pred_pos = y_pred_pos[:end_idx]
        
        # Convertir a tiempo real
        time_bins_pos = np.arange(len(y_true_pos))
        time_seconds_pos = time_bins_pos * (bin_duration_ms / 1000.0)
        
        # Convertir posición de mm a cm
        y_true_pos = y_true_pos / 10.0  # Convertir mm a cm
        y_pred_pos = y_pred_pos / 10.0  # Convertir mm a cm
        
        ax1.plot(time_seconds_pos, y_true_pos, label='Real Data', color='black', linewidth=2)
        ax1.plot(time_seconds_pos, y_pred_pos, label='Predictions', color='#63adf2', linewidth=2)
        ax1.set_title('Position', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time (seconds)', fontsize=12)
        ax1.set_ylabel('Position (cm)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # R² para posición (usar del fold completo si está disponible)
        if fold_r2_position is not None:
            r2_pos = fold_r2_position
        elif len(y_true_pos) == len(y_pred_pos):
            ss_res = np.sum((y_true_pos - y_pred_pos) ** 2)
            ss_tot = np.sum((y_true_pos - np.mean(y_true_pos)) ** 2)
            r2_pos = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        else:
            r2_pos = 0
            
        ax1.text(0.02, 0.98, f'R² = {r2_pos:.4f}', transform=ax1.transAxes, 
                fontsize=11, fontweight='bold', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Velocidad (derecha)
    y_true_vel = true_values[fold_to_plot]['velocity']
    y_pred_vel = predictions[fold_to_plot]['velocity']
    
    if len(y_true_vel) > 0 and len(y_pred_vel) > 0:
        # Aplicar límite temporal
        if time_limit is not None and time_limit > 0:
            end_idx = min(time_limit, len(y_true_vel), len(y_pred_vel))
            y_true_vel = y_true_vel[:end_idx]
            y_pred_vel = y_pred_vel[:end_idx]
        
        # Convertir a tiempo real
        time_bins_vel = np.arange(len(y_true_vel))
        time_seconds_vel = time_bins_vel * (bin_duration_ms / 1000.0)
        
        ax2.plot(time_seconds_vel, y_true_vel, label='Real Data', color='black', linewidth=2)
        ax2.plot(time_seconds_vel, y_pred_vel, label='Predictions', color='#ad84ce', linewidth=2)
        ax2.set_title('Speed', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time (seconds)', fontsize=12)
        ax2.set_ylabel('Speed (cm/s)', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # R² para velocidad (usar del fold completo si está disponible)
        if fold_r2_velocity is not None:
            r2_vel = fold_r2_velocity
        elif len(y_true_vel) == len(y_pred_vel):
            ss_res = np.sum((y_true_vel - y_pred_vel) ** 2)
            ss_tot = np.sum((y_true_vel - np.mean(y_true_vel)) ** 2)
            r2_vel = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        else:
            r2_vel = 0
            
        ax2.text(0.02, 0.98, f'R² = {r2_vel:.4f}', transform=ax2.transAxes, 
                fontsize=11, fontweight='bold', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Estilo limpio para ambos axes
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.suptitle(f'Predictions vs Real Data - {dataset_name}\n(Fold {fold_to_plot} - Representative)', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        # Cambiar extensión a .svg
        if save_path.endswith('.png'):
            save_path = save_path.replace('.png', '.svg')
        elif not save_path.endswith('.svg'):
            save_path += '.svg'
            
        plt.savefig(save_path, format='svg', bbox_inches='tight')
        print(f"📁 Combined graph saved to: {save_path}")
    
    plt.show()
    
    return fig, (ax1, ax2)

print("🎨 Funciones de visualización cargadas exitosamente")

# %%
# Celda 3: Generación de gráficos de predicciones para POSICIÓN

print("=" * 100)
print("GENERACIÓN DE GRÁFICOS DE PREDICCIONES - POSICIÓN")
print("=" * 100)

# Procesar cada dataset para posición
position_results = {}

for dataset_type, datasets in study_dirs.items():
    for dataset_name, study_path in datasets.items():
        descriptionDict = dataset_mapping[dataset_type].get(dataset_name, dataset_name)
        description = descriptionDict['label']
        print(f"\n📍 Procesando POSICIÓN para: {description}")
        print(f"   Ruta: {study_path}")
        
        try:
            # Cargar resultados de folds para encontrar fold cercano a mediana
            fold_results = load_fold_results(study_path)
            
            if not fold_results:
                print(f"   ❌ No se encontraron resultados de folds")
                continue
            
            # Encontrar fold más cercano a mediana para posición
            median_fold = get_fold_closest_to_median(fold_results, target='position')
            print(f"   🎯 Fold más cercano a mediana (posición): {median_fold}")
            
            # Extraer R² del fold para posición
            fold_r2_position = None
            if median_fold in fold_results:
                additional_scores = fold_results[median_fold].get('additional_test_r2_scores', [])
                if len(additional_scores) >= 1:
                    fold_r2_position = additional_scores[0]  # Posición es índice 0
            
            # Cargar predicciones y valores reales
            final_results = load_final_results(study_path)
            if not final_results:
                print(f"   ❌ No se pudo cargar final_results.json")
                continue

            #fold_raw_data = load_fold_raw_data(study_path, median_fold)

            fold_raw_data =fold_results[median_fold]
            
            if not fold_raw_data:
                print(f"   ❌ No se pudo cargar fold results para fold {median_fold}")
                continue
            
            #a=dataset_mapping[dataset_type][dataset_name]["mat_File"]
            #print(f"Type of var1: {type(a)}, value: {a}")
            #print(datasets)
            mat_file = os.path.join(base_dir, dataset_mapping[dataset_type][dataset_name]["mat_File"])
            #print(mat_file)
            testTrialsInFold=fold_raw_data["test_trials"]    
                
            predictions, true_values = extract_predictions_and_true_values(final_results)
            if not predictions or not true_values:
                print(f"   ❌ No se pudieron extraer predicciones")
                continue
            
            # Verificar que tenemos datos para posición
            if (median_fold in predictions and median_fold in true_values and
                len(predictions[median_fold]['position']) > 0 and 
                len(true_values[median_fold]['position']) > 0):
                
                print(f"   📊 Position data found: {len(predictions[median_fold]['position'])} points")
                
                # Crear gráfico de posición
                safe_name = dataset_name.replace('/', '_').replace(' ', '_')
                save_path = f'{plot_path}/position_{safe_name}.svg'
                # Determinar duración del bin según el dataset
                #bin_duration = 100 if '100ms' in dataset_name else 200  # ms
                
                fig, ax = plot_predictions_comparison(
                    predictions, true_values, median_fold,testTrials=testTrialsInFold,
                    target='position',
                    dataset_name=description,
                    #dataset_key=dataset_name,
                    dataset_key=mat_file,
                    save_path=save_path,
                    time_limit=timeBinsPlot,
                    bin_duration_ms=bin_duration,
                    figsize=(14, 6),
                    fold_r2=fold_r2_position  # Pasar R² del fold completo
                )
                
                if fig is not None:
                    position_results[description] = {
                        'success': True,
                        'fold': median_fold,
                        'n_points': len(predictions[median_fold]['position']),
                        'save_path': save_path
                    }
                    print(f"   ✅ Position graph generated successfully")
                else:
                    position_results[description] = {'success': False, 'error': 'Plot generation failed'}
            else:
                print(f"   ⚠️  No valid position data for fold {median_fold}")
                position_results[description] = {'success': False, 'error': 'No valid position data'}
                
        except Exception as e:
            print(f"   ❌ Error procesando {description}: {e}")
            position_results[description] = {'success': False, 'error': str(e)}

# Resumen de resultados para posición
print(f"\n📊 SUMMARY - POSITION GRAPHS:")
successful_pos = sum(1 for r in position_results.values() if r.get('success', False))
total_pos = len(position_results)
print(f"   ✅ Successful: {successful_pos}/{total_pos}")
print(f"   📁 Files generated:")
for desc, result in position_results.items():
    if result.get('success', False):
        print(f"      • {desc}: {result['save_path']}")

# %%
# Celda 4: Generación de gráficos de predicciones para VELOCIDAD

print("=" * 100)
print("PREDICTION GRAPHS GENERATION - SPEED")
print("=" * 100)

# Procesar cada dataset para velocidad
velocity_results = {}

for dataset_type, datasets in study_dirs.items():
    for dataset_name, study_path in datasets.items():
        # Same shape as in the position block above (dataset_mapping values
        # are `{"label": ..., "mat_File": ...}` dicts, not plain strings).
        descriptionDict = dataset_mapping[dataset_type].get(dataset_name, dataset_name)
        description = descriptionDict['label']
        mat_file = os.path.join(base_dir, descriptionDict['mat_File'])

        print(f"\n🚀 Procesando SPEED para: {description}")
        print(f"   Ruta: {study_path}")
        
        try:
            # Cargar resultados de folds para encontrar fold cercano a mediana
            fold_results = load_fold_results(study_path)
            
            if not fold_results:
                print(f"   ❌ No se encontraron resultados de folds")
                continue
            
            # Encontrar fold más cercano a mediana para velocidad
            median_fold = get_fold_closest_to_median(fold_results, target='velocity')
            print(f"   🎯 Fold más cercano a mediana (speed): {median_fold}")
            
            # Extraer R² del fold para velocidad
            fold_r2_velocity = None
            if median_fold in fold_results:
                additional_scores = fold_results[median_fold].get('additional_test_r2_scores', [])
                if len(additional_scores) >= 2:
                    fold_r2_velocity = additional_scores[1]  # Velocidad es índice 1
            
            # Cargar predicciones y valores reales
            final_results = load_final_results(study_path)
            if not final_results:
                print(f"   ❌ No se pudo cargar final_results.json")
                continue
                
            predictions, true_values = extract_predictions_and_true_values(final_results)
            if not predictions or not true_values:
                print(f"   ❌ No se pudieron extraer predicciones")
                continue
            
            # Verificar que tenemos datos para velocidad
            if (median_fold in predictions and median_fold in true_values and
                len(predictions[median_fold]['velocity']) > 0 and 
                len(true_values[median_fold]['velocity']) > 0):
                
                print(f"   📊 Datos de speed encontrados: {len(predictions[median_fold]['velocity'])} puntos")
                
                # Crear gráfico de speed
                safe_name = dataset_name.replace('/', '_').replace(' ', '_')
                save_path = f'{plot_path}/speed_{safe_name}.svg'
                
                # Determinar duración del bin según el dataset
                #bin_duration = 100 if '100ms' in dataset_name else 200  # ms
                
                fig, ax = plot_predictions_comparison(
                    predictions, true_values, median_fold,
                    testTrials=fold_results[median_fold]["test_trials"],
                    target='velocity',
                    dataset_name=description,
                    dataset_key=mat_file,
                    save_path=save_path,
                    time_limit=timeBinsPlot,
                    bin_duration_ms=bin_duration,
                    figsize=(14, 6),
                    fold_r2=fold_r2_velocity  # Pasar R² del fold completo
                )
                
                if fig is not None:
                    velocity_results[description] = {
                        'success': True,
                        'fold': median_fold,
                        'n_points': len(predictions[median_fold]['velocity']),
                        'save_path': save_path
                    }
                    print(f"   ✅ Gráfico de speed generado exitosamente")
                else:
                    velocity_results[description] = {'success': False, 'error': 'Plot generation failed'}
            else:
                print(f"   ⚠️  No hay datos de speed válidos para fold {median_fold}")
                velocity_results[description] = {'success': False, 'error': 'No valid velocity data'}
                
        except Exception as e:
            print(f"   ❌ Error procesando {description}: {e}")
            velocity_results[description] = {'success': False, 'error': str(e)}

# Resumen de resultados para velocidad
print(f"\n📊 RESUMEN - GRÁFICOS DE SPEED:")
successful_vel = sum(1 for r in velocity_results.values() if r.get('success', False))
total_vel = len(velocity_results)
print(f"   ✅ Exitosos: {successful_vel}/{total_vel}")
print(f"   📁 Archivos generados:")
for desc, result in velocity_results.items():
    if result.get('success', False):
        print(f"      • {desc}: {result['save_path']}")

    # %%
# # Celda 5: Gráficos separados de posición y velocidad para los datasets seleccionados

# print("=" * 100)
# print("GENERACIÓN DE GRÁFICOS SEPARADOS - POSICIÓN Y SPEED")
# print("=" * 100)

# # Procesar cada dataset para generar gráficos separados de posición y speed
# separate_results = {}

# for dataset_type, datasets in study_dirs.items():
#     for dataset_name, study_path in datasets.items():
#         description = dataset_mapping[dataset_type].get(dataset_name, dataset_name)
        
#         print(f"\n📊 Procesando GRÁFICOS SEPARADOS para: {description}")
#         print(f"   Ruta: {study_path}")
        
#         try:
#             # Cargar resultados de folds
#             fold_results = load_fold_results(study_path)
            
#             if not fold_results:
#                 print(f"   ❌ No se encontraron resultados de folds")
#                 continue
            
#             # Encontrar fold más cercano a la mediana GENERAL (combinada)
#             median_fold = get_fold_closest_to_median(fold_results, target='combined')
#             print(f"   🎯 Fold seleccionado (mediana general): {median_fold}")
            
#             # Extraer R² del fold para ambos targets usando el MISMO fold
#             fold_r2_position = None
#             fold_r2_velocity = None
            
#             if median_fold in fold_results:
#                 additional_scores = fold_results[median_fold].get('additional_test_r2_scores', [])
#                 if len(additional_scores) >= 2:
#                     fold_r2_position = additional_scores[0]  # Posición es índice 0
#                     fold_r2_velocity = additional_scores[1]  # Velocidad es índice 1
            
#             # Cargar predicciones y valores reales
#             final_results = load_final_results(study_path)
#             if not final_results:
#                 print(f"   ❌ No se pudo cargar final_results.json")
#                 continue
                
#             predictions, true_values = extract_predictions_and_true_values(final_results)
#             if not predictions or not true_values:
#                 print(f"   ❌ No se pudieron extraer predicciones")
#                 continue
            
#             separate_results[description] = {'position': None, 'velocity': None}
            
#             # === GENERAR GRÁFICO DE POSICIÓN ===
#             has_position = (median_fold in predictions and median_fold in true_values and
#                            len(predictions[median_fold]['position']) > 0 and 
#                            len(true_values[median_fold]['position']) > 0)
            
#             if has_position:
#                 print(f"   📍 Generando gráfico de POSICIÓN (fold {median_fold})")
#                 print(f"      • Datos: {len(predictions[median_fold]['position'])} puntos")
                
#                 safe_name = dataset_name.replace('/', '_').replace(' ', '_')
#                 save_path_pos = f'results/mlp/plots/predictions/position_{safe_name}.svg'
                
                
#                 # Determinar duración del bin según el dataset
#                 #bin_duration = 100 if '100ms' in dataset_name else 200  # ms
                
#                 fig_pos, ax_pos = plot_predictions_comparison(
#                     predictions, true_values, median_fold,
#                     target='position',
#                     dataset_name=description,
#                     save_path=save_path_pos,
#                     time_limit=timeBinsPlot, 
#                     bin_duration_ms=bin_duration,
#                     figsize=(14, 6),
#                     fold_r2=fold_r2_position  # Pasar R² del fold completo
#                 )
                
#                 if fig_pos is not None:
#                     separate_results[description]['position'] = {
#                         'success': True,
#                         'fold': median_fold,
#                         'n_points': len(predictions[median_fold]['position']),
#                         'save_path': save_path_pos,
#                         'r2': fold_r2_position
#                     }
#                     print(f"      ✅ Gráfico de posición generado: {save_path_pos}")
#                 else:
#                     separate_results[description]['position'] = {'success': False, 'error': 'Plot generation failed'}
#             else:
#                 print(f"   ⚠️  No hay datos válidos de posición para fold {median_fold}")
#                 separate_results[description]['position'] = {'success': False, 'error': 'No valid position data'}
            
#             # === GENERAR GRÁFICO DE SPEED ===
#             has_velocity = (median_fold in predictions and median_fold in true_values and
#                            len(predictions[median_fold]['velocity']) > 0 and 
#                            len(true_values[median_fold]['velocity']) > 0)
            
#             if has_velocity:
#                 print(f"   🚀 Generando gráfico de SPEED (fold {median_fold})")
#                 print(f"      • Datos: {len(predictions[median_fold]['velocity'])} puntos")
                
#                 safe_name = dataset_name.replace('/', '_').replace(' ', '_')
#                 save_path_vel = f'results/mlp/plots/predictions/velocity_{safe_name}.svg'
                
                
                
                
#                 fig_vel, ax_vel = plot_predictions_comparison(
#                     predictions, true_values, median_fold,
#                     target='velocity',
#                     dataset_name=description,
#                     save_path=save_path_vel,
#                     time_limit=timeBinsPlot, 
#                     bin_duration_ms=bin_duration,
#                     figsize=(14, 6),
#                     fold_r2=fold_r2_velocity  # Pasar R² del fold completo
#                 )
                
#                 if fig_vel is not None:
#                     separate_results[description]['velocity'] = {
#                         'success': True,
#                         'fold': median_fold,
#                         'n_points': len(predictions[median_fold]['velocity']),
#                         'save_path': save_path_vel,
#                         'r2': fold_r2_velocity
#                     }
#                     print(f"      ✅ Gráfico de speed generado: {save_path_vel}")
#                 else:
#                     separate_results[description]['velocity'] = {'success': False, 'error': 'Plot generation failed'}
#             else:
#                 print(f"   ⚠️  No hay datos válidos de speed para fold {median_fold}")
#                 separate_results[description]['velocity'] = {'success': False, 'error': 'No valid speed data'}
                
#         except Exception as e:
#             print(f"   ❌ Error procesando {description}: {e}")
#             separate_results[description] = {'position': {'success': False, 'error': str(e)}, 
#                                            'velocity': {'success': False, 'error': str(e)}}

# # Resumen de resultados separados
# print(f"\n📊 RESUMEN - GRÁFICOS SEPARADOS:")
# successful_pos_sep = sum(1 for r in separate_results.values() if r.get('position', {}).get('success', False))
# successful_vel_sep = sum(1 for r in separate_results.values() if r.get('velocity', {}).get('success', False))
# total_datasets = len(separate_results)

# print(f"   📍 Posición: {successful_pos_sep}/{total_datasets} exitosos")
# print(f"   🚀 Speed: {successful_vel_sep}/{total_datasets} exitosos")
# print(f"   📁 Archivos generados:")

# for desc, result in separate_results.items():
#     print(f"      • {desc}:")
#     if result.get('position', {}).get('success', False):
#         pos_data = result['position']
#         print(f"         - Posición (fold {pos_data['fold']}): {pos_data['save_path']}")
#     if result.get('velocity', {}).get('success', False):
#         vel_data = result['velocity']
#         print(f"         - Speed (fold {vel_data['fold']}): {vel_data['save_path']}")

# # Resumen general
# print(f"\n🎯 RESUMEN GENERAL DE VISUALIZACIONES:")
# print(f"   📍 Posición (celdas 3 + 5): {successful_pos_sep} gráficos")
# print(f"   🚀 Speed (celdas 4 + 5): {successful_vel_sep} gráficos") 
# print(f"   📁 Total archivos generados: {successful_pos_sep + successful_vel_sep}")

# # %%