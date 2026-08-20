# %% [markdown]
# # Análisis de Resultados de Experimentos MLP
# 
# Este script analiza los resultados de los experimentos MLP para todos los datasets:
# - L5 experto (200ms, 200ms historia corta, 100ms)
# - L3 experto
# - N31 experto
# - L7 naive
# - N53 naive
# - DG S19 hipocampo
# - CA3 S18 hipocampo
#
# ## Esquema de Colores y Orden Consistente:
# - **1. Naive**: #004E89 (Azul oscuro)
# - **2. Experto**: #2A6041 (Verde oscuro) 
# - **3. Hipocampo**: #9B2915 (Rojo oscuro)
# 
# Todos los gráficos siguen este orden para facilitar la comparación visual.

# %%
# Celda 1: Inspección y carga de resultados

import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
# Use a non-interactive backend so `plt.show()` doesn't pop up a window for
# every figure when this script is run from the CLI. Files still get saved.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 100)
print("ANÁLISIS DE RESULTADOS DE EXPERIMENTOS MLP")
print("=" * 100)

# Directorio base de resultados.
# Nuestros study dirs viven en `results/mlp_<dataset>_bins200ms_dual/` (no hay
# subdirs por region/dataset como en la pipeline original de Sebas).
results_base_dir = 'results'

plot_path = 'results/plots/dual/performance'
if not os.path.exists(plot_path):
    os.makedirs(plot_path)
    print(f"📁 Directorio '{plot_path}' creado")


# Mapeo automático de study dirs → display names y tipos.
#
# Nuestro naming convention (from scripts/run_all_experiments.py):
#     mlp_<REGION>_<ID>_<expert|naive>_bins200ms_dual
#   e.g.
#     mlp_Pir_L5_expert_bins200ms_dual   → ("Pir-Expert-L5", "pir_experto")
#     mlp_CA3_S17_expert_bins200ms_dual  → ("CA3-Expert-S17", "CA3_experto")
#     mlp_DG_S38_naive_bins200ms_dual    → ("DG-Naive-S38",   "DG_naive")
#
# `type` uses the cohort-in-Spanish suffix (`_experto` / `_naive`) so it plugs
# straight into the existing COLOR_SCHEME and DATASET_TYPE_ORDER further down.

REGION_DISPLAY = {"Pir": "Pir", "CA3": "CA3", "DG": "DG"}
COHORT_DISPLAY = {"expert": "Expert", "naive": "Naive"}
COHORT_TYPE_SUFFIX = {"expert": "_experto", "naive": "_naive"}

def parse_study_dir_name(study_dir_name):
    """Extract (display_name, dataset_type) from a `mlp_<...>_dual` directory name.

    Returns (None, None) if the name does not match the convention.
    """
    if not study_dir_name.startswith("mlp_") or not study_dir_name.endswith("_dual"):
        return None, None
    core = study_dir_name[len("mlp_"):-len("_dual")]  # e.g. "Pir_L5_expert_bins200ms"
    if not core.endswith("_bins200ms"):
        return None, None
    core = core[: -len("_bins200ms")]                 # e.g. "Pir_L5_expert"
    parts = core.split("_")
    if len(parts) < 3:
        return None, None
    region = parts[0]
    cohort = parts[-1]
    dataset_id = "_".join(parts[1:-1])
    if region not in REGION_DISPLAY or cohort not in COHORT_DISPLAY:
        return None, None
    display_name = f"{REGION_DISPLAY[region]}-{COHORT_DISPLAY[cohort]}-{dataset_id}"
    dataset_type = f"{region}{COHORT_TYPE_SUFFIX[cohort]}"
    # Normalize `Pir_experto` → `pir_experto` for the existing COLOR_SCHEME keys.
    if dataset_type.startswith("Pir"):
        dataset_type = "pir" + dataset_type[len("Pir"):]
    return display_name, dataset_type


# Datasets to exclude from every plot (catastrophic outliers whose R² folds
# collapse the shared y-axis on the position boxplot and skew any per-type
# mean/std). Kept as a constant so it's obvious in code review.
EXCLUDED_DATASETS_FROM_PLOTS = {
    "DG-Naive-S38",   # position folds e.g. [-2.39, -1.59, -1.05] — median -1.06
    "DG-Naive-S41",   # position folds e.g. [-181.87, -20.41, -18.98] — median -18.98
    "Pir-Expert-N59", # excluded per Facundo (2026-07-08)
}

# Discover every dual study dir under results_base_dir and build the mapping
# expected by the downstream code (nested dict:
# {dataset_type: {study_dir_name: display_name}}).
dataset_mapping = {}
for _entry in sorted(os.listdir(results_base_dir)) if os.path.exists(results_base_dir) else []:
    if not os.path.isdir(os.path.join(results_base_dir, _entry)):
        continue
    _display, _type = parse_study_dir_name(_entry)
    if _display is None:
        continue
    if _display in EXCLUDED_DATASETS_FROM_PLOTS:
        print(f"   ⏭️  Excluded from plots: {_display}")
        continue
    dataset_mapping.setdefault(_type, {})[_entry] = _display

# dataset_colors = {}
# for dataset in dataset_order:
#         dataset_type = boxplot_df[boxplot_df['Dataset'] == dataset]['Type'].iloc[0]
#         color = type_colors.get(dataset_type, '#333333')
#         dataset_colors[dataset] = color


type_mapping = {'pir_experto': 'Piriform-Expert', 'pir_naive': 'Piriform-Naive', 'CA3_experto': 'CA3-Expert', 'CA3_naive': 'CA3-Naive', 'DG_experto': 'DG-Expert', 'DG_naive': 'DG-Naive'}
             

# Mapeo de nombres entre results_df y CSV
name_mapping = {}
for dataset_type, datasets_dict in dataset_mapping.items():
    print(f"   {dataset_type}: {list(datasets_dict.keys())}")
    for dataset_name, description in datasets_dict.items():
        
        name_mapping[dataset_type + '/' + dataset_name] = description


# # Mapeo de nombres entre results_df y CSV
# name_mapping = {
#         'pir_experto/pir experto L5': 'L5 experto',
#         'pir_experto/pir experto L3': 'L3 experto', 
#         'pir_experto/pir experto N31': 'N31 experto',
#         'pir_naive/pir naive L7': 'L7 naive',
#         'pir_naive/pir naive N53': 'N53 naive',
#         # 'DG-experto/DG_S19_200ms': 'DG S19 hipocampo',
#         # 'CA3-experto/CA3_S18_200ms': 'CA3 S18 hipocampo',
#         # 'DG-experto/DG_S17_200ms': 'DG S17 hipocampo',
#         # 'DG-experto/DG_S18_200ms': 'DG S18 hipocampo',
#         # 'CA3-experto/CA3_S17_200ms': 'CA3 S17 hipocampo',
#         ## 'DG-naive/DG_S24_200ms': 'DG S24 hipocampo',
#         # 'DG-naive/DG_S38_200ms': 'DG S38 hipocampo',
#         # 'DG-naive/DG_S39_200ms': 'DG S39 hipocampo',
#         # 'CA3-naive/CA3_S30_200ms': 'CA3 S30 hipocampo',
#         # 'CA3-naive/CA3_S31_200ms': 'CA3 S31 hipocampo'
#         'DG-experto/DG_S19_200ms': 'S19 DG experto',
#         'CA3-experto/CA3_S18_200ms': 'S18 CA3 experto',
#         'DG-experto/DG_S17_200ms': 'S17 DG experto',
#         'DG-experto/DG_S18_200ms': 'S18 DG experto',
#         'CA3-experto/CA3_S17_200ms': 'S17 CA3 experto',
#         #'DG-naive/DG_S24_200ms': 'S24 DG naive',
#         'DG-naive/DG_S38_200ms': 'S38 DG naive',
#         'DG-naive/DG_S39_200ms': 'S39 DG naive',
#         'CA3-naive/CA3_S30_200ms': 'S30 CA3 naive',
#         'CA3-naive/CA3_S31_200ms': 'S31 CA3 naive'
#     }

# Orden fijo específico para todos los boxplots
fixed_order = [
        # Piriform — naive first, then expert (kept order + added N59)
        'Pir-Naive-N53',
        'Pir-Naive-L7',
        'Pir-Expert-N31',
        'Pir-Expert-L3',
        'Pir-Expert-L5',
        'Pir-Expert-N59',
        # CA3
        'CA3-Naive-S30',
        'CA3-Naive-S31',
        'CA3-Expert-S17',
        'CA3-Expert-S18',
        # DG — naive first, then expert (added S38 and S41 to the naive block)
        'DG-Naive-S24',
        'DG-Naive-S38',
        'DG-Naive-S39',
        'DG-Naive-S41',
        'DG-Expert-S17',
        'DG-Expert-S18',
        'DG-Expert-S19',
    ]


# Esquema de colores consistente para todos los gráficos
COLOR_SCHEME = {
    'pir_naive': "#58AAE9",      # Azul oscuro
    'pir_experto':  '#004E89',      # Azul oscuro
    'DG_experto': "#9B4615",    # Rojo oscuro
    'CA3_experto': "#146D02",   # Verde oscuro
    'DG_naive': "#EBA447",       # Rojo claro
    'CA3_naive': "#69DA85",      # Verde claro
    
}
# Orden consistente para todos los gráficos
DATASET_TYPE_ORDER = ['pir_naive', 'pir_experto', 'CA3_naive','CA3_experto', 'DG_naive', 'DG_experto']

# Usar esquema de colores consistente
type_colors = {
        'Piriform-Expert': COLOR_SCHEME['pir_experto'],
        'Piriform-Naive': COLOR_SCHEME['pir_naive'],
        'CA3-Expert': COLOR_SCHEME['CA3_experto'],
        'CA3-Naive': COLOR_SCHEME['CA3_naive'],
        'DG-Expert': COLOR_SCHEME['DG_experto'],
        'DG-Naive': COLOR_SCHEME['DG_naive'],
    }



# Función auxiliar para obtener color por tipo de dataset
def get_color_by_type(dataset_type):
    """Retorna el color correspondiente al tipo de dataset"""
    #return COLOR_SCHEME.get(dataset_type.lower(), '#333333')  # Color por defecto si no se encuentra
    return COLOR_SCHEME.get(dataset_type, '#333333')  # Color por defecto si no se encuentra

# Función auxiliar para obtener paleta de colores ordenada
def get_color_palette(dataset_types):
    """Retorna una paleta de colores basada en los tipos de datasets"""
    return [get_color_by_type(dtype) for dtype in dataset_types]

def get_ordered_datasets_by_type(boxplot_df):
    """Ordena los datasets según el orden fijo especificado"""
    
    
    # Filtrar solo los datasets que existen en los datos
    available_datasets = boxplot_df['Dataset'].unique()
    ordered_datasets = [dataset for dataset in fixed_order if dataset in available_datasets]
    
    # Agregar cualquier dataset que no esté en el orden fijo (por si acaso)
    for dataset in available_datasets:
        if dataset not in ordered_datasets:
            ordered_datasets.append(dataset)
    
    return ordered_datasets

def get_ordered_types(boxplot_df):
    """Retorna los tipos de datasets en el orden consistente"""
    available_types = boxplot_df['Type'].str.lower().unique()
    ordered_types = []
    
    for dataset_type in DATASET_TYPE_ORDER:
        if dataset_type in available_types:
            ordered_types.append(dataset_type.capitalize())
    
    return ordered_types

def print_color_scheme():
    """Imprime el esquema de colores actual"""
    print("🎨 ESQUEMA DE COLORES Y ORDEN CONSISTENTE:")
    for i, dataset_type in enumerate(DATASET_TYPE_ORDER, 1):
        color = COLOR_SCHEME[dataset_type]
        print(f"   {i}. {dataset_type.capitalize()}: {color}")
    print()

def find_study_directories(base_path, dataset_mapping):
    """Encuentra los study dirs para cada dataset del mapping.

    Layout esperado por este runner:
        <base_path>/<study_dir_name>/training_results/fold_N/results.json

    (a diferencia del layout original de Sebas, que tenía subdirs anidados por
    region/dataset y luego un study_YYYY-MM-DD_HH-MM-SS/).
    """
    study_dirs = {}
    for dataset_type, datasets_dict in dataset_mapping.items():
        study_dirs[dataset_type] = {}
        for dataset_name in datasets_dict.keys():
            study_path = os.path.join(base_path, dataset_name)
            if os.path.isdir(os.path.join(study_path, "training_results")):
                study_dirs[dataset_type][dataset_name] = study_path
                print(f"   ✅ Encontrado: {dataset_type}/{dataset_name}")
            else:
                print(f"   ⚠️  Sin training_results: {dataset_type}/{dataset_name}")
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

def extract_experiment_summary(study_dirs):
    """Extrae un resumen de todos los experimentos"""
    summary_data = []
    
    for dataset_type, datasets in study_dirs.items():
        for dataset_name, study_path in datasets.items():
            print(f"📊 Analizando: {dataset_name}")
            
            try:
                # Cargar resultados de todos los folds
                fold_results = load_fold_results(study_path)
                
                if fold_results:
                    # Extraer métricas de cada fold
                    fold_test_r2 = []
                    fold_val_r2 = []
                    fold_position_r2 = []
                    fold_velocity_r2 = []
                    
                    for fold_num, fold_data in fold_results.items():
                        fold_test_r2.append(fold_data['test_r2_score'])
                        fold_val_r2.append(fold_data['best_validation_r2_score'])
                        
                        # R2 scores adicionales (posición y velocidad por separado)
                        if 'additional_test_r2_scores' in fold_data and len(fold_data['additional_test_r2_scores']) >= 2:
                            fold_position_r2.append(fold_data['additional_test_r2_scores'][0])
                            fold_velocity_r2.append(fold_data['additional_test_r2_scores'][1])
                    
                    # Calcular estadísticas agregadas
                    summary_data.append({
                        'dataset_type': dataset_type,
                        'dataset_name': dataset_name,
                        'description': dataset_mapping[dataset_type].get(dataset_name, dataset_name),
                        'n_folds': len(fold_results),
                        'mean_test_r2': np.mean(fold_test_r2),
                        'std_test_r2': np.std(fold_test_r2),
                        'mean_val_r2': np.mean(fold_val_r2),
                        'std_val_r2': np.std(fold_val_r2),
                        'mean_position_r2': np.mean(fold_position_r2) if fold_position_r2 else np.nan,
                        'std_position_r2': np.std(fold_position_r2) if fold_position_r2 else np.nan,
                        'mean_velocity_r2': np.mean(fold_velocity_r2) if fold_velocity_r2 else np.nan,
                        'std_velocity_r2': np.std(fold_velocity_r2) if fold_velocity_r2 else np.nan,
                        'best_fold_test_r2': max(fold_test_r2),
                        'worst_fold_test_r2': min(fold_test_r2),
                        'fold_test_r2_list': fold_test_r2,
                        'study_path': study_path
                    })
                else:
                    print(f"   ⚠️  No se encontraron resultados de folds")
                    
            except Exception as e:
                print(f"   ❌ Error procesando {dataset_name}: {e}")
    
    return summary_data

# Buscar y cargar todos los resultados
print("🔍 Buscando directorios de estudios...")
print_color_scheme()
print("📋 Datasets a buscar:")
for dataset_type, datasets_dict in dataset_mapping.items():
    print(f"   {dataset_type}: {list(datasets_dict.keys())}")
print()

study_dirs = find_study_directories(results_base_dir, dataset_mapping)

print(f"📁 Estudios encontrados:")
total_studies = 0
for dataset_type, datasets in study_dirs.items():
    print(f"   {dataset_type}: {len(datasets)} datasets")
    total_studies += len(datasets)
print(f"   Total: {total_studies} estudios")

print(f"\n🔄 Extrayendo resultados...")
summary_data = extract_experiment_summary(study_dirs)

# Convertir a DataFrame para análisis
results_df = pd.DataFrame(summary_data)

print(f"\n✅ Datos extraídos exitosamente:")
print(f"   📊 {len(results_df)} experimentos analizados")
print(f"   📈 Columnas disponibles: {list(results_df.columns)}")

# %%
# Celda 2: Tabla resumen de resultados

print("=" * 120)
print("TABLA RESUMEN DE RESULTADOS R² (TEST)")
print("=" * 120)

if not results_df.empty:
    # Crear tabla ordenada por performance
    display_df = results_df.copy()
    display_df = display_df.sort_values('mean_test_r2', ascending=False)
    
    print(f"{'Rank':<4} {'Dataset':<30} {'Tipo':<12} {'R² Test':<12} {'±STD':<8} {'R² Val':<12} {'Pos R²':<10} {'Vel R²':<10} {'Folds':<6}")
    print("-" * 120)
    
    for idx, (_, row) in enumerate(display_df.iterrows(), 1):
        rank = f"{idx}"
        dataset = f"{row['description']}"
        dtype = f"{row['dataset_type']}"
        test_r2 = f"{row['mean_test_r2']:.4f}"
        std_r2 = f"±{row['std_test_r2']:.3f}"
        val_r2 = f"{row['mean_val_r2']:.4f}"
        pos_r2 = f"{row['mean_position_r2']:.4f}" if not pd.isna(row['mean_position_r2']) else "N/A"
        vel_r2 = f"{row['mean_velocity_r2']:.4f}" if not pd.isna(row['mean_velocity_r2']) else "N/A"
        folds = f"{row['n_folds']}"
        
        print(f"{rank:<4} {dataset:<30} {dtype:<12} {test_r2:<12} {std_r2:<8} {val_r2:<12} {pos_r2:<10} {vel_r2:<10} {folds:<6}")
    
    print(f"\n📈 Estadísticas generales:")
    print(f"   • R² Test promedio: {results_df['mean_test_r2'].mean():.4f} ± {results_df['mean_test_r2'].std():.4f}")
    print(f"   • Mejor performance: {results_df['mean_test_r2'].max():.4f} ({display_df.iloc[0]['description']})")
    print(f"   • Peor performance: {results_df['mean_test_r2'].min():.4f} ({display_df.iloc[-1]['description']})")
    
    # Análisis por tipo de dataset
    print(f"\n📊 Performance por tipo de dataset:")
    for dataset_type in DATASET_TYPE_ORDER:  #['piriform-experto', 'piriform-naive', 'CA3-experto', 'CA3-naive', 'DG-experto', 'DG-naive']
        type_data = results_df[results_df['dataset_type'] == dataset_type]
        if not type_data.empty:
            mean_r2 = type_data['mean_test_r2'].mean()
            std_r2 = type_data['mean_test_r2'].std()
            count = len(type_data)
            print(f"   • {dataset_type.capitalize()}: {mean_r2:.4f} ± {std_r2:.4f} (n={count})")

else:
    print("❌ No se encontraron datos para analizar")

# %%
# Celda 2.1: Gráfico de Cantidad de Datos por Dataset

print("=" * 100)
print("VISUALIZACIÓN DE CANTIDAD DE DATOS POR DATASET")
print("=" * 100)

def create_data_quantity_plot(results_df, save_path=None, figsize=(12, 8)):
    """Crea gráfico de dispersión: Neuronas vs Trials por dataset.

    Post-preprocess counts, tomados directamente de nuestros artefactos:
      • n_neurons, time_bins ← data/processed/preprocess_report.csv
        (columnas `kept_neurons` y `kept_bins`).
      • n_trials             ← results/<study>/training_results/fold_1/results.json
        (train_trials + val_trials + test_trials — todos los folds suman lo mismo).
      • features             ← n_neurons × (bins_before + bins_current + bins_after)
    """
    if results_df.empty:
        print("❌ No hay datos disponibles para el gráfico")
        return None, None

    # Cargar preprocess report (kept_neurons, kept_bins por dataset).
    try:
        report_df = pd.read_csv('data/processed/preprocess_report.csv').set_index('dataset')
    except Exception as e:
        print(f"❌ Error cargando preprocess_report.csv: {e}")
        return None, None

    # Ventana temporal en bins (bins_before + bins_current + bins_after).
    BINS_WINDOW = 5

    # Region dir map: needed to locate the pickle fallback for datasets not in
    # the CSV. Mirrors REGION_TO_DIR in 04-decoding-predictions.py.
    _region_to_dir = {"pir_experto": "piriform", "pir_naive": "piriform",
                      "CA3_experto": "ca3", "CA3_naive": "ca3",
                      "DG_experto": "dg", "DG_naive": "dg"}

    plot_data = []
    for _, row in results_df.iterrows():
        study_dir_name = row['dataset_name']            # e.g. mlp_Pir_L5_expert_bins200ms_dual
        stem = study_dir_name[len("mlp_"):-len("_dual")]  # e.g. Pir_L5_expert_bins200ms

        # Post-preprocess bins / neurons — CSV first, pickle fallback for rows
        # skipped in the last preprocess_all.py run (e.g. Pir_L5 whose output
        # already existed on the GPU and was skipped from the report).
        if stem in report_df.index:
            n_neurons = int(report_df.loc[stem, 'kept_neurons'])
            time_bins = int(report_df.loc[stem, 'kept_bins'])
        else:
            region_dir = _region_to_dir.get(row['dataset_type'])
            pkl_path = Path(f"data/processed/{region_dir}/{stem}_flat.pickle") if region_dir else None
            if pkl_path is not None and pkl_path.exists():
                import pickle
                with pkl_path.open('rb') as f:
                    X, _y, _T = pickle.load(f)
                time_bins = int(X.shape[0])
                n_neurons = int(X.shape[1] // BINS_WINDOW)
                print(f"   ↩️  {stem}: not in CSV — read from pickle "
                      f"({n_neurons} neurons, {time_bins} bins)")
            else:
                print(f"   ⚠️  {stem}: not in CSV and no local pickle, skip")
                continue

        # Post-preprocess trials from any fold's results.json (all folds sum the same).
        fold_json_path = os.path.join(row['study_path'], 'training_results', 'fold_1', 'results.json')
        try:
            with open(fold_json_path) as f:
                fold = json.load(f)
            n_trials = (len(fold.get('train_trials', []))
                        + len(fold.get('val_trials', []))
                        + len(fold.get('test_trials', [])))
        except Exception as e:
            print(f"   ⚠️  {stem}: no pude leer fold_1/results.json ({e}), skip")
            continue

        features = n_neurons * BINS_WINDOW
        plot_data.append({
            'dataset_name': study_dir_name,
            'description': row['description'],
            'dataset_type': row['dataset_type'],
            'n_neurons': n_neurons,
            'n_trials': n_trials,
            'time_bins': time_bins,
            'features': features,
            'mean_test_r2': row['mean_test_r2'],
        })
        print(f"   ✅ {row['description']}: {n_neurons} neurons, {n_trials} trials, {time_bins:,} bins")

    if not plot_data:
        print("❌ No se pudieron extraer datos de dimensiones")
        return None, None
    
    # Crear DataFrame para el gráfico
    plot_df = pd.DataFrame(plot_data)
    
    # Configurar el gráfico
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=figsize, dpi=200)
    
    
    for dataset_type in DATASET_TYPE_ORDER:
        type_data = plot_df[plot_df['dataset_type'] == dataset_type]
        if not type_data.empty:
            color = get_color_by_type(dataset_type)
            english_label = type_mapping.get(dataset_type, dataset_type.capitalize())
            ax.scatter(
                type_data['n_trials'],
                type_data['n_neurons'],
                marker='X',
                s=200,  # Tamaño de las cruces (matches position_vs_velocity_scatter)
                color=color,
                alpha=0.8,
                label=english_label,
                edgecolors='black',
                linewidth=1.5,
            )
    
    # Personalización del gráfico
    ax.set_xlabel('Number of Trials', fontsize=18, fontweight='bold')
    ax.set_ylabel('Number of Neurons', fontsize=18, fontweight='bold')
    ax.set_title('Data Quantity per Dataset\n(Neurons vs Trials)', 
                fontsize=20, fontweight='bold', pad=20)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Leyenda
    ax.legend(title='Dataset Type', title_fontsize=16, fontsize=15, 
             loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # Anotaciones para cada punto
    for _, row in plot_df.iterrows():
        # Nombre corto del dataset
        short_name = row['description'].replace(' - ', '\n')
        
        # Posicionamiento especial para evitar solapamientos
        if 'DG S19' in row['description']:
            # DG S19 hacia la izquierda
            xytext = (-5, 5)
            ha = 'right'
        else:
            # Resto de datasets hacia la derecha
            xytext = (5, 5)
            ha = 'left'
        
        ax.annotate(
            short_name,
            (row['n_trials'], row['n_neurons']),
            xytext=xytext,  # Offset del texto
            textcoords='offset points',
            fontsize=13,
            ha=ha,
            va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
        )
    
    # Estilo
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Ajustar límites para mejor visualización
    x_margin = (plot_df['n_trials'].max() - plot_df['n_trials'].min()) * 0.1
    y_margin = (plot_df['n_neurons'].max() - plot_df['n_neurons'].min()) * 0.1
    
    ax.set_xlim(plot_df['n_trials'].min() - x_margin, plot_df['n_trials'].max() + x_margin)
    ax.set_ylim(plot_df['n_neurons'].min() - y_margin, plot_df['n_neurons'].max() + y_margin)
    
    plt.tight_layout()
    
    # Guardar si se especifica path
    if save_path:
        # Detectar formato basado en la extensión
        if save_path.endswith('.svg'):
            plt.savefig(save_path, format='svg', bbox_inches='tight')
        else:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📁 Gráfico guardado en: {save_path}")
    
    plt.show()
    
    # Imprimir estadísticas
    print(f"\n📊 ESTADÍSTICAS DE CANTIDAD DE DATOS:")
    print(f"{'Dataset':<25} {'Tipo':<10} {'Neuronas':<10} {'Ensayos':<10} {'Time bins':<12} {'R² Medio':<10}")
    print("-" * 90)
    
    for _, row in plot_df.sort_values('mean_test_r2', ascending=False).iterrows():
        dataset = f"{row['description']}"
        dtype = f"{row['dataset_type']}"
        neurons = f"{row['n_neurons']}"
        trials = f"{row['n_trials']}"
        time_bins = f"{row['time_bins']:,}"
        r2 = f"{row['mean_test_r2']:.3f}"
        
        print(f"{dataset:<25} {dtype:<10} {neurons:<10} {trials:<10} {time_bins:<12} {r2:<10}")
    
    # Estadísticas por tipo
    print(f"\n📈 PROMEDIOS POR TIPO DE DATASET:")
    for dataset_type in DATASET_TYPE_ORDER:
        type_data = plot_df[plot_df['dataset_type'] == dataset_type]
        if not type_data.empty:
            avg_neurons = type_data['n_neurons'].mean()
            avg_trials = type_data['n_trials'].mean()
            avg_time_bins = type_data['time_bins'].mean()
            avg_r2 = type_data['mean_test_r2'].mean()
            print(f"   • {dataset_type.capitalize()}: {avg_neurons:.0f} neuronas, {avg_trials:.0f} ensayos, {avg_time_bins:,.0f} time bins, R²={avg_r2:.3f}")
    
    # Información adicional sobre la relación features/neuronas
    print(f"\n🔍 ANÁLISIS FEATURES vs NEURONAS:")
    for _, row in plot_df.iterrows():
        ratio = row['features'] / row['n_neurons']
        print(f"   • {row['description']}: {row['features']} features ÷ {row['n_neurons']} neuronas = {ratio:.1f} (history window)")
    
    return fig, ax


def create_data_quantity_plot_pre(results_df, save_path=None, figsize=(12, 8)):
    """Same as `create_data_quantity_plot` but reads raw .mat files.

    Sources:
      • n_neurons, time_bins ← neuronActivity.shape from data/raw/<region>/<stem>.mat
      • n_trials             ← len(trialFinalBin)

    Same visual style so pre/post plots can be compared side-by-side.
    """
    from scipy.io import loadmat

    if results_df.empty:
        print("❌ No hay datos disponibles para el gráfico")
        return None, None

    _region_to_dir = {"pir_experto": "piriform", "pir_naive": "piriform",
                      "CA3_experto": "ca3", "CA3_naive": "ca3",
                      "DG_experto": "dg", "DG_naive": "dg"}

    plot_data = []
    for _, row in results_df.iterrows():
        study_dir_name = row['dataset_name']
        stem = study_dir_name[len("mlp_"):-len("_dual")]
        region_dir = _region_to_dir.get(row['dataset_type'])
        if region_dir is None:
            print(f"   ⚠️  {stem}: unknown region for type {row['dataset_type']}, skip")
            continue
        mat_path = Path(f"data/raw/{region_dir}/{stem}.mat")
        if not mat_path.exists():
            print(f"   ⚠️  {stem}: raw .mat not found ({mat_path}), skip")
            continue
        m = loadmat(mat_path)
        time_bins = int(m['neuronActivity'].shape[0])
        n_neurons = int(m['neuronActivity'].shape[1])
        n_trials = int(len(m['trialFinalBin'].ravel()))
        plot_data.append({
            'description': row['description'],
            'dataset_type': row['dataset_type'],
            'n_neurons': n_neurons,
            'n_trials': n_trials,
            'time_bins': time_bins,
        })
        print(f"   ✅ {row['description']} (pre): {n_neurons} neurons, {n_trials} trials, {time_bins:,} bins")

    if not plot_data:
        print("❌ No pre-preprocess data available")
        return None, None

    plot_df = pd.DataFrame(plot_data)

    plt.style.use('default')
    fig, ax = plt.subplots(figsize=figsize, dpi=200)

    for dataset_type in DATASET_TYPE_ORDER:
        type_data = plot_df[plot_df['dataset_type'] == dataset_type]
        if not type_data.empty:
            color = get_color_by_type(dataset_type)
            english_label = type_mapping.get(dataset_type, dataset_type.capitalize())
            ax.scatter(
                type_data['n_trials'],
                type_data['n_neurons'],
                marker='X',
                s=200,
                color=color,
                alpha=0.8,
                label=english_label,
                edgecolors='black',
                linewidth=1.5,
            )

    ax.set_xlabel('Number of Trials', fontsize=18, fontweight='bold')
    ax.set_ylabel('Number of Neurons', fontsize=18, fontweight='bold')
    ax.set_title('Data Quantity per Dataset — Pre-preprocessing\n(Neurons vs Trials)',
                 fontsize=20, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(title='Dataset Type', title_fontsize=16, fontsize=15,
              loc='upper right', frameon=True, fancybox=True, shadow=True)

    for _, row in plot_df.iterrows():
        short_name = row['description'].replace(' - ', '\n')
        if 'DG S19' in row['description']:
            xytext, ha = (-5, 5), 'right'
        else:
            xytext, ha = (5, 5), 'left'
        ax.annotate(
            short_name,
            (row['n_trials'], row['n_neurons']),
            xytext=xytext, textcoords='offset points',
            fontsize=13, ha=ha, va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
        )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    x_margin = (plot_df['n_trials'].max() - plot_df['n_trials'].min()) * 0.1
    y_margin = (plot_df['n_neurons'].max() - plot_df['n_neurons'].min()) * 0.1
    ax.set_xlim(plot_df['n_trials'].min() - x_margin, plot_df['n_trials'].max() + x_margin)
    ax.set_ylim(plot_df['n_neurons'].min() - y_margin, plot_df['n_neurons'].max() + y_margin)

    plt.tight_layout()
    if save_path:
        if save_path.endswith('.svg'):
            plt.savefig(save_path, format='svg', bbox_inches='tight')
        else:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📁 Gráfico (pre-preprocessing) guardado en: {save_path}")
    plt.show()
    return fig, ax


# Generar los gráficos (pre y post preprocessing) si tenemos datos
if not results_df.empty:
    print("📊 Creando gráfico de cantidad de datos (post-preprocessing)...")
    fig_data, ax_data = create_data_quantity_plot(
        results_df,
        save_path=f'{plot_path}/data_quantity_plot.svg',
        figsize=(12, 8)
    )
    print("\n📊 Creando gráfico de cantidad de datos (pre-preprocessing)...")
    fig_data_pre, ax_data_pre = create_data_quantity_plot_pre(
        results_df,
        save_path=f'{plot_path}/data_quantity_plot_pre.svg',
        figsize=(12, 8)
    )
else:
    print("❌ No hay datos disponibles para el gráfico de cantidad de datos")


# %%
# Celda 3: Análisis de configuraciones óptimas

print("=" * 100)
print("ANÁLISIS DE HIPERPARÁMETROS ÓPTIMOS")
print("=" * 100)

def extract_best_hyperparameters(study_dirs, results_df):
    """Extrae los hiperparámetros del mejor trial de cada dataset"""
    hyperparams_data = []
    
    for _, row in results_df.iterrows():
        study_path = row['study_path']
        dataset_name = row['dataset_name']
        
        try:
            # Buscar el mejor trial en cada fold
            fold_results = load_fold_results(study_path)
            
            all_configs = []
            for fold_num, fold_data in fold_results.items():
                best_trial = fold_data['best_trial_number']
                
                # Cargar configuración del mejor trial
                config_path = os.path.join(
                    study_path, 'training_results', f'fold_{fold_num}', 
                    f'trial_{best_trial}'
                )
                
                # Buscar archivo de configuración
                if os.path.exists(config_path):
                    config_files = [f for f in os.listdir(config_path) if f.endswith('_config.json')]
                    if config_files:
                        config_file = os.path.join(config_path, config_files[0])
                        with open(config_file, 'r') as f:
                            config = json.load(f)
                        
                        all_configs.append({
                            'fold': fold_num,
                            'hidden_size': config.get('hidden_size', None),
                            'num_layers': config.get('num_layers', None),
                            'dropout': config.get('dropout', None),
                            'lr': config.get('lr', None),
                            'r2_score': config.get('r2_score', None)
                        })
            
            if all_configs:
                # Calcular estadísticas de hiperparámetros
                hidden_sizes = [c['hidden_size'] for c in all_configs if c['hidden_size'] is not None]
                num_layers = [c['num_layers'] for c in all_configs if c['num_layers'] is not None]
                dropouts = [c['dropout'] for c in all_configs if c['dropout'] is not None]
                lrs = [c['lr'] for c in all_configs if c['lr'] is not None]
                
                hyperparams_data.append({
                    'dataset_name': dataset_name,
                    'description': row['description'],
                    'dataset_type': row['dataset_type'],
                    'mean_test_r2': row['mean_test_r2'],
                    'mean_val_r2': row['mean_val_r2'],
                    'mean_position_r2': row['mean_position_r2'],
                    'mean_velocity_r2': row['mean_velocity_r2'],
                    'mean_hidden_size': np.mean(hidden_sizes) if hidden_sizes else None,
                    'std_hidden_size': np.std(hidden_sizes) if hidden_sizes else None,
                    'hidden_size_values': hidden_sizes,
                    'mean_num_layers': np.mean(num_layers) if num_layers else None,
                    'std_num_layers': np.std(num_layers) if num_layers else None,
                    'num_layers_values': num_layers,
                    'mean_dropout': np.mean(dropouts) if dropouts else None,
                    'std_dropout': np.std(dropouts) if dropouts else None,
                    'mean_lr': np.mean(lrs) if lrs else None,
                    'std_lr': np.std(lrs) if lrs else None,
                    'configs': all_configs
                })
                
        except Exception as e:
            print(f"   ⚠️  Error extrayendo hiperparámetros de {dataset_name}: {e}")
    
    return hyperparams_data

if not results_df.empty:
    print("🔄 Extrayendo hiperparámetros óptimos...")
    hyperparams_data = extract_best_hyperparameters(study_dirs, results_df)
    
    if hyperparams_data:
        hyperparams_df = pd.DataFrame(hyperparams_data)
        
        print(f"\n📋 HIPERPARÁMETROS ÓPTIMOS POR DATASET:")
        print(f"{'Dataset':<30} {'Hidden Size':<12} {'Layers':<8} {'Dropout':<10} {'LR (log10)':<12} {'R² Test':<10}")
        print("-" * 100)
        
        for _, row in hyperparams_df.sort_values('mean_test_r2', ascending=False).iterrows():
            dataset = f"{row['description']}"
            hidden = f"{row['mean_hidden_size']:.0f}±{row['std_hidden_size']:.0f}" if row['mean_hidden_size'] else "N/A"
            layers = f"{row['mean_num_layers']:.1f}±{row['std_num_layers']:.1f}" if row['mean_num_layers'] else "N/A"
            dropout = f"{row['mean_dropout']:.3f}±{row['std_dropout']:.3f}" if row['mean_dropout'] else "N/A"
            lr_log = f"{np.log10(row['mean_lr']):.2f}±{np.log10(row['std_lr']) if row['std_lr'] > 0 else 0:.2f}" if row['mean_lr'] else "N/A"
            r2 = f"{row['mean_test_r2']:.4f}"
            
            print(f"{dataset:<30} {hidden:<12} {layers:<8} {dropout:<10} {lr_log:<12} {r2:<10}")
        hyperparams_df.to_csv(f'{plot_path}/hyperparams.csv', index=False)

        # Análisis de tendencias
        print(f"\n🔍 TENDENCIAS EN HIPERPARÁMETROS:")
        
        # Hidden size vs performance
        valid_hidden = hyperparams_df.dropna(subset=['mean_hidden_size', 'mean_test_r2'])
        if not valid_hidden.empty:
            corr_hidden = valid_hidden['mean_hidden_size'].corr(valid_hidden['mean_test_r2'])
            print(f"   • Correlación Hidden Size vs R²: {corr_hidden:.3f}")
        
        # Num layers vs performance  
        valid_layers = hyperparams_df.dropna(subset=['mean_num_layers', 'mean_test_r2'])
        if not valid_layers.empty:
            corr_layers = valid_layers['mean_num_layers'].corr(valid_layers['mean_test_r2'])
            print(f"   • Correlación Num Layers vs R²: {corr_layers:.3f}")
            
        # Dropout vs performance
        valid_dropout = hyperparams_df.dropna(subset=['mean_dropout', 'mean_test_r2'])
        if not valid_dropout.empty:
            corr_dropout = valid_dropout['mean_dropout'].corr(valid_dropout['mean_test_r2'])
            print(f"   • Correlación Dropout vs R²: {corr_dropout:.3f}")


# %%
# Celda 4: Visualización comparativa para POSICIÓN

print("=" * 100)
print("ANÁLISIS DE PERFORMANCE - DECODIFICACIÓN DE POSICIÓN")
print("=" * 100)

def prepare_position_data_for_boxplot(results_df):
    """Prepara los datos de posición para visualización en boxplot"""
    boxplot_data = []
    
    for _, row in results_df.iterrows():
        dataset_name = row['description']
        dataset_type = row['dataset_type']
        
        # Extraer R² scores de posición de cada fold
        fold_results = load_fold_results(row['study_path'])
        
        for fold_num, fold_data in fold_results.items():
            if 'additional_test_r2_scores' in fold_data and len(fold_data['additional_test_r2_scores']) >= 2:
                position_r2 = fold_data['additional_test_r2_scores'][0]  # Primer valor = posición
                
                # Mapear tipo a inglés
                #type_mapping = {'piriform-experto': 'Piriform-Expert', 'piriform-naive': 'Piriform-Naive', 'CA3-experto': 'CA3-Expert', 'CA3-naive': 'CA3-Naive', 'DG-experto': 'DG-Expert', 'DG-naive': 'DG-Naive'}
                english_type = type_mapping.get(dataset_type, dataset_type.capitalize())
                
                boxplot_data.append({
                    'Dataset': dataset_name,
                    'R2_Score': position_r2,
                    'Fold': fold_num,
                    'Type': english_type,
                    'Target': 'Position'
                })
    
    return pd.DataFrame(boxplot_data)

def create_position_performance_boxplot(boxplot_df, save_path=None, figsize=(14, 8)):
    """Creates boxplot for position performance in ibioml style"""
    
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=figsize, dpi=200)
    
    
    # Ordenar datasets por tipo y luego por performance
    dataset_order = get_ordered_datasets_by_type(boxplot_df)
    
    # Crear paleta de colores como diccionario {dataset: color}
    dataset_colors = {}
    for dataset in dataset_order:
        dataset_type = boxplot_df[boxplot_df['Dataset'] == dataset]['Type'].iloc[0]
        color = type_colors.get(dataset_type, '#333333')
        dataset_colors[dataset] = color
    
    # Boxplot estilo ibioml
    sns.boxplot(
        data=boxplot_df, 
        x='Dataset', 
        y='R2_Score',
        hue='Dataset',
        order=dataset_order,
        palette=dataset_colors,
        fill=False,
        showfliers=False,
        linewidth=2,
        legend=False,
        ax=ax
    )
    
    # Stripplot con cruces
    sns.stripplot(
        data=boxplot_df,
        x='Dataset', 
        y='R2_Score',
        hue='Dataset',
        order=dataset_order,
        palette=dataset_colors,
        jitter=False,
        size=8,
        marker=".",
        alpha=0.7,
        legend=False,
        ax=ax
    )
    
    # Estilo ibioml
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    ax.set_xlabel('Dataset', fontsize=18, fontweight='bold')
    ax.set_ylabel('Accuracy (R²) - Position', fontsize=18, fontweight='bold')
    ax.set_title('Position Decoding Performance by Dataset\n(5-Fold Cross-Validation)', 
                fontsize=20, fontweight='bold', pad=20)
    
    plt.xticks(rotation=45, ha='right', fontsize=15)
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 0.8)

    # Leyenda con medianas
    medians = boxplot_df.groupby('Dataset')['R2_Score'].median()
    ordered_medians = [medians[dataset] for dataset in dataset_order]
    dataset_types = [boxplot_df[boxplot_df['Dataset'] == dataset]['Type'].iloc[0] for dataset in dataset_order]
    
    legend_elements = []
    for i, (dataset, median, dtype) in enumerate(zip(dataset_order, ordered_medians, dataset_types)):
        color = type_colors.get(dtype, '#333333')
        legend_elements.append(
            plt.Line2D([0], [0], color=color, lw=3, 
                      label=f'{dataset}: {median:.3f}')
        )
    
    ax.legend(
        handles=legend_elements,
        title="Position Accuracy (R²) - Median",
        title_fontsize=16,
        fontsize=15,
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )
    
    plt.tight_layout()
    
    if save_path:
        # Detectar formato basado en la extensión
        if save_path.endswith('.svg'):
            plt.savefig(save_path, format='svg', bbox_inches='tight')
        else:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📁 Gráfico de posición guardado en: {save_path}")
    
    plt.show()
    
    # Estadísticas de posición
    print(f"\n📊 ESTADÍSTICAS DE DECODIFICACIÓN DE POSICIÓN:")
    print(f"   • R² promedio: {boxplot_df['R2_Score'].mean():.4f} ± {boxplot_df['R2_Score'].std():.4f}")
    print(f"   • Mejor dataset: {dataset_order[0]} (R² = {ordered_medians[0]:.4f})")
    print(f"   • Peor dataset: {dataset_order[-1]} (R² = {ordered_medians[-1]:.4f})")
    
    # Estadísticas por tipo
    type_stats = boxplot_df.groupby('Type')['R2_Score'].agg(['mean', 'std', 'median']).round(4)
    print(f"\n📈 PERFORMANCE DE POSICIÓN POR TIPO:")
    for tipo in type_stats.index:
        stats = type_stats.loc[tipo]
        print(f"   • {tipo}: {stats['mean']:.4f} ± {stats['std']:.4f} (mediana: {stats['median']:.4f})")
    
    return fig, ax

# Generar visualización para posición
if not results_df.empty:
    print("🎯 Preparando datos de posición para visualización...")
    position_boxplot_df = prepare_position_data_for_boxplot(results_df)
    
    if not position_boxplot_df.empty:
        print(f"📊 Datos de posición preparados:")
        print(f"   • {len(position_boxplot_df)} puntos de datos")
        print(f"   • {position_boxplot_df['Dataset'].nunique()} datasets únicos")
        
        print(f"\n🎯 Creando boxplot de POSICIÓN...")
    #    print(position_boxplot_df)
        fig_pos, ax_pos = create_position_performance_boxplot(
            position_boxplot_df, 
            save_path=f'{plot_path}/position_performance_boxplot.svg',
            figsize=(15, 8)
        )
    else:
        print("❌ No se encontraron datos de posición")
else:
    print("❌ No hay datos disponibles para análisis de posición")

# %%
# Celda 5: Visualización comparativa para VELOCIDAD

print("=" * 100)
print("ANÁLISIS DE PERFORMANCE - DECODIFICACIÓN DE VELOCIDAD")
print("=" * 100)

def prepare_velocity_data_for_boxplot(results_df):
    """Prepara los datos de velocidad para visualización en boxplot"""
    boxplot_data = []
    
    for _, row in results_df.iterrows():
        dataset_name = row['description']
        dataset_type = row['dataset_type']
        
        # Extraer R² scores de velocidad de cada fold
        fold_results = load_fold_results(row['study_path'])
        
        for fold_num, fold_data in fold_results.items():
            if 'additional_test_r2_scores' in fold_data and len(fold_data['additional_test_r2_scores']) >= 2:
                velocity_r2 = fold_data['additional_test_r2_scores'][1]  # Segundo valor = velocidad
                
                # Mapear tipo a inglés
                #type_mapping = {'piriform-experto': 'Piriform-Expert', 'piriform-naive': 'Piriform-Naive', 'CA3-experto': 'CA3-Expert', 'CA3-naive': 'CA3-Naive', 'DG-experto': 'DG-Expert', 'DG-naive': 'DG-Naive'}
                english_type = type_mapping.get(dataset_type, dataset_type.capitalize())
                
                boxplot_data.append({
                    'Dataset': dataset_name,
                    'R2_Score': velocity_r2,
                    'Fold': fold_num,
                    'Type': english_type,
                    'Target': 'Velocity'
                })
    
    return pd.DataFrame(boxplot_data)

def create_velocity_performance_boxplot(boxplot_df, save_path=None, figsize=(14, 8)):
    """Creates boxplot for speed performance in ibioml style"""
    
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=figsize, dpi=200)
    

    
    # Ordenar datasets por tipo y luego por performance
    dataset_order = get_ordered_datasets_by_type(boxplot_df)
    
    # Crear paleta de colores como diccionario {dataset: color}
    dataset_colors = {}
    for dataset in dataset_order:
        dataset_type = boxplot_df[boxplot_df['Dataset'] == dataset]['Type'].iloc[0]
        color = type_colors.get(dataset_type, '#333333')
        dataset_colors[dataset] = color
    
    # Boxplot estilo ibioml
    sns.boxplot(
        data=boxplot_df, 
        x='Dataset', 
        y='R2_Score',
        hue='Dataset',
        order=dataset_order,
        palette=dataset_colors,
        fill=False,
        showfliers=False,
        linewidth=2,
        legend=False,
        ax=ax
    )
    
    # Stripplot con cruces
    sns.stripplot(
        data=boxplot_df,
        x='Dataset', 
        y='R2_Score',
        hue='Dataset',
        order=dataset_order,
        palette=dataset_colors,
        jitter=False,
        size=8,
        marker=".",
        alpha=0.7,
        legend=False,
        ax=ax
    )
    
    # Estilo ibioml
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    ax.set_xlabel('Dataset', fontsize=18, fontweight='bold')
    ax.set_ylabel('Accuracy (R²) - Speed', fontsize=18, fontweight='bold')
    ax.set_title('Speed Decoding Performance by Dataset\n(5-Fold Cross-Validation)', 
                fontsize=20, fontweight='bold', pad=20)
    
    plt.xticks(rotation=45, ha='right', fontsize=15)
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 0.8)

    # Leyenda con medianas
    medians = boxplot_df.groupby('Dataset')['R2_Score'].median()
    ordered_medians = [medians[dataset] for dataset in dataset_order]
    dataset_types = [boxplot_df[boxplot_df['Dataset'] == dataset]['Type'].iloc[0] for dataset in dataset_order]
    
    legend_elements = []
    for i, (dataset, median, dtype) in enumerate(zip(dataset_order, ordered_medians, dataset_types)):
        color = type_colors.get(dtype, '#333333')
        legend_elements.append(
            plt.Line2D([0], [0], color=color, lw=3, 
                      label=f'{dataset}: {median:.3f}')
        )
    
    ax.legend(
        handles=legend_elements,
        title="Speed Accuracy (R²) - Median",
        title_fontsize=16,
        fontsize=15,
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )
    
    plt.tight_layout()
    
    if save_path:
        # Detectar formato basado en la extensión
        if save_path.endswith('.svg'):
            plt.savefig(save_path, format='svg', bbox_inches='tight')
        else:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📁 Gráfico de velocidad guardado en: {save_path}")
    
    plt.show()
    
    # Estadísticas de velocidad
    print(f"\n📊 ESTADÍSTICAS DE DECODIFICACIÓN DE VELOCIDAD:")
    print(f"   • R² promedio: {boxplot_df['R2_Score'].mean():.4f} ± {boxplot_df['R2_Score'].std():.4f}")
    print(f"   • Mejor dataset: {dataset_order[0]} (R² = {ordered_medians[0]:.4f})")
    print(f"   • Peor dataset: {dataset_order[-1]} (R² = {ordered_medians[-1]:.4f})")
    
    # Estadísticas por tipo
    type_stats = boxplot_df.groupby('Type')['R2_Score'].agg(['mean', 'std', 'median']).round(4)
    print(f"\n📈 PERFORMANCE DE VELOCIDAD POR TIPO:")
    for tipo in type_stats.index:
        stats = type_stats.loc[tipo]
        print(f"   • {tipo}: {stats['mean']:.4f} ± {stats['std']:.4f} (mediana: {stats['median']:.4f})")
    
    return fig, ax

# Generar visualización para velocidad
if not results_df.empty:
    print("🚀 Preparando datos de velocidad para visualización...")
    velocity_boxplot_df = prepare_velocity_data_for_boxplot(results_df)
    
    if not velocity_boxplot_df.empty:
        print(f"📊 Datos de velocidad preparados:")
        print(f"   • {len(velocity_boxplot_df)} puntos de datos")
        print(f"   • {velocity_boxplot_df['Dataset'].nunique()} datasets únicos")
        
        print(f"\n🚀 Creando boxplot de VELOCIDAD...")
        fig_vel, ax_vel = create_velocity_performance_boxplot(
            velocity_boxplot_df, 
            save_path=f'{plot_path}/velocity_performance_boxplot.svg',
            figsize=(15, 8)
        )
    else:
        print("❌ No se encontraron datos de velocidad")
else:
    print("❌ No hay datos disponibles para análisis de velocidad")

# %%
# Celda 6: Gráfico bidimensional Posición vs Velocidad

print("=" * 100)
print("ANÁLISIS BIDIMENSIONAL: POSICIÓN vs VELOCIDAD")
print("=" * 100)

def create_position_vs_velocity_scatter(results_df, save_path=None, figsize=(12, 8)):
    """Creates scatter plot: Position Accuracy (R²) (X) vs Speed Accuracy (R²) (Y)"""
    
    if results_df.empty:
        print("❌ No hay datos disponibles para el gráfico")
        return None, None
    
    # Preparar datos para el gráfico
    scatter_data = []
    
    for _, row in results_df.iterrows():
        dataset_name = row['description']
        dataset_type = row['dataset_type']
        
        # Extraer R² scores de posición y velocidad de cada fold
        fold_results = load_fold_results(row['study_path'])
        
        position_scores = []
        velocity_scores = []
        
        for fold_num, fold_data in fold_results.items():
            if 'additional_test_r2_scores' in fold_data and len(fold_data['additional_test_r2_scores']) >= 2:
                position_r2 = fold_data['additional_test_r2_scores'][0]  # Posición
                velocity_r2 = fold_data['additional_test_r2_scores'][1]  # Velocidad
                
                position_scores.append(position_r2)
                velocity_scores.append(velocity_r2)
        
        if position_scores and velocity_scores:
            # Usar la mediana de cada dataset
            median_position = np.median(position_scores)
            median_velocity = np.median(velocity_scores)
            
            scatter_data.append({
                'dataset_name': row['dataset_name'],
                'description': dataset_name,
                'dataset_type': dataset_type,
                'position_r2': median_position,
                'velocity_r2': median_velocity,
                'position_scores': position_scores,
                'velocity_scores': velocity_scores
            })
    
    if not scatter_data:
        print("❌ No se pudieron extraer datos de R² para el gráfico")
        return None, None
    
    # Crear DataFrame para el gráfico
    scatter_df = pd.DataFrame(scatter_data)
    
    # Configurar el gráfico
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=figsize, dpi=200)
    
    # Crear scatter plot con cruces, coloreado por tipo de dataset
    for dataset_type in DATASET_TYPE_ORDER:
        type_data = scatter_df[scatter_df['dataset_type'] == dataset_type]
        if not type_data.empty:
            color = get_color_by_type(dataset_type)
            english_label = type_mapping.get(dataset_type, dataset_type.capitalize())
            ax.scatter(
                type_data['position_r2'],
                type_data['velocity_r2'],
                marker='X',
                s=200,  # Tamaño de las cruces
                color=color,
                alpha=0.8,
                label=english_label,
                edgecolors='black',
                linewidth=1.5
            )
    
    
    # Personalización del gráfico
    ax.set_xlabel('Position Accuracy (R²) - Median', fontsize=18, fontweight='bold')
    ax.set_ylabel('Speed Accuracy (R²) - Median', fontsize=18, fontweight='bold')
    ax.set_title('Performance Comparison: Position vs Speed\n(Median Accuracy (R²) per Dataset)', 
                fontsize=20, fontweight='bold', pad=20)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Leyenda fuera del área del plot, matchea el estilo de los boxplots.
    ax.legend(title='Dataset Type', title_fontsize=16, fontsize=15,
             loc='upper left', bbox_to_anchor=(1.02, 1),
             frameon=True, fancybox=True, shadow=True)
    
    # Anotaciones para cada punto
    for _, row in scatter_df.iterrows():
        # Nombre corto del dataset
        short_name = row['description'].replace(' - ', '\n')

        ax.annotate(
            short_name,
            (row['position_r2'], row['velocity_r2']),
            xytext=(8, 8),  # Offset del texto
            textcoords='offset points',
            fontsize=13,
            ha='left',
            va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
        )
    
    # Estilo
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Ajustar límites para mejor visualización (comenzando en 0.0)
    x_margin = scatter_df['position_r2'].max() * 0.1
    y_margin = scatter_df['velocity_r2'].max() * 0.1

    ax.set_xlim(0.0, scatter_df['position_r2'].max() + x_margin)
    ax.set_ylim(0.0, scatter_df['velocity_r2'].max() + y_margin)
    
    plt.tight_layout()
    
    # Guardar si se especifica path
    if save_path:
        # Detectar formato basado en la extensión
        if save_path.endswith('.svg'):
            plt.savefig(save_path, format='svg', bbox_inches='tight')
        else:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📁 Gráfico guardado en: {save_path}")
    
    plt.show()
    
    # Imprimir estadísticas
    print(f"\n📊 ANÁLISIS POSICIÓN vs VELOCIDAD:")
    print(f"{'Dataset':<25} {'Tipo':<10} {'R² Pos':<10} {'R² Vel':<10} {'Diferencia':<12} {'Mejor en':<12}")
    print("-" * 85)
    
    for _, row in scatter_df.sort_values('position_r2', ascending=False).iterrows():
        dataset = f"{row['description']}"
        dtype = f"{row['dataset_type']}"
        pos_r2 = f"{row['position_r2']:.3f}"
        vel_r2 = f"{row['velocity_r2']:.3f}"
        diff = row['position_r2'] - row['velocity_r2']
        diff_str = f"{diff:+.3f}"
        better_at = "Posición" if diff > 0 else "Velocidad" if diff < 0 else "Igual"
        
        print(f"{dataset:<25} {dtype:<10} {pos_r2:<10} {vel_r2:<10} {diff_str:<12} {better_at:<12}")
    
    # Estadísticas generales
    pos_mean = scatter_df['position_r2'].mean()
    vel_mean = scatter_df['velocity_r2'].mean()
    
    print(f"\n📈 ESTADÍSTICAS GENERALES:")
    print(f"   • R² Posición promedio: {pos_mean:.4f} ± {scatter_df['position_r2'].std():.4f}")
    print(f"   • R² Velocidad promedio: {vel_mean:.4f} ± {scatter_df['velocity_r2'].std():.4f}")
    print(f"   • Diferencia promedio (Pos - Vel): {pos_mean - vel_mean:+.4f}")
    
    # Correlación
    correlation = scatter_df['position_r2'].corr(scatter_df['velocity_r2'])
    print(f"   • Correlación Posición-Velocidad: {correlation:.4f}")
    
    if pos_mean > vel_mean:
        print(f"   🏆 En promedio, POSICIÓN es más fácil de decodificar")
    else:
        print(f"   🏆 En promedio, VELOCIDAD es más fácil de decodificar")
    
    # Análisis por cuadrantes
    pos_median = scatter_df['position_r2'].median()
    vel_median = scatter_df['velocity_r2'].median()
    
    print(f"\n🎯 ANÁLISIS POR CUADRANTES (mediana como referencia):")
    print(f"   • Mediana Posición: {pos_median:.3f}")
    print(f"   • Mediana Velocidad: {vel_median:.3f}")
    
    q1 = scatter_df[(scatter_df['position_r2'] >= pos_median) & (scatter_df['velocity_r2'] >= vel_median)]
    q2 = scatter_df[(scatter_df['position_r2'] < pos_median) & (scatter_df['velocity_r2'] >= vel_median)]
    q3 = scatter_df[(scatter_df['position_r2'] < pos_median) & (scatter_df['velocity_r2'] < vel_median)]
    q4 = scatter_df[(scatter_df['position_r2'] >= pos_median) & (scatter_df['velocity_r2'] < vel_median)]
    
    print(f"   • Q1 (Alta Pos, Alta Vel): {len(q1)} datasets - {list(q1['description'])}")
    print(f"   • Q2 (Baja Pos, Alta Vel): {len(q2)} datasets - {list(q2['description'])}")
    print(f"   • Q3 (Baja Pos, Baja Vel): {len(q3)} datasets - {list(q3['description'])}")
    print(f"   • Q4 (Alta Pos, Baja Vel): {len(q4)} datasets - {list(q4['description'])}")
    
    return fig, ax

# Generar el gráfico si tenemos datos
if not results_df.empty:
    print("📊 Creando gráfico bidimensional Posición vs Velocidad...")
    fig_scatter, ax_scatter = create_position_vs_velocity_scatter(
        results_df,
        save_path=f'{plot_path}/position_vs_velocity_scatter.svg',
        figsize=(12, 8)
    )
else:
    print("❌ No hay datos disponibles para el gráfico bidimensional")

# %%
