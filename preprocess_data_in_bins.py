#%% IMPORTO LIBRERIAS
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from scipy import stats
import pickle
import time
import sys
from utils.preprocessing_funcs import get_spikes_with_history, get_trial_bin_indices


#%% 
# Carga de datos
mat_contents = io.loadmat('datasets/L5_bins200ms_completo.mat')
neural_data = mat_contents['neuronActivity'].copy()
rewCtxt = mat_contents['rewCtxt'].copy()
trialFinalBin = np.ravel(mat_contents['trialFinalBin'].copy())
dPrime = np.ravel(mat_contents['dPrime'].copy())
criterion = np.ravel(mat_contents['criterion'].copy())
rewCtxt = rewCtxt.squeeze()

# Variables a decodificar
pos_binned = mat_contents['position'].copy()
vels_binned = mat_contents['velocity'].copy()

# Elección de la variable a decodificar 
y=pos_binned

# Inicializar array de trials
trial_ids = np.zeros_like(rewCtxt, dtype=int)

start = 0
for i, end in enumerate(trialFinalBin):
    trial_ids[start:end + 1] = i + 1  # i+1 para que el primer trial sea 1, no 0
    start = end + 1

# Tamaño de time bins en segundos
time_bins_size = 0.2

# Booleano para graficos
plotting = False

#%%
print("=== RESUMEN DE DATOS ===")

# Cantidad de neuronas
print("Cantidad de neuronas:", neural_data.shape[1])

# Cantidad de ensayos
print("Cantidad de ensayos (trials):", trialFinalBin.shape[0])

# Cantidad de time bins
print("Cantidad total de time bins:", neural_data.shape[0], "=", f"{(neural_data.shape[0]*time_bins_size):.2f}", "segundos", "=", f"{(neural_data.shape[0]*time_bins_size)/60:.2f}", "minutos")

# Cantidad de trials con contexto recompensado
# Esto se calcula contando cuántos trials terminan en un bin que corresponde a rewCtxt == 1
# Es decir, se usa trialFinalBin como índice
trial_rewCtxt = rewCtxt[trialFinalBin - 1]  # -1 porque los índices en Python empiezan en 0
n_rewarded_trials = np.sum(trial_rewCtxt == 1)
print("Cantidad de trials con contexto recompensado:", n_rewarded_trials)

# Duración promedio del trial (en bins y en segundos)
mean_trial_duration_bins = np.mean(mat_contents['trialDurationInBins'])
mean_trial_duration_seconds = mean_trial_duration_bins * time_bins_size
print("Duración promedio de los trials:", f"{mean_trial_duration_bins:.2f} bins, {mean_trial_duration_seconds:.2f} segundos")

# Actividad neuronal
print("Shape de la actividad neuronal:", neural_data.shape)

#%%
# Graficar duración de los trials
if plotting:
    plt.figure(figsize=(10, 6))
    plt.title('Duración de ensayos')
    plt.plot(mat_contents['trialDurationInBins'] * time_bins_size, label='Duración de los trials')
    plt.axhline(mean_trial_duration_seconds, color='r', linestyle='--', label='Duración media = ' + f"{mean_trial_duration_seconds:.2f}")
    plt.text(-len(mat_contents['trialDurationInBins'])*0.06, mean_trial_duration_seconds, f'{mean_trial_duration_seconds:.2f}', ha='right', va='center', fontsize=8, color='red')
    plt.xlabel('Índice del trial')
    plt.ylabel('Duración del trial (segundos)')
    plt.legend()
    plt.show()

#%%
# Graficar performance por ensayo
mean_dPrime = np.nanmean(dPrime)
if plotting:
    plt.figure(figsize=(10, 6))
    plt.title('Performance por ensayo')
    plt.plot(dPrime, label='dPrime')
    plt.axhline(mean_dPrime, color='r', linestyle='--', label='Rendimiento medio = ' + f"{mean_dPrime:.2f}")
    plt.xlabel('Índice del trial')
    plt.ylabel('dPrime')
    plt.xticks(np.arange(0, len(dPrime), len(dPrime)//5))
    plt.grid(True, axis='x')
    plt.legend()
    plt.show()

#%%%
# Graficar tasa de disparo media por neurona
total_spikes_per_neuron = np.sum(neural_data, axis=0)  # Sumar spikes por neurona
total_time_secs = neural_data.shape[0] * time_bins_size  # 200 ms por bin
mean_firing_rates = total_spikes_per_neuron / total_time_secs

# Coordenadas x
x = np.arange(len(mean_firing_rates))

# Calcular la media de tasas
mean_rate = np.mean(mean_firing_rates)

if plotting:
    plt.figure(figsize=(12, 5))

    for i in x:
        # Línea punteada vertical
        plt.plot([i, i], [0, mean_firing_rates[i]], linestyle='--', color='tab:blue', linewidth=1)

        # Círculo al final
        plt.plot(i, mean_firing_rates[i], marker='o', color='tab:blue')

        # Etiqueta con el valor encima del círculo
        plt.text(i, mean_firing_rates[i] + .25, f"{mean_firing_rates[i]:.2f}", 
                 ha='center', va='bottom', fontsize=8, rotation=0)

    # Línea horizontal con la media
    plt.axhline(mean_rate, linestyle='--', color='red', linewidth=1.5, label=f"Media: {mean_rate:.2f} Hz")
    plt.text(-len(mean_firing_rates) * 0.05, mean_rate, f"{mean_rate:.2f}", va='center', ha='right', fontsize=8, color='red')
    plt.legend()

    # Estética
    plt.xlabel("Índice de neurona")
    plt.ylabel("Tasa de disparo media (Hz)")
    plt.title("Tasa de disparo media por neurona")
    step = 4
    xticks = list(range(0, len(mean_firing_rates), step))
    if (len(mean_firing_rates) - 1) not in xticks:
        xticks.append(len(mean_firing_rates) - 1)
    plt.xticks(ticks=xticks, labels=xticks)
    plt.grid(axis='y', linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.show()


#%%
# Agrego el contexto a los datos
rewCtxt_neg = np.logical_not(rewCtxt).astype("uint8")
neural_data_with_ctxt = np.concatenate((neural_data, rewCtxt[:,np.newaxis], rewCtxt_neg[:,np.newaxis]), axis=1)

# Se agregaron dos neuronas que indican el contexto
print("Shape de la actividad neuronal con contexto:", neural_data_with_ctxt.shape)

#%% 
# Defino ventanas temporales y spikes with history
# Tomamos la actividad pasada y futura inmediata para cada time bin
bins_before = 5
bins_current = 1
bins_after = 5

# Obtengo los spikes con historia
X=get_spikes_with_history(neural_data_with_ctxt,bins_before,bins_after,bins_current)
# esto lo realizo ahora para luego poder eliminar los spikes sin historia que quedan al principio y al final del tensor X, es decir, los nans que se generan en el proceso de obtener los spikes con historia en donde no hay historia
print("Shape del X:", X.shape)


#%% 
# LIMPIEZA DE DATOS POR INACTIVIDAD
trialDurationInBins = np.ravel(mat_contents['trialDurationInBins'].copy())  
threshTrialDuration=np.mean(trialDurationInBins)+3*np.std(trialDurationInBins)

# Grafico la duracion de los trials
plt.figure(figsize=(10, 6))
plt.plot(trialDurationInBins, label='Duración de los trials')
plt.axhline(threshTrialDuration, color='r', linestyle='--', label='Umbral de duración')
plt.xlabel('Índice del trial')
plt.ylabel('Duración del trial (bins)')
plt.show()

trialsTooLong= np.ravel(np.where(trialDurationInBins>=threshTrialDuration))
indices_to_remove_trialDuration=[]

if trialsTooLong.size > 0:
    print('Indices de trials con duracion excesiva: ', trialsTooLong)

    for trial in trialsTooLong:
        startInd, endInd = get_trial_bin_indices(trialFinalBin, trial)
        indices_to_remove_trialDuration.extend(range(startInd,endInd))

    # Agrego los indices de los trials a remover por duracion excesiva 
    indices_to_remove_trialDuration=np.array(indices_to_remove_trialDuration)
    print("Índices de los time bins a eliminar:", indices_to_remove_trialDuration)
    print("Trials a remover por duracion excesiva: ", np.unique(trial_ids[indices_to_remove_trialDuration]))

else:
    print("No hay trials a remover por duracion excesiva")

    
#%% 
# CLEANING DE BOUNDARIES SIN HISTORY
# Removemos los primeros bins y los ultimos porque no tienen historia (son los creados por get_spikes_with_history), boundaries
first_indexes = np.arange(bins_before)
last_indexes = np.arange(X.shape[0]-bins_after,X.shape[0])

# Concatenamos con los indices que queremos remover por inactividad
indices_to_remove_temp = np.concatenate((first_indexes, indices_to_remove_trialDuration, last_indexes))
print("Indices a remover por ahora, sin historia y por inactividad:", indices_to_remove_temp)


#%% 
# LIMPIEZA DE DATOS POR LOW PERFORMANCE
# Filtrar los índices de los trials cuyo dPrime es menor que 2.8 o es NaN
low_performance_trials_indices = np.where((dPrime <= 2.8) | (np.isnan(dPrime)))[0]

# Mostrar los índices de los trials
print("Indices de trials con dPrime menor o igual a 2.8:", low_performance_trials_indices)
print("Cantidad de trials:", len(low_performance_trials_indices))

# Crear una lista para almacenar los índices de los time bins a eliminar
indices_to_remove_low_performance = []

for trial in low_performance_trials_indices:
    startInd, endInd = get_trial_bin_indices(trialFinalBin, trial)
    indices_to_remove_low_performance.extend(range(startInd,endInd))

# Convertir la lista a un array de numpy
indices_to_remove_low_performance = np.array(indices_to_remove_low_performance)
print("Trials a remover por bajo rendimiento: ", np.unique(trial_ids[indices_to_remove_low_performance]))

#%%
# GRAFICAR TRIALS CON LOW PERFORMANCE   
plt.figure(figsize=(10, 6))
plt.plot(dPrime, label='dPrime')
plt.axhline(2.8, color='r', linestyle='--', label='Umbral de rendimiento')
plt.xlabel('Índice del trial')
plt.ylabel('dPrime')
# x grid que marque 5 lugares equidistantes
plt.xticks(np.arange(0, len(dPrime), len(dPrime)//5))
plt.grid(True, axis='x')
plt.legend()
plt.show()


#%%
# Agregar los índices de bajo rendimiento a los índices a eliminar
rmv_time=np.where(np.isnan(y[:,0])) # indices en los que la posicion es NaN
print("Cantidad de time bins a eliminar por NaN:", len(rmv_time[0]))
indices_to_remove = np.union1d(rmv_time,np.union1d(indices_to_remove_temp, indices_to_remove_low_performance))

# Obtener trials a eliminar
trials_to_remove = np.unique(trial_ids[indices_to_remove])

print("Índices de los time bins a eliminar por bajo rendimiento:", indices_to_remove_low_performance)
print("Índices totales de los time bins a eliminar:", indices_to_remove)
print("Trials a eliminar:", trials_to_remove)
print("Cantidad de trials a eliminar:", len(trials_to_remove))

# Eliminar todos los time bins de esos trials
bins_to_remove = np.where(np.isin(trial_ids, trials_to_remove))[0]
print("Cantidad de time bins a eliminar:", len(bins_to_remove))
print("Trials a eliminar:", np.unique(trial_ids[bins_to_remove]))

print("indices_to_remove:", len(indices_to_remove))
print("bins_to_remove:", len(bins_to_remove))

#%%
# Eliminar los índices de los time bins de bajo rendimiento y cuando el animal se quedó quieto (con boundaries)
X = np.delete(X, bins_to_remove, 0)
y = np.delete(y, bins_to_remove, 0)
trial_ids = np.delete(trial_ids, bins_to_remove, axis=0)

print("Shape de X final:", X.shape)
print("Shape de y final:", y.shape)
print("Shape de trial_ids final:", trial_ids.shape)
print("IDs de trials finales:", np.unique(trial_ids))
print("Cantidad de trials finales:", len(np.unique(trial_ids)))


#%%
# Crear gráfico de firing rates por neurona
firingMinimo=1000
plt.figure(figsize=(12, 6))

# Obtener firing rates de todas las neuronas (sin modificar las últimas dos)
nd_sum_all = np.nansum(X[:,0,:], axis=0)
nd_sum_original = nd_sum_all.copy()  # Guardar valores originales

# Crear array de índices de neuronas
neuron_indices = np.arange(len(nd_sum_all))

# Crear máscara para neuronas que cumplen el criterio
neuronas_cumplen = nd_sum_all >= firingMinimo
neuronas_no_cumplen = nd_sum_all < firingMinimo

# Graficar todas las neuronas
plt.plot(neuron_indices, nd_sum_all, 'o-', color='blue', alpha=0.7, label='Todas las neuronas')

# Marcar en rojo las neuronas que no cumplen el criterio
plt.plot(neuron_indices[neuronas_no_cumplen], nd_sum_all[neuronas_no_cumplen], 'ro', 
         markersize=8, label=f'Neuronas con < {firingMinimo} spikes')

# Línea horizontal del umbral
plt.axhline(y=firingMinimo, color='red', linestyle='--', alpha=0.7, 
           label=f'Umbral mínimo ({firingMinimo} spikes)')

plt.xlabel('Índice de Neurona')
plt.ylabel('Número Total de Spikes')
plt.title('Firing Rates por Neurona - Criterio de Selección')
plt.legend()
plt.grid(True, alpha=0.3)

# Agregar texto con estadísticas
total_neuronas = len(nd_sum_all)
neuronas_eliminadas = np.sum(neuronas_no_cumplen)
neuronas_mantenidas = total_neuronas - neuronas_eliminadas

plt.text(0.02, 0.98, f'Total neuronas: {total_neuronas-2}\nEliminadas: {neuronas_eliminadas}\nMantenidas: {neuronas_mantenidas}', 
         transform=plt.gca().transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.show()

print(f"Neuronas eliminadas por bajo firing rate (< {firingMinimo} spikes): {neuronas_eliminadas}")
print(f"Neuronas mantenidas: {neuronas_mantenidas}")
print(f"Total de neuronas originales: {total_neuronas}")

#%% 
# AGREGO CLEANING DE NEURONAS CON POCOS SPIKES 
#Remove neurons with too few spikes
firingMinimo=1000
nd_sum=np.nansum(X[:,0,:],axis=0)
# Agregar ultimas dos neuronas como excepciones
nd_sum[-2:] = 1e6
rmv_nrn_clean=np.where(nd_sum<firingMinimo)

X = np.delete(X, rmv_nrn_clean, 2)
print("Shape de X final:", X.shape)
print("Trials finales:", np.unique(trial_ids))
print("Cantidad de trials finales:", len(np.unique(trial_ids)))
# Calcular estadísticas finales después del preprocessing
print("=== ESTADÍSTICAS FINALES DESPUÉS DEL PREPROCESSING ===")

# Cantidad de neuronas finales (excluyendo las 2 neuronas de contexto)
n_neuronas_finales = X.shape[2] - 2
print(f"Cantidad de neuronas finales: {n_neuronas_finales}")

# Cantidad de trials finales
trials_finales = np.unique(trial_ids)
n_trials_finales = len(trials_finales)
print(f"Cantidad de trials finales: {n_trials_finales}")

# Cantidad de time bins finales
n_time_bins_finales = X.shape[0]
tiempo_total_segundos = n_time_bins_finales * time_bins_size
tiempo_total_minutos = tiempo_total_segundos / 60
print(f"Cantidad de time bins finales: {n_time_bins_finales} = {tiempo_total_segundos:.2f} segundos = {tiempo_total_minutos:.2f} minutos")

# Calcular duración promedio de los trials finales
duraciones_trials_finales = []
for trial_id in trials_finales:
    trial_bins = np.where(trial_ids == trial_id)[0]
    duracion_bins = len(trial_bins)
    duracion_segundos = duracion_bins * time_bins_size
    duraciones_trials_finales.append(duracion_segundos)

duracion_promedio_trials = np.mean(duraciones_trials_finales)
print(f"Duración promedio de los trials finales: {duracion_promedio_trials:.2f} segundos")

# También mostrar estadísticas de duración
duracion_min_trials = np.min(duraciones_trials_finales)
duracion_max_trials = np.max(duraciones_trials_finales)
print(f"Duración mínima de trial: {duracion_min_trials:.2f} segundos")
print(f"Duración máxima de trial: {duracion_max_trials:.2f} segundos")


# Calcular trials recompensados y no recompensados finales
# Usar los trial_ids finales para obtener el contexto de cada trial
trials_rewCtxt_finales = []
for trial_id in trials_finales:
    # Obtener el último bin de cada trial para determinar su contexto
    trial_bins = np.where(trial_ids == trial_id)[0]
    if len(trial_bins) > 0:
        last_bin = trial_bins[-1]
        # Obtener el contexto del trial original (antes del preprocessing)
        # Necesitamos mapear de vuelta al trial original
        original_trial_index = int(trial_id) - 1  # trial_id empieza en 1
        if original_trial_index < len(trialFinalBin):
            context_value = rewCtxt[trialFinalBin[original_trial_index] - 1]
            trials_rewCtxt_finales.append(context_value)

trials_rewCtxt_finales = np.array(trials_rewCtxt_finales)
n_trials_recompensados = np.sum(trials_rewCtxt_finales == 1)
n_trials_no_recompensados = np.sum(trials_rewCtxt_finales == 0)

print(f"Cantidad de trials recompensados finales: {n_trials_recompensados}")
print(f"Cantidad de trials no recompensados finales: {n_trials_no_recompensados}")

# Verificar que la suma coincida
print(f"Verificación: {n_trials_recompensados} + {n_trials_no_recompensados} = {n_trials_recompensados + n_trials_no_recompensados} (debería ser igual a {n_trials_finales})")



# Flatten X: Esto lo necesito para entrenar los no recurrentes
X_flat = X.reshape(X.shape[0], (X.shape[1] * X.shape[2]))


#%% 
# GUARDO LOS DATOS
# Guardo los datos en un archivo pickle
with open('processed-datasets/L5_bins200ms_withCtxt_preprocessed.pickle', 'wb') as f:
    pickle.dump((X, y, trial_ids), f)
    
# Guardo los datos en un archivo pickle
with open('processed-datasets/L5_bins200ms_withCtxt_preprocessed_flat.pickle', 'wb') as f:
    pickle.dump((X_flat, y, trial_ids), f)
    
print("Datos guardados correctamente")
# %%
