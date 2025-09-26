#%% #Analis pendiente obtenida de curve_fit(lineal,campo_m,magnetizacion_ua_m_filtrada)
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import os
from astropy.io import ascii
from astropy.table import Table, Column, MaskedColumn
from uncertainties import ufloat,unumpy
from glob import glob
import seaborn as sns
from scipy.stats import shapiro
#%% Pendientes 
pend_300=glob('300_segunda/**/pendientes.txt',recursive=True)
pend_300.sort(reverse=True)

pend_270=glob('270/**/pendientes.txt',recursive=True)
pend_270.sort(reverse=True)

# pend_239=glob('239_15_to_5/**/pendientes.txt',recursive=True)
# pend_239.sort(reverse=True)

# pend_212=glob('212_15_to_5/**/pendientes.txt',recursive=True)
# pend_212.sort(reverse=True)

# pend_135=glob('135_15_to_5/**/pendientes.txt',recursive=True)
# pend_135.sort(reverse=True)

# pend_081=glob('081_15_to_5/**/pendientes.txt',recursive=True)
# pend_081.sort(reverse=True)

#%%
def leer_file_pendientes(archivo):
    data=np.loadtxt(archivo,skiprows=4)
    mean=np.mean(data[:-1])*1e14
    std=np.std(data[:-1])*1e14
    return ufloat(mean,std)
#%%
m_300 = [leer_file_pendientes(fpath) for fpath in pend_300]
m_270 = [leer_file_pendientes(fpath) for fpath in pend_270]
# m_239 = [leer_file_pendientes(fpath) for fpath in pend_239]
# m_212 = [leer_file_pendientes(fpath) for fpath in pend_212]
# m_135 = [leer_file_pendientes(fpath) for fpath in pend_135]
# m_081 = [leer_file_pendientes(fpath) for fpath in pend_081]

# Extraer solo los valores nominales (mean) para el heatmap
m_300_nominal = [val.n for val in m_300]
m_270_nominal = [val.n for val in m_270]
# m_239_nominal = [val.n for val in m_239]
# m_212_nominal = [val.n for val in m_212]
# m_135_nominal = [val.n for val in m_135]
# m_081_nominal = [val.n for val in m_081]

# Extraer solo las incertidumbres (std) 
m_300_err= [val.s for val in m_300]
m_270_err= [val.s for val in m_270]
# m_239_err= [val.s for val in m_239]
# m_212_err= [val.s for val in m_212]
# m_135_err= [val.s for val in m_135]
# m_081_err= [val.s for val in m_081]

#%%
m = [ m_270, m_300]
# Crear matriz para el heatmap (usando valores nominales)
m_nominal = np.array([ m_270_nominal, m_300_nominal])
m_err = [m_270_err, m_300_err]

frecuencias = [270, 300]  # kHz
H0 = [20, 24, 27, 31, 35, 38, 42, 46, 50, 53, 57]  # amplitud de campo

# Crear figura y ejes
plt.figure(figsize=(12, 6),constrained_layout=True)

# Crear heatmap con valores nominales
heatmap = sns.heatmap(
    m_nominal,
    xticklabels=H0,
    yticklabels=frecuencias,
    annot=m,  # Muestra los valores en las celdas
    fmt='.1uS',   # Formato de 3 decimales
    cmap='viridis',
    cbar_kws={'label': 'Pendiente m (x10^14) [Vs/A/m]'},
    linewidths=0.5,
    linecolor='gray'
)

# Configurar etiquetas y título
plt.xlabel('H$_0$ [kA/m]', fontsize=12, fontweight='bold')
plt.ylabel('Frecuencia [kHz]', fontsize=12, fontweight='bold')
plt.title('Heatmap de Pendiente m vs Frecuencia y Amplitud de campo H$_0$', fontsize=14, fontweight='bold')

# Rotar las etiquetas para mejor legibilidad
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.savefig('heatmap_pendiente_m_vs_frecuencia_amplitud_H0.png', dpi=300)
plt.show()
#%% Veo de corregir pendientes por inclinación del fondo
# def corregir_por_inclinacion(pendientes, devolver_pendientes_corregidas=True, 
#                              threshold_linear=1e-6, use_longdouble=True):
#     """
#     pendientes : iterable de floats (último elemento = referencia)
#     devolver_pendientes_corregidas : si True devuelve pendientes corregidas (tan(delta_theta)),
#                                      si False devuelve solo media y std de delta_theta (radianes)
#     threshold_linear : si |m| < threshold_linear se usa la aproximación lineal arctan(m) ~ m
#     use_longdouble : usar mayor precisión (np.longdouble) si está disponible
#     """
#     dtype = np.longdouble if use_longdouble else float
#     arr = np.array(pendientes, dtype=dtype)
#     m_ref = arr[-1]
#     ms = arr[:-1]

#     # criterio: si ambos (max absoluto) son muy pequeños, usar aproximación lineal
#     if np.max(np.abs(arr)) < threshold_linear:
#         # small-angle approx: delta_theta ≈ m - m_ref
#         delta_theta = ms - m_ref
#     else:
#         # fórmula exacta para diferencia de arctan:
#         # delta_theta = arctan((m_i - m_ref)/(1 + m_i*m_ref))
#         numer = ms - m_ref
#         denom = 1 + ms * m_ref
#         ratio = numer / denom
#         delta_theta = np.arctan(ratio)

#     # estadísticas sobre delta_theta
#     # por defecto devolvemos media/std de las PENDIENTES CORREGIDAS;
#     # pero puedes querer media/std de ángulos (radianes). Aquí calculamos ambas opciones.
#     media_delta = np.mean(delta_theta)
#     std_delta = np.std(delta_theta, ddof=1)

#     if devolver_pendientes_corregidas:
#         # convertir de nuevo a pendiente corregida: m_corr = tan(delta_theta)
#         pendientes_corr = np.tan(delta_theta)
#         media_m = np.mean(pendientes_corr)
#         std_m = np.std(pendientes_corr, ddof=1)
#         return {
#             "pendientes_corregidas": pendientes_corr,
#             "media_pendientes_corregidas": media_m,
#             "std_pendientes_corregidas": std_m,
#             "media_delta_theta_rad": media_delta,
#             "std_delta_theta_rad": std_delta
#         }
#     else:
#         return {
#             "media_delta_theta_rad": media_delta,
#             "std_delta_theta_rad": std_delta,
#             "delta_theta_array": delta_theta
#         }

# pendientes = [
#     1.164212e-13, 1.160360e-13, 1.177865e-13, 1.158811e-13,
#     1.170346e-13, 1.165802e-13, 1.162353e-13, 1.161804e-13,
#     1.169756e-13, 1.169290e-13, 1.173470e-13, 1.160335e-13,
#     1.163908e-13, 1.161268e-13, 1.165103e-13, 1.165971e-13,
#     1.161385e-13, 1.152694e-13, 1.168027e-13, 1.169992e-13,
#     1.157680e-13, 1.162035e-13, 2.218629e-14
# ]

# res = corregir_por_inclinacion(pendientes)
# for k,v in res.items():
#     print(k, ":", v)

# #%%
# import numpy as np
# import matplotlib.pyplot as plt

# # Tus pendientes
# pendientes = np.array([
#     1.164212e-13, 1.160360e-13, 1.177865e-13, 1.158811e-13,
#     1.170346e-13, 1.165802e-13, 1.162353e-13, 1.161804e-13,
#     1.169756e-13, 1.169290e-13, 1.173470e-13, 1.160335e-13,
#     1.163908e-13, 1.161268e-13, 1.165103e-13, 1.165971e-13,
#     1.161385e-13, 1.152694e-13, 1.168027e-13, 1.169992e-13,
#     1.157680e-13, 1.162035e-13, 2.218629e-14
# ])

# m_ref = pendientes[-1]
# ms = pendientes[:-1]

# # Rango de campo
# x = np.linspace(-24e3, 24e3, 200)  # -24 a 24 kA/m

# # Loop de gráficas
# for i, m in enumerate(ms, start=1):
#     # calcular pendiente corregida restando ángulos
#     delta_theta = np.arctan(m) - np.arctan(m_ref)
#     m_corr = np.tan(delta_theta)

#     # y-values
#     y_orig = m * x
#     y_ref = m_ref * x
#     y_corr = m_corr * x

#     # plot
#     plt.figure(figsize=(6,4))
#     plt.plot(x, y_orig, label=f'Original m{i}', color='blue')
#     plt.plot(x, y_ref, label='Referencia', color='red', linestyle='--')
#     plt.plot(x, y_corr, label='Corregida', color='green')
#     plt.xlabel('Campo (A/m)')
#     plt.ylabel('Respuesta (u.a.)')
#     plt.title(f'Comparación pendiente {i}')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

#%% Amplitudes de señal
amp_300=glob('300/**/amplitudes.txt',recursive=True)
amp_300.sort(reverse=True)
amp_270=glob('270/**/amplitudes.txt',recursive=True)
amp_270.sort(reverse=True)
# amp_239=glob('239_15_to_5/**/amplitudes.txt',recursive=True)
# amp_239.sort(reverse=True)
# amp_212=glob('212_15_to_5/**/amplitudes.txt',recursive=True)
# amp_212.sort(reverse=True)
# amp_135=glob('135_15_to_5/**/amplitudes.txt',recursive=True)
# amp_135.sort(reverse=True)
# amp_081=glob('081_15_to_5/**/amplitudes.txt',recursive=True)
# amp_081.sort(reverse=True)

def leer_file_amplitudes(archivo):
    data=np.loadtxt(archivo,skiprows=2)
    mean=np.mean(data[:-1])
    std=np.std(data[:-1])
    return ufloat(mean,std)

a_300 = [leer_file_amplitudes(fpath) for fpath in amp_300]
a_270 = [leer_file_amplitudes(fpath) for fpath in amp_270]
# a_239 = [leer_file_amplitudes(fpath) for fpath in amp_239]
# a_212 = [leer_file_amplitudes(fpath) for fpath in amp_212]
# a_135 = [leer_file_amplitudes(fpath) for fpath in amp_135]
#a_081 = [leer_file_amplitudes(fpath) for fpath in amp_081]  

# Extraer solo los valores nominales (mean) para el heatmap
a_300_nominal = [val.n for val in a_300]
a_270_nominal = [val.n for val in a_270]
# a_239_nominal = [val.n for val in a_239]
# a_212_nominal = [val.n for val in a_212]
# a_135_nominal = [val.n for val in a_135]
#a_081_nominal = [val.n for val in a_081]        

a = [ a_270, a_300]
# Crear matriz para el heatmap (usando valores nominales)
a_nominal = np.array([ a_270_nominal, a_300_nominal])

# Crear figura y ejes
plt.figure(figsize=(12, 6),constrained_layout=True)

# Crear heatmap con valores nominales
heatmap = sns.heatmap(
    a_nominal,
    xticklabels=H0,
    yticklabels=frecuencias,
    annot=a,  # Muestra los valores en las celdas
    fmt='.1uS',   # Formato de 3 decimales
    cmap='viridis',
    cbar_kws={'label': 'Amplitud de señal [mV]'},
    linewidths=0.5,
    linecolor='gray'
)

# Configurar etiquetas y título
plt.xlabel('H$_0$ [kA/m]', fontsize=12, fontweight='bold')
plt.ylabel('Frecuencia [kHz]', fontsize=12, fontweight='bold')
plt.title('Heatmap de Amplitud señal vs Frecuencia y Amplitud de campo H$_0$', fontsize=14, fontweight='bold')

# Rotar las etiquetas para mejor legibilidad
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.savefig('heatmap_amplitud_señal_vs_frecuencia_amplitud_H0.png', dpi=300)
plt.show()

# =============================================================================
#%%ANÁLISIS ESTADÍSTICO DE CORRELACIONES - SOLO PENDIENTES
# =============================================================================
print("="*60)
print("ANÁLISIS ESTADÍSTICO DE CORRELACIONES - PENDIENTES")
print("="*60)

import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from uncertainties import ufloat, unumpy
import uncertainties.umath as umath

#%% 1. CARGA DE DATOS PARA PENDIENTES CON INCERTEZAS
print("\n1. CARGA DE DATOS PENDIENTES CON INCERTEZAS")
print("-"*40)

# Crear arrays con valores nominales e incertezas
pendientes_ufloat = []
for i, freq in enumerate(frecuencias):
    for j, h0_val in enumerate(H0):
        # Crear objetos ufloat para cada medida
        pendiente_u = ufloat(m_nominal[i, j], m_err[i][j])
        pendientes_ufloat.append(pendiente_u)

# Crear DataFrame
corr_data_m = []
for i, freq in enumerate(frecuencias):
    for j, h0_val in enumerate(H0):
        corr_data_m.append({
            'Frecuencia_kHz': freq,
            'H0_kA_m': h0_val,
            'Pendiente': m_nominal[i, j],
            'Pendiente_err': m_err[i][j],
            'Pendiente_ufloat': ufloat(m_nominal[i, j], m_err[i][j])
        })

df_pendientes = pd.DataFrame(corr_data_m)

print(f"Dataset creado con {len(df_pendientes)} observaciones")
print("Primeras 5 filas:")
print(df_pendientes.head())

#%% 2. ANÁLISIS DESCRIPTIVO CON INCERTEZAS
print("\n\n2. ESTADÍSTICAS DESCRIPTIVAS CON INCERTEZAS")
print("-"*40)

# Función para calcular estadísticas con incertezas
def weighted_stats(values, errors):
    """Calcula estadísticas ponderadas por incertezas"""
    values = np.array(values)
    errors = np.array(errors)
    
    # Ponderación por inverso de la varianza
    weights = 1.0 / (errors**2)
    total_weight = np.sum(weights)
    
    # Media ponderada
    weighted_mean = np.sum(weights * values) / total_weight
    
    # Error de la media ponderada
    weighted_mean_err = 1.0 / np.sqrt(total_weight)
    
    # Varianza ponderada
    weighted_variance = np.sum(weights * (values - weighted_mean)**2) / total_weight
    weighted_std = np.sqrt(weighted_variance)
    
    return weighted_mean, weighted_mean_err, weighted_std

# Estadísticas generales considerando incertezas
print("ESTADÍSTICAS GENERALES CON INCERTEZAS:")
print("="*40)

weighted_mean, weighted_mean_err, weighted_std = weighted_stats(df_pendientes['Pendiente'], 
                                                                df_pendientes['Pendiente_err'])

print(f"Media ponderada: {weighted_mean:.3f} ± {weighted_mean_err:.3f}")
print(f"Desviación estándar ponderada: {weighted_std:.3f}")
print(f"Mínimo: {df_pendientes['Pendiente'].min():.3f}")
print(f"Máximo: {df_pendientes['Pendiente'].max():.3f}")
print(f"Rango: {df_pendientes['Pendiente'].max() - df_pendientes['Pendiente'].min():.3f}")

# Media no ponderada para comparación
simple_mean = df_pendientes['Pendiente'].mean()
simple_std = df_pendientes['Pendiente'].std()
print(f"\nMedia simple (sin ponderar): {simple_mean:.3e} ± {simple_std/np.sqrt(len(df_pendientes)):.3e}")
print(f"Diferencia relativa: {abs(weighted_mean - simple_mean)/simple_mean*100:.2f}%")



#%%
# Bien como interpreto que es la media ponderada? y la media simple?
# 2. ESTADÍSTICAS DESCRIPTIVAS CON INCERTEZAS
# ----------------------------------------
# ESTADÍSTICAS GENERALES CON INCERTEZAS:
# ========================================
# Media ponderada: 8.849 ± 0.009
# Desviación estándar ponderada: 0.584
# Mínimo: 8.110
# Máximo: 11.647
# Rango: 3.537

# Media simple (sin ponderar): 8.854e+00 ± 7.491e-02
# Diferencia relativa: 0.05%

# Con esa diferencia conviene tener en cuenta alguna de las 2 en los calculos y plost que siguen?
#%% 3. MATRIZ DE CORRELACIÓN CON INCERTEZAS
print("\n\n3. MATRIZ DE CORRELACIÓN CON INCERTEZAS")
print("-"*40)

from scipy.optimize import curve_fit
import numpy as np

# Función para correlación ponderada por incertezas
def weighted_correlation(x, y, y_err):
    """Calcula correlación considerando incertezas en y"""
    # Ajuste lineal ponderado
    def linear_func(x, a, b):
        return a * x + b
    
    # Ponderación por inverso de la varianza
    weights = 1.0 / (y_err**2)
    
    try:
        popt, pcov = curve_fit(linear_func, x, y, sigma=y_err, absolute_sigma=True)
        a, b = popt
        # Coeficiente de correlación a partir de la pendiente
        r = a * np.std(x) / np.std(y)
        return r, pcov[0, 0]  # Retorna correlación y varianza de la pendiente
    except:
        return np.nan, np.nan

# Función para bootstrap con incertezas
def bootstrap_correlation_with_errors(x, y, y_err, n_bootstrap=1000):
    """Calcula correlación con bootstrap considerando incertezas"""
    corr_samples = []
    n = len(x)
    
    for _ in range(n_bootstrap):
        # Muestrear considerando incertezas en y
        y_sample = np.random.normal(y, y_err)
        # Calcular correlación
        corr, _ = pearsonr(x, y_sample)
        corr_samples.append(corr)
    
    corr_mean = np.mean(corr_samples)
    corr_std = np.std(corr_samples)
    return corr_mean, corr_std

# Calcular correlaciones ponderadas
print("CORRELACIONES PONDERADAS POR INCERTEZAS:")
print("="*40)

# Correlación Pendiente vs Frecuencia (considerando incertezas)
r_freq_weighted, r_freq_var = weighted_correlation(
    df_pendientes['Frecuencia_kHz'], 
    df_pendientes['Pendiente'],
    df_pendientes['Pendiente_err'])

r_freq_bootstrap, r_freq_err = bootstrap_correlation_with_errors(
    df_pendientes['Frecuencia_kHz'],
    df_pendientes['Pendiente'],
    df_pendientes['Pendiente_err'],
    n_bootstrap=1000)

# Correlación Pendiente vs H0 (considerando incertezas)
r_h0_weighted, r_h0_var = weighted_correlation(
    df_pendientes['H0_kA_m'], 
    df_pendientes['Pendiente'],
    df_pendientes['Pendiente_err'])

r_h0_bootstrap, r_h0_err = bootstrap_correlation_with_errors(
    df_pendientes['H0_kA_m'],
    df_pendientes['Pendiente'],
    df_pendientes['Pendiente_err'],
    n_bootstrap=1000)

print("MÉTODO DE AJUSTE PONDERADO:")
print(f"Pendiente vs Frecuencia: r = {r_freq_weighted:.3f} ± {np.sqrt(r_freq_var):.3f}")
print(f"Pendiente vs H0: r = {r_h0_weighted:.3f} ± {np.sqrt(r_h0_var):.3f}")

print("\nMÉTODO BOOTSTRAP:")
print(f"Pendiente vs Frecuencia: r = {r_freq_bootstrap:.3f} ± {r_freq_err:.3f}")
print(f"Pendiente vs H0: r = {r_h0_bootstrap:.3f} ± {r_h0_err:.3f}")

# Correlaciones simples para comparación
print("\nCORRELACIONES SIMPLES (sin considerar incertezas):")
corr_matrix_pearson = df_pendientes[['Pendiente', 'Frecuencia_kHz', 'H0_kA_m']].corr(method='pearson')
print("Matriz de correlación de Pearson:")
print(corr_matrix_pearson.round(3))


#%% 4. REGRESIÓN PONDERADA POR INCERTEZAS
print("\n\n4. REGRESIÓN LINEAL PONDERADA")
print("-"*40)

# Regresión ponderada por incertezas
def weighted_linear_regression(x, y, y_err):
    """Regresión lineal ponderada por incertezas"""
    X = sm.add_constant(x)
    weights = 1.0 / (y_err**2)
    
    model = sm.WLS(y, X, weights=weights)
    results = model.fit()
    return results

# Regresión Pendiente vs Frecuencia (ponderada)
print("REGRESIÓN PONDERADA - Pendiente vs Frecuencia:")
model_freq_weighted = weighted_linear_regression(
    df_pendientes['Frecuencia_kHz'],
    df_pendientes['Pendiente'],
    df_pendientes['Pendiente_err']
)
print(model_freq_weighted.summary())

# Regresión Pendiente vs H0 (ponderada)
print("\nREGRESIÓN PONDERADA - Pendiente vs H0:")
model_h0_weighted = weighted_linear_regression(
    df_pendientes['H0_kA_m'],
    df_pendientes['Pendiente'],
    df_pendientes['Pendiente_err']
)
print(model_h0_weighted.summary())

#%% 5. REGRESIÓN MÚLTIPLE PONDERADA
print("\n\n5. REGRESIÓN MÚLTIPLE PONDERADA")
print("-"*40)

# Preparar variables para regresión múltiple ponderada
X = df_pendientes[['Frecuencia_kHz', 'H0_kA_m']]
X = sm.add_constant(X)
y = df_pendientes['Pendiente']
weights = 1.0 / (df_pendientes['Pendiente_err']**2)

# Modelo de regresión lineal múltiple ponderada
modelo_weighted = sm.WLS(y, X, weights=weights).fit()

print("RESUMEN DEL MODELO DE REGRESIÓN PONDERADA:")
print("="*50)
print(modelo_weighted.summary())

# Comparar con modelo no ponderado
modelo_simple = sm.OLS(y, X).fit()
print(f"\nCOMPARACIÓN DE MODELOS:")
print(f"R² ponderado: {modelo_weighted.rsquared:.3f}")
print(f"R² simple: {modelo_simple.rsquared:.3f}")
print(f"Diferencia: {modelo_weighted.rsquared - modelo_simple.rsquared:.3f}")

#%% 6. ANÁLISIS DE RESIDUOS PONDERADOS
print("\n\n6. ANÁLISIS DE RESIDUOS PONDERADOS")
print("-"*40)

# Calcular residuos ponderados
residuals_weighted = modelo_weighted.resid
residuals_standardized = residuals_weighted / df_pendientes['Pendiente_err']

print("ESTADÍSTICAS DE RESIDUOS PONDERADOS:")
print(f"Media de residuos: {np.mean(residuals_weighted):.3e}")
print(f"Desviación estándar de residuos: {np.std(residuals_weighted):.3e}")
print(f"Residuos estandarizados (deberían ser ~N(0,1)):")
print(f"  Media: {np.mean(residuals_standardized):.3f}")
print(f"  Desviación: {np.std(residuals_standardized):.3f}")

# Test de normalidad de residuos
_, p_residuals = shapiro(residuals_standardized)
print(f"Normalidad de residuos (Shapiro-Wilk): p = {p_residuals:.3e}")
print(f"Interpretación: {'Normal' if p_residuals > 0.05 else 'No normal'}")

#%% 7. COMPARACIÓN FINAL DE CORRELACIONES
print("\n\n7. COMPARACIÓN FINAL - CON Y SIN INCERTEZAS")
print("-"*40)

# Obtener correlaciones simples para comparación
corr_freq_simple, p_freq_simple = pearsonr(df_pendientes['Frecuencia_kHz'], df_pendientes['Pendiente'])
corr_h0_simple, p_h0_simple = pearsonr(df_pendientes['H0_kA_m'], df_pendientes['Pendiente'])

print("COMPARACIÓN DE CORRELACIONES:")
print("="*30)
print("PENDIENTE vs FRECUENCIA:")
print(f"  Simple:     r = {corr_freq_simple:.3f}, p = {p_freq_simple:.3e}")
print(f"  Ponderada:  r = {r_freq_bootstrap:.3f} ± {r_freq_err:.3f}")
print(f"  Diferencia: {abs(corr_freq_simple - r_freq_bootstrap):.3f}")

print("\nPENDIENTE vs H0:")
print(f"  Simple:     r = {corr_h0_simple:.3f}, p = {p_h0_simple:.3e}")
print(f"  Ponderada:  r = {r_h0_bootstrap:.3f} ± {r_h0_err:.3f}")
print(f"  Diferencia: {abs(corr_h0_simple - r_h0_bootstrap):.3f}")

# Interpretación de cambios
print(f"\nINTERPRETACIÓN:")
if abs(corr_freq_simple - r_freq_bootstrap) < 0.05:
    print("✓ Las incertezas no afectan significativamente la correlación con Frecuencia")
else:
    print("⚠️ Las incertezas afectan la correlación con Frecuencia")

if abs(corr_h0_simple - r_h0_bootstrap) < 0.05:
    print("✓ Las incertezas no afectan significativamente la correlación con H0")
else:
    print("⚠️ Las incertezas afectan la correlación con H0")

#%% 8. VISUALIZACIÓN DE CORRELACIONES PONDERADAS
print("\n\n8. GRÁFICO DE CORRELACIONES PONDERADAS")
print("-"*40)

plt.figure(figsize=(12, 5))

# Correlación con Frecuencia
plt.subplot(1, 2, 1)
plt.errorbar(df_pendientes['Frecuencia_kHz'], df_pendientes['Pendiente'],
             yerr=df_pendientes['Pendiente_err'], fmt='o', alpha=0.6,
             capsize=3, label='Datos con incertezas')
plt.xlabel('Frecuencia (kHz)')
plt.ylabel('Pendiente')
plt.title(f'Pendiente vs Frecuencia\nr = {r_freq_bootstrap:.3f} ± {r_freq_err:.3f}')
plt.grid(True, alpha=0.3)

# Correlación con H0
plt.subplot(1, 2, 2)
plt.errorbar(df_pendientes['H0_kA_m'], df_pendientes['Pendiente'],
             yerr=df_pendientes['Pendiente_err'], fmt='o', alpha=0.6,
             capsize=3, label='Datos con incertezas')
plt.xlabel('Campo H0 (kA/m)')
plt.ylabel('Pendiente')
plt.title(f'Pendiente vs H0\nr = {r_h0_bootstrap:.3f} ± {r_h0_err:.3f}')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
#% 8. VISUALIZACIÓN DE CORRELACIONES
print("\n\n8. VISUALIZACIÓN GRÁFICA")
print("-"*40)

# Crear figura con subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10),constrained_layout=True)
fig.suptitle('Análisis de Correlaciones - Pendientes', fontsize=16, fontweight='bold')

# Scatter plot Pendiente vs Frecuencia (coloreado por H0)
scatter1 = axes[0, 0].scatter(df_pendientes['Frecuencia_kHz'], df_pendientes['Pendiente'], 
                             c=df_pendientes['H0_kA_m'], cmap='viridis', alpha=0.7)
axes[0, 0].set_xlabel('Frecuencia (kHz)')
axes[0, 0].set_ylabel('Pendiente')
axes[0, 0].set_title('Pendiente vs Frecuencia (coloreado por H0)')
plt.colorbar(scatter1, ax=axes[0, 0], label='H0 (kA/m)')
axes[0, 0].grid(True, alpha=0.3)

# Scatter plot Pendiente vs H0 (coloreado por Frecuencia)
scatter2 = axes[0, 1].scatter(df_pendientes['H0_kA_m'], df_pendientes['Pendiente'], 
                             c=df_pendientes['Frecuencia_kHz'], cmap='plasma', alpha=0.7)
axes[0, 1].set_xlabel('Campo H0 (kA/m)')
axes[0, 1].set_ylabel('Pendiente')
axes[0, 1].set_title('Pendiente vs H0 (coloreado por Frecuencia)')
plt.colorbar(scatter2, ax=axes[0, 1], label='Frecuencia (kHz)')
axes[0, 1].grid(True, alpha=0.3)

# Heatmap de correlación
im = axes[1, 0].imshow(corr_matrix_pearson.values, cmap='coolwarm', vmin=-1, vmax=1)
axes[1, 0].set_xticks(range(len(corr_matrix_pearson.columns)))
axes[1, 0].set_yticks(range(len(corr_matrix_pearson.columns)))
axes[1, 0].set_xticklabels(corr_matrix_pearson.columns, rotation=45)
axes[1, 0].set_yticklabels(corr_matrix_pearson.columns)
axes[1, 0].set_title('Matriz de Correlación (Pearson)')

# Añadir valores numéricos al heatmap
for i in range(len(corr_matrix_pearson.columns)):
    for j in range(len(corr_matrix_pearson.columns)):
        axes[1, 0].text(j, i, f'{corr_matrix_pearson.iloc[i, j]:.2f}', 
                       ha='center', va='center', fontsize=10, 
                       color='white' if abs(corr_matrix_pearson.iloc[i, j]) > 0.5 else 'black')

plt.colorbar(im, ax=axes[1, 0])

# Gráfico de residuos del modelo
axes[1, 1].scatter(modelo_weighted.fittedvalues, modelo_weighted.resid, alpha=0.7)
axes[1, 1].axhline(y=0, color='red', linestyle='--')
axes[1, 1].set_xlabel('Valores Ajustados')
axes[1, 1].set_ylabel('Residuos')
axes[1, 1].set_title('Análisis de Residuos del Modelo Ponderado')
axes[1, 1].grid(True, alpha=0.3)
plt.savefig('analisis_correlaciones_pendientes.png', dpi=300)
plt.show()

#%% 9. EXPORTACIÓN DE RESULTADOS
print("\n\n9. EXPORTACIÓN DE RESULTADOS")
print("-"*40)

# Primero calcular las correlaciones simples si no están definidas
if 'corr_freq_pearson' not in locals():
    corr_freq_pearson, p_freq_pearson = pearsonr(df_pendientes['Frecuencia_kHz'], df_pendientes['Pendiente'])
    corr_freq_spearman, p_freq_spearman = spearmanr(df_pendientes['Frecuencia_kHz'], df_pendientes['Pendiente'])
    corr_h0_pearson, p_h0_pearson = pearsonr(df_pendientes['H0_kA_m'], df_pendientes['Pendiente'])
    corr_h0_spearman, p_h0_spearman = spearmanr(df_pendientes['H0_kA_m'], df_pendientes['Pendiente'])

# Asegurarse de que el modelo simple está definido
if 'modelo_simple' not in locals():
    X_simple = df_pendientes[['Frecuencia_kHz', 'H0_kA_m']]
    X_simple = sm.add_constant(X_simple)
    y_simple = df_pendientes['Pendiente']
    modelo_simple = sm.OLS(y_simple, X_simple).fit()

# Calcular coeficientes estandarizados si no están definidos
if 'coef_std' not in locals():
    coef_std = modelo_simple.params[1:] / df_pendientes['Pendiente'].std()

# Guardar datos
df_pendientes.to_csv('analisis_pendientes_completo.csv', index=False)
print("Datos de pendientes guardados en 'analisis_pendientes_completo.csv'")

# Guardar resumen estadístico
with open('resumen_analisis_pendientes.txt', 'w') as f:
    f.write("RESUMEN DE ANÁLISIS ESTADÍSTICO - PENDIENTES\n")
    f.write("="*60 + "\n\n")
    
    f.write("CORRELACIONES SIMPLES:\n")
    f.write(f"Pendiente vs Frecuencia (Pearson): r = {corr_freq_pearson:.3f}, p = {p_freq_pearson:.3e}\n")
    f.write(f"Pendiente vs Frecuencia (Spearman): rho = {corr_freq_spearman:.3f}, p = {p_freq_spearman:.3e}\n")
    f.write(f"Pendiente vs H0 (Pearson): r = {corr_h0_pearson:.3f}, p = {p_h0_pearson:.3e}\n")
    f.write(f"Pendiente vs H0 (Spearman): rho = {corr_h0_spearman:.3f}, p = {p_h0_spearman:.3e}\n\n")
    
    f.write("CORRELACIONES PONDERADAS (Bootstrap):\n")
    f.write(f"Pendiente vs Frecuencia: r = {r_freq_bootstrap:.3f} ± {r_freq_err:.3f}\n")
    f.write(f"Pendiente vs H0: r = {r_h0_bootstrap:.3f} ± {r_h0_err:.3f}\n\n")
    
    f.write("REGRESIÓN MÚLTIPLE SIMPLE:\n")
    f.write(f"R² = {modelo_simple.rsquared:.3f}\n")
    f.write(f"R² ajustado = {modelo_simple.rsquared_adj:.3f}\n")
    f.write(f"Intercepto = {modelo_simple.params['const']:.3e}\n")
    f.write(f"Coef. Frecuencia = {modelo_simple.params['Frecuencia_kHz']:.3e}\n")
    f.write(f"Coef. H0 = {modelo_simple.params['H0_kA_m']:.3e}\n\n")
    
    f.write("REGRESIÓN MÚLTIPLE PONDERADA:\n")
    f.write(f"R² = {modelo_weighted.rsquared:.3f}\n")
    f.write(f"R² ajustado = {modelo_weighted.rsquared_adj:.3f}\n")
    f.write(f"Intercepto = {modelo_weighted.params['const']:.3e}\n")
    f.write(f"Coef. Frecuencia = {modelo_weighted.params['Frecuencia_kHz']:.3e}\n")
    f.write(f"Coef. H0 = {modelo_weighted.params['H0_kA_m']:.3e}\n\n")
    
    f.write("ESTADÍSTICAS DESCRIPTIVAS:\n")
    f.write(f"Media ponderada: {weighted_mean:.3f} ± {weighted_mean_err:.3f}\n")
    f.write(f"Media simple: {simple_mean:.3f} ± {simple_std/np.sqrt(len(df_pendientes)):.3f}\n")
    f.write(f"Diferencia relativa: {abs(weighted_mean - simple_mean)/simple_mean*100:.2f}%\n")

print("Resumen estadístico guardado en 'resumen_analisis_pendientes.txt'")

#%% 10. INTERPRETACIÓN FINAL
print("\n\n10. INTERPRETACIÓN FINAL")
print("-"*40)
print("RESUMEN EJECUTIVO:")
print(f"• La pendiente muestra una correlación {'positiva' if corr_freq_pearson > 0 else 'negativa'} ")
print(f"  con la frecuencia (r = {corr_freq_pearson:.3f})")
print(f"• La pendiente muestra una correlación {'positiva' if corr_h0_pearson > 0 else 'negativa'} ")
print(f"  con el campo H0 (r = {corr_h0_pearson:.3f})")
print(f"• El modelo de regresión explica el {modelo_simple.rsquared*100:.1f}% de la variabilidad")
print(f"• La variable más influyente es: {'Frecuencia' if abs(coef_std['Frecuencia_kHz']) > abs(coef_std['H0_kA_m']) else 'Campo H0'}")

# Añadir interpretación de las correlaciones ponderadas
print(f"\nCONSIDERANDO INCERTEZAS:")
print(f"• Correlación ponderada Frecuencia: r = {r_freq_bootstrap:.3f} ± {r_freq_err:.3f}")
print(f"• Correlación ponderada H0: r = {r_h0_bootstrap:.3f} ± {r_h0_err:.3f}")

#%% ANÁLISIS ESTADÍSTICO DE CORRELACIONES - AMPLITUD DE SEÑAL
# =============================================================================
# ANÁLISIS ESTADÍSTICO DE CORRELACIONES - AMPLITUD DE SEÑAL
# =============================================================================
print("="*60)
print("ANÁLISIS ESTADÍSTICO DE CORRELACIONES - AMPLITUD DE SEÑAL")
print("="*60)

#% 1. PREPARACIÓN DE DATOS PARA AMPLITUDES
print("\n1. PREPARACIÓN DE DATOS")
print("-"*40)

# Crear DataFrame para análisis de correlación de amplitudes
corr_data_a = []
for i, freq in enumerate(frecuencias):
    for j, h0_val in enumerate(H0):
        corr_data_a.append({
            'Frecuencia_kHz': freq,
            'H0_kA_m': h0_val,
            'Amplitud_mV': a_nominal[i, j]  # Cambiado a Amplitud_mV para claridad
        })

df_amplitudes = pd.DataFrame(corr_data_a)

print(f"Dataset creado con {len(df_amplitudes)} observaciones")
print("Primeras 5 filas:")
print(df_amplitudes.head())

#% 2. ANÁLISIS DESCRIPTIVO BÁSICO
print("\n\n2. ESTADÍSTICAS DESCRIPTIVAS")
print("-"*40)

print("Estadísticas generales de amplitudes:")
print(f"Media: {df_amplitudes['Amplitud_mV'].mean():.3f} mV")
print(f"Desviación estándar: {df_amplitudes['Amplitud_mV'].std():.3f} mV")
print(f"Mínimo: {df_amplitudes['Amplitud_mV'].min():.3f} mV")
print(f"Máximo: {df_amplitudes['Amplitud_mV'].max():.3f} mV")
print(f"Rango: {df_amplitudes['Amplitud_mV'].max() - df_amplitudes['Amplitud_mV'].min():.3f} mV")

print("\nEstadísticas por frecuencia:")
stats_freq = df_amplitudes.groupby('Frecuencia_kHz')['Amplitud_mV'].agg(['mean', 'std', 'count'])
print(stats_freq.round(3))

print("\nEstadísticas por campo H0:")
stats_h0 = df_amplitudes.groupby('H0_kA_m')['Amplitud_mV'].agg(['mean', 'std', 'count'])
print(stats_h0.round(3))

#% 3. MATRIZ DE CORRELACIÓN COMPLETA
print("\n\n3. MATRIZ DE CORRELACIÓN")
print("-"*40)

# Matriz de correlación con todos los métodos
corr_matrix_pearson = df_amplitudes[['Amplitud_mV', 'Frecuencia_kHz', 'H0_kA_m']].corr(method='pearson')
corr_matrix_spearman = df_amplitudes[['Amplitud_mV', 'Frecuencia_kHz', 'H0_kA_m']].corr(method='spearman')

print("Matriz de correlación de Pearson:")
print(corr_matrix_pearson.round(3))

print("\nMatriz de correlación de Spearman (no paramétrica):")
print(corr_matrix_spearman.round(3))

#% 4. CORRELACIONES INDIVIDUALES CON SIGNIFICANCIA
print("\n\n4. CORRELACIONES INDIVIDUALES")
print("-"*40)

# Correlación Amplitud vs Frecuencia
corr_freq_pearson, p_freq_pearson = pearsonr(df_amplitudes['Amplitud_mV'], df_amplitudes['Frecuencia_kHz'])
corr_freq_spearman, p_freq_spearman = spearmanr(df_amplitudes['Amplitud_mV'], df_amplitudes['Frecuencia_kHz'])

print("Amplitud vs Frecuencia:")
print(f"  Pearson: r = {corr_freq_pearson:.3f}, p = {p_freq_pearson:.3e}")
print(f"  Spearman: rho = {corr_freq_spearman:.3f}, p = {p_freq_spearman:.3e}")

# Correlación Amplitud vs H0
corr_h0_pearson, p_h0_pearson = pearsonr(df_amplitudes['Amplitud_mV'], df_amplitudes['H0_kA_m'])
corr_h0_spearman, p_h0_spearman = spearmanr(df_amplitudes['Amplitud_mV'], df_amplitudes['H0_kA_m'])

print("\nAmplitud vs Campo H0:")
print(f"  Pearson: r = {corr_h0_pearson:.3f}, p = {p_h0_pearson:.3e}")
print(f"  Spearman: rho = {corr_h0_spearman:.3f}, p = {p_h0_spearman:.3e}")

# Interpretación de significancia
def interpretar_significancia(p_valor):
    if p_valor < 0.001:
        return "*** Muy significativa"
    elif p_valor < 0.01:
        return "** Muy significativa"
    elif p_valor < 0.05:
        return "* Significativa"
    else:
        return "No significativa"

print(f"\nInterpretación Pearson Frecuencia: {interpretar_significancia(p_freq_pearson)}")
print(f"Interpretación Spearman Frecuencia: {interpretar_significancia(p_freq_spearman)}")
print(f"Interpretación Pearson H0: {interpretar_significancia(p_h0_pearson)}")
print(f"Interpretación Spearman H0: {interpretar_significancia(p_h0_spearman)}")

#% 5. ANÁLISIS DE REGRESIÓN MÚLTIPLE
print("\n\n5. REGRESIÓN MÚLTIPLE")
print("-"*40)

# Preparar variables para regresión
X = df_amplitudes[['Frecuencia_kHz', 'H0_kA_m']]
X = sm.add_constant(X)  # Agregar intercepto
y = df_amplitudes['Amplitud_mV']

# Modelo de regresión lineal múltiple
modelo = sm.OLS(y, X).fit()

print("RESUMEN DEL MODELO DE REGRESIÓN:")
print("="*50)
print(modelo.summary())

# Coeficientes estandarizados para comparar importancia
print("\nCOEFICIENTES ESTANDARIZADOS (importancia relativa):")
coef_std = modelo.params[1:] / y.std()  # Estandarizar coeficientes
print(f"Frecuencia: {coef_std['Frecuencia_kHz']:.3f}")
print(f"Campo H0: {coef_std['H0_kA_m']:.3f}")

#% 6. ANÁLISIS DE INTERACCIÓN ENTRE VARIABLES
print("\n\n6. ANÁLISIS DE INTERACCIÓN")
print("-"*40)

# Modelo con término de interacción
X_interaction = df_amplitudes[['Frecuencia_kHz', 'H0_kA_m']].copy()
X_interaction['Interaccion'] = X_interaction['Frecuencia_kHz'] * X_interaction['H0_kA_m']
X_interaction = sm.add_constant(X_interaction)

modelo_interaction = sm.OLS(y, X_interaction).fit()

print("MODELO CON INTERACCIÓN:")
print("="*30)
print(f"Coeficiente de interacción: {modelo_interaction.params['Interaccion']:.3e}")
print(f"p-valor interacción: {modelo_interaction.pvalues['Interaccion']:.3e}")
print(f"Significancia: {interpretar_significancia(modelo_interaction.pvalues['Interaccion'])}")

# Comparar modelos (con y sin interacción)
print(f"\nComparación de modelos (R²):")
print(f"Sin interacción: {modelo.rsquared:.3f}")
print(f"Con interacción: {modelo_interaction.rsquared:.3f}")
print(f"Mejora: {modelo_interaction.rsquared - modelo.rsquared:.3f}")

#% 7. ANÁLISIS DE GRADIENTES ESPACIALES
print("\n\n7. ANÁLISIS DE GRADIENTES ESPACIALES")
print("-"*40)

def analizar_gradientes(matriz, nombre):
    """Analiza gradientes en ambas dimensiones"""
    grad_x = np.gradient(matriz, axis=1)  # Dirección H0 (horizontal)
    grad_y = np.gradient(matriz, axis=0)  # Dirección Frecuencia (vertical)
    
    print(f"\nGRADIENTES - {nombre}:")
    print(f"Dirección H0 (horizontal):")
    print(f"  Media: {np.mean(grad_x):.3f} mV/kA·m ± {np.std(grad_x):.3f}")
    print(f"  Mínimo: {np.min(grad_x):.3f}, Máximo: {np.max(grad_x):.3f}")
    
    print(f"Dirección Frecuencia (vertical):")
    print(f"  Media: {np.mean(grad_y):.3f} mV/kHz ± {np.std(grad_y):.3f}")
    print(f"  Mínimo: {np.min(grad_y):.3f}, Máximo: {np.max(grad_y):.3f}")
    
    return grad_x, grad_y

grad_x_a, grad_y_a = analizar_gradientes(a_nominal, "Amplitudes")

#% 8. VISUALIZACIÓN DE CORRELACIONES
print("\n\n8. VISUALIZACIÓN GRÁFICA")
print("-"*40)

# Crear figura con subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10),constrained_layout=True)
fig.suptitle('Análisis de Correlaciones - Amplitud de Señal', fontsize=16, fontweight='bold')

# Scatter plot Amplitud vs Frecuencia (coloreado por H0)
scatter1 = axes[0, 0].scatter(df_amplitudes['Frecuencia_kHz'], df_amplitudes['Amplitud_mV'], 
                             c=df_amplitudes['H0_kA_m'], cmap='viridis', alpha=0.7)
axes[0, 0].set_xlabel('Frecuencia (kHz)')
axes[0, 0].set_ylabel('Amplitud (mV)')
axes[0, 0].set_title('Amplitud vs Frecuencia (coloreado por H0)')
plt.colorbar(scatter1, ax=axes[0, 0], label='H0 (kA/m)')
axes[0, 0].grid(True, alpha=0.3)

# Scatter plot Amplitud vs H0 (coloreado por Frecuencia)
scatter2 = axes[0, 1].scatter(df_amplitudes['H0_kA_m'], df_amplitudes['Amplitud_mV'], 
                             c=df_amplitudes['Frecuencia_kHz'], cmap='plasma', alpha=0.7)
axes[0, 1].set_xlabel('Campo H0 (kA/m)')
axes[0, 1].set_ylabel('Amplitud (mV)')
axes[0, 1].set_title('Amplitud vs H0 (coloreado por Frecuencia)')
plt.colorbar(scatter2, ax=axes[0, 1], label='Frecuencia (kHz)')
axes[0, 1].grid(True, alpha=0.3)

# Heatmap de correlación
im = axes[1, 0].imshow(corr_matrix_pearson.values, cmap='coolwarm', vmin=-1, vmax=1)
axes[1, 0].set_xticks(range(len(corr_matrix_pearson.columns)))
axes[1, 0].set_yticks(range(len(corr_matrix_pearson.columns)))
axes[1, 0].set_xticklabels(corr_matrix_pearson.columns, rotation=45)
axes[1, 0].set_yticklabels(corr_matrix_pearson.columns)
axes[1, 0].set_title('Matriz de Correlación (Pearson)')

# Añadir valores numéricos al heatmap
for i in range(len(corr_matrix_pearson.columns)):
    for j in range(len(corr_matrix_pearson.columns)):
        axes[1, 0].text(j, i, f'{corr_matrix_pearson.iloc[i, j]:.2f}', 
                       ha='center', va='center', fontsize=10, 
                       color='white' if abs(corr_matrix_pearson.iloc[i, j]) > 0.5 else 'black')

plt.colorbar(im, ax=axes[1, 0])

# Gráfico de residuos del modelo
axes[1, 1].scatter(modelo.fittedvalues, modelo.resid, alpha=0.7)
axes[1, 1].axhline(y=0, color='red', linestyle='--')
axes[1, 1].set_xlabel('Valores Ajustados (mV)')
axes[1, 1].set_ylabel('Residuos (mV)')
axes[1, 1].set_title('Análisis de Residuos del Modelo')
axes[1, 1].grid(True, alpha=0.3)

plt.savefig('analisis_correlaciones_amplitud.png', dpi=300)
plt.show()

#% 9. ANÁLISIS ADICIONAL - TENDENCIAS POR GRUPOS
print("\n\n9. ANÁLISIS DE TENDENCIAS POR GRUPOS")
print("-"*40)

# Análisis de tendencias lineales por frecuencia
print("Tendencias Amplitud vs H0 por frecuencia:")
for freq in frecuencias:
    subset = df_amplitudes[df_amplitudes['Frecuencia_kHz'] == freq]
    if len(subset) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(subset['H0_kA_m'], subset['Amplitud_mV'])
        print(f"  {freq} kHz: pendiente = {slope:.3f} mV/(kA/m), r = {r_value:.3f}, p = {p_value:.3e}")

print("\nTendencias Amplitud vs Frecuencia por campo H0:")
for h0_val in H0:
    subset = df_amplitudes[df_amplitudes['H0_kA_m'] == h0_val]
    if len(subset) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(subset['Frecuencia_kHz'], subset['Amplitud_mV'])
        print(f"  H0 = {h0_val} kA/m: pendiente = {slope:.3f} mV/kHz, r = {r_value:.3f}, p = {p_value:.3e}")

#% 10. EXPORTACIÓN DE RESULTADOS
print("\n\n10. EXPORTACIÓN DE RESULTADOS")
print("-"*40)

# Guardar datos
df_amplitudes.to_csv('analisis_amplitudes_completo.csv', index=False)
print("Datos de amplitudes guardados en 'analisis_amplitudes_completo.csv'")

# Guardar resumen estadístico
with open('resumen_analisis_amplitudes.txt', 'w') as f:
    f.write("RESUMEN DE ANÁLISIS ESTADÍSTICO - AMPLITUDES\n")
    f.write("="*60 + "\n\n")
    
    f.write("ESTADÍSTICAS DESCRIPTIVAS:\n")
    f.write(f"Media: {df_amplitudes['Amplitud_mV'].mean():.3f} mV\n")
    f.write(f"Desviación estándar: {df_amplitudes['Amplitud_mV'].std():.3f} mV\n")
    f.write(f"Rango: {df_amplitudes['Amplitud_mV'].max() - df_amplitudes['Amplitud_mV'].min():.3f} mV\n\n")
    
    f.write("CORRELACIONES:\n")
    f.write(f"Amplitud vs Frecuencia (Pearson): r = {corr_freq_pearson:.3f}, p = {p_freq_pearson:.3e}\n")
    f.write(f"Amplitud vs Frecuencia (Spearman): rho = {corr_freq_spearman:.3f}, p = {p_freq_spearman:.3e}\n")
    f.write(f"Amplitud vs H0 (Pearson): r = {corr_h0_pearson:.3f}, p = {p_h0_pearson:.3e}\n")
    f.write(f"Amplitud vs H0 (Spearman): rho = {corr_h0_spearman:.3f}, p = {p_h0_spearman:.3e}\n\n")
    
    f.write("REGRESIÓN MÚLTIPLE:\n")
    f.write(f"R² = {modelo.rsquared:.3f}\n")
    f.write(f"R² ajustado = {modelo.rsquared_adj:.3f}\n")
    f.write(f"Intercepto = {modelo.params['const']:.3f}\n")
    f.write(f"Coef. Frecuencia = {modelo.params['Frecuencia_kHz']:.3f}\n")
    f.write(f"Coef. H0 = {modelo.params['H0_kA_m']:.3f}\n\n")
    
    f.write("GRADIENTES ESPACIALES:\n")
    f.write(f"Gradiente H0: {np.mean(grad_x_a):.3f} ± {np.std(grad_x_a):.3f} mV/(kA/m)\n")
    f.write(f"Gradiente Frecuencia: {np.mean(grad_y_a):.3f} ± {np.std(grad_y_a):.3f} mV/kHz\n")

print("Resumen estadístico guardado en 'resumen_analisis_amplitudes.txt'")

#% 11. INTERPRETACIÓN FINAL
print("\n\n11. INTERPRETACIÓN FINAL")
print("-"*40)
print("RESUMEN EJECUTIVO:")
print(f"• La amplitud muestra una correlación {'positiva' if corr_freq_pearson > 0 else 'negativa'} ")
print(f"  con la frecuencia (r = {corr_freq_pearson:.3f})")
print(f"• La amplitud muestra una correlación {'positiva' if corr_h0_pearson > 0 else 'negativa'} ")
print(f"  con el campo H0 (r = {corr_h0_pearson:.3f})")
print(f"• El modelo de regresión explica el {modelo.rsquared*100:.1f}% de la variabilidad")
print(f"• La variable más influyente es: {'Frecuencia' if abs(coef_std['Frecuencia_kHz']) > abs(coef_std['H0_kA_m']) else 'Campo H0'}")
print(f"• La amplitud promedio es de {df_amplitudes['Amplitud_mV'].mean():.1f} mV")

print("\n¡Análisis de amplitudes completado exitosamente!")
# %% Histogramas 
# =============================================================================
# HISTOGRAMAS DE PENDIENTES AGRUPADOS POR FRECUENCIA Y CAMPO H0
# =============================================================================
print("="*60)
print("HISTOGRAMAS DE PENDIENTES - ANÁLISIS DE DISTRIBUCIÓN")
print("="*60)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Crear DataFrame con los datos de pendientes
df_pendientes = pd.DataFrame({
    'Frecuencia_kHz': np.repeat(frecuencias, len(H0)),
    'H0_kA_m': list(H0) * len(frecuencias),
    'Pendiente': m_nominal.flatten()
})

#% 1. HISTOGRAMAS AGRUPADOS POR FRECUENCIA
print("\n1. HISTOGRAMAS AGRUPADOS POR FRECUENCIA")
print("-"*40)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Distribución de Pendientes por Frecuencia', fontsize=16, fontweight='bold')

# Colores diferentes para cada frecuencia
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

for i, freq in enumerate(frecuencias):
    row = i // 3
    col = i % 3
    
    # Filtrar datos para esta frecuencia
    data = df_pendientes[df_pendientes['Frecuencia_kHz'] == freq]['Pendiente']
    
    # Crear histograma
    n, bins, patches = axes[row, col].hist(data, bins=8, alpha=0.7, color=colors[i], 
                                          edgecolor='black', linewidth=0.5)
    
    # Añadir líneas estadísticas
    axes[row, col].axvline(data.mean(), color='red', linestyle='--', linewidth=2, 
                          label=f'Media: {data.mean():.2f}')
    axes[row, col].axvline(data.median(), color='green', linestyle='--', linewidth=2, 
                          label=f'Mediana: {data.median():.2f}')
    
    # Configuración del subplot
    axes[row, col].set_title(f'{freq} kHz', fontsize=12, fontweight='bold')
    axes[row, col].set_xlabel('Pendiente')
    axes[row, col].set_ylabel('Frecuencia')
    axes[row, col].grid(True, alpha=0.3)
    axes[row, col].legend()
    
    # Añadir texto con estadísticas
    stats_text = f'n = {len(data)}\nσ = {data.std():.2f}'
    axes[row, col].text(0.95, 0.95, stats_text, transform=axes[row, col].transAxes,
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Ajustar layout
plt.tight_layout()
plt.show()

#% 2. HISTOGRAMAS AGRUPADOS POR CAMPO H0
print("\n2. HISTOGRAMAS AGRUPADOS POR CAMPO H0")
print("-"*40)

fig, axes = plt.subplots(4, 3, figsize=(15, 12))
fig.suptitle('Distribución de Pendientes por Campo H0', fontsize=16, fontweight='bold')

# Reorganizar axes para tener 4 filas y 3 columnas (11 valores de H0)
axes = axes.flatten()

# Colormap para los campos H0
cmap = plt.cm.viridis
norm = plt.Normalize(min(H0), max(H0))

for i, h0_val in enumerate(H0):
    # Filtrar datos para este campo H0
    data = df_pendientes[df_pendientes['H0_kA_m'] == h0_val]['Pendiente']
    
    # Crear histograma
    n, bins, patches = axes[i].hist(data, bins=6, alpha=0.7, 
                                   color=cmap(norm(h0_val)), 
                                   edgecolor='black', linewidth=0.5)
    
    # Añadir líneas estadísticas
    axes[i].axvline(data.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Media: {data.mean():.2f}')
    axes[i].axvline(data.median(), color='green', linestyle='--', linewidth=2, 
                   label=f'Mediana: {data.median():.2f}')
    
    # Configuración del subplot
    axes[i].set_title(f'H0 = {h0_val} kA/m', fontsize=10, fontweight='bold')
    axes[i].set_xlabel('Pendiente')
    axes[i].set_ylabel('Frecuencia')
    axes[i].grid(True, alpha=0.3)
    axes[i].legend(fontsize=8)
    
    # Añadir texto con estadísticas
    stats_text = f'n = {len(data)}\nσ = {data.std():.2f}'
    axes[i].text(0.95, 0.95, stats_text, transform=axes[i].transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=8)

# Ocultar el último subplot si no se usa
if len(H0) < len(axes):
    for i in range(len(H0), len(axes)):
        axes[i].set_visible(False)

# Añadir colorbar para los campos H0
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Campo H0 (kA/m)', fontsize=12)

plt.tight_layout(rect=[0, 0, 0.9, 0.96])
plt.show()

#% 3. ANÁLISIS COMPARATIVO - BOXPLOTS
print("\n3. ANÁLISIS COMPARATIVO - BOXPLOTS")
print("-"*40)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6),constrained_layout=True)
fig.suptitle('Análisis Comparativo de Distribuciones de Pendientes', fontsize=16, fontweight='bold')

# Boxplot por frecuencia
boxplot_data_freq = [df_pendientes[df_pendientes['Frecuencia_kHz'] == freq]['Pendiente'] 
                     for freq in frecuencias]

ax1.boxplot(boxplot_data_freq, labels=frecuencias)
ax1.set_title('Distribución por Frecuencia', fontsize=14)
ax1.set_xlabel('Frecuencia (kHz)', fontsize=12)
ax1.set_ylabel('Pendiente', fontsize=12)

# Boxplot por campo H0
boxplot_data_h0 = [df_pendientes[df_pendientes['H0_kA_m'] == h0_val]['Pendiente'] 
                   for h0_val in H0]

ax2.boxplot(boxplot_data_h0, labels=H0)
ax2.set_title('Distribución por Campo H0', fontsize=14)
ax2.set_xlabel('Campo H0 (kA/m)', fontsize=12)
ax2.set_ylabel('Pendiente', fontsize=12)
ax2.tick_params(axis='x', rotation=45)
for a in [ax1, ax2]:
    a.legend() 
    a.grid(True, alpha=0.3)

plt.show()

#% 4. ESTADÍSTICAS DESCRIPTIVAS POR GRUPOS
print("\n4. ESTADÍSTICAS DESCRIPTIVAS POR GRUPOS")
print("-"*40)

print("ESTADÍSTICAS POR FRECUENCIA:")
print("="*30)
stats_freq = df_pendientes.groupby('Frecuencia_kHz')['Pendiente'].agg([
    'count', 'mean', 'std', 'min', 'max', 'median'
]).round(3)
print(stats_freq)

print("\nESTADÍSTICAS POR CAMPO H0:")
print("="*30)
stats_h0 = df_pendientes.groupby('H0_kA_m')['Pendiente'].agg([
    'count', 'mean', 'std', 'min', 'max', 'median'
]).round(3)
print(stats_h0)

#% 5. TEST DE NORMALIDAD POR GRUPOS
print("\n5. TEST DE NORMALIDAD (Shapiro-Wilk)")
print("-"*40)

from scipy.stats import shapiro

print("Normalidad por Frecuencia:")
for freq in frecuencias:
    data = df_pendientes[df_pendientes['Frecuencia_kHz'] == freq]['Pendiente']
    stat, p_value = shapiro(data)
    normal = "✓ Normal" if p_value > 0.05 else "✗ No normal"
    print(f"  {freq} kHz: p = {p_value:.3e} -> {normal}")

print("\nNormalidad por Campo H0:")
for h0_val in H0:
    data = df_pendientes[df_pendientes['H0_kA_m'] == h0_val]['Pendiente']
    stat, p_value = shapiro(data)
    normal = "✓ Normal" if p_value > 0.05 else "✗ No normal"
    print(f"  H0 = {h0_val} kA/m: p = {p_value:.3e} -> {normal}")

#% 6. GRÁFICO DE TENDENCIAS PROMEDIO
print("\n6. TENDENCIAS PROMEDIO")
print("-"*40)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Tendencia promedio por frecuencia
mean_by_freq = df_pendientes.groupby('Frecuencia_kHz')['Pendiente'].mean()
std_by_freq = df_pendientes.groupby('Frecuencia_kHz')['Pendiente'].std()

ax1.errorbar(frecuencias, mean_by_freq, yerr=std_by_freq, 
            fmt='o-', capsize=5, capthick=2, linewidth=2)
ax1.set_title('Pendiente Promedio vs Frecuencia', fontsize=14)
ax1.set_xlabel('Frecuencia (kHz)', fontsize=12)
ax1.set_ylabel('Pendiente Promedio', fontsize=12)
ax1.grid(True, alpha=0.3)

# Tendencia promedio por campo H0
mean_by_h0 = df_pendientes.groupby('H0_kA_m')['Pendiente'].mean()
std_by_h0 = df_pendientes.groupby('H0_kA_m')['Pendiente'].std()

ax2.errorbar(H0, mean_by_h0, yerr=std_by_h0, 
            fmt='o-', capsize=5, capthick=2, linewidth=2)
ax2.set_title('Pendiente Promedio vs Campo H0', fontsize=14)
ax2.set_xlabel('Campo H0 (kA/m)', fontsize=12)
ax2.set_ylabel('Pendiente Promedio', fontsize=12)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#% 7. EXPORTACIÓN DE RESULTADOS
print("\n7. EXPORTACIÓN DE RESULTADOS")
print("-"*40)

# Guardar estadísticas
stats_freq.to_csv('estadisticas_pendientes_por_frecuencia.csv')
stats_h0.to_csv('estadisticas_pendientes_por_campo_H0.csv')

print("Estadísticas guardadas en:")
print("- 'estadisticas_pendientes_por_frecuencia.csv'")
print("- 'estadisticas_pendientes_por_campo_H0.csv'")

print("\n¡Análisis de distribuciones completado!")
#%%

# =============================================================================
# HISTOGRAMA DE TODAS LAS MEDIDAS DE PENDIENTE
# =============================================================================
print("="*60)
print("HISTOGRAMA COMPLETO - TODAS LAS MEDIDAS DE PENDIENTE")
print("="*60)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm, gaussian_kde

# Crear array con todas las pendientes
todas_pendientes = m_nominal.flatten()

# Estadísticas descriptivas
media = np.mean(todas_pendientes)
mediana = np.median(todas_pendientes)
desviacion = np.std(todas_pendientes)
minimo = np.min(todas_pendientes)
maximo = np.max(todas_pendientes)
rango = maximo - minimo

print(f"ESTADÍSTICAS DE TODAS LAS PENDIENTES:")
print(f"Número de medidas: {len(todas_pendientes)}")
print(f"Media: {media:.3f}")
print(f"Mediana: {mediana:.3f}")
print(f"Desviación estándar: {desviacion:.3f}")
print(f"Mínimo: {minimo:.3f}")
print(f"Máximo: {maximo:.3f}")
print(f"Rango: {rango:.3f}")

#% 1. HISTOGRAMA BÁSICO CON TODAS LAS MEDIDAS
print("\n1. HISTOGRAMA BÁSICO")
print("-"*40)

plt.figure(figsize=(12, 8))

# Calcular número óptimo de bins (regla de Freedman-Diaconis)
q75, q25 = np.percentile(todas_pendientes, [75, 25])
iqr = q75 - q25
bin_width = 2 * iqr / (len(todas_pendientes) ** (1/3))
n_bins = int((maximo - minimo) / bin_width)
n_bins = min(n_bins, 20)  # Limitar a máximo 20 bins

# Crear histograma
n, bins, patches = plt.hist(todas_pendientes, bins=n_bins, alpha=0.7, 
                           color='skyblue', edgecolor='black', linewidth=0.8,
                           density=False)

# Añadir líneas de referencia
plt.axvline(media, color='red', linestyle='--', linewidth=3, 
           label=f'Media: {media:.2f}')
plt.axvline(mediana, color='green', linestyle='--', linewidth=3, 
           label=f'Mediana: {mediana:.2f}')
plt.axvline(media + desviacion, color='orange', linestyle=':', linewidth=2,
           label=f'±1σ: {media + desviacion:.2f}')
plt.axvline(media - desviacion, color='orange', linestyle=':', linewidth=2)

# Configuración del gráfico
plt.title('Distribución de Todas las Medidas de Pendiente\n(N = {} medidas)'.format(len(todas_pendientes)), 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Pendiente', fontsize=14)
plt.ylabel('Frecuencia', fontsize=14)
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=12)

# Añadir texto con estadísticas
stats_text = f'N = {len(todas_pendientes)}\nμ = {media:.2f}\nσ = {desviacion:.2f}'
plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
         verticalalignment='top', horizontalalignment='left',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
         fontsize=12)

plt.tight_layout()
plt.show()

#%% 2. HISTOGRAMA CON CURVA DE DENSIDAD
print("\n2. HISTOGRAMA CON CURVA DE DENSIDAD")
print("-"*40)

plt.figure(figsize=(10, 6.6),constrained_layout=True)

# Histograma
n, bins, patches = plt.hist(todas_pendientes, bins=n_bins, alpha=0.7, 
                           color='lightcoral', edgecolor='black', linewidth=0.8,
                           density=True)

# Calcular y plotear curva de densidad KDE
kde = gaussian_kde(todas_pendientes)
x_range = np.linspace(minimo - 0.1*rango, maximo + 0.1*rango, 1000)
plt.plot(x_range, kde(x_range), 'b-', linewidth=3, label='Curva de densidad (KDE)')

# Calcular y plotear distribución normal teórica
x_norm = np.linspace(minimo, maximo, 1000)
y_norm = norm.pdf(x_norm, media, desviacion)
plt.plot(x_norm, y_norm, 'g--', linewidth=2, label='Distribución normal teórica')

# Líneas de referencia
plt.axvline(media, color='red', linestyle='--', linewidth=3, 
           label=f'Media: {media:.2f}')
plt.axvline(mediana, color='green', linestyle='--', linewidth=3, 
           label=f'Mediana: {mediana:.2f}')

# Configuración
plt.title('Distribución de Todas las Pendientes con Curva de Densidad', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Pendiente', fontsize=14)
plt.ylabel('Densidad de probabilidad', fontsize=14)
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=12)

# Test de normalidad
from scipy.stats import shapiro, normaltest
stat_sw, p_sw = shapiro(todas_pendientes)
stat_ks, p_ks = normaltest(todas_pendientes)

norm_text = f'Shapiro-Wilk: p = {p_sw:.3e}\nD\'Agostino: p = {p_ks:.3e}'
plt.text(0.85, 0.5, norm_text, transform=plt.gca().transAxes,
         verticalalignment='center', horizontalalignment='center',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
         fontsize=11)

# plt.tight_layout()
plt.savefig('histograma_pendientes_con_densidad.png', dpi=300)
plt.show()

#%% 3. HISTOGRAMA POR AGRUPAMIENTO (FREQUENCIA vs H0)
print("\n3. HISTOGRAMA COMPARATIVO POR TIPO DE VARIABLE")
print("-"*40)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Agrupar por frecuencia
colors_freq = plt.cm.Set3(np.linspace(0, 1, len(frecuencias)))
for i, freq in enumerate(frecuencias):
    data = m_nominal[i, :]
    ax1.hist(data, bins=n_bins, alpha=0.6, color=colors_freq[i], 
             edgecolor='black', linewidth=0.5, label=f'{freq} kHz',
             density=True)

ax1.set_title('Distribución por Frecuencia', fontsize=14, fontweight='bold')
ax1.set_xlabel('Pendiente', fontsize=12)
ax1.set_ylabel('Densidad', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Agrupar por campo H0 (transponer la matriz)
colors_h0 = plt.cm.viridis(np.linspace(0, 1, len(H0)))
for j, h0_val in enumerate(H0):
    data = m_nominal[:, j]
    ax2.hist(data, bins=n_bins, alpha=0.6, color=colors_h0[j], 
             edgecolor='black', linewidth=0.5, label=f'{h0_val} kA/m',
             density=True)

ax2.set_title('Distribución por Campo H0', fontsize=14, fontweight='bold')
ax2.set_xlabel('Pendiente', fontsize=12)
ax2.set_ylabel('Densidad', fontsize=12)
ax2.legend(fontsize=8, ncol=2)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#% 4. HISTOGRAMA ACUMULATIVO
print("\n4. HISTOGRAMA ACUMULATIVO")
print("-"*40)

plt.figure(figsize=(12, 8))

# Histograma acumulativo
counts, bin_edges = np.histogram(todas_pendientes, bins=n_bins)
cumulative = np.cumsum(counts)
cumulative_percent = cumulative / len(todas_pendientes) * 100

plt.hist(todas_pendientes, bins=n_bins, alpha=0.7, color='lightgreen',
         edgecolor='black', linewidth=0.8, cumulative=True,
         label='Distribución acumulativa')

# Añadir línea de percentiles
for percentile in [25, 50, 75, 90]:
    value = np.percentile(todas_pendientes, percentile)
    plt.axvline(value, color='red', linestyle='--', alpha=0.7,
               label=f'P{percentile}: {value:.2f}')

plt.title('Distribución Acumulativa de Todas las Pendientes', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Pendiente', fontsize=14)
plt.ylabel('Frecuencia acumulada', fontsize=14)
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=11)

# Añadir percentiles
percentiles_text = f'P25: {np.percentile(todas_pendientes, 25):.2f}\n'
percentiles_text += f'P50: {np.percentile(todas_pendientes, 50):.2f}\n'
percentiles_text += f'P75: {np.percentile(todas_pendientes, 75):.2f}\n'
percentiles_text += f'P90: {np.percentile(todas_pendientes, 90):.2f}'

plt.text(0.02, 0.75, percentiles_text, transform=plt.gca().transAxes,
         verticalalignment='top', horizontalalignment='left',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
         fontsize=11)

plt.tight_layout()
plt.show()

#% 5. ANÁLISIS DE NORMALIDAD COMPLETO
print("\n5. ANÁLISIS DE NORMALIDAD COMPLETO")
print("-"*40)

from scipy.stats import shapiro, normaltest, anderson

print("TESTS DE NORMALIDAD PARA TODAS LAS PENDIENTES:")
print(f"Número de muestras: {len(todas_pendientes)}")

# Shapiro-Wilk test
stat_sw, p_sw = shapiro(todas_pendientes)
print(f"\nShapiro-Wilk Test:")
print(f"  Estadístico: {stat_sw:.4f}")
print(f"  p-valor: {p_sw:.3e}")
print(f"  Normalidad: {'SÍ' if p_sw > 0.05 else 'NO'}")

# D'Agostino's K^2 test
stat_da, p_da = normaltest(todas_pendientes)
print(f"\nD'Agostino K^2 Test:")
print(f"  Estadístico: {stat_da:.4f}")
print(f"  p-valor: {p_da:.3e}")
print(f"  Normalidad: {'SÍ' if p_da > 0.05 else 'NO'}")

# Anderson-Darling test
result_ad = anderson(todas_pendientes)
print(f"\nAnderson-Darling Test:")
print(f"  Estadístico: {result_ad.statistic:.4f}")
print("  Valores críticos:")
for i in range(len(result_ad.critical_values)):
    sl, cv = result_ad.significance_level[i], result_ad.critical_values[i]
    print(f"    {sl}%: {cv:.3f} - {'Normal' if result_ad.statistic < cv else 'No normal'}")

#% 6. EXPORTACIÓN DE DATOS
print("\n6. EXPORTACIÓN DE DATOS")
print("-"*40)

# Crear DataFrame con todas las medidas
df_todas_pendientes = pd.DataFrame({
    'Pendiente': todas_pendientes,
    'Frecuencia_kHz': np.repeat(frecuencias, len(H0)),
    'H0_kA_m': list(H0) * len(frecuencias)
})

# Guardar datos
df_todas_pendientes.to_csv('todas_las_pendientes.csv', index=False)

# Guardar estadísticas
estadisticas = {
    'N_medidas': len(todas_pendientes),
    'Media': media,
    'Mediana': mediana,
    'Desviacion_estandar': desviacion,
    'Minimo': minimo,
    'Maximo': maximo,
    'Rango': rango,
    'Shapiro_Wilk_p': p_sw,
    'D_Agostino_p': p_da
}

df_estadisticas = pd.DataFrame([estadisticas])
df_estadisticas.to_csv('estadisticas_todas_pendientes.csv', index=False)

print("Datos guardados en:")
print("- 'todas_las_pendientes.csv'")
print("- 'estadisticas_todas_pendientes.csv'")

print(f"\n¡Análisis completo de {len(todas_pendientes)} medidas realizado! ✅")
#%%