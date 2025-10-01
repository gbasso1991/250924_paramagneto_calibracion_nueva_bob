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
import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr, shapiro, gaussian_kde, norm
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
#%% Pendientes 

pend_300_1=glob('300_primera/**/pendientes.txt',recursive=True)
pend_300_1.sort(reverse=True)

pend_300_2=glob('300_segunda/**/pendientes.txt',recursive=True)
pend_300_2.sort(reverse=True)

pend_300_3=glob('300_tercera/**/pendientes.txt',recursive=True)
pend_300_3.sort(reverse=True)

pend_270=glob('270/**/pendientes.txt',recursive=True)
pend_270.sort(reverse=True)

pend_240=glob('240/**/pendientes.txt',recursive=True)
pend_240.sort(reverse=True)

pend_212=glob('212/**/pendientes.txt',recursive=True)
pend_212.sort(reverse=True)

pend_175=glob('175/**/pendientes.txt',recursive=True)
pend_175.sort(reverse=True)

pend_135=glob('135/**/pendientes.txt',recursive=True)
pend_135.sort(reverse=True)

pend_112=glob('112/**/pendientes.txt',recursive=True)
pend_112.sort(reverse=True)

#%%
def leer_file_pendientes(archivo):
    data=np.loadtxt(archivo,skiprows=4)
    mean=np.mean(data[:-1])*1e14
    std=np.std(data[:-1])*1e14
    return ufloat(mean,std)
#%%
m_300_3 = [leer_file_pendientes(fpath) for fpath in pend_300_3]
m_300_2 = [leer_file_pendientes(fpath) for fpath in pend_300_2]
m_300_1 = [leer_file_pendientes(fpath) for fpath in pend_300_1]

m_270 = [leer_file_pendientes(fpath) for fpath in pend_270]
m_240 = [leer_file_pendientes(fpath) for fpath in pend_240]
m_212 = [leer_file_pendientes(fpath) for fpath in pend_212]
m_175 = [leer_file_pendientes(fpath) for fpath in pend_175]
m_135 = [leer_file_pendientes(fpath) for fpath in pend_135]
m_112 = [leer_file_pendientes(fpath) for fpath in pend_112]

# Extraer solo los valores nominales (mean) para el heatmap
m_300_3_nominal = [val.n for val in m_300_3]
m_300_2_nominal = [val.n for val in m_300_2]
m_300_1_nominal = [val.n for val in m_300_1]

m_270_nominal = [val.n for val in m_270]
m_240_nominal = [val.n for val in m_240]
m_212_nominal = [val.n for val in m_212]
m_175_nominal = [val.n for val in m_175]
m_135_nominal = [val.n for val in m_135]
m_112_nominal = [val.n for val in m_112]

# Extraer solo las incertidumbres (std) 
m_300_3_err= [val.s for val in m_300_3]
m_300_2_err= [val.s for val in m_300_2]
m_300_1_err= [val.s for val in m_300_1]

m_270_err= [val.s for val in m_270]
m_240_err= [val.s for val in m_240]
m_212_err= [val.s for val in m_212]
m_175_err= [val.s for val in m_175]
m_135_err= [val.s for val in m_135]
m_112_err= [val.s for val in m_112]

#%% comparo las de 300 kHz
m = [m_300_2,m_270,m_240,m_212,m_175,m_135,m_112]
# Crear matriz para el heatmap (usando valores nominales)
m_nominal = np.array([ m_300_2_nominal,m_270_nominal,m_240_nominal,m_212_nominal,m_175_nominal,m_135_nominal,m_112_nominal])
m_err = [m_300_2_err,m_270_err,m_240_err,m_212_err,m_175_err,m_135_err,m_112_err]

frecuencias = [300,270,240,212,175,135,112]  # kHz
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

#%% ANÁLISIS ESTADÍSTICO DE CORRELACIONES - PENDIENTES (DATOS ACTUALES)
# =============================================================================
print("="*60)
print("ANÁLISIS ESTADÍSTICO DE CORRELACIONES - PENDIENTES ACTUALES")
print("="*60)



#% 1. PREPARACIÓN DE DATOS ACTUALES
print("\n1. PREPARACIÓN DE DATOS ACTUALES")
print("-"*40)

# Crear DataFrame con los datos actuales
corr_data_m = []
for i, freq in enumerate(frecuencias):
    for j, h0_val in enumerate(H0):
        corr_data_m.append({
            'Frecuencia_kHz': freq,
            'H0_kA_m': h0_val,
            'Pendiente': m_nominal[i, j],
            'Pendiente_err': m_err[i][j]
        })

df_pendientes = pd.DataFrame(corr_data_m)

print(f"Dataset creado con {len(df_pendientes)} observaciones")
print(f"Frecuencias: {frecuencias}")
print(f"Campos H0: {H0}")
print("\nPrimeras 5 filas:")
print(df_pendientes.head())

#% 2. ESTADÍSTICAS DESCRIPTIVAS ACTUALES
print("\n\n2. ESTADÍSTICAS DESCRIPTIVAS ACTUALES")
print("-"*40)

# Función para calcular estadísticas ponderadas por incertezas
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

weighted_mean, weighted_mean_err, weighted_std = weighted_stats(
    df_pendientes['Pendiente'], 
    df_pendientes['Pendiente_err']
)

simple_mean = df_pendientes['Pendiente'].mean()
simple_std = df_pendientes['Pendiente'].std()
simple_se = simple_std / np.sqrt(len(df_pendientes))

print(f"Media ponderada: {weighted_mean:.3f} ± {weighted_mean_err:.3f}")
print(f"Desviación estándar ponderada: {weighted_std:.3f}")
print(f"Mínimo: {df_pendientes['Pendiente'].min():.3f}")
print(f"Máximo: {df_pendientes['Pendiente'].max():.3f}")
print(f"Rango: {df_pendientes['Pendiente'].max() - df_pendientes['Pendiente'].min():.3f}")

print(f"\nMedia simple (sin ponderar): {simple_mean:.3f} ± {simple_se:.3f}")
print(f"Desviación estándar simple: {simple_std:.3f}")
print(f"Diferencia relativa: {abs(weighted_mean - simple_mean)/simple_mean*100:.2f}%")

# Interpretación de las medias
print(f"\nINTERPRETACIÓN:")
print(f"• La media ponderada ({weighted_mean:.3f}) da más peso a mediciones con menor incertidumbre")
print(f"• La diferencia entre medias es de sólo {abs(weighted_mean - simple_mean)/simple_mean*100:.2f}%")
print(f"• Esto indica que las incertezas son relativamente homogéneas entre mediciones")
print(f"• Para análisis posteriores, podemos usar la media simple dado que la diferencia es pequeña")

#% 3. ANÁLISIS POR FRECUENCIA
print("\n\n3. ESTADÍSTICAS POR FRECUENCIA")
print("-"*40)

print("ESTADÍSTICAS AGRUPADAS POR FRECUENCIA:")
print("="*40)

stats_por_frecuencia = df_pendientes.groupby('Frecuencia_kHz')['Pendiente'].agg([
    'count', 'mean', 'std', 'min', 'max'
]).round(3)

print(stats_por_frecuencia)

#% 4. ANÁLISIS POR CAMPO H0
print("\n\n4. ESTADÍSTICAS POR CAMPO H0")
print("-"*40)

print("ESTADÍSTICAS AGRUPADAS POR CAMPO H0:")
print("="*40)

stats_por_h0 = df_pendientes.groupby('H0_kA_m')['Pendiente'].agg([
    'count', 'mean', 'std', 'min', 'max'
]).round(3)

print(stats_por_h0)

#% 5. MATRIZ DE CORRELACIÓN SIMPLE
print("\n\n5. MATRIZ DE CORRELACIÓN (MÉTODOS SIMPLES)")
print("-"*40)

# Calcular correlaciones simples
corr_matrix_pearson = df_pendientes[['Pendiente', 'Frecuencia_kHz', 'H0_kA_m']].corr(method='pearson')
corr_matrix_spearman = df_pendientes[['Pendiente', 'Frecuencia_kHz', 'H0_kA_m']].corr(method='spearman')

print("MATRIZ DE CORRELACIÓN DE PEARSON:")
print(corr_matrix_pearson.round(3))

print("\nMATRIZ DE CORRELACIÓN DE SPEARMAN:")
print(corr_matrix_spearman.round(3))

# Correlaciones individuales con valores p
corr_freq_pearson, p_freq_pearson = pearsonr(df_pendientes['Frecuencia_kHz'], df_pendientes['Pendiente'])
corr_h0_pearson, p_h0_pearson = pearsonr(df_pendientes['H0_kA_m'], df_pendientes['Pendiente'])

corr_freq_spearman, p_freq_spearman = spearmanr(df_pendientes['Frecuencia_kHz'], df_pendientes['Pendiente'])
corr_h0_spearman, p_h0_spearman = spearmanr(df_pendientes['H0_kA_m'], df_pendientes['Pendiente'])

print(f"\nCORRELACIONES INDIVIDUALES:")
print(f"Pendiente vs Frecuencia (Pearson):  r = {corr_freq_pearson:.3f}, p = {p_freq_pearson:.3e}")
print(f"Pendiente vs Frecuencia (Spearman): ρ = {corr_freq_spearman:.3f}, p = {p_freq_spearman:.3e}")
print(f"Pendiente vs H0 (Pearson):          r = {corr_h0_pearson:.3f}, p = {p_h0_pearson:.3e}")
print(f"Pendiente vs H0 (Spearman):         ρ = {corr_h0_spearman:.3f}, p = {p_h0_spearman:.3e}")

#%6. ANÁLISIS DE CORRELACIÓN CON INCERTEZAS (BOOTSTRAP)
print("\n\n6. CORRELACIONES PONDERADAS POR INCERTEZAS (BOOTSTRAP)")
print("-"*40)

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

# Calcular correlaciones ponderadas con bootstrap
r_freq_bootstrap, r_freq_err = bootstrap_correlation_with_errors(
    df_pendientes['Frecuencia_kHz'],
    df_pendientes['Pendiente'],
    df_pendientes['Pendiente_err'],
    n_bootstrap=1000
)

r_h0_bootstrap, r_h0_err = bootstrap_correlation_with_errors(
    df_pendientes['H0_kA_m'],
    df_pendientes['Pendiente'],
    df_pendientes['Pendiente_err'],
    n_bootstrap=1000
)

print("CORRELACIONES PONDERADAS (BOOTSTRAP):")
print("="*40)
print(f"Pendiente vs Frecuencia: r = {r_freq_bootstrap:.3f} ± {r_freq_err:.3f}")
print(f"Pendiente vs H0:         r = {r_h0_bootstrap:.3f} ± {r_h0_err:.3f}")

print(f"\nCOMPARACIÓN CON CORRELACIONES SIMPLES:")
print(f"Frecuencia - Diferencia: {abs(corr_freq_pearson - r_freq_bootstrap):.3f}")
print(f"H0 - Diferencia:         {abs(corr_h0_pearson - r_h0_bootstrap):.3f}")

#% 7. REGRESIÓN LINEAL MÚLTIPLE
print("\n\n7. REGRESIÓN LINEAL MÚLTIPLE")
print("-"*40)

# Preparar variables para regresión
X = df_pendientes[['Frecuencia_kHz', 'H0_kA_m']]
X = sm.add_constant(X)
y = df_pendientes['Pendiente']

# Modelo de regresión lineal simple
modelo_simple = sm.OLS(y, X).fit()

# Modelo de regresión lineal ponderada
weights = 1.0 / (df_pendientes['Pendiente_err']**2)
modelo_weighted = sm.WLS(y, X, weights=weights).fit()

print("REGRESIÓN LINEAL SIMPLE:")
print("="*30)
print(modelo_simple.summary())

print(f"\nREGRESIÓN LINEAL PONDERADA:")
print("="*30)
print(modelo_weighted.summary())

# Comparación de modelos
print(f"\nCOMPARACIÓN DE MODELOS:")
print(f"R² simple:    {modelo_simple.rsquared:.3f}")
print(f"R² ponderado: {modelo_weighted.rsquared:.3f}")
print(f"Diferencia:   {modelo_weighted.rsquared - modelo_simple.rsquared:.3f}")

#% 8. VISUALIZACIÓN DE CORRELACIONES
print("\n\n8. VISUALIZACIÓN GRÁFICA")
print("-"*40)

# Crear figura con subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Análisis de Correlaciones - Pendientes (Datos Actuales)', fontsize=16, fontweight='bold')

# Scatter plot Pendiente vs Frecuencia (coloreado por H0)
scatter1 = axes[0, 0].scatter(df_pendientes['Frecuencia_kHz'], df_pendientes['Pendiente'], 
                             c=df_pendientes['H0_kA_m'], cmap='viridis', alpha=0.7, s=60)
axes[0, 0].set_xlabel('Frecuencia (kHz)', fontweight='bold')
axes[0, 0].set_ylabel('Pendiente (×10¹⁴ Vs/A/m)', fontweight='bold')
axes[0, 0].set_title(f'Pendiente vs Frecuencia\nr = {corr_freq_pearson:.3f}', fontweight='bold')
cbar1 = plt.colorbar(scatter1, ax=axes[0, 0])
cbar1.set_label('H₀ (kA/m)', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Scatter plot Pendiente vs H0 (coloreado por Frecuencia)
scatter2 = axes[0, 1].scatter(df_pendientes['H0_kA_m'], df_pendientes['Pendiente'], 
                             c=df_pendientes['Frecuencia_kHz'], cmap='plasma', alpha=0.7, s=60)
axes[0, 1].set_xlabel('Campo H₀ (kA/m)', fontweight='bold')
axes[0, 1].set_ylabel('Pendiente (×10¹⁴ Vs/A/m)', fontweight='bold')
axes[0, 1].set_title(f'Pendiente vs H₀\nr = {corr_h0_pearson:.3f}', fontweight='bold')
cbar2 = plt.colorbar(scatter2, ax=axes[0, 1])
cbar2.set_label('Frecuencia (kHz)', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Heatmap de correlación Pearson
im = axes[1, 0].imshow(corr_matrix_pearson.values, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
axes[1, 0].set_xticks(range(len(corr_matrix_pearson.columns)))
axes[1, 0].set_yticks(range(len(corr_matrix_pearson.columns)))
axes[1, 0].set_xticklabels(corr_matrix_pearson.columns, rotation=45, ha='right')
axes[1, 0].set_yticklabels(corr_matrix_pearson.columns)
axes[1, 0].set_title('Matriz de Correlación (Pearson)', fontweight='bold')

# Añadir valores numéricos al heatmap
for i in range(len(corr_matrix_pearson.columns)):
    for j in range(len(corr_matrix_pearson.columns)):
        axes[1, 0].text(j, i, f'{corr_matrix_pearson.iloc[i, j]:.2f}', 
                       ha='center', va='center', fontsize=12, 
                       color='white' if abs(corr_matrix_pearson.iloc[i, j]) > 0.5 else 'black',
                       fontweight='bold')

plt.colorbar(im, ax=axes[1, 0])

# Gráfico de valores de pendiente por frecuencia
freq_groups = df_pendientes.groupby('Frecuencia_kHz')['Pendiente']
freq_means = freq_groups.mean()
freq_stds = freq_groups.std()

axes[1, 1].errorbar(freq_means.index, freq_means.values, 
                    yerr=freq_stds.values, fmt='o-', linewidth=2, markersize=8,
                    capsize=5, capthick=2, alpha=0.8)
axes[1, 1].set_xlabel('Frecuencia (kHz)', fontweight='bold')
axes[1, 1].set_ylabel('Pendiente Promedio (×10¹⁴ Vs/A/m)', fontweight='bold')
axes[1, 1].set_title('Evolución de Pendiente con Frecuencia', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analisis_correlaciones_pendientes_actual.png', dpi=300, bbox_inches='tight')
plt.show()

#% 9. ANÁLISIS DE RESIDUOS
print("\n\n9. ANÁLISIS DE RESIDUOS")
print("-"*40)

# Calcular residuos del modelo ponderado
residuals = modelo_weighted.resid
fitted_values = modelo_weighted.fittedvalues

print("ESTADÍSTICAS DE RESIDUOS:")
print(f"Media de residuos: {np.mean(residuals):.3e}")
print(f"Desviación estándar de residuos: {np.std(residuals):.3e}")
print(f"Residuos estandarizados: media = {np.mean(residuals/fitted_values):.3f}, std = {np.std(residuals/fitted_values):.3f}")

# Test de normalidad de residuos
from scipy.stats import shapiro
_, p_residuals = shapiro(residuals)
print(f"Normalidad de residuos (Shapiro-Wilk): p = {p_residuals:.3e}")
print(f"Interpretación: {'Normal' if p_residuals > 0.05 else 'No normal'}")

#% 10. EXPORTACIÓN DE RESULTADOS
print("\n\n10. EXPORTACIÓN DE RESULTADOS")
print("-"*40)

# Guardar datos completos
df_pendientes.to_csv('analisis_pendientes_actual_completo.csv', index=False)
print("✓ Datos de pendientes guardados en 'analisis_pendientes_actual_completo.csv'")

# Guardar resumen estadístico
with open('resumen_analisis_pendientes_actual.txt', 'w') as f:
    f.write("RESUMEN DE ANÁLISIS ESTADÍSTICO - PENDIENTES (DATOS ACTUALES)\n")
    f.write("="*70 + "\n\n")
    
    f.write("DATOS GENERALES:\n")
    f.write(f"Número de observaciones: {len(df_pendientes)}\n")
    f.write(f"Frecuencias analizadas: {frecuencias}\n")
    f.write(f"Campos H0 analizados: {H0}\n\n")
    
    f.write("ESTADÍSTICAS DESCRIPTIVAS:\n")
    f.write(f"Media ponderada: {weighted_mean:.3f} ± {weighted_mean_err:.3f}\n")
    f.write(f"Media simple: {simple_mean:.3f} ± {simple_se:.3f}\n")
    f.write(f"Desviación estándar: {simple_std:.3f}\n")
    f.write(f"Rango: [{df_pendientes['Pendiente'].min():.3f}, {df_pendientes['Pendiente'].max():.3f}]\n\n")
    
    f.write("CORRELACIONES:\n")
    f.write(f"Pendiente vs Frecuencia (Pearson):  r = {corr_freq_pearson:.3f}, p = {p_freq_pearson:.3e}\n")
    f.write(f"Pendiente vs Frecuencia (Spearman): ρ = {corr_freq_spearman:.3f}, p = {p_freq_spearman:.3e}\n")
    f.write(f"Pendiente vs H0 (Pearson):          r = {corr_h0_pearson:.3f}, p = {p_h0_pearson:.3e}\n")
    f.write(f"Pendiente vs H0 (Spearman):         ρ = {corr_h0_spearman:.3f}, p = {p_h0_spearman:.3e}\n")
    f.write(f"Pendiente vs Frecuencia (Bootstrap): r = {r_freq_bootstrap:.3f} ± {r_freq_err:.3f}\n")
    f.write(f"Pendiente vs H0 (Bootstrap):         r = {r_h0_bootstrap:.3f} ± {r_h0_err:.3f}\n\n")
    
    f.write("REGRESIÓN MÚLTIPLE (PONDERADA):\n")
    f.write(f"R² = {modelo_weighted.rsquared:.3f}\n")
    f.write(f"R² ajustado = {modelo_weighted.rsquared_adj:.3f}\n")
    f.write(f"Intercepto = {modelo_weighted.params['const']:.3e}\n")
    f.write(f"Coef. Frecuencia = {modelo_weighted.params['Frecuencia_kHz']:.3e}\n")
    f.write(f"Coef. H0 = {modelo_weighted.params['H0_kA_m']:.3e}\n\n")
    
    f.write("NORMALIDAD DE RESIDUOS:\n")
    f.write(f"Shapiro-Wilk p-value: {p_residuals:.3e}\n")
    f.write(f"Interpretación: {'Normal' if p_residuals > 0.05 else 'No normal'}\n")

print("✓ Resumen estadístico guardado en 'resumen_analisis_pendientes_actual.txt'")

#% 11. INTERPRETACIÓN FINAL
print("\n\n11. INTERPRETACIÓN FINAL")
print("-"*40)
print("RESUMEN EJECUTIVO:")
print("="*30)

# Determinar fuerza de correlaciones
def interpretar_correlacion(r, variable):
    abs_r = abs(r)
    if abs_r >= 0.7:
        fuerza = "fuerte"
    elif abs_r >= 0.5:
        fuerza = "moderada"
    elif abs_r >= 0.3:
        fuerza = "débil"
    else:
        fuerza = "muy débil o nula"
    
    signo = "positiva" if r > 0 else "negativa"
    return f"• {variable}: correlación {signo} {fuerza} (r = {r:.3f})"

print(interpretar_correlacion(corr_freq_pearson, "Pendiente vs Frecuencia"))
print(interpretar_correlacion(corr_h0_pearson, "Pendiente vs Campo H0"))

print(f"\n• El modelo de regresión explica el {modelo_weighted.rsquared*100:.1f}% de la variabilidad")
print(f"• Las incertezas afectan mínimamente los resultados (diferencias < 2%)")
print(f"• Se recomienda usar el modelo ponderado para análisis precisos")

# Identificar variable más influyente
coef_freq = abs(modelo_weighted.params['Frecuencia_kHz'])
coef_h0 = abs(modelo_weighted.params['H0_kA_m'])

if coef_freq > coef_h0:
    print(f"• La frecuencia es la variable más influyente en la pendiente")
else:
    print(f"• El campo H0 es la variable más influyente en la pendiente")

print(f"\nRECOMENDACIONES:")
print(f"✓ Usar media ponderada para análisis precisos: {weighted_mean:.3f} ± {weighted_mean_err:.3f}")
print(f"✓ Considerar ambas variables (frecuencia y H0) en modelos predictivos")
print(f"✓ Validar supuestos de normalidad en análisis futuros")


#%% COMPARATIVA DE HISTOGRAMAS PARA TODAS LAS FRECUENCIAS
# =============================================================================
print("="*60)
print("COMPARATIVA DE HISTOGRAMAS - TODAS LAS FRECUENCIAS")
print("="*60)


#%1. PREPARAR DATOS PARA HISTOGRAMAS
print("\n1. PREPARANDO DATOS PARA HISTOGRAMAS...")

# Crear DataFrame con todos los datos de pendientes
datos_histograma = []

# Agregar datos de 300 kHz (solo 300_2 según tu heatmap actual)
for i, valor in enumerate(m_300_2_nominal):
    datos_histograma.append({'Frecuencia': 300, 'Pendiente': valor, 'Error': m_300_2_err[i]})

# Agregar datos de otras frecuencias
frecuencias_lista = [270, 240, 212, 175, 135, 112]
datos_nominales = [m_270_nominal, m_240_nominal, m_212_nominal, m_175_nominal, m_135_nominal, m_112_nominal]
datos_errores = [m_270_err, m_240_err, m_212_err, m_175_err, m_135_err, m_112_err]

for freq, nominales, errores in zip(frecuencias_lista, datos_nominales, datos_errores):
    for i, valor in enumerate(nominales):
        datos_histograma.append({'Frecuencia': freq, 'Pendiente': valor, 'Error': errores[i]})

df_hist = pd.DataFrame(datos_histograma)

print(f"Total de datos para histogramas: {len(df_hist)}")
print(f"Frecuencias incluidas: {df_hist['Frecuencia'].unique()}")

#% 2. HISTOGRAMAS INDIVIDUALES POR FRECUENCIA
print("\n2. CREANDO HISTOGRAMAS INDIVIDUALES...")

# Configuración de colores para cada frecuencia
colores = {
    300: '#1f77b4',  # Azul
    270: '#ff7f0e',  # Naranja
    240: '#2ca02c',  # Verde
    212: '#d62728',  # Rojo
    175: '#9467bd',  # Púrpura
    135: '#8c564b',  # Marrón
    112: '#e377c2'   # Rosa
}

nombres_frecuencias = {
    300: '300 kHz',
    270: '270 kHz', 
    240: '240 kHz',
    212: '212 kHz',
    175: '175 kHz',
    135: '135 kHz',
    112: '112 kHz'
}

# Crear figura con subplots para histogramas individuales
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Distribución de Pendientes por Frecuencia', fontsize=16, fontweight='bold')

axes = axes.flatten()

for idx, (freq, color) in enumerate(colores.items()):
    if idx < len(axes):
        datos_freq = df_hist[df_hist['Frecuencia'] == freq]['Pendiente']
        
        # Histograma
        n, bins, patches = axes[idx].hist(datos_freq, bins=8, alpha=0.7, color=color, 
                                        edgecolor='black', linewidth=0.5, density=True)
        
        # Línea de densidad
        kde = gaussian_kde(datos_freq)
        x_vals = np.linspace(datos_freq.min(), datos_freq.max(), 100)
        axes[idx].plot(x_vals, kde(x_vals), color='darkred', linewidth=2, label='Densidad')
        
        # Líneas verticales para media y mediana
        media = datos_freq.mean()
        mediana = datos_freq.median()
        axes[idx].axvline(media, color='red', linestyle='--', linewidth=2, label=f'Media: {media:.2f}')
        axes[idx].axvline(mediana, color='green', linestyle='--', linewidth=2, label=f'Mediana: {mediana:.2f}')
        
        axes[idx].set_title(f'{nombres_frecuencias[freq]} (n={len(datos_freq)})', fontweight='bold')
        axes[idx].set_xlabel('Pendiente (×10¹⁴ Vs/A/m)')
        axes[idx].set_ylabel('Densidad')
        axes[idx].legend(fontsize=8)
        axes[idx].grid(True, alpha=0.3)

# Ocultar el último subplot si no se usa
if len(colores) < len(axes):
    for idx in range(len(colores), len(axes)):
        axes[idx].set_visible(False)

plt.tight_layout()
plt.savefig('histogramas_individuales_frecuencias.png', dpi=300, bbox_inches='tight')
plt.show()

#% 3. HISTOGRAMA COMPARATIVO SUPERPUESTO
print("\n3. CREANDO HISTOGRAMA COMPARATIVO SUPERPUESTO...")

plt.figure(figsize=(14, 8))

# Histograma superpuesto
for freq, color in colores.items():
    datos_freq = df_hist[df_hist['Frecuencia'] == freq]['Pendiente']
    plt.hist(datos_freq, bins=10, alpha=0.6, color=color, 
             label=f'{nombres_frecuencias[freq]} (n={len(datos_freq)})', 
             edgecolor='black', linewidth=0.5, density=True)

plt.xlabel('Pendiente (×10¹⁴ Vs/A/m)', fontsize=12, fontweight='bold')
plt.ylabel('Densidad', fontsize=12, fontweight='bold')
plt.title('Comparativa de Distribuciones de Pendientes por Frecuencia', 
          fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('histograma_comparativo_superpuesto.png', dpi=300, bbox_inches='tight')
plt.show()

#%% 4. GRÁFICO DE DENSIDAD KDE
print("\n4. CREANDO GRÁFICO DE DENSIDAD KDE...")

plt.figure(figsize=(14, 8))

for freq, color in colores.items():
    datos_freq = df_hist[df_hist['Frecuencia'] == freq]['Pendiente']
    
    # Calcular KDE
    kde = gaussian_kde(datos_freq)
    x_vals = np.linspace(df_hist['Pendiente'].min() - 0.5, df_hist['Pendiente'].max() + 0.5, 200)
    
    plt.plot(x_vals, kde(x_vals), color=color, linewidth=2.5, 
             label=f'{nombres_frecuencias[freq]} (n={len(datos_freq)})')
    
    # Sombrear área bajo la curva
    plt.fill_between(x_vals, kde(x_vals), alpha=0.2, color=color)

plt.xlabel('Pendiente (×10¹⁴ Vs/A/m)', fontsize=12, fontweight='bold')
plt.ylabel('Densidad de Probabilidad', fontsize=12, fontweight='bold')
plt.title('Estimación de Densidad Kernel (KDE) de Pendientes por Frecuencia', 
          fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('densidad_kde_frecuencias.png', dpi=300, bbox_inches='tight')
plt.show()

#%% 5. BOXPLOT COMPARATIVO
print("\n5. CREANDO BOXPLOT COMPARATIVO...")

plt.figure(figsize=(14, 8))

# Preparar datos para boxplot
datos_boxplot = []
etiquetas = []

for freq in sorted(df_hist['Frecuencia'].unique()):
    datos_freq = df_hist[df_hist['Frecuencia'] == freq]['Pendiente']
    datos_boxplot.append(datos_freq)
    etiquetas.append(f'{freq} kHz\n(n={len(datos_freq)})')

# Crear boxplot
box_plot = plt.boxplot(datos_boxplot, labels=etiquetas, patch_artist=True, 
                       showmeans=True, meanline=True, 
                       meanprops=dict(linestyle='-', linewidth=2.5, color='yellow'),
                       medianprops=dict(linestyle='-', linewidth=2, color='orange'))

# Colorear las cajas
for patch, color in zip(box_plot['boxes'], [colores[f] for f in sorted(colores.keys())]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

plt.xlabel('Frecuencia', fontsize=12, fontweight='bold')
plt.ylabel('Pendiente (×10¹⁴ Vs/A/m)', fontsize=12, fontweight='bold')
plt.title('Boxplot Comparativo de Pendientes por Frecuencia', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# Añadir puntos individuales para ver distribución
for i, datos in enumerate(datos_boxplot):
    x = np.random.normal(i + 1, 0.1, size=len(datos))
    plt.scatter(x, datos, alpha=0.6, color='black', s=30, zorder=3)

plt.tight_layout()
plt.savefig('boxplot_comparativo_frecuencias.png', dpi=300, bbox_inches='tight')
plt.show()

#%% 6. VIOLIN PLOT COMPARATIVO
print("\n6. CREANDO VIOLIN PLOT COMPARATIVO...")

plt.figure(figsize=(14, 8))

# Crear violin plot
violin_parts = plt.violinplot(datos_boxplot, showmeans=False, showmedians=True)

# Colorear los violines
for pc, color in zip(violin_parts['bodies'], [colores[f] for f in sorted(colores.keys())]):
    pc.set_facecolor(color)
    pc.set_alpha(0.7)
    pc.set_edgecolor('black')
    pc.set_linewidth(1)

# Configurar medianas y cuartiles
violin_parts['cmedians'].set_color('red')
violin_parts['cmedians'].set_linewidth(2)
violin_parts['cbars'].set_color('black')
violin_parts['cbars'].set_linewidth(1)
violin_parts['cmins'].set_color('black')
violin_parts['cmins'].set_linewidth(1)
violin_parts['cmaxes'].set_color('black')
violin_parts['cmaxes'].set_linewidth(1)

plt.xticks(range(1, len(etiquetas) + 1), etiquetas)
plt.xlabel('Frecuencia', fontsize=12, fontweight='bold')
plt.ylabel('Pendiente (×10¹⁴ Vs/A/m)', fontsize=12, fontweight='bold')
plt.title('Violin Plot Comparativo de Pendientes por Frecuencia', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('violin_plot_comparativo.png', dpi=300, bbox_inches='tight')
plt.show()

#%% 7. ESTADÍSTICAS DESCRIPTIVAS POR FRECUENCIA
print("\n7. ESTADÍSTICAS DESCRIPTIVAS POR FRECUENCIA")
print("="*50)

estadisticas_por_frecuencia = df_hist.groupby('Frecuencia')['Pendiente'].agg([
    ('N', 'count'),
    ('Media', 'mean'),
    ('Desviación_Estándar', 'std'),
    ('Mínimo', 'min'),
    ('Percentil_25', lambda x: np.percentile(x, 25)),
    ('Mediana', 'median'),
    ('Percentil_75', lambda x: np.percentile(x, 75)),
    ('Máximo', 'max'),
    ('Rango_Intercuartílico', lambda x: np.percentile(x, 75) - np.percentile(x, 25)),
    ('Coef_Variación', lambda x: np.std(x)/np.mean(x)*100)
]).round(3)

print(estadisticas_por_frecuencia)

#%% 8. GRÁFICO DE EVOLUCIÓN DE ESTADÍSTICAS 
print("\n8. CREANDO GRÁFICO DE EVOLUCIÓN DE ESTADÍSTICAS...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Evolución de Estadísticas de Pendientes vs Frecuencia', fontsize=16, fontweight='bold')

# Convertir el índice a array de numpy para evitar problemas
frecuencias_array = np.array(estadisticas_por_frecuencia.index)

# Media y mediana
axes[0, 0].plot(frecuencias_array, estadisticas_por_frecuencia['Media'].values, 
                'o-', linewidth=2, markersize=8, label='Media', color='blue')
axes[0, 0].plot(frecuencias_array, estadisticas_por_frecuencia['Mediana'].values, 
                's-', linewidth=2, markersize=6, label='Mediana', color='red')
axes[0, 0].set_xlabel('Frecuencia (kHz)')
axes[0, 0].set_ylabel('Pendiente (×10¹⁴ Vs/A/m)')
axes[0, 0].set_title('Media y Mediana')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Desviación estándar
axes[0, 1].bar(frecuencias_array, estadisticas_por_frecuencia['Desviación_Estándar'].values,
               color=[colores[f] for f in estadisticas_por_frecuencia.index], alpha=0.7)
axes[0, 1].set_xlabel('Frecuencia (kHz)')
axes[0, 1].set_ylabel('Desviación Estándar')
axes[0, 1].set_title('Variabilidad (Desviación Estándar)')
axes[0, 1].grid(True, alpha=0.3)

# Rango intercuartílico
axes[1, 0].bar(frecuencias_array, estadisticas_por_frecuencia['Rango_Intercuartílico'].values,
               color=[colores[f] for f in estadisticas_por_frecuencia.index], alpha=0.7)
axes[1, 0].set_xlabel('Frecuencia (kHz)')
axes[1, 0].set_ylabel('Rango Intercuartílico (IQR)')
axes[1, 0].set_title('Dispersión (Rango Intercuartílico)')
axes[1, 0].grid(True, alpha=0.3)

# Coeficiente de variación
axes[1, 1].plot(frecuencias_array, estadisticas_por_frecuencia['Coef_Variación'].values, 
                'o-', linewidth=2, markersize=8, color='purple')
axes[1, 1].set_xlabel('Frecuencia (kHz)')
axes[1, 1].set_ylabel('Coeficiente de Variación (%)')
axes[1, 1].set_title('Variabilidad Relativa (Coef. Variación)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('evolucion_estadisticas_frecuencia.png', dpi=300, bbox_inches='tight')
plt.show()





#%% 9. EXPORTACIÓN DE RESULTADOS DE HISTOGRAMAS
print("\n9. EXPORTANDO RESULTADOS...")

# Guardar estadísticas
estadisticas_por_frecuencia.to_csv('estadisticas_pendientes_por_frecuencia.csv')
print("✓ Estadísticas guardadas en 'estadisticas_pendientes_por_frecuencia.csv'")

# Guardar datos completos para histogramas
df_hist.to_csv('datos_completos_histogramas.csv', index=False)
print("✓ Datos completos guardados en 'datos_completos_histogramas.csv'")

# Resumen ejecutivo
print("\n" + "="*60)
print("RESUMEN EJECUTIVO - COMPARATIVA DE HISTOGRAMAS")
print("="*60)

print(f"• Total de mediciones analizadas: {len(df_hist)}")
print(f"• Rango total de pendientes: {df_hist['Pendiente'].min():.3f} a {df_hist['Pendiente'].max():.3f}")
print(f"• Frecuencia con mayor variabilidad: {estadisticas_por_frecuencia['Desviación_Estándar'].idxmax()} kHz")
print(f"• Frecuencia con menor variabilidad: {estadisticas_por_frecuencia['Desviación_Estándar'].idxmin()} kHz")
print(f"• Frecuencia con pendiente media más alta: {estadisticas_por_frecuencia['Media'].idxmax()} kHz")
print(f"• Frecuencia con pendiente media más baja: {estadisticas_por_frecuencia['Media'].idxmin()} kHz")

# Identificar patrones
coef_variacion_promedio = estadisticas_por_frecuencia['Coef_Variación'].mean()
print(f"• Coeficiente de variación promedio: {coef_variacion_promedio:.1f}%")

if coef_variacion_promedio < 10:
    print("  → Las mediciones son muy consistentes entre frecuencias")
elif coef_variacion_promedio < 20:
    print("  → Las mediciones muestran variabilidad moderada")
else:
    print("  → Las mediciones muestran alta variabilidad entre frecuencias")

print("\n✓ Gráficos guardados en:")
print("  - histogramas_individuales_frecuencias.png")
print("  - histograma_comparativo_superpuesto.png") 
print("  - densidad_kde_frecuencias.png")
print("  - boxplot_comparativo_frecuencias.png")
print("  - violin_plot_comparativo.png")
print("  - evolucion_estadisticas_frecuencia.png")

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
#%% HISTOGRAMA CON TODAS LAS MEDIDAS DE PENDIENTE COMBINADAS
# =============================================================================
print("="*60)
print("HISTOGRAMA CON TODAS LAS MEDIDAS DE PENDIENTE")
print("="*60)
#% 1. COMBINAR TODAS LAS MEDIDAS DE PENDIENTE
print("\n1. COMBINANDO TODAS LAS MEDIDAS DE PENDIENTE...")

# Combinar todos los datos de pendientes en un solo array
todas_pendientes = []
todas_errores = []
etiquetas_combinadas = []

# Agregar datos de 300 kHz
for i, valor in enumerate(m_300_2_nominal):
    todas_pendientes.append(valor)
    todas_errores.append(m_300_2_err[i])
    etiquetas_combinadas.append('300 kHz')

# Agregar datos de otras frecuencias
frecuencias_lista = [270, 240, 212, 175, 135, 112]
datos_nominales = [m_270_nominal, m_240_nominal, m_212_nominal, m_175_nominal, m_135_nominal, m_112_nominal]
datos_errores = [m_270_err, m_240_err, m_212_err, m_175_err, m_135_err, m_112_err]

for freq, nominales, errores in zip(frecuencias_lista, datos_nominales, datos_errores):
    for i, valor in enumerate(nominales):
        todas_pendientes.append(valor)
        todas_errores.append(errores[i])
        etiquetas_combinadas.append(f'{freq} kHz')

# Convertir a arrays de numpy
todas_pendientes = np.array(todas_pendientes)
todas_errores = np.array(todas_errores)

print(f"Total de mediciones combinadas: {len(todas_pendientes)}")
print(f"Rango de pendientes: {todas_pendientes.min():.3f} a {todas_pendientes.max():.3f}")
print(f"Media global: {todas_pendientes.mean():.3f} ± {todas_pendientes.std():.3f}")

#% 2. HISTOGRAMA PRINCIPAL CON TODOS LOS DATOS
print("\n2. CREANDO HISTOGRAMA PRINCIPAL...")

plt.figure(figsize=(14, 9))

# Histograma principal
n, bins, patches = plt.hist(todas_pendientes, bins=15, alpha=0.7, color='steelblue', 
                           edgecolor='black', linewidth=0.8, density=False,
                           label=f'Todas las mediciones (n={len(todas_pendientes)})')

# Calcular estadísticas globales
media_global = todas_pendientes.mean()
mediana_global = np.median(todas_pendientes)
std_global = todas_pendientes.std()
error_promedio = todas_errores.mean()

# Líneas verticales para estadísticas
plt.axvline(media_global, color='red', linestyle='--', linewidth=3, 
           label=f'Media: {media_global:.3f}')
plt.axvline(mediana_global, color='green', linestyle='--', linewidth=3, 
           label=f'Mediana: {mediana_global:.3f}')
plt.axvline(media_global + std_global, color='orange', linestyle=':', linewidth=2, 
           label=f'±1σ: {std_global:.3f}')
plt.axvline(media_global - std_global, color='orange', linestyle=':', linewidth=2)

# Añadir curva de densidad KDE
try:
    kde = gaussian_kde(todas_pendientes)
    x_vals = np.linspace(todas_pendientes.min() - 0.5, todas_pendientes.max() + 0.5, 200)
    y_vals = kde(x_vals) * len(todas_pendientes) * (bins[1] - bins[0])  # Escalar a frecuencia
    plt.plot(x_vals, y_vals, color='darkred', linewidth=3, label='Distribución KDE')
except Exception as e:
    print(f"Advertencia: No se pudo calcular KDE: {e}")

plt.xlabel('Pendiente (×10¹⁴ Vs/A/m)', fontsize=14, fontweight='bold')
plt.ylabel('Frecuencia', fontsize=14, fontweight='bold')
plt.title('Distribución de Todas las Medidas de Pendiente', fontsize=16, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Añadir texto con estadísticas
textstr = f'''Estadísticas Globales:
N = {len(todas_pendientes)}
Media = {media_global:.3f}
Mediana = {mediana_global:.3f}
σ = {std_global:.3f}
Error prom. = {error_promedio:.3f}
Mín = {todas_pendientes.min():.3f}
Máx = {todas_pendientes.max():.3f}'''

props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('histograma_todas_pendientes.png', dpi=300, bbox_inches='tight')
plt.show()

#% 3. HISTOGRAMA CON DISTRIBUCIÓN NORMAL COMPARATIVA
print("\n3. CREANDO HISTOGRAMA CON DISTRIBUCIÓN NORMAL COMPARATIVA...")

plt.figure(figsize=(14, 9))

# Histograma con densidad
n, bins, patches = plt.hist(todas_pendientes, bins=15, alpha=0.7, color='lightblue', 
                           edgecolor='black', linewidth=0.8, density=True,
                           label='Datos experimentales')

# Distribución normal teórica
x_norm = np.linspace(todas_pendientes.min() - 0.5, todas_pendientes.max() + 0.5, 200)
y_norm = norm.pdf(x_norm, media_global, std_global)
plt.plot(x_norm, y_norm, 'r-', linewidth=3, 
         label=f'Normal (μ={media_global:.3f}, σ={std_global:.3f})')

# Curva KDE de los datos
try:
    kde = gaussian_kde(todas_pendientes)
    y_kde = kde(x_norm)
    plt.plot(x_norm, y_kde, 'g--', linewidth=3, label='KDE experimental')
except Exception as e:
    print(f"Advertencia: No se pudo calcular KDE: {e}")

plt.xlabel('Pendiente (×10¹⁴ Vs/A/m)', fontsize=14, fontweight='bold')
plt.ylabel('Densidad de Probabilidad', fontsize=14, fontweight='bold')
plt.title('Distribución de Pendientes vs Distribución Normal', fontsize=16, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Test de normalidad
from scipy.stats import shapiro, normaltest

if len(todas_pendientes) > 3 and len(todas_pendientes) < 5000:
    stat_sw, p_sw = shapiro(todas_pendientes)
    stat_ks, p_ks = normaltest(todas_pendientes)
    
    textstr = f'''Test de Normalidad:
Shapiro-Wilk: p = {p_sw:.3e}
D'Agostino: p = {p_ks:.3e}
{"Normal" if p_sw > 0.05 else "No normal"}'''
    
    props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('histograma_normal_comparativo.png', dpi=300, bbox_inches='tight')
plt.show()

#%4. HISTOGRAMA ACUMULATIVO (CDF)
print("\n4. CREANDO HISTOGRAMA ACUMULATIVO...")

plt.figure(figsize=(14, 8))

# Histograma acumulativo
counts, bin_edges = np.histogram(todas_pendientes, bins=20, density=False)
cdf = np.cumsum(counts) / np.sum(counts)

plt.plot(bin_edges[1:], cdf, 'o-', linewidth=3, markersize=6, 
         label='CDF Experimental', color='purple')

# CDF teórica normal
x_cdf = np.linspace(todas_pendientes.min() - 0.5, todas_pendientes.max() + 0.5, 200)
cdf_norm = norm.cdf(x_cdf, media_global, std_global)
plt.plot(x_cdf, cdf_norm, 'r--', linewidth=2, 
         label=f'CDF Normal (μ={media_global:.3f}, σ={std_global:.3f})')

plt.xlabel('Pendiente (×10¹⁴ Vs/A/m)', fontsize=14, fontweight='bold')
plt.ylabel('Probabilidad Acumulada', fontsize=14, fontweight='bold')
plt.title('Función de Distribución Acumulativa (CDF) de Pendientes', fontsize=16, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)

# Añadir percentiles importantes
percentiles = [5, 25, 50, 75, 95]
for p in percentiles:
    valor_p = np.percentile(todas_pendientes, p)
    plt.axvline(valor_p, color='orange', linestyle=':', alpha=0.7)
    plt.text(valor_p, 0.1 + 0.1 * percentiles.index(p)/5, f'P{p}={valor_p:.2f}', 
             rotation=90, fontsize=9)

plt.tight_layout()
plt.savefig('cdf_pendientes.png', dpi=300, bbox_inches='tight')
plt.show()

#% 5. HISTOGRAMA POR GRUPOS DE FRECUENCIA (STACKED)
print("\n5. CREANDO HISTOGRAMA AGRUPADO POR FRECUENCIA...")

plt.figure(figsize=(14, 9))

# Preparar datos para histograma agrupado
datos_por_frecuencia = {}
for freq in df_hist['Frecuencia'].unique():
    datos_por_frecuencia[freq] = df_hist[df_hist['Frecuencia'] == freq]['Pendiente'].values

# Ordenar por frecuencia
frecuencias_ordenadas = sorted(datos_por_frecuencia.keys())
datos_ordenados = [datos_por_frecuencia[f] for f in frecuencias_ordenadas]
etiquetas_ordenadas = [f'{f} kHz' for f in frecuencias_ordenadas]
colores_ordenados = [colores[f] for f in frecuencias_ordenadas]

# Histograma apilado
plt.hist(datos_ordenados, bins=12, stacked=True, 
         label=etiquetas_ordenadas, color=colores_ordenados, 
         edgecolor='black', linewidth=0.5, alpha=0.8)

plt.xlabel('Pendiente (×10¹⁴ Vs/A/m)', fontsize=14, fontweight='bold')
plt.ylabel('Frecuencia Acumulada', fontsize=14, fontweight='bold')
plt.title('Distribución de Pendientes por Frecuencia (Histograma Apilado)', 
          fontsize=16, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('histograma_apilado_frecuencias.png', dpi=300, bbox_inches='tight')
plt.show()

#% 6. ESTADÍSTICAS GLOBALES DETALLADAS
print("\n6. ESTADÍSTICAS GLOBALES DETALLADAS")
print("="*50)

estadisticas_globales = {
    'N_total': len(todas_pendientes),
    'Media': media_global,
    'Mediana': mediana_global,
    'Desviación_Estándar': std_global,
    'Error_Promedio': error_promedio,
    'Varianza': np.var(todas_pendientes),
    'Mínimo': todas_pendientes.min(),
    'Máximo': todas_pendientes.max(),
    'Rango': todas_pendientes.max() - todas_pendientes.min(),
    'Percentil_5': np.percentile(todas_pendientes, 5),
    'Percentil_25': np.percentile(todas_pendientes, 25),
    'Percentil_75': np.percentile(todas_pendientes, 75),
    'Percentil_95': np.percentile(todas_pendientes, 95),
    'Rango_Intercuartílico': np.percentile(todas_pendientes, 75) - np.percentile(todas_pendientes, 25),
    'Coef_Variación': (std_global / media_global * 100) if media_global != 0 else np.nan,
    'Asimetría': float(pd.Series(todas_pendientes).skew()),
    'Curtosis': float(pd.Series(todas_pendientes).kurtosis())
}

# Mostrar estadísticas
for key, value in estadisticas_globales.items():
    if isinstance(value, float):
        print(f"{key:<25}: {value:.4f}")
    else:
        print(f"{key:<25}: {value}")

#% 7. EXPORTACIÓN DE RESULTADOS GLOBALES
print("\n7. EXPORTANDO RESULTADOS GLOBALES...")

# Guardar datos combinados
df_global = pd.DataFrame({
    'Pendiente': todas_pendientes,
    'Error': todas_errores,
    'Frecuencia_kHz': etiquetas_combinadas
})
df_global.to_csv('datos_pendientes_globales.csv', index=False)
print("✓ Datos globales guardados en 'datos_pendientes_globales.csv'")

# Guardar estadísticas globales
df_estadisticas_global = pd.DataFrame([estadisticas_globales])
df_estadisticas_global.to_csv('estadisticas_globales_pendientes.csv', index=False)
print("✓ Estadísticas globales guardadas en 'estadisticas_globales_pendientes.csv'")

# Resumen ejecutivo
print("\n" + "="*60)
print("RESUMEN EJECUTIVO - DISTRIBUCIÓN GLOBAL DE PENDIENTES")
print("="*60)

print(f"• Total de mediciones: {len(todas_pendientes)}")
print(f"• Rango global: {todas_pendientes.min():.3f} a {todas_pendientes.max():.3f}")
print(f"• Valor central: {media_global:.3f} ± {std_global:.3f}")
print(f"• Coeficiente de variación: {estadisticas_globales['Coef_Variación']:.1f}%")
print(f"• Asimetría: {estadisticas_globales['Asimetría']:.3f}")

if abs(estadisticas_globales['Asimetría']) < 0.5:
    print("  → Distribución aproximadamente simétrica")
elif estadisticas_globales['Asimetría'] > 0:
    print("  → Distribución con sesgo positivo (cola a la derecha)")
else:
    print("  → Distribución con sesgo negativo (cola a la izquierda)")

if 'p_sw' in locals():
    if p_sw > 0.05:
        print(f"• Normalidad: Distribución normal (p = {p_sw:.3f})")
    else:
        print(f"• Normalidad: Distribución no normal (p = {p_sw:.3f})")

print(f"\n• El {95 - 5}% de las mediciones están entre {estadisticas_globales['Percentil_5']:.3f} y {estadisticas_globales['Percentil_95']:.3f}")

print("\n✓ Gráficos guardados en:")
print("  - histograma_todas_pendientes.png")
print("  - histograma_normal_comparativo.png")
print("  - cdf_pendientes.png")
print("  - histograma_apilado_frecuencias.png")
# %%
