"""
Módulo de Reportes Sintéticos (reporting.py)
-------------------------------------------
Genera comparaciones estadísticas globales, rankings y resúmenes de rendimiento
entre los diferentes algoritmos e inicializaciones.
"""

# =============================================================================
# LIBRERÍAS ESTÁNDAR
# =============================================================================
import os
from typing import Any, List, Optional, Tuple

# =============================================================================
# LIBRERÍAS DE TERCEROS
# =============================================================================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import BoundaryNorm, ListedColormap

# =============================================================================
# MÓDULOS LOCALES (snp_tag)
# =============================================================================
from snp_tag.engine.stats_logic import (compute_kruskal_dunn)
from snp_tag.utils.ai_exports import exportar_gemelo_ia_csv
from snp_tag.utils.terminal import (imprimir_grafico_guardado,
                                    imprimir_subseccion)
from snp_tag.constants import HIGHER_IS_BETTER_METRICS, METRICS_DISPLAY_NAMES

def _calcular_intervalos_dunn(means_series: pd.Series, p_dunn: pd.DataFrame, metrica_objetivo: str) -> tuple:
    """Calcula los grupos ordenados, medias e intervalos maximales de equivalencia."""
    is_higher_better = metrica_objetivo in HIGHER_IS_BETTER_METRICS
    sorted_groups = means_series.sort_values(ascending=not is_higher_better)
    groups = sorted_groups.index.tolist()
    means = sorted_groups.values.tolist()
    n = len(groups)
    
    intervals = []
    for i in range(n):
        for j in range(n-1, i, -1):
            g1, g2 = groups[i], groups[j]
            if g1 in p_dunn.index and g2 in p_dunn.columns:
                p = p_dunn.loc[g1, g2]
            elif g2 in p_dunn.index and g1 in p_dunn.columns:
                p = p_dunn.loc[g2, g1]
            else:
                continue
                
            if p >= 0.05:
                intervals.append((i, j))
                break
                
    maximal_intervals = []
    for i, j in intervals:
        is_maximal = True
        for (i2, j2) in intervals:
            if (i2 <= i and j2 >= j) and not (i2 == i and j2 == j):
                is_maximal = False
                break
        if is_maximal:
            maximal_intervals.append((i, j))
            
    return groups, means, maximal_intervals

def dibujar_diagrama_equivalencia_ax(ax: plt.Axes, groups: list, means: list, maximal_intervals: list, metrica_objetivo: str, titulo: str, incluir_titulo: bool = True) -> list:
    """
    Dibuja un diagrama de barras de equivalencia estadística sobre un eje específico usando intervalos precalculados.
    Devuelve los datos de cliques para posible exportación.
    """
    n = len(groups)
            
    y_coords = np.arange(n, 0, -1)
    
    ax.scatter([0]*n, y_coords, color='#F1C40F', zorder=5, label='Valor medio', 
               edgecolors='white', linewidths=1.5, s=100)
    
    for idx, (g, m, y) in enumerate(zip(groups, means, y_coords)):
        ax.annotate(f"{m:.4f}", xy=(0, y), xytext=(-15, 0), textcoords="offset points", 
                    ha='right', va='center', fontsize=10, color='#7F8C8D')
        ax.annotate(g, xy=(0, y), xytext=(-65, 0), textcoords="offset points", 
                    ha='right', va='center', fontsize=10, color='#2C3E50', fontweight='bold')
        
    cliques_data = []
    for line_idx, (i, j) in enumerate(maximal_intervals):
        x_line = line_idx + 1
        y_start = y_coords[i]
        y_end = y_coords[j]
        label = 'Equivalencia estadística (p ≥ 0.05)' if line_idx == 0 else None
        ax.plot([x_line, x_line], [y_start, y_end], color='#222222', linewidth=4, solid_capstyle='round', label=label)
        
        for k in range(i, j+1):
            ax.plot([0, x_line], [y_coords[k], y_coords[k]], color='#58585A', linestyle='-', linewidth=1.5, alpha=0.3)
            
        cliques_data.append(f"Equivalence_Line_{line_idx+1}: {groups[i]} to {groups[j]}")

    ax.set_ylim(0.5, n + 0.5)
    right_limit = max(1, len(maximal_intervals)) + 0.5
    
    max_g_str = max(groups, key=len) if groups else ""
    ax.annotate("0.0000", xy=(right_limit, y_coords[0] if len(y_coords) > 0 else 1), xytext=(15, 0), textcoords="offset points", 
                ha='left', va='center', fontsize=10, alpha=0.0)
    ax.annotate(max_g_str, xy=(right_limit, y_coords[0] if len(y_coords) > 0 else 1), xytext=(65, 0), textcoords="offset points", 
                ha='left', va='center', fontsize=10, fontweight='bold', alpha=0.0)
    
    ax.set_xlim(left=-0.2, right=right_limit)

    if incluir_titulo:
        metrica_mostrar = METRICS_DISPLAY_NAMES.get(metrica_objetivo, metrica_objetivo)
        ax.set_title(f'{metrica_mostrar}', pad=10, fontweight='bold', color='#2C3E50')
    
    ax.axis('off')
    return cliques_data


def graficar_diagrama_equivalencia(means_series: pd.Series, p_dunn: pd.DataFrame, metrica_objetivo: str, dir_salida: str, etiqueta_modo: str, col_group: str, titulo: str, dpi: int = 100) -> None:
    """
    Genera un diagrama de barras de equivalencia estadística individual basado en los p-valores del post-hoc de Dunn.
    """
    n = len(means_series)
    groups, means, maximal_intervals = _calcular_intervalos_dunn(means_series, p_dunn, metrica_objetivo)
    
    fig, ax = plt.subplots(figsize=(max(8, len(maximal_intervals)*0.8 + 6), max(4, n*0.4 + 1.5)))
    
    cliques_data = dibujar_diagrama_equivalencia_ax(ax, groups, means, maximal_intervals, metrica_objetivo, titulo, incluir_titulo=False)
    
    metrica_mostrar = METRICS_DISPLAY_NAMES.get(metrica_objetivo, metrica_objetivo)
    plt.suptitle(f'Ranking y Equivalencia Estadística de Dunn\n{metrica_mostrar} | {titulo}', y=1.05, fontweight='bold', color='#2C3E50')
    fig.legend(title='Leyenda', title_fontsize='small', loc='lower center', 
               bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=True, fontsize='small',
               facecolor='#f8f9fa', edgecolor='#cccccc')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    ruta_cd = os.path.join(dir_salida, f"equivalencia_dunn_{metrica_objetivo.lower()}_{col_group}_{etiqueta_modo}.png")
    plt.savefig(ruta_cd, dpi=dpi, bbox_inches='tight')
    
    df_export = pd.DataFrame({"Group": groups, "Mean": means})
    if cliques_data:
        df_export["Equivalence_Lines"] = str(cliques_data)
        
    exportar_gemelo_ia_csv(ruta_cd, df_datos=df_export)
    
    espacios = " " * 9
    print(f"{espacios}    ", end="")
    imprimir_grafico_guardado(ruta_cd, f"Diagrama de Equivalencias")
    plt.close()

def graficar_diagrama_equivalencia_grid(df_runs: pd.DataFrame, dir_salida: str, metricas: list, etiqueta_modo: str, col_group: str, titulo: str, dpi: int = 100) -> None:
    """
    Genera un grid (cuadrícula) con todos los diagramas de equivalencia de Dunn para un col_group.
    """
    df_plot = df_runs.copy()
    if 'config' not in df_plot.columns:
        df_plot['config'] = df_plot['algorithm'] + '-' + df_plot['init'] + '-' + df_plot['crossover']
    
    n_metricas = len(metricas)
    if n_metricas == 0:
        return
        
    cols = min(3, n_metricas)
    rows = (n_metricas + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4 * rows))
    if n_metricas == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Para la leyenda global
    handles_added = False
    
    for idx, metrica in enumerate(metricas):
        ax = axes[idx]
        stat, p_val, p_dunn = compute_kruskal_dunn(df_plot, metrica, col_group)
        if p_dunn is None:
            ax.axis('off')
            metrica_mostrar = METRICS_DISPLAY_NAMES.get(metrica, metrica)
            ax.set_title(f'{metrica_mostrar}', pad=10, fontweight='bold', color='#2C3E50')
            ax.text(0.5, 0.5, 'Kruskal-Wallis omitido', ha='center', va='center', color='gray')
            continue
            
        means_series = df_plot.groupby(col_group)[metrica].mean()
        groups, means, maximal_intervals = _calcular_intervalos_dunn(means_series, p_dunn, metrica)
        dibujar_diagrama_equivalencia_ax(ax, groups, means, maximal_intervals, metrica, titulo)
        
        if not handles_added and ax.get_legend_handles_labels()[0]:
            handles, labels = ax.get_legend_handles_labels()
            handles_added = True
            
    # Ocultar ejes vacíos si los hay
    for idx in range(n_metricas, len(axes)):
        axes[idx].axis('off')
        
    plt.suptitle(f'Ranking y Equivalencia Estadística de Dunn | {titulo}', fontsize=20, fontweight='bold', color='#2C3E50', y=1.02)
    
    if handles_added:
        fig.legend(handles, labels, title='Leyenda', title_fontsize='large', loc='lower center', 
                   bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=True, fontsize='medium',
                   facecolor='#f8f9fa', edgecolor='#cccccc')
                   
    plt.tight_layout()
    
    ruta_cd = os.path.join(dir_salida, f"equivalencia_dunn_grid_{col_group}_{etiqueta_modo}.png")
    plt.savefig(ruta_cd, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    espacios = " " * 9
    print(f"{espacios}    ", end="")
    imprimir_grafico_guardado(ruta_cd, f"Grid Diagrama Equivalencias ({titulo})")


def graficar_rendimiento_tiempo(df_runs: pd.DataFrame, dir_salida: str, etiqueta_modo: str, dpi: int = 100) -> None:
    """
    Visualiza el tiempo de ejecución promedio por cada configuración algorítmica.

    Parámetros:
    -----------
    df_runs : pd.DataFrame
        Dataset con los tiempos de ejecución de cada iteración independiente.
    dir_salida : str
        Directorio destino para exportar la figura.
    etiqueta_modo : str
        Sufijo identificador del experimento.
    dpi : int
        Resolución de la imagen generada.
    """
    df_plot = df_runs.copy()
    df_plot['config'] = df_plot['algorithm'] + '-' + df_plot['init'] + '-' + df_plot['crossover']
    n_configs = df_plot['config'].nunique()
    ancho_dinamico = max(12, n_configs * 0.35)
    

    
    # 2. Media +- Std (Barplot)
    plt.figure(figsize=(ancho_dinamico, 6))
    sns.barplot(data=df_plot, x='config', y='time_seg', hue='config', legend=False, errorbar='sd', capsize=.2)
    plt.title('Media ± Desviación Estándar del Tiempo de Ejecución')
    plt.xticks(rotation=35, ha='right', rotation_mode='anchor')
    plt.tight_layout()
    ruta_std = os.path.join(dir_salida, f'media_std_tiempo_{etiqueta_modo}.png')
    plt.savefig(ruta_std, dpi=dpi, bbox_inches='tight')
    exportar_gemelo_ia_csv(ruta_std, df_datos=df_plot[['config', 'time_seg']])
    imprimir_grafico_guardado(ruta_std, "Media ± std tiempo ejecución")
    

        
    plt.close('all')

def graficar_comparativa_objetivos(df_runs: pd.DataFrame, dir_salida: str, etiqueta_modo: str, dpi: int = 100) -> None:
    """
    Genera un heatmap comparativo de rendimiento escalado entre todas las métricas.

    Parámetros:
    -----------
    df_runs : pd.DataFrame
        Dataset consolidado con las métricas finales.
    dir_salida : str
        Ruta de exportación.
    etiqueta_modo : str
        Sufijo identificador.
    dpi : int
        Calidad de imagen.
    """
    # 1. Agregación y preparación
    cols_met = ['Range', 'SumMin', 'MinSum', 'MaxToleranceRate', 'AvgToleranceRate', 'AvgHammingDistance', 'Hypervolume', 'IGD+', 'GD+']
    disponibles = [c for c in cols_met if c in df_runs.columns]
    
    resumen = df_runs.groupby(['algorithm', 'init', 'crossover'])[disponibles].mean().reset_index()
    resumen['method'] = resumen['algorithm'] + '-' + resumen['init'] + '-' + resumen['crossover']
    
    heat_df_plot = resumen.set_index('method')[disponibles].copy()
    heat_norm_better = heat_df_plot.copy()
    
    # 2. Lógica de Normalización de Calidad (Legacy)
    higher_is_better = ['Hypervolume', 'Range', 'MaxToleranceRate', 'AvgToleranceRate', 'AvgHammingDistance']
    lower_is_better = ['SumMin', 'MinSum', 'IGD+', 'GD+']
    
    for col in disponibles:
        c_min, c_max = heat_df_plot[col].min(), heat_df_plot[col].max()
        c_range = c_max - c_min
        if c_range == 0:
            heat_norm_better[col] = 1.0
        elif any(k in col for k in higher_is_better):
            heat_norm_better[col] = (heat_df_plot[col] - c_min) / c_range
        else:
            # Para métricas donde menos es mejor, invertimos el rango [0, 1]
            heat_norm_better[col] = (c_max - heat_df_plot[col]) / c_range
    
    n_configs = len(heat_df_plot)
    alto_dinamico = max(8.0, n_configs * 0.35)
    ancho_dinamico = max(14.0, len(disponibles) * 0.8)

    # 3. Graficado exacto
    plt.figure(figsize=(ancho_dinamico, alto_dinamico))
    ax = sns.heatmap(heat_norm_better, annot=heat_df_plot, fmt='.3f', cmap='RdYlGn', linewidths=0.5)
    
    # Personalizar la barra de color (leyenda)
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Peor', 'Mejor'])
    
    plt.title('Comparativa de Algoritmos: Rojo (Peor) vs Verde (Mejor)', fontsize=15, pad=20)
    plt.tight_layout()
    
    ruta_h = os.path.join(dir_salida, f"heatmap_comparativa_{etiqueta_modo}.png")
    plt.savefig(ruta_h, dpi=dpi, bbox_inches='tight')
    exportar_gemelo_ia_csv(ruta_h, df_datos=heat_df_plot.reset_index())
    imprimir_grafico_guardado(ruta_h, "Mapa de calor comparativo (Benchmark)")
    plt.close()

def graficar_violin_metricas(df_runs: pd.DataFrame, dir_salida: str, etiqueta_modo: str,
                             dpi: int = 100, emitir_log: bool = True) -> List[Tuple[str, str]]:
    """
    Genera diagramas de violín para representar las distribuciones de rendimiento.

    Parámetros:
    -----------
    df_runs : pd.DataFrame
        Dataset con métricas finales.
    dir_salida : str
        Directorio base de guardado.
    etiqueta_modo : str
        Sufijo del archivo.
    dpi : int
        Resolución del gráfico.
    emitir_log : bool
        Indica si se reporta la generación en la consola.

    Retorna:
    --------
    List[Tuple[str, str]]
        Rutas y descripciones de las imágenes exportadas.
    """
    metricas = ['Range', 'SumMin', 'MinSum', 'MaxToleranceRate', 'AvgToleranceRate', 'AvgHammingDistance', 'Hypervolume', 'IGD+', 'GD+']
    disponibles = [m for m in metricas if m in df_runs.columns]
    if not disponibles:
        return []
    df_plot = df_runs.copy()
    df_plot['config'] = df_plot['algorithm'] + '-' + df_plot['init'] + '-' + df_plot['crossover']
    n_configs = df_plot['config'].nunique()
    ancho_dinamico = max(12.0, n_configs * 0.35)
    artefactos = []
    
    # 2. Individuales
    for m in disponibles:
        plt.figure(figsize=(ancho_dinamico, 6))
        sns.violinplot(data=df_plot, x='config', y=m, inner="quart", hue='config', legend=False)
        sns.stripplot(data=df_plot, x='config', y=m, color="black", alpha=0.3, size=3)
        plt.title(f'Distribución Detallada: {m} (Violin Plot)')
        plt.xticks(rotation=35, ha='right', rotation_mode='anchor')
        plt.tight_layout()
        ruta_i = os.path.join(dir_salida, f'violin_metricas_{m}_{etiqueta_modo}.png')
        plt.savefig(ruta_i, dpi=dpi, bbox_inches='tight')
        exportar_gemelo_ia_csv(ruta_i, df_datos=df_plot[['config', m]])
        artefactos.append((ruta_i, f"Distribución {m} (Violin)"))
        if emitir_log:
            imprimir_grafico_guardado(ruta_i, f"Distribución {m} (Violin)")
    plt.close('all')
    return artefactos

def graficar_media_std_metricas(df_runs: pd.DataFrame, dir_salida: str, etiqueta_modo: str,
                                dpi: int = 100, emitir_log: bool = True) -> List[Tuple[str, str]]:
    """
    Genera diagramas de barras con intervalos de confianza de desviación estándar.

    Parámetros:
    -----------
    df_runs : pd.DataFrame
        Dataset con métricas finales.
    dir_salida : str
        Directorio destino.
    etiqueta_modo : str
        Sufijo de nomenclatura.
    dpi : int
        Resolución visual.
    emitir_log : bool
        Manejo de registros por consola.

    Retorna:
    --------
    List[Tuple[str, str]]
        Lista de recursos de imagen exportados.
    """
    metricas = ['Range', 'SumMin', 'MinSum', 'MaxToleranceRate', 'AvgToleranceRate', 'AvgHammingDistance', 'Hypervolume', 'IGD+', 'GD+']
    disponibles = [m for m in metricas if m in df_runs.columns]
    if not disponibles:
        return []
    df_plot = df_runs.copy()
    df_plot['config'] = df_plot['algorithm'] + '-' + df_plot['init'] + '-' + df_plot['crossover']
    n_configs = df_plot['config'].nunique()
    ancho_dinamico = max(12.0, n_configs * 0.35)
    artefactos = []

    # 2. Individuales (Barplots)
    for m in disponibles:
        plt.figure(figsize=(ancho_dinamico, 6))
        sns.barplot(data=df_plot, x='config', y=m, hue='config', legend=False, errorbar='sd', capsize=.2)
        plt.title(f'Media ± Desviación Estándar de {m}')
        plt.xticks(rotation=35, ha='right', rotation_mode='anchor')
        plt.tight_layout()
        ruta_i = os.path.join(dir_salida, f'media_std_metricas_{m}_{etiqueta_modo}.png')
        plt.savefig(ruta_i, dpi=dpi, bbox_inches='tight')
        exportar_gemelo_ia_csv(ruta_i, df_datos=df_plot[['config', m]])
        artefactos.append((ruta_i, f"Media ± std {m}"))
        if emitir_log:
            imprimir_grafico_guardado(ruta_i, f"Media ± std {m}")
    plt.close('all')
    return artefactos

def graficar_boxplot_metricas(df_runs: pd.DataFrame, dir_salida: str, etiqueta_modo: str,
                              dpi: int = 100, emitir_log: bool = True) -> List[Tuple[str, str]]:
    """
    Genera diagramas de caja para evaluar la dispersión intercuartil.

    Parámetros:
    -----------
    df_runs : pd.DataFrame
        Dataset estadístico.
    dir_salida : str
        Destino en disco.
    etiqueta_modo : str
        Clave identificadora.
    dpi : int
        Nitidez de salida.
    emitir_log : bool
        Logs por terminal.

    Retorna:
    --------
    List[Tuple[str, str]]
        Detalles de los gráficos exportados.
    """
    metricas = ['Range', 'SumMin', 'MinSum', 'MaxToleranceRate', 'AvgToleranceRate', 'AvgHammingDistance', 'Hypervolume', 'IGD+', 'GD+']
    disponibles = [m for m in metricas if m in df_runs.columns]
    if not disponibles:
        return []
    
    df_plot = df_runs.copy()
    df_plot['config'] = df_plot['algorithm'] + '-' + df_plot['init'] + '-' + df_plot['crossover']
    n_configs = df_plot['config'].nunique()
    ancho_dinamico = max(12.0, n_configs * 0.35)
    artefactos = []
    
    # 2. Individuales
    for m in disponibles:
        plt.figure(figsize=(max(10.0, n_configs * 0.35), 6))
        sns.boxplot(data=df_plot, x='config', y=m, hue='config', legend=False)
        plt.title(f'Distribución de {m} (Boxplot)')
        plt.xticks(rotation=35, ha='right', rotation_mode='anchor')
        plt.tight_layout()
        ruta_i = os.path.join(dir_salida, f'boxplot_metricas_{m}_{etiqueta_modo}.png')
        plt.savefig(ruta_i, dpi=dpi, bbox_inches='tight')
        exportar_gemelo_ia_csv(ruta_i, df_datos=df_plot[['config', m]])
        artefactos.append((ruta_i, f"Distribución {m} (Boxplot)"))
        if emitir_log:
            imprimir_grafico_guardado(ruta_i, f"Distribución {m} (Boxplot)")
    plt.close('all')
    return artefactos


def graficar_analisis_kruskal_dunn(df_runs: pd.DataFrame, dir_salida: str, metrica_objetivo: str, etiqueta_modo: str, col_group: str = 'config', dpi: int = 100, indent: int = 9, graficar: bool = True, dir_equivalencia: str = None) -> None:
    """
    Evalúa contrastes no paramétricos multivariables y exporta un heatmap de los p-values.

    Parámetros:
    -----------
    df_runs : pd.DataFrame
        Histórico de rendimiento por ejecución.
    dir_salida : str
        Carpeta destino.
    metrica_objetivo : str
        Nombre del indicador de rendimiento evaluado.
    etiqueta_modo : str
        Distintivo del modo.
    dpi : int
        Puntos por pulgada.
    indent : int
        Espacios de margen en terminal.
    """
    if df_runs.empty or metrica_objetivo not in df_runs.columns: return
    
    espacios = " " * indent
    df_plot = df_runs.copy()
    if 'config' not in df_plot.columns:
        df_plot['config'] = df_plot['algorithm'] + '-' + df_plot['init'] + '-' + df_plot['crossover']
    
    n_configs = df_plot[col_group].nunique()
    
    stat, p_val, p_dunn = compute_kruskal_dunn(df_plot, metrica_objetivo, col_group)
    
    titulos = {
        'config': 'Configuración (Alg + Init + Cruce)',
        'algorithm': 'Algoritmo',
        'init': 'Inicialización',
        'crossover': 'Cruce'
    }
    titulo = titulos.get(col_group, col_group)

    sub_line = "─" * (len(metrica_objetivo) + len(titulo) + 31)
    if stat is None:
        print(f"\n{espacios}📊  \033[1mVALIDACIÓN ESTADÍSTICA ({metrica_objetivo} | {titulo})\033[0m")
        print(f"{espacios}{sub_line}")
        print(f"{espacios}    ⚠️  Kruskal-Wallis omitido: se requieren al menos 2 grupos.\n")
        return

    print(f"\n{espacios}📊  \033[1mVALIDACIÓN ESTADÍSTICA ({metrica_objetivo} | {titulo})\033[0m")
    print(f"{espacios}{sub_line}")
    print(f"{espacios}    Estadístico H (Kruskal-Wallis): {stat:.4f}")
    print(f"{espacios}    P-valor: {p_val:.4e}")
    print(f"{espacios}    Significativo (p < 0.05): {'Sí' if p_val < 0.05 else 'No'}\n")
    
    if p_dunn is not None and graficar:
        
        tamano_hm = max(14.0, n_configs * 0.4)
        plt.figure(figsize=(tamano_hm, tamano_hm))
        mask = np.triu(np.ones_like(p_dunn, dtype=bool))
        cmap = ListedColormap(['#F1C40F', '#F9E79F', '#58585A'])
        norm = BoundaryNorm([0, 0.01, 0.05, 1.0], cmap.N)
        
        ax = sns.heatmap(p_dunn, mask=mask, annot=False, cmap=cmap, norm=norm, 
                    linewidths=0.5, linecolor='white',
                    cbar_kws={"ticks": [0.005, 0.03, 0.525]})
        cbar = ax.collections[0].colorbar
        cbar.set_ticklabels(['Alta Sig. (p < 0.01)', 'Sig. (p < 0.05)', 'No Sig. (p \u2265 0.05)'])
        cbar.set_label("Nivel de Significancia", size=12)
        
        plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
        plt.suptitle(f'Post-hoc de Dunn: {metrica_objetivo}', fontsize=16, y=0.98)
        plt.tight_layout()
        
        ruta_h = os.path.join(dir_salida, f"heatmap_dunn_{metrica_objetivo.lower()}_{col_group}_{etiqueta_modo}.png")
        plt.savefig(ruta_h, dpi=dpi, bbox_inches='tight')
        exportar_gemelo_ia_csv(ruta_h, df_datos=p_dunn)
        
        print(f"{espacios}    ", end="")
        imprimir_grafico_guardado(ruta_h, f"Heatmap de Comparaciones Dunn (Post-hoc)")
        plt.close()
        
        # Generar Diagrama de Equivalencia (CD)
        means_series = df_plot.groupby(col_group)[metrica_objetivo].mean()
        dir_eq = dir_equivalencia if dir_equivalencia else dir_salida
        graficar_diagrama_equivalencia(means_series, p_dunn, metrica_objetivo, dir_eq, etiqueta_modo, col_group, titulo, dpi)



