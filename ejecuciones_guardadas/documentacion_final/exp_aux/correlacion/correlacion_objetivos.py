"""
correlacion_objetivos.py
------------------------
Calcula la correlación de Spearman entre los cuatro objetivos del problema
de etiquetado de SNPs usando todas las soluciones de los frentes de Pareto
finales de ambos experimentos (ABS y PROP).

Objetivos:
  f1  = Compacidad       (minimizar nº de SNPs)         → columna original
  f2  = Tolerancia       (maximizar dist. Hamming mín)  → negada en CSV: f2_transformed = -f2
  f3  = Disimilitud      (maximizar dist. Hamming media)→ negada en CSV: f3_transformed = -f3
  f4  = Balance          (minimizar varianza)            → columna original

Hipótesis a verificar:
  H1: f1 y f4 están correlacionadas (más SNPs → más varianza)
  H2: f2 y f3 están correlacionadas (mayor tolerancia mín → mayor distancia media)
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas de datos
# ---------------------------------------------------------------------------
BASE = Path("/home/cofer/Documents/University/TFG/snp_tag_tfg/snp_tag/input")
CSV_ABS  = BASE / "ABS"  / "frentes_pareto_full_31.csv"
CSV_PROP = BASE / "PROP" / "frentes_pareto_full_31.csv"

OUT_DIR = Path("/home/cofer/.gemini/antigravity-ide/brain/"
               "8a493ece-273c-4358-8a63-eb5e11e9f4fc/scratch")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Cargar y revertir transformaciones (los objetivos en el CSV están negados)
# ---------------------------------------------------------------------------
def load_and_fix(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["experimento"] = label
    # Revertir la negación de maximización → valores positivos reales
    df["f1"] = df["f1_compactness"]                      # ya positivo (minimizar)
    df["f2"] = -df["f2_transformed_tolerance"]           # deshacer negación → maximizar
    df["f3"] = -df["f3_transformed_hamming_avg"]         # deshacer negación → maximizar
    df["f4"] = df["f4_balance_var"]                      # ya positivo (minimizar)
    return df[["experimento", "algorithm", "f1", "f2", "f3", "f4"]]

frames = []
for path, label in [(CSV_ABS, "ABS"), (CSV_PROP, "PROP")]:
    if path.exists():
        frames.append(load_and_fix(path, label))
        print(f"✓ {label}: {len(frames[-1]):,} soluciones cargadas")
    else:
        print(f"✗ No encontrado: {path}")

if not frames:
    raise FileNotFoundError("No se encontró ningún CSV de frentes de Pareto.")

df = pd.concat(frames, ignore_index=True)
print(f"\nTotal soluciones: {len(df):,}\n")

# ---------------------------------------------------------------------------
# Función para calcular y mostrar matriz de correlación de Spearman
# ---------------------------------------------------------------------------
def spearman_matrix(data: pd.DataFrame, title: str):
    cols = ["f1", "f2", "f3", "f4"]
    labels = ["f1\nCompacidad", "f2\nTolerancia", "f3\nDisimilitud", "f4\nBalance"]
    n = len(cols)
    rho_mat = np.zeros((n, n))
    p_mat   = np.zeros((n, n))

    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"{'Par':<22}  {'ρ Spearman':>12}  {'p-valor':>12}  {'Significativo':>14}")
    print(f"{'-'*62}")

    for i in range(n):
        for j in range(n):
            rho, p = stats.spearmanr(data[cols[i]], data[cols[j]])
            rho_mat[i, j] = rho
            p_mat[i, j]   = p
            if i < j:
                sig = "*** " if p < 0.001 else ("**  " if p < 0.01 else ("*   " if p < 0.05 else "n.s."))
                print(f"  {cols[i]} ↔ {cols[j]:<16}  {rho:>+12.4f}  {p:>12.2e}  {sig:>14}")

    # Heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Correlación
    sns.heatmap(rho_mat, ax=axes[0], annot=True, fmt=".3f", vmin=-1, vmax=1,
                cmap="RdBu_r", xticklabels=labels, yticklabels=labels,
                linewidths=0.5)
    axes[0].set_title(f"Correlación de Spearman (ρ)\n{title}", fontsize=11)

    # p-valores (log scale)
    p_log = -np.log10(np.clip(p_mat, 1e-300, 1))
    sns.heatmap(p_log, ax=axes[1], annot=True, fmt=".1f", vmin=0,
                cmap="YlOrRd", xticklabels=labels, yticklabels=labels,
                linewidths=0.5)
    axes[1].set_title(f"Significancia: -log₁₀(p)\n{title}", fontsize=11)

    plt.tight_layout()
    fname = OUT_DIR / f"corr_spearman_{title.replace(' ', '_').lower()}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  → Figura guardada: {fname}")

    return rho_mat, p_mat

# ---------------------------------------------------------------------------
# 1) Análisis global (todos los experimentos juntos)
# ---------------------------------------------------------------------------
spearman_matrix(df, "Global (ABS + PROP)")

# ---------------------------------------------------------------------------
# 2) Por experimento
# ---------------------------------------------------------------------------
for exp in df["experimento"].unique():
    subset = df[df["experimento"] == exp]
    spearman_matrix(subset, f"Experimento {exp}")

# ---------------------------------------------------------------------------
# 3) Resumen de las hipótesis de interés
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("  RESUMEN — Hipótesis de correlación")
print(f"{'='*60}")
for exp_label, subset in [("Global", df)] + [(e, df[df["experimento"]==e]) for e in df["experimento"].unique()]:
    r12, p12 = stats.spearmanr(subset["f1"], subset["f4"])  # compacidad ↔ balance
    r23, p23 = stats.spearmanr(subset["f2"], subset["f3"])  # tolerancia ↔ disimilitud
    print(f"\n  [{exp_label}]")
    print(f"    H1 — f1 (Compacidad) ↔ f4 (Balance):    ρ={r12:+.4f}  p={p12:.2e}")
    print(f"    H2 — f2 (Tolerancia) ↔ f3 (Disimilitud): ρ={r23:+.4f}  p={p23:.2e}")

print("\nListo. Figuras en:", OUT_DIR)
