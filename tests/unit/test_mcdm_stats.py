"""
Unit tests for MCDM decision logic and non-parametric statistical tests.
"""

import pytest
import numpy as np
import pandas as pd

from snp_tag.engine.mcdm_logic import (
    normalizar_minimizacion,
    detectar_knee_points,
    evaluar_estrategias_mcdm,
)
from snp_tag.engine.stats_logic import compute_kruskal_dunn


def test_normalizar_minimizacion():
    """
    Verifies min-max normalisation and direction inversion:
    Col 0 (Compacidad): lower is better -> standard min-max.
    Col 1 (Tolerancia): higher is better -> inverted (1 - x).
    Col 2 (Disimilitud): higher is better -> inverted (1 - x).
    Col 3 (Balance): lower is better -> standard min-max.
    """
    F_real = np.array([
        [10.0, 1.0, 0.2, 0.01],
        [20.0, 5.0, 0.8, 0.05],
    ])

    F_norm = normalizar_minimizacion(F_real)

    assert F_norm.shape == F_real.shape
    # Col 0 (Compacidad): 10.0 should be 0.0 (best), 20.0 should be 1.0 (worst)
    assert F_norm[0, 0] == 0.0
    assert F_norm[1, 0] == 1.0

    # Col 1 (Tolerancia): 1.0 was worst (min) -> now 1.0, 5.0 was best (max) -> now 0.0
    assert F_norm[0, 1] == 1.0
    assert F_norm[1, 1] == 0.0

    # Col 2 (Disimilitud): 0.2 was worst -> now 1.0, 0.8 was best -> now 0.0
    assert F_norm[0, 2] == 1.0
    assert F_norm[1, 2] == 0.0

    # Col 3 (Balance): 0.01 was best -> now 0.0, 0.05 was worst -> now 1.0
    assert F_norm[0, 3] == 0.0
    assert F_norm[1, 3] == 1.0


def test_evaluar_estrategias_mcdm():
    """
    Verifies that MCDM strategies select valid solution indices.
    """
    n_solutions = 10
    rng = np.random.default_rng(42)
    F_norm = rng.uniform(0.0, 1.0, size=(n_solutions, 4))
    pesos = np.array([0.4, 0.2, 0.2, 0.2])

    I_asf, I_pw, pw_mat, I_knee = evaluar_estrategias_mcdm(F_norm, pesos)

    # Selected indices must be within [0, n_solutions - 1]
    assert 0 <= I_asf[0] < n_solutions
    assert 0 <= I_pw[0] < n_solutions
    assert pw_mat.shape == (n_solutions, 4)
    if len(I_knee) > 0:
        assert np.all((I_knee >= 0) & (I_knee < n_solutions))


def test_compute_kruskal_dunn():
    """
    Tests Kruskal-Wallis test and Dunn's post-hoc pairwise testing.
    """
    # Create dataset with distinct groups
    df = pd.DataFrame({
        "config": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
        "hypervolume": np.concatenate([
            np.random.normal(0.8, 0.05, 20),
            np.random.normal(0.5, 0.05, 20),
            np.random.normal(0.2, 0.05, 20),
        ]),
    })

    stat, p_val, p_dunn = compute_kruskal_dunn(df, metrica_objetivo="hypervolume", col_group="config")

    assert stat is not None
    assert p_val < 0.001
    assert p_dunn is not None
    assert p_dunn.shape == (3, 3)
