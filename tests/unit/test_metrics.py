"""
Unit tests for Pareto indicators, objective decoding, and feasibility masking.
"""

import pytest
import numpy as np

from snp_tag.engine.metrics_logic import (
    decodificar_objetivos_reales,
    mascara_soluciones_factibles,
    filtrar_soluciones_factibles,
    calcular_hipervolumen,
    normalización_min_max,
)


def test_decodificar_objetivos_reales():
    """
    Tests physical metric decoding from optimization space (neg and inverse).
    """
    # [k=5, f2=-2.0, f3=-10.0, f4=0.5]
    F_neg = np.array([[5.0, -2.0, -10.0, 0.5]])
    dec_neg = decodificar_objetivos_reales(F_neg, modo_transformacion_objetivos="neg")
    assert dec_neg["compacidad"][0] == 5.0
    assert dec_neg["min_cobertura"][0] == 2.0
    assert dec_neg["hamming_prom_real"][0] == 10.0
    assert dec_neg["balance_var"][0] == 0.5

    # [k=5, f2=0.5 (1/2), f3=0.1 (1/10), f4=0.5]
    F_inv = np.array([[5.0, 0.5, 0.1, 0.5]])
    dec_inv = decodificar_objetivos_reales(F_inv, modo_transformacion_objetivos="inverse")
    assert dec_inv["min_cobertura"][0] == pytest.approx(2.0)
    assert dec_inv["hamming_prom_real"][0] == pytest.approx(10.0)


def test_mascara_y_filtro_soluciones_factibles():
    """
    Tests feasibility masking where min_cobertura >= 1.0 is required.
    """
    # Solution 0: min_cobertura = 2.0 (feasible)
    # Solution 1: min_cobertura = 0.0 (infeasible)
    # Solution 2: min_cobertura = 1.0 (feasible)
    F = np.array([
        [5.0, -2.0, -10.0, 0.5],
        [5.0, 0.0, -10.0, 0.5],
        [5.0, -1.0, -10.0, 0.5],
    ])

    mask = mascara_soluciones_factibles(F, modo_transformacion_objetivos="neg")
    assert np.array_equal(mask, np.array([True, False, True]))

    F_filtrado = filtrar_soluciones_factibles(F, modo_transformacion_objetivos="neg")
    assert F_filtrado.shape == (2, 4)
    assert np.array_equal(F_filtrado, F[[0, 2]])


def test_calcular_hipervolumen():
    """
    Tests hypervolume computation on a normalised front.
    """
    # Two complementary solutions bounded in [0, 1]^4
    F_norm = np.array([
        [0.2, 0.8, 0.5, 0.5],
        [0.8, 0.2, 0.5, 0.5],
    ])

    hv = calcular_hipervolumen(F_norm)
    assert isinstance(hv, float)
    assert 0.0 < hv < 1.0

    # Empty front returns np.nan per module design
    hv_empty = calcular_hipervolumen(np.empty((0, 4)))
    assert np.isnan(hv_empty)


def test_normalizacion_min_max():
    """
    Tests min-max normalisation against ideal and nadir bounds.
    """
    F = np.array([
        [2.0, 4.0, 6.0, 8.0],
        [4.0, 8.0, 12.0, 16.0],
    ])
    ideal = np.array([0.0, 0.0, 0.0, 0.0])
    nadir = np.array([10.0, 20.0, 30.0, 40.0])

    F_norm = normalización_min_max(F, ideal, nadir)
    assert F_norm.shape == (2, 4)
    assert np.all(F_norm >= 0.0)
    assert np.all(F_norm <= 1.0)
    assert F_norm[0, 0] == pytest.approx(0.2)
    assert F_norm[1, 1] == pytest.approx(0.4)
