"""
Unit tests for ProblemaTagSNP evaluation and objective transformation.
"""

import pytest
import numpy as np

from snp_tag.core.problem import (
    ProblemaTagSNP,
    transformar_objetivos_a_minimizacion,
    evaluar_poblacion_vectorizado,
)


def test_problem_evaluation_shapes(toy_problem_absolute):
    """
    Verifies that evaluating a population returns F matrix of shape (M, 4)
    and constraint matrix G of shape (M, 1).
    """
    pop_size = 15
    n_vars = toy_problem_absolute.n_var
    
    # Random boolean population
    rng = np.random.default_rng(123)
    X = rng.choice([True, False], size=(pop_size, n_vars))
    # Ensure at least one SNP is active per individual
    X[:, 0] = True

    out = {}
    toy_problem_absolute._evaluate(X, out)

    assert "F" in out
    assert "G" in out
    assert out["F"].shape == (pop_size, 4)
    assert out["G"].shape == (pop_size, 1)
    assert not np.isnan(out["F"]).any()
    assert not np.isinf(out["F"]).any()


def test_empty_individual_penalization(toy_problem_absolute):
    """
    Verifies that an individual with 0 SNPs selected is marked infeasible
    and severely penalised.
    """
    n_vars = toy_problem_absolute.n_var
    X = np.zeros((1, n_vars), dtype=bool)

    out = {}
    toy_problem_absolute._evaluate(X, out)

    # Constraint G <= 0 indicates feasibility; G > 0 indicates constraint violation
    assert out["G"][0, 0] > 0
    # Objectives should reflect penalty (e.g., massive values)
    assert out["F"][0, 0] > 1e4 or out["F"][0, 1] > 1e4


def test_absolute_vs_proportional_evaluation(toy_problem_absolute, toy_problem_proportional):
    """
    Verifies that proportional evaluation divides Hamming distance (f3) by k
    and variance (f4) by k^2, matching Ting et al. proportional formulation.
    """
    n_vars = toy_problem_absolute.n_var
    # Select all SNPs
    X = np.ones((1, n_vars), dtype=bool)

    out_abs = {}
    toy_problem_absolute._evaluate(X, out_abs)

    out_prop = {}
    toy_problem_proportional._evaluate(X, out_prop)

    F_abs = out_abs["F"][0]
    F_prop = out_prop["F"][0]

    # Compactness (f1, index 0) is identical (k = 20)
    assert F_abs[0] == pytest.approx(F_prop[0])

    # Tolerance (f2, index 1) is identical (kept in absolute terms per Ting formulation)
    assert F_abs[1] == pytest.approx(F_prop[1])

    # Hamming distance (f3, index 2) in proportional is divided by k = 20
    assert F_prop[2] == pytest.approx(F_abs[2] / n_vars)

    # Variance (f4, index 3) in proportional is divided by k^2 = 400
    assert F_prop[3] == pytest.approx(F_abs[3] / (n_vars ** 2))


def test_transformar_objetivos_neg_vs_inverse():
    """
    Verifies objective transformation methods:
    'neg' simply negates values; 'inverse' computes 1 / value.
    """
    k = np.array([5.0])
    cobertura = np.array([2.0])
    hamming = np.array([10.0])
    varianza = np.array([0.5])

    F_neg = transformar_objetivos_a_minimizacion(k, cobertura, hamming, varianza, modo_transformacion="neg")
    assert F_neg[0, 1] == -2.0
    assert F_neg[0, 2] == -10.0

    F_inv = transformar_objetivos_a_minimizacion(k, cobertura, hamming, varianza, modo_transformacion="inverse")
    assert F_inv[0, 1] == pytest.approx(1.0 / 2.0)
    assert F_inv[0, 2] == pytest.approx(1.0 / 10.0)
