"""
Unit tests for initialisation and sampling strategies.
"""

import pytest
import numpy as np

from snp_tag.core.sampling import (
    MuestreoAleatorioDisperso,
    MuestreoGreedyMultiCobertura,
    MuestreoGreedyTing,
    MuestreoGreedyHolistico,
)


def test_muestreo_aleatorio_disperso(toy_problem_absolute):
    """
    Verifies MuestreoAleatorioDisperso generates valid boolean matrices
    without empty chromosomes.
    """
    pop_size = 20
    sampler = MuestreoAleatorioDisperso(prob=0.1, semilla=42)
    pop = sampler.do(toy_problem_absolute, n_samples=pop_size)
    X = pop.get("X")

    assert isinstance(X, np.ndarray)
    assert X.shape == (pop_size, toy_problem_absolute.n_var)
    assert X.dtype == bool
    # Non-empty chromosomes constraint
    assert np.all(X.sum(axis=1) > 0)


def test_muestreo_greedy_multi_cobertura(toy_problem_absolute, toy_haplotypes, toy_pair_idx):
    """
    Verifies MuestreoGreedyMultiCobertura generates valid chromosomes.
    """
    pop_size = 15
    sampler = MuestreoGreedyMultiCobertura(
        H=toy_haplotypes,
        pair_idx=toy_pair_idx,
        max_cobertura_objetivo=3.0,
        semilla=42,
    )
    pop = sampler.do(toy_problem_absolute, n_samples=pop_size)
    X = pop.get("X")

    assert isinstance(X, np.ndarray)
    assert X.shape == (pop_size, toy_problem_absolute.n_var)
    assert X.dtype == bool
    assert np.all(X.sum(axis=1) > 0)


def test_muestreo_greedy_ting(toy_problem_absolute, toy_haplotypes, toy_pair_idx):
    """
    Verifies Ting's greedy sampling strategy generates valid chromosomes.
    """
    pop_size = 15
    sampler = MuestreoGreedyTing(
        H=toy_haplotypes,
        pair_idx=toy_pair_idx,
        ratio_greedy=0.7,
        semilla=42,
    )
    pop = sampler.do(toy_problem_absolute, n_samples=pop_size)
    X = pop.get("X")

    assert isinstance(X, np.ndarray)
    assert X.shape == (pop_size, toy_problem_absolute.n_var)
    assert X.dtype == bool
    assert np.all(X.sum(axis=1) > 0)


def test_muestreo_greedy_holistico(toy_problem_absolute, toy_haplotypes, toy_pair_idx):
    """
    Verifies Holistic greedy sampling produces valid chromosomes.
    """
    pop_size = 10
    sampler = MuestreoGreedyHolistico(
        H=toy_haplotypes,
        pair_idx=toy_pair_idx,
        max_k=10,
        semilla=42,
    )
    pop = sampler.do(toy_problem_absolute, n_samples=pop_size)
    X = pop.get("X")

    assert isinstance(X, np.ndarray)
    assert X.shape == (pop_size, toy_problem_absolute.n_var)
    assert X.dtype == bool
    assert np.all(X.sum(axis=1) > 0)
