"""
Unit tests for genetic operators, repair operator, and algorithm builders.
"""

import pytest
import numpy as np
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.core.problem import Problem
from pymoo.core.population import Population

from snp_tag.core.algorithm import ReparacionSNP, fabricar_algoritmo, construir_direcciones_referencia


class MockProblem(Problem):
    def __init__(self, n_var):
        super().__init__(n_var=n_var, n_obj=4, xl=0, xu=1, vtype=bool)


def test_mutation_rate_statistical_invariant():
    """
    STATISTICAL INVARIANT TEST:
    Verifies that BitflipMutation configured with prob=1.0 and prob_var=1/N
    yields an expected average of ~1.0 flipped bit per individual across the population.
    This test serves as a regression guard against the (1/N)^2 silent degradation bug.
    """
    n_vars = 1000
    n_individuals = 200
    problem = MockProblem(n_var=n_vars)
    
    # 1. Correct configuration (prob=1.0, prob_var=1.0 / n_vars)
    mutation_correct = BitflipMutation(prob=1.0, prob_var=1.0 / n_vars)
    
    # Start with all-zero matrix wrapped in a Population object
    pop_zeros = Population.new("X", np.zeros((n_individuals, n_vars), dtype=bool))
    mutated_pop = mutation_correct.do(problem, pop_zeros)
    mutated = mutated_pop.get("X")
    
    flips_per_individual = np.sum(mutated, axis=1)
    mean_flips = np.mean(flips_per_individual)
    
    # E[X] = 1.0, with N=1000 and M=200, standard error is 1 / sqrt(200) ~ 0.07
    # 99.9% confidence interval is well within [0.75, 1.25]
    assert 0.75 <= mean_flips <= 1.25, (
        f"Expected ~1.0 mean flips per individual, but got {mean_flips:.4f}. "
        "Check prob vs prob_var in BitflipMutation!"
    )
    # Ensure almost all individuals underwent evaluation
    individuals_with_mutation = np.sum(flips_per_individual > 0)
    assert individuals_with_mutation > 80, (
        f"Too few mutating individuals: {individuals_with_mutation} out of {n_individuals}"
    )

    # 2. Buggy legacy configuration (prob=1.0 / n_vars, prob_var=None)
    # Verify that the old faulty configuration drastically fails this invariant
    mutation_buggy = BitflipMutation(prob=1.0 / n_vars)
    pop_zeros_buggy = Population.new("X", np.zeros((n_individuals, n_vars), dtype=bool))
    mutated_buggy = mutation_buggy.do(problem, pop_zeros_buggy).get("X")
    flips_buggy = np.sum(mutated_buggy)
    assert flips_buggy <= 5, (
        "Buggy configuration was expected to produce practically zero flips"
    )


@pytest.mark.xfail(strict=True, reason="REGRESSION PROOF: Demonstrates that an invariant check catches the legacy (1/N)^2 bug")
def test_regression_catches_legacy_buggy_mutation():
    """
    Subjects the legacy faulty configuration:
        BitflipMutation(prob=1.0 / n_vars)
    to the standard genetic algorithm invariant:
        E[flips per individual] ~ 1.0.

    When executed with --runxfail, this test FAILS with an AssertionError,
    proving mathematically and programmatically that this test catches the bug.
    """
    n_vars = 772
    n_individuals = 220
    problem = MockProblem(n_var=n_vars)

    # Legacy buggy configuration from previous runs
    mutation_buggy = BitflipMutation(prob=1.0 / n_vars)

    pop_zeros = Population.new("X", np.zeros((n_individuals, n_vars), dtype=bool))
    mutated = mutation_buggy.do(problem, pop_zeros).get("X")
    flips_per_individual = np.sum(mutated, axis=1)
    mean_flips = np.mean(flips_per_individual)

    # The invariant requirement: average ~1.0 bit flip per chromosome
    assert 0.75 <= mean_flips <= 1.25, (
        f"CRITICAL DEFECT DETECTED: Expected ~1.0 mean flips per individual, "
        f"but got {mean_flips:.4f}! Mutation has silently collapsed."
    )


def test_reparacion_snp_empty_individual():
    """
    Verifies that ReparacionSNP fixes all-false chromosomes by activating at least one SNP,
    while leaving valid non-empty chromosomes unaltered.
    """
    n_vars = 20
    problem = MockProblem(n_var=n_vars)
    repair = ReparacionSNP()

    # Matrix with 3 empty individuals and 2 non-empty individuals
    Z = np.zeros((5, n_vars), dtype=bool)
    Z[3, 5] = True
    Z[4, [2, 7, 11]] = True

    pop = Population.new("X", Z.copy())
    pop_repaired = repair.do(problem, pop)
    Z_repaired = pop_repaired.get("X")

    # All individuals must now have >= 1 active SNP
    assert np.all(Z_repaired.sum(axis=1) >= 1)

    # Non-empty individuals should remain intact
    assert Z_repaired[3, 5] == True
    assert np.array_equal(Z_repaired[4], Z[4])


@pytest.mark.parametrize("algo_name", ["NSGA2", "SPEA2", "NSGA3", "UNSGA3", "RVEA", "SMSEMOA", "AGEMOEA2"])
@pytest.mark.parametrize("crossover_name", ["UX", "HUX", "1P", "2P"])
def test_fabricar_algoritmo_builds_all_models(
    toy_problem_absolute, toy_haplotypes, default_config, algo_name, crossover_name
):
    """
    Tests that fabricar_algoritmo constructs all 7 algorithms with all 4 crossover variants.
    """
    dirs_ref, _ = construir_direcciones_referencia(tam_poblacion=default_config.tam_poblacion, n_obj=4)
    algo = fabricar_algoritmo(
        problema=toy_problem_absolute,
        H=toy_haplotypes,
        nombre_algo=algo_name,
        nombre_init="random_dense",
        nombre_crossover=crossover_name,
        cfg=default_config,
        semilla=42,
        dirs_ref=dirs_ref,
    )
    assert algo is not None
    assert hasattr(algo, "pop_size")
    assert algo.pop_size == default_config.tam_poblacion
