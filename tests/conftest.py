"""
Shared fixtures and test helpers for snp_tag_tfg test suite.
"""

import itertools
import pytest
import numpy as np

from snp_tag.config import construir_configuracion
from snp_tag.core.problem import ProblemaTagSNP


@pytest.fixture
def toy_haplotypes():
    """
    Returns a deterministic toy binary haplotype matrix (6 haplotypes x 20 SNPs)
    where all haplotype pairs have non-zero Hamming distance.
    """
    rng = np.random.default_rng(42)
    # Generate 6 haplotypes with structured LD
    block1 = rng.choice([0, 1], size=(6, 1), p=[0.7, 0.3])
    block1_expanded = np.repeat(block1, 10, axis=1)
    
    block2 = rng.choice([0, 1], size=(6, 1), p=[0.4, 0.6])
    block2_expanded = np.repeat(block2, 10, axis=1)
    
    H = np.hstack([block1_expanded, block2_expanded]).astype(np.int8)
    
    # Guarantee every pair has distinct SNPs by embedding unique diagonal markers
    for i in range(6):
        H[i, i] = 1
        H[i, i + 6] = 0
    return H


@pytest.fixture
def toy_pair_idx(toy_haplotypes):
    """
    All pairwise combinations of the toy haplotypes (6 choose 2 = 15 pairs).
    """
    n_hap = toy_haplotypes.shape[0]
    pairs = list(itertools.combinations(range(n_hap), 2))
    return np.array(pairs, dtype=int)


@pytest.fixture
def toy_problem_absolute(toy_haplotypes, toy_pair_idx):
    """
    ProblemaTagSNP configured with absolute evaluation and dataset limits.
    """
    return ProblemaTagSNP(
        H=toy_haplotypes,
        pair_idx=toy_pair_idx,
        normalizar_busqueda=False,
        modo_transformacion_objetivos="neg",
        modo_evaluacion="absolute",
        cap_tolerancia=3.0,
    )


@pytest.fixture
def toy_problem_proportional(toy_haplotypes, toy_pair_idx):
    """
    ProblemaTagSNP configured with proportional evaluation.
    """
    return ProblemaTagSNP(
        H=toy_haplotypes,
        pair_idx=toy_pair_idx,
        normalizar_busqueda=False,
        modo_transformacion_objetivos="neg",
        modo_evaluacion="proportional",
        cap_tolerancia=3.0,
    )


@pytest.fixture
def default_config():
    """
    Returns a valid ConfiguracionExperimento with linked absolute configuration.
    """
    return construir_configuracion(
        modo="fast",
        data_source="synthetic",
        overrides={
            "modo_evaluacion": "absolute",
            "modo_normalizacion": "static_dataset_limits",
            "n_snps": 20,
            "num_bloques": 2,
        },
    )
