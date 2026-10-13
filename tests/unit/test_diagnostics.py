"""
Unit tests for genomic diagnostics and Linkage Disequilibrium (LD) analysis.
"""

import pytest
import numpy as np

from snp_tag.engine.diagnostics_logic import (
    calcular_ld_completo,
    detectar_bloques_ld,
    analizar_similitud_genotipica,
)


def test_calcular_ld_completo(toy_haplotypes):
    """
    Verifies Linkage Disequilibrium Pearson correlation calculations.
    """
    n_snps = toy_haplotypes.shape[1]
    media_ld, corrs, corr_full = calcular_ld_completo(toy_haplotypes)

    assert isinstance(media_ld, float)
    assert 0.0 <= media_ld <= 1.0
    assert corr_full.shape == (n_snps, n_snps)

    # Upper triangular vector length: n_snps * (n_snps - 1) / 2
    expected_pairs = n_snps * (n_snps - 1) // 2
    assert len(corrs) == expected_pairs


def test_detectar_bloques_ld(toy_haplotypes):
    """
    Verifies automatic LD recombination hotspot block boundary detection.
    """
    n_snps = toy_haplotypes.shape[1]
    bloques = detectar_bloques_ld(toy_haplotypes, ventana_suavizado=3, umbral=0.1, min_snps_bloque=5)

    assert isinstance(bloques, list)
    assert len(bloques) >= 1
    # First block starts at 0
    assert bloques[0][0] == 0
    # Last block ends at n_snps
    assert bloques[-1][1] == n_snps

    # Blocks must be contiguous
    for i in range(len(bloques) - 1):
        assert bloques[i][1] == bloques[i + 1][0]


def test_analizar_similitud_genotipica(toy_haplotypes):
    """
    Tests pairwise Hamming distances computation across all haplotype pairs.
    """
    n_hap = toy_haplotypes.shape[0] # 6 haplotypes
    expected_pairs = n_hap * (n_hap - 1) // 2 # 15 pairs

    dvals = analizar_similitud_genotipica(toy_haplotypes)

    assert isinstance(dvals, np.ndarray)
    assert len(dvals) == expected_pairs
    assert np.all(dvals >= 0)
