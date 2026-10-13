"""
Unit tests for data loaders, LD generation, and haplotype matrix filtering.
"""

import pytest
import numpy as np

from snp_tag.data.loader import (
    filtrar_snps_monomorficos,
    generar_bloque_haplotipico_ld,
    cargar_dataset_objetivo,
)


def test_filtrar_snps_monomorficos():
    """
    Verifies monomorphic SNPs (all 0s or all 1s) are correctly detected and stripped.
    """
    H = np.array([
        [0, 1, 0, 1, 0],
        [0, 0, 0, 1, 1],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 1, 1],
    ], dtype=np.int8)

    H_filt, idx_validos = filtrar_snps_monomorficos(H)

    assert H_filt.shape == (4, 2)
    assert np.array_equal(idx_validos, np.array([1, 4]))
    assert np.array_equal(H_filt[:, 0], H[:, 1])
    assert np.array_equal(H_filt[:, 1], H[:, 4])


def test_generar_bloque_haplotipico_ld():
    """
    Verifies synthetic LD haplotype generator constructs matrices with correct
    shapes and binary values.
    """
    n_hap = 20
    n_snps = 100
    tam_bloque = 25

    H, n_bloques, intentos = generar_bloque_haplotipico_ld(
        n_haplotipos=n_hap,
        n_snps=n_snps,
        tam_bloque=tam_bloque,
        prob_flip=0.02,
        semilla=42,
    )

    assert H.shape == (n_hap, n_snps)
    assert set(np.unique(H)).issubset({0, 1})
    assert n_bloques >= 0
    assert intentos >= 0


def test_cargar_dataset_objetivo_synthetic(default_config):
    """
    Tests cargar_dataset_objetivo in synthetic mode.
    """
    H, snp_ids, posiciones, hap_ids = cargar_dataset_objetivo(default_config)
    assert H.shape[0] == default_config.n_haplotipos
    assert H.shape[1] == default_config.n_snps
    assert len(snp_ids) == default_config.n_snps
    assert len(posiciones) == default_config.n_snps
    assert len(hap_ids) == default_config.n_haplotipos
