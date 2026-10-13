"""
Integration smoke tests for evolutionary execution.
"""

import pytest
import numpy as np

from snp_tag.config import construir_configuracion
from snp_tag.core.algorithm import construir_direcciones_referencia
from snp_tag.pipelines.evolution_pipeline import _ejecutar_replica_individual
from snp_tag.utils.filesystem import crear_arbol_directorios_dataset


def test_evolution_smoke_run(toy_haplotypes, toy_pair_idx, tmp_path):
    """
    Smoke test: executes a single minimal replica in fast mode
    to verify end-to-end execution of genetic search without errors.
    """
    n_snps = toy_haplotypes.shape[1]

    cfg = construir_configuracion(
        modo="fast",
        data_source="synthetic",
        overrides={
            "dir_salida_base": str(tmp_path),
            "modo_evaluacion": "absolute",
            "modo_normalizacion": "static_dataset_limits",
        },
    )
    # Align SNPs with toy dataset
    cfg.n_snps = n_snps
    cfg.pm = 1.0 / n_snps

    # Build output directory structure
    _, carpetas = crear_arbol_directorios_dataset(cfg, "synthetic")
    cfg.carpetas = carpetas

    dirs_ref, _ = construir_direcciones_referencia(tam_poblacion=cfg.tam_poblacion, n_obj=4)

    args = {
        "algo": "NSGA2",
        "init": "random_dense",
        "crossover": "UX",
        "replica": 1,
        "semilla": 42,
        "H": toy_haplotypes,
        "pair_idx": toy_pair_idx,
        "cfg": cfg,
        "dirs_ref": dirs_ref,
    }

    resultado = _ejecutar_replica_individual(args)

    assert resultado is not None
    assert resultado.algoritmo == "NSGA2"
    assert resultado.inicializacion == "random_dense"
    assert resultado.crossover == "UX"
    assert 0 < resultado.X_final.shape[0] <= cfg.tam_poblacion
    assert resultado.X_final.shape[1] == n_snps
    assert resultado.F_final.shape[1] == 4
    assert not np.isnan(resultado.F_final).any()
    assert resultado.tiempo_seg > 0.0
