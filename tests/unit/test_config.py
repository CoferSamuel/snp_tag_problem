"""
Unit tests for configuration loading and parameter validation.
"""

import pytest
from snp_tag.config import (
    construir_configuracion,
    resolver_modo_evaluacion,
    resolver_modo_normalizacion,
)


def test_linked_configurations_validation():
    """
    Ensures coupled configurations enforce compatibility:
    - proportional evaluation requires static_proportional_limits.
    - absolute evaluation requires static_dataset_limits (or non-proportional).
    """
    # 1. Proportional with static_dataset_limits must fail
    with pytest.raises(ValueError, match="Incompatibilidad detectada"):
        construir_configuracion(
            modo="fast",
            data_source="synthetic",
            overrides={
                "modo_evaluacion": "proportional",
                "modo_normalizacion": "static_dataset_limits",
            },
        )

    # 2. Absolute with static_proportional_limits must fail
    with pytest.raises(ValueError, match="Incompatibilidad detectada"):
        construir_configuracion(
            modo="fast",
            data_source="synthetic",
            overrides={
                "modo_evaluacion": "absolute",
                "modo_normalizacion": "static_proportional_limits",
            },
        )

    # 3. Valid pairing 1: proportional + static_proportional_limits
    cfg_prop = construir_configuracion(
        modo="fast",
        data_source="synthetic",
        overrides={
            "modo_evaluacion": "proportional",
            "modo_normalizacion": "static_proportional_limits",
        },
    )
    assert cfg_prop.modo_evaluacion == "proportional"
    assert cfg_prop.modo_normalizacion == "static_proportional_limits"

    # 4. Valid pairing 2: absolute + static_dataset_limits
    cfg_abs = construir_configuracion(
        modo="fast",
        data_source="synthetic",
        overrides={
            "modo_evaluacion": "absolute",
            "modo_normalizacion": "static_dataset_limits",
        },
    )
    assert cfg_abs.modo_evaluacion in ("absolute", "absoluta")
    assert cfg_abs.modo_normalizacion == "static_dataset_limits"


def test_mutation_probability_calculation():
    """
    Tests that mutation probability (pm) is correctly derived as 1 / n_snps.
    """
    cfg = construir_configuracion(
        modo="fast",
        data_source="synthetic",
        overrides={
            "num_bloques": 10,
            "n_snps": 500,
            "modo_evaluacion": "absolute",
            "modo_normalizacion": "static_dataset_limits",
        },
    )
    expected_snps = 500
    assert cfg.n_snps == expected_snps
    assert cfg.pm == pytest.approx(1.0 / expected_snps, rel=1e-5)


@pytest.mark.parametrize("profile_name,expected_gen,expected_pop", [
    ("fast", 2, 10),
    ("medium", 50, 84),
    ("high", 100, 120),
    ("full", 500, 220),
])
def test_profiles_parameters(profile_name, expected_gen, expected_pop):
    """
    Verifies profiles load correct generation and population size presets.
    """
    cfg = construir_configuracion(
        modo=profile_name,
        data_source="synthetic",
        overrides={
            "modo_evaluacion": "absolute",
            "modo_normalizacion": "static_dataset_limits",
        },
    )
    assert cfg.n_generaciones == expected_gen
    assert cfg.tam_poblacion == expected_pop
