from pathlib import Path

import numpy as np
import yaml

from src.calibrated_emission import (
    evaluate_distribution_fit,
    fit_uncensored_distribution,
    regime_reference,
    relative_physical_response,
    sample_calibrated_emissions,
)
from src.data_audit import load_and_audit_data
from src.emission_twin import fit_emission_twin


ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT.parent / "Cement_ESP_Data.csv"
CONFIG = yaml.safe_load((ROOT / "config" / "model.yaml").read_text(encoding="utf-8"))


def _models():
    audit = load_and_audit_data(CSV, 50.0)
    twin = fit_emission_twin(audit, CONFIG["emission_prior"])
    calibrated = CONFIG["calibrated_model"]
    distribution = fit_uncensored_distribution(
        audit.frame,
        threshold=calibrated["threshold_mgNm3"],
        resolution=calibrated["resolution_mgNm3"],
        dirichlet_alpha=calibrated["dirichlet_alpha"],
    )
    return audit, twin, distribution


def test_uncensored_distribution_matches_attachment():
    audit, _, distribution = _models()
    assert len(distribution.observations) == 4539
    assert np.isclose(distribution.point_prediction, 49.76138576779026)
    assert np.isclose(np.std(distribution.observations, ddof=1), 0.1800784459593051)
    assert np.isclose(distribution.probabilities.sum(), 1.0)
    draws = distribution.sample(20_000, seed=2026, draw_distribution=False)
    assert abs(draws.mean() - distribution.point_prediction) < 0.01

    validation = evaluate_distribution_fit(audit.frame)
    assert validation["ks_distance"].mean() < 0.05
    assert validation["wasserstein_mgNm3"].mean() < 0.02
    assert validation["absolute_mean_error_mgNm3"].mean() < 0.02
    assert 0.94 <= validation["interval_coverage"].mean() <= 0.99


def test_relative_physical_layer_is_anchored_and_directional():
    audit, twin, _ = _models()
    conditions, reference = regime_reference(audit.frame)
    base = relative_physical_response(
        twin, reference, conditions, reference, conditions
    )
    assert np.isclose(base, 1.0)

    raised_voltage = reference.copy()
    raised_voltage[:4] *= 1.02
    long_cycles = reference.copy()
    long_cycles[4:] *= 1.05
    assert (
        relative_physical_response(
            twin, raised_voltage, conditions, reference, conditions
        )
        < 1.0
    )
    assert (
        relative_physical_response(
            twin, long_cycles, conditions, reference, conditions
        )
        > 1.0
    )
    high_load = dict(conditions)
    high_load["c_in"] *= 1.05
    assert (
        relative_physical_response(
            twin, reference, high_load, reference, conditions
        )
        > 1.0
    )


def test_calibrated_samples_are_finite_and_reproducible():
    audit, twin, distribution = _models()
    conditions, reference = regime_reference(audit.frame)
    first = sample_calibrated_emissions(
        twin,
        distribution,
        audit.frame,
        reference,
        reference_policy=reference,
        n_scenarios=1000,
        seed=2026,
    )
    second = sample_calibrated_emissions(
        twin,
        distribution,
        audit.frame,
        reference,
        reference_policy=reference,
        n_scenarios=1000,
        seed=2026,
    )
    assert np.all(np.isfinite(first))
    assert np.all(first > 0)
    assert np.allclose(first, second)
