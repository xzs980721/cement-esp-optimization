from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import r2_score

from src.data_audit import _fit_right_censored_normal, load_and_audit_data
from src.emission_twin import fit_emission_twin
from src.power_model import fit_power_surrogate
from src.q1_diagnostics import q1_model_data_consistency


ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT.parent / "Cement_ESP_Data.csv"
CONFIG = yaml.safe_load((ROOT / "config" / "model.yaml").read_text(encoding="utf-8"))


def test_data_invariants():
    audit = load_and_audit_data(CSV, 50.0)
    assert audit.summary["rows"] == 10080
    assert audit.summary["complete_minute_grid"]
    assert audit.summary["output_missing"] == 50
    assert audit.summary["output_censored"] == 5491
    assert abs(audit.summary["output_censored_fraction"] - 5491 / 10030) < 1e-12
    assert abs(audit.summary["output_censored_fraction_all_rows"] - 5491 / 10080) < 1e-12


def test_censored_normal_recovers_synthetic_parameters():
    rng = np.random.default_rng(2026)
    latent = rng.normal(50.15, 0.30, size=50000)
    observed = np.minimum(latent, 50.0)
    fit = _fit_right_censored_normal(observed, 50.0)
    assert abs(fit["mu"] - 50.15) < 0.03
    assert abs(fit["sigma"] - 0.30) < 0.03


def test_power_model_accuracy_and_physical_signs():
    audit = load_and_audit_data(CSV, 50.0)
    model = fit_power_surrogate(audit.frame)
    prediction = model.predict_frame(audit.frame)
    assert r2_score(audit.frame["P_total_kW"], prediction) > 0.995
    assert model.cv_metrics["rmse_kW"].mean() < 7.0
    assert np.all(model.raw_coefficients > 0)


def test_emission_twin_directions():
    audit = load_and_audit_data(CSV, 50.0)
    twin = fit_emission_twin(audit, CONFIG["emission_prior"])
    expected_weights = twin.u_ref**2 / np.sum(twin.u_ref**2)
    assert np.allclose(twin.stage_weights, expected_weights)
    assert np.isclose(twin.stage_weights.sum(), 1.0)
    assert np.isclose(twin.temperature_optimum, audit.frame["Temp_C"].mean())
    assert np.isclose(twin.temp_ref, audit.frame["Temp_C"].mean())
    assert np.isclose(twin.temperature_scale, 25.0)
    reference = np.r_[twin.u_ref, twin.t_ref]
    raised_voltage = reference.copy(); raised_voltage[:4] *= 1.02
    long_cycles = reference.copy(); long_cycles[4:] *= 1.20
    base = twin.predict_deterministic(reference, twin.temp_ref, twin.c_in_ref, twin.q_ref)[0]
    assert twin.predict_deterministic(raised_voltage, twin.temp_ref, twin.c_in_ref, twin.q_ref)[0] < base
    assert twin.predict_deterministic(long_cycles, twin.temp_ref, twin.c_in_ref, twin.q_ref)[0] > base


def test_q1_consistency_separates_marginal_calibration_from_conditional_fit():
    audit = load_and_audit_data(CSV, 50.0)
    twin = fit_emission_twin(audit, CONFIG["emission_prior"])
    diagnostics = q1_model_data_consistency(audit, twin).set_index("metric")

    assert diagnostics.loc["censor_fraction", "absolute_gap"] < 0.002
    assert diagnostics.loc["uncensored_mean_mgNm3", "absolute_gap"] < 0.01
    auc = diagnostics.loc["conditional_censor_auc", "model_or_diagnostic"]
    correlation = diagnostics.loc[
        "uncensored_prediction_correlation", "model_or_diagnostic"
    ]
    assert 0.45 < auc < 0.55
    assert abs(correlation) < 0.05
