from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"


def test_expected_artifacts_exist():
    assert (ROOT / "answers" / "逐问解答.md").exists()
    assert len(list((ROOT / "outputs" / "figures").glob("fig*.png"))) >= 11
    assert (TABLES / "optimal_policy_10.csv").exists()
    assert (TABLES / "q1_model_data_consistency.csv").exists()


def test_policy_feasibility_and_q4_logic():
    policy10 = pd.read_csv(TABLES / "optimal_policy_10.csv")
    policy5_raw = pd.read_csv(TABLES / "optimal_policy_5_raw_bounds.csv")
    policy5_conditional = pd.read_csv(TABLES / "optimal_policy_5_conditional_extension.csv")
    q4 = pd.read_csv(TABLES / "q4_energy_increase.csv")
    assert policy10["feasible"].all()
    assert (~policy5_raw["feasible"]).sum() >= 1
    assert policy5_conditional["feasible"].all()
    assert np.allclose(policy5_conditional["voltage_upper_factor"], 1.10)
    assert np.allclose(q4["voltage_upper_factor"], 1.10)
    assert (q4["increase_pct"] > 0).all()
    weighted_10 = (q4["share"] * q4["power_10_kW"]).sum()
    weighted_5 = (q4["share"] * q4["power_5_conditional_kW"]).sum()
    increase = 100 * (weighted_5 - weighted_10) / weighted_10
    assert 4.5 < increase < 6.0


def test_calibrated_model_artifacts_and_constraints():
    expected = [
        "calibrated_uncensored_pmf.csv",
        "calibrated_distribution_daily_cv.csv",
        "calibrated_regime_reference_check.csv",
        "calibrated_optimal_policy_10.csv",
        "calibrated_optimal_policy_10_110pct.csv",
        "calibrated_optimal_policy_5_110pct.csv",
        "calibrated_q4_energy_increase.csv",
    ]
    assert all((TABLES / filename).exists() for filename in expected)
    validation = pd.read_csv(TABLES / "calibrated_distribution_daily_cv.csv")
    assert validation["ks_distance"].mean() < 0.05
    assert validation["wasserstein_mgNm3"].mean() < 0.02
    assert validation["absolute_mean_error_mgNm3"].mean() < 0.02
    assert 0.94 <= validation["interval_coverage"].mean() <= 0.99

    reference = pd.read_csv(TABLES / "calibrated_regime_reference_check.csv")
    assert len(reference) == 8
    assert np.allclose(reference["reference_multiplier"], 1.0, atol=1e-12)

    policy10 = pd.read_csv(TABLES / "calibrated_optimal_policy_10.csv")
    policy10_wide = pd.read_csv(TABLES / "calibrated_optimal_policy_10_110pct.csv")
    policy5_wide = pd.read_csv(TABLES / "calibrated_optimal_policy_5_110pct.csv")
    q4 = pd.read_csv(TABLES / "calibrated_q4_energy_increase.csv")
    assert (~policy10["feasible"]).sum() == 1
    assert policy10_wide["feasible"].all()
    assert policy5_wide["feasible"].all()
    assert (q4["increase_pct"] > 0).all()
    weighted10 = (q4["share"] * q4["power_10_110pct_kW"]).sum()
    weighted5 = (q4["share"] * q4["power_5_110pct_kW"]).sum()
    increase = 100.0 * (weighted5 - weighted10) / weighted10
    assert 4.5 < increase < 6.5
