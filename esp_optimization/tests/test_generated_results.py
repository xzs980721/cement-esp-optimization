from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"


def test_expected_artifacts_exist():
    assert (ROOT / "answers" / "逐问解答.md").exists()
    assert len(list((ROOT / "outputs" / "figures").glob("fig*.png"))) >= 11
    assert (TABLES / "optimal_policy_10.csv").exists()


def test_policy_feasibility_and_q4_logic():
    policy10 = pd.read_csv(TABLES / "optimal_policy_10.csv")
    policy5_raw = pd.read_csv(TABLES / "optimal_policy_5_raw_bounds.csv")
    policy5_conditional = pd.read_csv(TABLES / "optimal_policy_5_conditional_extension.csv")
    q4 = pd.read_csv(TABLES / "q4_energy_increase.csv")
    assert policy10["feasible"].all()
    assert (~policy5_raw["feasible"]).sum() >= 1
    assert policy5_conditional["feasible"].all()
    assert (q4["increase_pct"] > 0).all()
    weighted_10 = (q4["share"] * q4["power_10_kW"]).sum()
    weighted_5 = (q4["share"] * q4["power_5_conditional_kW"]).sum()
    increase = 100 * (weighted_5 - weighted_10) / weighted_10
    assert 3.0 < increase < 10.0

