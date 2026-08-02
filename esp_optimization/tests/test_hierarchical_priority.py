from pathlib import Path

import numpy as np
import pandas as pd

from src.hierarchical_priority import (
    cycle_constraint_relief_table,
    hierarchical_priority_table,
)


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"


class _FakePower:
    @staticmethod
    def predict_policy(policy: np.ndarray) -> float:
        policy = np.asarray(policy, dtype=float)
        return float(np.sum(policy[:4] ** 2) + 1000.0 * np.sum(1.0 / policy[4:]))


class _FakeTwin:
    def __init__(self, dust_scale: float):
        self.dust_scale = dust_scale

    @staticmethod
    def scenario_pack(regime_frame, n_scenarios: int, seed: int):
        return np.zeros(n_scenarios, dtype=float)

    @staticmethod
    def predict_scenarios(policy: np.ndarray, scenarios: np.ndarray) -> np.ndarray:
        policy = np.asarray(policy, dtype=float)
        concentration = 100.0 * np.exp(-0.01 * np.sum(policy[:4] ** 2))
        concentration *= 1.0 + 0.001 * np.sum(policy[4:])
        return np.full(len(scenarios), concentration, dtype=float)

    def dust_load_index(self, policy: np.ndarray, regime_frame) -> float:
        return float(self.dust_scale * np.sum(np.asarray(policy, dtype=float)[4:]))


def _fixture_policy():
    policy = np.array([10.0, 10.5, 11.0, 11.5, 100.0, 105.0, 110.0, 115.0])
    bounds = [(5.0, 15.0)] * 4 + [(50.0, 180.0)] * 4
    frame = pd.DataFrame({"dummy": [1, 2, 3]})
    return policy, bounds, frame


def test_cycle_relief_is_positive_and_voltage_relief_is_zero():
    policy, bounds, frame = _fixture_policy()
    twin = _FakeTwin(dust_scale=1.45 / np.sum(policy[4:]))
    relief = cycle_constraint_relief_table(policy, frame, twin, _FakePower(), bounds)
    assert (relief["constraint_delta_power_kW"] > 0).all()
    assert (relief["dust_load_relief"] > 0).all()
    assert (relief["dust_relief_per_added_kW"] > 0).all()

    actions = hierarchical_priority_table(
        policy,
        frame,
        twin,
        _FakePower(),
        bounds,
        seed=2026,
        regime=7,
        n_scenarios=1000,
    )
    voltage = actions.loc[actions["parameter_type"] == "电压"]
    assert np.allclose(voltage["dust_relief_per_added_kW"], 0.0)


def test_cycle_relief_uses_reverse_difference_at_lower_bound():
    policy, bounds, frame = _fixture_policy()
    policy[4] = bounds[4][0]
    twin = _FakeTwin(dust_scale=1.45 / np.sum(policy[4:]))
    relief = cycle_constraint_relief_table(policy, frame, twin, _FakePower(), bounds)
    first_field = relief.loc[relief["parameter"] == "T1_s"].iloc[0]
    assert first_field["constraint_direction"] == "下限处反向差分"
    assert first_field["constraint_delta_power_kW"] > 0
    assert first_field["dust_load_relief"] > 0
    assert first_field["dust_relief_per_added_kW"] > 0


def test_active_constraint_selects_period_before_voltage():
    policy, bounds, frame = _fixture_policy()
    twin = _FakeTwin(dust_scale=1.45 / np.sum(policy[4:]))
    actions = hierarchical_priority_table(
        policy,
        frame,
        twin,
        _FakePower(),
        bounds,
        seed=2026,
        regime=7,
        n_scenarios=1000,
    )
    assert actions["dust_constraint_active"].all()
    assert actions.iloc[0]["parameter_type"] == "振打周期"
    assert actions.iloc[0]["primary_action_class"] == "周期优先"
    assert actions.loc[actions["parameter_type"] == "振打周期", "operational_rank"].max() <= 4
    assert actions.loc[actions["parameter_type"] == "电压", "operational_rank"].min() >= 5


def test_inactive_constraint_selects_direct_voltage_benefit():
    policy, bounds, frame = _fixture_policy()
    twin = _FakeTwin(dust_scale=1.20 / np.sum(policy[4:]))
    actions = hierarchical_priority_table(
        policy,
        frame,
        twin,
        _FakePower(),
        bounds,
        seed=2026,
        regime=1,
        n_scenarios=1000,
    )
    assert not actions["dust_constraint_active"].any()
    assert actions.iloc[0]["parameter_type"] == "电压"
    assert actions.iloc[0]["primary_action_class"] == "电压优先"


def test_generated_hierarchical_results_are_consistent():
    summary = pd.read_csv(TABLES / "hierarchical_priority_all_regimes.csv")
    stability = pd.read_csv(TABLES / "hierarchical_priority_seed_stability.csv")
    revalidation = pd.read_csv(TABLES / "hierarchical_priority_q2_revalidation.csv")

    assert summary.loc[summary["label"].isin(["R1", "R2", "R3"]), "primary_action_class"].eq(
        "电压优先"
    ).all()
    assert summary.loc[summary["label"].isin(["R4", "R5", "R6", "R7", "R8"]), "primary_action_class"].eq(
        "周期优先"
    ).all()
    assert summary.loc[summary["label"] == "R2", "primary_parameter"].iloc[0] == "U3_kV"
    assert "T3_s" in summary.loc[summary["label"] == "R8", "primary_parameter"].iloc[0]
    assert "T4_s" in summary.loc[summary["label"] == "R8", "primary_parameter"].iloc[0]
    for tolerance in (1e-6, 1e-4, 1e-3):
        classified_active = summary["relative_dust_slack"] <= tolerance
        assert summary.loc[~classified_active, "label"].tolist() == ["R1", "R2", "R3"]
        assert summary.loc[classified_active, "label"].tolist() == ["R4", "R5", "R6", "R7", "R8"]
    assert stability.groupby("label")["primary_action_class"].nunique().eq(1).all()
    assert stability.groupby("label").size().eq(12).all()
    assert revalidation["feasible"].all()
    assert revalidation["power_abs_error_kW"].max() < 1e-8
    assert revalidation["p95_abs_error_mgNm3"].max() < 1e-8
