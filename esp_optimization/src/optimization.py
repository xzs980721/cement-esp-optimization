from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

from .emission_twin import EmissionScenarios, EmissionTwin
from .power_model import PowerSurrogate


POLICY_COLUMNS = [f"U{i}_kV" for i in range(1, 5)] + [f"T{i}_s" for i in range(1, 5)]


@dataclass
class OptimizationResult:
    limit: float
    table: pd.DataFrame
    policies: dict[int, np.ndarray]
    training_scenarios: dict[int, EmissionScenarios]


def optimize_single_regime(
    twin: EmissionTwin,
    power: PowerSurrogate,
    full_frame: pd.DataFrame,
    regime_frame: pd.DataFrame,
    regime: int,
    limit: float,
    optimization_config: dict[str, Any],
    seed: int,
    voltage_upper_factor: float = 1.0,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    policy, summary, _ = _optimize_one(
        regime=regime,
        regime_frame=regime_frame.reset_index(drop=True),
        full_frame=full_frame,
        twin=twin,
        power=power,
        limit=float(limit),
        confidence=float(optimization_config["confidence"]),
        scenario_quantile_buffer=float(
            optimization_config.get("scenario_quantile_buffer", 0.025)
        ),
        n_scenarios=int(optimization_config["training_scenarios"]),
        verification_scenarios=int(optimization_config["verification_scenarios"]),
        maxiter=int(optimization_config["differential_evolution_maxiter"]),
        popsize=int(optimization_config["differential_evolution_popsize"]),
        dust_limit=float(optimization_config["dust_load_limit"]),
        penalty_scale=float(optimization_config["penalty_scale"]),
        seed=seed,
        voltage_upper_factor=voltage_upper_factor,
    )
    return policy, summary


def policy_bounds(
    frame: pd.DataFrame, voltage_upper_factor: float = 1.0
) -> list[tuple[float, float]]:
    bounds = [
        (float(frame[column].min()), float(frame[column].max())) for column in POLICY_COLUMNS
    ]
    return [
        (lower, upper * voltage_upper_factor if index < 4 else upper)
        for index, (lower, upper) in enumerate(bounds)
    ]


def _policy_summary(
    policy: np.ndarray,
    regime_frame: pd.DataFrame,
    twin: EmissionTwin,
    power: PowerSurrogate,
    scenarios: EmissionScenarios,
    limit: float,
    confidence: float,
    dust_limit: float,
) -> dict[str, float | bool]:
    concentrations = twin.predict_scenarios(policy, scenarios)
    p95 = float(np.quantile(concentrations, confidence))
    return {
        "power_kW": power.predict_policy(policy),
        "c_mean_mgNm3": float(np.mean(concentrations)),
        "c_median_mgNm3": float(np.median(concentrations)),
        "c_p95_mgNm3": p95,
        "compliance_probability": float(np.mean(concentrations <= limit)),
        "dust_load_index": twin.dust_load_index(policy, regime_frame),
        "feasible": bool(
            p95 <= limit + 1e-6
            and twin.dust_load_index(policy, regime_frame) <= dust_limit + 1e-6
        ),
    }


def _optimize_one(
    regime: int,
    regime_frame: pd.DataFrame,
    full_frame: pd.DataFrame,
    twin: EmissionTwin,
    power: PowerSurrogate,
    limit: float,
    confidence: float,
    scenario_quantile_buffer: float,
    n_scenarios: int,
    verification_scenarios: int,
    maxiter: int,
    popsize: int,
    dust_limit: float,
    penalty_scale: float,
    seed: int,
    voltage_upper_factor: float,
) -> tuple[np.ndarray, dict[str, float | bool], EmissionScenarios]:
    bounds = policy_bounds(full_frame, voltage_upper_factor=voltage_upper_factor)
    training = twin.scenario_pack(regime_frame, n_scenarios=n_scenarios, seed=seed)
    design_confidence = min(0.995, confidence + scenario_quantile_buffer)

    def constraints(policy: np.ndarray) -> tuple[float, float]:
        concentration = twin.predict_scenarios(policy, training)
        p95 = float(np.quantile(concentration, design_confidence))
        dust = twin.dust_load_index(policy, regime_frame)
        return p95, dust

    def penalized_objective(policy: np.ndarray) -> float:
        p95, dust = constraints(policy)
        emission_violation = max(0.0, p95 / limit - 1.0)
        dust_violation = max(0.0, dust / dust_limit - 1.0)
        return float(
            power.predict_policy(policy)
            + penalty_scale * (emission_violation**2 + dust_violation**2)
        )

    global_result = differential_evolution(
        penalized_objective,
        bounds=bounds,
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        tol=2e-5,
        polish=False,
        updating="immediate",
        workers=1,
    )

    local_constraints = [
        {"type": "ineq", "fun": lambda x: limit - constraints(x)[0]},
        {"type": "ineq", "fun": lambda x: dust_limit - constraints(x)[1]},
    ]
    local_result = minimize(
        lambda x: power.predict_policy(x),
        global_result.x,
        method="SLSQP",
        bounds=bounds,
        constraints=local_constraints,
        options={"maxiter": 350, "ftol": 1e-8, "disp": False},
    )
    candidates = [np.asarray(global_result.x, dtype=float)]
    if local_result.success:
        candidates.append(np.asarray(local_result.x, dtype=float))

    feasible_candidates: list[np.ndarray] = []
    for candidate in candidates:
        p95, dust = constraints(candidate)
        if p95 <= limit * 1.0005 and dust <= dust_limit * 1.0005:
            feasible_candidates.append(candidate)
    if feasible_candidates:
        policy = min(feasible_candidates, key=power.predict_policy)
    else:
        policy = min(candidates, key=penalized_objective)

    verification = twin.scenario_pack(
        regime_frame,
        n_scenarios=verification_scenarios,
        seed=seed + 100000,
    )
    summary = _policy_summary(
        policy,
        regime_frame,
        twin,
        power,
        verification,
        limit,
        confidence,
        dust_limit,
    )
    summary.update(
        {
            "regime": regime,
            "limit_mgNm3": limit,
            "global_iterations": int(global_result.nit),
            "local_success": bool(local_result.success),
            "design_confidence": design_confidence,
            "voltage_upper_factor": voltage_upper_factor,
        }
    )
    return policy, summary, training


def optimize_policy(
    twin: EmissionTwin,
    power: PowerSurrogate,
    frame: pd.DataFrame,
    regime_labels: np.ndarray,
    limit: float,
    optimization_config: dict[str, Any],
    seed: int = 2026,
    voltage_upper_factor: float = 1.0,
) -> OptimizationResult:
    rows: list[dict[str, float | bool]] = []
    policies: dict[int, np.ndarray] = {}
    scenario_packs: dict[int, EmissionScenarios] = {}
    for regime in sorted(np.unique(regime_labels)):
        regime_frame = frame.loc[regime_labels == regime].reset_index(drop=True)
        policy, summary, scenarios = _optimize_one(
            regime=int(regime),
            regime_frame=regime_frame,
            full_frame=frame,
            twin=twin,
            power=power,
            limit=float(limit),
            confidence=float(optimization_config["confidence"]),
            scenario_quantile_buffer=float(
                optimization_config.get("scenario_quantile_buffer", 0.025)
            ),
            n_scenarios=int(optimization_config["training_scenarios"]),
            verification_scenarios=int(optimization_config["verification_scenarios"]),
            maxiter=int(optimization_config["differential_evolution_maxiter"]),
            popsize=int(optimization_config["differential_evolution_popsize"]),
            dust_limit=float(optimization_config["dust_load_limit"]),
            penalty_scale=float(optimization_config["penalty_scale"]),
            seed=seed + 1000 * int(regime) + int(limit * 10),
            voltage_upper_factor=voltage_upper_factor,
        )
        policies[int(regime)] = policy
        scenario_packs[int(regime)] = scenarios
        summary.update({column: float(value) for column, value in zip(POLICY_COLUMNS, policy)})
        history_power = float(regime_frame["P_total_kW"].mean())
        summary["history_power_mean_kW"] = history_power
        summary["saving_vs_history_pct"] = 100.0 * (
            history_power - float(summary["power_kW"])
        ) / history_power
        rows.append(summary)
    table = pd.DataFrame(rows).sort_values("regime").reset_index(drop=True)
    return OptimizationResult(
        limit=float(limit),
        table=table,
        policies=policies,
        training_scenarios=scenario_packs,
    )


def policy_priority_table(
    policy: np.ndarray,
    regime_frame: pd.DataFrame,
    twin: EmissionTwin,
    power: PowerSurrogate,
    bounds: list[tuple[float, float]],
    seed: int,
    n_scenarios: int = 10000,
) -> pd.DataFrame:
    scenarios = twin.scenario_pack(regime_frame, n_scenarios=n_scenarios, seed=seed)
    base_c = float(np.quantile(twin.predict_scenarios(policy, scenarios), 0.95))
    base_p = power.predict_policy(policy)
    rows: list[dict[str, float | str]] = []
    for index, column in enumerate(POLICY_COLUMNS):
        changed = policy.copy()
        if index < 4:
            if policy[index] * 1.01 <= bounds[index][1] + 1e-9:
                changed[index] = min(bounds[index][1], policy[index] * 1.01)
                direction = "提高1%"
                reverse = False
            else:
                changed[index] = max(bounds[index][0], policy[index] * 0.99)
                direction = "上限处反向差分"
                reverse = True
        else:
            if policy[index] * 0.99 >= bounds[index][0] - 1e-9:
                changed[index] = max(bounds[index][0], policy[index] * 0.99)
                direction = "缩短1%"
                reverse = False
            else:
                changed[index] = min(bounds[index][1], policy[index] * 1.01)
                direction = "下限处反向差分"
                reverse = True
        new_c = float(np.quantile(twin.predict_scenarios(changed, scenarios), 0.95))
        new_p = power.predict_policy(changed)
        if reverse:
            delta_power = base_p - new_p
            reduction = new_c - base_c
        else:
            delta_power = new_p - base_p
            reduction = base_c - new_c
        rows.append(
            {
                "parameter": column,
                "direction": direction,
                "delta_power_kW": delta_power,
                "p95_reduction_mgNm3": reduction,
                "reduction_per_added_kW": reduction / delta_power if delta_power > 1e-9 else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("reduction_per_added_kW", ascending=False)


def power_increment_attribution(
    policy_10: np.ndarray,
    policy_5: np.ndarray,
    power: PowerSurrogate,
) -> pd.DataFrame:
    coef = power.raw_coefficients
    voltage_increment = coef[:4] * (policy_5[:4] ** 2 - policy_10[:4] ** 2)
    rapping_increment = coef[4:8] * (1.0 / policy_5[4:] - 1.0 / policy_10[4:])
    rows = [
        {"component": f"U{i + 1}", "increment_kW": float(voltage_increment[i])}
        for i in range(4)
    ]
    rows.extend(
        {"component": f"T{i + 1}", "increment_kW": float(rapping_increment[i])}
        for i in range(4)
    )
    table = pd.DataFrame(rows)
    total = float(table["increment_kW"].sum())
    table["share_pct"] = np.where(abs(total) > 1e-9, 100.0 * table["increment_kW"] / total, np.nan)
    return table
