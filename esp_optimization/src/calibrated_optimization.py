from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

from .calibrated_emission import (
    CalibratedScenarios,
    UncensoredDistribution,
    build_calibrated_scenarios,
    regime_reference,
)
from .emission_twin import EmissionTwin
from .optimization import POLICY_COLUMNS, policy_bounds
from .power_model import PowerSurrogate


@dataclass
class CalibratedOptimizationResult:
    limit: float
    table: pd.DataFrame
    policies: dict[int, np.ndarray]
    reference_policies: dict[int, np.ndarray]
    training_scenarios: dict[int, CalibratedScenarios]


def _summary(
    policy: np.ndarray,
    scenarios: CalibratedScenarios,
    regime_frame: pd.DataFrame,
    twin: EmissionTwin,
    power: PowerSurrogate,
    limit: float,
    confidence: float,
    dust_limit: float,
) -> dict[str, float | bool]:
    concentration = scenarios.predict(twin, policy)
    p95 = float(np.quantile(concentration, confidence))
    dust = float(twin.dust_load_index(policy, regime_frame))
    return {
        "power_kW": float(power.predict_policy(policy)),
        "c_mean_mgNm3": float(np.mean(concentration)),
        "c_median_mgNm3": float(np.median(concentration)),
        "c_p95_mgNm3": p95,
        "compliance_probability": float(np.mean(concentration <= limit)),
        "dust_load_index": dust,
        "feasible": bool(p95 <= limit + 1e-6 and dust <= dust_limit + 1e-6),
    }


def _repair_on_scenarios(
    policy: np.ndarray,
    bounds: list[tuple[float, float]],
    scenarios: CalibratedScenarios,
    regime_frame: pd.DataFrame,
    twin: EmissionTwin,
    power: PowerSurrogate,
    limit: float,
    design_confidence: float,
    dust_limit: float,
) -> tuple[np.ndarray, bool]:
    def metrics(candidate: np.ndarray) -> tuple[float, float]:
        concentration = scenarios.predict(twin, candidate)
        return (
            float(np.quantile(concentration, design_confidence)),
            float(twin.dust_load_index(candidate, regime_frame)),
        )

    candidate = np.asarray(policy, dtype=float).copy()
    ceiling = candidate.copy()
    ceiling[:4] = np.asarray([bound[1] for bound in bounds[:4]], dtype=float)
    ceiling[4:] = np.minimum(
        candidate[4:], np.asarray([bound[1] for bound in bounds[4:]], dtype=float)
    )
    ceiling_q, ceiling_dust = metrics(ceiling)
    if ceiling_q > limit + 1e-6 or ceiling_dust > dust_limit + 1e-6:
        return candidate, False

    lower_mix, upper_mix = 0.0, 1.0
    for _ in range(38):
        mix = 0.5 * (lower_mix + upper_mix)
        trial = candidate.copy()
        trial[:4] += mix * (ceiling[:4] - candidate[:4])
        q_value, dust = metrics(trial)
        if q_value <= limit and dust <= dust_limit:
            upper_mix = mix
        else:
            lower_mix = mix
    repaired = candidate.copy()
    repaired[:4] += upper_mix * (ceiling[:4] - candidate[:4])
    constraints = [
        {
            "type": "ineq",
            "fun": lambda x: limit
            - float(np.quantile(scenarios.predict(twin, x), design_confidence)),
        },
        {
            "type": "ineq",
            "fun": lambda x: dust_limit - twin.dust_load_index(x, regime_frame),
        },
    ]
    local = minimize(
        lambda x: power.predict_policy(x),
        repaired,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 300, "ftol": 1e-8, "disp": False},
    )
    candidates = [repaired]
    if local.success:
        candidates.append(np.asarray(local.x, dtype=float))
    feasible = []
    for trial in candidates:
        q_value, dust = metrics(trial)
        if q_value <= limit * 1.0005 and dust <= dust_limit * 1.0005:
            feasible.append(trial)
    if not feasible:
        return candidate, False
    return min(feasible, key=power.predict_policy), True


def _optimize_regime(
    distribution: UncensoredDistribution,
    twin: EmissionTwin,
    power: PowerSurrogate,
    full_frame: pd.DataFrame,
    regime_frame: pd.DataFrame,
    regime: int,
    limit: float,
    optimization_config: dict[str, Any],
    seed: int,
    voltage_upper_factor: float,
    initial_policy: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, float | bool], CalibratedScenarios, np.ndarray]:
    bounds = policy_bounds(full_frame, voltage_upper_factor=voltage_upper_factor)
    _, reference_policy = regime_reference(regime_frame)
    confidence = float(optimization_config["confidence"])
    design_confidence = min(
        0.995,
        confidence + float(optimization_config.get("scenario_quantile_buffer", 0.025)),
    )
    dust_limit = float(optimization_config["dust_load_limit"])
    training = build_calibrated_scenarios(
        twin=twin,
        distribution=distribution,
        regime_frame=regime_frame,
        reference_policy=reference_policy,
        n_scenarios=int(optimization_config["training_scenarios"]),
        seed=seed,
    )

    def metrics(policy: np.ndarray) -> tuple[float, float]:
        concentration = training.predict(twin, policy)
        return (
            float(np.quantile(concentration, design_confidence)),
            float(twin.dust_load_index(policy, regime_frame)),
        )

    penalty_scale = float(optimization_config["penalty_scale"])

    def objective(policy: np.ndarray) -> float:
        p95, dust = metrics(policy)
        emission_violation = max(0.0, p95 / limit - 1.0)
        dust_violation = max(0.0, dust / dust_limit - 1.0)
        return float(
            power.predict_policy(policy)
            + penalty_scale * (emission_violation**2 + dust_violation**2)
        )

    global_result = differential_evolution(
        objective,
        bounds=bounds,
        seed=seed,
        maxiter=int(optimization_config["differential_evolution_maxiter"]),
        popsize=int(optimization_config["differential_evolution_popsize"]),
        tol=2e-5,
        polish=False,
        updating="immediate",
        workers=1,
    )
    constraints = [
        {"type": "ineq", "fun": lambda x: limit - metrics(x)[0]},
        {"type": "ineq", "fun": lambda x: dust_limit - metrics(x)[1]},
    ]
    lower = np.asarray([bound[0] for bound in bounds], dtype=float)
    upper = np.asarray([bound[1] for bound in bounds], dtype=float)
    starts = [np.asarray(global_result.x, dtype=float)]
    starts.append(np.clip(reference_policy, lower, upper))
    if initial_policy is not None:
        starts.append(np.clip(np.asarray(initial_policy, dtype=float), lower, upper))
    ceiling = starts[-1].copy()
    ceiling[:4] = upper[:4]
    starts.append(ceiling)

    candidates: list[np.ndarray] = []
    local_success = False
    for start in starts:
        candidates.append(start)
        local = minimize(
            lambda x: power.predict_policy(x),
            start,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 350, "ftol": 1e-8, "disp": False},
        )
        local_success = local_success or bool(local.success)
        if local.success:
            candidates.append(np.asarray(local.x, dtype=float))
    feasible = []
    for candidate in candidates:
        p95, dust = metrics(candidate)
        if p95 <= limit * 1.0005 and dust <= dust_limit * 1.0005:
            feasible.append(candidate)
    policy = min(feasible, key=power.predict_policy) if feasible else min(candidates, key=objective)

    verification = build_calibrated_scenarios(
        twin=twin,
        distribution=distribution,
        regime_frame=regime_frame,
        reference_policy=reference_policy,
        n_scenarios=int(optimization_config["verification_scenarios"]),
        seed=seed + 100_000,
    )
    summary = _summary(
        policy,
        verification,
        regime_frame,
        twin,
        power,
        limit,
        confidence,
        dust_limit,
    )
    repaired = False
    if not bool(summary["feasible"]):
        repair_pack = build_calibrated_scenarios(
            twin=twin,
            distribution=distribution,
            regime_frame=regime_frame,
            reference_policy=reference_policy,
            n_scenarios=int(optimization_config["verification_scenarios"]),
            seed=seed + 200_000,
        )
        policy, repaired = _repair_on_scenarios(
            policy,
            bounds,
            repair_pack,
            regime_frame,
            twin,
            power,
            limit,
            design_confidence,
            dust_limit,
        )
        if repaired:
            verification = build_calibrated_scenarios(
                twin=twin,
                distribution=distribution,
                regime_frame=regime_frame,
                reference_policy=reference_policy,
                n_scenarios=int(optimization_config["verification_scenarios"]),
                seed=seed + 300_000,
            )
            summary = _summary(
                policy,
                verification,
                regime_frame,
                twin,
                power,
                limit,
                confidence,
                dust_limit,
            )
    summary.update(
        {
            "regime": int(regime),
            "limit_mgNm3": float(limit),
            "global_iterations": int(global_result.nit),
            "local_success": bool(local_success),
            "design_confidence": design_confidence,
            "voltage_upper_factor": float(voltage_upper_factor),
            "verification_repair": bool(repaired),
        }
    )
    return policy, summary, training, reference_policy


def optimize_calibrated_policy(
    distribution: UncensoredDistribution,
    twin: EmissionTwin,
    power: PowerSurrogate,
    frame: pd.DataFrame,
    regime_labels: np.ndarray,
    limit: float,
    optimization_config: dict[str, Any],
    seed: int = 2026,
    voltage_upper_factor: float = 1.0,
    initial_policies: dict[int, np.ndarray] | None = None,
) -> CalibratedOptimizationResult:
    rows: list[dict[str, float | bool]] = []
    policies: dict[int, np.ndarray] = {}
    references: dict[int, np.ndarray] = {}
    scenarios: dict[int, CalibratedScenarios] = {}
    for regime_value in sorted(np.unique(regime_labels)):
        regime = int(regime_value)
        regime_frame = frame.loc[regime_labels == regime].reset_index(drop=True)
        policy, summary, training, reference = _optimize_regime(
            distribution=distribution,
            twin=twin,
            power=power,
            full_frame=frame,
            regime_frame=regime_frame,
            regime=regime,
            limit=float(limit),
            optimization_config=optimization_config,
            seed=seed + 1000 * regime + int(limit * 10),
            voltage_upper_factor=voltage_upper_factor,
            initial_policy=(
                initial_policies.get(regime) if initial_policies is not None else None
            ),
        )
        policies[regime] = policy
        references[regime] = reference
        scenarios[regime] = training
        summary.update(
            {column: float(value) for column, value in zip(POLICY_COLUMNS, policy)}
        )
        history_power = float(regime_frame["P_total_kW"].mean())
        summary["history_power_mean_kW"] = history_power
        summary["saving_vs_history_pct"] = 100.0 * (
            history_power - float(summary["power_kW"])
        ) / history_power
        rows.append(summary)
    return CalibratedOptimizationResult(
        limit=float(limit),
        table=pd.DataFrame(rows).sort_values("regime").reset_index(drop=True),
        policies=policies,
        reference_policies=references,
        training_scenarios=scenarios,
    )


def calibrated_policy_priority_table(
    policy: np.ndarray,
    scenarios: CalibratedScenarios,
    regime_frame: pd.DataFrame,
    twin: EmissionTwin,
    power: PowerSurrogate,
    bounds: list[tuple[float, float]],
    dust_limit: float = 1.45,
    perturbation: float = 0.01,
) -> pd.DataFrame:
    policy = np.asarray(policy, dtype=float)
    base_concentration = float(np.quantile(scenarios.predict(twin, policy), 0.95))
    base_power = float(power.predict_policy(policy))
    base_dust = float(twin.dust_load_index(policy, regime_frame))
    rows: list[dict[str, float | str | bool]] = []
    for index, parameter in enumerate(POLICY_COLUMNS):
        changed = policy.copy()
        if index < 4:
            normal = policy[index] * (1.0 + perturbation) <= bounds[index][1] + 1e-9
            changed[index] = (
                min(bounds[index][1], policy[index] * (1.0 + perturbation))
                if normal
                else max(bounds[index][0], policy[index] * (1.0 - perturbation))
            )
            direction = "提高1%" if normal else "上限处反向差分"
        else:
            normal = policy[index] * (1.0 - perturbation) >= bounds[index][0] - 1e-9
            changed[index] = (
                max(bounds[index][0], policy[index] * (1.0 - perturbation))
                if normal
                else min(bounds[index][1], policy[index] * (1.0 + perturbation))
            )
            direction = "缩短1%" if normal else "下限处反向差分"
        new_concentration = float(
            np.quantile(scenarios.predict(twin, changed), 0.95)
        )
        new_power = float(power.predict_policy(changed))
        new_dust = float(twin.dust_load_index(changed, regime_frame))
        if normal:
            delta_power = new_power - base_power
            reduction = base_concentration - new_concentration
            dust_relief = base_dust - new_dust
        else:
            delta_power = base_power - new_power
            reduction = new_concentration - base_concentration
            dust_relief = new_dust - base_dust
        rows.append(
            {
                "parameter": parameter,
                "parameter_type": "电压" if index < 4 else "振打周期",
                "direction": direction,
                "delta_power_kW": delta_power,
                "p95_reduction_mgNm3": reduction,
                "reduction_per_added_kW": (
                    reduction / delta_power if delta_power > 1e-9 else np.nan
                ),
                "dust_load_relief": dust_relief,
                "dust_relief_per_added_kW": (
                    dust_relief / delta_power if delta_power > 1e-9 else np.nan
                ),
            }
        )
    table = pd.DataFrame(rows)
    relative_slack = float((dust_limit - base_dust) / dust_limit)
    active = bool(relative_slack <= 1e-4)
    if active:
        primary = (
            table.loc[table["parameter_type"] == "振打周期"]
            .sort_values("dust_relief_per_added_kW", ascending=False)
            .iloc[0]["parameter"]
        )
        priority_class = "周期优先"
    else:
        primary = table.sort_values(
            "reduction_per_added_kW", ascending=False
        ).iloc[0]["parameter"]
        priority_class = "电压优先" if str(primary).startswith("U") else "周期优先"
    table["dust_load_index"] = base_dust
    table["relative_dust_slack"] = relative_slack
    table["dust_constraint_active"] = active
    table["primary_action_class"] = priority_class
    table["primary_parameter"] = primary
    return table.sort_values("reduction_per_added_kW", ascending=False, ignore_index=True)
