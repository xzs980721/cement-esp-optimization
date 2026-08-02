from __future__ import annotations

import numpy as np
import pandas as pd

from .emission_twin import EmissionTwin
from .optimization import POLICY_COLUMNS, policy_priority_table
from .power_model import PowerSurrogate


def cycle_constraint_relief_table(
    policy: np.ndarray,
    regime_frame: pd.DataFrame,
    twin: EmissionTwin,
    power: PowerSurrogate,
    bounds: list[tuple[float, float]],
    perturbation: float = 0.01,
) -> pd.DataFrame:
    """Return dust-load relief per added kW for the four rapping periods.

    A normal perturbation shortens a period by ``perturbation``.  At a lower
    bound, the function uses the adjacent longer-period point and converts the
    backward difference to the same "shorten the period" direction.
    """

    policy = np.asarray(policy, dtype=float)
    if policy.shape != (8,):
        raise ValueError("Policy must contain U1..U4,T1..T4.")
    if len(bounds) != 8:
        raise ValueError("Bounds must contain eight (lower, upper) pairs.")
    if not 0.0 < perturbation < 1.0:
        raise ValueError("Perturbation must lie between zero and one.")

    base_power = float(power.predict_policy(policy))
    base_dust = float(twin.dust_load_index(policy, regime_frame))
    rows: list[dict[str, float | str]] = []
    for index in range(4, 8):
        changed = policy.copy()
        shortened = policy[index] * (1.0 - perturbation)
        if shortened >= bounds[index][0] - 1e-9:
            changed[index] = max(bounds[index][0], shortened)
            direction = f"缩短{100 * perturbation:.0f}%"
            delta_power = float(power.predict_policy(changed) - base_power)
            dust_relief = float(base_dust - twin.dust_load_index(changed, regime_frame))
        else:
            changed[index] = min(bounds[index][1], policy[index] * (1.0 + perturbation))
            direction = "下限处反向差分"
            delta_power = float(base_power - power.predict_policy(changed))
            dust_relief = float(twin.dust_load_index(changed, regime_frame) - base_dust)
        relief_per_kw = dust_relief / delta_power if delta_power > 1e-9 else np.nan
        rows.append(
            {
                "parameter": POLICY_COLUMNS[index],
                "constraint_direction": direction,
                "constraint_delta_power_kW": delta_power,
                "dust_load_relief": dust_relief,
                "dust_relief_per_added_kW": relief_per_kw,
            }
        )
    return pd.DataFrame(rows).sort_values(
        "dust_relief_per_added_kW", ascending=False, ignore_index=True
    )


def _within_relative_gap(values: pd.Series, best: float, tolerance: float) -> pd.Series:
    denominator = max(abs(best), 1e-12)
    return (best - values) / denominator <= tolerance + 1e-12


def hierarchical_priority_table(
    policy: np.ndarray,
    regime_frame: pd.DataFrame,
    twin: EmissionTwin,
    power: PowerSurrogate,
    bounds: list[tuple[float, float]],
    seed: int,
    regime: int | None = None,
    n_scenarios: int = 10000,
    dust_limit: float = 1.45,
    active_tolerance: float = 1e-4,
    tie_relative_tolerance: float = 0.02,
) -> pd.DataFrame:
    """Rank controls by feasibility first and direct efficiency second.

    If the relative dust-load slack is within ``active_tolerance`` of zero,
    rapping periods are ranked first by dust-load relief per added kW.  Once
    slack is restored, voltages are ranked by direct p95 reduction per added
    kW.  Otherwise all variables are ranked directly by the latter metric.
    """

    if dust_limit <= 0:
        raise ValueError("Dust limit must be positive.")
    if not 0 <= active_tolerance < 1:
        raise ValueError("Active tolerance must lie in [0, 1).")
    if not 0 <= tie_relative_tolerance < 1:
        raise ValueError("Tie tolerance must lie in [0, 1).")

    direct = policy_priority_table(
        policy=policy,
        regime_frame=regime_frame,
        twin=twin,
        power=power,
        bounds=bounds,
        seed=seed,
        n_scenarios=n_scenarios,
    ).rename(
        columns={
            "delta_power_kW": "direct_delta_power_kW",
            "p95_reduction_mgNm3": "direct_p95_reduction_mgNm3",
            "reduction_per_added_kW": "direct_benefit_mgNm3_per_kW",
        }
    )
    relief = cycle_constraint_relief_table(
        policy=policy,
        regime_frame=regime_frame,
        twin=twin,
        power=power,
        bounds=bounds,
    )
    actions = direct.merge(relief, on="parameter", how="left")
    actions["parameter_type"] = np.where(
        actions["parameter"].str.startswith("U"), "电压", "振打周期"
    )
    for column in [
        "constraint_delta_power_kW",
        "dust_load_relief",
        "dust_relief_per_added_kW",
    ]:
        actions[column] = actions[column].fillna(0.0)

    dust_index = float(twin.dust_load_index(policy, regime_frame))
    relative_slack = float((dust_limit - dust_index) / dust_limit)
    constraint_active = bool(relative_slack <= active_tolerance)
    voltage = actions.loc[actions["parameter_type"] == "电压"].copy()
    periods = actions.loc[actions["parameter_type"] == "振打周期"].copy()
    best_voltage = voltage.sort_values(
        "direct_benefit_mgNm3_per_kW", ascending=False
    ).iloc[0]

    actions["is_primary_action"] = False
    actions["is_tied_primary"] = False
    actions["is_secondary_action"] = False
    if constraint_active:
        periods = periods.sort_values("dust_relief_per_added_kW", ascending=False)
        best_value = float(periods.iloc[0]["dust_relief_per_added_kW"])
        tied_mask = _within_relative_gap(
            periods["dust_relief_per_added_kW"], best_value, tie_relative_tolerance
        )
        tied_parameters = periods.loc[tied_mask, "parameter"].tolist()
        primary_parameter = "、".join(tied_parameters)
        primary_class = "周期优先"
        secondary_parameter = str(best_voltage["parameter"])
        actions.loc[actions["parameter"].isin(tied_parameters), "is_tied_primary"] = True
        actions.loc[actions["parameter"] == tied_parameters[0], "is_primary_action"] = True
        actions.loc[
            actions["parameter"] == secondary_parameter, "is_secondary_action"
        ] = True
        period_order = periods["parameter"].tolist()
        voltage_order = voltage.sort_values(
            "direct_benefit_mgNm3_per_kW", ascending=False
        )["parameter"].tolist()
        rank_order = period_order + voltage_order
        actions["operational_role"] = np.where(
            actions["parameter_type"] == "振打周期",
            "一级：恢复尘负荷安全裕量",
            "二级：约束恢复后的直接减排",
        )
    else:
        ordered = actions.sort_values("direct_benefit_mgNm3_per_kW", ascending=False)
        tied_parameters = [str(ordered.iloc[0]["parameter"])]
        primary_parameter = tied_parameters[0]
        primary_class = (
            "电压优先"
            if str(ordered.iloc[0]["parameter_type"]) == "电压"
            else "周期优先"
        )
        secondary_candidates = ordered.loc[~ordered["parameter"].isin(tied_parameters)]
        secondary_parameter = (
            str(secondary_candidates.iloc[0]["parameter"])
            if len(secondary_candidates)
            else "—"
        )
        actions.loc[actions["parameter"].isin(tied_parameters), "is_tied_primary"] = True
        actions.loc[actions["parameter"] == tied_parameters[0], "is_primary_action"] = True
        if secondary_parameter != "—":
            actions.loc[
                actions["parameter"] == secondary_parameter, "is_secondary_action"
            ] = True
        rank_order = ordered["parameter"].tolist()
        actions["operational_role"] = "直接减排效率排序"

    ranks = {parameter: rank + 1 for rank, parameter in enumerate(rank_order)}
    actions["operational_rank"] = actions["parameter"].map(ranks).astype(int)
    actions["regime"] = int(regime) if regime is not None else -1
    actions["label"] = f"R{int(regime) + 1}" if regime is not None else "—"
    actions["dust_load_index"] = dust_index
    actions["relative_dust_slack"] = relative_slack
    actions["dust_constraint_active"] = constraint_active
    actions["primary_action_class"] = primary_class
    actions["primary_parameter"] = primary_parameter
    actions["secondary_parameter"] = secondary_parameter

    columns = [
        "regime",
        "label",
        "parameter",
        "parameter_type",
        "direction",
        "direct_delta_power_kW",
        "direct_p95_reduction_mgNm3",
        "direct_benefit_mgNm3_per_kW",
        "constraint_direction",
        "constraint_delta_power_kW",
        "dust_load_relief",
        "dust_relief_per_added_kW",
        "dust_load_index",
        "relative_dust_slack",
        "dust_constraint_active",
        "primary_action_class",
        "primary_parameter",
        "secondary_parameter",
        "operational_role",
        "operational_rank",
        "is_primary_action",
        "is_tied_primary",
        "is_secondary_action",
    ]
    return actions[columns].sort_values("operational_rank", ignore_index=True)


def summarize_hierarchical_priority(actions: pd.DataFrame) -> dict[str, object]:
    """Collapse one regime's action table to a report-ready summary row."""

    if actions.empty:
        raise ValueError("Action table cannot be empty.")
    first = actions.iloc[0]
    primary_rows = actions.loc[actions["is_tied_primary"]]
    primary_direct = float(primary_rows["direct_benefit_mgNm3_per_kW"].max())
    primary_relief = float(primary_rows["dust_relief_per_added_kW"].max())
    return {
        "regime": int(first["regime"]),
        "label": str(first["label"]),
        "dust_load_index": float(first["dust_load_index"]),
        "relative_dust_slack": float(first["relative_dust_slack"]),
        "dust_constraint_active": bool(first["dust_constraint_active"]),
        "primary_action_class": str(first["primary_action_class"]),
        "primary_parameter": str(first["primary_parameter"]),
        "primary_direct_benefit_mgNm3_per_kW": primary_direct,
        "primary_dust_relief_per_added_kW": primary_relief,
        "secondary_parameter": str(first["secondary_parameter"]),
    }
