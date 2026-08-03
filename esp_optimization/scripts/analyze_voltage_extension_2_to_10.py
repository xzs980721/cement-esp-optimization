from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_audit import load_and_audit_data
from src.emission_twin import fit_emission_twin
from src.optimization import (
    POLICY_COLUMNS,
    _policy_summary,
    _repair_on_independent_scenarios,
    policy_bounds,
    power_increment_attribution,
)
from src.power_model import fit_power_surrogate
from src.regimes import segment_regimes
from src.reporting import COLORS, setup_plot_style, save_figure


FACTORS = np.arange(1.02, 1.101, 0.01)


def _continuation_optimize(
    previous_policy: np.ndarray,
    previous_factor: float,
    factor: float,
    regime: int,
    regime_frame: pd.DataFrame,
    full_frame: pd.DataFrame,
    twin,
    power,
    optimization_config: dict,
    seed: int,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    """Refine one adjacent voltage-bound scenario using deterministic warm starts."""

    bounds = policy_bounds(full_frame, voltage_upper_factor=factor)
    lower = np.array([bound[0] for bound in bounds], dtype=float)
    upper = np.array([bound[1] for bound in bounds], dtype=float)
    confidence = float(optimization_config["confidence"])
    design_confidence = min(
        0.995,
        confidence + float(optimization_config.get("scenario_quantile_buffer", 0.025)),
    )
    dust_limit = float(optimization_config["dust_load_limit"])
    training = twin.scenario_pack(
        regime_frame,
        n_scenarios=int(optimization_config["training_scenarios"]),
        seed=seed,
    )

    def design_metrics(policy: np.ndarray) -> tuple[float, float]:
        concentration = twin.predict_scenarios(policy, training)
        return (
            float(np.quantile(concentration, design_confidence)),
            float(twin.dust_load_index(policy, regime_frame)),
        )

    constraints = [
        {"type": "ineq", "fun": lambda x: 5.0 - design_metrics(x)[0]},
        {"type": "ineq", "fun": lambda x: dust_limit - design_metrics(x)[1]},
    ]
    scale = factor / previous_factor
    starts: list[np.ndarray] = [np.clip(previous_policy, lower, upper)]
    voltage_scaled = previous_policy.copy()
    voltage_scaled[:4] *= scale
    starts.append(np.clip(voltage_scaled, lower, upper))
    voltage_ceiling = previous_policy.copy()
    voltage_ceiling[:4] = upper[:4]
    starts.append(np.clip(voltage_ceiling, lower, upper))
    for period_mix in (0.25, 0.50, 0.75):
        candidate = voltage_ceiling.copy()
        candidate[4:] += period_mix * (upper[4:] - candidate[4:])
        starts.append(np.clip(candidate, lower, upper))

    candidates: list[np.ndarray] = []
    local_success = False
    for start in starts:
        q_value, dust = design_metrics(start)
        if q_value <= 5.0 * 1.0005 and dust <= dust_limit * 1.0005:
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
        q_value, dust = design_metrics(local.x)
        if q_value <= 5.0 * 1.0005 and dust <= dust_limit * 1.0005:
            candidates.append(np.asarray(local.x, dtype=float))

    if not candidates:
        candidates = [np.clip(previous_policy, lower, upper)]
    policy = min(candidates, key=power.predict_policy)
    verification = twin.scenario_pack(
        regime_frame,
        n_scenarios=int(optimization_config["verification_scenarios"]),
        seed=seed + 100000,
    )
    summary = _policy_summary(
        policy,
        regime_frame,
        twin,
        power,
        verification,
        5.0,
        confidence,
        dust_limit,
    )
    verification_repair = False
    if not bool(summary["feasible"]):
        repair_scenarios = twin.scenario_pack(
            regime_frame,
            n_scenarios=int(optimization_config["verification_scenarios"]),
            seed=seed + 200000,
        )
        repaired, verification_repair = _repair_on_independent_scenarios(
            policy,
            bounds,
            regime_frame,
            twin,
            power,
            repair_scenarios,
            5.0,
            design_confidence,
            dust_limit,
        )
        if verification_repair:
            policy = repaired
            verification = twin.scenario_pack(
                regime_frame,
                n_scenarios=int(optimization_config["verification_scenarios"]),
                seed=seed + 300000,
            )
            summary = _policy_summary(
                policy,
                regime_frame,
                twin,
                power,
                verification,
                5.0,
                confidence,
                dust_limit,
            )
    summary.update(
        {
            "regime": regime,
            "limit_mgNm3": 5.0,
            "local_success": local_success,
            "design_confidence": design_confidence,
            "voltage_upper_factor": factor,
            "verification_repair": verification_repair,
        }
    )
    return policy, summary


def _load_context():
    config_path = PROJECT_ROOT / "config" / "model.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(config["project"]["seed"])
    audit = load_and_audit_data(
        (PROJECT_ROOT / config["project"]["input_csv"]).resolve(),
        censor_threshold=float(config["data"]["censor_threshold_mgNm3"]),
    )
    power = fit_power_surrogate(audit.frame)
    regime_config = config["regimes"]
    regimes = segment_regimes(
        audit.frame,
        rolling_window=int(config["data"]["rolling_window_minutes"]),
        k_min=int(regime_config["k_min"]),
        k_max=int(regime_config["k_max"]),
        switch_penalty=float(regime_config["switch_penalty"]),
        minimum_share=float(regime_config["minimum_share"]),
        minimum_median_dwell=int(regime_config["minimum_median_dwell_minutes"]),
        selection_metric=str(regime_config["selection_metric"]),
        seed=seed,
    )
    twin = fit_emission_twin(audit, config["emission_prior"])
    return config, seed, audit, power, regimes, twin


def _plot(summary: pd.DataFrame, path: Path) -> None:
    setup_plot_style()
    x = summary["extension_pct"].to_numpy()
    fig = plt.figure(figsize=(9.2, 5.4))
    grid = fig.add_gridspec(2, 2, width_ratios=(1.12, 1), hspace=0.42, wspace=0.30)
    ax_power = fig.add_subplot(grid[:, 0])
    ax_attribution = fig.add_subplot(grid[0, 1])
    ax_r8 = fig.add_subplot(grid[1, 1])

    # (a) Weighted power trend
    ax_power.plot(
        x, summary["weighted_power_kW"], marker="o", color=COLORS[0], lw=1.4,
    )
    ax_power.axvspan(3, 10, color=COLORS[1], alpha=0.08, lw=0)
    ax_power.set_title("(a) 全周期加权功率")
    ax_power.set_ylabel("功率 / kW")
    ax_power.set_xlabel("电压上界扩展 / %")
    selected = {2, 3, 10}
    for xi, yi in zip(x, summary["weighted_power_kW"]):
        if int(round(xi)) in selected:
            offset = (6, 7) if int(round(xi)) == 2 else ((-6, 7) if int(round(xi)) == 10 else (0, 7))
            alignment = "left" if int(round(xi)) == 2 else ("right" if int(round(xi)) == 10 else "center")
            ax_power.annotate(
                f"{yi:.1f}", (xi, yi), xytext=offset,
                textcoords="offset points", ha=alignment, fontsize=7.5,
            )
    ax_power.text(
        6.5, 0.04, "边际变化趋缓区间", transform=ax_power.get_xaxis_transform(),
        ha="center", va="bottom", fontsize=7.5, color=COLORS[1],
    )

    # (b) Stacked power increment attribution
    ax_attribution.bar(
        x,
        summary["weighted_voltage_increment_kW"],
        color=COLORS[0],
        label="电压项",
        width=0.7,
    )
    ax_attribution.bar(
        x,
        summary["weighted_rapping_increment_kW"],
        bottom=summary["weighted_voltage_increment_kW"],
        color=COLORS[2],
        label="振打周期项",
        width=0.7,
    )
    ax_attribution.set_title("(b) 相对 10 mg·Nm$^{-3}$ 策略的增量功率")
    ax_attribution.set_ylabel("增量功率 / kW")
    ax_attribution.set_xlabel("电压上界扩展 / %")
    ax_attribution.legend(ncol=2, loc="upper right", fontsize=7)

    # (c) R8 power increase
    ax_r8.plot(
        x, summary["r8_increase_pct"], marker="s", color=COLORS[3], lw=1.2
    )
    ax_r8.axvspan(3, 10, color=COLORS[3], alpha=0.05, lw=0)
    ax_r8.set_title("(c) 最高负荷 R8 的功率增幅")
    ax_r8.set_ylabel("功率增幅 / %")
    ax_r8.set_xlabel("电压上界扩展 / %")
    for xi, yi in zip(x, summary["r8_increase_pct"]):
        if int(round(xi)) in selected:
            offset = (6, 6) if int(round(xi)) == 2 else ((-6, 6) if int(round(xi)) == 10 else (0, 6))
            alignment = "left" if int(round(xi)) == 2 else ("right" if int(round(xi)) == 10 else "center")
            ax_r8.annotate(
                f"{yi:.2f}%", (xi, yi), xytext=offset,
                textcoords="offset points", ha=alignment, fontsize=7.5,
            )

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.11, top=0.94)
    save_figure(fig, path)


def main() -> None:
    config, seed, audit, power, regimes, twin = _load_context()
    table_dir = PROJECT_ROOT / "outputs" / "tables"
    figure_dir = PROJECT_ROOT / "outputs" / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    policy10 = pd.read_csv(table_dir / "optimal_policy_10.csv").sort_values("regime")
    policy_grid_path = table_dir / "q4_voltage_extension_2_to_10_policies.csv"
    if not policy_grid_path.exists():
        raise FileNotFoundError(
            "The 2% warm-start policies are missing; run this analysis from the "
            "published result bundle or regenerate the 2% scenario first."
        )
    policy5_base = pd.read_csv(policy_grid_path)
    policy5_base = policy5_base.loc[
        np.isclose(policy5_base["extension_pct"], 2)
    ].drop(columns="extension_pct").sort_values("regime")
    centers = regimes.centers.sort_values("regime").reset_index(drop=True)
    shares = centers["share"].to_numpy(dtype=float)
    baseline_power = policy10["power_kW"].to_numpy(dtype=float)
    weighted_power_10 = float(np.dot(shares, baseline_power))
    base_bounds = policy_bounds(audit.frame)
    common_seed = seed + 900
    previous_factor = 1.02
    previous_policies = {
        int(row["regime"]): row[POLICY_COLUMNS].to_numpy(dtype=float)
        for _, row in policy5_base.iterrows()
    }

    summary_rows: list[dict[str, object]] = []
    policy_rows: list[pd.DataFrame] = []
    attribution_rows: list[pd.DataFrame] = []

    for factor in FACTORS:
        if np.isclose(factor, 1.02):
            result_table = policy5_base.copy().reset_index(drop=True)
        else:
            rows: list[dict[str, object]] = []
            current_policies: dict[int, np.ndarray] = {}
            for regime in range(regimes.n_regimes):
                regime_frame = audit.frame.loc[regimes.labels == regime].reset_index(drop=True)
                optimizer_seed = common_seed + 1000 * regime + 50
                policy, summary = _continuation_optimize(
                    previous_policies[regime],
                    previous_factor,
                    float(factor),
                    regime,
                    regime_frame,
                    audit.frame,
                    twin,
                    power,
                    config["optimization"],
                    optimizer_seed,
                )
                summary.update(
                    {column: float(value) for column, value in zip(POLICY_COLUMNS, policy)}
                )
                rows.append(summary)
                current_policies[regime] = policy
            result_table = pd.DataFrame(rows).sort_values("regime").reset_index(drop=True)
            previous_policies = current_policies
            previous_factor = float(factor)
        result_table.insert(0, "extension_pct", round(100.0 * (factor - 1.0)))
        policy_rows.append(result_table)

        weighted_power = float(np.dot(shares, result_table["power_kW"].to_numpy(dtype=float)))
        weighted_voltage_increment = 0.0
        weighted_rapping_increment = 0.0
        for regime in range(regimes.n_regimes):
            policy_10 = policy10.loc[policy10["regime"] == regime, POLICY_COLUMNS].iloc[0].to_numpy(dtype=float)
            policy_5 = result_table.loc[result_table["regime"] == regime, POLICY_COLUMNS].iloc[0].to_numpy(dtype=float)
            attribution = power_increment_attribution(policy_10, policy_5, power)
            attribution.insert(0, "regime", regime)
            attribution.insert(0, "extension_pct", round(100.0 * (factor - 1.0)))
            attribution_rows.append(attribution)
            weighted_voltage_increment += shares[regime] * float(
                attribution.loc[attribution["component"].str.startswith("U"), "increment_kW"].sum()
            )
            weighted_rapping_increment += shares[regime] * float(
                attribution.loc[attribution["component"].str.startswith("T"), "increment_kW"].sum()
            )

        r8 = result_table.loc[result_table["regime"] == 7].iloc[0]
        r8_baseline_power = float(policy10.loc[policy10["regime"] == 7, "power_kW"].iloc[0])
        upper_limits = np.array([bound[1] for bound in base_bounds[:4]], dtype=float) * factor
        voltage_values = result_table[[f"U{i}_kV" for i in range(1, 5)]].to_numpy(dtype=float)
        max_voltage_utilization = float(np.max(voltage_values / upper_limits))
        summary_rows.append(
            {
                "extension_pct": round(100.0 * (factor - 1.0)),
                "voltage_upper_factor": float(factor),
                "all_regimes_feasible": bool(result_table["feasible"].all()),
                "infeasible_regimes": int((~result_table["feasible"]).sum()),
                "weighted_power_kW": weighted_power,
                "weighted_increase_vs_10_pct": 100.0 * (weighted_power - weighted_power_10) / weighted_power_10,
                "weighted_voltage_increment_kW": weighted_voltage_increment,
                "weighted_rapping_increment_kW": weighted_rapping_increment,
                "max_voltage_utilization": max_voltage_utilization,
                "minimum_compliance_probability": float(result_table["compliance_probability"].min()),
                "maximum_dust_load_index": float(result_table["dust_load_index"].max()),
                "r8_power_kW": float(r8["power_kW"]),
                "r8_increase_pct": 100.0 * (float(r8["power_kW"]) - r8_baseline_power) / r8_baseline_power,
                "r8_c_p95_mgNm3": float(r8["c_p95_mgNm3"]),
                "r8_compliance_probability": float(r8["compliance_probability"]),
                "r8_dust_load_index": float(r8["dust_load_index"]),
                **{column: float(r8[column]) for column in POLICY_COLUMNS},
            }
        )
        print(
            f"{100 * (factor - 1):.0f}%: weighted={weighted_power:.3f} kW, "
            f"increase={summary_rows[-1]['weighted_increase_vs_10_pct']:.3f}%, "
            f"R8={summary_rows[-1]['r8_increase_pct']:.3f}%",
            flush=True,
        )

    summary = pd.DataFrame(summary_rows)
    policies = pd.concat(policy_rows, ignore_index=True)
    attributions = pd.concat(attribution_rows, ignore_index=True)
    summary.to_csv(table_dir / "q4_voltage_extension_2_to_10_comparison.csv", index=False)
    policies.to_csv(table_dir / "q4_voltage_extension_2_to_10_policies.csv", index=False)
    attributions.to_csv(table_dir / "q4_voltage_extension_2_to_10_attribution.csv", index=False)
    _plot(summary, figure_dir / "fig15_voltage_extension_2_to_10.png")

    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
