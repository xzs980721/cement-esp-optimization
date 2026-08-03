from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_audit import load_and_audit_data
from src.emission_twin import fit_emission_twin
from src.optimization import POLICY_COLUMNS, policy_bounds, policy_priority_table
from src.power_model import fit_power_surrogate
from src.regimes import segment_regimes
from src.reporting import COLORS, setup_plot_style, save_figure


PARAMETER_ORDER = ["U1_kV", "U2_kV", "U3_kV", "U4_kV", "T1_s", "T2_s", "T3_s", "T4_s"]
PARAMETER_LABELS = {
    "U1_kV": "$U_1$",
    "U2_kV": "$U_2$",
    "U3_kV": "$U_3$",
    "U4_kV": "$U_4$",
    "T1_s": "$T_1$",
    "T2_s": "$T_2$",
    "T3_s": "$T_3$",
    "T4_s": "$T_4$",
}


def _load_models(config_path: Path):
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(config["project"]["seed"])
    csv_path = (PROJECT_ROOT / config["project"]["input_csv"]).resolve()
    audit = load_and_audit_data(
        csv_path,
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


def _plot_benefit_matrix(wide: pd.DataFrame, path: Path) -> None:
    setup_plot_style()
    voltage = wide[["U1_kV", "U2_kV", "U3_kV", "U4_kV"]]
    rapping = wide[["T1_s", "T2_s", "T3_s", "T4_s"]]

    fig, axes = plt.subplots(
        1, 2, figsize=(12.5, 5.6),
        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.28},
    )
    sns.heatmap(
        voltage, annot=True, fmt=".4f", cmap="YlGnBu",
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "减排收益 / (mg/Nm3 per kW)", "shrink": 0.82},
        annot_kws={"fontsize": 8},
        ax=axes[0],
    )
    sns.heatmap(
        rapping, annot=True, fmt=".4f", cmap="YlOrBr",
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "减排收益 / (mg/Nm3 per kW)", "shrink": 0.82},
        annot_kws={"fontsize": 8},
        ax=axes[1],
    )
    for ax, title, columns in [
        (axes[0], "(a) 提高电压 1% 的单位新增功率减排收益", voltage.columns),
        (axes[1], "(b) 缩短振打周期 1% 的单位新增功率减排收益", rapping.columns),
    ]:
        ax.set_title(title, pad=10)
        ax.set_xlabel("控制变量")
        ax.set_ylabel("典型工况")
        ax.set_xticklabels(
            [PARAMETER_LABELS[col] for col in columns], rotation=0,
        )
        ax.set_yticklabels(wide.index, rotation=0)
        ax.tick_params(axis="both", labelsize=8)

    fig.suptitle("八类工况的单位新增功率减排收益", y=1.01)
    fig.tight_layout()
    save_figure(fig, path)


def main() -> None:
    config_path = PROJECT_ROOT / "config" / "model.yaml"
    config, seed, audit, power, regimes, twin = _load_models(config_path)
    policy_table = pd.read_csv(PROJECT_ROOT / "outputs" / "tables" / "optimal_policy_10.csv")
    bounds = policy_bounds(audit.frame)
    n_scenarios = int(config["optimization"]["verification_scenarios"])

    frames: list[pd.DataFrame] = []
    # Retain the two representative-regime seeds used by the existing paper so
    # that R2 and R8 reproduce its reported marginal-benefit values exactly.
    representative_seeds = {1: seed + 30000, 7: seed + 31000}
    for regime in range(regimes.n_regimes):
        regime_frame = audit.frame.loc[regimes.labels == regime].reset_index(drop=True)
        policy_row = policy_table.loc[policy_table["regime"] == regime].iloc[0]
        policy = policy_row[POLICY_COLUMNS].to_numpy(dtype=float)
        priority = policy_priority_table(
            policy,
            regime_frame,
            twin,
            power,
            bounds,
            seed=representative_seeds.get(regime, seed + 32000 + regime),
            n_scenarios=n_scenarios,
        )
        priority.insert(0, "regime", regime)
        priority.insert(1, "label", f"R{regime + 1}")
        priority["parameter_order"] = priority["parameter"].map(
            {parameter: index for index, parameter in enumerate(PARAMETER_ORDER)}
        )
        frames.append(priority)

    long_table = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["regime", "parameter_order"])
        .drop(columns="parameter_order")
        .reset_index(drop=True)
    )
    wide = (
        long_table.pivot(index="label", columns="parameter", values="reduction_per_added_kW")
        .reindex(index=[f"R{i + 1}" for i in range(regimes.n_regimes)], columns=PARAMETER_ORDER)
    )
    wide.index.name = "工况"

    summary_rows: list[dict[str, object]] = []
    for regime in range(regimes.n_regimes):
        subset = long_table.loc[long_table["regime"] == regime].copy()
        best_overall = subset.loc[subset["reduction_per_added_kW"].idxmax()]
        best_voltage = subset.loc[
            subset["parameter"].str.startswith("U"), "reduction_per_added_kW"
        ].idxmax()
        best_rapping = subset.loc[
            subset["parameter"].str.startswith("T"), "reduction_per_added_kW"
        ].idxmax()
        summary_rows.append(
            {
                "regime": regime,
                "label": f"R{regime + 1}",
                "best_parameter": best_overall["parameter"],
                "best_benefit": best_overall["reduction_per_added_kW"],
                "best_voltage_parameter": long_table.loc[best_voltage, "parameter"],
                "best_voltage_benefit": long_table.loc[best_voltage, "reduction_per_added_kW"],
                "best_rapping_parameter": long_table.loc[best_rapping, "parameter"],
                "best_rapping_benefit": long_table.loc[best_rapping, "reduction_per_added_kW"],
            }
        )
    summary = pd.DataFrame(summary_rows)

    table_dir = PROJECT_ROOT / "outputs" / "tables"
    figure_dir = PROJECT_ROOT / "outputs" / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    long_table.to_csv(
        table_dir / "all_regime_marginal_benefit_long.csv", index=False, encoding="utf-8-sig"
    )
    wide.reset_index().to_csv(
        table_dir / "all_regime_marginal_benefit_matrix.csv", index=False, encoding="utf-8-sig"
    )
    summary.to_csv(
        table_dir / "all_regime_marginal_benefit_summary.csv", index=False, encoding="utf-8-sig"
    )
    _plot_benefit_matrix(wide, figure_dir / "fig12_all_regime_marginal_benefit.png")

    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(wide.round(6).to_string())
        print("\nTop actions")
        print(summary.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
