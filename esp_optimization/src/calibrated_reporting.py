from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from .calibrated_emission import (
    UncensoredDistribution,
    regime_reference,
    relative_physical_response,
)
from .emission_twin import EmissionTwin, T_COLUMNS, U_COLUMNS
from .q1_diagnostics import _historical_twin_score
from .reporting import COLORS, NEUTRAL_GRAY, RISK_RED, SAFE_TEAL, save_figure, setup_plot_style


def plot_calibrated_distribution_comparison(
    frame: pd.DataFrame,
    twin: EmissionTwin,
    distribution: UncensoredDistribution,
    path: Path,
    seed: int = 2026,
) -> None:
    setup_plot_style()
    observed = frame.loc[frame["C_out_mgNm3"] < distribution.threshold, "C_out_mgNm3"].to_numpy()
    old = _historical_twin_score(frame, twin)[frame["C_out_mgNm3"].to_numpy() < distribution.threshold]
    calibrated = distribution.sample(len(observed), seed=seed, draw_distribution=False)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))

    axes[0].boxplot(
        [observed, old, calibrated],
        tick_labels=["未触顶观测", "旧绝对物理预测", "新观测分布层"],
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": COLORS[0], "alpha": 0.55},
        medianprops={"color": RISK_RED, "linewidth": 1.2},
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"出口浓度 / (mg·Nm$^{-3}$)，对数尺度")
    axes[0].set_title("(a) 预测分布尺度比较")
    axes[0].tick_params(axis="x", rotation=10)

    for values, label, color, style in [
        (observed, "未触顶观测", COLORS[0], "-"),
        (calibrated, "新观测分布层", SAFE_TEAL, "--"),
        (old, "旧绝对物理预测", RISK_RED, ":"),
    ]:
        ordered = np.sort(values)
        probability = np.arange(1, len(ordered) + 1) / len(ordered)
        axes[1].plot(ordered, probability, label=label, color=color, ls=style, lw=1.4)
    axes[1].set_xlim(48.7, 50.02)
    axes[1].set_xlabel(r"出口浓度 / (mg·Nm$^{-3}$)")
    axes[1].set_ylabel("经验累积概率")
    axes[1].set_title("(b) 量程以下分布的局部比较")
    axes[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    save_figure(fig, path)


def _binned_uncensored_effect(
    frame: pd.DataFrame, column: str, n_bins: int = 10
) -> pd.DataFrame:
    data = frame.loc[frame["C_out_mgNm3"] < 50, [column, "C_out_mgNm3"]].dropna().copy()
    data["bin"] = pd.qcut(data[column], n_bins, duplicates="drop")
    rows = []
    for _, group in data.groupby("bin", observed=True):
        values = group["C_out_mgNm3"].to_numpy(dtype=float)
        n = len(values)
        mean = float(np.mean(values))
        half = float(student_t.ppf(0.975, n - 1) * np.std(values, ddof=1) / np.sqrt(n))
        rows.append(
            {
                "x": float(group[column].median()),
                "mean": mean,
                "lower": mean - half,
                "upper": mean + half,
            }
        )
    return pd.DataFrame(rows)


def plot_calibrated_factor_comparison(
    frame: pd.DataFrame,
    twin: EmissionTwin,
    path: Path,
) -> None:
    setup_plot_style()
    working = frame.copy()
    working["voltage_index"] = (
        (working[U_COLUMNS].to_numpy(dtype=float) / twin.u_ref) ** 2
    ) @ twin.stage_weights
    working["rapping_index"] = (
        working[T_COLUMNS].to_numpy(dtype=float) / twin.t_ref
    ) @ twin.stage_weights
    conditions, reference_policy = regime_reference(working)
    specifications = [
        ("入口浓度", "C_in_gNm3", r"g·Nm$^{-3}$"),
        ("烟气流量", "Q_Nm3h", r"$10^3$ Nm$^3$·h$^{-1}$"),
        ("温度", "Temp_C", r"$^\circ$C"),
        ("综合电压指数", "voltage_index", "相对参考值"),
        ("综合振打指数", "rapping_index", "相对参考值"),
    ]
    fig, axes = plt.subplots(5, 2, figsize=(9.5, 12.0))
    for row, (label, column, unit) in enumerate(specifications):
        observed = _binned_uncensored_effect(working, column)
        x_values = observed["x"].to_numpy(dtype=float)
        model_values = []
        for value in x_values:
            candidate_conditions = dict(conditions)
            candidate_policy = reference_policy.copy()
            if column == "C_in_gNm3":
                candidate_conditions["c_in"] = float(value)
            elif column == "Q_Nm3h":
                candidate_conditions["q"] = float(value)
            elif column == "Temp_C":
                candidate_conditions["temp_c"] = float(value)
            elif column == "voltage_index":
                candidate_policy[:4] = reference_policy[:4] * np.sqrt(float(value))
            else:
                candidate_policy[4:] = reference_policy[4:] * float(value)
            model_values.append(
                relative_physical_response(
                    twin,
                    candidate_policy,
                    candidate_conditions,
                    reference_policy,
                    conditions,
                )
            )
        x_plot = x_values / 1000.0 if column == "Q_Nm3h" else x_values
        axes[row, 0].plot(x_plot, model_values, color=COLORS[0], lw=1.4)
        axes[row, 0].axhline(1.0, color=NEUTRAL_GRAY, lw=0.7, ls="--")
        axes[row, 0].set_ylabel("物理相对倍率")
        axes[row, 0].set_title(f"{label}：物理层")

        yerr = np.vstack(
            [
                observed["mean"] - observed["lower"],
                observed["upper"] - observed["mean"],
            ]
        )
        axes[row, 1].errorbar(
            x_plot,
            observed["mean"],
            yerr=yerr,
            color=SAFE_TEAL,
            marker="o",
            markersize=3.2,
            lw=1.0,
            capsize=2,
        )
        axes[row, 1].axhline(
            float(distribution_mean(working)), color=NEUTRAL_GRAY, lw=0.7, ls="--"
        )
        axes[row, 1].set_ylim(49.70, 49.82)
        axes[row, 1].set_ylabel(r"未触顶均值 / (mg·Nm$^{-3}$)")
        axes[row, 1].set_title(f"{label}：数据层")
        axes[row, 0].set_xlabel(unit)
        axes[row, 1].set_xlabel(unit)
    fig.suptitle("数据层无显著条件效应与物理层相对响应的证据分离", y=0.998)
    fig.tight_layout()
    save_figure(fig, path)


def distribution_mean(frame: pd.DataFrame) -> float:
    return float(
        frame.loc[frame["C_out_mgNm3"] < 50, "C_out_mgNm3"].mean()
    )


def plot_calibrated_policy_comparison(
    comparison: pd.DataFrame,
    path: Path,
) -> None:
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.3, 4.0))
    x = np.arange(len(comparison))
    width = 0.36
    axes[0].bar(
        x - width / 2,
        comparison["old_power_10_kW"],
        width,
        label="旧绝对模型",
        color=COLORS[6],
    )
    new_bars = axes[0].bar(
        x + width / 2,
        comparison["calibrated_power_10_kW"],
        width,
        label="新双层模型",
        color=COLORS[0],
    )
    for bar, feasible in zip(new_bars, comparison["calibrated_feasible_10"]):
        if not bool(feasible):
            bar.set_facecolor(RISK_RED)
            bar.set_hatch("//")
    axes[0].set_xticks(x, comparison["label"])
    axes[0].set_ylabel("10限值最优功率 / kW")
    axes[0].set_title("(a) 各工况10限值功率")
    axes[0].legend(fontsize=7, frameon=False)

    axes[1].bar(
        comparison["label"],
        comparison["calibrated_power_change_pct"],
        color=[COLORS[1] if value <= 0 else RISK_RED for value in comparison["calibrated_power_change_pct"]],
    )
    axes[1].axhline(0.0, color=NEUTRAL_GRAY, lw=0.7)
    axes[1].set_ylabel("新模型相对旧模型功率变化 / %")
    axes[1].set_title("(b) 策略变化幅度")
    for index, row in comparison.reset_index(drop=True).iterrows():
        if not bool(row["calibrated_feasible_10"]):
            axes[1].text(index, 0.5, "历史边界\n不可行", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    save_figure(fig, path)
