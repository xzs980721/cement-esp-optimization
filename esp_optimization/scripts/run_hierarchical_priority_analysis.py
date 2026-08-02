from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_audit import load_and_audit_data
from src.emission_twin import fit_emission_twin
from src.hierarchical_priority import (
    hierarchical_priority_table,
    summarize_hierarchical_priority,
)
from src.optimization import POLICY_COLUMNS, policy_bounds
from src.power_model import fit_power_surrogate
from src.regimes import segment_regimes
from src.reporting import COLORS, setup_plot_style


ACTIVE_TOLERANCE = 1e-4
TIE_TOLERANCE = 0.02
N_SCENARIOS = 10000


def _markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    def render(value: object) -> str:
        if pd.isna(value):
            return "—"
        if isinstance(value, (bool, np.bool_)):
            return "是" if bool(value) else "否"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{digits}f}"
        return str(value)

    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(render(value) for value in row) + " |")
    return "\n".join(lines)


def _load_context():
    config_path = PROJECT_ROOT / "config" / "model.yaml"
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


def _verification_seed(
    base_seed: int,
    regime: int,
    limit: float,
    repaired: bool,
) -> int:
    optimizer_seed = base_seed + 1000 * regime + int(limit * 10)
    return optimizer_seed + (300000 if repaired else 100000)


def _revalidate_table(
    source: pd.DataFrame,
    base_seed: int,
    limit: float,
    audit,
    regimes,
    twin,
    power,
    dust_limit: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, source_row in source.sort_values("regime").iterrows():
        regime = int(source_row["regime"])
        policy = source_row[POLICY_COLUMNS].to_numpy(dtype=float)
        regime_frame = audit.frame.loc[regimes.labels == regime].reset_index(drop=True)
        repaired = bool(source_row["verification_repair"])
        scenarios = twin.scenario_pack(
            regime_frame,
            n_scenarios=N_SCENARIOS,
            seed=_verification_seed(base_seed, regime, limit, repaired),
        )
        concentration = twin.predict_scenarios(policy, scenarios)
        recalculated = {
            "regime": regime,
            "label": f"R{regime + 1}",
            "limit_mgNm3": limit,
            "power_kW": float(power.predict_policy(policy)),
            "c_mean_mgNm3": float(np.mean(concentration)),
            "c_p95_mgNm3": float(np.quantile(concentration, 0.95)),
            "compliance_probability": float(np.mean(concentration <= limit)),
            "dust_load_index": float(twin.dust_load_index(policy, regime_frame)),
        }
        recalculated["feasible"] = bool(
            recalculated["c_p95_mgNm3"] <= limit + 1e-6
            and recalculated["dust_load_index"] <= dust_limit + 1e-6
        )
        recalculated["power_abs_error_kW"] = abs(
            recalculated["power_kW"] - float(source_row["power_kW"])
        )
        recalculated["p95_abs_error_mgNm3"] = abs(
            recalculated["c_p95_mgNm3"] - float(source_row["c_p95_mgNm3"])
        )
        recalculated["probability_abs_error"] = abs(
            recalculated["compliance_probability"]
            - float(source_row["compliance_probability"])
        )
        rows.append(recalculated)
    return pd.DataFrame(rows)


def _priority_for_policies(
    policy_table: pd.DataFrame,
    audit,
    regimes,
    twin,
    power,
    bounds,
    seeds: dict[int, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    action_tables: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for regime in range(regimes.n_regimes):
        regime_frame = audit.frame.loc[regimes.labels == regime].reset_index(drop=True)
        policy = policy_table.loc[
            policy_table["regime"] == regime, POLICY_COLUMNS
        ].iloc[0].to_numpy(dtype=float)
        actions = hierarchical_priority_table(
            policy=policy,
            regime_frame=regime_frame,
            twin=twin,
            power=power,
            bounds=bounds,
            seed=seeds[regime],
            regime=regime,
            n_scenarios=N_SCENARIOS,
            dust_limit=1.45,
            active_tolerance=ACTIVE_TOLERANCE,
            tie_relative_tolerance=TIE_TOLERANCE,
        )
        action_tables.append(actions)
        summaries.append(summarize_hierarchical_priority(actions))
    return pd.concat(action_tables, ignore_index=True), pd.DataFrame(summaries)


def _seed_stability(
    policy10: pd.DataFrame,
    audit,
    regimes,
    twin,
    power,
    bounds,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for regime in [1, 7]:
        frame = audit.frame.loc[regimes.labels == regime].reset_index(drop=True)
        policy = policy10.loc[
            policy10["regime"] == regime, POLICY_COLUMNS
        ].iloc[0].to_numpy(dtype=float)
        for repeat in range(12):
            actions = hierarchical_priority_table(
                policy=policy,
                regime_frame=frame,
                twin=twin,
                power=power,
                bounds=bounds,
                seed=seed + 50000 + 1000 * regime + repeat,
                regime=regime,
                n_scenarios=N_SCENARIOS,
                dust_limit=1.45,
                active_tolerance=ACTIVE_TOLERANCE,
                tie_relative_tolerance=TIE_TOLERANCE,
            )
            summary = summarize_hierarchical_priority(actions)
            summary.update({"repeat": repeat + 1, "seed": seed + 50000 + 1000 * regime + repeat})
            rows.append(summary)
    return pd.DataFrame(rows)


def _plot_dual_metrics(actions: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    setup_plot_style()
    labels = summary["label"].tolist()
    best_voltage: list[float] = []
    best_period: list[float] = []
    best_relief: list[float] = []
    for label in labels:
        subset = actions.loc[actions["label"] == label]
        best_voltage.append(
            float(
                subset.loc[
                    subset["parameter_type"] == "电压",
                    "direct_benefit_mgNm3_per_kW",
                ].max()
            )
        )
        best_period.append(
            float(
                subset.loc[
                    subset["parameter_type"] == "振打周期",
                    "direct_benefit_mgNm3_per_kW",
                ].max()
            )
        )
        best_relief.append(float(subset["dust_relief_per_added_kW"].max()))

    x = np.arange(len(labels))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))
    axes[0].bar(x - width / 2, best_voltage, width, label="最佳电压变量", color=COLORS[0])
    axes[0].bar(x + width / 2, best_period, width, label="最佳周期变量", color=COLORS[2])
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("mg/Nm³ per kW")
    axes[0].set_title("直接减排收益 $E_j$")
    axes[0].legend()

    colors = [COLORS[3] if active else COLORS[1] for active in summary["dust_constraint_active"]]
    axes[1].bar(x, best_relief, color=colors)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("尘负荷指数 per kW")
    axes[1].set_title("周期变量的约束松弛收益 $G_{T_i}$")
    axes[1].legend(
        handles=[
            Patch(facecolor=COLORS[1], label="约束不活跃：电压优先"),
            Patch(facecolor=COLORS[3], label="约束活跃：周期优先"),
        ],
        loc="lower right",
    )
    fig.suptitle("八类工况的分层控制判据", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=230, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _draw_box(ax, xy, width, height, text, edge, fill):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.025",
        linewidth=1.8,
        edgecolor=edge,
        facecolor=fill,
    )
    ax.add_patch(box)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=11)
    return box


def _plot_control_flow(summary: pd.DataFrame, path: Path) -> None:
    setup_plot_style()
    r2 = summary.loc[summary["label"] == "R2"].iloc[0]
    r8 = summary.loc[summary["label"] == "R8"].iloc[0]
    fig, ax = plt.subplots(figsize=(13.2, 5.6))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 5.6)
    ax.axis("off")
    lanes = [
        (4.05, "低负荷 R2", r2, COLORS[0], "#EAF2F8"),
        (1.20, "高负荷 R8", r8, COLORS[3], "#FDEDEC"),
    ]
    for y, label, row, color, fill in lanes:
        ax.text(0.15, y + 0.42, label, fontsize=13, fontweight="bold", color=color, va="center")
        slack = max(0.0, 100 * float(row["relative_dust_slack"]))
        primary = str(row["primary_parameter"]).replace("_kV", "").replace("_s", "")
        secondary = str(row["secondary_parameter"]).replace("_kV", "").replace("_s", "")
        texts = (
            [
                f"$I_M={row['dust_load_index']:.4f}$\n裕量 {slack:.2f}%",
                "尘负荷约束不活跃",
                "按直接收益 $E_j$ 排序",
                f"先调 {primary}\n其次 {secondary}",
            ]
            if label.endswith("R2")
            else [
                f"$I_M={row['dust_load_index']:.4f}$\n裕量 {slack:.2f}%",
                "尘负荷约束活跃",
                "按松弛收益 $G_{T_i}$ 排序",
                f"先调 {primary}\n恢复裕量后调 {secondary}",
            ]
        )
        xs = [2.05, 4.85, 7.65, 10.45]
        for index, (x, text) in enumerate(zip(xs, texts)):
            _draw_box(ax, (x, y), 2.15, 0.84, text, color, fill)
            if index < 3:
                ax.add_patch(
                    FancyArrowPatch(
                        (x + 2.15, y + 0.42),
                        (xs[index + 1], y + 0.42),
                        arrowstyle="-|>",
                        mutation_scale=14,
                        linewidth=1.6,
                        color=color,
                    )
                )
    ax.set_title("R2 与 R8 的词典序控制流程", fontsize=16, fontweight="bold", pad=14)
    fig.savefig(path, dpi=230, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _write_report(
    policy10: pd.DataFrame,
    summary10: pd.DataFrame,
    actions10: pd.DataFrame,
    revalidation10: pd.DataFrame,
    q4_recheck: pd.DataFrame,
    stability: pd.DataFrame,
    q4_screen: pd.DataFrame,
    bootstrap: pd.DataFrame,
    voltage_upper_factor: float,
    path: Path,
) -> None:
    q2 = revalidation10.merge(
        summary10[
            [
                "regime",
                "relative_dust_slack",
                "dust_constraint_active",
                "primary_action_class",
                "primary_parameter",
                "secondary_parameter",
            ]
        ],
        on="regime",
    )
    q2_display = q2[
        [
            "label",
            "power_kW",
            "c_p95_mgNm3",
            "compliance_probability",
            "dust_load_index",
            "relative_dust_slack",
            "primary_action_class",
            "primary_parameter",
        ]
    ].copy()
    q2_display.columns = [
        "工况",
        "功率/kW",
        "浓度P95",
        "达标概率",
        "尘负荷指数",
        "相对裕量",
        "一级动作",
        "首选变量",
    ]
    q2_display["达标概率"] *= 100
    q2_display["相对裕量"] = (100 * q2_display["相对裕量"]).clip(lower=0.0)

    q2_policy = policy10[["regime", *POLICY_COLUMNS]].copy()
    q2_policy["regime"] = q2_policy["regime"].astype(int).map(lambda value: f"R{value + 1}")
    q2_policy.columns = [
        "工况",
        "$U_1$/kV",
        "$U_2$/kV",
        "$U_3$/kV",
        "$U_4$/kV",
        "$T_1$/s",
        "$T_2$/s",
        "$T_3$/s",
        "$T_4$/s",
    ]

    representative = summary10.loc[summary10["label"].isin(["R2", "R8"])].copy()
    representative = representative[
        [
            "label",
            "dust_load_index",
            "relative_dust_slack",
            "primary_action_class",
            "primary_parameter",
            "secondary_parameter",
        ]
    ]
    representative.columns = ["工况", "$I_M$", "相对裕量", "一级动作", "首选变量", "二级变量"]
    representative["相对裕量"] = (100 * representative["相对裕量"]).clip(lower=0.0)

    stability_counts = (
        stability.groupby(["label", "primary_action_class"], as_index=False)
        .size()
        .rename(columns={"label": "工况", "primary_action_class": "优先类别", "size": "出现次数/12"})
    )

    q4_weighted_10 = float(np.sum(q4_recheck["share"] * q4_recheck["power_10_kW"]))
    q4_weighted_5 = float(np.sum(q4_recheck["share"] * q4_recheck["power_5_kW"]))
    weighted_increase = 100 * (q4_weighted_5 - q4_weighted_10) / q4_weighted_10
    r8_increase = float(q4_recheck.loc[q4_recheck["label"] == "R8", "increase_pct"].iloc[0])
    valid_bootstrap = bootstrap.loc[bootstrap["feasible_10"] & bootstrap["feasible_5"]]
    interval = np.quantile(valid_bootstrap["increase_pct"], [0.025, 0.975])
    raw_infeasible = q4_recheck.loc[~q4_recheck["raw_5_feasible"], "label"].tolist()
    extension_row = q4_screen.loc[
        np.isclose(q4_screen["voltage_upper_factor"], voltage_upper_factor)
    ].iloc[0]
    extension_pct = 100.0 * (voltage_upper_factor - 1.0)

    q4_display = q4_recheck[
        [
            "label",
            "power_10_kW",
            "power_5_kW",
            "increase_pct",
            "raw_5_feasible",
            "conditional_5_feasible",
            "priority_10",
            "primary_parameter_10",
        ]
    ].copy()
    q4_display.columns = [
        "工况",
        "10限值功率/kW",
        "5限值功率/kW",
        "增幅/%",
        "历史边界可行",
        f"{extension_pct:.0f}%扩展可行",
        "一级动作",
        "首选变量",
    ]

    r2_actions = actions10.loc[actions10["label"] == "R2"]
    r8_actions = actions10.loc[actions10["label"] == "R8"]
    r2_u3 = float(
        r2_actions.loc[r2_actions["parameter"] == "U3_kV", "direct_benefit_mgNm3_per_kW"].iloc[0]
    )
    r8_t3 = float(
        r8_actions.loc[r8_actions["parameter"] == "T3_s", "dust_relief_per_added_kW"].iloc[0]
    )
    r8_t4 = float(
        r8_actions.loc[r8_actions["parameter"] == "T4_s", "dust_relief_per_added_kW"].iloc[0]
    )

    text = rf"""# 分层控制优先级模型：四问重新计算结果

## 1. 模型改进

保留原删失排放数字孪生、功率代理及 95% 机会约束，不改变分场权重、温度修正、$\kappa_M=0.18$ 和 $\delta=0.07$。改进仅作用于控制动作排序，采用“安全可行性优先、直接减排效率其次”的词典序规则。

直接减排收益仍为

$$
E_j=\frac{{q_{{0.95}}(C\mid\boldsymbol x)-q_{{0.95}}(C\mid\boldsymbol x+\Delta x_j\boldsymbol e_j)}}
{{P(\boldsymbol x+\Delta x_j\boldsymbol e_j)-P(\boldsymbol x)}}.
$$

定义尘负荷相对裕量

$$
s_M=\frac{{1.45-I_M}}{{1.45}}.
$$

当 $s_M\le10^{{-4}}$ 时认为尘负荷约束活跃。此时先按

$$
G_{{T_i}}=\frac{{I_M(\boldsymbol x)-I_M(\boldsymbol x_{{T_i,-1\%}})}}
{{P(\boldsymbol x_{{T_i,-1\%}})-P(\boldsymbol x)}}
$$

选择振打周期；恢复安全裕量后，再按 $E_j$ 调节电压。该规则没有引入人为综合权重。

## 2. 问题一：因素关系、平均浓度与振打峰值

电压仍通过 $U_i^2$ 进入 D--A 去除指数，主要控制稳态穿透。令 $r_i=T_i/T_{{i,0}}$、$L=C_{{\rm in}}Q/(C_{{\rm in,0}}Q_0)$，则平均出口浓度写为

$$
\bar C_{{\rm out}}=1000C_{{\rm in}}\exp(-D_{{\rm eff}})R_{{\rm rap}},
\qquad
R_{{\rm rap}}=1+\delta L\sum_{{i=1}}^4w_ir_i,
$$

其中

$$
D_{{\rm eff}}=D_0\frac{{Q_0}}{{Q}}f_\Theta(\Theta)
\sum_{{i=1}}^4w_i\left(\frac{{U_i}}{{U_{{i,0}}}}\right)^2
\exp\left[-\kappa_ML\sum_{{i=1}}^4w_i\max(r_i-1,0)^2\right].
$$

因此，振打周期同时作用于平均浓度和瞬时峰值：

1. 平均浓度方面，周期延长会提高线性再飞扬因子；当 $r_i>1$ 时，还会通过积灰惩罚降低去除指数。其对数导数为

   $$
   \frac{{\partial\ln \bar C_{{\rm out}}}}{{\partial T_i}}
   =\frac{{\delta Lw_i}}{{T_{{i,0}}R_{{\rm rap}}}}
   +\frac{{2\kappa_MLw_i(r_i-1)_+D_{{\rm eff}}}}{{T_{{i,0}}}}>0,
   $$

   故在模型适用范围内，延长任一振打周期均会提高平均出口浓度。
2. 瞬时峰值方面，一阶尘量守恒给出 $M_{{\rm rap,i}}\approx\eta_i\dot M_iT_i$，从而

   $$
   C_{{\rm peak,i}}-\bar C_{{\rm out}}
   \propto\frac{{\eta_i\dot M_iT_i}}{{Q\Delta t_i}}.
   $$

   在脱落比例 $\eta_i$ 与脉冲持续时间 $\Delta t_i$ 不变的局部近似下，单次峰值随 $T_i$ 近似线性增加；周期缩短会降低单次峰值，但增加机械动作功率。
3. 数据缺少实际振打触发时刻，故峰值结论仍限于归一化相位和相对风险，不解释为实测秒级峰值。

## 3. 问题二：八类工况最低功率及分层动作

固定原随机种子，对 $10\,\mathrm{{mg/Nm^3}}$ 策略重新生成独立 10,000 情景，八类工况均通过排放与尘负荷约束。R1--R3 具有明确尘负荷裕量，归入电压优先层；R4--R8 位于 $I_M=1.45$ 的活跃边界，归入周期优先层。

{_markdown_table(q2_display, digits=4)}

这里的“周期优先”表示先恢复约束裕量，而不是周期的直接 $E_j$ 高于电压。

对应的八变量最优设定如下：

{_markdown_table(q2_policy, digits=2)}

## 4. 问题三：两种代表工况

{_markdown_table(representative, digits=4)}

- **低负荷 R2：** $I_M=1.0982$，相对裕量为 24.26%，尘负荷约束不活跃。按直接收益排序，$U_3$ 的 $E_j={r2_u3:.4f}\,\mathrm{{mg/Nm^3/kW}}$，故优先调节电压，首选 $U_3$，其次 $U_1$。
- **高负荷 R8：** $I_M=1.45$，相对裕量为 0，必须先调振打周期。$T_3,T_4$ 的 $G_T$ 分别为 {r8_t3:.6f} 和 {r8_t4:.6f}，相对差异小于 2%，按并列优先处理；恢复裕量后再优先调节 $U_4$。

12 组独立情景重复结果如下，说明“R2 电压优先、R8 周期优先”的类别结论不依赖单一随机种子：

{_markdown_table(stability_counts, digits=0)}

## 5. 问题四：限值收紧的可行性与电耗增幅

历史边界内不满足 $5\,\mathrm{{mg/Nm^3}}$ 的工况仍为 {"、".join(raw_infeasible)}。条件情景不再搜索临界可行扩展量，而是直接将四场历史电压上界设为原值的 {voltage_upper_factor:.2f} 倍；其加权功率由 {q4_weighted_10:.2f} kW 增至 {q4_weighted_5:.2f} kW，增幅为 **{weighted_increase:.2f}%**。R8 增幅为 **{r8_increase:.2f}%**，12 组重采样经验范围为 **[{interval[0]:.2f}%, {interval[1]:.2f}%]**。该情景下八类工况均满足约束。

{_markdown_table(q4_display, digits=2)}

分层动作解释为：从 $10$ 限值策略向 $5$ 限值策略过渡时，高负荷工况先通过 $T_3/T_4$ 恢复尘负荷安全裕量，再利用扩展后的电压空间降低稳态穿透。该顺序同时解释了 R8 中电压强化与周期缩短共同造成的功率增量。

## 6. 数值结论的变化

- 问题一至问题三的核心排放模型、功率模型和 $10\,\mathrm{{mg/Nm^3}}$ 策略不变；问题四的主情景改为电压上界 110%，相应功率、增幅、归因与重采样范围均以该情景重新计算。
- 改进发生在第三问的“优先”定义：原 $E_j$ 是直接平均减排效率，新模型增加活跃约束层，能够区分低负荷电压优先和高负荷周期优先。
- 固定种子复算的最大功率误差、P95 误差和达标概率误差均记录在复核表中。

## 7. 结论边界

分层优先级是现场动作顺序，不表示振打周期的直接减排收益超过电压。附件仍缺少二次电流和实际振打事件，因此上线前需用低量程出口浓度与 PLC 振打记录重新标定。
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    config, seed, audit, power, regimes, twin = _load_context()
    table_dir = PROJECT_ROOT / "outputs" / "tables"
    figure_dir = PROJECT_ROOT / "outputs" / "figures"
    answer_dir = PROJECT_ROOT / "answers"
    for directory in [table_dir, figure_dir, answer_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    policy10 = pd.read_csv(table_dir / "optimal_policy_10.csv")
    policy5_raw = pd.read_csv(table_dir / "optimal_policy_5_raw_bounds.csv")
    policy5 = pd.read_csv(table_dir / "optimal_policy_5_conditional_extension.csv")
    q4 = pd.read_csv(table_dir / "q4_energy_increase.csv")
    q4_screen = pd.read_csv(table_dir / "q4_voltage_extension_screen.csv")
    bootstrap = pd.read_csv(table_dir / "q4_high_regime_bootstrap.csv")
    dust_limit = float(config["optimization"]["dust_load_limit"])
    voltage_upper_factor = float(
        config["optimization"]["q4_voltage_upper_extension_factor"]
    )

    seeds10 = {
        regime: (seed + 30000 if regime == 1 else seed + 31000 if regime == 7 else seed + 32000 + regime)
        for regime in range(regimes.n_regimes)
    }
    actions10, summary10 = _priority_for_policies(
        policy10,
        audit,
        regimes,
        twin,
        power,
        policy_bounds(audit.frame),
        seeds10,
    )
    seeds5 = {regime: seed + 61000 + regime for regime in range(regimes.n_regimes)}
    actions5, summary5 = _priority_for_policies(
        policy5,
        audit,
        regimes,
        twin,
        power,
        policy_bounds(audit.frame, voltage_upper_factor=voltage_upper_factor),
        seeds5,
    )

    revalidation10 = _revalidate_table(
        policy10, seed, 10.0, audit, regimes, twin, power, dust_limit
    )
    revalidation5_raw = _revalidate_table(
        policy5_raw, seed + 500, 5.0, audit, regimes, twin, power, dust_limit
    )
    revalidation5 = _revalidate_table(
        policy5, seed + 900, 5.0, audit, regimes, twin, power, dust_limit
    )
    stability = _seed_stability(
        policy10,
        audit,
        regimes,
        twin,
        power,
        policy_bounds(audit.frame),
        seed,
    )

    q4_recheck = q4.copy()
    q4_recheck = q4_recheck.rename(columns={"power_5_conditional_kW": "power_5_kW"})
    q4_recheck = q4_recheck.merge(
        summary10[
            ["regime", "dust_load_index", "primary_action_class", "primary_parameter"]
        ].rename(
            columns={
                "dust_load_index": "dust_10",
                "primary_action_class": "priority_10",
                "primary_parameter": "primary_parameter_10",
            }
        ),
        on="regime",
    ).merge(
        summary5[
            ["regime", "dust_load_index", "primary_action_class", "primary_parameter"]
        ].rename(
            columns={
                "dust_load_index": "dust_5",
                "primary_action_class": "priority_5",
                "primary_parameter": "primary_parameter_5",
            }
        ),
        on="regime",
    )

    actions10.to_csv(
        table_dir / "hierarchical_priority_actions.csv", index=False, encoding="utf-8-sig"
    )
    summary10.to_csv(
        table_dir / "hierarchical_priority_all_regimes.csv", index=False, encoding="utf-8-sig"
    )
    stability.to_csv(
        table_dir / "hierarchical_priority_seed_stability.csv", index=False, encoding="utf-8-sig"
    )
    revalidation10.to_csv(
        table_dir / "hierarchical_priority_q2_revalidation.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(
        [
            revalidation5_raw.assign(scenario="历史边界"),
            revalidation5.assign(
                scenario=f"电压上界为历史上界的{100 * voltage_upper_factor:.0f}%"
            ),
        ],
        ignore_index=True,
    ).to_csv(
        table_dir / "hierarchical_priority_q4_revalidation.csv", index=False, encoding="utf-8-sig"
    )
    q4_recheck.to_csv(
        table_dir / "hierarchical_priority_q4_recheck.csv", index=False, encoding="utf-8-sig"
    )

    _plot_dual_metrics(
        actions10,
        summary10,
        figure_dir / "fig13_hierarchical_priority_metrics.png",
    )
    _plot_control_flow(
        summary10,
        figure_dir / "fig14_r2_r8_control_flow.png",
    )
    _write_report(
        policy10=policy10,
        summary10=summary10,
        actions10=actions10,
        revalidation10=revalidation10,
        q4_recheck=q4_recheck,
        stability=stability,
        q4_screen=q4_screen,
        bootstrap=bootstrap,
        voltage_upper_factor=voltage_upper_factor,
        path=answer_dir / "分层优先级新模型四问结果.md",
    )

    print(summary10.to_string(index=False))
    print("\nRevalidation maxima")
    print(
        revalidation10[
            ["power_abs_error_kW", "p95_abs_error_mgNm3", "probability_abs_error"]
        ].max().to_string()
    )


if __name__ == "__main__":
    main()
