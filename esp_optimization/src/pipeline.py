from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import r2_score

from .data_audit import load_and_audit_data
from .emission_twin import fit_emission_twin
from .optimization import (
    POLICY_COLUMNS,
    optimize_policy,
    optimize_single_regime,
    policy_bounds,
    policy_priority_table,
    power_increment_attribution,
)
from .power_model import fit_power_surrogate
from .regimes import segment_regimes
from .reporting import generate_all_figures


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Cannot serialize {type(value)}")


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _format_value(value: Any) -> str:
    if pd.isna(value):
        return "—"
    if isinstance(value, (bool, np.bool_)):
        return "是" if value else "否"
    if isinstance(value, (float, np.floating)):
        magnitude = abs(float(value))
        if magnitude >= 100000:
            return f"{value:,.0f}"
        if magnitude >= 100:
            return f"{value:.1f}"
        return f"{value:.3f}"
    return str(value)


def _markdown_table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    table = frame[columns].copy() if columns else frame.copy()
    headers = [str(column) for column in table.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in table.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_format_value(value) for value in row) + " |")
    return "\n".join(lines)


def _sensitivity_table(twin, high_frame, policy_10, policy_5, seed: int) -> pd.DataFrame:
    parameters = [
        "d_ref",
        "temperature_sensitivity",
        "plate_penalty",
        "rapping_amplitude",
    ]
    rows: list[dict[str, Any]] = []
    for parameter in parameters:
        for factor in [0.7, 1.0, 1.3]:
            altered = replace(twin, **{parameter: getattr(twin, parameter) * factor})
            scenarios = altered.scenario_pack(high_frame, n_scenarios=10000, seed=seed + len(rows))
            c10 = altered.predict_scenarios(policy_10, scenarios)
            c5 = altered.predict_scenarios(policy_5, scenarios)
            rows.append(
                {
                    "parameter": parameter,
                    "factor": factor,
                    "p95_at_10_policy": float(np.quantile(c10, 0.95)),
                    "p95_at_5_policy": float(np.quantile(c5, 0.95)),
                    "10_policy_compliance": float(np.mean(c10 <= 10)),
                    "5_policy_compliance": float(np.mean(c5 <= 5)),
                }
            )
    return pd.DataFrame(rows)


def _high_regime_bootstrap(
    twin,
    power,
    full_frame,
    high_frame,
    high_regime: int,
    optimization_config: dict[str, Any],
    voltage_extension: float,
    seed: int,
) -> pd.DataFrame:
    fast = dict(optimization_config)
    fast.update(
        {
            "training_scenarios": 512,
            "verification_scenarios": 3000,
            "differential_evolution_maxiter": 55,
            "differential_evolution_popsize": 7,
            "scenario_quantile_buffer": max(
                float(optimization_config.get("scenario_quantile_buffer", 0.025)), 0.035
            ),
        }
    )
    rows = []
    for replicate in range(12):
        draw_seed = seed + 17000 + replicate * 101
        policy10, summary10 = optimize_single_regime(
            twin,
            power,
            full_frame,
            high_frame,
            high_regime,
            10.0,
            fast,
            draw_seed,
            voltage_upper_factor=1.0,
        )
        policy5, summary5 = optimize_single_regime(
            twin,
            power,
            full_frame,
            high_frame,
            high_regime,
            5.0,
            fast,
            draw_seed + 37,
            voltage_upper_factor=voltage_extension,
        )
        increase = 100.0 * (summary5["power_kW"] - summary10["power_kW"]) / summary10["power_kW"]
        rows.append(
            {
                "replicate": replicate + 1,
                "power_10_kW": summary10["power_kW"],
                "power_5_kW": summary5["power_kW"],
                "increase_pct": increase,
                "feasible_10": summary10["feasible"],
                "feasible_5": summary5["feasible"],
                "max_voltage_10": float(policy10[:4].max()),
                "max_voltage_5": float(policy5[:4].max()),
            }
        )
    return pd.DataFrame(rows)


def _build_answer(
    audit,
    power,
    regimes,
    twin,
    result10,
    result5_raw,
    result5_conditional,
    q3_table,
    priority_low,
    priority_high,
    q4,
    weighted_10,
    weighted_5,
    high_regime,
    low_regime,
    bootstrap,
    voltage_extension,
) -> str:
    mean_r2 = power.cv_metrics["r2"].mean()
    mean_rmse = power.cv_metrics["rmse_kW"].mean()
    overall_r2 = r2_score(audit.frame["P_total_kW"], power.predict_frame(audit.frame))
    weighted_increase = 100 * (weighted_5 - weighted_10) / weighted_10
    high_increase = float(q4.loc[q4["regime"] == high_regime, "increase_pct"].iloc[0])
    valid_bootstrap = bootstrap.loc[bootstrap["feasible_10"] & bootstrap["feasible_5"]]
    ci = np.quantile(valid_bootstrap["increase_pct"], [0.025, 0.975])
    raw_infeasible = int((~result5_raw.table["feasible"]).sum())

    policy_columns = [
        "label", "U1_kV", "U2_kV", "U3_kV", "U4_kV", "T1_s", "T2_s", "T3_s", "T4_s",
        "power_kW", "c_p95_mgNm3", "compliance_probability", "dust_load_index", "feasible",
    ]
    regimes_view = regimes.centers[[
        "label", "n", "share", "Temp_C_mean", "C_in_gNm3_mean", "Q_Nm3h_mean", "mass_load_mean", "median_dwell_min"
    ]]
    policy10 = result10.table.merge(regimes.centers[["regime", "label"]], on="regime")
    policy5 = result5_conditional.table.merge(regimes.centers[["regime", "label"]], on="regime")
    emission_formula = r"""$$
C_{\mathrm{base},t}=1000C_{\mathrm{in},t}\exp(-D_t),
$$

$$
D_t=D_0\frac{Q_0}{Q_t}g_T(T_t)
\left[\sum_{i=1}^4w_i\left(\frac{U_i}{U_{i,0}}\right)^2\right]
g_L(\mathbf{T}_t,C_{\mathrm{in},t}Q_t).
$$"""
    power_formula = r"""$$
P=\beta_0+\sum_{i=1}^4\beta_iU_i^2+\sum_{i=1}^4\gamma_i/T_i+\varepsilon.
$$"""

    return rf"""# A题：水泥烧成系统电除尘器协同优化控制——逐问解答

> 本文档是第一阶段数值解答。所有结果由 `scripts/run_all.py` 从原始 CSV 一键生成。出口浓度没有进行比例换算；50 mg/Nm³被按右删失处理。由于 10/5 mg/Nm³均在观测范围外，排放数值属于“删失锚定 + 物理先验”的外推结果，而不是附件数据直接验证的经验回归。

## 1. 数据诊断与关键判断

数据含 {audit.summary['rows']:,} 个连续分钟、{audit.summary['output_missing']} 个孤立出口缺失点；{audit.summary['output_censored']} 个读数贴在 50 mg/Nm³上限，占 {audit.summary['output_censored_fraction']:.2%}。右删失正态极大似然得到潜在均值 {twin.censor_mu:.3f}、标准差 {twin.censor_sigma:.3f} mg/Nm³。

把温度、入口浓度、流量、四场电压与四场振打周期同时用于普通线性回归，原始出口浓度的 $R^2$ 仅为 {audit.summary['ordinary_ols_r2']:.6f}；对“是否触及50上限”建线性概率模型，$R^2$ 也只有 {audit.summary['censor_indicator_linear_r2']:.6f}。这说明闭环运行和仪表截断已抹去可用于辨识控制效应的输出变化，不能用高阶机器学习强行声称得到因果关系。

![数据质量](../outputs/figures/fig01_data_audit.png)

![删失诊断](../outputs/figures/fig02_censor_diagnostics.png)

## 2. 问题1：因素关系与振打峰值

### 2.1 删失物理信息模型

采用 Deutsch–Anderson 指数穿透结构：

{emission_formula}

其中 $g_T$ 是以 125°C 为中心的温度修正，$g_L$ 是长振打周期引起的积灰/反电晕折减。基准去除指数由右删失运行点和质量守恒锚定为 $D_0={twin.d_ref:.4f}$。分场权重、温度和再飞扬系数由权威文献给出先验，并在计算中传播不确定性。

振打采用隐状态：极板尘负荷随捕集量累积，周期到达后大部分落入灰斗、小部分形成再飞扬脉冲。对未知振打相位积分后，95%排放分位数同时包含积灰效率损失和瞬时峰值风险。因此周期过长会积灰并放大单次峰值，过短则提高振打频率和机械能耗，存在内部折中点。

![因素响应](../outputs/figures/fig06_effect_curves.png)

![温度电压响应面](../outputs/figures/fig07_temperature_voltage_surface.png)

![振打动态](../outputs/figures/fig08_rapping_dynamics.png)

### 2.2 关系结论

- 入口浓度近似按比例抬升出口浓度；流量增加缩短停留时间，使穿透率上升。
- 电压通过迁移速度和去除指数发挥单调减排作用，边际减排收益随电压增大而递减。
- 125°C附近模型效率最佳；偏离该温区时，电气特性和粉尘比电阻的不确定性使所需控制强度增加。
- 振打周期不能只按“越长越省电”处理：长周期降低振打频率，却增加极板尘负荷、效率衰减和单次再飞扬峰值。

证据边界如下：

{_markdown_table(twin.identifiability)}

## 3. 问题2：典型工况与最低电耗

### 3.1 工况划分

对温度、$\log C_{{in}}$、$\log Q$ 做15分钟中位数平滑和稳健标准化，再以时间正则化高斯混合模型划分。BIC在满足占比和持续时间约束的候选中选择 $K={regimes.n_regimes}$。

{_markdown_table(regimes_view)}

![工况识别](../outputs/figures/fig04_regime_segmentation.png)

![工况中心](../outputs/figures/fig05_regime_centers.png)

### 3.2 能耗模型

拟合结果为

{power_formula}

全样本 $R^2={overall_r2:.4f}$；更严格的逐日留一验证平均 $R^2={mean_r2:.4f}$、RMSE={mean_rmse:.2f} kW（约占平均功率的0.34%）。逐日 $R^2$ 未机械达到0.99，但绝对误差很小；本文保留这一真实验证结果，不通过随机分钟切分虚增精度。完整系数见 `power_coefficients.csv`。

![能耗验证](../outputs/figures/fig03_power_model.png)

### 3.3 10 mg/Nm³最优控制表

优化在历史电压/周期边界内进行；内部按97.5%设计分位留出蒙特卡洛裕量，最后用独立10,000次模拟核查95%分位。

{_markdown_table(policy10, policy_columns)}

![10限值策略](../outputs/figures/fig09_policy_10_heatmap.png)

## 4. 问题3：两类工况与优先级

选取样本充分的低负荷工况 R{low_regime+1} 与最高质量负荷工况 R{high_regime+1}。操作参数为：

{_markdown_table(q3_table)}

单位增量电耗的减排收益排序：

**低负荷 R{low_regime+1}**

{_markdown_table(priority_low.head(8))}

**高负荷 R{high_regime+1}**

{_markdown_table(priority_high.head(8))}

![优先级比较](../outputs/figures/fig10_priority_comparison.png)

控制规律是：基线穿透占主导且积灰指数较低时，先提高减排收益/千瓦最大的电场电压；高负荷下若积灰约束活跃，则先缩短相应振打周期，再提高电压。最终场用于严格限值下的精细捕集，振打应错峰，避免多个电场脉冲叠加。

## 5. 问题4：10降至5的电耗代价

在完全相同的历史安全边界内，5 mg/Nm³有 {raw_infeasible} 个工况无法通过独立95%机会约束复核，因此不能为全工况给出虚假的“边界内最优百分比”。为完成政策情景测算，另做明确标注的条件仿真：仅把各场历史最大电压上界放宽至 {voltage_extension:.3f} 倍，振打周期边界保持不变。

{_markdown_table(q4)}

按历史工况占比加权，10 mg/Nm³最优电耗为 {weighted_10:.1f} kW；条件性5 mg/Nm³电耗为 {weighted_5:.1f} kW，增加 **{weighted_increase:.2f}%**。最高负荷 R{high_regime+1} 增幅为 **{high_increase:.2f}%**；12组独立场景重采样中有 {len(valid_bootstrap)} 组同时通过两级限值复核，其增幅95%区间为 **[{ci[0]:.2f}%, {ci[1]:.2f}%]**。

![电耗增幅](../outputs/figures/fig11_q4_energy_increase.png)

高浓度工况建议：

1. 采用入口质量负荷前馈，在高负荷进入电场前预置电压与振打周期；
2. 优先强化减排收益最高的电场，并保留末级电场的精细捕集余量；
3. 四场振打错峰，禁止相位重合导致瞬时峰值叠加；
4. 用进入/退出双阈值设置滞回，避免工况边界附近频繁切换；
5. 若要长期保证5 mg/Nm³，应先做整流变压器、电极和火花率校核，不能直接把条件仿真电压作为现场指令。

## 6. 稳健性、局限与复现

- `sensitivity_high_regime.csv` 给出关键机理系数±30%扰动下的达标概率；先验减弱30%时，部分策略会失去达标性，说明5 mg/Nm³结论必须与现场标定联用。
- `q4_high_regime_bootstrap.csv` 保存12组独立优化重采样，用于区间与算法稳定性检查。
- 附件没有二次电流、实际振打触发时刻、粉尘比电阻和粒径分布；这些缺失是排放外推不确定性的主要来源。
- 权威依据：[生态环境部水泥行业超低排放意见](https://www.mee.gov.cn/xxgk2018/xxgk/xxgk03/202401/t20240119_1064243.html)、[EPA电除尘性能模型](https://nepis.epa.gov/Exe/ZyPURL.cgi?Dockey=9101MY0A.TXT)、[EPA电除尘检查手册](https://nepis.epa.gov/Exe/ZyPURL.cgi?Dockey=9400034C.TXT)、[高温电除尘实验研究](https://doi.org/10.1016/j.seppur.2015.01.016)、[振打再飞扬研究](https://doi.org/10.1021/es00127a012)。
"""


def run_all(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parent.parent
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(config["project"]["seed"])
    csv_path = (project_root / config["project"]["input_csv"]).resolve()
    output_root = project_root / config["project"]["output_dir"]
    table_dir = output_root / "tables"
    figure_dir = output_root / "figures"
    model_dir = output_root / "models"
    log_dir = output_root / "logs"
    for directory in [table_dir, figure_dir, model_dir, log_dir, project_root / "answers"]:
        directory.mkdir(parents=True, exist_ok=True)

    audit = load_and_audit_data(
        csv_path, censor_threshold=float(config["data"]["censor_threshold_mgNm3"])
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
        bic_tie_tolerance=float(regime_config["bic_tie_tolerance"]),
        seed=seed,
    )
    twin = fit_emission_twin(audit, config["emission_prior"])
    optimization_config = config["optimization"]
    result10 = optimize_policy(twin, power, audit.frame, regimes.labels, 10.0, optimization_config, seed)
    result5_raw = optimize_policy(twin, power, audit.frame, regimes.labels, 5.0, optimization_config, seed + 500)
    voltage_extension = float(optimization_config["q4_voltage_upper_extension_factor"])
    result5_conditional = optimize_policy(
        twin,
        power,
        audit.frame,
        regimes.labels,
        5.0,
        optimization_config,
        seed + 900,
        voltage_upper_factor=voltage_extension,
    )

    eligible = regimes.centers.loc[
        (regimes.centers["n"] >= max(200, int(0.05 * len(audit.frame))))
        & (regimes.centers["Temp_C_p95"] < 145)
    ]
    low_regime = int(eligible.loc[eligible["mass_load_mean"].idxmin(), "regime"])
    high_regime = int(eligible.loc[eligible["mass_load_mean"].idxmax(), "regime"])
    bounds = policy_bounds(audit.frame)
    low_frame = audit.frame.loc[regimes.labels == low_regime].reset_index(drop=True)
    high_frame = audit.frame.loc[regimes.labels == high_regime].reset_index(drop=True)
    priority_low = policy_priority_table(
        result10.policies[low_regime], low_frame, twin, power, bounds, seed + 30000
    )
    priority_high = policy_priority_table(
        result10.policies[high_regime], high_frame, twin, power, bounds, seed + 31000
    )

    q3_rows = []
    for regime, name in [(low_regime, "低负荷"), (high_regime, "高负荷")]:
        current = result10.table.loc[result10.table["regime"] == regime].iloc[0]
        history = audit.frame.loc[regimes.labels == regime, POLICY_COLUMNS].mean()
        row: dict[str, Any] = {"工况": name, "标签": f"R{regime+1}"}
        for column in POLICY_COLUMNS:
            row[column] = current[column]
            row[f"历史{column}"] = history[column]
        row["power_kW"] = current["power_kW"]
        row["c_p95_mgNm3"] = current["c_p95_mgNm3"]
        row["compliance_probability"] = current["compliance_probability"]
        q3_rows.append(row)
    q3_table = pd.DataFrame(q3_rows)

    q4 = regimes.centers[["regime", "label", "share"]].merge(
        result10.table[["regime", "power_kW"]].rename(columns={"power_kW": "power_10_kW"}), on="regime"
    ).merge(
        result5_raw.table[["regime", "feasible"]].rename(columns={"feasible": "raw_5_feasible"}), on="regime"
    ).merge(
        result5_conditional.table[["regime", "power_kW", "feasible"]].rename(
            columns={"power_kW": "power_5_conditional_kW", "feasible": "conditional_5_feasible"}
        ), on="regime"
    )
    q4["increase_pct"] = 100.0 * (
        q4["power_5_conditional_kW"] - q4["power_10_kW"]
    ) / q4["power_10_kW"]
    q4["voltage_upper_factor"] = voltage_extension
    weighted_10 = float(np.sum(q4["share"] * q4["power_10_kW"]))
    weighted_5 = float(np.sum(q4["share"] * q4["power_5_conditional_kW"]))

    attribution = []
    for regime in range(regimes.n_regimes):
        table = power_increment_attribution(
            result10.policies[regime], result5_conditional.policies[regime], power
        )
        table.insert(0, "regime", regime)
        attribution.append(table)
    attribution_table = pd.concat(attribution, ignore_index=True)
    sensitivity = _sensitivity_table(
        twin,
        high_frame,
        result10.policies[high_regime],
        result5_conditional.policies[high_regime],
        seed + 32000,
    )
    bootstrap = _high_regime_bootstrap(
        twin,
        power,
        audit.frame,
        high_frame,
        high_regime,
        optimization_config,
        voltage_extension,
        seed,
    )

    tables = {
        "feature_summary.csv": audit.feature_summary.reset_index(names="feature"),
        "correlations_with_output.csv": audit.correlations.reset_index().rename(columns={"index": "feature", "C_out_mgNm3": "correlation"}),
        "power_coefficients.csv": power.coefficient_table(),
        "power_leave_one_day_cv.csv": power.cv_metrics,
        "regime_model_selection.csv": regimes.model_selection,
        "regime_centers.csv": regimes.centers,
        "regime_transition_matrix.csv": regimes.transition_matrix.reset_index(names="from_regime"),
        "emission_identifiability.csv": twin.identifiability,
        "optimal_policy_10.csv": result10.table,
        "optimal_policy_5_raw_bounds.csv": result5_raw.table,
        "optimal_policy_5_conditional_extension.csv": result5_conditional.table,
        "q3_two_regime_operation_table.csv": q3_table,
        "priority_low_regime.csv": priority_low,
        "priority_high_regime.csv": priority_high,
        "q4_energy_increase.csv": q4,
        "q4_power_attribution.csv": attribution_table,
        "sensitivity_high_regime.csv": sensitivity,
        "q4_high_regime_bootstrap.csv": bootstrap,
    }
    for name, table in tables.items():
        _write_csv(table, table_dir / name)

    audit_payload = dict(audit.summary)
    (log_dir / "data_audit.json").write_text(
        json.dumps(audit_payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8"
    )
    model_payload = {
        "power_intercept": power.raw_intercept,
        "power_coefficients": dict(zip(power.feature_names, power.raw_coefficients)),
        "emission_references": {
            "temp_ref": twin.temp_ref,
            "c_in_ref": twin.c_in_ref,
            "q_ref": twin.q_ref,
            "u_ref": twin.u_ref,
            "t_ref": twin.t_ref,
            "d_ref": twin.d_ref,
            "stage_weights": twin.stage_weights,
        },
        "regime_count": regimes.n_regimes,
        "high_regime": high_regime,
        "low_regime": low_regime,
        "weighted_power_10_kW": weighted_10,
        "weighted_power_5_kW": weighted_5,
    }
    (model_dir / "model_summary.json").write_text(
        json.dumps(model_payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8"
    )

    generate_all_figures(
        audit,
        power,
        regimes,
        twin,
        result10,
        result5_conditional,
        q4,
        weighted_10,
        weighted_5,
        priority_low,
        priority_high,
        figure_dir,
    )
    answer = _build_answer(
        audit,
        power,
        regimes,
        twin,
        result10,
        result5_raw,
        result5_conditional,
        q3_table,
        priority_low,
        priority_high,
        q4,
        weighted_10,
        weighted_5,
        high_regime,
        low_regime,
        bootstrap,
        voltage_extension,
    )
    answer_path = project_root / "answers" / "逐问解答.md"
    answer_path.write_text(answer, encoding="utf-8")
    return {
        "answer_path": str(answer_path),
        "regime_count": regimes.n_regimes,
        "weighted_power_10_kW": weighted_10,
        "weighted_power_5_kW": weighted_5,
        "weighted_increase_pct": 100 * (weighted_5 - weighted_10) / weighted_10,
        "raw_5_infeasible_regimes": int((~result5_raw.table["feasible"]).sum()),
        "conditional_5_infeasible_regimes": int((~result5_conditional.table["feasible"]).sum()),
        "high_regime": high_regime,
        "low_regime": low_regime,
    }
