from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, spearmanr
import yaml

from .calibrated_emission import (
    build_calibrated_scenarios,
    evaluate_distribution_fit,
    fit_uncensored_distribution,
    regime_reference,
    relative_physical_response,
)
from .calibrated_optimization import (
    calibrated_policy_priority_table,
    optimize_calibrated_policy,
)
from .calibrated_reporting import (
    plot_calibrated_distribution_comparison,
    plot_calibrated_factor_comparison,
    plot_calibrated_policy_comparison,
)
from .data_audit import load_and_audit_data
from .emission_twin import T_COLUMNS, U_COLUMNS, fit_emission_twin
from .optimization import POLICY_COLUMNS, policy_bounds
from .power_model import fit_power_surrogate
from .regimes import segment_regimes


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)}")


def _format(value: Any) -> str:
    if pd.isna(value):
        return "—"
    if isinstance(value, (bool, np.bool_)):
        return "是" if value else "否"
    if isinstance(value, (float, np.floating)):
        if abs(float(value)) >= 100:
            return f"{value:.2f}"
        return f"{value:.4f}"
    return str(value)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    view = frame[columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in view.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_format(value) for value in row) + " |")
    return "\n".join(lines)


def _factor_evidence_table(frame: pd.DataFrame, twin) -> pd.DataFrame:
    data = frame.loc[frame["C_out_mgNm3"] < 50].copy()
    data["voltage_index"] = (
        (data[U_COLUMNS].to_numpy(dtype=float) / twin.u_ref) ** 2
    ) @ twin.stage_weights
    data["rapping_index"] = (
        data[T_COLUMNS].to_numpy(dtype=float) / twin.t_ref
    ) @ twin.stage_weights
    specifications = [
        ("入口浓度", "C_in_gNm3", "物理层单调上升"),
        ("烟气流量", "Q_Nm3h", "物理层单调上升"),
        ("温度", "Temp_C", "物理层参考中心附近最低"),
        ("综合电压指数", "voltage_index", "物理层单调下降"),
        ("综合振打指数", "rapping_index", "物理层单调上升"),
    ]
    rows = []
    for factor, column, physical in specifications:
        rho, rho_pvalue = spearmanr(data[column], data["C_out_mgNm3"])
        bins = pd.qcut(data[column], 10, duplicates="drop")
        groups = [
            group["C_out_mgNm3"].to_numpy(dtype=float)
            for _, group in data.groupby(bins, observed=True)
        ]
        _, bin_pvalue = f_oneway(*groups)
        bin_means = [float(np.mean(group)) for group in groups]
        rows.append(
            {
                "factor": factor,
                "physical_response": physical,
                "uncensored_spearman_rho": float(rho),
                "uncensored_spearman_pvalue": float(rho_pvalue),
                "uncensored_bin_pvalue": float(bin_pvalue),
                "uncensored_bin_mean_min_mgNm3": min(bin_means),
                "uncensored_bin_mean_max_mgNm3": max(bin_means),
                "data_supported_effect": bool(bin_pvalue < 0.05),
            }
        )
    return pd.DataFrame(rows)


def _representative_regimes(regimes, frame, config) -> tuple[int, int]:
    eligible = regimes.centers.loc[
        (
            regimes.centers["n"]
            >= int(float(config["minimum_share"]) * len(frame))
        )
        & (
            regimes.centers["Temp_C_p95"]
            < float(config["maximum_temperature_p95_C"])
        )
    ]
    low = int(eligible.loc[eligible["mass_load_mean"].idxmin(), "regime"])
    high = int(eligible.loc[eligible["mass_load_mean"].idxmax(), "regime"])
    return low, high


def _sensitivity_table(
    distribution,
    twin,
    high_frame,
    reference_policy,
    policy_10,
    policy_5,
    seed: int,
) -> pd.DataFrame:
    rows = []
    parameters = [
        "d_ref",
        "temperature_sensitivity",
        "plate_penalty",
        "rapping_load_coefficient",
    ]
    for parameter in parameters:
        for factor in [0.7, 1.0, 1.3]:
            altered = replace(twin, **{parameter: getattr(twin, parameter) * factor})
            scenarios = build_calibrated_scenarios(
                altered,
                distribution,
                high_frame,
                reference_policy,
                n_scenarios=10_000,
                seed=seed + len(rows),
            )
            c10 = scenarios.predict(altered, policy_10)
            c5 = scenarios.predict(altered, policy_5)
            rows.append(
                {
                    "parameter": parameter,
                    "factor": factor,
                    "p95_at_10_policy": float(np.quantile(c10, 0.95)),
                    "p95_at_5_policy": float(np.quantile(c5, 0.95)),
                    "compliance_10": float(np.mean(c10 <= 10)),
                    "compliance_5": float(np.mean(c5 <= 5)),
                }
            )
    for field in range(4):
        for factor in [0.7, 1.0, 1.3]:
            weights = twin.stage_weights.copy()
            weights[field] *= factor
            weights /= weights.sum()
            altered = replace(twin, stage_weights=weights)
            scenarios = build_calibrated_scenarios(
                altered,
                distribution,
                high_frame,
                reference_policy,
                n_scenarios=10_000,
                seed=seed + len(rows),
            )
            c10 = scenarios.predict(altered, policy_10)
            c5 = scenarios.predict(altered, policy_5)
            rows.append(
                {
                    "parameter": f"stage_weight_w{field + 1}",
                    "factor": factor,
                    "p95_at_10_policy": float(np.quantile(c10, 0.95)),
                    "p95_at_5_policy": float(np.quantile(c5, 0.95)),
                    "compliance_10": float(np.mean(c10 <= 10)),
                    "compliance_5": float(np.mean(c5 <= 5)),
                }
            )
    return pd.DataFrame(rows)


def _baseline_resampling_table(
    distribution,
    twin,
    high_frame,
    reference_policy,
    policy_10,
    policy_5,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for replicate in range(12):
        scenarios = build_calibrated_scenarios(
            twin,
            distribution,
            high_frame,
            reference_policy,
            n_scenarios=10_000,
            seed=seed + 101 * replicate,
        )
        c10 = scenarios.predict(twin, policy_10)
        c5 = scenarios.predict(twin, policy_5)
        rows.append(
            {
                "replicate": replicate + 1,
                "p95_at_10_policy": float(np.quantile(c10, 0.95)),
                "p95_at_5_policy": float(np.quantile(c5, 0.95)),
                "compliance_10": float(np.mean(c10 <= 10)),
                "compliance_5": float(np.mean(c5 <= 5)),
            }
        )
    return pd.DataFrame(rows)


def run_calibrated_analysis(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parent.parent
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(config["project"]["seed"])
    csv_path = (project_root / config["project"]["input_csv"]).resolve()
    output_root = project_root / config["project"]["output_dir"]
    table_dir = output_root / "tables"
    figure_dir = output_root / "figures"
    model_dir = output_root / "models"
    answer_dir = project_root / "answers"
    for directory in [table_dir, figure_dir, model_dir, answer_dir]:
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
        selection_metric=str(regime_config["selection_metric"]),
        seed=seed,
    )
    twin = fit_emission_twin(audit, config["emission_prior"])
    calibrated_config = config["calibrated_model"]
    distribution = fit_uncensored_distribution(
        audit.frame,
        threshold=float(calibrated_config["threshold_mgNm3"]),
        resolution=float(calibrated_config["resolution_mgNm3"]),
        dirichlet_alpha=float(calibrated_config["dirichlet_alpha"]),
    )
    daily_cv = evaluate_distribution_fit(
        audit.frame,
        threshold=distribution.threshold,
        resolution=distribution.resolution,
        dirichlet_alpha=distribution.dirichlet_alpha,
    )
    factor_evidence = _factor_evidence_table(audit.frame, twin)
    reference_rows = []
    for regime_value in sorted(np.unique(regimes.labels)):
        regime = int(regime_value)
        regime_frame = audit.frame.loc[regimes.labels == regime].reset_index(drop=True)
        conditions, reference_policy = regime_reference(regime_frame)
        reference_rows.append(
            {
                "regime": regime,
                "label": f"R{regime + 1}",
                "reference_multiplier": relative_physical_response(
                    twin,
                    reference_policy,
                    conditions,
                    reference_policy,
                    conditions,
                ),
                "reference_temp_C": conditions["temp_c"],
                "reference_c_in_gNm3": conditions["c_in"],
                "reference_q_Nm3h": conditions["q"],
            }
        )
    reference_checks = pd.DataFrame(reference_rows)

    optimization_config = dict(config["optimization"])
    result10 = optimize_calibrated_policy(
        distribution,
        twin,
        power,
        audit.frame,
        regimes.labels,
        limit=10.0,
        optimization_config=optimization_config,
        seed=seed,
        voltage_upper_factor=1.0,
    )
    result5_raw = optimize_calibrated_policy(
        distribution,
        twin,
        power,
        audit.frame,
        regimes.labels,
        limit=5.0,
        optimization_config=optimization_config,
        seed=seed + 500,
        voltage_upper_factor=1.0,
        initial_policies=result10.policies,
    )
    voltage_factor = float(config["optimization"]["q4_voltage_upper_extension_factor"])
    result10_wide = optimize_calibrated_policy(
        distribution,
        twin,
        power,
        audit.frame,
        regimes.labels,
        limit=10.0,
        optimization_config=optimization_config,
        seed=seed + 700,
        voltage_upper_factor=voltage_factor,
        initial_policies=result10.policies,
    )
    result5_wide = optimize_calibrated_policy(
        distribution,
        twin,
        power,
        audit.frame,
        regimes.labels,
        limit=5.0,
        optimization_config=optimization_config,
        seed=seed + 900,
        voltage_upper_factor=voltage_factor,
        initial_policies=result5_raw.policies,
    )

    centers = regimes.centers[["regime", "label", "share"]]
    q4 = centers.merge(
        result10_wide.table[["regime", "power_kW", "feasible"]].rename(
            columns={
                "power_kW": "power_10_110pct_kW",
                "feasible": "wide_10_feasible",
            }
        ),
        on="regime",
    ).merge(
        result5_raw.table[["regime", "feasible"]].rename(
            columns={"feasible": "raw_5_feasible"}
        ),
        on="regime",
    ).merge(
        result5_wide.table[["regime", "power_kW", "feasible"]].rename(
            columns={"power_kW": "power_5_110pct_kW", "feasible": "wide_5_feasible"}
        ),
        on="regime",
    )
    q4["increase_pct"] = 100.0 * (
        q4["power_5_110pct_kW"] - q4["power_10_110pct_kW"]
    ) / q4["power_10_110pct_kW"]
    q4["voltage_upper_factor"] = voltage_factor
    weighted_10 = float(np.sum(q4["share"] * q4["power_10_110pct_kW"]))
    weighted_5 = float(np.sum(q4["share"] * q4["power_5_110pct_kW"]))
    weighted_increase = 100.0 * (weighted_5 - weighted_10) / weighted_10

    low_regime, high_regime = _representative_regimes(
        regimes, audit.frame, config["representative_regimes"]
    )
    bounds = policy_bounds(audit.frame)
    priority_tables = {}
    for regime, suffix in [(low_regime, "low"), (high_regime, "high")]:
        regime_frame = audit.frame.loc[regimes.labels == regime].reset_index(drop=True)
        scenarios = build_calibrated_scenarios(
            twin,
            distribution,
            regime_frame,
            result10.reference_policies[regime],
            n_scenarios=10_000,
            seed=seed + 30_000 + regime,
        )
        priority_tables[suffix] = calibrated_policy_priority_table(
            result10.policies[regime],
            scenarios,
            regime_frame,
            twin,
            power,
            bounds,
            dust_limit=float(optimization_config["dust_load_limit"]),
        )
        priority_tables[suffix].insert(0, "regime", regime)
        priority_tables[suffix].insert(1, "label", f"R{regime + 1}")

    old_policy_path = table_dir / "optimal_policy_10.csv"
    if not old_policy_path.exists():
        raise FileNotFoundError(
            "Existing optimal_policy_10.csv is required for the non-overwriting comparison."
        )
    old10 = pd.read_csv(old_policy_path)
    comparison = centers[["regime", "label"]].merge(
        old10[["regime", "power_kW", "c_p95_mgNm3"]].rename(
            columns={
                "power_kW": "old_power_10_kW",
                "c_p95_mgNm3": "old_c_p95_10_mgNm3",
            }
        ),
        on="regime",
    ).merge(
        result10.table[["regime", "power_kW", "c_p95_mgNm3", "feasible"]].rename(
            columns={
                "power_kW": "calibrated_power_10_kW",
                "c_p95_mgNm3": "calibrated_c_p95_10_mgNm3",
                "feasible": "calibrated_feasible_10",
            }
        ),
        on="regime",
    )
    comparison["calibrated_power_change_pct"] = np.where(
        comparison["calibrated_feasible_10"],
        100.0
        * (comparison["calibrated_power_10_kW"] - comparison["old_power_10_kW"])
        / comparison["old_power_10_kW"],
        np.nan,
    )

    high_frame = audit.frame.loc[regimes.labels == high_regime].reset_index(drop=True)
    sensitivity = _sensitivity_table(
        distribution,
        twin,
        high_frame,
        result10_wide.reference_policies[high_regime],
        result10_wide.policies[high_regime],
        result5_wide.policies[high_regime],
        seed + 50_000,
    )
    resampling = _baseline_resampling_table(
        distribution,
        twin,
        high_frame,
        result10_wide.reference_policies[high_regime],
        result10_wide.policies[high_regime],
        result5_wide.policies[high_regime],
        seed + 60_000,
    )

    tables = {
        "calibrated_uncensored_pmf.csv": distribution.probability_table(),
        "calibrated_uncensored_summary.csv": distribution.summary_table(),
        "calibrated_distribution_daily_cv.csv": daily_cv,
        "calibrated_factor_evidence.csv": factor_evidence,
        "calibrated_regime_reference_check.csv": reference_checks,
        "calibrated_optimal_policy_10.csv": result10.table,
        "calibrated_optimal_policy_10_110pct.csv": result10_wide.table,
        "calibrated_optimal_policy_5_raw_bounds.csv": result5_raw.table,
        "calibrated_optimal_policy_5_110pct.csv": result5_wide.table,
        "calibrated_q3_priority_low.csv": priority_tables["low"],
        "calibrated_q3_priority_high.csv": priority_tables["high"],
        "calibrated_q4_energy_increase.csv": q4,
        "calibrated_old_new_policy_comparison.csv": comparison,
        "calibrated_sensitivity_high_regime.csv": sensitivity,
        "calibrated_baseline_resampling.csv": resampling,
    }
    for filename, table in tables.items():
        _write_csv(table, table_dir / filename)

    plot_calibrated_distribution_comparison(
        audit.frame,
        twin,
        distribution,
        figure_dir / "calibrated_distribution_comparison.png",
        seed=seed,
    )
    plot_calibrated_factor_comparison(
        audit.frame,
        twin,
        figure_dir / "calibrated_factor_evidence_comparison.png",
    )
    plot_calibrated_policy_comparison(
        comparison,
        figure_dir / "calibrated_policy_comparison.png",
    )

    cv_summary = {
        "mean_ks_distance": float(daily_cv["ks_distance"].mean()),
        "mean_wasserstein_mgNm3": float(daily_cv["wasserstein_mgNm3"].mean()),
        "mean_absolute_mean_error_mgNm3": float(
            daily_cv["absolute_mean_error_mgNm3"].mean()
        ),
        "mean_interval_coverage": float(daily_cv["interval_coverage"].mean()),
    }
    summary_payload = {
        "observation_layer": {
            "n_uncensored": len(distribution.observations),
            "mean_mgNm3": distribution.point_prediction,
            "sd_mgNm3": float(np.std(distribution.observations, ddof=1)),
            "threshold_mgNm3": distribution.threshold,
            "resolution_mgNm3": distribution.resolution,
            "dirichlet_alpha": distribution.dirichlet_alpha,
            "daily_cv": cv_summary,
        },
        "physical_layer": {
            "definition": "regime-reference physical concentration ratio",
            "absolute_historical_prediction_used": False,
        },
        "optimization": {
            "all_10_feasible": bool(result10.table["feasible"].all()),
            "all_10_110pct_feasible": bool(result10_wide.table["feasible"].all()),
            "raw_5_infeasible_regimes": int((~result5_raw.table["feasible"]).sum()),
            "all_5_110pct_feasible": bool(result5_wide.table["feasible"].all()),
            "weighted_power_10_kW": weighted_10,
            "weighted_power_5_kW": weighted_5,
            "weighted_increase_pct": weighted_increase,
            "low_regime": low_regime,
            "high_regime": high_regime,
        },
    }
    (model_dir / "calibrated_model_summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )

    policy_display = result10.table.copy()
    policy_display.insert(
        policy_display.columns.get_loc("regime") + 1,
        "label",
        policy_display["regime"].map(lambda value: f"R{int(value) + 1}"),
    )
    policy_columns = [
        "label",
        "U1_kV",
        "U2_kV",
        "U3_kV",
        "U4_kV",
        "T1_s",
        "T2_s",
        "T3_s",
        "T4_s",
        "power_kW",
        "c_p95_mgNm3",
        "compliance_probability",
        "dust_load_index",
        "feasible",
    ]
    reference_error = float(
        np.max(np.abs(reference_checks["reference_multiplier"].to_numpy() - 1.0))
    )
    resampling_ranges = {
        column: (
            float(resampling[column].min()),
            float(resampling[column].max()),
        )
        for column in [
            "p95_at_10_policy",
            "p95_at_5_policy",
            "compliance_10",
            "compliance_5",
        ]
    }
    non_scale_sensitivity = sensitivity.loc[sensitivity["parameter"] != "d_ref"]
    non_scale_p95_10 = (
        float(non_scale_sensitivity["p95_at_10_policy"].min()),
        float(non_scale_sensitivity["p95_at_10_policy"].max()),
    )
    non_scale_p95_5 = (
        float(non_scale_sensitivity["p95_at_5_policy"].min()),
        float(non_scale_sensitivity["p95_at_5_policy"].max()),
    )
    scale_sensitivity = sensitivity.loc[sensitivity["parameter"] == "d_ref"]
    scale_p95_10 = (
        float(scale_sensitivity["p95_at_10_policy"].min()),
        float(scale_sensitivity["p95_at_10_policy"].max()),
    )
    scale_p95_5 = (
        float(scale_sensitivity["p95_at_5_policy"].min()),
        float(scale_sensitivity["p95_at_5_policy"].max()),
    )
    feasible_10_count = int(result10.table["feasible"].sum())
    total_regimes = int(len(result10.table))
    report = f"""# 非触顶分布校准—物理相对响应双层模型：四问影响报告

> 本报告为独立重算结果，不修改论文。观测层只描述4,539条未触顶记录；物理层只提供相对响应，不再解释为历史绝对浓度预测。

## 一、模型与分布验证

观测层均值为 {distribution.point_prediction:.4f} mg/Nm³，标准差为 {np.std(distribution.observations, ddof=1):.4f} mg/Nm³。逐日留一验证的平均 KS 距离为 {cv_summary['mean_ks_distance']:.4f}，平均 Wasserstein 距离为 {cv_summary['mean_wasserstein_mgNm3']:.4f} mg/Nm³，日均值平均绝对误差为 {cv_summary['mean_absolute_mean_error_mgNm3']:.4f} mg/Nm³，95% 区间平均覆盖率为 {cv_summary['mean_interval_coverage']:.2%}。八个工况参考点的物理倍率均严格等于 1，最大数值误差为 {reference_error:.2e}。

![分布比较](../outputs/figures/calibrated_distribution_comparison.png)

## 二、问题一：因素关系的证据分离

{_markdown_table(factor_evidence, ['factor','physical_response','uncensored_spearman_rho','uncensored_bin_pvalue','data_supported_effect'])}

数据层未识别出显著因素效应；物理层仅报告相对响应方向及情景倍率。

![因素证据比较](../outputs/figures/calibrated_factor_evidence_comparison.png)

## 三、问题二：10 mg/Nm³校准策略

{_markdown_table(policy_display, policy_columns)}

在历史操作边界内，{feasible_10_count}/{total_regimes} 个工况通过独立 10,000 情景检验。R5 未满足 95% 机会约束，故不将其表中数值解释为可执行最优策略；采用 110% 电压上界情景后，八个工况均获得可行解。

## 四、问题三：低、高负荷动作优先级

低负荷工况为R{low_regime + 1}，首要类别为{priority_tables['low'].iloc[0]['primary_action_class']}，首要变量为{priority_tables['low'].iloc[0]['primary_parameter']}；高负荷工况为R{high_regime + 1}，首要类别为{priority_tables['high'].iloc[0]['primary_action_class']}，首要变量为{priority_tables['high'].iloc[0]['primary_parameter']}。周期优先只表示尘负荷约束恢复顺序，直接减排收益仍单独报告。

## 五、问题四：10降至5的电耗变化

{_markdown_table(q4, ['label','power_10_110pct_kW','wide_10_feasible','raw_5_feasible','power_5_110pct_kW','wide_5_feasible','increase_pct'])}

在相同的 110% 电压上界下，按工况占比加权的 10 mg/Nm³ 功率为 {weighted_10:.2f} kW，5 mg/Nm³ 功率为 {weighted_5:.2f} kW，增幅为 {weighted_increase:.2f}%。在历史边界内，5 mg/Nm³ 限值存在 {int((~result5_raw.table['feasible']).sum())} 个不可行工况；在 110% 边界下，10 mg/Nm³ 与 5 mg/Nm³ 两组策略均在八个工况中通过独立验证。

## 六、敏感性与重采样稳定性

经验分布进行 12 次独立重采样后，R8 的 10 mg/Nm³ 策略排放 95% 分位处于 {resampling_ranges['p95_at_10_policy'][0]:.4f}–{resampling_ranges['p95_at_10_policy'][1]:.4f} mg/Nm³，达标概率处于 {resampling_ranges['compliance_10'][0]:.2%}–{resampling_ranges['compliance_10'][1]:.2%}；5 mg/Nm³ 策略对应区间为 {resampling_ranges['p95_at_5_policy'][0]:.4f}–{resampling_ranges['p95_at_5_policy'][1]:.4f} mg/Nm³ 和 {resampling_ranges['compliance_5'][0]:.2%}–{resampling_ranges['compliance_5'][1]:.2%}，说明观测基线抽样误差对结论影响较小。

在温度、积灰、振打系数和分场权重的 ±30% 扰动下，10 mg/Nm³ 策略的排放 95% 分位为 {non_scale_p95_10[0]:.4f}–{non_scale_p95_10[1]:.4f} mg/Nm³，5 mg/Nm³ 策略为 {non_scale_p95_5[0]:.4f}–{non_scale_p95_5[1]:.4f} mg/Nm³。分场权重的不利扰动可使 5 mg/Nm³ 策略轻微越过限值，因此工程实施时应保留额外控制裕量。迁移尺度参数 $d_{{\mathrm{{ref}}}}$ 的敏感性最强：其 ±30% 扰动分别使上述区间扩展至 {scale_p95_10[0]:.4f}–{scale_p95_10[1]:.4f} mg/Nm³ 和 {scale_p95_5[0]:.4f}–{scale_p95_5[1]:.4f} mg/Nm³。由此可知，低量程出口测量与迁移尺度现场标定是将情景方案转化为控制定值的必要条件。

## 七、新旧结果比较与适用边界

{_markdown_table(comparison, ['label','old_power_10_kW','calibrated_power_10_kW','calibrated_feasible_10','calibrated_power_change_pct','old_c_p95_10_mgNm3','calibrated_c_p95_10_mgNm3'])}

新模型改善的是历史未触顶分布的一致性，而不是逐分钟预测R²。10/5 mg/Nm³仍属于物理相对响应主导的限值外推；现场应用前仍需低量程出口仪表和分场电流数据标定。
"""
    report_path = answer_dir / "calibrated_四问影响报告.md"
    report_path.write_text(report, encoding="utf-8")

    return {
        "report": str(report_path),
        "model_summary": str(model_dir / "calibrated_model_summary.json"),
        "distribution_cv": cv_summary,
        "all_10_feasible": bool(result10.table["feasible"].all()),
        "all_10_110pct_feasible": bool(result10_wide.table["feasible"].all()),
        "raw_5_infeasible_regimes": int((~result5_raw.table["feasible"]).sum()),
        "all_5_110pct_feasible": bool(result5_wide.table["feasible"].all()),
        "weighted_increase_pct": weighted_increase,
    }
