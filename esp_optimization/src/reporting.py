from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

from .data_audit import AuditResult
from .emission_twin import EmissionTwin
from .optimization import OptimizationResult, POLICY_COLUMNS
from .power_model import PowerSurrogate
from .regimes import RegimeModel


COLORS = ["#145DA0", "#2E8B57", "#F39C12", "#C0392B", "#7D3C98", "#008C95", "#6C757D", "#D35400"]


def setup_plot_style() -> None:
    font_candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    ]
    registered_fonts: list[str] = []
    for font_path in font_candidates:
        if font_path.exists():
            fm.fontManager.addfont(str(font_path))
            registered_fonts.append(fm.FontProperties(fname=str(font_path)).get_name())
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "font.sans-serif": registered_fonts
            + ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 130,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "axes.titleweight": "bold",
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def plot_data_audit(audit: AuditResult, path: Path) -> None:
    frame = audit.frame
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
    axes[0, 0].plot(frame["timestamp"], frame["Temp_C"], lw=0.7, label="温度/°C")
    axes[0, 0].set_title("7天入口温度")
    axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    twin = axes[0, 0].twinx()
    twin.plot(frame["timestamp"], frame["C_in_gNm3"], lw=0.6, color=COLORS[3], alpha=0.7)
    twin.set_ylabel("入口浓度 g/Nm³", color=COLORS[3])

    axes[0, 1].plot(frame["timestamp"], frame["Q_Nm3h"] / 1000.0, lw=0.7, color=COLORS[1])
    axes[0, 1].set_title("烟气流量")
    axes[0, 1].set_ylabel("千 Nm³/h")
    axes[0, 1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    axes[1, 0].plot(frame["timestamp"], frame["C_out_mgNm3"], lw=0.55, color=COLORS[3])
    axes[1, 0].axhline(50, ls="--", color="black", lw=1, label="右删失上限 50")
    missing = frame["C_out_mgNm3"].isna()
    axes[1, 0].scatter(frame.loc[missing, "timestamp"], np.full(missing.sum(), 48.65), marker="x", s=15, label="缺失")
    axes[1, 0].set_title("出口浓度：大量读数贴于仪表上限")
    axes[1, 0].set_ylabel("mg/Nm³")
    axes[1, 0].legend(loc="lower right")
    axes[1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    uncensored = frame.loc[frame["C_out_mgNm3"] < 50, "C_out_mgNm3"]
    axes[1, 1].hist(uncensored, bins=35, color=COLORS[0], alpha=0.8, label="未删失读数")
    axes[1, 1].bar([50], [audit.summary["output_censored"]], width=0.03, color=COLORS[3], label="删失读数")
    axes[1, 1].set_title(f"右删失比例 {audit.summary['output_censored_fraction']:.2%}")
    axes[1, 1].set_xlabel("C_out (mg/Nm³)")
    axes[1, 1].set_ylabel("频数")
    axes[1, 1].legend()
    fig.suptitle("数据质量与右删失证据", fontsize=16)
    fig.tight_layout()
    _save(fig, path)


def plot_censor_diagnostics(audit: AuditResult, path: Path) -> None:
    correlation = audit.correlations.drop("C_out_mgNm3").sort_values(key=np.abs)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    colors = [COLORS[3] if value < 0 else COLORS[0] for value in correlation]
    axes[0].barh(correlation.index, correlation.values, color=colors)
    axes[0].axvline(0, color="black", lw=0.8)
    axes[0].set_title("原始变量与出口浓度相关系数")
    axes[0].set_xlabel("Pearson r")
    fit = audit.summary["censored_normal"]
    values = audit.frame["C_out_mgNm3"].dropna().to_numpy()
    uncensored = values[values < 50]
    x = np.linspace(48.7, 50.5, 300)
    from scipy.stats import norm

    axes[1].hist(uncensored, bins=35, density=True, alpha=0.55, color=COLORS[0], label="未删失样本")
    axes[1].plot(x, norm.pdf(x, fit["mu"], fit["sigma"]), color=COLORS[3], lw=2, label="删失正态MLE")
    axes[1].axvline(50, color="black", ls="--", label="删失点")
    axes[1].set_title(f"潜在均值={fit['mu']:.3f}, σ={fit['sigma']:.3f}")
    axes[1].legend()
    fig.suptitle("删失分布与低可识别性诊断", fontsize=15)
    fig.tight_layout()
    _save(fig, path)


def plot_power_fit(frame: pd.DataFrame, model: PowerSurrogate, path: Path) -> None:
    prediction = model.predict_frame(frame)
    residual = frame["P_total_kW"].to_numpy() - prediction
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    axes[0].scatter(frame["P_total_kW"], prediction, s=4, alpha=0.25, color=COLORS[0])
    lo = min(frame["P_total_kW"].min(), prediction.min())
    hi = max(frame["P_total_kW"].max(), prediction.max())
    axes[0].plot([lo, hi], [lo, hi], color=COLORS[3], lw=1.5)
    axes[0].set(xlabel="实测功率/kW", ylabel="预测功率/kW", title="预测—实测")
    axes[1].plot(frame["timestamp"], residual, lw=0.5, color=COLORS[1])
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_title("功率残差时序")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    cv = model.cv_metrics
    axes[2].bar(cv["held_out_day"].str[-5:], cv["r2"], color=COLORS[0])
    axes[2].set_ylim(max(0.95, cv["r2"].min() - 0.01), 1.0)
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].set_title(f"逐日留一验证，平均R²={cv['r2'].mean():.4f}")
    fig.suptitle("能耗代理模型验证", fontsize=15)
    fig.tight_layout()
    _save(fig, path)


def plot_regimes(frame: pd.DataFrame, regimes: RegimeModel, path: Path) -> None:
    pca = PCA(n_components=2).fit_transform(regimes.scaled_features)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [2, 1]})
    step = max(1, len(frame) // 5000)
    for regime in range(regimes.n_regimes):
        mask = regimes.labels[::step] == regime
        axes[0].scatter(pca[::step][mask, 0], pca[::step][mask, 1], s=7, alpha=0.55, label=f"R{regime + 1}", color=COLORS[regime % len(COLORS)])
    axes[0].set(title="工况在稳健特征空间中的分布", xlabel="PC1", ylabel="PC2")
    axes[0].legend(ncol=min(8, regimes.n_regimes), loc="upper center", fontsize=8)
    axes[1].step(frame["timestamp"], regimes.labels + 1, where="post", lw=0.8, color=COLORS[0])
    axes[1].set_yticks(range(1, regimes.n_regimes + 1))
    axes[1].set_yticklabels([f"R{i}" for i in range(1, regimes.n_regimes + 1)])
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    axes[1].set_title("时间正则化后的工况序列")
    fig.suptitle(f"典型工况识别（轮廓系数选择K={regimes.n_regimes}）", fontsize=15)
    fig.tight_layout()
    _save(fig, path)


def plot_regime_centers(regimes: RegimeModel, path: Path) -> None:
    columns = ["Temp_C_mean", "C_in_gNm3_mean", "Q_Nm3h_mean", "mass_load_mean", "P_history_mean_kW"]
    values = regimes.centers[columns].to_numpy(dtype=float)
    z = (values - values.mean(axis=0)) / values.std(axis=0)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.heatmap(
        z,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        xticklabels=["温度", "入口浓度", "流量", "质量负荷", "历史功率"],
        yticklabels=regimes.centers["label"],
        ax=ax,
    )
    ax.set_title("典型工况中心（列内标准化）")
    _save(fig, path)


def plot_effect_curves(twin: EmissionTwin, path: Path) -> None:
    reference_policy = np.r_[twin.u_ref, twin.t_ref]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    cin = np.linspace(18, 72, 100)
    axes[0, 0].plot(cin, twin.predict_deterministic(reference_policy, np.full_like(cin, twin.temp_ref), cin, np.full_like(cin, twin.q_ref)), color=COLORS[0])
    axes[0, 0].set(title="入口浓度效应", xlabel="C_in (g/Nm³)", ylabel="预测C_out")
    flow = np.linspace(4.0e5, 5.15e5, 100)
    axes[0, 1].plot(flow / 1000, twin.predict_deterministic(reference_policy, np.full_like(flow, twin.temp_ref), np.full_like(flow, twin.c_in_ref), flow), color=COLORS[1])
    axes[0, 1].set(title="流量/停留时间效应", xlabel="千 Nm³/h", ylabel="预测C_out")
    multipliers = np.linspace(0.8, 1.2, 100)
    for i in range(4):
        values = []
        for multiplier in multipliers:
            policy = reference_policy.copy(); policy[i] = twin.u_ref[i] * multiplier
            values.append(twin.predict_deterministic(policy, twin.temp_ref, twin.c_in_ref, twin.q_ref)[0])
        axes[1, 0].plot(multipliers, values, label=f"U{i+1}", color=COLORS[i])
    axes[1, 0].set(title="分场电压单调效应", xlabel="相对基准电压", ylabel="预测C_out")
    axes[1, 0].legend()
    for i in range(4):
        values = []
        for multiplier in multipliers:
            policy = reference_policy.copy(); policy[4+i] = twin.t_ref[i] * multiplier
            values.append(twin.predict_deterministic(policy, twin.temp_ref, twin.c_in_ref, twin.q_ref)[0])
        axes[1, 1].plot(multipliers, values, label=f"T{i+1}", color=COLORS[i])
    axes[1, 1].set(title="振打周期的积灰—再飞扬折中", xlabel="相对基准周期", ylabel="预测C_out")
    axes[1, 1].legend()
    fig.suptitle("物理信息模型的单因素响应", fontsize=15)
    fig.tight_layout()
    _save(fig, path)


def plot_temperature_voltage_surface(twin: EmissionTwin, path: Path) -> None:
    temp = np.linspace(110, 160, 80)
    voltage_scale = np.linspace(0.85, 1.22, 80)
    surface = np.empty((len(temp), len(voltage_scale)))
    for i, value in enumerate(temp):
        for j, scale in enumerate(voltage_scale):
            policy = np.r_[twin.u_ref * scale, twin.t_ref]
            surface[i, j] = twin.predict_deterministic(policy, value, twin.c_in_ref, twin.q_ref)[0]
    fig, ax = plt.subplots(figsize=(9, 5.8))
    contour = ax.contourf(voltage_scale, temp, np.log10(surface), levels=25, cmap="viridis_r")
    cbar = fig.colorbar(contour, ax=ax); cbar.set_label("log10(C_out)")
    ax.contour(voltage_scale, temp, surface, levels=[5, 10, 50], colors=["white", "yellow", "red"], linewidths=1.2)
    ax.set(title="温度—电压协同响应面", xlabel="四电场电压统一倍率", ylabel="入口温度/°C")
    _save(fig, path)


def plot_rapping_dynamics(twin: EmissionTwin, path: Path) -> None:
    minutes = np.arange(0, 70)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)
    for period, color in zip([180, 233, 290], [COLORS[1], COLORS[0], COLORS[3]]):
        phase = 0.0; loads = []; peaks = []
        for _ in minutes:
            phase += 60.0 / period
            peak = 0.0
            if phase >= 1.0:
                phase -= 1.0; peak = period / 233.0
            loads.append(phase); peaks.append(peak)
        axes[0].plot(minutes, loads, color=color, label=f"T={period}s")
        axes[1].stem(minutes, peaks, linefmt=color, markerfmt=" ", basefmt=" ", label=f"T={period}s")
    axes[0].set(title="归一化振打相位", ylabel="周期内积灰进度"); axes[0].legend()
    axes[1].set(title="一阶质量守恒下的相对脉冲示意", xlabel="时间/min", ylabel="相对峰值")
    fig.suptitle("振打周期对积灰与瞬时峰值的归一化示意", fontsize=15)
    fig.tight_layout()
    _save(fig, path)


def plot_policy_heatmap(result: OptimizationResult, path: Path, title: str) -> None:
    data = result.table[POLICY_COLUMNS].copy()
    normalized = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-12)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.heatmap(normalized, annot=data.round(1), fmt="", cmap="YlGnBu", yticklabels=[f"R{i+1}" for i in result.table["regime"]], ax=ax)
    ax.set_title(title + "（颜色为列内归一化，标注为实际值）")
    _save(fig, path)


def plot_q4_energy(q4: pd.DataFrame, weighted_10: float, weighted_5: float, path: Path) -> None:
    x = np.arange(len(q4)); width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].bar(x - width/2, q4["power_10_kW"], width, label="10 mg/Nm³", color=COLORS[0])
    axes[0].bar(x + width/2, q4["power_5_conditional_kW"], width, label="5 mg/Nm³", color=COLORS[3])
    axes[0].set_xticks(x, q4["label"]); axes[0].set_ylabel("kW"); axes[0].set_title("各工况最优电耗"); axes[0].legend()
    axes[1].bar(q4["label"], q4["increase_pct"], color=[COLORS[i % len(COLORS)] for i in range(len(q4))])
    weighted_pct = 100 * (weighted_5 - weighted_10) / weighted_10
    axes[1].axhline(weighted_pct, color="black", ls="--", label=f"加权增幅 {weighted_pct:.1f}%")
    axes[1].set_ylabel("增幅/%"); axes[1].set_title("限值收紧的电耗代价"); axes[1].legend()
    fig.suptitle("10→5 mg/Nm³ 的条件性电耗增幅", fontsize=15)
    fig.tight_layout(); _save(fig, path)


def plot_priority(priority_low: pd.DataFrame, priority_high: pd.DataFrame, path: Path) -> None:
    merged = priority_low[["parameter", "reduction_per_added_kW"]].merge(
        priority_high[["parameter", "reduction_per_added_kW"]], on="parameter", suffixes=("_low", "_high")
    ).set_index("parameter")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    merged.plot(kind="bar", ax=ax, color=[COLORS[1], COLORS[3]])
    ax.set(title="低负荷/高负荷工况的单位增量电耗减排收益", ylabel="mg/Nm³ per kW", xlabel="操作变量")
    ax.legend(["低负荷", "高负荷"]); ax.tick_params(axis="x", rotation=0)
    _save(fig, path)


def generate_all_figures(
    audit: AuditResult,
    power: PowerSurrogate,
    regimes: RegimeModel,
    twin: EmissionTwin,
    result_10: OptimizationResult,
    result_5_conditional: OptimizationResult,
    q4: pd.DataFrame,
    weighted_10: float,
    weighted_5: float,
    priority_low: pd.DataFrame,
    priority_high: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_plot_style()
    plot_data_audit(audit, output_dir / "fig01_data_audit.png")
    plot_censor_diagnostics(audit, output_dir / "fig02_censor_diagnostics.png")
    plot_power_fit(audit.frame, power, output_dir / "fig03_power_model.png")
    plot_regimes(audit.frame, regimes, output_dir / "fig04_regime_segmentation.png")
    plot_regime_centers(regimes, output_dir / "fig05_regime_centers.png")
    plot_effect_curves(twin, output_dir / "fig06_effect_curves.png")
    plot_temperature_voltage_surface(twin, output_dir / "fig07_temperature_voltage_surface.png")
    plot_rapping_dynamics(twin, output_dir / "fig08_rapping_dynamics.png")
    plot_policy_heatmap(result_10, output_dir / "fig09_policy_10_heatmap.png", "10 mg/Nm³最优参数")
    plot_priority(priority_low, priority_high, output_dir / "fig10_priority_comparison.png")
    plot_q4_energy(q4, weighted_10, weighted_5, output_dir / "fig11_q4_energy_increase.png")
