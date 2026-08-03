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

# -- Nature 级统一色板 ----------------------------------------------------------
# 色盲友好、视觉均匀，按视觉权重排序
COLORS = [
    "#2166AC",  # 蓝 -- 主序列 / 电压
    "#439F7A",  # 绿 -- 次序列 / 低负荷 / 尘负荷安全
    "#F08C39",  # 橙 -- 暖色调 / 振打周期
    "#D83432",  # 红 -- 警告 / 删失 / 高负荷 / 约束活跃
    "#8C6BB1",  # 紫 -- 第五类别
    "#339FA0",  # 青 -- 第六类别
    "#8C8C8C",  # 灰 -- 中性 / 参考线
    "#B8713D",  # 棕 -- 末尾类别
]

# -- 统一绘图风格 ----------------------------------------------------------------


def setup_plot_style() -> None:
    """应用 Nature 级统一 matplotlib 样式。

    特点:
    - 无衬线字体, 微软雅黑优先, 适配中文
    - 300 dpi 输出, 150 dpi 屏幕
    - 白底浅灰网格, 隐藏上/右脊线
    - 统一字号: 基础 9pt, 刻度 8pt, 轴标题 10pt, 总标题 11pt
    """
    font_root = Path("C:/Windows/Fonts")
    candidates = [
        font_root / "msyh.ttc",
        font_root / "msyhbd.ttc",
        font_root / "simhei.ttf",
    ]
    registered: list[str] = []
    for p in candidates:
        if p.exists():
            try:
                fm.fontManager.addfont(str(p))
                name = fm.FontProperties(fname=str(p)).get_name()
                if name not in registered:
                    registered.append(name)
            except Exception:
                pass

    fallback = registered + [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
        "Arial",
        "Helvetica",
    ]

    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.0)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": fallback,
            "font.size": 9,
            "axes.unicode_minus": False,
            # Figure
            "figure.dpi": 150,
            "figure.facecolor": "white",
            "figure.titlesize": 11,
            "figure.titleweight": "bold",
            # Axes
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.labelweight": "normal",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.5,
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            # Grid
            "grid.alpha": 0.15,
            "grid.linewidth": 0.3,
            "grid.color": "#999999",
            # Ticks
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.major.width": 0.4,
            "ytick.major.width": 0.4,
            # Lines
            "lines.linewidth": 1.0,
            "lines.markersize": 4,
            "lines.markeredgewidth": 0.0,
            # Legend
            "legend.fontsize": 8,
            "legend.frameon": False,
            "legend.title_fontsize": 8,
            # Save
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "savefig.pad_inches": 0.05,
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    """保存图像为 PNG (300 dpi), 然后关闭."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white", edgecolor="none")
    plt.close(fig)


# ===== 各图表函数 ===============================================================


def plot_data_audit(audit: AuditResult, path: Path) -> None:
    frame = audit.frame
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.0))

    # (a) 入口温度时序
    ax = axes[0, 0]
    ax.plot(frame["timestamp"], frame["Temp_C"], lw=0.8, color=COLORS[0])
    ax.set_title("(a) 7天入口温度")
    ax.set_ylabel("温度 / C")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    twin = ax.twinx()
    twin.plot(
        frame["timestamp"], frame["C_in_gNm3"],
        lw=0.6, color=COLORS[3], alpha=0.65,
    )
    twin.set_ylabel("入口浓度 / (g/Nm3)", color=COLORS[3])
    twin.tick_params(axis="y", colors=COLORS[3])
    twin.spines["right"].set_color(COLORS[3])

    # (b) 烟气流量
    ax = axes[0, 1]
    ax.plot(frame["timestamp"], frame["Q_Nm3h"] / 1000.0, lw=0.8, color=COLORS[1])
    ax.set_title("(b) 烟气流量")
    ax.set_ylabel("流量 / (千 Nm3/h)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    # (c) 出口浓度与删失
    ax = axes[1, 0]
    ax.plot(frame["timestamp"], frame["C_out_mgNm3"], lw=0.6, color=COLORS[3])
    ax.axhline(50, ls="--", color="#333333", lw=0.8, label="右删失上限 50")
    missing = frame["C_out_mgNm3"].isna()
    ax.scatter(
        frame.loc[missing, "timestamp"],
        np.full(missing.sum(), 48.65),
        marker="x", s=12, color=COLORS[7], label="缺失读数",
    )
    ax.set_title("(c) 出口浓度: 大量读数贴于仪表上限")
    ax.set_ylabel("C_out / (mg/Nm3)")
    ax.legend(loc="lower right", fontsize=7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    # (d) 删失比例直方图
    ax = axes[1, 1]
    uncensored = frame.loc[frame["C_out_mgNm3"] < 50, "C_out_mgNm3"]
    ax.hist(uncensored, bins=35, color=COLORS[0], alpha=0.75, label="未删失读数")
    ax.bar(
        [50], [audit.summary["output_censored"]],
        width=0.03, color=COLORS[3], label="删失读数",
    )
    ax.set_title(
        f"(d) 右删失比例 {audit.summary['output_censored_fraction']:.2%}"
    )
    ax.set_xlabel("C_out / (mg/Nm3)")
    ax.set_ylabel("频数")
    ax.legend(fontsize=7)

    fig.suptitle("数据质量与右删失证据")
    fig.tight_layout()
    save_figure(fig, path)


def plot_censor_diagnostics(audit: AuditResult, path: Path) -> None:
    from scipy.stats import norm

    correlation = audit.correlations.drop("C_out_mgNm3").sort_values(key=np.abs)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # (a) 相关系数条形图
    bar_colors = [COLORS[3] if v < 0 else COLORS[0] for v in correlation.values]
    axes[0].barh(correlation.index, correlation.values, color=bar_colors, height=0.6)
    axes[0].axvline(0, color="#333333", lw=0.6)
    axes[0].set_title("(a) 原始变量与出口浓度相关系数")
    axes[0].set_xlabel("Pearson r")
    axes[0].tick_params(axis="y", labelsize=7)

    # (b) 删失正态 MLE 拟合
    fit = audit.summary["censored_normal"]
    values = audit.frame["C_out_mgNm3"].dropna().to_numpy()
    uncensored = values[values < 50]
    x = np.linspace(48.7, 50.5, 300)
    axes[1].hist(
        uncensored, bins=35, density=True, alpha=0.5,
        color=COLORS[0], label="未删失样本",
    )
    axes[1].plot(
        x, norm.pdf(x, fit["mu"], fit["sigma"]),
        color=COLORS[3], lw=1.6, label="删失正态 MLE",
    )
    axes[1].axvline(50, color="#333333", ls="--", lw=0.8, label="删失点")
    axes[1].set_title(f"(b) 潜在均值 = {fit['mu']:.3f}, sigma = {fit['sigma']:.3f}")
    axes[1].set_xlabel("C_out / (mg/Nm3)")
    axes[1].legend(fontsize=7)

    fig.suptitle("删失分布与低可识别性诊断")
    fig.tight_layout()
    save_figure(fig, path)


def plot_power_fit(frame: pd.DataFrame, model: PowerSurrogate, path: Path) -> None:
    prediction = model.predict_frame(frame)
    residual = frame["P_total_kW"].to_numpy() - prediction
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))

    # (a) 预测 vs 实测
    lo = min(frame["P_total_kW"].min(), prediction.min())
    hi = max(frame["P_total_kW"].max(), prediction.max())
    axes[0].scatter(
        frame["P_total_kW"], prediction,
        s=5, alpha=0.25, color=COLORS[0], edgecolors="none",
    )
    axes[0].plot([lo, hi], [lo, hi], color=COLORS[3], lw=1.0)
    axes[0].set_xlabel("实测功率 / kW")
    axes[0].set_ylabel("预测功率 / kW")
    axes[0].set_title("(a) 预测 vs 实测")

    # (b) 残差时序
    axes[1].plot(frame["timestamp"], residual, lw=0.5, color=COLORS[1])
    axes[1].axhline(0, color="#333333", lw=0.6)
    axes[1].set_title("(b) 功率残差时序")
    axes[1].set_ylabel("残差 / kW")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    # (c) 逐日留一 R2
    cv = model.cv_metrics
    axes[2].bar(cv["held_out_day"].str[-5:], cv["r2"], color=COLORS[0], width=0.6)
    axes[2].set_ylim(max(0.95, cv["r2"].min() - 0.01), 1.0)
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].set_title(f"(c) 逐日留一验证, 平均 R2 = {cv['r2'].mean():.4f}")
    axes[2].set_ylabel("R2")

    fig.suptitle("能耗代理模型验证")
    fig.tight_layout()
    save_figure(fig, path)


def plot_regimes(frame: pd.DataFrame, regimes: RegimeModel, path: Path) -> None:
    pca = PCA(n_components=2).fit_transform(regimes.scaled_features)
    fig, axes = plt.subplots(
        2, 1, figsize=(12.8, 8.5),
        gridspec_kw={"height_ratios": [2, 1]},
    )

    # (a) PCA 散点图
    step = max(1, len(frame) // 5000)
    for ri in range(regimes.n_regimes):
        mask = regimes.labels[::step] == ri
        axes[0].scatter(
            pca[::step][mask, 0], pca[::step][mask, 1],
            s=6, alpha=0.45, edgecolors="none",
            label=f"R{ri + 1}", color=COLORS[ri % len(COLORS)],
        )
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].set_title("(a) 工况在稳健特征空间中的分布")
    axes[0].legend(
        ncol=min(8, regimes.n_regimes), loc="upper center",
        fontsize=7, markerscale=0.8,
    )

    # (b) 工况时间序列
    axes[1].step(
        frame["timestamp"], regimes.labels + 1,
        where="post", lw=0.6, color=COLORS[0],
    )
    axes[1].set_yticks(range(1, regimes.n_regimes + 1))
    axes[1].set_yticklabels([f"R{i}" for i in range(1, regimes.n_regimes + 1)])
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    axes[1].set_title("(b) 时间正则化后的工况序列")
    axes[1].set_ylabel("工况")

    fig.suptitle(f"典型工况识别 (轮廓系数选择 K = {regimes.n_regimes})")
    fig.tight_layout()
    save_figure(fig, path)


def plot_regime_centers(regimes: RegimeModel, path: Path) -> None:
    columns = [
        "Temp_C_mean", "C_in_gNm3_mean", "Q_Nm3h_mean",
        "mass_load_mean", "P_history_mean_kW",
    ]
    values = regimes.centers[columns].to_numpy(dtype=float)
    z = (values - values.mean(axis=0)) / values.std(axis=0)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    sns.heatmap(
        z, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        linewidths=0.5, linecolor="white",
        xticklabels=["温度", "入口浓度", "流量", "质量负荷", "历史功率"],
        yticklabels=regimes.centers["label"],
        annot_kws={"fontsize": 8},
        cbar_kws={"shrink": 0.82, "label": "Z-score"},
        ax=ax,
    )
    ax.set_title("典型工况中心 (列内标准化)")
    ax.tick_params(axis="both", labelsize=8)
    save_figure(fig, path)


def plot_effect_curves(twin: EmissionTwin, path: Path) -> None:
    reference_policy = np.r_[twin.u_ref, twin.t_ref]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))

    # (a) 入口浓度效应
    cin = np.linspace(18, 72, 100)
    axes[0, 0].plot(
        cin,
        twin.predict_deterministic(
            reference_policy,
            np.full_like(cin, twin.temp_ref), cin, np.full_like(cin, twin.q_ref),
        ),
        color=COLORS[0], lw=1.2,
    )
    axes[0, 0].set_xlabel("C_in / (g/Nm3)")
    axes[0, 0].set_ylabel("预测 C_out / (mg/Nm3)")
    axes[0, 0].set_title("(a) 入口浓度效应")

    # (b) 流量/停留时间效应
    flow = np.linspace(4.0e5, 5.15e5, 100)
    axes[0, 1].plot(
        flow / 1000,
        twin.predict_deterministic(
            reference_policy,
            np.full_like(flow, twin.temp_ref),
            np.full_like(flow, twin.c_in_ref), flow,
        ),
        color=COLORS[1], lw=1.2,
    )
    axes[0, 1].set_xlabel("流量 / (千 Nm3/h)")
    axes[0, 1].set_ylabel("预测 C_out / (mg/Nm3)")
    axes[0, 1].set_title("(b) 流量 / 停留时间效应")

    # (c) 分场电压单调效应
    multipliers = np.linspace(0.8, 1.2, 100)
    for i in range(4):
        values = [
            twin.predict_deterministic(
                _voltage_perturb(reference_policy, i, m),
                twin.temp_ref, twin.c_in_ref, twin.q_ref,
            )[0]
            for m in multipliers
        ]
        axes[1, 0].plot(multipliers, values, label=f"U{i+1}", color=COLORS[i], lw=1.0)
    axes[1, 0].set_xlabel("相对基准电压")
    axes[1, 0].set_ylabel("预测 C_out / (mg/Nm3)")
    axes[1, 0].set_title("(c) 分场电压单调效应")
    axes[1, 0].legend(fontsize=7, ncol=2)

    # (d) 振打周期积灰-再飞扬折中
    for i in range(4):
        values = [
            twin.predict_deterministic(
                _rapping_perturb(reference_policy, i, m),
                twin.temp_ref, twin.c_in_ref, twin.q_ref,
            )[0]
            for m in multipliers
        ]
        axes[1, 1].plot(
            multipliers, values, label=f"T{i+1}",
            color=COLORS[i], lw=1.0, marker=".", markersize=2,
        )
    axes[1, 1].set_xlabel("相对基准周期")
    axes[1, 1].set_ylabel("预测 C_out / (mg/Nm3)")
    axes[1, 1].set_title("(d) 振打周期的积灰-再飞扬折中")
    axes[1, 1].legend(fontsize=7, ncol=2)

    fig.suptitle("物理信息模型的单因素响应")
    fig.tight_layout()
    save_figure(fig, path)


def _voltage_perturb(policy: np.ndarray, field: int, multiplier: float) -> np.ndarray:
    p = policy.copy()
    p[field] = p[field] * multiplier
    return p


def _rapping_perturb(policy: np.ndarray, field: int, multiplier: float) -> np.ndarray:
    p = policy.copy()
    p[4 + field] = p[4 + field] * multiplier
    return p


def plot_temperature_voltage_surface(twin: EmissionTwin, path: Path) -> None:
    temp = np.linspace(110, 160, 80)
    voltage_scale = np.linspace(0.85, 1.22, 80)
    surface = np.empty((len(temp), len(voltage_scale)))
    for i, t_val in enumerate(temp):
        for j, scale in enumerate(voltage_scale):
            policy = np.r_[twin.u_ref * scale, twin.t_ref]
            surface[i, j] = twin.predict_deterministic(
                policy, t_val, twin.c_in_ref, twin.q_ref,
            )[0]

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    contour = ax.contourf(
        voltage_scale, temp, np.log10(surface), levels=25, cmap="viridis_r",
    )
    cbar = fig.colorbar(contour, ax=ax, shrink=0.85)
    cbar.set_label("log10(C_out)")
    cbar.ax.tick_params(labelsize=7)
    ax.contour(
        voltage_scale, temp, surface, levels=[5, 10, 50],
        colors=["white", "yellow", "red"], linewidths=0.8,
        linestyles=["-", "--", "-."],
    )
    ax.set_xlabel("四电场电压统一倍率")
    ax.set_ylabel("入口温度 / C")
    ax.set_title("温度-电压协同响应面")
    save_figure(fig, path)


def plot_rapping_dynamics(twin: EmissionTwin, path: Path) -> None:
    minutes = np.arange(0, 70)
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 6), sharex=True)

    for period, color in zip(
        [180, 233, 290], [COLORS[1], COLORS[0], COLORS[3]],
    ):
        phase = 0.0
        loads, peaks = [], []
        for _ in minutes:
            phase += 60.0 / period
            peak = 0.0
            if phase >= 1.0:
                phase -= 1.0
                peak = period / 233.0
            loads.append(phase)
            peaks.append(peak)
        axes[0].plot(minutes, loads, color=color, label=f"T = {period} s", lw=1.0)
        axes[1].stem(
            minutes, peaks, linefmt=color, markerfmt=" ", basefmt=" ",
            label=f"T = {period} s",
        )

    axes[0].set_ylabel("归一化积灰进度")
    axes[0].set_title("(a) 归一化振打相位")
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel("时间 / min")
    axes[1].set_ylabel("相对峰值")
    axes[1].set_title("(b) 一阶质量守恒下的相对脉冲示意")

    fig.suptitle("振打周期对积灰与瞬时峰值的归一化示意")
    fig.tight_layout()
    save_figure(fig, path)


def plot_policy_heatmap(result: OptimizationResult, path: Path, title: str) -> None:
    data = result.table[POLICY_COLUMNS].copy()
    normalized = (data - data.min(axis=0)) / (
        data.max(axis=0) - data.min(axis=0) + 1e-12
    )
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    sns.heatmap(
        normalized, annot=data.round(1), fmt="", cmap="YlGnBu",
        linewidths=0.5, linecolor="white",
        yticklabels=[f"R{i+1}" for i in result.table["regime"]],
        annot_kws={"fontsize": 7.5},
        cbar_kws={"shrink": 0.8, "label": "列内归一化值"},
        ax=ax,
    )
    ax.set_title(title + "  (颜色: 列内归一化; 标注: 实际值)")
    ax.set_xlabel("控制变量")
    ax.set_ylabel("工况")
    ax.tick_params(axis="both", labelsize=8)
    save_figure(fig, path)


def plot_q4_energy(
    q4: pd.DataFrame, weighted_10: float, weighted_5: float, path: Path,
) -> None:
    x = np.arange(len(q4))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # (a) 各工况最优电耗
    axes[0].bar(x - width / 2, q4["power_10_kW"], width,
                label="10 mg/Nm3", color=COLORS[0])
    axes[0].bar(x + width / 2, q4["power_5_conditional_kW"], width,
                label="5 mg/Nm3", color=COLORS[3])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(q4["label"])
    axes[0].set_ylabel("功率 / kW")
    axes[0].set_title("(a) 各工况最优电耗")
    axes[0].legend(fontsize=7)

    # (b) 限值收紧的电耗增幅
    axes[1].bar(
        q4["label"], q4["increase_pct"],
        color=[COLORS[i % len(COLORS)] for i in range(len(q4))],
        width=0.6,
    )
    weighted_pct = 100 * (weighted_5 - weighted_10) / weighted_10
    axes[1].axhline(
        weighted_pct, color="#333333", ls="--", lw=0.8,
        label=f"加权增幅 {weighted_pct:.1f}%",
    )
    axes[1].set_ylabel("增幅 / %")
    axes[1].set_title("(b) 限值收紧的电耗代价")
    axes[1].legend(fontsize=7)

    fig.suptitle("10 -> 5 mg/Nm3 的条件性电耗增幅")
    fig.tight_layout()
    save_figure(fig, path)


def plot_priority(
    priority_low: pd.DataFrame, priority_high: pd.DataFrame, path: Path,
) -> None:
    merged = (
        priority_low[["parameter", "reduction_per_added_kW"]]
        .merge(
            priority_high[["parameter", "reduction_per_added_kW"]],
            on="parameter", suffixes=("_low", "_high"),
        )
        .set_index("parameter")
    )
    fig, ax = plt.subplots(figsize=(9, 4.8))
    merged.plot(kind="bar", ax=ax, color=[COLORS[1], COLORS[3]], width=0.7)
    ax.set_ylabel("减排收益 / (mg/Nm3 per kW)")
    ax.set_xlabel("操作变量")
    ax.set_title("低负荷/高负荷工况的单位增量电耗减排收益")
    ax.legend(["低负荷", "高负荷"], fontsize=7)
    ax.tick_params(axis="x", rotation=0)
    save_figure(fig, path)


# ===== 批量生成 ================================================================


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
    """统一风格批量生成 fig01--fig11."""
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
    plot_policy_heatmap(
        result_10, output_dir / "fig09_policy_10_heatmap.png",
        "10 mg/Nm3 最优参数",
    )
    plot_priority(priority_low, priority_high, output_dir / "fig10_priority_comparison.png")
    plot_q4_energy(q4, weighted_10, weighted_5, output_dir / "fig11_q4_energy_increase.png")
