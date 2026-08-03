from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

from .data_audit import AuditResult
from .emission_twin import EmissionTwin
from .optimization import OptimizationResult, POLICY_COLUMNS
from .power_model import PowerSurrogate
from .regimes import RegimeModel

# -- 论文统一色板 ---------------------------------------------------------------
# 颜色按语义固定：电压/基准=蓝，安全/低负荷=青绿，振打=橙，风险/高负荷=红。
COLORS = [
    "#2F6690",  # 主蓝 -- 电压 / 基准策略 / 主要结果
    "#3A8D8C",  # 青绿 -- 安全 / 低负荷 / 次要序列
    "#E39A32",  # 琥珀橙 -- 振打周期 / 能耗增量
    "#C84A42",  # 砖红 -- 风险 / 高负荷 / 删失
    "#9A78B5",  # 柔紫 -- 补充类别
    "#6BA6A6",  # 浅青 -- 补充类别
    "#747474",  # 中性灰 -- 参考线 / 历史值
    "#9A6B46",  # 暖棕 -- 补充类别
]

PRIMARY_BLUE, SAFE_TEAL, RAPPING_AMBER, RISK_RED, _, _, NEUTRAL_GRAY, _ = COLORS

# 同一控制量使用同一色相，以线型辅助区分不同电场。
VOLTAGE_COLORS = ["#244E70", "#2F6690", "#5D8FB5", "#8CB4D0"]
RAPPING_COLORS = ["#A96013", "#C97B20", "#E39A32", "#F0B766"]

# R1--R8 只在工况分类图中使用，降低饱和度并避免与控制语义混淆。
REGIME_COLORS = [
    "#2F6690", "#6C9BC3", "#3A8D8C", "#78B7A4",
    "#E39A32", "#C97845", "#9A78B5", "#747474",
]

VOLTAGE_CMAP = LinearSegmentedColormap.from_list(
    "esp_voltage", ["#F3F7FA", "#8CB4D0", "#2F6690", "#244E70"],
)
RAPPING_CMAP = LinearSegmentedColormap.from_list(
    "esp_rapping", ["#FFF8E8", "#F0B766", "#E39A32", "#A96013"],
)
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "esp_diverging", [PRIMARY_BLUE, "#F4F5F6", RISK_RED],
)
SURFACE_CMAP = LinearSegmentedColormap.from_list(
    "esp_surface", ["#F3C969", SAFE_TEAL, PRIMARY_BLUE, "#244E70"],
)

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
        "PingFang SC",
        "Hiragino Sans GB",
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
            "font.size": 8.5,
            "axes.unicode_minus": False,
            # Figure
            "figure.dpi": 150,
            "figure.facecolor": "white",
            "figure.titlesize": 10,
            "figure.titleweight": "semibold",
            # Axes
            "axes.titlesize": 9,
            "axes.titleweight": "semibold",
            "axes.labelsize": 8.5,
            "axes.labelweight": "normal",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.6,
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            # Grid
            "grid.alpha": 0.12,
            "grid.linewidth": 0.4,
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
            "lines.linewidth": 1.2,
            "lines.markersize": 3.5,
            "lines.markeredgewidth": 0.0,
            # Legend
            "legend.fontsize": 8,
            "legend.frameon": False,
            "legend.title_fontsize": 8,
            # Save
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "savefig.pad_inches": 0.05,
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    """保存高分辨率 PNG，并同步保存可用于排版的矢量 PDF。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white", edgecolor="none")
    if path.suffix.lower() == ".png":
        fig.savefig(path.with_suffix(".pdf"), facecolor="white", edgecolor="none")
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
        lw=0.6, color=RAPPING_AMBER, alpha=0.72,
    )
    twin.set_ylabel("入口浓度 / (g/Nm3)", color=RAPPING_AMBER)
    twin.tick_params(axis="y", colors=RAPPING_AMBER)
    twin.spines["right"].set_color(RAPPING_AMBER)

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
        marker="x", s=12, color=NEUTRAL_GRAY, label="缺失读数",
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
    bar_colors = [RAPPING_AMBER if v < 0 else PRIMARY_BLUE for v in correlation.values]
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
    fig = plt.figure(figsize=(9.2, 5.2))
    grid = fig.add_gridspec(2, 2, width_ratios=(1.05, 1), hspace=0.42, wspace=0.32)
    ax_fit = fig.add_subplot(grid[:, 0])
    ax_residual = fig.add_subplot(grid[0, 1])
    ax_cv = fig.add_subplot(grid[1, 1])

    # (a) 预测 vs 实测
    lo = min(frame["P_total_kW"].min(), prediction.min())
    hi = max(frame["P_total_kW"].max(), prediction.max())
    ax_fit.scatter(
        frame["P_total_kW"], prediction,
        s=4, alpha=0.18, color=COLORS[0], edgecolors="none", rasterized=True,
    )
    ax_fit.plot([lo, hi], [lo, hi], color=NEUTRAL_GRAY, lw=1.1)
    ax_fit.set_xlabel("实测功率 / kW")
    ax_fit.set_ylabel("预测功率 / kW")
    ax_fit.set_title("(a) 预测值与实测值")
    ax_fit.set_aspect("equal", adjustable="box")
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    ax_fit.text(
        0.04, 0.96, f"MAE = {mae:.2f} kW\nRMSE = {rmse:.2f} kW",
        transform=ax_fit.transAxes, va="top", fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2.5},
    )

    # (b) 残差时序
    ax_residual.plot(
        frame["timestamp"], residual, lw=0.45, alpha=0.65,
        color=COLORS[1], rasterized=True,
    )
    ax_residual.axhline(0, color="#333333", lw=0.7)
    ax_residual.set_title("(b) 残差时序")
    ax_residual.set_ylabel("残差 / kW")
    ax_residual.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    # (c) 逐日留一 R2
    cv = model.cv_metrics
    bars = ax_cv.bar(
        cv["held_out_day"].str[-5:], cv["r2"], color=COLORS[0], width=0.62,
    )
    ax_cv.set_ylim(max(0.95, cv["r2"].min() - 0.01), 1.0)
    ax_cv.tick_params(axis="x", rotation=30)
    ax_cv.set_title(f"(c) 逐日留一验证（平均 $R^2$ = {cv['r2'].mean():.4f}）")
    ax_cv.set_ylabel("$R^2$")
    for bar, value in zip(bars, cv["r2"]):
        ax_cv.text(
            bar.get_x() + bar.get_width() / 2, value + 0.0006,
            f"{value:.3f}", ha="center", va="bottom", fontsize=6.5,
        )

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.94)
    save_figure(fig, path)


def plot_regimes(frame: pd.DataFrame, regimes: RegimeModel, path: Path) -> None:
    pca = PCA(n_components=2).fit_transform(regimes.scaled_features)
    fig, axes = plt.subplots(
        2, 1, figsize=(9.2, 5.8),
        gridspec_kw={"height_ratios": [2.25, 0.55], "hspace": 0.28},
    )

    # (a) PCA 散点图
    step = max(1, len(frame) // 3200)
    for ri in range(regimes.n_regimes):
        mask = regimes.labels[::step] == ri
        axes[0].scatter(
            pca[::step][mask, 0], pca[::step][mask, 1],
            s=7, alpha=0.52, edgecolors="none", rasterized=True,
            color=REGIME_COLORS[ri % len(REGIME_COLORS)],
        )
        center = np.median(pca[regimes.labels == ri], axis=0)
        axes[0].annotate(
            f"R{ri + 1}", center, ha="center", va="center", fontsize=8,
            fontweight="semibold",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.8},
        )
    axes[0].set_xlabel("第一主成分 PC1")
    axes[0].set_ylabel("第二主成分 PC2")
    axes[0].set_title("(a) 稳健特征空间中的工况分布")

    # (b) 工况时间序列
    time_values = mdates.date2num(frame["timestamp"])
    axes[1].imshow(
        (regimes.labels + 1)[None, :], aspect="auto", interpolation="nearest",
        cmap=ListedColormap(REGIME_COLORS[:regimes.n_regimes]),
        extent=[time_values[0], time_values[-1], 0, 1],
        vmin=0.5, vmax=regimes.n_regimes + 0.5,
    )
    axes[1].set_yticks([])
    axes[1].set_xlim(time_values[0], time_values[-1])
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    axes[1].set_title("(b) 时间正则化后的工况序列", pad=5)
    axes[1].set_xlabel("日期")
    axes[1].grid(False)
    for spine in axes[1].spines.values():
        spine.set_visible(False)

    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.11, top=0.95)
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
        z, annot=True, fmt=".2f", cmap=DIVERGING_CMAP, center=0,
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
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.2))

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
    axes[0, 0].set_xlabel(r"入口浓度 $C_{\mathrm{in}}$ / (g·Nm$^{-3}$)")
    axes[0, 0].set_ylabel(r"预测浓度 $C_{\mathrm{out}}$ / (mg·Nm$^{-3}$)")
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
    axes[0, 1].set_xlabel(r"烟气流量 / ($10^3$ Nm$^3$·h$^{-1}$)")
    axes[0, 1].set_ylabel(r"预测浓度 $C_{\mathrm{out}}$ / (mg·Nm$^{-3}$)")
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
        axes[1, 0].plot(
            multipliers, values, label=rf"$U_{i+1}$", color=VOLTAGE_COLORS[i],
            lw=1.25, ls=("-", "--", "-.", ":")[i],
        )
    axes[1, 0].set_xlabel("相对基准电压")
    axes[1, 0].set_ylabel(r"预测浓度 $C_{\mathrm{out}}$ / (mg·Nm$^{-3}$)")
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
            multipliers, values, label=rf"$T_{i+1}$",
            color=RAPPING_COLORS[i], lw=1.25, ls=("-", "--", "-.", ":")[i],
        )
    axes[1, 1].set_xlabel("相对基准周期")
    axes[1, 1].set_ylabel(r"预测浓度 $C_{\mathrm{out}}$ / (mg·Nm$^{-3}$)")
    axes[1, 1].set_title("(d) 振打周期的积灰-再飞扬折中")
    axes[1, 1].legend(fontsize=7, ncol=2)

    for ax in axes.flat:
        ax.axhline(50, color="#666666", lw=0.7, ls="--", alpha=0.75)
    for ax in axes[1, :]:
        ax.axvline(1.0, color="#666666", lw=0.7, ls="--", alpha=0.75)
    fig.subplots_adjust(
        left=0.09, right=0.98, bottom=0.09, top=0.95,
        hspace=0.42, wspace=0.28,
    )
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
        voltage_scale, temp, np.log10(surface), levels=25, cmap=SURFACE_CMAP,
    )
    cbar = fig.colorbar(contour, ax=ax, shrink=0.85)
    cbar.set_label("log10(C_out)")
    cbar.ax.tick_params(labelsize=7)
    ax.contour(
        voltage_scale, temp, surface, levels=[5, 10, 50],
        colors=["#F4F5F6", RAPPING_AMBER, RISK_RED], linewidths=0.9,
        linestyles=["-", "--", "-."],
    )
    ax.set_xlabel("四电场电压统一倍率")
    ax.set_ylabel("入口温度 / C")
    ax.set_title("温度-电压协同响应面")
    save_figure(fig, path)


def plot_rapping_dynamics(twin: EmissionTwin, path: Path) -> None:
    minutes = np.arange(0, 70)
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 6), sharex=True)

    for line_index, (period, color) in enumerate(zip(
        [180, 233, 290],
        [PRIMARY_BLUE, RAPPING_AMBER, "#9A78B5"],
    )):
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
        line_style = ("-", "--", "-.")[line_index]
        axes[0].plot(
            minutes, loads, color=color, label=f"T = {period} s",
            lw=1.1, ls=line_style,
        )
        _, stemlines, _ = axes[1].stem(
            minutes, peaks, linefmt=color, markerfmt=" ", basefmt=" ",
            label=f"T = {period} s",
        )
        plt.setp(stemlines, linestyle=line_style, linewidth=0.9)

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
    voltages = data.iloc[:, :4]
    periods = data.iloc[:, 4:]
    labels = [f"R{i+1}" for i in result.table["regime"]]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.7), sharey=True)
    sns.heatmap(
        voltages, annot=True, fmt=".1f", cmap=VOLTAGE_CMAP,
        vmin=float(voltages.to_numpy().min()),
        vmax=float(voltages.to_numpy().max()),
        linewidths=0.6, linecolor="white", yticklabels=labels,
        xticklabels=[r"$U_1$", r"$U_2$", r"$U_3$", r"$U_4$"],
        annot_kws={"fontsize": 7.5},
        cbar_kws={"shrink": 0.82, "label": "电压 / kV"}, ax=axes[0],
    )
    sns.heatmap(
        periods, annot=True, fmt=".0f", cmap=RAPPING_CMAP,
        vmin=float(periods.to_numpy().min()),
        vmax=float(periods.to_numpy().max()),
        linewidths=0.6, linecolor="white", yticklabels=labels,
        xticklabels=[r"$T_1$", r"$T_2$", r"$T_3$", r"$T_4$"],
        annot_kws={"fontsize": 7.5},
        cbar_kws={"shrink": 0.82, "label": "振打周期 / s"}, ax=axes[1],
    )
    axes[0].set_title("(a) 四电场电压")
    axes[1].set_title("(b) 四电场振打周期")
    axes[0].set_ylabel("典型工况")
    axes[1].set_ylabel("")
    for ax in axes:
        ax.set_xlabel("控制变量")
        ax.tick_params(axis="both", labelsize=8, rotation=0)
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.12, top=0.92, wspace=0.26)
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
        color=[PRIMARY_BLUE] * (len(q4) - 1) + [RISK_RED],
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
