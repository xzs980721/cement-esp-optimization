from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm


EXPECTED_COLUMNS = [
    "timestamp",
    "Temp_C",
    "C_in_gNm3",
    "Q_Nm3h",
    "U1_kV",
    "U2_kV",
    "U3_kV",
    "U4_kV",
    "T1_s",
    "T2_s",
    "T3_s",
    "T4_s",
    "C_out_mgNm3",
    "P_total_kW",
]

OPERATING_COLUMNS = [
    "Temp_C",
    "C_in_gNm3",
    "Q_Nm3h",
    "U1_kV",
    "U2_kV",
    "U3_kV",
    "U4_kV",
    "T1_s",
    "T2_s",
    "T3_s",
    "T4_s",
]


@dataclass
class AuditResult:
    frame: pd.DataFrame
    summary: dict[str, Any]
    feature_summary: pd.DataFrame
    correlations: pd.Series


def _fit_right_censored_normal(values: np.ndarray, threshold: float) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    uncensored = values[values < threshold]
    n_censored = int(np.sum(values >= threshold))

    def negative_log_likelihood(theta: np.ndarray) -> float:
        mu, log_sigma = theta
        sigma = np.exp(log_sigma)
        z = (uncensored - mu) / sigma
        log_pdf = -log_sigma - 0.5 * np.log(2 * np.pi) - 0.5 * z**2
        survival = np.clip(norm.sf((threshold - mu) / sigma), 1e-300, 1.0)
        return float(-(log_pdf.sum() + n_censored * np.log(survival)))

    initial_sigma = max(float(np.std(uncensored)), 0.05)
    fit = minimize(
        negative_log_likelihood,
        x0=np.array([threshold, np.log(initial_sigma)]),
        method="L-BFGS-B",
        bounds=[(threshold - 5.0, threshold + 5.0), (np.log(0.01), np.log(10.0))],
    )
    mu = float(fit.x[0])
    sigma = float(np.exp(fit.x[1]))
    alpha = (threshold - mu) / sigma
    expected_above = mu + sigma * norm.pdf(alpha) / max(norm.sf(alpha), 1e-12)
    return {
        "mu": mu,
        "sigma": sigma,
        "expected_latent_if_censored": float(expected_above),
        "log_likelihood": float(-fit.fun),
        "converged": bool(fit.success),
    }


def _ordinary_r2(frame: pd.DataFrame) -> float:
    valid = frame["C_out_mgNm3"].notna()
    x = frame.loc[valid, OPERATING_COLUMNS].to_numpy(dtype=float)
    y = frame.loc[valid, "C_out_mgNm3"].to_numpy(dtype=float)
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    design = np.column_stack([np.ones(len(x)), x])
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    prediction = design @ coef
    return float(1.0 - np.sum((y - prediction) ** 2) / np.sum((y - y.mean()) ** 2))


def _censor_indicator_r2(frame: pd.DataFrame, threshold: float) -> float:
    valid = frame["C_out_mgNm3"].notna()
    x = frame.loc[valid, OPERATING_COLUMNS].to_numpy(dtype=float)
    y = (frame.loc[valid, "C_out_mgNm3"].to_numpy(dtype=float) >= threshold).astype(float)
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    design = np.column_stack([np.ones(len(x)), x])
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    prediction = design @ coef
    return float(1.0 - np.sum((y - prediction) ** 2) / np.sum((y - y.mean()) ** 2))


def load_and_audit_data(csv_path: str | Path, censor_threshold: float = 50.0) -> AuditResult:
    csv_path = Path(csv_path)
    frame = pd.read_csv(csv_path, encoding="utf-8-sig", parse_dates=["timestamp"])
    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    frame = frame[EXPECTED_COLUMNS].sort_values("timestamp").reset_index(drop=True)
    if frame["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps were found.")

    deltas = frame["timestamp"].diff().dropna()
    complete_minute_grid = bool((deltas == pd.Timedelta(minutes=1)).all())
    output = frame["C_out_mgNm3"]
    censored_normal = _fit_right_censored_normal(output.to_numpy(), censor_threshold)

    summary = {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "start": frame["timestamp"].min().isoformat(sep=" "),
        "end": frame["timestamp"].max().isoformat(sep=" "),
        "complete_minute_grid": complete_minute_grid,
        "output_missing": int(output.isna().sum()),
        "output_censored": int((output >= censor_threshold).sum()),
        "output_censored_fraction": float((output >= censor_threshold).mean()),
        "output_min": float(output.min()),
        "output_max": float(output.max()),
        "ordinary_ols_r2": _ordinary_r2(frame),
        "censor_indicator_linear_r2": _censor_indicator_r2(frame, censor_threshold),
        "censored_normal": censored_normal,
    }

    feature_summary = frame.drop(columns="timestamp").describe().T
    feature_summary["missing"] = frame.drop(columns="timestamp").isna().sum()
    correlations = frame.select_dtypes(include=[np.number]).corr()["C_out_mgNm3"].sort_values()
    return AuditResult(frame=frame, summary=summary, feature_summary=feature_summary, correlations=correlations)

