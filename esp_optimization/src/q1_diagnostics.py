from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from .data_audit import AuditResult
from .emission_twin import EmissionTwin, T_COLUMNS, U_COLUMNS


def _historical_twin_score(frame: pd.DataFrame, twin: EmissionTwin) -> np.ndarray:
    """Evaluate the deterministic twin at each recorded operating point."""

    u_ratio_sq = (frame[U_COLUMNS].to_numpy(dtype=float) / twin.u_ref) ** 2
    t_ratio = frame[T_COLUMNS].to_numpy(dtype=float) / twin.t_ref
    stage_score = u_ratio_sq @ twin.stage_weights
    long_cycle_excess = np.maximum(t_ratio - 1.0, 0.0) ** 2
    plate_index = long_cycle_excess @ twin.stage_weights
    load_ratio = (
        frame["C_in_gNm3"].to_numpy(dtype=float)
        * frame["Q_Nm3h"].to_numpy(dtype=float)
        / (twin.c_in_ref * twin.q_ref)
    )
    temperature_factor = np.exp(
        -twin.temperature_sensitivity
        * ((frame["Temp_C"].to_numpy(dtype=float) - twin.temperature_optimum)
           / twin.temperature_scale) ** 2
    )
    plate_factor = np.exp(-twin.plate_penalty * load_ratio * plate_index)
    capture_exponent = (
        twin.d_ref
        * (twin.q_ref / frame["Q_Nm3h"].to_numpy(dtype=float))
        * temperature_factor
        * stage_score
        * plate_factor
    )
    accumulated_dust = t_ratio @ twin.stage_weights
    rapping_factor = 1.0 + twin.rapping_load_coefficient * load_ratio * accumulated_dust
    return (
        1000.0
        * frame["C_in_gNm3"].to_numpy(dtype=float)
        * np.exp(-capture_exponent)
        * rapping_factor
    )


def q1_model_data_consistency(
    audit: AuditResult,
    twin: EmissionTwin,
) -> pd.DataFrame:
    """Separate marginal censor calibration from conditional identifiability.

    The censored-normal anchor can be checked against the marginal outlet
    distribution.  The full physical twin is additionally evaluated as a
    ranking score for the censor event; an AUC near 0.5 and a near-zero
    uncensored correlation indicate that the recorded closed-loop data do not
    identify the conditional factor-response surface.
    """

    frame = audit.frame
    threshold = float(twin.censor_threshold)
    output = frame["C_out_mgNm3"].to_numpy(dtype=float)
    valid = np.isfinite(output)
    uncensored = valid & (output < threshold)
    censored = output[valid] >= threshold

    alpha = (threshold - twin.censor_mu) / twin.censor_sigma
    below_probability = float(norm.cdf(alpha))
    fitted_censor_fraction = float(norm.sf(alpha))
    fitted_uncensored_mean = float(
        twin.censor_mu
        - twin.censor_sigma * norm.pdf(alpha) / max(below_probability, 1e-12)
    )
    probabilities = np.asarray([0.25, 0.50, 0.75])
    fitted_quantiles = twin.censor_mu + twin.censor_sigma * norm.ppf(
        probabilities * below_probability
    )
    observed_quantiles = np.quantile(output[uncensored], probabilities)

    twin_score = _historical_twin_score(frame, twin)
    censor_auc = float(roc_auc_score(censored.astype(int), twin_score[valid]))
    uncensored_correlation = float(
        np.corrcoef(twin_score[uncensored], output[uncensored])[0, 1]
    )

    rows = [
        {
            "metric": "censor_fraction",
            "observed": float(censored.mean()),
            "model_or_diagnostic": fitted_censor_fraction,
            "absolute_gap": abs(float(censored.mean()) - fitted_censor_fraction),
            "interpretation": "marginal censored-normal calibration",
        },
        {
            "metric": "uncensored_mean_mgNm3",
            "observed": float(output[uncensored].mean()),
            "model_or_diagnostic": fitted_uncensored_mean,
            "absolute_gap": abs(float(output[uncensored].mean()) - fitted_uncensored_mean),
            "interpretation": "conditional mean below the censor threshold",
        },
    ]
    for label, observed, fitted in zip(
        ("uncensored_q25_mgNm3", "uncensored_q50_mgNm3", "uncensored_q75_mgNm3"),
        observed_quantiles,
        fitted_quantiles,
    ):
        rows.append(
            {
                "metric": label,
                "observed": float(observed),
                "model_or_diagnostic": float(fitted),
                "absolute_gap": abs(float(observed) - float(fitted)),
                "interpretation": "conditional quantile below the censor threshold",
            }
        )
    rows.extend(
        [
            {
                "metric": "conditional_censor_auc",
                "observed": 0.5,
                "model_or_diagnostic": censor_auc,
                "absolute_gap": abs(censor_auc - 0.5),
                "interpretation": "0.5 denotes no conditional discrimination",
            },
            {
                "metric": "uncensored_prediction_correlation",
                "observed": 0.0,
                "model_or_diagnostic": uncensored_correlation,
                "absolute_gap": abs(uncensored_correlation),
                "interpretation": "near zero denotes no conditional tracking",
            },
            {
                "metric": "ordinary_ols_r2",
                "observed": 0.0,
                "model_or_diagnostic": float(audit.summary["ordinary_ols_r2"]),
                "absolute_gap": abs(float(audit.summary["ordinary_ols_r2"])),
                "interpretation": "raw operating variables have negligible explanatory power",
            },
        ]
    )
    return pd.DataFrame(rows)
