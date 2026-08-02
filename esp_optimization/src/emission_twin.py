from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .data_audit import AuditResult


U_COLUMNS = [f"U{i}_kV" for i in range(1, 5)]
T_COLUMNS = [f"T{i}_s" for i in range(1, 5)]


def _derive_stage_weights(u_ref: np.ndarray, prior: dict[str, Any]) -> np.ndarray:
    """Return a transparent prior center for the four field contributions."""
    method = prior.get("stage_weight_method", "configured")
    if method == "reference_voltage_squared":
        raw = np.asarray(u_ref, dtype=float) ** 2
    elif method == "configured":
        raw = np.asarray(prior["stage_weights"], dtype=float)
    else:
        raise ValueError(f"Unknown stage-weight method: {method}")
    if raw.shape != (4,) or np.any(~np.isfinite(raw)) or np.any(raw <= 0):
        raise ValueError("Stage weights must contain four positive finite values.")
    return raw / raw.sum()


def _derive_temperature_reference(frame: pd.DataFrame, prior: dict[str, Any]) -> float:
    """Return the documented operating-temperature reference."""
    method = prior.get("temperature_reference_method", "data_mean")
    if method == "data_mean":
        reference = float(frame["Temp_C"].mean())
    else:
        raise ValueError(f"Unknown temperature-reference method: {method}")
    if not np.isfinite(reference):
        raise ValueError("Temperature reference must be finite.")
    return reference


def _lognormal_draws(rng: np.random.Generator, mean: float, relative_sd: float, n: int) -> np.ndarray:
    cv2 = relative_sd**2
    sigma2 = np.log1p(cv2)
    mu = np.log(mean) - 0.5 * sigma2
    return rng.lognormal(mean=mu, sigma=np.sqrt(sigma2), size=n)


@dataclass
class EmissionScenarios:
    temp_c: np.ndarray
    c_in: np.ndarray
    q: np.ndarray
    d_ref: np.ndarray
    stage_weights: np.ndarray
    temperature_sensitivity: np.ndarray
    plate_penalty: np.ndarray
    rapping_load_coefficient: np.ndarray
    residual: np.ndarray


@dataclass
class EmissionTwin:
    censor_threshold: float
    censor_mu: float
    censor_sigma: float
    temp_ref: float
    c_in_ref: float
    q_ref: float
    u_ref: np.ndarray
    t_ref: np.ndarray
    stage_weights: np.ndarray
    stage_weight_concentration: float
    temperature_optimum: float
    temperature_sensitivity: float
    temperature_scale: float
    plate_penalty: float
    rapping_load_coefficient: float
    residual_sigma: float
    removal_relative_sd: float
    prior_relative_sd: float
    d_ref: float
    identifiability: pd.DataFrame

    def scenario_pack(
        self,
        regime_frame: pd.DataFrame,
        n_scenarios: int,
        seed: int,
        prior_scale: float = 1.0,
    ) -> EmissionScenarios:
        rng = np.random.default_rng(seed)
        sampled = regime_frame.iloc[
            rng.choice(len(regime_frame), size=n_scenarios, replace=True)
        ]
        relative_sd = self.prior_relative_sd * prior_scale
        concentration = max(self.stage_weight_concentration / max(prior_scale**2, 0.1), 8.0)
        weights = rng.dirichlet(self.stage_weights * concentration, size=n_scenarios)
        return EmissionScenarios(
            temp_c=sampled["Temp_C"].to_numpy(dtype=float),
            c_in=sampled["C_in_gNm3"].to_numpy(dtype=float),
            q=sampled["Q_Nm3h"].to_numpy(dtype=float),
            d_ref=np.clip(
                rng.normal(
                    self.d_ref,
                    self.d_ref * self.removal_relative_sd * prior_scale,
                    size=n_scenarios,
                ),
                0.25 * self.d_ref,
                None,
            ),
            stage_weights=weights,
            temperature_sensitivity=_lognormal_draws(
                rng, self.temperature_sensitivity, relative_sd, n_scenarios
            ),
            plate_penalty=_lognormal_draws(rng, self.plate_penalty, relative_sd, n_scenarios),
            rapping_load_coefficient=_lognormal_draws(
                rng, self.rapping_load_coefficient, relative_sd, n_scenarios
            ),
            residual=rng.normal(0.0, self.residual_sigma * prior_scale, size=n_scenarios),
        )

    def predict_scenarios(self, policy: np.ndarray, scenarios: EmissionScenarios) -> np.ndarray:
        policy = np.asarray(policy, dtype=float)
        if policy.shape != (8,):
            raise ValueError("Policy must contain U1..U4,T1..T4.")
        u_ratio_sq = (policy[:4] / self.u_ref) ** 2
        t_ratio = policy[4:] / self.t_ref
        stage_score = scenarios.stage_weights @ u_ratio_sq
        long_cycle_excess = np.maximum(t_ratio - 1.0, 0.0) ** 2
        plate_index = scenarios.stage_weights @ long_cycle_excess
        load_ratio = (scenarios.c_in * scenarios.q) / (self.c_in_ref * self.q_ref)
        temperature_factor = np.exp(
            -scenarios.temperature_sensitivity
            * ((scenarios.temp_c - self.temperature_optimum) / self.temperature_scale) ** 2
        )
        plate_factor = np.exp(-scenarios.plate_penalty * load_ratio * plate_index)
        capture_exponent = (
            scenarios.d_ref
            * (self.q_ref / scenarios.q)
            * temperature_factor
            * stage_score
            * plate_factor
        )
        accumulated_dust_term = scenarios.stage_weights @ t_ratio
        rapping_factor = 1.0 + (
            scenarios.rapping_load_coefficient * load_ratio * accumulated_dust_term
        )
        concentration = (
            1000.0
            * scenarios.c_in
            * np.exp(-capture_exponent)
            * rapping_factor
            * np.exp(scenarios.residual)
        )
        return np.clip(concentration, 1e-6, 1e6)

    def predict_deterministic(
        self,
        policy: np.ndarray,
        temp_c: float | np.ndarray,
        c_in: float | np.ndarray,
        q: float | np.ndarray,
        parameter_scale: dict[str, float] | None = None,
    ) -> np.ndarray:
        scale = parameter_scale or {}
        temp = np.atleast_1d(np.asarray(temp_c, dtype=float))
        cin = np.broadcast_to(np.asarray(c_in, dtype=float), temp.shape)
        flow = np.broadcast_to(np.asarray(q, dtype=float), temp.shape)
        n = len(temp)
        scenarios = EmissionScenarios(
            temp_c=temp,
            c_in=cin,
            q=flow,
            d_ref=np.full(n, self.d_ref * scale.get("d_ref", 1.0)),
            stage_weights=np.tile(self.stage_weights, (n, 1)),
            temperature_sensitivity=np.full(
                n, self.temperature_sensitivity * scale.get("temperature_sensitivity", 1.0)
            ),
            plate_penalty=np.full(n, self.plate_penalty * scale.get("plate_penalty", 1.0)),
            rapping_load_coefficient=np.full(
                n,
                self.rapping_load_coefficient
                * scale.get("rapping_load_coefficient", 1.0),
            ),
            residual=np.zeros(n),
        )
        return self.predict_scenarios(policy, scenarios)

    def dust_load_index(self, policy: np.ndarray, regime_frame: pd.DataFrame) -> float:
        policy = np.asarray(policy, dtype=float)
        load_ratio = (
            regime_frame["C_in_gNm3"].to_numpy(dtype=float)
            * regime_frame["Q_Nm3h"].to_numpy(dtype=float)
            / (self.c_in_ref * self.q_ref)
        )
        weighted_cycle = float(np.dot(self.stage_weights, policy[4:] / self.t_ref))
        return float(np.quantile(load_ratio, 0.95) * weighted_cycle)


def fit_emission_twin(audit: AuditResult, prior: dict[str, Any]) -> EmissionTwin:
    frame = audit.frame
    censor_fit = audit.summary["censored_normal"]
    temp_ref = _derive_temperature_reference(frame, prior)
    c_in_ref = float(frame["C_in_gNm3"].median())
    q_ref = float(frame["Q_Nm3h"].median())
    u_ref = frame[U_COLUMNS].median().to_numpy(dtype=float)
    t_ref = frame[T_COLUMNS].median().to_numpy(dtype=float)
    stage_weights = _derive_stage_weights(u_ref, prior)

    rap_at_reference = 1.0 + float(prior["rapping_load_coefficient"])
    # The dataset only identifies the operating-point removal exponent. Shape
    # parameters are literature-informed and their uncertainty is propagated.
    d_ref = float(
        np.log(1000.0 * c_in_ref * rap_at_reference / float(censor_fit["mu"]))
    )
    valid_output = frame["C_out_mgNm3"].dropna()
    identifiability = pd.DataFrame(
        [
            {
                "quantity": "右删失位置与噪声",
                "evidence": "数据",
                "diagnostic": f"n={len(valid_output)}, censor={audit.summary['output_censored_fraction']:.3%}",
                "status": "可识别",
            },
            {
                "quantity": "工况/操作量对出口浓度的净效应",
                "evidence": "数据",
                "diagnostic": f"OLS R2={audit.summary['ordinary_ols_r2']:.6f}",
                "status": "不可识别",
            },
            {
                "quantity": "删失事件的条件依赖",
                "evidence": "数据",
                "diagnostic": f"linear R2={audit.summary['censor_indicator_linear_r2']:.6f}",
                "status": "不可识别",
            },
            {
                "quantity": "基准去除指数",
                "evidence": "删失锚定+质量守恒",
                "diagnostic": f"D_ref={d_ref:.4f}",
                "status": "弱识别",
            },
            {
                "quantity": "分场权重",
                "evidence": "附件中位电压+D-A电压平方代理",
                "diagnostic": "w=" + np.array2string(stage_weights, precision=4, separator=","),
                "status": "代理识别",
            },
            {
                "quantity": "温度参考中心",
                "evidence": "附件温度均值",
                "diagnostic": f"Theta_ref={temp_ref:.3f} C",
                "status": "可识别",
            },
            {
                "quantity": "温度/积灰/再飞扬系数",
                "evidence": "物理先验",
                "diagnostic": "基准传播并作30%压力测试",
                "status": "先验主导",
            },
        ]
    )
    return EmissionTwin(
        censor_threshold=float(audit.summary["output_max"]),
        censor_mu=float(censor_fit["mu"]),
        censor_sigma=float(censor_fit["sigma"]),
        temp_ref=temp_ref,
        c_in_ref=c_in_ref,
        q_ref=q_ref,
        u_ref=u_ref,
        t_ref=t_ref,
        stage_weights=stage_weights,
        stage_weight_concentration=float(prior["stage_weight_concentration"]),
        temperature_optimum=temp_ref,
        temperature_sensitivity=float(prior["temperature_sensitivity"]),
        temperature_scale=float(prior["temperature_scale_C"]),
        plate_penalty=float(prior["plate_penalty"]),
        rapping_load_coefficient=float(prior["rapping_load_coefficient"]),
        residual_sigma=float(prior["log_residual_sigma"]),
        removal_relative_sd=float(prior["removal_scale_relative_sd"]),
        prior_relative_sd=float(prior["prior_relative_sd"]),
        d_ref=d_ref,
        identifiability=identifiability,
    )
