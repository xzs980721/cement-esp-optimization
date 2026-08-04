from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from .emission_twin import EmissionScenarios, EmissionTwin, T_COLUMNS, U_COLUMNS


@dataclass
class UncensoredDistribution:
    """Discrete distribution of outlet readings strictly below the cap."""

    threshold: float
    resolution: float
    support_steps: np.ndarray
    counts: np.ndarray
    dirichlet_alpha: float
    observations: np.ndarray

    @property
    def values(self) -> np.ndarray:
        return self.threshold - self.resolution * self.support_steps

    @property
    def posterior_parameters(self) -> np.ndarray:
        return self.counts.astype(float) + self.dirichlet_alpha

    @property
    def probabilities(self) -> np.ndarray:
        parameters = self.posterior_parameters
        return parameters / parameters.sum()

    @property
    def point_prediction(self) -> float:
        return float(np.mean(self.observations))

    def sample(
        self,
        n_scenarios: int,
        seed: int,
        draw_distribution: bool = True,
    ) -> np.ndarray:
        if n_scenarios <= 0:
            raise ValueError("n_scenarios must be positive.")
        rng = np.random.default_rng(seed)
        probabilities = (
            rng.dirichlet(self.posterior_parameters)
            if draw_distribution
            else self.probabilities
        )
        indices = rng.choice(
            len(self.support_steps), size=n_scenarios, replace=True, p=probabilities
        )
        return self.values[indices]

    def probability_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "gap_step": self.support_steps,
                "gap_mgNm3": self.resolution * self.support_steps,
                "outlet_mgNm3": self.values,
                "count": self.counts,
                "empirical_probability": self.counts / self.counts.sum(),
                "posterior_mean_probability": self.probabilities,
            }
        )

    def summary_table(self) -> pd.DataFrame:
        probabilities = [0.005, 0.025, 0.05, 0.25, 0.50, 0.75, 0.95, 0.975, 0.995]
        rows: list[dict[str, float | str]] = [
            {"metric": "n_uncensored", "value": float(len(self.observations))},
            {"metric": "mean_mgNm3", "value": float(np.mean(self.observations))},
            {"metric": "sd_mgNm3", "value": float(np.std(self.observations, ddof=1))},
            {"metric": "min_mgNm3", "value": float(np.min(self.observations))},
            {"metric": "max_mgNm3", "value": float(np.max(self.observations))},
        ]
        rows.extend(
            {
                "metric": f"q{100 * probability:g}_mgNm3",
                "value": float(np.quantile(self.observations, probability)),
            }
            for probability in probabilities
        )
        return pd.DataFrame(rows)


@dataclass
class CalibratedScenarios:
    """Scenario pack with an empirical baseline and a physical response ratio."""

    baseline: np.ndarray
    physical_scenarios: EmissionScenarios
    reference_concentration: np.ndarray
    reference_policy: np.ndarray
    reference_conditions: dict[str, float]

    def predict(self, twin: EmissionTwin, policy: np.ndarray) -> np.ndarray:
        numerator = twin.predict_scenarios(policy, self.physical_scenarios)
        ratio = numerator / np.maximum(self.reference_concentration, 1e-12)
        return np.clip(self.baseline * ratio, 1e-6, 1e6)


def fit_uncensored_distribution(
    frame: pd.DataFrame,
    threshold: float = 50.0,
    resolution: float = 0.01,
    dirichlet_alpha: float = 0.5,
) -> UncensoredDistribution:
    """Fit the discrete empirical distribution of readings below ``threshold``."""

    if resolution <= 0:
        raise ValueError("resolution must be positive.")
    if dirichlet_alpha <= 0:
        raise ValueError("dirichlet_alpha must be positive.")
    values = frame.loc[
        frame["C_out_mgNm3"].notna() & (frame["C_out_mgNm3"] < threshold),
        "C_out_mgNm3",
    ].to_numpy(dtype=float)
    if len(values) == 0:
        raise ValueError("No uncensored outlet observations were found.")
    steps = np.rint((threshold - values) / resolution).astype(int)
    if np.any(steps <= 0):
        raise ValueError("All fitted observations must lie strictly below the threshold.")
    support_steps = np.arange(1, int(steps.max()) + 1, dtype=int)
    counts = np.bincount(steps, minlength=len(support_steps) + 1)[1:]
    return UncensoredDistribution(
        threshold=float(threshold),
        resolution=float(resolution),
        support_steps=support_steps,
        counts=counts,
        dirichlet_alpha=float(dirichlet_alpha),
        observations=values,
    )


def evaluate_distribution_fit(
    frame: pd.DataFrame,
    threshold: float = 50.0,
    resolution: float = 0.01,
    dirichlet_alpha: float = 0.5,
) -> pd.DataFrame:
    """Leave one day out and evaluate distributional, not pointwise, fit."""

    usable = frame.loc[
        frame["C_out_mgNm3"].notna() & (frame["C_out_mgNm3"] < threshold)
    ].copy()
    usable["date"] = usable["timestamp"].dt.date
    rows: list[dict[str, float | str]] = []
    for date in sorted(usable["date"].unique()):
        train = usable.loc[usable["date"] != date]
        test = usable.loc[usable["date"] == date, "C_out_mgNm3"].to_numpy(dtype=float)
        fitted = fit_uncensored_distribution(
            train,
            threshold=threshold,
            resolution=resolution,
            dirichlet_alpha=dirichlet_alpha,
        )
        train_values = fitted.observations
        lower, upper = np.quantile(train_values, [0.025, 0.975])
        ks = ks_2samp(train_values, test)
        rows.append(
            {
                "date": str(date),
                "n_train": int(len(train_values)),
                "n_test": int(len(test)),
                "train_mean_mgNm3": float(np.mean(train_values)),
                "test_mean_mgNm3": float(np.mean(test)),
                "absolute_mean_error_mgNm3": float(
                    abs(np.mean(test) - np.mean(train_values))
                ),
                "ks_distance": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
                "wasserstein_mgNm3": float(wasserstein_distance(train_values, test)),
                "interval_lower_mgNm3": float(lower),
                "interval_upper_mgNm3": float(upper),
                "interval_coverage": float(np.mean((test >= lower) & (test <= upper))),
            }
        )
    return pd.DataFrame(rows)


def regime_reference(
    regime_frame: pd.DataFrame,
) -> tuple[dict[str, float], np.ndarray]:
    if regime_frame.empty:
        raise ValueError("regime_frame cannot be empty.")
    conditions = {
        "temp_c": float(regime_frame["Temp_C"].median()),
        "c_in": float(regime_frame["C_in_gNm3"].median()),
        "q": float(regime_frame["Q_Nm3h"].median()),
    }
    policy = regime_frame[U_COLUMNS + T_COLUMNS].median().to_numpy(dtype=float)
    return conditions, policy


def _reference_scenarios(
    scenarios: EmissionScenarios,
    reference_conditions: dict[str, float],
) -> EmissionScenarios:
    n = len(scenarios.temp_c)
    return replace(
        scenarios,
        temp_c=np.full(n, reference_conditions["temp_c"], dtype=float),
        c_in=np.full(n, reference_conditions["c_in"], dtype=float),
        q=np.full(n, reference_conditions["q"], dtype=float),
        residual=np.zeros(n, dtype=float),
    )


def build_calibrated_scenarios(
    twin: EmissionTwin,
    distribution: UncensoredDistribution,
    regime_frame: pd.DataFrame,
    reference_policy: np.ndarray | None,
    n_scenarios: int,
    seed: int,
    prior_scale: float = 1.0,
) -> CalibratedScenarios:
    conditions, historical_reference = regime_reference(regime_frame)
    reference = (
        historical_reference
        if reference_policy is None
        else np.asarray(reference_policy, dtype=float)
    )
    if reference.shape != (8,):
        raise ValueError("reference_policy must contain U1..U4,T1..T4.")
    physical = twin.scenario_pack(
        regime_frame, n_scenarios=n_scenarios, seed=seed, prior_scale=prior_scale
    )
    physical = replace(physical, residual=np.zeros(n_scenarios, dtype=float))
    reference_pack = _reference_scenarios(physical, conditions)
    denominator = twin.predict_scenarios(reference, reference_pack)
    baseline = distribution.sample(
        n_scenarios=n_scenarios,
        seed=seed + 500_003,
        draw_distribution=True,
    )
    return CalibratedScenarios(
        baseline=baseline,
        physical_scenarios=physical,
        reference_concentration=denominator,
        reference_policy=reference.copy(),
        reference_conditions=conditions,
    )


def relative_physical_response(
    twin: EmissionTwin,
    policy: np.ndarray,
    conditions: dict[str, float],
    reference_policy: np.ndarray,
    reference_conditions: dict[str, float],
    parameter_scale: dict[str, float] | None = None,
) -> float:
    numerator = twin.predict_deterministic(
        np.asarray(policy, dtype=float),
        conditions["temp_c"],
        conditions["c_in"],
        conditions["q"],
        parameter_scale=parameter_scale,
    )[0]
    denominator = twin.predict_deterministic(
        np.asarray(reference_policy, dtype=float),
        reference_conditions["temp_c"],
        reference_conditions["c_in"],
        reference_conditions["q"],
        parameter_scale=parameter_scale,
    )[0]
    return float(numerator / max(float(denominator), 1e-12))


def sample_calibrated_emissions(
    twin: EmissionTwin,
    distribution: UncensoredDistribution,
    regime_frame: pd.DataFrame,
    policy: np.ndarray,
    reference_policy: np.ndarray | None = None,
    n_scenarios: int = 10_000,
    seed: int = 2026,
    prior_scale: float = 1.0,
) -> np.ndarray:
    scenarios = build_calibrated_scenarios(
        twin=twin,
        distribution=distribution,
        regime_frame=regime_frame,
        reference_policy=reference_policy,
        n_scenarios=n_scenarios,
        seed=seed,
        prior_scale=prior_scale,
    )
    return scenarios.predict(twin, np.asarray(policy, dtype=float))
