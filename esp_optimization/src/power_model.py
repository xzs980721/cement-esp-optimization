from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


U_COLUMNS = [f"U{i}_kV" for i in range(1, 5)]
T_COLUMNS = [f"T{i}_s" for i in range(1, 5)]
BASE_FEATURES = [f"U{i}_sq" for i in range(1, 5)] + [f"inv_T{i}" for i in range(1, 5)]
EXTENDED_FEATURES = BASE_FEATURES + ["Temp_C", "Q_scaled"]


def _power_features(frame: pd.DataFrame, extended: bool = False) -> pd.DataFrame:
    data = {f"U{i}_sq": frame[f"U{i}_kV"].to_numpy(dtype=float) ** 2 for i in range(1, 5)}
    data.update({f"inv_T{i}": 1.0 / frame[f"T{i}_s"].to_numpy(dtype=float) for i in range(1, 5)})
    if extended:
        data["Temp_C"] = frame["Temp_C"].to_numpy(dtype=float)
        data["Q_scaled"] = frame["Q_Nm3h"].to_numpy(dtype=float) / 100000.0
    return pd.DataFrame(data, index=frame.index)


@dataclass
class PowerSurrogate:
    feature_names: list[str]
    scaler: StandardScaler
    estimator: HuberRegressor
    raw_intercept: float
    raw_coefficients: np.ndarray
    cv_metrics: pd.DataFrame
    selected_model: str

    def predict_frame(self, frame: pd.DataFrame) -> np.ndarray:
        extended = self.selected_model == "extended"
        features = _power_features(frame, extended=extended)[self.feature_names]
        return self.estimator.predict(self.scaler.transform(features))

    def predict_policy(self, policy: np.ndarray) -> float:
        policy = np.asarray(policy, dtype=float)
        if policy.shape != (8,):
            raise ValueError("Policy must contain U1..U4,T1..T4.")
        values = np.concatenate([policy[:4] ** 2, 1.0 / policy[4:]])
        if self.selected_model == "extended":
            raise ValueError("Extended power model requires a condition frame.")
        return float(self.raw_intercept + values @ self.raw_coefficients)

    def coefficient_table(self) -> pd.DataFrame:
        return pd.DataFrame({"feature": self.feature_names, "coefficient": self.raw_coefficients})


def _fit_huber(features: pd.DataFrame, target: np.ndarray) -> tuple[StandardScaler, HuberRegressor]:
    scaler = StandardScaler().fit(features)
    estimator = HuberRegressor(epsilon=1.35, alpha=1e-6, max_iter=1000).fit(
        scaler.transform(features), target
    )
    return scaler, estimator


def _cross_validate(frame: pd.DataFrame, extended: bool) -> pd.DataFrame:
    metrics: list[dict[str, float | str]] = []
    days = frame["timestamp"].dt.date
    for day in sorted(days.unique()):
        train = days != day
        test = days == day
        x_train = _power_features(frame.loc[train], extended=extended)
        x_test = _power_features(frame.loc[test], extended=extended)
        y_train = frame.loc[train, "P_total_kW"].to_numpy(dtype=float)
        y_test = frame.loc[test, "P_total_kW"].to_numpy(dtype=float)
        scaler, model = _fit_huber(x_train, y_train)
        prediction = model.predict(scaler.transform(x_test))
        metrics.append(
            {
                "held_out_day": str(day),
                "r2": r2_score(y_test, prediction),
                "mae_kW": mean_absolute_error(y_test, prediction),
                "rmse_kW": float(np.sqrt(mean_squared_error(y_test, prediction))),
            }
        )
    return pd.DataFrame(metrics)


def fit_power_surrogate(frame: pd.DataFrame) -> PowerSurrogate:
    base_cv = _cross_validate(frame, extended=False)
    extended_cv = _cross_validate(frame, extended=True)
    base_rmse = float(base_cv["rmse_kW"].mean())
    extended_rmse = float(extended_cv["rmse_kW"].mean())
    use_extended = extended_rmse < 0.98 * base_rmse

    # The optimizer needs a condition-independent objective. If the extended model
    # does not materially improve out-of-day prediction, the physical base form wins.
    if use_extended:
        selected = "extended"
        features = _power_features(frame, extended=True)
        cv = extended_cv
    else:
        selected = "base"
        features = _power_features(frame, extended=False)
        cv = base_cv

    scaler, estimator = _fit_huber(features, frame["P_total_kW"].to_numpy(dtype=float))
    raw_coefficients = estimator.coef_ / scaler.scale_
    raw_intercept = float(estimator.intercept_ - np.dot(raw_coefficients, scaler.mean_))

    # Fall back to the base model for optimization if a tiny residual condition term
    # happens to pass the threshold. Its coefficients are reported separately.
    if selected == "extended":
        base_features = _power_features(frame, extended=False)
        scaler, estimator = _fit_huber(base_features, frame["P_total_kW"].to_numpy(dtype=float))
        raw_coefficients = estimator.coef_ / scaler.scale_
        raw_intercept = float(estimator.intercept_ - np.dot(raw_coefficients, scaler.mean_))
        features = base_features
        selected = "base"
        cv = base_cv

    return PowerSurrogate(
        feature_names=list(features.columns),
        scaler=scaler,
        estimator=estimator,
        raw_intercept=raw_intercept,
        raw_coefficients=np.asarray(raw_coefficients, dtype=float),
        cv_metrics=cv,
        selected_model=selected,
    )

