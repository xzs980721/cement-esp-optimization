from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture


CONDITION_COLUMNS = ["Temp_C", "C_in_gNm3", "Q_Nm3h"]


def _condition_matrix(frame: pd.DataFrame, rolling_window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    smoothed = frame[CONDITION_COLUMNS].rolling(
        rolling_window, center=True, min_periods=1
    ).median()
    transformed = np.column_stack(
        [
            smoothed["Temp_C"].to_numpy(dtype=float),
            np.log(smoothed["C_in_gNm3"].to_numpy(dtype=float)),
            np.log(smoothed["Q_Nm3h"].to_numpy(dtype=float)),
        ]
    )
    median = np.median(transformed, axis=0)
    iqr = np.percentile(transformed, 75, axis=0) - np.percentile(transformed, 25, axis=0)
    scaled = (transformed - median) / iqr
    return scaled, median, iqr


def _viterbi_regularize(cost: np.ndarray, switch_penalty: float) -> np.ndarray:
    n, k = cost.shape
    dp = np.empty((n, k), dtype=float)
    parent = np.empty((n, k), dtype=np.int16)
    dp[0] = cost[0]
    transition = np.full((k, k), switch_penalty, dtype=float)
    np.fill_diagonal(transition, 0.0)
    for t in range(1, n):
        candidates = dp[t - 1][:, None] + transition
        parent[t] = np.argmin(candidates, axis=0)
        dp[t] = cost[t] + np.min(candidates, axis=0)
    labels = np.empty(n, dtype=int)
    labels[-1] = int(np.argmin(dp[-1]))
    for t in range(n - 1, 0, -1):
        labels[t - 1] = int(parent[t, labels[t]])
    return labels


def _segments(labels: np.ndarray) -> list[tuple[int, int, int]]:
    starts = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
    ends = np.r_[starts[1:], len(labels)]
    return [(int(start), int(end), int(labels[start])) for start, end in zip(starts, ends)]


def _merge_short_runs(labels: np.ndarray, cost: np.ndarray, minimum_length: int) -> np.ndarray:
    labels = labels.copy()
    for _ in range(25):
        changed = False
        runs = _segments(labels)
        for index, (start, end, label) in enumerate(runs):
            if end - start >= minimum_length:
                continue
            candidates: list[int] = []
            if index > 0:
                candidates.append(runs[index - 1][2])
            if index + 1 < len(runs):
                candidates.append(runs[index + 1][2])
            if not candidates:
                continue
            replacement = min(candidates, key=lambda c: float(cost[start:end, c].mean()))
            labels[start:end] = replacement
            changed = True
        if not changed:
            break
    return labels


def _dwell_statistics(labels: np.ndarray, k: int) -> dict[int, float]:
    dwell: dict[int, list[int]] = {i: [] for i in range(k)}
    for start, end, label in _segments(labels):
        dwell[label].append(end - start)
    return {i: float(np.median(values)) if values else 0.0 for i, values in dwell.items()}


@dataclass
class RegimeModel:
    labels: np.ndarray
    n_regimes: int
    centers: pd.DataFrame
    model_selection: pd.DataFrame
    transition_matrix: pd.DataFrame
    scaled_features: np.ndarray
    robust_median: np.ndarray
    robust_iqr: np.ndarray
    selected_gmm: GaussianMixture


def segment_regimes(
    frame: pd.DataFrame,
    rolling_window: int = 15,
    k_min: int = 3,
    k_max: int = 8,
    switch_penalty: float = 8.0,
    minimum_share: float = 0.01,
    minimum_median_dwell: int = 15,
    bic_tie_tolerance: float = 10.0,
    seed: int = 2026,
) -> RegimeModel:
    scaled, robust_median, robust_iqr = _condition_matrix(frame, rolling_window)
    candidates: list[dict[str, object]] = []

    rng = np.random.default_rng(seed)
    sample_indices = rng.choice(len(frame), size=min(2500, len(frame)), replace=False)
    for k in range(k_min, k_max + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            n_init=8,
            max_iter=500,
            reg_covar=1e-5,
            random_state=seed + k,
        ).fit(scaled)
        probabilities = np.clip(gmm.predict_proba(scaled), 1e-300, 1.0)
        cost = -np.log(probabilities)
        labels = _viterbi_regularize(cost, switch_penalty=switch_penalty)
        labels = _merge_short_runs(labels, cost, minimum_median_dwell)
        shares = np.bincount(labels, minlength=k) / len(labels)
        dwell = _dwell_statistics(labels, k)
        active = shares > 0
        valid = bool(
            active.all()
            and shares.min() >= minimum_share
            and min(dwell.values()) >= minimum_median_dwell
        )
        try:
            silhouette = float(silhouette_score(scaled[sample_indices], labels[sample_indices]))
        except ValueError:
            silhouette = float("nan")
        candidates.append(
            {
                "k": k,
                "bic": float(gmm.bic(scaled)),
                "silhouette": silhouette,
                "minimum_share": float(shares.min()),
                "minimum_median_dwell": float(min(dwell.values())),
                "transitions": int(np.sum(labels[1:] != labels[:-1])),
                "valid": valid,
                "gmm": gmm,
                "labels": labels,
            }
        )

    valid_candidates = [candidate for candidate in candidates if candidate["valid"]]
    pool = valid_candidates if valid_candidates else candidates
    best_bic = min(float(candidate["bic"]) for candidate in pool)
    near_best = [
        candidate for candidate in pool if float(candidate["bic"]) <= best_bic + bic_tie_tolerance
    ]
    selected = min(near_best, key=lambda candidate: int(candidate["k"]))
    gmm = selected["gmm"]
    labels = np.asarray(selected["labels"], dtype=int)
    k = int(selected["k"])

    # Give regimes stable, interpretable numbers from low to high inlet mass load.
    raw_load = frame["C_in_gNm3"].to_numpy() * frame["Q_Nm3h"].to_numpy()
    load_means = np.array([raw_load[labels == j].mean() for j in range(k)])
    ordering = np.argsort(load_means)
    mapping = {int(old): int(new) for new, old in enumerate(ordering)}
    labels = np.array([mapping[int(label)] for label in labels], dtype=int)

    rows: list[dict[str, float | int | str]] = []
    for regime in range(k):
        subset = frame.loc[labels == regime]
        load = subset["C_in_gNm3"] * subset["Q_Nm3h"]
        run_lengths = [end - start for start, end, lab in _segments(labels) if lab == regime]
        rows.append(
            {
                "regime": regime,
                "label": f"R{regime + 1}",
                "n": int(len(subset)),
                "share": float(len(subset) / len(frame)),
                "Temp_C_mean": float(subset["Temp_C"].mean()),
                "Temp_C_p95": float(subset["Temp_C"].quantile(0.95)),
                "C_in_gNm3_mean": float(subset["C_in_gNm3"].mean()),
                "Q_Nm3h_mean": float(subset["Q_Nm3h"].mean()),
                "mass_load_mean": float(load.mean()),
                "P_history_mean_kW": float(subset["P_total_kW"].mean()),
                "median_dwell_min": float(np.median(run_lengths)),
            }
        )
    centers = pd.DataFrame(rows)

    counts = np.zeros((k, k), dtype=float)
    for left, right in zip(labels[:-1], labels[1:]):
        counts[left, right] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    transition = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0)
    transition_matrix = pd.DataFrame(
        transition,
        index=[f"R{i + 1}" for i in range(k)],
        columns=[f"R{i + 1}" for i in range(k)],
    )

    selection_table = pd.DataFrame(
        [
            {
                key: value
                for key, value in candidate.items()
                if key not in {"gmm", "labels"}
            }
            for candidate in candidates
        ]
    )
    selection_table["selected"] = selection_table["k"] == k
    return RegimeModel(
        labels=labels,
        n_regimes=k,
        centers=centers,
        model_selection=selection_table,
        transition_matrix=transition_matrix,
        scaled_features=scaled,
        robust_median=robust_median,
        robust_iqr=robust_iqr,
        selected_gmm=gmm,
    )

