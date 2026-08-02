from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_audit import load_and_audit_data
from src.regimes import segment_regimes


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit smoothing-window and transition-penalty settings.")
    parser.add_argument("--config", default="config/model.yaml")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    root = config_path.parent.parent
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    audit = load_and_audit_data(
        (root / config["project"]["input_csv"]).resolve(),
        censor_threshold=float(config["data"]["censor_threshold_mgNm3"]),
    )
    base = config["regimes"]
    rows: list[dict[str, float | int | str]] = []

    def evaluate(audit_name: str, setting: float, window: int, penalty: float) -> None:
        model = segment_regimes(
            audit.frame,
            rolling_window=window,
            k_min=8,
            k_max=8,
            switch_penalty=penalty,
            minimum_share=float(base["minimum_share"]),
            minimum_median_dwell=int(base["minimum_median_dwell_minutes"]),
            selection_metric="silhouette",
            seed=int(config["project"]["seed"]),
        )
        row = model.model_selection.iloc[0]
        rows.append(
            {
                "audit": audit_name,
                "setting": setting,
                "silhouette": float(row["silhouette"]),
                "minimum_share": float(row["minimum_share"]),
                "minimum_median_dwell": float(row["minimum_median_dwell"]),
                "transitions": int(row["transitions"]),
            }
        )

    for window in (10, 15, 20):
        evaluate("rolling_window_minutes", window, window, float(base["switch_penalty"]))
    for penalty in (4.0, 8.0, 12.0):
        evaluate(
            "switch_penalty",
            penalty,
            int(config["data"]["rolling_window_minutes"]),
            penalty,
        )

    output = root / "outputs/tables/regime_regularization_sensitivity.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(output)


if __name__ == "__main__":
    main()
