from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import MODEL_FEATURES, load_raw_data, prepare_dataset


def evaluate_errors(
    data_path: Path, model_path: Path, output_dir: Path
) -> dict[str, object]:
    """Evaluate the saved final model on the same address-based test split."""
    data, _ = prepare_dataset(load_raw_data(data_path))
    _, test_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(
            data, data["target"], groups=data["listing_group"]
        )
    )
    test = data.iloc[test_idx].copy()
    model = joblib.load(model_path)
    test["prediction"] = model.predict(test[MODEL_FEATURES])
    test["absolute_error"] = (test["target"] - test["prediction"]).abs()

    bins = [0, 200_000, 500_000, 1_000_000, np.inf]
    labels = ["до 200 тыс.", "200–500 тыс.", "500 тыс.–1 млн", "от 1 млн"]
    test["price_segment"] = pd.cut(
        test["target"], bins=bins, labels=labels, include_lowest=True
    )
    segment_mae = (
        test.groupby("price_segment", observed=True)
        .apply(
            lambda frame: mean_absolute_error(
                frame["target"], frame["prediction"]
            ),
            include_groups=False,
        )
        .round(2)
    )

    sample = test.sample(n=min(8_000, len(test)), random_state=42)
    lower = min(sample["target"].min(), sample["prediction"].min())
    upper = max(sample["target"].max(), sample["prediction"].max())
    plt.figure(figsize=(7, 6))
    plt.scatter(sample["target"], sample["prediction"], alpha=0.15, s=8)
    plt.plot([lower, upper], [lower, upper], linestyle="--", color="black")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Фактическая и предсказанная стоимость")
    plt.xlabel("Фактическая стоимость, долларов")
    plt.ylabel("Прогноз, долларов")
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "actual_vs_predicted.png", dpi=130)
    plt.close()

    report = {
        "test_rows": len(test),
        "median_absolute_error": round(float(test["absolute_error"].median()), 2),
        "mae_by_price_segment": {
            str(key): float(value) for key, value in segment_mae.items()
        },
    }
    (output_dir / "error_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    result = evaluate_errors(
        Path("data.csv.zip"),
        Path("models/hist_model.joblib"),
        Path("reports"),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
