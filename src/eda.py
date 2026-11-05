from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kruskal, spearmanr

from src.data import MODEL_FEATURES, load_raw_data, prepare_dataset


def build_eda(data_path: Path, output_dir: Path) -> dict:
    """Build a small set of figures and statistics for the project report."""
    data, audit = prepare_dataset(load_raw_data(data_path))
    output_dir.mkdir(parents=True, exist_ok=True)

    price_quantiles = data["target"].quantile([0.01, 0.5, 0.9, 0.99])
    missing = (
        data[MODEL_FEATURES].isna().mean().mul(100).sort_values(ascending=False)
    )

    plt.figure(figsize=(9, 5))
    data["target"].clip(upper=price_quantiles.loc[0.99]).hist(bins=50)
    plt.title("Распределение стоимости недвижимости (до 99-го перцентиля)")
    plt.xlabel("Стоимость, долларов")
    plt.ylabel("Количество объявлений")
    plt.tight_layout()
    plt.savefig(output_dir / "price_distribution.png", dpi=130)
    plt.close()

    plt.figure(figsize=(9, 5))
    missing[missing > 0].sort_values().plot.barh()
    plt.title("Доля пропусков после очистки")
    plt.xlabel("Пропуски, %")
    plt.tight_layout()
    plt.savefig(output_dir / "missing_values.png", dpi=130)
    plt.close()

    sample = data.dropna(subset=["sqft"]).sample(
        n=min(8_000, data["sqft"].notna().sum()), random_state=42
    )
    plt.figure(figsize=(9, 5))
    plt.scatter(sample["sqft"], sample["target"], alpha=0.15, s=8)
    plt.yscale("log")
    plt.title("Связь площади и стоимости")
    plt.xlabel("Площадь, sqft")
    plt.ylabel("Стоимость, долларов (логарифмическая шкала)")
    plt.tight_layout()
    plt.savefig(output_dir / "price_vs_sqft.png", dpi=130)
    plt.close()

    top_states = data["state"].value_counts().head(10).index
    state_prices = (
        data.loc[data["state"].isin(top_states)]
        .groupby("state")["target"]
        .median()
        .sort_values()
    )
    plt.figure(figsize=(9, 5))
    state_prices.plot.barh()
    plt.title("Медианная стоимость в штатах с наибольшим числом объявлений")
    plt.xlabel("Медианная стоимость, долларов")
    plt.tight_layout()
    plt.savefig(output_dir / "median_price_by_state.png", dpi=130)
    plt.close()

    sqft_price = data[["sqft", "target"]].dropna()
    spearman_stat, spearman_pvalue = spearmanr(
        sqft_price["sqft"], sqft_price["target"]
    )
    state_groups = [
        group["target"].to_numpy()
        for _, group in data.loc[data["state"].isin(top_states)].groupby("state")
    ]
    kruskal_stat, kruskal_pvalue = kruskal(*state_groups)
    state_sample_size = sum(len(group) for group in state_groups)
    state_effect_size = (kruskal_stat - len(state_groups) + 1) / (
        state_sample_size - len(state_groups)
    )

    summary = {
        "audit": audit,
        "target": {
            "mean": round(float(data["target"].mean()), 2),
            "median": round(float(data["target"].median()), 2),
            "p90": round(float(price_quantiles.loc[0.9]), 2),
            "p99": round(float(price_quantiles.loc[0.99]), 2),
        },
        "sqft_target_spearman": round(
            float(data[["sqft", "target"]].corr(method="spearman").iloc[0, 1]), 3
        ),
        "missing_percent": {
            key: round(float(value), 2)
            for key, value in missing.items()
            if value > 0
        },
        "median_price_by_top_state": {
            key: round(float(value), 2) for key, value in state_prices.items()
        },
        "statistical_tests": {
            "sqft_price_spearman": {
                "statistic": round(float(spearman_stat), 3),
                "pvalue": float(spearman_pvalue),
                "sample_size": len(sqft_price),
            },
            "price_by_state_kruskal": {
                "statistic": round(float(kruskal_stat), 2),
                "pvalue": float(kruskal_pvalue),
                "effect_size_epsilon_squared": round(float(state_effect_size), 3),
                "sample_size": state_sample_size,
            },
        },
    }
    (output_dir / "eda_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    result = build_eda(Path("data.csv.zip"), Path("reports"))
    print(json.dumps(result, ensure_ascii=False, indent=2))
