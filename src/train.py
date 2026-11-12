from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.data import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    load_raw_data,
    prepare_dataset,
)
from src.modeling import QuantileClippedRegressor


def _metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "rmse": round(float(root_mean_squared_error(y_true, y_pred)), 2),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def build_model() -> QuantileClippedRegressor:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=100,
                ),
            ),
        ]
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )
    regressor = Pipeline(
        [
            ("preprocessing", preprocessing),
            ("regressor", Ridge(alpha=10.0, solver="lsqr")),
        ]
    )
    log_target_model = TransformedTargetRegressor(
        regressor=regressor, func=np.log1p, inverse_func=np.expm1
    )
    return QuantileClippedRegressor(
        estimator=log_target_model,
        lower_quantile=0.01,
        upper_quantile=0.99,
    )


def build_hist_model() -> QuantileClippedRegressor:
    """A compact nonlinear model used as the main model in the project."""
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    max_categories=100,
                ),
            ),
        ]
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )
    categorical_mask = [False] * len(NUMERIC_FEATURES) + [True] * len(
        CATEGORICAL_FEATURES
    )
    regressor = Pipeline(
        [
            ("preprocessing", preprocessing),
            (
                "regressor",
                HistGradientBoostingRegressor(
                    learning_rate=0.08,
                    max_iter=150,
                    max_leaf_nodes=31,
                    l2_regularization=1.0,
                    categorical_features=categorical_mask,
                    random_state=42,
                ),
            ),
        ]
    )
    log_target_model = TransformedTargetRegressor(
        regressor=regressor, func=np.log1p, inverse_func=np.expm1
    )
    return QuantileClippedRegressor(
        estimator=log_target_model,
        lower_quantile=0.01,
        upper_quantile=0.99,
    )


def build_random_forest_model() -> QuantileClippedRegressor:
    """Tree bagging model used as the third, non-linear experiment."""
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    max_categories=100,
                ),
            ),
        ]
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )
    regressor = Pipeline(
        [
            ("preprocessing", preprocessing),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=60,
                    max_depth=18,
                    min_samples_leaf=3,
                    max_features=0.8,
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )
    log_target_model = TransformedTargetRegressor(
        regressor=regressor, func=np.log1p, inverse_func=np.expm1
    )
    return QuantileClippedRegressor(
        estimator=log_target_model,
        lower_quantile=0.01,
        upper_quantile=0.99,
    )


def train(data_path: Path, output_dir: Path) -> dict:
    raw = load_raw_data(data_path)
    data, audit = prepare_dataset(raw)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(
        splitter.split(data, data["target"], groups=data["listing_group"])
    )
    train_data = data.iloc[train_idx]
    test_data = data.iloc[test_idx]

    x_train = train_data[MODEL_FEATURES]
    y_train = train_data["target"]
    x_test = test_data[MODEL_FEATURES]
    y_test = test_data["target"]

    baseline = DummyRegressor(strategy="median").fit(x_train, y_train)
    baseline_metrics = _metrics(y_test, baseline.predict(x_test))

    model = build_model().fit(x_train, y_train)
    model_metrics = _metrics(y_test, model.predict(x_test))

    random_forest = build_random_forest_model().fit(x_train, y_train)
    random_forest_metrics = _metrics(y_test, random_forest.predict(x_test))

    hist_model = build_hist_model().fit(x_train, y_train)
    hist_metrics = _metrics(y_test, hist_model.predict(x_test))

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "ridge_model.joblib")
    joblib.dump(random_forest, output_dir / "random_forest_model.joblib")
    joblib.dump(hist_model, output_dir / "hist_model.joblib")

    report = {
        "audit": audit,
        "split": {"train_rows": len(train_data), "test_rows": len(test_data)},
        "baseline": baseline_metrics,
        "ridge_log_target_clipped": model_metrics,
        "random_forest": random_forest_metrics,
        "hist_gradient_boosting": hist_metrics,
        "features": MODEL_FEATURES,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the first house-price model.")
    parser.add_argument("--data", type=Path, default=Path("data.csv.zip"))
    parser.add_argument("--output", type=Path, default=Path("models"))
    args = parser.parse_args()
    print(json.dumps(train(args.data, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
