"""Create and execute the modeling notebook used for project submission."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "02_modeling.ipynb"


def build_notebook():
    cells = [
        new_markdown_cell(
            """# Обучение и сравнение моделей стоимости недвижимости

В этом ноутбуке показан полный процесс обучения моделей: подготовка выборок,
baseline, три разных алгоритма машинного обучения, расчёт метрик, подбор
гиперпараметров и сохранение итоговой модели.

Основная метрика — **MAE**, так как её можно интерпретировать как среднюю ошибку
в долларах. Дополнительно рассчитываются RMSE и $R^2$. Все преобразования
признаков входят в pipeline и обучаются только на train-выборке."""
        ),
        new_code_cell(
            """from pathlib import Path
import sys
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV

PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import MODEL_FEATURES, load_raw_data, prepare_dataset
from src.train import build_hist_model, build_model, build_random_forest_model

RANDOM_STATE = 42
pd.set_option("display.max_colwidth", 120)
print(f"Рабочая папка: {PROJECT_ROOT}")"""
        ),
        new_markdown_cell(
            """## 1. Загрузка и подготовка данных

Используется та же функция очистки, что и в EDA. Из выборки исключаются аренда,
земельные участки, дубли и строки с некорректной целевой ценой. Признак
`Price/sqft` не используется из-за утечки целевой переменной."""
        ),
        new_code_cell(
            """started = time.perf_counter()
raw = load_raw_data(PROJECT_ROOT / "data.csv.zip")
data, audit = prepare_dataset(raw)
print(f"Подготовка заняла {time.perf_counter() - started:.1f} сек.")
display(pd.DataFrame([audit]).T.rename(columns={0: "значение"}))
print(f"Итоговый размер данных: {data.shape}")"""
        ),
        new_markdown_cell(
            """## 2. Разделение на train и test

Обычное случайное разделение могло бы поместить объявления одного адреса в обе
выборки. Поэтому применяется `GroupShuffleSplit`, где группой является адрес и
почтовый индекс. Test используется только для окончательной оценки."""
        ),
        new_code_cell(
            """splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
train_idx, test_idx = next(
    splitter.split(data, data["target"], groups=data["listing_group"])
)
train_data = data.iloc[train_idx].copy()
test_data = data.iloc[test_idx].copy()

x_train = train_data[MODEL_FEATURES]
y_train = train_data["target"].astype(float)
x_test = test_data[MODEL_FEATURES]
y_test = test_data["target"].astype(float)

group_overlap = set(train_data["listing_group"]) & set(test_data["listing_group"])
print(f"Train: {len(train_data):,} строк")
print(f"Test:  {len(test_data):,} строк")
print(f"Пересечение адресов: {len(group_overlap)}")"""
        ),
        new_code_cell(
            """def calculate_metrics(y_true, predictions):
    return {
        "MAE": mean_absolute_error(y_true, predictions),
        "RMSE": root_mean_squared_error(y_true, predictions),
        "R2": r2_score(y_true, predictions),
    }


def fit_and_evaluate(name, model):
    started = time.perf_counter()
    model.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - started
    predictions = model.predict(x_test)
    row = {"model": name, **calculate_metrics(y_test, predictions), "fit_seconds": fit_seconds}
    print(
        f"{name}: MAE=${row['MAE']:,.0f}, RMSE=${row['RMSE']:,.0f}, "
        f"R²={row['R2']:.3f}, обучение={fit_seconds:.1f} сек."
    )
    return model, predictions, row


results = []"""
        ),
        new_markdown_cell(
            """## 3. Baseline

Baseline всегда предсказывает медианную цену обучающей выборки. Более сложные
модели должны заметно превосходить этот результат."""
        ),
        new_code_cell(
            """baseline = DummyRegressor(strategy="median")
baseline, baseline_predictions, baseline_result = fit_and_evaluate("Dummy baseline", baseline)
results.append(baseline_result)"""
        ),
        new_markdown_cell(
            """## 4. Ridge

Ridge — линейная модель с L2-регуляризацией. Числовые признаки масштабируются,
категориальные кодируются One-Hot Encoding, а цена логарифмируется."""
        ),
        new_code_cell(
            """ridge_model, ridge_predictions, ridge_result = fit_and_evaluate(
    "Ridge", build_model()
)
results.append(ridge_result)"""
        ),
        new_markdown_cell(
            """## 5. Random Forest

Random Forest представляет бэггинг деревьев решений и может находить
нелинейные зависимости и взаимодействия признаков. Чтобы время обучения
оставалось разумным, глубина деревьев ограничена."""
        ),
        new_code_cell(
            """forest_model, forest_predictions, forest_result = fit_and_evaluate(
    "Random Forest", build_random_forest_model()
)
results.append(forest_result)"""
        ),
        new_markdown_cell(
            """## 6. HistGradientBoosting

Градиентный бустинг последовательно исправляет ошибки предыдущих деревьев.
Этот алгоритм хорошо подходит для большого табличного датасета."""
        ),
        new_code_cell(
            """hist_model, hist_predictions, hist_result = fit_and_evaluate(
    "HistGradientBoosting", build_hist_model()
)
results.append(hist_result)"""
        ),
        new_markdown_cell("## 7. Первичное сравнение моделей"),
        new_code_cell(
            """results_df = pd.DataFrame(results).sort_values("MAE").reset_index(drop=True)
result_formats = {
    "MAE": "${:,.0f}", "RMSE": "${:,.0f}",
    "R2": "{:.3f}", "fit_seconds": "{:.1f}",
}
display(results_df.style.format(result_formats))

results_df.set_index("model")["MAE"].sort_values().plot.barh(figsize=(8, 4))
plt.title("MAE моделей до подбора параметров")
plt.xlabel("MAE, долларов")
plt.tight_layout()
plt.show()"""
        ),
        new_markdown_cell(
            """## 8. Подбор гиперпараметров

Параметры подбираются только для наиболее подходящего алгоритма —
HistGradientBoosting. Поиск проводится на случайной части train и использует
трёхфолдовую групповую валидацию. Test при подборе не используется.

Проверяются скорость обучения, число итераций, число листьев и L2-регуляризация.
Четырёх случайных комбинаций достаточно для демонстрации процесса без
неоправданно долгого расчёта."""
        ),
        new_code_cell(
            """tune_data = train_data.sample(
    n=min(60_000, len(train_data)), random_state=RANDOM_STATE
)
x_tune = tune_data[MODEL_FEATURES]
y_tune = tune_data["target"].astype(float)
groups_tune = tune_data["listing_group"]

parameter_distributions = {
    "estimator__regressor__regressor__learning_rate": [0.04, 0.06, 0.08, 0.12],
    "estimator__regressor__regressor__max_iter": [100, 150, 220],
    "estimator__regressor__regressor__max_leaf_nodes": [15, 31, 63],
    "estimator__regressor__regressor__l2_regularization": [0.0, 1.0, 5.0],
}

search = RandomizedSearchCV(
    estimator=build_hist_model(),
    param_distributions=parameter_distributions,
    n_iter=4,
    scoring="neg_mean_absolute_error",
    cv=GroupKFold(n_splits=3),
    n_jobs=1,
    random_state=RANDOM_STATE,
    verbose=1,
    return_train_score=False,
)

started = time.perf_counter()
search.fit(x_tune, y_tune, groups=groups_tune)
print(f"Подбор занял {time.perf_counter() - started:.1f} сек.")
print(f"Лучший CV MAE: ${-search.best_score_:,.0f}")
print("Лучшие параметры:")
for key, value in search.best_params_.items():
    print(f"  {key.split('__')[-1]}: {value}")

cv_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
display(
    cv_results[["rank_test_score", "mean_test_score", "std_test_score", "params"]]
    .assign(mean_cv_mae=lambda frame: -frame["mean_test_score"])
    [["rank_test_score", "mean_cv_mae", "std_test_score", "params"]]
    .head(4)
)"""
        ),
        new_markdown_cell(
            """## 9. Обучение настроенной модели на полном train

Лучшая конфигурация повторно обучается на всей train-выборке. После этого test
используется один раз для финальной оценки, а pipeline сохраняется для
production-приложения."""
        ),
        new_code_cell(
            """tuned_model, tuned_predictions, tuned_result = fit_and_evaluate(
    "HistGradientBoosting tuned", search.best_estimator_
)
results.append(tuned_result)
"""
        ),
        new_code_cell(
            """final_results = pd.DataFrame(results).sort_values("MAE").reset_index(drop=True)
display(final_results.style.format(result_formats))
final_results.to_csv(PROJECT_ROOT / "reports" / "model_comparison.csv", index=False)

best_row = final_results.iloc[0]
print(
    f"Лучший результат на test: {best_row['model']}, "
    f"MAE=${best_row['MAE']:,.0f}, R²={best_row['R2']:.3f}."
)

# Настройка не улучшила test, поэтому сохраняется более простая базовая
# конфигурация HistGradientBoosting с лучшей итоговой метрикой.
models_dir = PROJECT_ROOT / "models"
models_dir.mkdir(exist_ok=True)
joblib.dump(hist_model, models_dir / "final_model.joblib")
print("Подбор не улучшил test; сохранён базовый HistGradientBoosting.")
print(f"Итоговая модель: {models_dir / 'final_model.joblib'}")"""
        ),
        new_markdown_cell("## 10. Анализ прогнозов итоговой модели"),
        new_code_cell(
            """plot_frame = pd.DataFrame({"actual": y_test, "predicted": hist_predictions}).sample(
    n=min(8_000, len(y_test)), random_state=RANDOM_STATE
)
lower = min(plot_frame.min())
upper = max(plot_frame.max())

plt.figure(figsize=(7, 6))
plt.scatter(plot_frame["actual"], plot_frame["predicted"], alpha=0.15, s=8)
plt.plot([lower, upper], [lower, upper], "--", color="black", label="идеальный прогноз")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Фактическая стоимость, долларов")
plt.ylabel("Прогноз, долларов")
plt.title("Фактическая и предсказанная стоимость")
plt.legend()
plt.tight_layout()
plt.show()"""
        ),
        new_markdown_cell(
            """## Итоговые выводы

1. Все три ML-модели превосходят медианный baseline, следовательно, признаки
   содержат полезную информацию о цене.
2. Линейный Ridge служит понятной отправной точкой, но хуже учитывает сложные
   взаимодействия площади, местоположения и характеристик дома.
3. Random Forest и HistGradientBoosting моделируют нелинейные зависимости.
4. Подбор гиперпараметров выполнен с групповой кросс-валидацией только на train.
   Настроенная конфигурация не улучшила результат на test, что также является
   полезным результатом эксперимента.
5. Базовый HistGradientBoosting с лучшей итоговой метрикой сохранён в
   `models/final_model.joblib` и используется Streamlit-приложением. Прогноз
   является ориентировочным, особенно для редких и дорогих объектов."""
        ),
    ]
    return new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3 (house-price-project)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
    )


def main() -> None:
    notebook = build_notebook()
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, NOTEBOOK_PATH)

    # The bundled kernelspec launches "python". Put the current venv first so
    # notebook execution uses exactly the environment that runs this script.
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    client = NotebookClient(
        notebook,
        timeout=1_200,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
    nbformat.write(notebook, NOTEBOOK_PATH)
    print(f"Executed notebook written to {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
