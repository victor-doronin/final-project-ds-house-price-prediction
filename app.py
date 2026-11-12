from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.data import MODEL_FEATURES


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "final_model.joblib"


@st.cache_resource
def load_model():
    """Load the fitted preprocessing and prediction pipeline once."""
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="Оценка стоимости дома", page_icon="🏠")
st.title("Оценка стоимости недвижимости")
st.write(
    "Введите основные характеристики объекта. Модель вернёт ориентировочную "
    "стоимость в долларах США."
)

with st.form("prediction_form"):
    left, right = st.columns(2)
    with left:
        sqft = st.number_input("Площадь, sqft", min_value=100, max_value=20_000, value=1800)
        beds = st.number_input("Спальни", min_value=0, max_value=20, value=3)
        baths = st.number_input("Ванные", min_value=0.0, max_value=20.0, value=2.0, step=0.5)
        stories = st.number_input("Этажность", min_value=1.0, max_value=100.0, value=1.0, step=1.0)
        year_built = st.number_input("Год постройки", min_value=1700, max_value=2026, value=1990)
        lot_size = st.number_input(
            "Площадь участка, sqft",
            min_value=100,
            max_value=10_000_000,
            value=7000,
        )
    with right:
        property_type = st.selectbox(
            "Тип объекта",
            ["single_family", "condo_coop", "townhouse", "multi_family", "manufactured", "other"],
        )
        state = st.text_input("Штат (например, TX)", value="TX")
        city = st.text_input("Город", value="Dallas")
        zipcode = st.text_input("Почтовый индекс", value="75201")
        has_pool = st.checkbox("Есть бассейн")
        has_fireplace = st.checkbox("Есть камин")

    with st.expander("Дополнительные характеристики"):
        remodeled_year = st.number_input(
            "Год ремонта (0 — неизвестно)", min_value=0, max_value=2026, value=0
        )
        heating = st.text_input("Отопление", value="missing")
        cooling = st.text_input("Охлаждение", value="missing")
        parking = st.text_input("Парковка", value="missing")

    submitted = st.form_submit_button("Рассчитать стоимость")

if submitted:
    row = {
        "sqft": float(sqft),
        "beds": float(beds),
        "baths": float(baths),
        "stories": float(stories),
        "year_built": float(year_built),
        "remodeled_year": np.nan if remodeled_year == 0 else float(remodeled_year),
        "lot_size_sqft": float(lot_size),
        "has_pool": int(has_pool),
        "has_fireplace": int(has_fireplace),
        "status": "for_sale",
        "property_type": property_type,
        "city": city.strip().lower() or "missing",
        "state": state.strip().lower() or "missing",
        "zipcode": zipcode.strip().lower() or "missing",
        "heating": heating.strip().lower() or "missing",
        "cooling": cooling.strip().lower() or "missing",
        "parking": parking.strip().lower() or "missing",
    }
    features = pd.DataFrame([row])[MODEL_FEATURES]
    prediction = float(load_model().predict(features)[0])
    st.metric("Ориентировочная стоимость", f"${prediction:,.0f}")
    st.caption(
        "Прогноз носит информационный характер. Ошибка выше для дорогих и "
        "редких объектов."
    )
