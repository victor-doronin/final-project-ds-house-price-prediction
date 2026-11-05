from __future__ import annotations

import ast
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RAW_COLUMNS = [
    "status",
    "private pool",
    "propertyType",
    "street",
    "baths",
    "homeFacts",
    "fireplace",
    "city",
    "sqft",
    "zipcode",
    "beds",
    "state",
    "stories",
    "PrivatePool",
    "target",
]

NUMERIC_FEATURES = [
    "sqft",
    "beds",
    "baths",
    "stories",
    "year_built",
    "remodeled_year",
    "lot_size_sqft",
    "has_pool",
    "has_fireplace",
]

CATEGORICAL_FEATURES = [
    "status",
    "property_type",
    "city",
    "state",
    "zipcode",
    "heating",
    "cooling",
    "parking",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_raw_data(zip_path: str | Path) -> pd.DataFrame:
    """Read the real CSV member and ignore macOS metadata in the ZIP archive."""
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open("data.csv") as csv_file:
            return pd.read_csv(csv_file, usecols=RAW_COLUMNS, dtype="string")


def _extract_number(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.replace(",", "", regex=False)
    extracted = cleaned.str.extract(r"(-?\d+(?:\.\d+)?)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def _bounded(series: pd.Series, lower: float, upper: float) -> pd.Series:
    return series.where(series.between(lower, upper))


def _clean_text(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .replace("", pd.NA)
        .fillna("missing")
    )


def _normalize_property_type(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).strip().lower()
    if not text:
        return "missing"
    if "land" in text or "lot" in text:
        return "land"
    if "single" in text or "detached" in text:
        return "single_family"
    if "condo" in text or "co-op" in text or text == "coop":
        return "condo_coop"
    if "town" in text or "row home" in text:
        return "townhouse"
    if "multi" in text or "duplex" in text or "triplex" in text:
        return "multi_family"
    if "mobile" in text or "manufactured" in text:
        return "manufactured"
    return "other"


def _normalize_status(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).strip().lower()
    if not text:
        return "missing"
    if "rent" in text or "lease" in text:
        return "rental"
    if "foreclos" in text or "pre-foreclos" in text:
        return "foreclosure"
    if "auction" in text:
        return "auction"
    if "pending" in text or "contract" in text or "contingent" in text:
        return "pending"
    if "new construction" in text or text == "new":
        return "new_construction"
    if "active" in text or "for sale" in text or text == "p":
        return "for_sale"
    return "other"


def _parse_home_facts(value: Any) -> tuple[Any, ...]:
    empty = (None, None, None, None, None, None)
    if pd.isna(value):
        return empty
    try:
        parsed = ast.literal_eval(str(value))
        facts = parsed.get("atAGlanceFacts", [])
    except (ValueError, SyntaxError, AttributeError):
        return empty

    values: dict[str, Any] = {}
    for fact in facts:
        if isinstance(fact, dict):
            label = str(fact.get("factLabel", "")).strip().lower()
            values[label] = fact.get("factValue")

    # Price/sqft is deliberately not returned: it leaks the target price.
    return (
        values.get("year built"),
        values.get("remodeled year"),
        values.get("heating"),
        values.get("cooling"),
        values.get("parking"),
        values.get("lotsize"),
    )


def _parse_lot_size(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower().replace(",", "")
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return np.nan
    size = float(match.group(1))
    if "acre" in text:
        size *= 43_560
    return size if 100 <= size <= 10_000_000 else np.nan


def _is_present(series: pd.Series) -> pd.Series:
    text = _clean_text(series)
    negative = {"missing", "no", "none", "false", "0", "not applicable"}
    return (~text.isin(negative)).astype("int8")


def prepare_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean raw listings and return model-ready features plus an audit report."""
    report: dict[str, int] = {"raw_rows": len(raw)}
    data = raw.drop_duplicates().copy()
    report["duplicate_rows_removed"] = len(raw) - len(data)

    target_text = data["target"].astype("string")
    rental_mask = target_text.str.contains(r"/mo|month|rent", case=False, na=False)
    data["target"] = _extract_number(target_text)
    report["rental_rows_removed"] = int(rental_mask.sum())
    data = data.loc[~rental_mask].copy()

    data["property_type"] = data["propertyType"].map(_normalize_property_type)
    land_mask = data["property_type"].eq("land")
    report["land_rows_removed"] = int(land_mask.sum())
    data = data.loc[~land_mask].copy()

    valid_target = data["target"].between(10_000, 10_000_000)
    report["invalid_target_rows_removed"] = int((~valid_target).sum())
    data = data.loc[valid_target].copy()

    data["sqft"] = _bounded(_extract_number(data["sqft"]), 100, 20_000)
    data["beds"] = _bounded(_extract_number(data["beds"]), 0, 20)
    data["baths"] = _bounded(_extract_number(data["baths"]), 0, 20)

    story_text = _clean_text(data["stories"]).replace(
        {"one": "1", "two": "2", "three": "3", "four": "4"}
    )
    data["stories"] = _bounded(_extract_number(story_text), 1, 100)

    fact_columns = [
        "year_built",
        "remodeled_year",
        "heating",
        "cooling",
        "parking",
        "lot_size_raw",
    ]
    facts = pd.DataFrame(
        data["homeFacts"].map(_parse_home_facts).tolist(),
        columns=fact_columns,
        index=data.index,
    )
    data = pd.concat([data, facts], axis=1)
    data["year_built"] = _bounded(_extract_number(data["year_built"]), 1700, 2026)
    data["remodeled_year"] = _bounded(
        _extract_number(data["remodeled_year"]), 1700, 2026
    )
    data["lot_size_sqft"] = data["lot_size_raw"].map(_parse_lot_size)

    data["has_pool"] = (
        _is_present(data["private pool"]) | _is_present(data["PrivatePool"])
    ).astype("int8")
    data["has_fireplace"] = _is_present(data["fireplace"])

    data["status"] = data["status"].map(_normalize_status)
    for column in ["city", "state", "zipcode", "heating", "cooling", "parking"]:
        data[column] = _clean_text(data[column])

    street = _clean_text(data["street"])
    data["listing_group"] = street + "|" + data["zipcode"]

    result = data[MODEL_FEATURES + ["target", "listing_group"]].copy()
    report["prepared_rows"] = len(result)
    report["rows_with_missing_sqft"] = int(result["sqft"].isna().sum())
    report["rows_with_missing_beds"] = int(result["beds"].isna().sum())
    report["rows_with_missing_baths"] = int(result["baths"].isna().sum())
    return result, report
