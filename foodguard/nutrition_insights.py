"""Deterministic nutrition insight and Taiwan DRIs lookup engine.

The engine owns parsing, unit conversion, scaling, profile matching and
percentages.  An LLM may explain its output, but it never creates these
numbers.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .context import parse_consumption_amount


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRI_PATH = PROJECT_ROOT / "data" / "dri_references.json"

NUTRIENT_META: dict[str, dict[str, Any]] = {
    "calories_kcal": {"nutrient": "calories", "label": "熱量", "unit": "kcal", "aliases": ("熱量", "卡", "大卡", "kcal", "calories")},
    "protein_g": {"nutrient": "protein", "label": "蛋白質", "unit": "g", "aliases": ("蛋白質", "蛋白", "protein")},
    "fat_g": {"nutrient": "fat", "label": "脂肪", "unit": "g", "aliases": ("脂肪", "fat")},
    "saturated_fat_g": {"nutrient": "saturated_fat", "label": "飽和脂肪", "unit": "g", "aliases": ("飽和脂肪", "飽和脂肪酸", "saturated fat")},
    "trans_fat_g": {"nutrient": "trans_fat", "label": "反式脂肪", "unit": "g", "aliases": ("反式脂肪", "trans fat")},
    "carbohydrate_g": {"nutrient": "carbohydrate", "label": "碳水化合物", "unit": "g", "aliases": ("碳水化合物", "碳水", "醣類", "carbohydrate")},
    "sugar_g": {"nutrient": "sugar", "label": "糖", "unit": "g", "aliases": ("糖", "糖類", "sugar")},
    "sodium_mg": {"nutrient": "sodium", "label": "鈉", "unit": "mg", "aliases": ("鈉", "sodium")},
    "calcium_mg": {"nutrient": "calcium", "label": "鈣", "unit": "mg", "aliases": ("鈣", "calcium")},
    "iron_mg": {"nutrient": "iron", "label": "鐵", "unit": "mg", "aliases": ("鐵", "iron")},
    "potassium_mg": {"nutrient": "potassium", "label": "鉀", "unit": "mg", "aliases": ("鉀", "potassium")},
    "vitamin_d_ug": {"nutrient": "vitamin_d", "label": "維生素D", "unit": "µg", "aliases": ("維生素d", "維生素 D", "vitamin d")},
    "magnesium_mg": {"nutrient": "magnesium", "label": "鎂", "unit": "mg", "aliases": ("鎂", "magnesium")},
    "zinc_mg": {"nutrient": "zinc", "label": "鋅", "unit": "mg", "aliases": ("鋅", "zinc")},
    "fiber_g": {"nutrient": "fiber", "label": "膳食纖維", "unit": "g", "aliases": ("膳食纖維", "纖維", "fiber")},
}

NUTRIENT_ALIASES = {
    alias.casefold(): meta["nutrient"]
    for meta in NUTRIENT_META.values()
    for alias in meta["aliases"]
}


def _load_dri_records() -> list[dict[str, Any]]:
    with DRI_PATH.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return list(payload.get("records", []))


def parse_user_profile(message: str, previous: dict[str, Any] | None = None) -> dict[str, Any]:
    """Parse only explicitly stated profile facts; never infer disease or goals."""

    profile = {"age": None, "sex": None, "weight_kg": None, "calories_kcal": None, "life_stage": None}
    if isinstance(previous, dict):
        profile.update({key: previous.get(key) for key in profile})
    text = str(message or "")
    age = re.search(r"(?<!\d)(\d{1,3})\s*歲", text)
    if age:
        profile["age"] = int(age.group(1))
    weight = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:公斤|kg)", text, re.I)
    if weight:
        profile["weight_kg"] = float(weight.group(1))
    calories = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:大卡|kcal)", text, re.I)
    if calories:
        profile["calories_kcal"] = float(calories.group(1))
    if any(term in text.lower() for term in ("男性", "男生", "male", "man")):
        profile["sex"] = "male"
    elif any(term in text.lower() for term in ("女性", "女生", "female", "woman")):
        profile["sex"] = "female"
    if profile["age"] is not None:
        profile["life_stage"] = "adult" if profile["age"] >= 19 else "child_or_adolescent"
    return profile


def parse_active_nutrient(message: str, previous: str | None = None) -> str | None:
    text = str(message or "").casefold()
    for alias, nutrient in sorted(NUTRIENT_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if alias in text:
            return nutrient
    return previous


def _unit_factor(unit: str) -> tuple[str, float] | None:
    normalized = str(unit or "").strip().lower().replace("μ", "µ")
    return {
        "g": ("g", 1.0), "公克": ("g", 1.0), "克": ("g", 1.0),
        "mg": ("g", 0.001), "毫克": ("g", 0.001),
        "µg": ("g", 0.000001), "ug": ("g", 0.000001), "微克": ("g", 0.000001),
        "ml": ("ml", 1.0), "毫升": ("ml", 1.0),
        "l": ("ml", 1000.0), "公升": ("ml", 1000.0), "升": ("ml", 1000.0),
        "kcal": ("kcal", 1.0), "大卡": ("kcal", 1.0),
    }.get(normalized)


def _nutrition_basis(nutrition: dict[str, Any]) -> tuple[float | None, str | None]:
    basis = nutrition.get("nutrition_basis", {}) if isinstance(nutrition, dict) else {}
    if isinstance(basis, dict) and basis.get("amount") and basis.get("unit"):
        factor = _unit_factor(str(basis["unit"]))
        if factor:
            return float(basis["amount"]) * factor[1], factor[0]
        return float(basis["amount"]), str(basis["unit"]).lower()
    return None, None


def parse_portion(message: str, nutrition: dict[str, Any]) -> dict[str, Any] | None:
    """Parse ml/g/L or serving/package expressions without guessing missing data."""

    explicit = parse_consumption_amount(message)
    if explicit:
        return {**explicit, "source": "explicit_amount"}
    text = str(message or "").lower()
    basis_amount, basis_unit = _nutrition_basis(nutrition)
    servings = nutrition.get("servings_per_package") if isinstance(nutrition, dict) else None
    if basis_amount is None or basis_unit is None:
        return None
    if re.search(r"半\s*(?:瓶|包|盒|罐)", text):
        if servings:
            return {"amount": basis_amount * float(servings) / 2, "unit": basis_unit, "raw": "半包裝", "source": "half_package"}
        return None
    package_match = re.search(r"(一|1|兩|二|2|三|3)\s*(?:瓶|包|盒|罐)", text)
    if package_match and servings:
        count = {"一": 1, "1": 1, "兩": 2, "二": 2, "2": 2, "三": 3, "3": 3}[package_match.group(1)]
        return {"amount": basis_amount * float(servings) * count, "unit": basis_unit, "raw": package_match.group(0), "source": "package"}
    serving_match = re.search(r"(半|一|1|兩|二|2|三|3)\s*份", text)
    if serving_match:
        count = {"半": 0.5, "一": 1, "1": 1, "兩": 2, "二": 2, "2": 2, "三": 3, "3": 3}[serving_match.group(1)]
        return {"amount": basis_amount * count, "unit": basis_unit, "raw": serving_match.group(0), "source": "serving"}
    return None


def scale_nutrition(nutrition: dict[str, Any], portion: dict[str, Any] | None) -> dict[str, Any]:
    if not portion:
        return {"status": "no_consumption_amount", "scaled_values": {}, "multiplier": None}
    basis_amount, basis_unit = _nutrition_basis(nutrition)
    if basis_amount is None or basis_unit is None:
        return {"status": "missing_nutrition_basis", "scaled_values": {}, "multiplier": None, "missing_information": ["nutrition_basis"]}
    portion_factor = _unit_factor(str(portion.get("unit", "")))
    if not portion_factor or portion_factor[0] != basis_unit.lower():
        return {"status": "unit_mismatch", "scaled_values": {}, "multiplier": None, "missing_information": ["compatible_unit"]}
    multiplier = (float(portion["amount"]) * portion_factor[1]) / basis_amount
    values = nutrition.get("values", {}) if isinstance(nutrition, dict) else {}
    scaled = {field: round(float(value) * multiplier, 4) for field, value in values.items() if isinstance(value, (int, float)) and not isinstance(value, bool)}
    return {"status": "calculated", "amount": float(portion["amount"]), "unit": portion["unit"], "basis_amount": basis_amount, "basis_unit": basis_unit, "multiplier": multiplier, "scaled_values": scaled}


def lookup_dri(nutrient: str, profile: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return matched official records, or explicitly report missing selectors."""

    profile = profile if isinstance(profile, dict) else {}
    nutrient = NUTRIENT_ALIASES.get(str(nutrient).casefold(), str(nutrient).casefold())
    records = [dict(item) for item in _load_dri_records() if item.get("nutrient") == nutrient]
    age = profile.get("age")
    sex = profile.get("sex")
    if age is not None:
        records = [item for item in records if item.get("age_min", 0) <= age and (item.get("age_max") is None or age <= item["age_max"])]
    if sex:
        records = [item for item in records if item.get("sex") in {sex, "all"}]
    missing: list[str] = []
    candidates = records
    if not age:
        missing.append("age")
    if any(item.get("sex") not in {"all", None} for item in records) and not sex:
        missing.append("sex")
    selected = records if len(records) == 1 or (sex and age is not None) else []
    status = "found" if selected else ("needs_profile" if records else "not_found")
    if not records and age is not None:
        candidates = [dict(item) for item in _load_dri_records() if item.get("nutrient") == nutrient]
    return {"status": status, "nutrient": nutrient, "records": selected or candidates, "selected": selected[0] if len(selected) == 1 else None, "missing_information": sorted(set(missing)), "source": "data/dri_references.json"}


def _comparison_label(reference_type: str) -> str:
    return {"UL": "可耐受最高攝取量", "AI": "足夠攝取量參考值", "RDA": "建議攝取量", "EAR": "平均需要量", "CDRR": "慢性疾病風險降低攝取量"}.get(reference_type, "參考值")


def build_nutrition_insights(product: dict[str, Any], portion: dict[str, Any] | None = None, profile: dict[str, Any] | None = None, active_nutrient: str | None = None) -> dict[str, Any]:
    nutrition = product.get("nutrition", {}) if isinstance(product, dict) else {}
    profile = profile if isinstance(profile, dict) else {}
    scaled_result = scale_nutrition(nutrition, portion)
    values = nutrition.get("values", {}) if isinstance(nutrition, dict) else {}
    present = [field for field, value in values.items() if field in NUTRIENT_META and isinstance(value, (int, float))]
    priority = ["calories_kcal", "saturated_fat_g", "sugar_g", "sodium_mg", "protein_g", "fat_g", "carbohydrate_g", "fiber_g"]
    important = [field for field in priority if field in present] + [field for field in present if field not in priority]
    missing: list[str] = []
    comparisons: list[dict[str, Any]] = []
    if portion and scaled_result.get("status") == "calculated":
        for field, actual in scaled_result["scaled_values"].items():
            meta = NUTRIENT_META.get(field)
            if not meta:
                continue
            reference = lookup_dri(meta["nutrient"], profile)
            selected = reference.get("selected")
            if selected and selected.get("unit") in {"g/day", "mg/day", "µg/day"}:
                ref = float(selected["reference_value"])
                comparisons.append({"nutrient": meta["nutrient"], "actual": actual, "reference_value": ref, "reference_type": selected["reference_type"], "unit": selected["unit"], "percentage": round(float(actual) / ref * 100, 2), "comparison_label": _comparison_label(selected["reference_type"]), "source": selected["source"]})
            elif selected and selected.get("unit") == "g/1000 kcal":
                calories = profile.get("calories_kcal")
                if calories:
                    ref = float(selected["reference_value"]) * float(calories) / 1000
                    comparisons.append({"nutrient": meta["nutrient"], "actual": actual, "reference_value": round(ref, 4), "reference_type": selected["reference_type"], "unit": "g/day", "percentage": round(float(actual) / ref * 100, 2), "comparison_label": _comparison_label(selected["reference_type"]), "source": selected["source"], "formula": selected.get("formula")})
                else:
                    missing.append("calories_kcal")
            elif reference.get("missing_information"):
                missing.extend(reference["missing_information"])
    insights = []
    for field in important[:6]:
        meta = NUTRIENT_META[field]
        insights.append({"nutrient": meta["nutrient"], "label": meta["label"], "field": field, "reason": "label_value_present_without_inventing_a_health_threshold"})
    if scaled_result.get("status") == "calculated" and scaled_result.get("amount", 0) >= 1000:
        insights.append({"type": "portion_context", "message": "這個份量相對較大，請以實際一次或一天的攝取情境理解；系統不據此做健康結論。"})
    return {"status": "ok", "calculated_intake": scaled_result.get("scaled_values", {}), "scale": scaled_result, "reference_comparisons": comparisons, "important_nutrients": important, "missing_information": sorted(set(missing)), "proactive_insights": insights, "active_nutrient": active_nutrient, "user_profile": profile}
