"""Auditable structured nutrition-claim rules extracted from official sources."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RULES_PATH = PROJECT_ROOT / "data" / "nutrition_claim_rules.json"


@lru_cache(maxsize=1)
def load_claim_rules() -> tuple[dict[str, Any], ...]:
    if not RULES_PATH.exists():
        return ()
    raw = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    return tuple(item for item in raw if isinstance(item, dict))


def _claim_type(claim: str) -> str | None:
    if any(term in claim for term in ("無", "零", "不含")):
        return "zero"
    if any(term in claim for term in ("低", "少", "薄", "微", "略含")):
        return "low"
    if any(term in claim for term in ("高", "多", "強化", "富含")):
        return "high"
    return None


def _nutrient_field(claim: str) -> tuple[str, str] | None:
    aliases = (
        (("膳食纖維", "高纖", "纖維"), "膳食纖維", "fiber_g"),
        (("蛋白質", "蛋白"), "蛋白質", "protein_g"),
        (("鈉",), "鈉", "sodium_mg"),
        # 「低脂」是法規與包裝上常見的簡寫，不能只接受完整的「低脂肪」。
        (("脂肪", "脂"), "脂肪", "fat_g"),
        (("糖",), "糖", "sugar_g"),
    )
    for terms, label, field in aliases:
        if any(term in claim for term in terms):
            return label, field
    return None


def find_claim_rule(claim: str) -> dict[str, Any] | None:
    """Match a claim to a stored rule; never infer a numeric threshold."""

    text = re.sub(r"\s+", "", str(claim or ""))
    kind = _claim_type(text)
    nutrient = _nutrient_field(text)
    if not kind or not nutrient:
        return None
    label, field = nutrient
    for rule in load_claim_rules():
        if rule.get("claim_type") == kind and rule.get("nutrient") == label:
            return {**rule, "field": field}
    return None


def _form_and_basis(nutrition: dict[str, Any]) -> tuple[str, float] | None:
    basis = nutrition.get("nutrition_basis", {})
    if isinstance(basis, dict) and basis.get("amount"):
        unit = str(basis.get("unit", "")).lower()
        return ("liquid" if unit in {"ml", "毫升"} else "solid", float(basis["amount"]))
    serving = str(nutrition.get("serving_size") or "")
    match = re.search(r"(\d+(?:\.\d+)?)\s*(公克|克|g|毫升|ml|mL)", serving)
    if match:
        return ("liquid" if match.group(2).lower() in {"ml", "毫升"} else "solid", float(match.group(1)))
    return None


def evaluate_claim_rule(claim: str, nutrition: dict[str, Any]) -> dict[str, Any] | None:
    rule = find_claim_rule(claim)
    if rule is None:
        return None
    values = nutrition.get("values", {})
    actual = values.get(rule["field"]) if isinstance(values, dict) else None
    form_basis = _form_and_basis(nutrition)
    if actual is None or form_basis is None:
        return {"rule": rule, "evaluation": None}
    form, amount = form_basis
    if amount <= 0:
        return {"rule": rule, "evaluation": None}
    normalized_actual = round(float(actual) / amount * 100, 6)
    threshold = rule.get("threshold", {}).get(form)
    if threshold is None:
        return {"rule": rule, "evaluation": None}
    evaluation: dict[str, Any] = {
        "basis": "每100毫升" if form == "liquid" else "每100公克",
        "actual": normalized_actual,
        "input_value": float(actual),
        "input_amount": amount,
        "input_unit": "ml" if form == "liquid" else "g",
        "threshold": float(threshold),
        "comparison": rule["operator"],
        "met": normalized_actual <= float(threshold) if rule["operator"] == "<=" else normalized_actual >= float(threshold),
        "food_form": form,
        "source_page": rule.get("source_page"),
        "source_url": rule.get("source_url"),
        "additional_requirements": rule.get("additional_requirements", []),
    }
    if form == "liquid" and rule.get("threshold_per_100kcal") is not None:
        calories = values.get("calories_kcal") if isinstance(values, dict) else None
        if isinstance(calories, (int, float)) and float(calories) > 0:
            calories_per_100ml = float(calories) / amount * 100
            nutrient_per_100kcal = normalized_actual / calories_per_100ml * 100
            if nutrient_per_100kcal >= float(rule["threshold_per_100kcal"]):
                evaluation = {
                    **evaluation,
                    "basis": "每100大卡",
                    "actual": round(nutrient_per_100kcal, 6),
                    "threshold": float(rule["threshold_per_100kcal"]),
                    "met": True,
                }
    return {
        "rule": rule,
        "evaluation": evaluation,
    }
