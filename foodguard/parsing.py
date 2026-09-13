"""Parse user-entered food-label text into stable structured data."""

from __future__ import annotations

import re
from typing import Any


FIELD_LABELS = {
    "calories_kcal": "熱量",
    "protein_g": "蛋白質",
    "fat_g": "脂肪",
    "saturated_fat_g": "飽和脂肪",
    "trans_fat_g": "反式脂肪",
    "carbohydrate_g": "碳水化合物",
    "sugar_g": "糖",
    "sodium_mg": "鈉",
}

FIELD_ALIASES = {
    "calories_kcal": ("熱量", "熱量", "calories", "kcal"),
    "protein_g": ("蛋白質", "protein"),
    "fat_g": ("脂肪", "fat"),
    "saturated_fat_g": ("飽和脂肪", "飽和脂肪酸", "saturated fat"),
    "trans_fat_g": ("反式脂肪", "反式脂肪酸", "trans fat"),
    "carbohydrate_g": ("碳水化合物", "碳水", "carbohydrate"),
    "sugar_g": ("糖", "sugar"),
    "sodium_mg": ("鈉", "sodium"),
}

NO_CLAIM_VALUES = {"", "無", "無宣稱", "沒有", "無營養宣稱", "none", "no"}


def _clean_text(value: Any) -> str:
    return str(value or "").replace("\r\n", "\n").strip()


def parse_ingredients(value: Any) -> list[str]:
    """Split an ingredient field without assigning legal meaning to it."""

    if isinstance(value, list):
        raw_items = [str(item) for item in value]
    else:
        raw_items = re.split(r"[、,，;；\n]+", _clean_text(value))
    items: list[str] = []
    for item in raw_items:
        cleaned = re.sub(r"^[\s•·\-\d.、]+", "", item).strip()
        if cleaned and cleaned not in items:
            items.append(cleaned)
    return items


def parse_claims(value: Any) -> list[str]:
    if isinstance(value, list):
        raw_items = [str(item) for item in value]
    else:
        raw_items = re.split(r"[、,，;；\n]+", _clean_text(value))
    claims: list[str] = []
    for item in raw_items:
        cleaned = item.strip()
        if cleaned and cleaned.casefold() not in {x.casefold() for x in NO_CLAIM_VALUES}:
            if cleaned not in claims:
                claims.append(cleaned)
    return claims


def _number_after_alias(text: str, aliases: tuple[str, ...]) -> float | None:
    for alias in sorted(aliases, key=len, reverse=True):
        match = re.search(
            rf"{re.escape(alias)}\s*(?:[:：=]|是)?\s*([0-9]+(?:\.[0-9]+)?)",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            return float(match.group(1))
    return None


def parse_nutrition(value: Any) -> dict[str, Any]:
    """Parse common label fields while retaining the original text."""

    if isinstance(value, dict):
        raw_text = _clean_text(value.get("raw_text", ""))
        source_values = value.get("values", value)
        if not raw_text:
            raw_text = "\n".join(
                f"{key}: {item}"
                for key, item in value.items()
                if key not in {"raw_text", "values", "provided_fields"}
            )
    else:
        raw_text = _clean_text(value)
        source_values = {}

    values: dict[str, float] = {}
    for field, aliases in FIELD_ALIASES.items():
        direct_value = source_values.get(field) if isinstance(source_values, dict) else None
        parsed = None
        if isinstance(direct_value, (int, float)) and not isinstance(direct_value, bool):
            parsed = float(direct_value)
        elif direct_value is not None:
            parsed = _number_after_alias(f"{FIELD_LABELS[field]} {direct_value}", (FIELD_LABELS[field],))
        if parsed is None:
            parsed = _number_after_alias(raw_text, aliases)
        if parsed is not None:
            values[field] = parsed

    serving_match = re.search(
        r"每\s*(?:一份量|份量|一份|份)?\s*([0-9]+(?:\.[0-9]+)?\s*(?:公克|克|g|毫升|ml|mL))",
        raw_text,
        flags=re.IGNORECASE,
    )
    package_match = re.search(
        r"本包裝\s*(?:含|有)\s*([0-9]+(?:\.[0-9]+)?)\s*份", raw_text
    )

    provided_fields = [FIELD_LABELS[field] for field in FIELD_LABELS if field in values]
    return {
        "raw_text": raw_text,
        "values": values,
        "provided_fields": provided_fields,
        "serving_size": serving_match.group(1) if serving_match else None,
        "servings_per_package": float(package_match.group(1)) if package_match else None,
    }


def parse_product_data(
    product_name: Any, ingredients: Any, nutrition: Any, claims: Any
) -> dict[str, Any]:
    """Create the canonical product object passed through the analysis flow."""

    return {
        "product_name": _clean_text(product_name),
        "ingredients": parse_ingredients(ingredients),
        "nutrition": parse_nutrition(nutrition),
        "claims": parse_claims(claims),
    }
