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
    "fiber_g": "膳食纖維",
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
    "fiber_g": ("膳食纖維", "膳食纖維", "fiber", "fibre"),
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


_UNIT_PATTERN = r"(公克|克|g|毫克|mg|微克|μg|ug|毫升|ml|mL|公升|升|l|L|kcal|千卡)"
_DEFAULT_FIELD_UNITS = {
    "calories_kcal": "kcal",
    "protein_g": "g",
    "fat_g": "g",
    "saturated_fat_g": "g",
    "trans_fat_g": "g",
    "carbohydrate_g": "g",
    "sugar_g": "g",
    "sodium_mg": "mg",
    "fiber_g": "g",
}


def _number_and_unit_after_alias(
    text: str, aliases: tuple[str, ...]
) -> tuple[float, str | None] | None:
    """Read a value even when its unit is omitted or placed after the number."""

    for alias in sorted(aliases, key=len, reverse=True):
        match = re.search(
            rf"{re.escape(alias)}\s*(?:[:：=]|是)?\s*([0-9]+(?:\.[0-9]+)?)\s*{_UNIT_PATTERN}?",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            return float(match.group(1)), match.group(2)
    return None


def _normalize_field_value(field: str, value: float, unit: str | None) -> float:
    """Normalize explicit units to the canonical field unit; missing units use the field default."""

    if not unit:
        return value
    normalized = unit.casefold()
    if field == "calories_kcal":
        if normalized in {"千卡", "kcal"}:
            return value
        return value
    if field == "sodium_mg":
        if normalized in {"g", "公克", "克"}:
            return value * 1000
        if normalized in {"μg", "ug", "微克"}:
            return value / 1000
        return value
    if normalized in {"mg", "毫克"}:
        return value / 1000
    if normalized in {"μg", "ug", "微克"}:
        return value / 1_000_000
    if normalized in {"kg", "公斤", "公克"}:
        return value * 1000 if normalized in {"kg", "公斤"} else value
    return value


def parse_nutrition(value: Any) -> dict[str, Any]:
    """Parse common label fields while retaining the original text."""

    explicit_basis: dict[str, Any] = {}
    explicit_serving_size: Any = None
    if isinstance(value, dict):
        raw_text = _clean_text(value.get("raw_text", ""))
        source_values = value.get("values", value)
        if isinstance(value.get("nutrition_basis"), dict):
            explicit_basis = value["nutrition_basis"]
        explicit_serving_size = value.get("serving_size")
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
        parsed_unit: str | None = None
        if isinstance(direct_value, (int, float)) and not isinstance(direct_value, bool):
            parsed = float(direct_value)
        elif direct_value is not None:
            parsed_with_unit = _number_and_unit_after_alias(
                f"{FIELD_LABELS[field]} {direct_value}", (FIELD_LABELS[field],)
            )
            if parsed_with_unit:
                parsed, parsed_unit = parsed_with_unit
        if parsed is None:
            parsed_with_unit = _number_and_unit_after_alias(raw_text, aliases)
            if parsed_with_unit:
                parsed, parsed_unit = parsed_with_unit
        if parsed is not None:
            values[field] = _normalize_field_value(field, parsed, parsed_unit)

    serving_match = re.search(
        r"每\s*(?:一份量|份量|一份|份)?\s*(?:[:：=]\s*)?(?:約\s*)?"
        r"([0-9]+(?:\.[0-9]+)?)\s*(公克|克|g|毫升|ml|mL|公升|升|l|L)?",
        raw_text,
        flags=re.IGNORECASE,
    )
    package_match = re.search(
        r"本\s*包裝\s*(?:(?:含|有)\s*)?(?:[:：=]\s*)?(?:約\s*)?"
        r"([0-9]+(?:\.[0-9]+)?)\s*份",
        raw_text,
    )

    if serving_match:
        basis_amount = float(serving_match.group(1))
        unit = (serving_match.group(2) or "g").lower()
        basis_unit = "ml" if unit in {"ml", "毫升", "l", "公升", "升"} else "g"
        if basis_unit == "ml" and unit in {"l", "公升", "升"}:
            basis_amount *= 1000
        serving_size = f"{serving_match.group(1)} {serving_match.group(2) or 'g'}"
    elif explicit_basis.get("amount") is not None:
        # MCP tools receive the already-normalized product dictionary. Keep
        # its explicit basis instead of losing it during the second parse.
        basis_amount = float(explicit_basis["amount"])
        basis_unit = str(explicit_basis.get("unit") or "g").lower()
        if basis_unit in {"毫升", "ml", "l", "公升", "升"}:
            basis_unit = "ml"
        else:
            basis_unit = "g"
        serving_size = str(explicit_serving_size or f"{basis_amount:g} {basis_unit}")
    else:
        basis_amount = None
        basis_unit = None
        serving_size = explicit_serving_size

    provided_fields = [FIELD_LABELS[field] for field in FIELD_LABELS if field in values]
    return {
        "raw_text": raw_text,
        "values": values,
        "provided_fields": provided_fields,
        "serving_size": serving_size,
        "servings_per_package": float(package_match.group(1)) if package_match else None,
        "nutrition_basis": {
            "amount": basis_amount,
            "unit": basis_unit,
            "source": "serving_size" if serving_match else (explicit_basis.get("source") if explicit_basis else None),
        },
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
