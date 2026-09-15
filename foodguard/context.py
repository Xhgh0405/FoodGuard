"""Conversation-context parsing and deterministic consumption calculations."""

from __future__ import annotations

import re
from typing import Any


AMOUNT_RE = re.compile(
    r"(?<![\d.])([0-9]+(?:\.[0-9]+)?)\s*(毫升|公克|克|公斤|ml|mL|l|L|g|kg)(?![a-z])",
    re.IGNORECASE,
)

STANDARD_AMOUNT_RE = re.compile(
    r"(?<![\d.])([0-9]+(?:\.[0-9]+)?)\s*(µg|μg|ug|微克|mg|毫克|g|公克|克|kg|公斤|ml|毫升|l|公升|升|kcal|大卡)(?![a-z])",
    re.IGNORECASE,
)

PIECE_AMOUNT_RE = re.compile(
    r"(?<![\d.])([0-9]+(?:\.[0-9]+)?|[一二兩三四五六七八九十百]+)\s*(根|支|瓶|罐|份|片|顆|個)(?!\w)"
)
DURATION_RE = re.compile(
    r"([0-9]+(?:\.[0-9]+)?|[一二兩三四五六七八九十百]+)\s*(年|個月|月|週|星期|天)"
)


def _chinese_number(value: str) -> float | None:
    if value.isdigit():
        return float(value)
    digits = {"一": 1, "二": 2, "兩": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    if value == "十":
        return 10.0
    if value.startswith("十"):
        return float(10 + digits.get(value[1:], 0))
    if "十" in value:
        left, right = value.split("十", 1)
        return float(digits.get(left, 1) * 10 + digits.get(right, 0))
    if value in digits:
        return float(digits[value])
    return None


def parse_exposure_context(
    message: str, previous: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Extract explicit amount, frequency, duration and preparation signals.

    Missing slots are retained from the previous turn.  The parser never
    invents a gram amount for phrases such as「每天吃兩根」; it records the
    piece count and frequency instead.
    """

    context: dict[str, Any] = {
        "amount": None,
        "unit": None,
        "frequency": None,
        "duration": None,
        "preparation_method": None,
    }
    if isinstance(previous, dict):
        context.update({key: previous.get(key) for key in context})
    text = str(message or "")
    amount = parse_consumption_amount(text)
    if amount:
        context["amount"] = amount["amount"]
        context["unit"] = amount["unit"]
    else:
        piece = PIECE_AMOUNT_RE.search(text)
        if piece:
            value = _chinese_number(piece.group(1))
            if value is not None:
                context["amount"] = value
                context["unit"] = piece.group(2)

    frequency_terms = (
        (("一個月一次", "一月一次", "一個月吃一次", "每月一次"), "monthly_once"),
        (("每天", "每日", "天天", "daily", "per day"), "daily"),
        (("每週", "每星期", "一週", "每周", "weekly", "per week"), "weekly"),
        (("每月", "每個月", "一個月", "monthly", "per month"), "monthly"),
        (("偶爾", "偶尔", "occasionally"), "occasional"),
        (("一次",), "once"),
    )
    for terms, normalized in frequency_terms:
        if any(term.casefold() in text.casefold() for term in terms):
            context["frequency"] = normalized
            break

    duration = DURATION_RE.search(text)
    if duration and any(term in text for term in ("已經", "已经", "持續", "持續了", "吃了", "喝了", "年來", "months", "years")):
        value = _chinese_number(duration.group(1))
        if value is not None:
            unit = {"年": "years", "個月": "months", "月": "months", "週": "weeks", "星期": "weeks", "天": "days"}[duration.group(2)]
            context["duration"] = f"{value:g} {unit}"

    preparation_terms = (
        (("燒焦", "焦黑", "炭烤", "燒烤", "炙烤"), "charred_or_grilled"),
        (("煙燻", "熏製", "煙製"), "smoked"),
        (("油炸", "油炸", "炸"), "deep_fried"),
        (("烘焙", "焙烤", "烤"), "baked_or_roasted"),
        (("醃製", "醃漬", "醃"), "preserved_or_pickled"),
        (("生食", "生吃"), "raw"),
    )
    for terms, normalized in preparation_terms:
        if any(term in text for term in terms):
            context["preparation_method"] = normalized
            break
    return context


def parse_consumption_amount(message: str) -> dict[str, Any] | None:
    """Parse an explicit amount from a follow-up, without guessing omitted units."""

    match = STANDARD_AMOUNT_RE.search(str(message or "")) or AMOUNT_RE.search(str(message or ""))
    if not match:
        return None
    unit = match.group(2).lower()
    if unit in {"l", "公升", "升"}:
        amount, normalized = float(match.group(1)) * 1000, "ml"
    elif unit in {"kg", "公斤"}:
        amount, normalized = float(match.group(1)) * 1000, "g"
    elif unit in {"毫升", "ml"}:
        amount, normalized = float(match.group(1)), "ml"
    elif unit in {"mg", "毫克"}:
        amount, normalized = float(match.group(1)), "mg"
    elif unit in {"µg", "μg", "ug", "微克"}:
        amount, normalized = float(match.group(1)), "µg"
    elif unit in {"kcal", "大卡"}:
        amount, normalized = float(match.group(1)), "kcal"
    else:
        amount, normalized = float(match.group(1)), "g"
    return {"amount": amount, "unit": normalized, "raw": match.group(0)}


def calculate_consumption_nutrients(
    nutrition: dict[str, Any], amount: float, unit: str
) -> dict[str, Any]:
    """Scale label values to an explicitly supplied consumption amount."""

    basis = nutrition.get("nutrition_basis", {}) if isinstance(nutrition, dict) else {}
    basis_amount = basis.get("amount") if isinstance(basis, dict) else None
    basis_unit = basis.get("unit") if isinstance(basis, dict) else None
    unit_factors = {
        "g": ("g", 1.0), "公克": ("g", 1.0), "克": ("g", 1.0),
        "kg": ("g", 1000.0), "公斤": ("g", 1000.0),
        "mg": ("g", 0.001), "毫克": ("g", 0.001),
        "µg": ("g", 0.000001), "μg": ("g", 0.000001), "ug": ("g", 0.000001), "微克": ("g", 0.000001),
        "ml": ("ml", 1.0), "毫升": ("ml", 1.0), "l": ("ml", 1000.0), "公升": ("ml", 1000.0), "升": ("ml", 1000.0),
        "kcal": ("kcal", 1.0), "大卡": ("kcal", 1.0),
    }
    basis_factor = unit_factors.get(str(basis_unit).lower()) if basis_unit else None
    amount_factor = unit_factors.get(str(unit).lower())
    if not basis_amount or not basis_factor or not amount_factor or basis_factor[0] != amount_factor[0]:
        return {
            "status": "insufficient_input",
            "message": "需要知道營養標示的基準量與消費量使用相同單位，才能換算。",
            "amount": amount,
            "unit": unit,
            "scaled_values": {},
        }
    multiplier = (float(amount) * amount_factor[1]) / (float(basis_amount) * basis_factor[1])
    values = nutrition.get("values", {}) if isinstance(nutrition, dict) else {}
    scaled = {
        field: round(float(value) * multiplier, 4)
        for field, value in values.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    return {
        "status": "calculated",
        "amount": amount,
        "unit": unit,
        "basis_amount": float(basis_amount),
        "basis_unit": unit,
        "multiplier": multiplier,
        "scaled_values": scaled,
    }
