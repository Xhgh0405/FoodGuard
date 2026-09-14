"""Conversation-context parsing and deterministic consumption calculations."""

from __future__ import annotations

import re
from typing import Any


AMOUNT_RE = re.compile(
    r"(?<![\d.])([0-9]+(?:\.[0-9]+)?)\s*(毫升|公克|克|公斤|ml|mL|l|L|g|kg)(?![a-z])",
    re.IGNORECASE,
)


def parse_consumption_amount(message: str) -> dict[str, Any] | None:
    """Parse an explicit amount from a follow-up, without guessing omitted units."""

    match = AMOUNT_RE.search(str(message or ""))
    if not match:
        return None
    unit = match.group(2).lower()
    if unit in {"l"}:
        amount, normalized = float(match.group(1)) * 1000, "ml"
    elif unit in {"kg", "公斤"}:
        amount, normalized = float(match.group(1)) * 1000, "g"
    elif unit in {"毫升", "ml"}:
        amount, normalized = float(match.group(1)), "ml"
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
    if not basis_amount or not basis_unit or str(basis_unit).lower() != str(unit).lower():
        return {
            "status": "insufficient_input",
            "message": "需要知道營養標示的基準量與消費量使用相同單位，才能換算。",
            "amount": amount,
            "unit": unit,
            "scaled_values": {},
        }
    multiplier = float(amount) / float(basis_amount)
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
