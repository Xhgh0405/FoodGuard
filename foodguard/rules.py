"""Deterministic, source-aware checks for food-label analysis.

These rules classify user input and compare values only when a retrieved
source exposes the relevant requirement. They do not define legal numbers.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from .parsing import FIELD_LABELS
from .claim_rules import evaluate_claim_rule


ALLERGEN_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("甲殼類及其製品", ("蝦", "蟹", "甲殼類", "蝦米", "蝦皮")),
    ("芒果及其製品", ("芒果",)),
    ("花生及其製品", ("花生",)),
    (
        "牛奶、羊奶及其製品",
        (
            "奶油乳酪",
            "乳酪",
            "起司",
            "鮮奶油",
            "牛奶",
            "牛乳",
            "鮮奶",
            "生乳",
            "羊奶",
            "奶油",
            "奶粉",
            "乳粉",
            "乳製品",
            "乳類",
            "奶類",
            "含乳",
            "乳清",
        ),
    ),
    ("蛋及其製品", ("雞蛋", "鴨蛋", "蛋黃", "蛋白", "全蛋", "蛋粉", "蛋")),
    ("堅果類及其製品", ("堅果", "杏仁", "核桃", "腰果", "榛果", "開心果", "夏威夷豆")),
    ("芝麻及其製品", ("芝麻",)),
    ("含麩質之穀物及其製品", ("小麥", "小麥麵粉", "麵粉", "大麥", "黑麥", "燕麥", "麩質")),
    ("大豆及其製品", ("大豆", "黃豆", "豆粉", "豆漿", "豆腐", "大豆蛋白")),
    ("魚類及其製品", ("魚", "魚粉", "魚露", "柴魚")),
    ("亞硫酸鹽類製品", ("亞硫酸", "偏亞硫酸", "二氧化硫")),
)

NON_EGG_PROTEIN_TERMS = (
    "大豆蛋白",
    "黃豆蛋白",
    "分離大豆蛋白",
    "soy protein",
    "isolated soy protein",
    "豌豆蛋白",
    "乳清蛋白",
    "蛋白質",
)


def _matches_allergen(category: str, ingredient: str, terms: tuple[str, ...]) -> list[str]:
    """Match a label ingredient without treating plant protein as egg."""

    text = str(ingredient)
    normalized = text.lower()
    if category == "蛋及其製品" and any(term.lower() in normalized for term in NON_EGG_PROTEIN_TERMS):
        # Keep explicit egg ingredients valid, while preventing the generic
        # substring「蛋白」/「蛋」from contaminating soy or whey protein.
        if not any(term in text for term in ("雞蛋", "鴨蛋", "蛋黃", "全蛋", "蛋粉")):
            return []
    return [term for term in terms if term.lower() in normalized]

NUTRITION_ORDER = tuple(FIELD_LABELS.values())
FIELD_BY_LABEL = {label: field for field, label in FIELD_LABELS.items()}


def analyse_allergens(ingredients: Iterable[str]) -> dict[str, Any]:
    detected: list[dict[str, Any]] = []
    for category, terms in ALLERGEN_PATTERNS:
        matched_ingredients: list[str] = []
        matched_terms: list[str] = []
        for ingredient in ingredients:
            hits = _matches_allergen(category, str(ingredient), terms)
            if hits:
                matched_ingredients.append(ingredient)
                matched_terms.extend(hit for hit in hits if hit not in matched_terms)
        if matched_ingredients:
            detected.append(
                {
                    "category": category,
                    "source_ingredients": matched_ingredients,
                    "matched_terms": matched_terms,
                }
            )

    if detected:
        summary = f"偵測到 {len(detected)} 類可能需要注意的過敏原。"
        recommendations = ["請確認包裝是否依相關規定標示過敏原警語。"]
        status = "warning"
    else:
        summary = "目前輸入成分未辨識到常見過敏原文字。"
        recommendations = ["仍建議依完整配方與原料規格進行人工確認。"]
        status = "info"
    findings = [
        {
            "category": item["category"],
            "source_ingredients": item["source_ingredients"],
        }
        for item in detected
    ]
    return {
        "status": status,
        "detection_status": "detected" if detected else "not_detected",
        "title": "過敏原分析",
        "summary": summary,
        "findings": findings,
        "detected_allergens": detected,
        "reasoning": "先將成分文字歸併為不重複的過敏原類別，再以相關來源確認標示要求。",
        "recommendations": recommendations,
    }


def required_nutrition_fields(evidence: Iterable[dict[str, Any]]) -> list[str]:
    """Extract the standard field names explicitly appearing in evidence."""

    text = " ".join(str(item.get("text", "")) for item in evidence)
    return [label for label in NUTRITION_ORDER if label in text]


def analyse_nutrition_label(
    nutrition: dict[str, Any], evidence: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    required = required_nutrition_fields(evidence)
    provided = list(nutrition.get("provided_fields", []))
    missing = [field for field in required if field not in provided]

    if not evidence or not required:
        status = "insufficient_evidence"
        summary = "目前找不到足夠依據確認營養標示的完整要求。"
    elif missing:
        status = "warning"
        summary = (
            f"營養標示目前已辨識 {len(provided)} 項；尚未提供或無法辨識 "
            f"{len(missing)} 項：{', '.join(missing)}。未提供不代表食品沒有這些營養素。"
        )
    else:
        status = "pass"
        summary = "依目前來源可辨識的主要欄位均已提供；格式、單位與實際包裝仍需另外確認。"

    findings = [
        {"field": field, "status": "missing" if field in missing else "provided"}
        for field in required
    ]
    return {
        "status": status,
        "regulation_evidence_status": "sufficient" if evidence and required else "insufficient",
        "title": "營養標示完整性",
        "summary": summary,
        "findings": findings,
        "required_fields": required,
        "provided_fields": provided,
        "missing_fields": missing,
        "parsed_nutrition": nutrition,
        "reasoning": "以來源中明確出現的營養標示項目，與解析後的輸入欄位逐項比較。",
        "recommendations": ["請再確認包裝上的標題、份量、單位、數值格式及其他適用標示。"],
    }


def _claim_nutrient(claim: str) -> tuple[str, str] | None:
    candidates = (
        ("蛋白質", "protein_g"),
        ("膳食纖維", "fiber_g"),
        ("鈣", "calcium_mg"),
        ("鐵", "iron_mg"),
        ("維生素C", "vitamin_c_mg"),
        ("脂肪", "fat_g"),
        ("鈉", "sodium_mg"),
        ("糖", "sugar_g"),
    )
    for label, field in candidates:
        if label in claim:
            return label, field
    return None


def _extract_high_threshold(claim: str, evidence: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    nutrient = _claim_nutrient(claim)
    if nutrient is None or "高" not in claim and "富含" not in claim and "多" not in claim:
        return None
    label, field = nutrient
    pattern = re.compile(
        rf"{re.escape(label)}\s+([0-9]+(?:\.[0-9]+)?)\s*公克\s+([0-9]+(?:\.[0-9]+)?)\s*公克\s+([0-9]+(?:\.[0-9]+)?)\s*公克",
        flags=re.IGNORECASE,
    )
    for item in evidence:
        text = str(item.get("text", ""))
        match = pattern.search(text)
        if not match:
            continue
        return {
            "nutrient": label,
            "field": field,
            "solid_threshold": float(match.group(1)),
            "liquid_threshold_per_100ml": float(match.group(2)),
            "liquid_threshold_per_100kcal": float(match.group(3)),
            "comparison": ">=",
            "evidence_page": item.get("page"),
        }
    return None


def analyse_nutrition_claim(
    claim: str, nutrition: dict[str, Any], evidence: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    claims = [item.strip() for item in str(claim or "").split("、") if item.strip()]
    if not claims:
        return {
            "status": "not_applicable",
            "regulation_evidence_status": "not_required",
            "title": "營養宣稱",
            "summary": "目前未提供營養宣稱，因此不需要進行營養宣稱合規判定。",
            "findings": [],
            "reasoning": "輸入為空或表示無宣稱。",
            "recommendations": [],
            "claims": [],
        }

    evidence = list(evidence)
    # Evaluate the versioned structured rule before requiring text evidence.
    # This allows complete label data to be judged even when the optional
    # vector index is unavailable, while incomplete input remains explicitly
    # marked as insufficient.
    structured = evaluate_claim_rule(claims[0], nutrition)
    evaluation: dict[str, Any] | None = structured.get("evaluation") if structured else None
    if not evidence and not structured:
        return {
            "status": "insufficient_evidence",
            "regulation_evidence_status": "insufficient",
            "title": "營養宣稱",
            "summary": "目前找不到足夠依據確認此營養宣稱。",
            "findings": [],
            "reasoning": "未取得與宣稱直接相關的來源。",
            "recommendations": ["請補充適用的官方營養宣稱規範。"],
            "claims": claims,
        }

    # Keep the structured rule even when the user's label is missing a value
    # or serving basis.  The rule itself is still useful evidence and lets the
    # UI explain exactly what information is missing.
    threshold = (
        structured.get("rule")
        if structured
        else _extract_high_threshold(claims[0], evidence)
    )
    status = "insufficient_evidence" if not evidence else "warning"
    summary = "已找到營養宣稱相關依據，但目前無法從來源完整抽取適用條件。"
    recommendations = ["若產品型態、基準量或宣稱文字不同，請再確認適用條件。"]
    if structured and evaluation:
        status = "pass" if evaluation["met"] else "fail"
        comparison = evaluation.get("comparison", "")
        summary = (
            f"「{claims[0]}」的{evaluation['basis']}數值為 {evaluation['actual']}，"
            f"與官方門檻 {comparison} {evaluation['threshold']} 比較後，"
            f"{ '符合' if evaluation['met'] else '不符合' }數值條件；仍須一併符合其他標示規定。"
        )
    elif structured:
        rule = structured.get("rule", {})
        field = rule.get("field", "對應營養素")
        values = nutrition.get("values", {})
        actual = values.get(field) if isinstance(values, dict) else None
        if actual is None:
            summary = f"「{claims[0]}」目前缺少「{rule.get('nutrient', field)}」數值，因此暫時無法完成判定。"
            recommendations = [f"請補充營養標示中的「{rule.get('nutrient', field)}」數值。"]
        else:
            summary = "目前已有營養數值，但缺少每份的公克／毫升基準量，暫時無法換算判定。"
            recommendations = ["請補充每一份量及單位（公克或毫升），才能依每100公克／毫升判讀。"]
    elif threshold and "solid_threshold" in threshold:
        values = nutrition.get("values", {})
        actual = values.get(threshold["field"])
        serving_size = str(nutrition.get("serving_size") or "")
        if actual is not None and re.search(r"(?:毫升|ml|mL)", serving_size):
            size_match = re.search(r"([0-9]+(?:\.[0-9]+)?)", serving_size)
            if size_match and float(size_match.group(1)) > 0:
                actual_per_100ml = actual / float(size_match.group(1)) * 100
                evaluation = {
                    "basis": "每100毫升",
                    "actual": actual_per_100ml,
                    "threshold": threshold["liquid_threshold_per_100ml"],
                    "comparison": threshold["comparison"],
                    "met": actual_per_100ml >= threshold["liquid_threshold_per_100ml"],
                }
        elif actual is not None:
            evaluation = {
                "basis": "每份（尚未換算為法規基準）",
                "actual": actual,
                "threshold": threshold["solid_threshold"],
                "comparison": threshold["comparison"],
                "met": actual >= threshold["solid_threshold"],
            }
        if evaluation is not None:
            status = "pass" if evaluation["met"] else "fail"
            summary = (
                f"「{claims[0]}」的{evaluation['basis']}數值為 {evaluation['actual']}，"
                f"與來源門檻 {evaluation['comparison']} {evaluation['threshold']} 比較後，"
                f"{ '符合' if evaluation['met'] else '不符合' }數值條件。"
            )

    return {
        "status": status,
        "regulation_evidence_status": "sufficient" if evidence or (structured and evaluation) else "insufficient",
        "title": "營養宣稱",
        "summary": summary,
        "findings": [{"claim": item, "evaluation": evaluation} for item in claims],
        "claim": "、".join(claims),
        "claims": claims,
        "threshold": threshold,
        "numeric_evaluation": evaluation,
        "reasoning": "先辨識宣稱，再只使用來源中可抽取的條件進行數值比較。",
        "recommendations": recommendations,
    }
