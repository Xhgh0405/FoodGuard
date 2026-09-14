"""Structured, source-aware health-risk topic detection.

This module never converts a food name into a personal cancer probability.  It
only maps explicit product/question signals to official hazard summaries and
keeps exposure context separate from the hazard classification.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any


RULES_PATH = Path(__file__).resolve().parent.parent / "data" / "health_risk_knowledge.json"


@lru_cache(maxsize=1)
def load_health_risk_knowledge() -> tuple[dict[str, Any], ...]:
    if not RULES_PATH.exists():
        return ()
    raw = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    return tuple(item for item in raw if isinstance(item, dict))


def _product_text(product: dict[str, Any] | None) -> str:
    if not isinstance(product, dict):
        return ""
    ingredients = product.get("ingredients", [])
    if not isinstance(ingredients, list):
        ingredients = [ingredients]
    return " ".join(
        [str(product.get("product_name", "")), *(str(item) for item in ingredients)]
    )


def _explicit_topic(text: str, topic: str) -> bool:
    explicit_terms = {
        "aflatoxin": ("黃麴毒素", "黃麴黴素", "aflatoxin"),
        "acrylamide": ("丙烯醯胺", "丙烯酰胺", "acrylamide"),
        "high_temperature_cooking": ("苯駢芘", "多環芳香族", "PAH", "異環胺"),
        "nitrosation": ("亞硝胺", "N-亞硝基", "nitrosamine"),
    }
    return any(term.casefold() in text.casefold() for term in explicit_terms.get(topic, ()))


def _detection_status(topic: str, text: str, product: dict[str, Any] | None) -> str:
    if topic in {"processed_meat", "red_meat", "alcohol", "aspartame"}:
        return "detected"
    if topic == "aflatoxin":
        return "possible" if any(term in text for term in ("發霉", "霉", "污染", "黃麴")) else "possible"
    if topic == "acrylamide":
        return "detected" if _explicit_topic(text, topic) else "possible"
    if topic in {"high_temperature_cooking", "nitrosation"}:
        return "detected" if _explicit_topic(text, topic) else "possible"
    return "possible"


def _evidence_category(item: dict[str, Any]) -> str:
    group = item.get("iarc_group")
    if group == "1":
        return "A"
    if group in {"2A", "2B"}:
        return "B"
    return "C"


def has_health_risk_signal(product: dict[str, Any] | None) -> bool:
    text = _product_text(product)
    return any(
        any(str(alias).casefold() in text.casefold() for alias in item.get("aliases", []))
        for item in load_health_risk_knowledge()
    )


def detect_health_risk_topics(
    question: str,
    product: dict[str, Any] | None = None,
    exposure_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    question = str(question or "")
    product_text = _product_text(product)
    combined = f"{product_text} {question}"
    topics: list[dict[str, Any]] = []
    for knowledge in load_health_risk_knowledge():
        aliases = [str(alias) for alias in knowledge.get("aliases", [])]
        matched = [alias for alias in aliases if alias.casefold() in combined.casefold()]
        if not matched:
            continue
        item = deepcopy(knowledge)
        item["matched_signals"] = matched
        item["detection_status"] = _detection_status(item["topic"], combined, product)
        item["evidence_category"] = _evidence_category(item)
        item["product_signal"] = bool(product_text and any(
            alias.casefold() in product_text.casefold() for alias in aliases
        ))
        topics.append(item)

    # Do not let a generic「豬肉」signal hide the stronger processed-meat
    # classification if the product also contains sausage/ham/bacon terms.
    if any(item["topic"] == "processed_meat" for item in topics):
        topics = [item for item in topics if item["topic"] != "red_meat"]

    return {
        "status": "evidence_found" if topics else "insufficient_evidence",
        "health_risk_intent": True,
        "detection_status": "detected" if topics else "not_enough_evidence",
        "evidence_category": "A/B/C" if topics else "D",
        "risk_topics": topics,
        "exposure_context": deepcopy(exposure_context or {}),
        "current_product_used": bool(product),
        "summary": (
            "已辨識到可對應官方健康風險資料的主題；以下區分危害分類與實際暴露風險。"
            if topics
            else "目前沒有足夠產品、成分或加工方式資料對應到健康風險主題。"
        ),
        "reasoning": "先用產品與問題中的明確訊號建立風險主題，再交由官方結構化資料說明 hazard；不估算個人罹癌機率。",
        "recommendations": [
            "若要進一步談實際風險，請補充每次份量、頻率、持續期間與料理方式。"
        ] if topics else ["沒有被列入目前資料庫不等於已證明安全，仍需查明具體成分或污染物。"],
    }


def health_risk_sources(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn structured official entries into UI/MCP evidence records."""

    sources: list[dict[str, Any]] = []
    for item in result.get("risk_topics", []):
        if not isinstance(item, dict):
            continue
        organizations = item.get("source_organizations", [])
        document = " / ".join(str(value) for value in organizations) or "官方健康風險資料"
        sources.append(
            {
                "document": document,
                "page": "web",
                "text": str(item.get("evidence_summary", "")),
                "relevant_excerpt": str(item.get("evidence_summary", "")),
                "quote": str(item.get("evidence_summary", "")),
                "score": 1.0,
                "knowledge_domain": "health_risk",
                "source_url": item.get("source_url"),
                "source_title": item.get("source_title"),
                "topic": item.get("topic"),
            }
        )
    return sources
