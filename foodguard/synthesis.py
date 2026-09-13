"""Turn retrieved evidence into a compact, grounded reasoning context.

The retriever deliberately keeps the complete source records for audit and
the developer view.  This module is the boundary used before an LLM sees
evidence: it ranks source types, removes near duplicates, and extracts only
short grounded statements and numeric/conditional statements.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable


NOT_ENOUGH_EVIDENCE = "目前知識庫找不到足夠依據"
_SPLIT_RE = re.compile(r"[。！？；\n]+")
_NUMBER_RE = re.compile(
    r"(?:\d+(?:\.\d+)?|\d+\s*/\s*\d+)\s*(?:%|％|mg|g|公克|克|毫克|大卡|kcal|毫升|ml|mL|份|歲|年)"
    r"|(?:每|低於|不超過|不得超過|至少|超過|少於|大於|小於)[^，。；\n]{0,45}\d+(?:\.\d+)?"
)
_CONDITION_WORDS = ("若", "如", "適用", "限於", "依", "按照", "每", "以", "不得", "應", "須")


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _source_name(record: dict[str, Any]) -> str:
    return _clean(record.get("document") or record.get("source") or "未提供文件")


def infer_knowledge_domain(record: dict[str, Any]) -> str:
    """Infer a display-only domain label from metadata, never from legal facts."""

    explicit = _clean(record.get("knowledge_domain"))
    if explicit:
        return explicit
    name = _source_name(record).lower()
    if any(word in name for word in ("diabetes", "糖尿病")):
        return "disease_guidance"
    if any(word in name for word in ("hypertension", "高血壓")):
        return "disease_guidance"
    if any(word in name for word in ("kidney", "腎臟")):
        return "disease_guidance"
    if any(word in name for word in ("lipid", "血脂", "高血脂")):
        return "disease_guidance"
    if any(word in name for word in ("dri", "營養素", "nutrient")):
        return "nutrition_reference"
    if any(word in name for word in ("guide", "指引", "手冊", "衛教")):
        return "official_guide"
    if any(word in name for word in ("regulation", "法規", "標示", "公告", "應遵行")):
        return "regulation"
    return "unknown"


def _authority_priority(record: dict[str, Any]) -> int:
    """Prefer formal rules while retaining other official evidence for context."""

    domain = infer_knowledge_domain(record)
    name = _source_name(record).lower()
    if domain == "regulation" or any(
        word in name for word in ("法規", "公告", "應遵行", "規定")
    ):
        return 4
    if domain in {"official_guide", "nutrition_reference"} or any(
        word in name for word in ("指引", "手冊", "dri", "參考攝取")
    ):
        return 3
    if any(word in name for word in ("q&a", "qa", "問答", "常見問題")):
        return 2
    if domain == "disease_guidance" or any(word in name for word in ("衛教", "health")):
        return 1
    return 2


def _numeric_tokens(text: str) -> list[str]:
    return [match.group(0).strip() for match in _NUMBER_RE.finditer(text)]


def _candidate_sentences(text: str) -> list[str]:
    candidates: list[str] = []
    for part in _SPLIT_RE.split(_clean(text)):
        sentence = _clean(part).strip(" ：:、")
        if len(sentence) >= 8:
            candidates.append(sentence[:260])
    return candidates


def _is_duplicate(text: str, existing: Iterable[str], threshold: float = 0.9) -> bool:
    normalized = re.sub(r"\s+", "", text)
    return any(
        normalized == re.sub(r"\s+", "", item)
        or SequenceMatcher(None, normalized, re.sub(r"\s+", "", item)).ratio() >= threshold
        for item in existing
    )


def _valid_records(evidence: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw in evidence:
        if not isinstance(raw, dict):
            continue
        text = _clean(raw.get("text"))
        document = _source_name(raw)
        if not text or document == "未提供文件":
            continue
        record = dict(raw)
        record["document"] = document
        record["page"] = raw.get("page", "—")
        try:
            record["score"] = float(raw.get("score", 0.0))
        except (TypeError, ValueError):
            record["score"] = 0.0
        record["knowledge_domain"] = infer_knowledge_domain(record)
        records.append(record)
    return records


def synthesize_evidence(
    evidence: Iterable[dict[str, Any]],
    *,
    product_context: dict[str, Any] | None = None,
    max_sources: int = 8,
) -> dict[str, Any]:
    """Create grounded concepts for answer generation.

    The returned object intentionally does not contain the original ``text``
    field.  ``sources`` has only a short relevant excerpt for the LLM and UI;
    callers may keep the original records separately for auditing/debugging.
    """

    del product_context  # reserved for future domain-aware extraction
    records = _valid_records(evidence)
    records.sort(key=lambda item: (_authority_priority(item), item["score"]), reverse=True)

    unique: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    for record in records:
        page_key = (record["document"], str(record["page"]))
        if page_key in seen_keys and any(
            _is_duplicate(record["text"], [item["text"]], threshold=0.86)
            for item in unique
            if (item["document"], str(item["page"])) == page_key
        ):
            continue
        unique.append(record)
        seen_keys.add(page_key)

    key_points: list[str] = []
    conditions: list[str] = []
    numeric_rules: list[str] = []
    compact_sources: list[dict[str, Any]] = []
    for record in unique[:max_sources]:
        sentences = _candidate_sentences(record["text"])
        selected = sentences[:3]
        for sentence in selected:
            if not _is_duplicate(sentence, key_points):
                key_points.append(sentence)
            if any(word in sentence for word in _CONDITION_WORDS) and not _is_duplicate(
                sentence, conditions
            ):
                conditions.append(sentence)
            if _numeric_tokens(sentence) and not _is_duplicate(sentence, numeric_rules):
                numeric_rules.append(sentence)

        excerpt = _clean(record["text"])
        if len(excerpt) > 240:
            excerpt = excerpt[:240].rstrip() + "…"
        compact_sources.append(
            {
                "document": record["document"],
                "page": record["page"],
                "score": record["score"],
                "knowledge_domain": record["knowledge_domain"],
                "relevant_excerpt": excerpt,
            }
        )

    numeric_signatures = {
        tuple(_numeric_tokens(sentence))
        for sentence in numeric_rules
        if _numeric_tokens(sentence)
    }
    conflicts: list[str] = []
    if len(numeric_signatures) > 1 and len(numeric_rules) > 1:
        conflicts.append("不同來源或段落出現不同數值／條件，需確認適用範圍與現行版本。")

    insufficient: list[str] = []
    if not key_points:
        insufficient.append(NOT_ENOUGH_EVIDENCE)

    return {
        "key_points": key_points[:12],
        "conditions": conditions[:8],
        "numeric_rules": numeric_rules[:8],
        "conflicts": conflicts,
        "insufficient_information": insufficient,
        "sources": compact_sources,
    }
