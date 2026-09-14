"""FoodGuard MCP Server using the Official MCP Python SDK.

The tools perform a small deterministic input analysis first, retrieve
task-specific evidence second, and return structured findings. Retrieved
chunks are evidence only; they are never presented as the final conclusion.
"""

from __future__ import annotations

from typing import Any, Iterable

from mcp.server import MCPServer

from foodguard.evidence import retrieve_evidence
from foodguard.parsing import parse_claims, parse_ingredients, parse_nutrition
from foodguard.rules import (
    analyse_allergens,
    analyse_nutrition_claim,
    analyse_nutrition_label,
)
from rag import search as rag_search


NOT_ENOUGH_EVIDENCE = "目前知識庫找不到足夠依據"

mcp = MCPServer(
    "FoodGuard MCP Server",
    instructions="Return structured, source-grounded food-label analysis.",
)


def _source_records(matches: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    for match in matches:
        if not all(key in match for key in ("source", "page", "text", "score")):
            continue
        identity = (str(match["source"]), int(match["page"]), str(match["text"]))
        if identity in seen:
            continue
        seen.add(identity)
        sources.append(
            {
                "document": str(match["source"]),
                "page": int(match["page"]),
                "text": str(match["text"]),
                "quote": str(match["text"]),
                "score": float(match["score"]),
                "chunk_id": match.get("chunk_id"),
            }
        )
    return sources


def _retrieve_regulation(
    query: str,
    *,
    source_hints: tuple[str, ...] = (),
    top_k: int = 3,
    extra_queries: tuple[str, ...] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Retrieve and filter evidence for one task; no legal rule is defined here."""

    debug: dict[str, Any] = {"queries": [], "retrieval": []}
    matches: list[dict[str, Any]] = []
    try:
        for current_query in (query, *extra_queries):
            found, current_debug = retrieve_evidence(
                current_query,
                source_hints=source_hints,
                top_k=top_k,
                search_function=rag_search,
            )
            debug["queries"].append(current_query)
            debug["retrieval"].append(current_debug)
            matches.extend(found)
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        debug["error"] = str(exc)
        return [], debug

    sources = _source_records(matches)
    sources.sort(key=lambda item: item["score"], reverse=True)
    return sources[:top_k], debug


def _with_evidence_guard(
    rule_result: dict[str, Any], sources: list[dict[str, Any]]
) -> dict[str, Any]:
    """Prevent a local input hint from becoming a legal conclusion without evidence."""

    result = dict(rule_result)
    if not sources and result.get("status") != "not_applicable":
        result["status"] = "insufficient_evidence"
        result["message"] = NOT_ENOUGH_EVIDENCE
        result["summary"] = NOT_ENOUGH_EVIDENCE
        result["reasoning"] = "已完成輸入整理，但沒有足夠的相關法規來源支持判斷。"
        result["recommendations"] = ["請加入對應的官方法規文件後再重新分析。"]
    return result


def _response(
    result: dict[str, Any],
    sources: list[dict[str, Any]],
    debug_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"result": result, "sources": sources}
    if debug_evidence is not None:
        payload["debug_evidence"] = debug_evidence
    return payload


@mcp.tool(
    name="search_food_regulation",
    description=(
        "Search imported official food-regulation sources. Use this tool for broad food-law "
        "questions and return only evidence from the local source collection."
    ),
)
def search_food_regulation(query: str) -> dict[str, Any]:
    """Search regulation evidence without turning retrieved text into a conclusion."""

    query = str(query or "").strip()
    if not query:
        return _response(
            {
                "status": "insufficient_evidence",
                "title": "食品法規搜尋",
                "summary": NOT_ENOUGH_EVIDENCE,
                "findings": [],
                "reasoning": "查詢不可為空。",
                "recommendations": ["請輸入要查詢的食品法規主題。"],
                "query": query,
            },
            [],
        )

    extra_queries: tuple[str, ...] = ()
    if any(term in query for term in ("食品法", "食品法規", "食品法律", "食品標示規定有哪些")):
        extra_queries = ("食品安全衛生管理法 食品標示 食品宣傳 廣告",)
    elif any(term in query for term in ("糖尿病", "高血壓", "腎臟病", "腎病", "高血脂", "血脂")):
        # Short questions often contain only a disease and「可以嗎」. Search
        # the matching official guide explicitly instead of relying on one
        # generic embedding query.
        extra_queries = (
            f"{query} 飲食 注意事項",
            "糖尿病與我 飲食 飲品 碳水化合物" if "糖尿病" in query else f"{query} 飲食指引",
        )
    sources, debug = _retrieve_regulation(
        query, top_k=5, extra_queries=extra_queries
    )
    result = {
        "status": "pass" if sources else "insufficient_evidence",
        "title": "食品法規搜尋",
        "summary": "已找到與問題相關的法規依據。" if sources else NOT_ENOUGH_EVIDENCE,
        "findings": [],
        "reasoning": "以下內容是檢索到的證據，應依問題再進行判讀。",
        "recommendations": [],
        "query": query,
    }
    if not sources:
        result["message"] = NOT_ENOUGH_EVIDENCE
    return _response(result, sources, debug)


@mcp.tool()
def check_allergens(ingredients: str | list[str]) -> dict[str, Any]:
    """Classify possible allergens, then retrieve only allergen-related evidence."""

    parsed_ingredients = parse_ingredients(ingredients)
    query = "食品過敏原標示規定 " + " ".join(parsed_ingredients)
    sources, debug = _retrieve_regulation(
        query,
        source_hints=("過敏原",),
        top_k=3,
    )
    result = _with_evidence_guard(analyse_allergens(parsed_ingredients), sources)
    result.update({"query": query, "ingredients": parsed_ingredients})
    return _response(result, sources, debug)


@mcp.tool()
def check_nutrition_label(nutrition_data: dict[str, Any]) -> dict[str, Any]:
    """Compare parsed nutrition fields with fields explicitly found in evidence."""

    nutrition = parse_nutrition(nutrition_data)
    fields = " ".join(nutrition.get("provided_fields", []))
    query = f"包裝食品營養標示應遵行事項 必要標示項目 {fields}".strip()
    sources, debug = _retrieve_regulation(
        query,
        source_hints=("營養標示",),
        top_k=3,
    )
    result = _with_evidence_guard(analyse_nutrition_label(nutrition, sources), sources)
    result.update({"query": query})
    return _response(result, sources, debug)


@mcp.tool()
def check_nutrition_claim(
    claim: str, nutrition_data: dict[str, Any]
) -> dict[str, Any]:
    """Skip empty claims; otherwise retrieve claim evidence and evaluate source thresholds."""

    claims = parse_claims(claim)
    nutrition = parse_nutrition(nutrition_data)
    if not claims:
        result = analyse_nutrition_claim("", nutrition, [])
        result.update({"query": "", "nutrition_data": nutrition})
        return _response(result, [], {"queries": [], "skipped": "claim_not_provided"})

    normalized_claim = "、".join(claims)
    query = f"包裝食品營養宣稱應遵行事項 {normalized_claim} 含量標準"
    sources, debug = _retrieve_regulation(
        query,
        source_hints=("營養宣稱",),
        top_k=3,
    )
    result = _with_evidence_guard(
        analyse_nutrition_claim(normalized_claim, nutrition, sources), sources
    )
    result.update({"query": query, "nutrition_data": nutrition})
    return _response(result, sources, debug)


def main() -> None:
    """Run the server over the local stdio transport."""

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
