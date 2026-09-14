"""FoodGuard MCP Server using the Official MCP Python SDK.

The tools perform a small deterministic input analysis first, retrieve
task-specific evidence second, and return structured findings. Retrieved
chunks are evidence only; they are never presented as the final conclusion.
"""

from __future__ import annotations

from typing import Any, Iterable

from mcp.server import MCPServer

from foodguard.evidence import retrieve_evidence
from foodguard.context import calculate_consumption_nutrients as calculate_scaled_nutrients
from foodguard.health_risk import (
    detect_health_risk_topics,
    health_risk_sources,
)
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


DISEASE_GUIDE_HINTS = ("disease_guides", "糖尿病", "高血壓", "腎臟病", "高血脂")


def _claim_query_expansion(claim: str) -> tuple[str, ...]:
    """Improve retrieval recall without changing the requested claim."""

    text = str(claim or "").strip()
    expansions: dict[str, tuple[str, ...]] = {
        "高蛋白": ("高蛋白", "高蛋白質", "蛋白質 高 多 富含", "包裝食品營養宣稱 蛋白質"),
        "無糖": ("無糖", "零糖", "不含糖", "糖 無 不含 零", "包裝食品營養宣稱 糖"),
        "低糖": ("低糖", "少糖", "糖 低 少", "包裝食品營養宣稱 糖"),
        "低鈉": ("低鈉", "少鈉", "鈉 低 少", "包裝食品營養宣稱 鈉"),
        "低脂": ("低脂", "少脂", "脂肪 低 少", "包裝食品營養宣稱 脂肪"),
        "高纖": ("高纖", "高纖維", "膳食纖維 高 多 富含", "包裝食品營養宣稱 膳食纖維"),
    }
    for key, values in expansions.items():
        if key in text:
            return tuple(item for item in values if item not in text)
    return ()


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
    result["regulation_evidence_status"] = "sufficient" if sources else "insufficient"
    if "detected_allergens" in result:
        # Detection is a deterministic classification of the supplied label;
        # missing RAG evidence must not erase that useful product finding.
        result["regulation_evidence_message"] = (
            "已找到相關法規來源。" if sources else "目前知識庫缺少足夠的相關規範來源。"
        )
        return result
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


@mcp.tool(
    name="search_disease_guideline",
    description=(
        "Search official disease dietary guidance in the local source collection. "
        "Use for diabetes, hypertension, kidney disease, or dyslipidemia questions; "
        "this tool does not diagnose or set a personal intake limit."
    ),
)
def search_disease_guideline(
    disease: str, query: str = "", product_context: dict[str, Any] | None = None
) -> dict[str, Any]:
    disease = str(disease or "").strip()
    query = str(query or "").strip()
    if not disease and not query:
        return _response(
            {"status": "insufficient_evidence", "summary": NOT_ENOUGH_EVIDENCE, "message": NOT_ENOUGH_EVIDENCE},
            [],
        )
    expanded = (
        f"{disease} 飲食 注意事項 官方指引 {query} "
        + ("糖尿病與我 碳水化合物 飲品" if "糖尿病" in f"{disease}{query}" else "")
    ).strip()
    sources, debug = _retrieve_regulation(
        expanded, source_hints=DISEASE_GUIDE_HINTS, top_k=5,
        extra_queries=(f"{disease} 飲食指南", f"{disease} 生活保健"),
    )
    for source in sources:
        source["knowledge_domain"] = "disease_guidance"
    result = {
        "status": "pass" if sources else "insufficient_evidence",
        "title": "疾病飲食指引",
        "summary": "已找到官方疾病飲食指引，可用於整理注意事項。" if sources else NOT_ENOUGH_EVIDENCE,
        "knowledge_domain": "disease_guidance",
        "disease": disease,
        "product_context_used": bool(product_context),
        "findings": [],
        "reasoning": "疾病資料僅用於飲食注意事項整理，不直接推導個人醫療結論。",
        "recommendations": ["請依個人用藥、檢驗結果與醫囑決定實際份量。"] if sources else [],
    }
    if not sources:
        result["message"] = NOT_ENOUGH_EVIDENCE
    return _response(result, sources, debug)


@mcp.tool(
    name="search_health_risk",
    description=(
        "Search structured official health-risk evidence for carcinogenic hazards, "
        "food contaminants, additives, and high-temperature processing. Separate hazard "
        "classification from personal exposure risk; never estimate cancer probability."
    ),
)
def search_health_risk(
    question: str,
    product_context: dict[str, Any] | None = None,
    exposure_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = detect_health_risk_topics(question, product_context, exposure_context)
    sources = health_risk_sources(result)
    result["regulation_evidence_status"] = "sufficient" if sources else "insufficient"
    result["source_organizations"] = sorted(
        {
            organization
            for topic in result.get("risk_topics", [])
            for organization in topic.get("source_organizations", [])
        }
    )
    result["hazard_classifications"] = [
        {
            "topic": topic.get("topic"),
            "agent": topic.get("agent"),
            "iarc_group": topic.get("iarc_group"),
            "classification_label": topic.get("classification_label"),
        }
        for topic in result.get("risk_topics", [])
    ]
    return _response(
        result,
        sources,
        {
            "knowledge_base": "data/health_risk_knowledge.json",
            "retrieval_mode": "structured_official_sources",
        },
    )


@mcp.tool(
    name="calculate_consumption_nutrients",
    description="Scale parsed nutrition-label values to an explicitly supplied consumption amount.",
)
def calculate_consumption_nutrients(
    nutrition_data: dict[str, Any], consumption_amount: float, consumption_unit: str
) -> dict[str, Any]:
    result = calculate_scaled_nutrients(
        parse_nutrition(nutrition_data), float(consumption_amount), str(consumption_unit)
    )
    result["title"] = "消費量營養換算"
    result["reasoning"] = "以標示基準量與使用者明確提供的消費量做比例換算，不推定個人安全上限。"
    return _response(result, [], {"calculation": "deterministic", "source_count": 0})


@mcp.tool()
def check_allergens(ingredients: str | list[str]) -> dict[str, Any]:
    """Classify possible allergens, then retrieve only allergen-related evidence."""

    parsed_ingredients = parse_ingredients(ingredients)
    query = "食品過敏原標示規定 " + " ".join(parsed_ingredients)
    sources, debug = _retrieve_regulation(
        query,
        source_hints=("食品過敏原標示規定",),
        top_k=3,
    )
    result = _with_evidence_guard(analyse_allergens(parsed_ingredients), sources)
    result.update({"query": query, "ingredients": parsed_ingredients, "knowledge_domain": "regulation"})
    return _response(result, sources, debug)


@mcp.tool()
def check_nutrition_label(nutrition_data: dict[str, Any]) -> dict[str, Any]:
    """Compare parsed nutrition fields with fields explicitly found in evidence."""

    nutrition = parse_nutrition(nutrition_data)
    fields = " ".join(nutrition.get("provided_fields", []))
    query = f"包裝食品營養標示應遵行事項 必要標示項目 {fields}".strip()
    sources, debug = _retrieve_regulation(
        query,
        source_hints=("營養標示應遵行事項",),
        top_k=3,
    )
    result = _with_evidence_guard(analyse_nutrition_label(nutrition, sources), sources)
    result.update({"query": query, "knowledge_domain": "regulation"})
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
    extra_queries = (
        f"包裝食品營養宣稱應遵行事項 {normalized_claim} 表一 表二 表三 固體 液體",
        f"{normalized_claim} 高 多 低 少 無 零 每100公克 每100毫升",
        *_claim_query_expansion(normalized_claim),
    )
    sources, debug = _retrieve_regulation(
        query,
        source_hints=("營養宣稱應遵行事項",),
        top_k=3,
        extra_queries=extra_queries,
    )
    result = _with_evidence_guard(
        analyse_nutrition_claim(normalized_claim, nutrition, sources), sources
    )
    result.update({"query": query, "nutrition_data": nutrition})
    result["knowledge_domain"] = "regulation"
    return _response(result, sources, debug)


def main() -> None:
    """Run the server over the local stdio transport."""

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
