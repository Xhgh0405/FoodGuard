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
from foodguard.food_database import search_food_composition as lookup_food_composition
from foodguard.health_risk import (
    detect_health_risk_topics,
    health_risk_sources,
)
from foodguard.nutrition_insights import (
    build_nutrition_insights,
    lookup_dri,
    parse_active_nutrient,
    parse_portion,
    parse_user_profile,
)
from foodguard.parsing import parse_claims, parse_ingredients, parse_nutrition
from foodguard.rules import (
    analyse_allergens,
    analyse_nutrition_claim,
    analyse_nutrition_label,
    evaluate_claim_rule,
)
from foodguard.web_search import fetch_web_page, search_web
from rag import search as rag_search


NOT_ENOUGH_EVIDENCE = "目前知識庫找不到足夠依據"

mcp = MCPServer(
    "FoodGuard MCP Server",
    instructions="Return structured, source-grounded food-label analysis.",
)


DISEASE_GUIDE_HINTS = ("disease_guides",)
DISEASE_GUIDE_PATHS = {
    "糖尿病": "disease_guides/diabetes",
    "高血壓": "disease_guides/hypertension",
    "腎臟病": "disease_guides/kidney_disease",
    "高血脂": "disease_guides/dyslipidemia",
}
DISEASE_DIETARY_QUERIES = {
    "糖尿病": "糖尿病病人飲食原則 醣類 食物 份量 高纖 適量油脂",
    "高血壓": "高血壓 飲食原則 鈉 鹽分 食物 份量",
    "腎臟病": "腎臟病 飲食原則 蛋白質 鈉 鉀 份量",
    "高血脂": "高血脂 飲食原則 脂肪 飽和脂肪 膽固醇 食物",
}


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
        result["recommendations"] = [
            "請擴充本機官方資料庫，或開啟 Web Search 後再重新分析。"
        ]
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
    source_hints: tuple[str, ...] = ()
    if any(term in query for term in ("過敏原", "乳製品", "牛奶", "羊奶", "奶類", "乳類")):
        # A generic「過敏原」query can be outranked by unrelated food-law
        # pages. Constrain this lookup to the dedicated official allergen
        # documents and expand with relevant dairy terms.
        source_hints = ("食品過敏原標示規定",)
        extra_queries = (
            "食品過敏原標示規定 乳製品 牛奶 羊奶",
            "食品過敏原標示規定 過敏原警語 標示方式",
            "食品過敏原標示規定 Q&A 乳類",
        )
    elif any(term in query for term in ("食品法", "食品法規", "食品法律", "食品標示規定有哪些")):
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
        query, source_hints=source_hints, top_k=5, extra_queries=extra_queries
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
def search_food_composition(query: str, limit: int = 5) -> dict[str, Any]:
    """Look up official TFDA food-composition data, separate from legal RAG."""

    result = lookup_food_composition(query, limit=limit)
    sources: list[dict[str, Any]] = []
    if result.get("status") == "pass":
        sources.append(
            {
                "document": "TFDA 臺灣食品成分資料庫官方開放資料",
                "page": "database",
                "text": "依食品樣品名稱查詢官方每100克食品成分資料。",
                "quote": "TFDA 臺灣食品成分資料庫官方開放資料",
                "score": 1.0,
                "knowledge_domain": "food_database",
                "source_url": "https://www.fda.gov.tw/TC/siteList.aspx?sid=284",
            }
        )
    return _response(result, sources, {"structured_lookup": True})


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
    matched_disease = next(
        (name for name in DISEASE_DIETARY_QUERIES if name in f"{disease}{query}"),
        disease,
    )
    dietary_query = DISEASE_DIETARY_QUERIES.get(
        matched_disease, f"{disease} 飲食原則 食物 份量 注意事項"
    )
    expanded = f"{dietary_query} {query}".strip()
    source_hints = (
        (DISEASE_GUIDE_PATHS[matched_disease],)
        if matched_disease in DISEASE_GUIDE_PATHS
        else DISEASE_GUIDE_HINTS
    )
    sources, debug = _retrieve_regulation(
        expanded, source_hints=source_hints, top_k=5,
        extra_queries=(dietary_query, f"{disease} 飲食指南 {query}", f"{disease} 生活保健"),
    )
    # When a disease folder has both a PDF and an HPA provenance page, prefer
    # the actual guide: the downloaded HTML page contains site navigation that
    # can outrank the guide's dietary pages.  Keep HTML-only disease folders
    # searchable because some official guides are currently stored that way.
    source_documents = [str(source.get("document", "")) for source in sources]
    pdf_directories = {
        document.rsplit("/", 1)[0]
        for document in source_documents
        if document.lower().endswith(".pdf") and "/" in document
    }
    if pdf_directories:
        sources = [
            source for source in sources
            if not (
                str(source.get("document", "")).lower().endswith((".html", ".htm"))
                and str(source.get("document", "")).rsplit("/", 1)[0] in pdf_directories
            )
        ]
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
            "evidence_category": topic.get("evidence_category"),
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


@mcp.tool(
    name="lookup_dri_reference",
    description="Look up an official structured Taiwan DRIs reference without inventing a demographic value.",
)
def lookup_dri_reference(
    nutrient: str, user_profile: dict[str, Any] | None = None
) -> dict[str, Any]:
    result = lookup_dri(nutrient, user_profile)
    result["knowledge_domain"] = "nutrition_reference"
    return _response(
        result,
        [
            {
                "document": result.get("source", "data/dri_references.json"),
                "page": "structured",
                "text": "Official structured Taiwan DRIs record",
                "quote": "Official structured Taiwan DRIs record",
                "score": 1.0,
                "knowledge_domain": "nutrition_reference",
                "source_url": "https://www.hpa.gov.tw/Pages/Detail.aspx?nodeid=4248&pid=12285",
            }
        ],
        {"lookup": "data/dri_references.json", "deterministic": True},
    )


@mcp.tool(
    name="analyze_nutrition_insights",
    description="Calculate scaled intake, official DRI comparisons and proactive nutrition insights deterministically.",
)
def analyze_nutrition_insights(
    product_context: dict[str, Any],
    consumption_context: dict[str, Any] | None = None,
    user_profile: dict[str, Any] | None = None,
    active_nutrient: str | None = None,
) -> dict[str, Any]:
    product = product_context if isinstance(product_context, dict) else {}
    portion = None
    if isinstance(consumption_context, dict) and consumption_context.get("amount") is not None:
        portion = dict(consumption_context)
    result = build_nutrition_insights(product, portion, user_profile, active_nutrient)
    result["engine_status"] = result.get("status")
    result["status"] = "pass"
    result["summary"] = "已依標示基準量計算可推導的營養重點；沒有資料支持的健康門檻不會自行建立。"
    result["knowledge_domain"] = "nutrition_reference"
    return _response(
        result,
        [
            {
                "document": "data/dri_references.json",
                "page": "structured",
                "text": "Official structured Taiwan DRIs records",
                "quote": "Official structured Taiwan DRIs records",
                "score": 1.0,
                "knowledge_domain": "nutrition_reference",
                "source_url": "https://www.hpa.gov.tw/Pages/Detail.aspx?nodeid=4248&pid=12285",
            }
        ],
        {"engine": "Nutrition Insight Engine", "deterministic": True},
    )


def _web_sources(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Adapt the public web schema to FoodGuard's auditable source records."""

    sources: list[dict[str, Any]] = []
    for item in result.get("results", []):
        if not isinstance(item, dict) or not item.get("url"):
            continue
        rank = int(item.get("rank", len(sources) + 1))
        sources.append(
            {
                "document": item.get("publisher") or item.get("title") or "Web source",
                "page": "web",
                "text": item.get("snippet", ""),
                "quote": item.get("snippet", ""),
                "score": max(0.01, 1.0 / rank),
                "knowledge_domain": "web",
                "source_url": item.get("url"),
                "title": item.get("title"),
                "publisher": item.get("publisher"),
                "published_date": item.get("published_date"),
                "retrieved_at": item.get("retrieved_at"),
            }
        )
    return sources


@mcp.tool(
    name="web_search",
    description=(
        "Search the configured web provider and return cleaned title, URL, publisher, "
        "date and snippet records. Use for current information or when local evidence is insufficient."
    ),
)
def web_search(
    query: str,
    domains: list[str] | None = None,
    recency_days: int | None = None,
    max_results: int = 5,
) -> dict[str, Any]:
    result = search_web(query, domains, recency_days, max_results)
    return _response(result, _web_sources(result), {"provider": result.get("provider"), "web_search": True})


@mcp.tool(
    name="fetch_web_page",
    description="Fetch readable text from one HTTP(S) result when a search snippet is not enough.",
)
def fetch_web_page_tool(url: str, max_chars: int = 12000) -> dict[str, Any]:
    result = fetch_web_page(url, max_chars)
    source = []
    if result.get("status") == "ok":
        source.append(
            {
                "document": result.get("title") or result.get("url"),
                "page": "web",
                "text": result.get("text", ""),
                "quote": result.get("text", "")[:1000],
                "score": 1.0,
                "knowledge_domain": "web",
                "source_url": result.get("url"),
            }
        )
    return _response(result, source, {"web_fetch": True})


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
    structured_results = [
        (item, evaluate_claim_rule(item, nutrition)) for item in claims
    ]
    structured_results = [
        (item, result) for item, result in structured_results if result is not None
    ]
    rule_sources = []
    for item, structured in structured_results:
        if not isinstance(structured.get("rule"), dict):
            continue
        # The structured file is an official, versioned rule source. Keep one
        # auditable source per claim, including when several claims are entered.
        rule = structured["rule"]
        rule_sources.append(
            {
                "document": rule.get("source_document", "data/nutrition_claim_rules.json"),
                "page": rule.get("source_page", "structured"),
                "text": (
                    f"官方結構化規則：{rule.get('claim', item)}；"
                    f"每100公克 {rule.get('operator', '')} {rule.get('threshold', {}).get('solid')}"
                    f"{rule.get('threshold_unit', '')}；每100毫升 {rule.get('operator', '')} "
                    f"{rule.get('threshold', {}).get('liquid')} {rule.get('threshold_unit', '')}。"
                ),
                "quote": f"官方結構化營養宣稱規則（{item}）",
                "score": 1.0,
                "knowledge_domain": "regulation",
                "source_url": rule.get("source_url"),
            }
        )
    base_result = analyse_nutrition_claim(normalized_claim, nutrition, sources)
    if rule_sources:
        base_result["structured_rule_sources"] = rule_sources
        # Keep the singular key for older UI/state payloads.
        base_result["structured_rule_source"] = rule_sources[0]
        # A complete label can be compared against the versioned structured
        # rule without RAG. Incomplete input remains insufficient and keeps
        # the recommendation to add data or use Web Search.
        result = base_result
    else:
        result = _with_evidence_guard(base_result, sources)
    # If there is no local structured rule, supplement the answer with
    # official web results when network search is enabled. This does not turn
    # snippets into numeric legal conclusions; it only adds auditable leads.
    # `structured` used to refer to the loop variable above.  When no claim
    # produced a structured rule (for example an unsupported or incomplete
    # claim), that variable was never assigned and the tool raised
    # UnboundLocalError.  The caller then misreported the tool failure as
    # missing RAG evidence.  The presence of structured rule sources is the
    # actual condition needed here.
    if not rule_sources:
        web_result = search_web(
            f"{normalized_claim} 食品營養宣稱 官方規定",
            domains=["fda.gov.tw", "mohw.gov.tw"],
            max_results=5,
        )
        web_sources = _web_sources(web_result)
        if web_sources:
            sources.extend(web_sources)
            result["web_search_status"] = "supplemented"
            result["recommendations"] = ["本機規則不足，以上已補充官方網站搜尋結果；正式判定仍需確認現行規範全文。"]
        else:
            result["web_search_status"] = web_result.get("status", "unavailable")
            result["recommendations"] = ["請擴充本機官方資料庫，或開啟 Web Search 後再重新判讀。"]
    result.update({"query": query, "nutrition_data": nutrition})
    result["knowledge_domain"] = "regulation"
    return _response(result, sources, debug)


def main() -> None:
    """Run the server over the local stdio transport."""

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
