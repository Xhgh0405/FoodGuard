"""MCP protocol client and CLI for FoodGuard.

This client launches ``mcp_server.py`` as a subprocess and communicates only
through the Official MCP Python SDK. It never imports server tool functions.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from dotenv import load_dotenv
from mcp import Client
from mcp.client.stdio import StdioServerParameters
from openai import AsyncOpenAI

from foodguard.synthesis import NOT_ENOUGH_EVIDENCE, synthesize_evidence
from foodguard.context import parse_consumption_amount, parse_exposure_context
from foodguard.health_risk import has_health_risk_signal


PROJECT_ROOT = Path(__file__).resolve().parent

SYSTEM_PROMPT = f"""你是 FoodGuard 食品標示法規助理。

規則：
1. 任何食品法規問題，先選擇適當的 MCP tool 取得本機 RAG 證據，再回答。
2. 只能使用 MCP tool 回傳的 sources 作為法規依據，不得自行補充、猜測或創造法規名稱、門檻、數字。
3. 若 sources 是空的，必須明確說：{NOT_ENOUGH_EVIDENCE}
4. 工具回傳的 evidence_summary 是整理後依據；不要把 raw chunk、PDF 原文、chunk_id 或相似度分數當成回答。
5. 使用者的追問可能省略主詞。請根據完整 conversation history 與目前食品資料理解上下文，再決定 MCP tool 與 query。
6. 對「食品法有哪些」「食品標示有哪些規定」這類廣泛問題，一律先呼叫 search_food_regulation；只能整理搜尋結果明確涵蓋的主題。
7. 如果搜尋結果只涵蓋特定主題，請說明目前只能確認那些主題，不要憑記憶列出未出現在來源中的規定。
8. 使用者若詢問與食品無關的問題，請簡短說明本服務只處理食品標示與營養宣稱。
9. 回答只輸出自然語言：先講結論，再用 3 至 5 點條列重點；不要逐段重複檢索文字。
10. Client 會另外呈現來源，回答中不要自行建立「法規來源」段落，也不要貼出原文。
"""

ANALYSIS_SYSTEM_PROMPT = f"""你是 FoodGuard 食品標示分析助手。

你只能依照提供的 PRODUCT_DATA、RULE_RESULTS 與 SYNTHESIZED_EVIDENCE 進行判斷。
SYNTHESIZED_EVIDENCE 是已去重、分級、濃縮的證據，不是可以照抄的原文。
你必須先理解食品資料，再用最相關的整理後重點支持結論。
不得捏造不存在的法規、標準、門檻、條號或數字。
如果證據不足以支持明確判斷，必須明確說：{NOT_ENOUGH_EVIDENCE}。
回答使用台灣繁體中文，先講整體結論，再用 3 至 5 點條列重點；不要複製原始檢索段落。
保留「可能、建議、需要確認、應」的語氣差異，不要把建議改成強制要求。
營養標示回答聚焦已提供欄位、缺少欄位與需確認項目；宣稱回答聚焦宣稱與輸入數值是否有證據支持；疾病指引只能說明飲食注意事項，不做診斷。
不要在回答中顯示 raw chunk、PDF、chunk_id、embedding 或相似度分數；Client 會另外顯示來源。
"""


class ChatCompletionsClient(Protocol):
    @property
    def chat(self) -> Any: ...


@dataclass(frozen=True)
class ClientResponse:
    answer: str
    sources: list[dict[str, Any]]
    tool_calls: list[str]
    evidence_synthesis: dict[str, Any] | None = None
    diagnostics: dict[str, Any] | None = None


@dataclass(frozen=True)
class LocalToolDescriptor:
    """The small subset of an MCP Tool needed by the OpenAI tool schema."""

    name: str
    description: str
    input_schema: dict[str, Any]


_LOCAL_TOOL_SCHEMAS: dict[str, tuple[str, dict[str, Any]]] = {
    "search_food_regulation": (
        "Search imported official food-regulation sources.",
        {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    ),
    "search_disease_guideline": (
        "Search official disease dietary guidance.",
        {
            "type": "object",
            "properties": {
                "disease": {"type": "string"},
                "query": {"type": "string"},
                "product_context": {"type": "object"},
            },
            "required": ["disease"],
        },
    ),
    "search_health_risk": (
        "Search structured official health-risk evidence and separate hazard from exposure risk.",
        {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "product_context": {"type": "object"},
                "exposure_context": {"type": "object"},
            },
            "required": ["question"],
        },
    ),
    "calculate_consumption_nutrients": (
        "Scale nutrition-label values to an explicitly supplied amount.",
        {
            "type": "object",
            "properties": {
                "nutrition_data": {"type": "object"},
                "consumption_amount": {"type": "number"},
                "consumption_unit": {"type": "string"},
            },
            "required": ["nutrition_data", "consumption_amount", "consumption_unit"],
        },
    ),
    "check_allergens": (
        "Classify possible allergens in the supplied ingredients.",
        {
            "type": "object",
            "properties": {"ingredients": {"oneOf": [{"type": "string"}, {"type": "array"}]}},
            "required": ["ingredients"],
        },
    ),
    "check_nutrition_label": (
        "Check nutrition-label completeness.",
        {"type": "object", "properties": {"nutrition_data": {"type": "object"}}, "required": ["nutrition_data"]},
    ),
    "check_nutrition_claim": (
        "Check a nutrition claim against the structured official rules.",
        {
            "type": "object",
            "properties": {"claim": {"type": "string"}, "nutrition_data": {"type": "object"}},
            "required": ["claim", "nutrition_data"],
        },
    ),
}


def _get_setting(name: str, default: str = "") -> str:
    """Read settings from local environment first, then Streamlit Secrets."""

    value = os.getenv(name, "").strip()
    if value:
        return value
    try:
        import streamlit as st

        return str(st.secrets.get(name, default)).strip()
    except Exception:
        return default


def _load_settings(require_key: bool = True) -> tuple[str, str, str | None]:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = _get_setting("OPENAI_API_KEY")
    if require_key and (not api_key or api_key == "your_openai_api_key_here"):
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Copy .env.example to .env and set the real API key."
        )
    model = _get_setting("LLM_MODEL") or _get_setting("OPENAI_MODEL", "gpt-4o-mini")
    base_url = _get_setting("OPENAI_BASE_URL") or None
    return api_key, model, base_url


def _configured_provider(base_url: str | None = None) -> str:
    explicit = _get_setting("LLM_PROVIDER")
    if explicit:
        return explicit.lower()
    configured_url = base_url or _get_setting("OPENAI_BASE_URL")
    if "11434" in configured_url or "ollama" in configured_url.lower():
        return "ollama"
    return "cloud" if configured_url or _get_setting("OPENAI_API_KEY") else "none"


def _configure_console_encoding() -> None:
    """Keep Windows CLI output from failing on Unicode extracted from PDFs."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            continue


def _tool_to_openai_schema(tool: Any) -> dict[str, Any]:
    schema = getattr(tool, "input_schema", None)
    if schema is None:
        schema = getattr(tool, "inputSchema", {})
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": schema,
        },
    }


def _message_to_dict(message: Any) -> dict[str, Any]:
    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_none=True)
    output: dict[str, Any] = {"role": message.role, "content": message.content}
    if getattr(message, "tool_calls", None):
        output["tool_calls"] = message.tool_calls
    return output


def _payload_from_mcp_result(result: Any) -> dict[str, Any]:
    structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict):
        payload = dict(structured)
    else:
        payload = {}
        for block in getattr(result, "content", []):
            text = getattr(block, "text", None)
            if not text:
                continue
            try:
                candidate = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                payload = candidate
                break

    if not isinstance(payload.get("sources"), list):
        payload["sources"] = []
    if "result" not in payload:
        payload["result"] = {
            "status": "mcp_error" if getattr(result, "is_error", False) else "unknown",
            "message": "MCP tool did not return a structured result.",
        }
    return payload


def _collect_sources(payload: dict[str, Any], target: list[dict[str, Any]]) -> None:
    for source in payload.get("sources", []):
        if not isinstance(source, dict):
            continue
        if not all(key in source for key in ("document", "page", "text", "score")):
            continue
        if source not in target:
            target.append(dict(source))


def _looks_like_food_question(message: str) -> bool:
    """Identify food-related messages that deserve a retrieval safety net."""

    keywords = (
        "食品",
        "食物",
        "食安",
        "標示",
        "營養",
        "過敏",
        "宣稱",
        "成分",
        "糖",
        "蛋白質",
        "脂肪",
        "鈉",
        "熱量",
    )
    return any(keyword in message for keyword in keywords)


def _fallback_query(history: list[dict[str, Any]], current: str) -> str:
    """Keep omitted subjects, such as '那糖呢？', in a fallback query."""

    previous_questions = [
        str(item.get("content", ""))
        for item in history
        if item.get("role") == "user" and item.get("content")
    ]
    if previous_questions and previous_questions[-1].strip() == current.strip():
        previous_questions.pop()
    previous_questions = previous_questions[-1:]
    return "；".join(previous_questions + [current]).strip()


def _fallback_intent(message: str) -> str:
    """Route common questions safely when no generative model is available."""

    compact = re.sub(r"\s+", "", message)
    if parse_consumption_amount(message):
        return "consumption"
    if any(
        term in compact
        for term in (
            "致癌", "癌症", "癌", "健康風險", "風險", "安全嗎", "長期吃", "添加物安全", "污染物",
            "加工肉", "阿斯巴甜", "黃麴毒素", "丙烯醯胺", "亞硝酸鹽", "亞硝胺",
            "燒焦", "焦黑", "燒烤", "煙燻", "醃製",
        )
    ):
        return "health_risk"
    if any(term in compact for term in ("一天", "每日", "一天最多", "幾個", "幾份", "可以吃多少")):
        return "intake"
    if any(term in compact for term in ("過敏", "過敏原", "奶類", "乳類", "雞蛋", "蛋類")):
        return "allergen"
    if any(term in compact for term in ("宣稱", "高蛋白", "低鈉", "無糖", "零糖", "高纖", "低脂")):
        return "claim"
    if any(term in compact for term in ("營養標示", "標示完整", "缺少欄位")):
        return "nutrition_label"
    if any(term in compact for term in ("高血壓", "糖尿病", "腎臟病", "腎病", "高血脂", "血脂")):
        return "health_guidance"
    return "search"


def _claim_from_question(message: str, current_claims: list[str]) -> str:
    match = re.search(
        r"(高|低|無|零|不含|富含|多)\s*(蛋白質|蛋白|膳食纖維|纖維|糖|鈉|脂肪|鈣|鐵)",
        message,
    )
    if match:
        nutrient = {"蛋白": "蛋白質", "纖維": "膳食纖維"}.get(match.group(2), match.group(2))
        return f"{match.group(1)}{nutrient}"
    return "、".join(current_claims)


def _intake_clarification(product_data: dict[str, Any] | None) -> str:
    product = product_data or {}
    product_name = str(product.get("product_name") or "這項食品")
    nutrition = product.get("nutrition", {}) if isinstance(product.get("nutrition"), dict) else {}
    serving_size = nutrition.get("serving_size")
    serving_note = (
        f"目前營養標示只有每份量 {serving_size}，仍不知道一份等於幾個。"
        if serving_size
        else "目前也缺少每份重量或每份包含幾個。"
    )
    return (
        f"目前無法只依「成年男性」判定一天最多可以吃幾個{product_name}。\n\n"
        f"- {serving_note}\n"
        "- 還需要確認你想控制的是熱量、糖、鈉、脂肪或其他營養素。\n"
        "- 個人疾病、用藥與飲食目標也會影響建議，系統不能直接做醫療診斷。\n\n"
        "請補充「每個重量或一份有幾個」以及想控制的營養項目，我才能依官方資料換算。"
    )


def _allergen_answer(question: str, result: dict[str, Any]) -> str:
    detected = [item for item in result.get("detected_allergens", []) if isinstance(item, dict)]
    if not detected:
        return str(result.get("summary") or NOT_ENOUGH_EVIDENCE)

    focused = detected
    if any(term in question for term in ("奶", "乳")):
        focused = [item for item in detected if "奶" in str(item.get("category", ""))] or detected
    elif "蛋" in question:
        focused = [item for item in detected if "蛋" in str(item.get("category", ""))] or detected

    descriptions = []
    for item in focused:
        ingredients = "、".join(str(value) for value in item.get("source_ingredients", []))
        descriptions.append(f"{ingredients or '相關成分'} → {item.get('category', '未分類過敏原')}")

    lines = ["依目前產品成分，系統辨識到：", *[f"- {item}" for item in descriptions]]
    if "為什麼" in question:
        lines.extend(
            [
                "",
                "這裡代表該成分被歸入官方標示需注意的過敏原類別，不代表每個人食用後都會過敏。",
                "現有法規來源主要說明標示要求；若要解釋個人的過敏原因，仍需要醫療專業評估。",
            ]
        )
    else:
        lines.append("請再確認包裝是否清楚標示相關過敏原警語。")
    return "\n".join(lines)


def _health_risk_answer(result: dict[str, Any], product: dict[str, Any] | None = None) -> str:
    topics = [item for item in result.get("risk_topics", []) if isinstance(item, dict)]
    product_name = str((product or {}).get("product_name") or "目前食品")
    exposure = result.get("exposure_context") or {}
    if not topics:
        return (
            f"目前沒有足夠資料把「{product_name}」對應到特定健康風險主題。\n"
            "- 這不等於已證明安全，也不代表一定有致癌物。\n"
            "- 請提供具體成分、食品類型、加工方式或檢驗資料，才能查詢官方評估。"
        )

    lines = ["【目前判讀】"]
    for topic in topics:
        classification = topic.get("classification_label") or "目前沒有單一 IARC 分類"
        lines.append(
            f"- {topic.get('agent', topic.get('topic'))}：{classification}。"
            "這是 hazard（是否具有造成危害的證據）分類，不是吃一次就會罹癌的個人 risk。"
        )
    lines.append("\n【為什麼】")
    for topic in topics[:3]:
        lines.append(f"- {topic.get('evidence_summary', '')}")
    lines.append("\n【目前產品資料】")
    for topic in topics[:4]:
        status = topic.get("detection_status", "possible")
        label = {"detected": "產品／問題中有明確訊號", "possible": "只有食品或製程脈絡，尚未證明實際含有", "not_enough_evidence": "資料不足"}.get(status, status)
        lines.append(f"- {topic.get('topic')}: {label}")
    known_exposure = [
        f"份量 {exposure.get('amount')} {exposure.get('unit')}"
        if exposure.get("amount") is not None and exposure.get("unit") else None,
        f"頻率 {exposure.get('frequency')}" if exposure.get("frequency") else None,
        f"期間 {exposure.get('duration')}" if exposure.get("duration") else None,
        f"料理方式 {exposure.get('preparation_method')}" if exposure.get("preparation_method") else None,
    ]
    lines.append("- 暴露資訊：" + ("、".join(item for item in known_exposure if item) if any(known_exposure) else "目前未提供份量、頻率或期間"))
    lines.append("\n【限制】")
    lines.append("- 目前只能說明官方 hazard 證據與可能暴露脈絡，不能依這些資料估算你的個人罹癌機率。")
    if not exposure.get("frequency") or not exposure.get("duration"):
        lines.append("- 若要進一步討論實際風險，仍需補充食用頻率與持續期間；沒有被 IARC 分類也不等於已證明安全。")
    return "\n".join(lines)


def _structured_fallback_answer(
    intent: str,
    question: str,
    payload: dict[str, Any],
    product: dict[str, Any] | None = None,
) -> str:
    result = payload.get("result", {}) if isinstance(payload.get("result"), dict) else {}
    if intent == "consumption":
        if result.get("status") != "calculated":
            return str(result.get("message") or NOT_ENOUGH_EVIDENCE)
        labels = {
            "sugar_g": "糖", "carbohydrate_g": "碳水化合物", "protein_g": "蛋白質",
            "sodium_mg": "鈉", "fat_g": "脂肪", "calories_kcal": "熱量",
        }
        product_name = str((product or {}).get("product_name") or "目前食品")
        lines = [f"依目前食品「{product_name}」標示，{result.get('amount'):g}{result.get('unit')}的換算結果為："]
        lines.extend(
            f"- {labels[field]}：{value:g}"
            for field, value in result.get("scaled_values", {}).items()
            if field in labels
        )
        disease_result = payload.get("disease_result", {})
        if isinstance(disease_result, dict) and disease_result.get("sources"):
            disease_name = disease_result.get("result", {}).get("disease") or "疾病"
            lines.append(
                f"已同時帶入{disease_name}官方飲食指引；上述數值只能用來比較此次攝取量，"
                "不能直接推導個人的每日安全上限。"
            )
        else:
            lines.append("這是依標示基準量的比例換算，不代表個人的每日安全上限。")
        lines.append("是否適合仍須配合整餐碳水化合物、用藥、血糖與醫囑判斷。")
        return "\n".join(lines)
    if intent == "allergen":
        return _allergen_answer(question, result)
    if intent == "health_risk":
        return _health_risk_answer(result, product)
    if intent == "nutrition_label":
        missing = "、".join(str(item) for item in result.get("missing_fields", []))
        answer = str(result.get("summary") or NOT_ENOUGH_EVIDENCE)
        return f"{answer}\n- 需要補充或確認：{missing}" if missing else answer
    if intent == "claim":
        evaluation = result.get("numeric_evaluation")
        answer = str(result.get("summary") or NOT_ENOUGH_EVIDENCE)
        if isinstance(evaluation, dict):
            return (
                f"{answer}\n- 比較基準：{evaluation.get('basis', '來源條件')}\n"
                f"- 輸入值：{evaluation.get('actual')}\n"
                f"- 來源門檻：{evaluation.get('comparison')} {evaluation.get('threshold')}"
            )
        return answer

    sources = payload.get("sources", [])
    if sources:
        titles: list[str] = []
        for source in sources:
            title = Path(str(source.get("document", ""))).stem
            if title and title not in titles:
                titles.append(title)
        topics = "、".join(titles[:4])
        if intent == "health_guidance":
            disease = next(
                (
                    term
                    for term in ("糖尿病", "高血壓", "腎臟病", "高血脂")
                    if term in question
                ),
                "目前疾病",
            )
            product = product or {}
            product_name = str(product.get("product_name") or "").strip()
            nutrition = product.get("nutrition", {})
            nutrition = nutrition if isinstance(nutrition, dict) else {}
            values = nutrition.get("values", {})
            values = values if isinstance(values, dict) else {}
            value_labels = (
                ("糖", "sugar_g"),
                ("碳水化合物", "carbohydrate_g"),
                ("份量", "serving_size"),
            )
            provided = []
            for label, field in value_labels:
                value = nutrition.get(field) if field == "serving_size" else values.get(field)
                if value not in (None, ""):
                    unit = "" if field == "serving_size" else " g"
                    provided.append(f"{label} {value}{unit}")
            missing = [
                label
                for label, field in value_labels
                if (nutrition.get(field) if field == "serving_size" else values.get(field)) in (None, "")
            ]
            ingredients = "、".join(str(item) for item in product.get("ingredients", []))
            sugar_terms = ("砂糖", "蔗糖", "葡萄糖", "果糖", "糖漿", "蜂蜜", "麥芽糊精")
            ingredient_note = (
                "成分文字中出現糖類相關原料，需一併看每份糖與碳水化合物。"
                if any(term in ingredients for term in sugar_terms)
                else "目前成分文字未足以判定其糖分影響。"
            )
            if not product_name:
                return (
                    f"已找到與{disease}相關的官方飲食資料，但目前沒有可比對的產品。"
                    "\n- 請先填入產品名稱、成分、每份份量、糖與碳水化合物。"
                    "\n- 疾病資料能提供飲食注意事項，不能單獨把所有飲品判定為可以或不可以。"
                )
            if missing:
                return (
                    f"針對目前產品「{product_name}」，已找到與{disease}相關的官方飲食資料，"
                    f"但目前缺少：{ '、'.join(missing) }，所以還不能可靠判定是否適合飲用。"
                    f"\n- 已提供：{ '、'.join(provided) if provided else '尚未提供糖尿病判斷所需的營養資料' }。"
                    f"\n- {ingredient_note}"
                    "\n- 請補齊標示資料，並依個人用藥與醫囑決定份量；這不取代醫療診斷。"
                )
            return (
                f"針對目前產品「{product_name}」，目前輸入的每份資料為：{'、'.join(provided)}。"
                f"\n- {ingredient_note}"
                f"\n- 官方{disease}資料可用來提供飲食注意事項，但沒有一個適用所有人的單一「可以喝／不能喝」判定。"
                "\n- 是否適合仍要配合飲用份量、整餐碳水化合物、用藥與個人醫囑；不能只用產品名稱下結論。"
            )
        return f"目前可查到與問題相關的官方資料主題：{topics}。請把問題縮小到特定標示、成分或營養宣稱。"
    return NOT_ENOUGH_EVIDENCE


def _synthesize_payloads(
    payloads: list[dict[str, Any]],
    product_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine MCP evidence before it is placed in an LLM prompt."""

    evidence: list[dict[str, Any]] = []
    for payload in payloads:
        result = payload.get("result", {})
        domain = result.get("knowledge_domain") if isinstance(result, dict) else None
        for source in payload.get("sources", []):
            if not isinstance(source, dict):
                continue
            record = dict(source)
            if domain and not record.get("knowledge_domain"):
                record["knowledge_domain"] = domain
            evidence.append(record)
    return synthesize_evidence(evidence, product_context=product_context)


def _llm_tool_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Expose only synthesized evidence to the LLM, keeping raw evidence internal."""

    return {
        "result": payload.get("result", {}),
        "evidence_summary": _synthesize_payloads([payload]),
    }


def _answer_with_sources(answer: str, sources: list[dict[str, Any]]) -> str:
    """Keep evidence structured separately from the public answer text."""

    answer = _sanitize_answer(answer, sources)
    if not answer:
        answer = NOT_ENOUGH_EVIDENCE
    if not sources:
        if NOT_ENOUGH_EVIDENCE not in answer:
            answer = f"{answer}\n\n{NOT_ENOUGH_EVIDENCE}".strip()
        return answer
    return answer


FOLLOWUP_SYNTHESIS_PROMPT = """你是 FoodGuard 的回答整理器。
只能使用輸入中的產品資料、消費量計算結果與官方指引摘要；不要創造門檻、醫療診斷或個人每日上限。
回答台灣繁體中文，先講與目前產品直接相關的結論，再列出 2 至 4 個重點。
若資料不足，要指出缺少哪個欄位；若有計算結果，必須保留數值與單位。
不要輸出原始 chunk、相似度、chain-of-thought 或「資料來源」段落。
"""


async def _llm_followup_answer(
    llm: Any,
    model: str,
    question: str,
    intent: str,
    product: dict[str, Any],
    payload: dict[str, Any],
    history: list[dict[str, Any]],
) -> str | None:
    """Ask the configured model to phrase already-verified follow-up facts."""

    disease_payload = payload.get("disease_result")
    evidence_payloads = [payload]
    if isinstance(disease_payload, dict):
        evidence_payloads.append(disease_payload)
    prompt = {
        "intent": intent,
        "new_user_message": question,
        "current_product": product,
        "relevant_tool_results": {
            "calculation": payload.get("result", {}),
            "disease_guideline": (
                disease_payload.get("result", {})
                if isinstance(disease_payload, dict)
                else None
            ),
        },
        "evidence_summary": _synthesize_payloads(evidence_payloads, product),
        "recent_conversation_history": [
            item for item in history[-8:] if item.get("role") in {"user", "assistant"}
        ],
    }
    try:
        completion = await llm.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": FOLLOWUP_SYNTHESIS_PROMPT},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            temperature=0,
            max_tokens=280,
        )
        answer = (getattr(completion.choices[0].message, "content", "") or "").strip()
        return answer or None
    except Exception:
        return None


def _sanitize_answer(answer: str, sources: list[dict[str, Any]]) -> str:
    """Prevent an accidental verbatim long-chunk echo in the public answer."""

    cleaned = (answer or "").strip()
    for source in sources:
        raw_text = " ".join(str(source.get("text", "")).split())
        if len(raw_text) >= 80 and raw_text in cleaned:
            cleaned = cleaned.replace(raw_text, "（依官方資料整理）")
    return cleaned


def _deterministic_analysis_summary(
    product_data: dict[str, Any], task_results: dict[str, dict[str, Any]]
) -> str:
    """Provide a safe local summary if an LLM is unavailable."""

    product_name = product_data.get("product_name") or "此食品"
    allergen = task_results.get("allergens", {}).get("result", {})
    nutrition = task_results.get("nutrition_label", {}).get("result", {})
    claim = task_results.get("nutrition_claim", {}).get("result", {})
    lines = [f"{product_name}的分析重點："]
    lines.append(f"- {allergen.get('summary', NOT_ENOUGH_EVIDENCE)}")
    lines.append(f"- {nutrition.get('summary', NOT_ENOUGH_EVIDENCE)}")
    lines.append(f"- {claim.get('summary', NOT_ENOUGH_EVIDENCE)}")
    health_risk = task_results.get("health_risk", {}).get("result", {})
    if health_risk:
        lines.append(f"- 健康風險：{health_risk.get('summary', NOT_ENOUGH_EVIDENCE)}")
    return "\n".join(lines)


class FoodGuardMCPClient:
    """A stateful client for FoodGuard MCP.

    Stdio is the normal transport.  Some Windows-hosted Streamlit runtimes
    deny the overlapped named pipes used by the MCP SDK, though.  In that
    case we keep the same client orchestration and call the exact server tool
    functions in-process as a transport fallback.  This is still the real
    parsing -> RAG -> rule pipeline; it only avoids the blocked pipe.
    """

    def __init__(
        self,
        server_script: Path | str = PROJECT_ROOT / "mcp_server.py",
        model: str | None = None,
        llm_client: ChatCompletionsClient | None = None,
        max_tool_rounds: int = 4,
        require_api_key: bool = True,
    ) -> None:
        api_key, configured_model, base_url = _load_settings(
            require_key=require_api_key and llm_client is None
        )
        self.model = model or configured_model
        self.server_script = Path(server_script).resolve()
        self.max_tool_rounds = max_tool_rounds
        self.history: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self._mcp: Client | None = None
        self._tools: list[Any] = []
        self._local_tools: dict[str, Any] = {}
        self.transport_mode = "stdio"
        self.transport_fallback_reason: str | None = None
        self._llm = llm_client
        self.provider = _configured_provider(base_url)
        self.current_product: dict[str, Any] | None = None
        self.current_analysis: dict[str, Any] | None = None
        self.consumption_context: dict[str, Any] | None = None
        self._last_llm_used = False
        self._last_parsed_followup: dict[str, Any] | None = None
        self._last_previous_context: str | None = None
        self.exposure_context: dict[str, Any] = {
            "amount": None,
            "unit": None,
            "frequency": None,
            "duration": None,
            "preparation_method": None,
        }
        self._last_health_risk_result: dict[str, Any] | None = None
        if self._llm is None and api_key:
            try:
                timeout = float(_get_setting("OPENAI_TIMEOUT", "20"))
            except ValueError:
                timeout = 20.0
            self._llm = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                max_retries=0,
            )

    def _diagnostics(
        self,
        *,
        intent: str | None,
        tool_calls: list[str],
        sources: list[dict[str, Any]],
        answer_source: str,
        fallback_reason: str | None = None,
        llm_used: bool | None = None,
    ) -> dict[str, Any]:
        domains = sorted({str(item.get("knowledge_domain")) for item in sources if item.get("knowledge_domain")})
        result: dict[str, Any] = {
            "llm_used": self._last_llm_used if llm_used is None else llm_used,
            "llm_provider": self.provider,
            "llm_model": self.model if self.provider != "none" else None,
            "current_product": (self.current_product or {}).get("product_name"),
            "conversation_turns": sum(1 for item in self.history if item.get("role") == "user"),
            "conversation_messages": len(self.history),
            "intent": intent,
            "mcp_tools_called": list(tool_calls),
            "mcp_transport": self.transport_mode,
            "rag_domains": domains,
            "evidence_count": len(sources),
            "evidence_status": "available" if sources else "insufficient",
            "answer_source": answer_source,
        }
        if self._last_parsed_followup:
            result["parsed_follow_up"] = dict(self._last_parsed_followup)
        if self._last_previous_context:
            result["previous_context"] = self._last_previous_context
        if self.consumption_context:
            result["consumption_context"] = dict(self.consumption_context)
        health_result = self._last_health_risk_result or {}
        if health_result:
            result["health_risk_intent"] = True
            result["risk_topic"] = [item.get("topic") for item in health_result.get("risk_topics", [])]
            result["hazard_classification"] = health_result.get("hazard_classifications", [])
            result["evidence_category"] = health_result.get("evidence_category")
            result["exposure_context"] = dict(self.exposure_context)
            result["health_risk_sources"] = health_result.get("source_organizations", [])
        else:
            result["health_risk_intent"] = intent == "health_risk"
        if fallback_reason:
            result["fallback_reason"] = fallback_reason
        return result

    def set_current_context(
        self,
        product_data: dict[str, Any] | None = None,
        analysis: dict[str, Any] | None = None,
    ) -> None:
        """Attach the active product to follow-up questions without raw evidence."""

        self.current_product = copy.deepcopy(product_data) if product_data else None
        self.current_analysis = copy.deepcopy(analysis) if analysis else None
        if self.current_product is not None:
            nutrition = self.current_product.get("nutrition")
            if isinstance(nutrition, dict) and nutrition.get("nutrition_basis"):
                self.current_product["nutrition_basis"] = copy.deepcopy(
                    nutrition["nutrition_basis"]
                )
            if self.current_analysis:
                mappings = {
                    "allergens": "allergen_analysis",
                    "nutrition_label": "label_analysis",
                    "nutrition_claim": "claim_analysis",
                }
                for source_key, target_key in mappings.items():
                    payload = self.current_analysis.get(source_key)
                    if isinstance(payload, dict):
                        self.current_product[target_key] = copy.deepcopy(
                            payload.get("result", payload)
                        )
                health_payload = self.current_analysis.get("health_risk")
                if isinstance(health_payload, dict):
                    self.current_product["health_context_analysis"] = copy.deepcopy(
                        health_payload.get("result", health_payload)
                    )
                for key in ("health_context_analysis", "consumption_context"):
                    if key in self.current_analysis:
                        self.current_product[key] = copy.deepcopy(self.current_analysis[key])
            context = self.current_product.get("consumption_context")
            self.consumption_context = copy.deepcopy(context) if isinstance(context, dict) else None
            exposure = self.current_product.get("exposure_context")
            if isinstance(exposure, dict):
                self.exposure_context.update(copy.deepcopy(exposure))

    def _add_context_message(self) -> None:
        if not self.current_product and not self.current_analysis:
            return
        context = {
            "product": self.current_product or {},
            "analysis": {
                name: payload.get("result", {})
                for name, payload in (self.current_analysis or {}).items()
                if isinstance(payload, dict) and name in {"allergens", "nutrition_label", "nutrition_claim"}
            },
        }
        message = {
            "role": "system",
            "content": (
                "目前對話的食品資料與既有判讀結果如下。追問省略主詞時，優先以此脈絡理解；"
                "若要補充法規，仍須呼叫 MCP tool。\n"
                + json.dumps(context, ensure_ascii=False)
            ),
        }
        if any(item.get("content") == message["content"] for item in self.history):
            return
        insert_at = 1 if self.history and self.history[0].get("role") == "system" else 0
        self.history.insert(insert_at, message)

    async def __aenter__(self) -> "FoodGuardMCPClient":
        parameters = StdioServerParameters(
            command=sys.executable,
            args=[str(self.server_script)],
            cwd=PROJECT_ROOT,
        )
        self._mcp = Client(parameters)
        try:
            await self._mcp.__aenter__()
        except (PermissionError, FileNotFoundError) as exc:
            # Windows sandboxed/hosted Streamlit processes can reject the
            # named pipe that the official stdio transport creates.  Do not
            # turn that infrastructure error into three false "no evidence"
            # product results: use the same server functions locally.
            self._mcp = None
            self._local_tools = self._load_local_tools()
            self.transport_mode = "in_process"
            self.transport_fallback_reason = f"{type(exc).__name__}: {exc}"
        await self.refresh_tools()
        return self

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self._mcp is not None:
            await self._mcp.__aexit__(exc_type, exc_value, traceback)
            self._mcp = None
        self._local_tools = {}

    @property
    def conversation_history(self) -> list[dict[str, Any]]:
        return list(self.history)

    @property
    def tool_names(self) -> list[str]:
        return [str(getattr(tool, "name", "")) for tool in self._tools]

    async def refresh_tools(self) -> list[str]:
        if self._mcp is None and not self._local_tools:
            raise RuntimeError("MCP client is not connected.")
        if self._mcp is not None:
            listed = await self._mcp.list_tools()
            self._tools = list(listed.tools)
        else:
            self._tools = [
                LocalToolDescriptor(name, description, schema)
                for name, (description, schema) in _LOCAL_TOOL_SCHEMAS.items()
                if name in self._local_tools
            ]
        return self.tool_names

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self._mcp is None and not self._local_tools:
            raise RuntimeError("MCP client is not connected.")
        if name not in self.tool_names:
            return {
                "result": {"status": "unknown_tool", "message": f"MCP tool not available: {name}"},
                "sources": [],
            }
        if self._mcp is not None:
            mcp_result = await self._mcp.call_tool(name, arguments)
            return _payload_from_mcp_result(mcp_result)
        result = self._local_tools[name](**arguments)
        if hasattr(result, "__await__"):
            result = await result
        if not isinstance(result, dict):
            raise TypeError(f"Local MCP tool {name} returned a non-object result.")
        return result

    @staticmethod
    def _load_local_tools() -> dict[str, Any]:
        """Load the server's registered tool functions for pipe-free fallback."""

        from mcp_server import (
            calculate_consumption_nutrients,
            check_allergens,
            check_nutrition_claim,
            check_nutrition_label,
            search_disease_guideline,
            search_food_regulation,
            search_health_risk,
        )

        return {
            "search_food_regulation": search_food_regulation,
            "check_allergens": check_allergens,
            "check_nutrition_label": check_nutrition_label,
            "check_nutrition_claim": check_nutrition_claim,
            "calculate_consumption_nutrients": calculate_consumption_nutrients,
            "search_disease_guideline": search_disease_guideline,
            "search_health_risk": search_health_risk,
        }

    async def summarize_analysis(
        self,
        product_data: dict[str, Any],
        task_results: dict[str, dict[str, Any]],
    ) -> str:
        """Use the configured LLM to summarize structured rule results safely."""

        self._last_llm_used = False
        fallback = _deterministic_analysis_summary(product_data, task_results)
        if self._llm is None:
            return fallback

        evidence_summary = _synthesize_payloads(list(task_results.values()), product_data)
        prompt_payload = {
            "PRODUCT_DATA": product_data,
            "RULE_RESULTS": {
                name: payload.get("result", {})
                for name, payload in task_results.items()
            },
            "SYNTHESIZED_EVIDENCE": evidence_summary,
        }
        try:
            completion = await self._llm.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(prompt_payload, ensure_ascii=False),
                    },
                ],
                temperature=0,
                max_tokens=350,
            )
            answer = (getattr(completion.choices[0].message, "content", "") or "").strip()
            self._last_llm_used = bool(answer)
            return answer or fallback
        except Exception:
            self._last_llm_used = False
            return fallback

    async def _ask_without_llm(
        self, user_message: str, *, append_user: bool = True
    ) -> ClientResponse:
        """Route to structured tools and answer safely without a paid LLM."""

        self._last_llm_used = False
        self._last_parsed_followup = None
        self._last_health_risk_result = None
        previous_users = [
            str(item.get("content", ""))
            for item in self.history
            if item.get("role") == "user" and item.get("content")
        ]
        self._last_previous_context = previous_users[-1] if previous_users else None
        contextual_query = _fallback_query(self.history, user_message)
        intent = _fallback_intent(user_message)
        if intent == "search" and (
            len(user_message.strip()) <= 12
            or _fallback_intent(contextual_query) == "health_risk"
        ):
            contextual_intent = _fallback_intent(contextual_query)
            if contextual_intent != "search":
                intent = contextual_intent
        if append_user:
            self.history.append({"role": "user", "content": user_message})
        sources: list[dict[str, Any]] = []
        called_tools: list[str] = []
        payload: dict[str, Any] = {"result": {}, "sources": []}

        if intent == "intake":
            answer = _intake_clarification(self.current_product)
        else:
            product = self.current_product or {}
            tool_name = "search_food_regulation"
            arguments: dict[str, Any] = {"query": contextual_query}
            if intent == "allergen":
                ingredients = product.get("ingredients", [])
                if not ingredients:
                    answer = "請先提供食品成分，才能辨識可能的過敏原。"
                    answer = _answer_with_sources(answer, sources)
                    self.history.append({"role": "assistant", "content": answer})
                    return ClientResponse(
                        answer=answer,
                        sources=sources,
                        tool_calls=called_tools,
                        evidence_synthesis=synthesize_evidence(sources),
                        diagnostics=self._diagnostics(
                            intent=intent,
                            tool_calls=called_tools,
                            sources=sources,
                            answer_source="fallback",
                        ),
                    )
                tool_name = "check_allergens"
                arguments = {"ingredients": ingredients}
            elif intent == "nutrition_label":
                tool_name = "check_nutrition_label"
                arguments = {"nutrition_data": product.get("nutrition", {})}
            elif intent == "claim":
                tool_name = "check_nutrition_claim"
                arguments = {
                    "claim": _claim_from_question(
                        contextual_query, list(product.get("claims", []))
                    ),
                    "nutrition_data": product.get("nutrition", {}),
                }
            elif intent == "consumption":
                parsed_amount = parse_consumption_amount(user_message)
                if not parsed_amount:
                    answer = "請提供明確的消費量，例如 2000 ml 或 500 公克。"
                    self.history.append({"role": "assistant", "content": answer})
                    return ClientResponse(
                        answer=answer,
                        sources=[],
                        tool_calls=[],
                        evidence_synthesis=synthesize_evidence([]),
                        diagnostics=self._diagnostics(
                            intent=intent, tool_calls=[], sources=[], answer_source="fallback"
                        ),
                    )
                self._last_parsed_followup = {
                    "consumption_amount": parsed_amount["amount"],
                    "consumption_unit": parsed_amount["unit"],
                    "raw": parsed_amount["raw"],
                }
                self.consumption_context = dict(self._last_parsed_followup)
                if self.current_product is not None:
                    self.current_product["consumption_context"] = copy.deepcopy(
                        self.consumption_context
                    )
                tool_name = "calculate_consumption_nutrients"
                arguments = {
                    "nutrition_data": product.get("nutrition", {}),
                    "consumption_amount": parsed_amount["amount"],
                    "consumption_unit": parsed_amount["unit"],
                }
            elif intent == "health_guidance":
                disease = next(
                    (term for term in ("糖尿病", "高血壓", "腎臟病", "高血脂") if term in contextual_query),
                    "",
                )
                tool_name = "search_disease_guideline"
                arguments = {
                    "disease": disease,
                    "query": contextual_query,
                    "product_context": product,
                }
            elif intent == "health_risk":
                self.exposure_context = parse_exposure_context(
                    user_message, self.exposure_context
                )
                if self.current_product is not None:
                    self.current_product["exposure_context"] = copy.deepcopy(
                        self.exposure_context
                    )
                tool_name = "search_health_risk"
                arguments = {
                    "question": contextual_query,
                    "product_context": product,
                    "exposure_context": self.exposure_context,
                }

            try:
                payload = await self.call_tool(tool_name, arguments)
                called_tools.append(tool_name)
                _collect_sources(payload, sources)
                if intent == "consumption":
                    previous_context = contextual_query
                    disease = next(
                        (
                            term
                            for term in ("糖尿病", "高血壓", "腎臟病", "高血脂")
                            if term in previous_context
                        ),
                        "",
                    )
                    if disease:
                        disease_payload = await self.call_tool(
                            "search_disease_guideline",
                            {
                                "disease": disease,
                                "query": previous_context,
                                "product_context": product,
                            },
                        )
                        payload["disease_result"] = disease_payload
                        called_tools.append("search_disease_guideline")
                        _collect_sources(disease_payload, sources)
                if intent == "health_risk":
                    self._last_health_risk_result = copy.deepcopy(payload.get("result", {}))
            except Exception:
                payload = {
                    "result": {"status": "insufficient_evidence", "summary": NOT_ENOUGH_EVIDENCE},
                    "sources": [],
                }
            answer = _structured_fallback_answer(
                intent, contextual_query, payload, product=product
            )

        answer_source = "rule_engine" if intent == "consumption" else ("mixed" if called_tools else "fallback")
        if intent in {"health_guidance", "health_risk", "consumption"} and self._llm is not None:
            generated = await _llm_followup_answer(
                self._llm,
                self.model,
                user_message,
                intent,
                product,
                payload,
                self.history,
            )
            if generated:
                answer = generated
                self._last_llm_used = True
                answer_source = "llm"
        if intent == "consumption" and payload.get("result", {}).get("status") == "calculated":
            # The calculation is a deterministic product-label operation and
            # does not require a RAG source; only sanitize any model wording.
            answer = _sanitize_answer(answer, sources) or NOT_ENOUGH_EVIDENCE
        else:
            answer = _answer_with_sources(answer, sources)
        self.history.append({"role": "assistant", "content": answer})
        return ClientResponse(
            answer=answer,
            sources=sources,
            tool_calls=called_tools,
            evidence_synthesis=synthesize_evidence(sources),
            diagnostics=self._diagnostics(
                intent=intent,
                tool_calls=called_tools,
                sources=sources,
                answer_source=answer_source,
            ),
        )

    async def _safe_analysis_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Keep one MCP task failure from cancelling the whole product analysis."""

        try:
            return await self.call_tool(name, arguments)
        except Exception as exc:
            return {
                "result": {
                    "status": "insufficient_evidence",
                    "title": name,
                    "summary": NOT_ENOUGH_EVIDENCE,
                    "findings": [],
                    "reasoning": "此項目暫時無法取得工具結果，因此不做猜測性判定。",
                    "recommendations": ["請確認規範查核服務與官方資料來源可正常使用。"],
                    "message": NOT_ENOUGH_EVIDENCE,
                },
                "sources": [],
                "debug_evidence": {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            }

    async def analyze_product(self, product_data: dict[str, Any]) -> dict[str, Any]:
        """Run the three MCP checks, then synthesize their structured results."""

        ingredients = product_data.get("ingredients", [])
        nutrition = product_data.get("nutrition", {})
        claims = product_data.get("claims", [])
        task_results = {
            "allergens": await self._safe_analysis_tool(
                "check_allergens", {"ingredients": ingredients}
            ),
            "nutrition_label": await self._safe_analysis_tool(
                "check_nutrition_label", {"nutrition_data": nutrition}
            ),
            "nutrition_claim": await self._safe_analysis_tool(
                "check_nutrition_claim",
                {"claim": "、".join(claims), "nutrition_data": nutrition},
            ),
        }
        if has_health_risk_signal(product_data):
            task_results["health_risk"] = await self._safe_analysis_tool(
                "search_health_risk",
                {
                    "question": "目前食品可能涉及哪些官方健康風險主題？",
                    "product_context": product_data,
                    "exposure_context": {},
                },
            )
        self.current_product = copy.deepcopy(product_data)
        self.current_analysis = copy.deepcopy(task_results)
        overall_summary = await self.summarize_analysis(product_data, task_results)
        evidence_synthesis = _synthesize_payloads(list(task_results.values()), product_data)
        all_sources = [
            source
            for payload in task_results.values()
            for source in payload.get("sources", [])
        ]
        return {
            "product_data": product_data,
            **task_results,
            "overall_summary": overall_summary,
            "evidence_synthesis": evidence_synthesis,
            "tool_calls": [
                "check_allergens",
                "check_nutrition_label",
                "check_nutrition_claim",
                *(["search_health_risk"] if "health_risk" in task_results else []),
            ],
            "diagnostics": self._diagnostics(
                intent="product_analysis",
                tool_calls=[
                    "check_allergens",
                    "check_nutrition_label",
                    "check_nutrition_claim",
                    *(["search_health_risk"] if "health_risk" in task_results else []),
                ],
                sources=all_sources,
                answer_source="llm" if self._llm else "mixed",
            ),
        }

    async def ask(self, user_message: str) -> ClientResponse:
        if self._mcp is None and not self._local_tools:
            raise RuntimeError("MCP client is not connected.")
        if not user_message.strip():
            raise ValueError("user_message must not be empty")

        self._last_llm_used = False
        self._last_parsed_followup = None
        previous_users = [
            str(item.get("content", ""))
            for item in self.history
            if item.get("role") == "user" and item.get("content")
        ]
        self._last_previous_context = previous_users[-1] if previous_users else None
        self._add_context_message()
        # The small local model is slow and unreliable at deciding whether to
        # call a tool. Health questions can be routed deterministically first,
        # which both reduces latency and guarantees the disease guide is used.
        routed_intent = _fallback_intent(user_message)
        if routed_intent == "search":
            routed_intent = _fallback_intent(_fallback_query(self.history, user_message))
        if routed_intent in {"health_guidance", "health_risk", "consumption"}:
            return await self._ask_without_llm(user_message)
        if self._llm is None:
            return await self._ask_without_llm(user_message)

        self.history.append({"role": "user", "content": user_message})
        openai_tools = [_tool_to_openai_schema(tool) for tool in self._tools]
        sources: list[dict[str, Any]] = []
        called_tools: list[str] = []
        fallback_attempted = False

        for _ in range(self.max_tool_rounds):
            try:
                completion = await self._llm.chat.completions.create(
                    model=self.model,
                    messages=self.history,
                    tools=openai_tools,
                    tool_choice="auto",
                    temperature=0,
                    max_tokens=500,
                )
            except Exception as exc:
                # Expired keys, quota errors and unavailable local models all
                # use the same deterministic intent router as no-LLM mode.
                response = await self._ask_without_llm(user_message, append_user=False)
                diagnostics = dict(response.diagnostics or {})
                diagnostics["fallback_reason"] = f"{type(exc).__name__}: {exc}"
                return ClientResponse(
                    answer=response.answer,
                    sources=response.sources,
                    tool_calls=response.tool_calls,
                    evidence_synthesis=response.evidence_synthesis,
                    diagnostics=diagnostics,
                )
            message = completion.choices[0].message
            tool_calls = getattr(message, "tool_calls", None) or []
            if not tool_calls:
                assistant_text = getattr(message, "content", "") or ""
                if (
                    not sources
                    and not fallback_attempted
                    and _looks_like_food_question(user_message)
                    and "search_food_regulation" in self.tool_names
                ):
                    fallback_attempted = True
                    fallback_arguments = {
                        "query": _fallback_query(self.history, user_message)
                    }
                    fallback_id = "foodguard-fallback-search"
                    payload = await self.call_tool(
                        "search_food_regulation", fallback_arguments
                    )
                    called_tools.append("search_food_regulation")
                    _collect_sources(payload, sources)
                    self.history.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": fallback_id,
                                    "type": "function",
                                    "function": {
                                        "name": "search_food_regulation",
                                        "arguments": json.dumps(
                                            fallback_arguments, ensure_ascii=False
                                        ),
                                    },
                                }
                            ],
                        }
                    )
                    self.history.append(
                        {
                            "role": "tool",
                            "tool_call_id": fallback_id,
                            "name": "search_food_regulation",
                            "content": json.dumps(
                                _llm_tool_payload(payload), ensure_ascii=False
                            ),
                        }
                    )
                    continue
                self.history.append({"role": "assistant", "content": assistant_text})
                self._last_llm_used = bool(assistant_text)
                return ClientResponse(
                    answer=_answer_with_sources(assistant_text, sources),
                    sources=sources,
                    tool_calls=called_tools,
                    evidence_synthesis=synthesize_evidence(sources),
                    diagnostics=self._diagnostics(
                        intent=_fallback_intent(user_message),
                        tool_calls=called_tools,
                        sources=sources,
                        answer_source="llm" if assistant_text else "fallback",
                    ),
                )

            self.history.append(_message_to_dict(message))
            for tool_call in tool_calls:
                name = tool_call.function.name
                called_tools.append(name)
                try:
                    arguments = json.loads(tool_call.function.arguments or "{}")
                    if not isinstance(arguments, dict):
                        raise ValueError("tool arguments must be a JSON object")
                    payload = await self.call_tool(name, arguments)
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    payload = {
                        "result": {"status": "invalid_tool_arguments", "message": str(exc)},
                        "sources": [],
                    }
                _collect_sources(payload, sources)
                self.history.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": name,
                        "content": json.dumps(
                            _llm_tool_payload(payload), ensure_ascii=False
                        ),
                    }
                )

        raise RuntimeError("LLM exceeded the maximum MCP tool-call rounds.")


async def _interactive_cli(demo: bool = False) -> None:
    async with FoodGuardMCPClient() as client:
        print("MCP tools:", ", ".join(client.tool_names))
        questions = (
            ["高蛋白食品的宣稱有什麼規定？", "那糖呢？"]
            if demo
            else iter(lambda: input("\nQ> "), "")
        )
        for question in questions:
            if not question.strip():
                break
            response = await client.ask(question)
            print(f"\nA> {response.answer}")
            print(f"\n[conversation messages: {len(client.conversation_history)}]")


def main() -> None:
    _configure_console_encoding()
    parser = argparse.ArgumentParser(description="FoodGuard MCP Client CLI")
    parser.add_argument("--demo", action="store_true", help="run the required two-turn Q1/Q2 conversation")
    args = parser.parse_args()
    try:
        asyncio.run(_interactive_cli(demo=args.demo))
    except (RuntimeError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
