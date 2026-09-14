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
    model = _get_setting("OPENAI_MODEL", "gpt-4o-mini")
    base_url = _get_setting("OPENAI_BASE_URL") or None
    return api_key, model, base_url


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


def _structured_fallback_answer(
    intent: str,
    question: str,
    payload: dict[str, Any],
) -> str:
    result = payload.get("result", {}) if isinstance(payload.get("result"), dict) else {}
    if intent == "allergen":
        return _allergen_answer(question, result)
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
            return (
                "目前找到相關官方資料，但現有結構化規則不足以直接產生個人飲食上限。"
                "請提供目前食品的每份營養數值與要注意的疾病項目。"
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
    return "\n".join(lines)


class FoodGuardMCPClient:
    """A stateful client that talks to FoodGuard MCP over stdio."""

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
        self._llm = llm_client
        self.current_product: dict[str, Any] | None = None
        self.current_analysis: dict[str, Any] | None = None
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

    def set_current_context(
        self,
        product_data: dict[str, Any] | None = None,
        analysis: dict[str, Any] | None = None,
    ) -> None:
        """Attach the active product to follow-up questions without raw evidence."""

        self.current_product = copy.deepcopy(product_data) if product_data else None
        self.current_analysis = copy.deepcopy(analysis) if analysis else None

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
        await self._mcp.__aenter__()
        await self.refresh_tools()
        return self

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self._mcp is not None:
            await self._mcp.__aexit__(exc_type, exc_value, traceback)
            self._mcp = None

    @property
    def conversation_history(self) -> list[dict[str, Any]]:
        return list(self.history)

    @property
    def tool_names(self) -> list[str]:
        return [tool.name for tool in self._tools]

    async def refresh_tools(self) -> list[str]:
        if self._mcp is None:
            raise RuntimeError("MCP client is not connected.")
        listed = await self._mcp.list_tools()
        self._tools = list(listed.tools)
        return self.tool_names

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self._mcp is None:
            raise RuntimeError("MCP client is not connected.")
        if name not in self.tool_names:
            return {
                "result": {"status": "unknown_tool", "message": f"MCP tool not available: {name}"},
                "sources": [],
            }
        mcp_result = await self._mcp.call_tool(name, arguments)
        return _payload_from_mcp_result(mcp_result)

    async def summarize_analysis(
        self,
        product_data: dict[str, Any],
        task_results: dict[str, dict[str, Any]],
    ) -> str:
        """Use the configured LLM to summarize structured rule results safely."""

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
            return answer or fallback
        except Exception:
            return fallback

    async def _ask_without_llm(
        self, user_message: str, *, append_user: bool = True
    ) -> ClientResponse:
        """Route to structured tools and answer safely without a paid LLM."""

        contextual_query = _fallback_query(self.history, user_message)
        intent = _fallback_intent(user_message)
        if intent == "search" and len(user_message.strip()) <= 12:
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

            try:
                payload = await self.call_tool(tool_name, arguments)
                called_tools.append(tool_name)
                _collect_sources(payload, sources)
            except Exception:
                payload = {
                    "result": {"status": "insufficient_evidence", "summary": NOT_ENOUGH_EVIDENCE},
                    "sources": [],
                }
            answer = _structured_fallback_answer(intent, contextual_query, payload)

        answer = _answer_with_sources(answer, sources)
        self.history.append({"role": "assistant", "content": answer})
        return ClientResponse(
            answer=answer,
            sources=sources,
            tool_calls=called_tools,
            evidence_synthesis=synthesize_evidence(sources),
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
        overall_summary = await self.summarize_analysis(product_data, task_results)
        evidence_synthesis = _synthesize_payloads(list(task_results.values()), product_data)
        return {
            "product_data": product_data,
            **task_results,
            "overall_summary": overall_summary,
            "evidence_synthesis": evidence_synthesis,
            "tool_calls": [
                "check_allergens",
                "check_nutrition_label",
                "check_nutrition_claim",
            ],
        }

    async def ask(self, user_message: str) -> ClientResponse:
        if self._mcp is None:
            raise RuntimeError("MCP client is not connected.")
        if not user_message.strip():
            raise ValueError("user_message must not be empty")

        self._add_context_message()
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
            except Exception:
                # Expired keys, quota errors and unavailable local models all
                # use the same deterministic intent router as no-LLM mode.
                return await self._ask_without_llm(user_message, append_user=False)
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
                return ClientResponse(
                    answer=_answer_with_sources(assistant_text, sources),
                    sources=sources,
                    tool_calls=called_tools,
                    evidence_synthesis=synthesize_evidence(sources),
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
