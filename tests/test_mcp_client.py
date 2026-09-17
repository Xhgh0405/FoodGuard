from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from mcp_client import FoodGuardMCPClient


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@dataclass
class FakeFunction:
    name: str
    arguments: str


@dataclass
class FakeToolCall:
    id: str
    function: FakeFunction


class FakeMessage:
    role = "assistant"

    def __init__(self, content: str = "", tool_calls: list[FakeToolCall] | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls or []

    def model_dump(self, exclude_none: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in self.tool_calls
            ]
        return payload


class FakeCompletion:
    def __init__(self, message: FakeMessage) -> None:
        self.choices = [type("Choice", (), {"message": message})()]


class FakeCompletions:
    def __init__(self) -> None:
        self.requests: list[list[dict[str, Any]]] = []
        self.turn = 0

    async def create(self, *, messages: list[dict[str, Any]], **kwargs: Any) -> FakeCompletion:
        self.requests.append(messages)
        last = messages[-1]
        if last["role"] == "user":
            self.turn += 1
            if self.turn == 1:
                query = "高蛋白食品的營養宣稱規定"
            else:
                assert any(
                    item["role"] == "user" and item["content"] == "高蛋白食品的宣稱有什麼規定？"
                    for item in messages
                )
                query = "延續高蛋白宣稱脈絡，糖的食品營養宣稱規定"
            call = FakeToolCall(
                id=f"call-{self.turn}",
                function=FakeFunction(
                    name="search_food_regulation",
                    arguments=json.dumps({"query": query}, ensure_ascii=False),
                ),
            )
            return FakeCompletion(FakeMessage(tool_calls=[call]))
        return FakeCompletion(FakeMessage(content="已根據 MCP/RAG 檢索結果整理回答。"))


class FakeLLM:
    def __init__(self) -> None:
        self.chat = type("Chat", (), {"completions": FakeCompletions()})()


@pytest.mark.anyio
async def test_multiturn_context_and_real_stdio_mcp_connection() -> None:
    fake_llm = FakeLLM()
    async with FoodGuardMCPClient(llm_client=fake_llm) as client:
        assert {
            "search_food_regulation",
            "check_allergens",
            "check_nutrition_label",
            "check_nutrition_claim",
        } <= set(client.tool_names)

        direct_tool_calls = [
            await client.call_tool("search_food_regulation", {"query": "食品標示"}),
            await client.call_tool("check_allergens", {"ingredients": "牛奶、大豆蛋白"}),
            await client.call_tool("check_nutrition_label", {"nutrition_data": {"熱量": "180 kcal"}}),
            await client.call_tool(
                "check_nutrition_claim",
                {"claim": "高蛋白", "nutrition_data": {"蛋白質": "12 g"}},
            ),
        ]
        assert all(set(payload) >= {"result", "sources"} for payload in direct_tool_calls)

        first = await client.ask("高蛋白食品的宣稱有什麼規定？")
        second = await client.ask("那糖呢？")

        assert first.tool_calls == ["search_food_regulation"]
        assert second.tool_calls == ["search_food_regulation"]
        assert len(client.conversation_history) > 1
        assert first.sources
        assert second.sources
        assert "資料來源：" not in first.answer
        assert "資料來源：" not in second.answer


@pytest.mark.anyio
async def test_product_analysis_isolates_one_tool_failure(monkeypatch) -> None:
    client = FoodGuardMCPClient(llm_client=FakeLLM(), require_api_key=False)

    async def fake_call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "check_nutrition_label":
            raise RuntimeError("simulated MCP task failure")
        return {
            "result": {
                "status": "pass",
                "title": name,
                "summary": "ok",
                "findings": [],
                "reasoning": "ok",
                "recommendations": [],
            },
            "sources": [],
        }

    monkeypatch.setattr(client, "call_tool", fake_call_tool)
    result = await client.analyze_product(
        {
            "product_name": "測試食品",
            "ingredients": ["牛奶"],
            "nutrition": {"provided_fields": [], "values": {}},
            "claims": [],
        }
    )

    assert result["allergens"]["result"]["status"] == "pass"
    assert result["nutrition_label"]["result"]["status"] == "insufficient_evidence"
    assert result["nutrition_label"]["debug_evidence"]["error_type"] == "RuntimeError"
    assert result["nutrition_claim"]["result"]["status"] == "pass"


@pytest.mark.anyio
async def test_representative_questions_use_product_context_and_compact_answer():
    fake_llm = FakeLLM()
    questions = [
        "這個食品有哪些過敏原？",
        "為什麼奶油乳酪算乳類？",
        "那雞蛋呢？",
        "高蛋白是什麼意思？",
        "這個可以標高蛋白嗎？",
        "高血壓的人這個要注意什麼？",
        "為什麼？",
    ]
    async with FoodGuardMCPClient(llm_client=fake_llm) as client:
        client.set_current_context(
            {"product_name": "測試飲品", "ingredients": ["牛奶", "雞蛋"]},
            {"allergens": {"result": {"summary": "需要確認"}}},
        )
        responses = [await client.ask(question) for question in questions]

    assert any(response.sources for response in responses)
    assert len(responses) == len(questions)
    assert all("重點引用：" not in response.answer for response in responses)
    assert any(
        any("測試飲品" in str(message.get("content", "")) for message in request)
        for request in fake_llm.chat.completions.requests
    )


@pytest.mark.anyio
async def test_no_llm_intake_question_requests_missing_details_without_search(monkeypatch):
    client = FoodGuardMCPClient(llm_client=FakeLLM(), require_api_key=False)
    client._llm = None
    client.set_current_context(
        {
            "product_name": "重乳酪蛋糕",
            "ingredients": ["奶油乳酪", "雞蛋"],
            "nutrition": {"serving_size": "100 公克", "values": {"calories_kcal": 345}},
            "claims": [],
        }
    )

    async def unexpected_tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError(f"ambiguous intake question must not call {name}")

    monkeypatch.setattr(client, "call_tool", unexpected_tool_call)
    response = await client._ask_without_llm("我是一個成年男性一天最多吃幾個")

    assert response.tool_calls == []
    assert "目前無法只依「成年男性」判定" in response.answer
    assert "每份量 100 公克" in response.answer
    assert "目前知識庫找不到足夠依據" in response.answer
    assert "營養宣稱應遵行事項" not in response.answer


@pytest.mark.anyio
async def test_no_llm_allergen_question_uses_product_ingredients(monkeypatch):
    client = FoodGuardMCPClient(llm_client=FakeLLM(), require_api_key=False)
    client._llm = None
    client.set_current_context(
        {
            "product_name": "重乳酪蛋糕",
            "ingredients": ["奶油乳酪", "雞蛋"],
            "nutrition": {},
            "claims": [],
        }
    )
    calls: list[tuple[str, dict[str, Any]]] = []

    async def fake_call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        calls.append((name, arguments))
        return {
            "result": {
                "status": "warning",
                "summary": "偵測到可能需要注意的過敏原。",
                "detected_allergens": [
                    {
                        "category": "牛奶、羊奶及其製品",
                        "source_ingredients": ["奶油乳酪"],
                    },
                    {"category": "蛋及其製品", "source_ingredients": ["雞蛋"]},
                ],
            },
            "sources": [
                {
                    "document": "食品過敏原標示規定.pdf",
                    "page": 1,
                    "text": "官方過敏原標示資料。",
                    "score": 0.9,
                }
            ],
        }

    monkeypatch.setattr(client, "call_tool", fake_call_tool)
    response = await client._ask_without_llm("為什麼奶類會過敏")

    assert calls == [("check_allergens", {"ingredients": ["奶油乳酪", "雞蛋"]})]
    assert "奶油乳酪 → 牛奶、羊奶及其製品" in response.answer
    assert "不代表每個人食用後都會過敏" in response.answer
    assert response.sources
    assert "資料來源：" not in response.answer
    assert "營養宣稱應遵行事項" not in response.answer


@pytest.mark.anyio
async def test_general_allergen_label_question_searches_regulation_not_product(monkeypatch):
    client = FoodGuardMCPClient(llm_client=None, require_api_key=False)
    client._llm = None
    client._mcp = object()
    client.set_current_context(
        {
            "product_name": "重乳酪蛋糕",
            "ingredients": ["奶油乳酪", "雞蛋"],
            "nutrition": {},
            "claims": [],
        }
    )
    calls: list[tuple[str, dict[str, Any]]] = []

    async def fake_call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        calls.append((name, arguments))
        return {
            "result": {"status": "pass", "summary": "已找到與問題相關的法規依據。"},
            "sources": [
                {
                    "document": "食品過敏原標示規定.pdf",
                    "page": 1,
                    "text": "食品過敏原標示規定：乳製品應依規定清楚標示。",
                    "score": 0.9,
                }
            ],
        }

    monkeypatch.setattr(client, "call_tool", fake_call_tool)
    response = await client.ask("台灣食品標示中，乳製品需要如何標示過敏原？")

    assert calls == [
        (
            "search_food_regulation",
            {"query": "台灣食品標示中，乳製品需要如何標示過敏原？"},
        )
    ]
    assert "乳製品應依規定清楚標示" in response.answer
    assert response.diagnostics["intent"] == "food_regulation"


@pytest.mark.anyio
async def test_llm_error_uses_same_safe_intake_fallback(monkeypatch):
    class FailingCompletions:
        async def create(self, **kwargs: Any) -> FakeCompletion:
            raise RuntimeError("simulated quota error")

    failing_llm = type(
        "FailingLLM",
        (),
        {"chat": type("Chat", (), {"completions": FailingCompletions()})()},
    )()
    client = FoodGuardMCPClient(llm_client=failing_llm, require_api_key=False)
    client._mcp = object()
    client.set_current_context(
        {"product_name": "重乳酪蛋糕", "nutrition": {}, "ingredients": [], "claims": []}
    )

    async def unexpected_tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError(f"ambiguous intake question must not call {name}")

    monkeypatch.setattr(client, "call_tool", unexpected_tool_call)
    response = await client.ask("我是一個成年男性一天最多吃幾個")

    assert response.tool_calls == []
    assert "目前無法只依「成年男性」判定" in response.answer
    assert sum(item.get("role") == "user" for item in client.history) == 1


@pytest.mark.anyio
async def test_diabetes_followup_uses_product_and_explicit_consumption_amount(monkeypatch):
    client = FoodGuardMCPClient(llm_client=FakeLLM(), require_api_key=False)
    client._llm = None
    client._mcp = object()
    client.set_current_context(
        {
            "product_name": "高蛋白豆漿",
            "ingredients": ["黃豆蛋白"],
            "nutrition": {
                "raw_text": "每100毫升\n糖 4 公克\n碳水化合物 5 公克",
                "values": {"sugar_g": 4, "carbohydrate_g": 5},
                "nutrition_basis": {"amount": 100, "unit": "ml"},
            },
            "claims": ["高蛋白"],
        }
    )

    async def fake_call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "search_disease_guideline":
            return {
                "result": {"status": "pass", "disease": "糖尿病", "knowledge_domain": "disease_guidance"},
                "sources": [{"document": "糖尿病與我.pdf", "page": 10, "text": "糖尿病飲食指引", "score": 0.9, "knowledge_domain": "disease_guidance"}],
            }
        if name == "calculate_consumption_nutrients":
            return {
                "result": {
                    "status": "calculated",
                    "amount": 2000.0,
                    "unit": "ml",
                    "scaled_values": {"sugar_g": 80.0, "carbohydrate_g": 100.0},
                },
                "sources": [],
            }
        raise AssertionError(name)

    monkeypatch.setattr(client, "call_tool", fake_call_tool)
    first = await client.ask("糖尿病能喝嗎？")
    second = await client.ask("喝2000毫升")

    assert first.tool_calls == ["search_disease_guideline"]
    assert second.tool_calls == ["calculate_consumption_nutrients", "search_disease_guideline"]
    assert "高蛋白豆漿" in second.answer
    assert "糖：80" in second.answer
    assert "碳水化合物：100" in second.answer
    assert "缺少份量" not in second.answer
    assert second.diagnostics["llm_used"] is False
    assert second.diagnostics["current_product"] == "高蛋白豆漿"
    assert second.diagnostics["parsed_follow_up"]["consumption_amount"] == 2000.0
    assert second.diagnostics["consumption_context"]["consumption_unit"] == "ml"
