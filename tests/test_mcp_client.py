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
        assert "法規來源：" in first.answer
        assert "法規來源：" in second.answer


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
