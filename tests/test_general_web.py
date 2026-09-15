from __future__ import annotations

from typing import Any

import pytest

import mcp_server
from mcp_client import FoodGuardMCPClient, _fallback_intent


def test_router_supports_general_and_volatile_intents() -> None:
    assert _fallback_intent("Python 是什麼？") == "general_knowledge"
    assert _fallback_intent("日本首都是哪裡？") == "general_knowledge"
    assert _fallback_intent("最近 TFDA 有更新食品標示規範嗎？") == "current_information"
    assert _fallback_intent("這瓶豆漿的糖是多少？") == "current_product_question"
    assert _fallback_intent("無糖標準是什麼？") == "food_regulation"
    assert _fallback_intent("加工肉有致癌風險嗎？") == "health_risk"


@pytest.mark.anyio
async def test_general_question_has_offline_answer_without_food_refusal() -> None:
    client = FoodGuardMCPClient(llm_client=None, require_api_key=False)
    client._llm = None
    client._mcp = object()

    response = await client.ask("Python 是什麼？")

    assert response.tool_calls == []
    assert "程式語言" in response.answer
    assert "只能回答食品" not in response.answer
    assert response.diagnostics["intent"] == "general_knowledge"
    assert response.diagnostics["answer_mode"] == "fallback"


@pytest.mark.anyio
async def test_current_information_calls_web_search_and_keeps_result_schema(monkeypatch) -> None:
    client = FoodGuardMCPClient(llm_client=None, require_api_key=False)
    client._llm = None
    client._mcp = object()
    calls: list[tuple[str, dict[str, Any]]] = []

    async def fake_call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        calls.append((name, arguments))
        return {
            "result": {
                "status": "ok",
                "query": arguments["query"],
                "provider": "fake",
                "results": [
                    {
                        "title": "TFDA announcement",
                        "url": "https://www.fda.gov.tw/example",
                        "snippet": "Official update",
                        "publisher": "TFDA",
                        "published_date": "2026-09-01",
                        "retrieved_at": "2026-09-15T00:00:00+00:00",
                    }
                ],
            },
            "sources": [
                {
                    "title": "TFDA announcement",
                    "url": "https://www.fda.gov.tw/example",
                    "snippet": "Official update",
                    "publisher": "TFDA",
                    "published_date": "2026-09-01",
                    "retrieved_at": "2026-09-15T00:00:00+00:00",
                }
            ],
        }

    monkeypatch.setattr(client, "call_tool", fake_call_tool)
    response = await client.ask("最近 TFDA 有更新食品標示規範嗎？")

    assert calls[0][0] == "web_search"
    assert calls[0][1]["domains"] == ["fda.gov.tw", "mohw.gov.tw"]
    assert response.diagnostics["web_search_used"] is True
    assert response.diagnostics["web_query"]
    assert response.sources[0]["source_url"].startswith("https://")
    assert "TFDA announcement" in response.answer


@pytest.mark.anyio
async def test_product_latest_claim_uses_product_rule_and_web(monkeypatch) -> None:
    client = FoodGuardMCPClient(llm_client=None, require_api_key=False)
    client._llm = None
    client._mcp = object()
    client.set_current_context(
        {
            "product_name": "高蛋白豆漿",
            "ingredients": ["黃豆"],
            "nutrition": {"values": {"protein_g": 12}, "nutrition_basis": {"amount": 100, "unit": "ml"}},
            "claims": ["高蛋白"],
        }
    )
    calls: list[str] = []

    async def fake_call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        calls.append(name)
        if name == "check_nutrition_claim":
            return {"result": {"status": "pass", "summary": "產品規則結果"}, "sources": []}
        if name == "web_search":
            return {
                "result": {"status": "ok", "query": arguments["query"], "provider": "fake", "results": []},
                "sources": [],
            }
        raise AssertionError(name)

    monkeypatch.setattr(client, "call_tool", fake_call_tool)
    response = await client.ask("這瓶符合最新的高蛋白規定嗎？")

    assert calls == ["check_nutrition_claim", "web_search"]
    assert response.diagnostics["current_product"] == "高蛋白豆漿"
    assert response.diagnostics["web_search_attempted"] is True
    assert response.diagnostics["web_search_used"] is False
    assert "目前無法存取即時網路資訊" in response.answer or "目前可查到" in response.answer


def test_web_search_disabled_is_structured_fallback(monkeypatch) -> None:
    monkeypatch.setenv("WEB_SEARCH_ENABLED", "false")
    result = mcp_server.web_search("最新食品標示")

    assert result["result"]["status"] == "disabled"
    assert result["result"]["results"] == []
    assert result["sources"] == []

