from __future__ import annotations

from typing import Any

import pytest

import mcp_server
from mcp_client import (
    FoodGuardMCPClient,
    _claim_from_question,
    _current_product_answer,
    _fallback_intent,
    _is_contextual_followup,
)


def test_router_supports_general_and_volatile_intents() -> None:
    assert _fallback_intent("Python 是什麼？") == "general_knowledge"
    assert _fallback_intent("日本首都是哪裡？") == "general_knowledge"
    assert _fallback_intent("最近 TFDA 有更新食品標示規範嗎？") == "current_information"
    assert _fallback_intent("這瓶豆漿的糖是多少？") == "current_product_question"
    assert _fallback_intent("無糖標準是什麼？") == "food_regulation"
    assert _fallback_intent("台灣食品標示中，乳製品需要如何標示過敏原？") == "food_regulation"
    assert _fallback_intent(
        "生產設備曾處理花生，是否一定要加註本產品生產製程廠房其設備或生產管線有處理花生？",
        {"product_name": "高纖燕麥果蛋白飲"},
    ) == "food_regulation"
    assert _fallback_intent(
        "這產品營養標示符合食品法嗎？",
        {"product_name": "高纖燕麥果蛋白飲"},
    ) == "food_regulation"
    assert _fallback_intent(
        "這產品的營養標示是否合法？",
        {"product_name": "低糖燕麥飲"},
    ) == "food_regulation"
    assert _fallback_intent("這個食品有哪些過敏原？", {"product_name": "豆漿"}) == "allergen"
    assert _fallback_intent("加工肉有致癌風險嗎？") == "health_risk"
    assert _fallback_intent("熱狗是什麼顏色的？") == "general_knowledge"


def test_product_reference_has_priority_over_daily_nutrition_reference() -> None:
    product = {
        "product_name": "果乾",
        "nutrition": {
            "values": {"protein_g": 2},
            "nutrition_basis": {"amount": 30, "unit": "g"},
        },
    }

    assert _fallback_intent("果乾的蛋白質多少", product) == "current_product_question"
    assert _fallback_intent("果乾有哪些成分", product) == "current_product_question"
    assert _fallback_intent("蛋白質一天多少", product) == "nutrition_reference"
    assert _fallback_intent("成人每天需要多少蛋白質", product) == "nutrition_reference"
    assert _fallback_intent("糖尿病可以吃果乾嗎", product) == "disease_guidance"


def test_loaded_product_handles_omitted_name_in_field_followups() -> None:
    product = {
        "product_name": "果乾",
        "nutrition": {"values": {"protein_g": 2, "sugar_g": 12}},
    }

    assert _fallback_intent("蛋白質多少", product) == "current_product_question"
    assert _fallback_intent("糖多少", product) == "current_product_question"
    assert _fallback_intent("成分有哪些", product) == "current_product_question"
    assert _fallback_intent("蛋白質一天多少", product) == "nutrition_reference"


def test_claim_legality_question_is_not_product_nutrient_lookup() -> None:
    product = {
        "product_name": "低脂無糖優酪乳",
        "nutrition": {"values": {"fat_g": 1.5, "sugar_g": 2}},
    }

    question = "這款產品的「低脂」與「無糖」宣稱是否合法？"
    assert _fallback_intent(question, product) == "claim"
    assert _claim_from_question(question, []) == "低脂肪、無糖"


def test_product_summary_answers_ingredient_and_label_questions() -> None:
    answer = _current_product_answer(
        "果乾的成分標示",
        {
            "product_name": "果乾",
            "ingredients": ["蘋果", "砂糖", "二氧化硫"],
            "nutrition": {
                "nutrition_basis": {"amount": 30, "unit": "g"},
                "values": {"protein_g": 2, "sugar_g": 12, "sodium_mg": 40},
            },
        },
    )

    assert "蘋果、砂糖、二氧化硫" in answer
    assert "蛋白質 2 g" in answer
    assert "糖 12 g" in answer
    assert "每 30g" in answer


def test_product_name_nutrient_words_do_not_trigger_single_field_lookup() -> None:
    product = {
        "product_name": "花生燕麥蛋白飲",
        "ingredients": ["水", "燕麥", "花生", "大豆蛋白", "砂糖"],
        "nutrition": {
            "nutrition_basis": {"amount": 300, "unit": "ml"},
            "values": {
                "calories_kcal": 180,
                "protein_g": 12,
                "fat_g": 6,
                "carbohydrate_g": 22,
                "sugar_g": 10,
                "sodium_mg": 120,
            },
        },
    }

    answer = _current_product_answer("花生燕麥蛋白飲的營養標示", product)

    assert "營養標示：" in answer
    assert "蛋白質 12 g" in answer
    assert "糖 10 g" in answer
    assert "鈉 120 mg" in answer
    assert "蛋白質是 12 g" not in answer

    bare_name_answer = _current_product_answer("花生燕麥蛋白飲", product)
    assert "蛋白質是 12 g" not in bare_name_answer


@pytest.mark.anyio
async def test_product_specific_question_returns_product_value_not_dri() -> None:
    client = FoodGuardMCPClient(llm_client=None, require_api_key=False)
    client._llm = None
    client._mcp = object()
    client.set_current_context(
        {
            "product_name": "果乾",
            "ingredients": ["蘋果", "砂糖"],
            "nutrition": {
                "nutrition_basis": {"amount": 30, "unit": "g"},
                "values": {"protein_g": 2},
            },
        }
    )

    response = await client.ask("蛋白質多少")

    assert response.diagnostics["intent"] == "current_product_question"
    assert "蛋白質是 2 g" in response.answer
    assert "RDA" not in response.answer


def test_general_questions_are_not_contextual_food_followups() -> None:
    assert _is_contextual_followup("元智大學是私立學校嗎？") is False
    assert _is_contextual_followup("Python 是什麼？") is False
    assert _is_contextual_followup("那糖呢？") is True
    assert _is_contextual_followup("如果一個月吃一次呢？") is True


def test_common_geography_question_has_offline_answer() -> None:
    from mcp_client import _general_fallback_answer

    answer = _general_fallback_answer("台灣位於哪裡？")
    assert "東亞" in answer
    assert "LLM_PROVIDER" not in answer


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
async def test_explicit_general_question_does_not_inherit_previous_health_context() -> None:
    client = FoodGuardMCPClient(llm_client=None, require_api_key=False)
    client._llm = None
    client._mcp = object()
    client.set_current_context(
        {"product_name": "含丙烯醯胺食品", "ingredients": ["馬鈴薯"]}
    )
    client.history.extend(
        [
            {"role": "user", "content": "丙烯醯胺會致癌嗎？"},
            {"role": "assistant", "content": "這是上一個健康風險問題的回答。"},
        ]
    )

    response = await client.ask("元智大學是私立學校嗎？")

    assert response.tool_calls == []
    assert response.diagnostics["intent"] == "general_knowledge"
    assert "致癌" not in response.answer
    assert "丙烯醯胺" not in response.answer


@pytest.mark.anyio
async def test_general_llm_turn_receives_no_food_history() -> None:
    calls: list[dict[str, Any]] = []

    class Completions:
        async def create(self, **kwargs: Any):
            calls.append(kwargs)
            message = type("Message", (), {"content": "是，元智大學是私立大學。"})()
            return type("Completion", (), {"choices": [type("Choice", (), {"message": message})()]})()

    fake_llm = type(
        "FakeLLM",
        (),
        {"chat": type("Chat", (), {"completions": Completions()})()},
    )()
    client = FoodGuardMCPClient(llm_client=fake_llm, require_api_key=False)
    client._mcp = object()
    client.set_current_context(
        {"product_name": "含丙烯醯胺食品", "ingredients": ["馬鈴薯"]}
    )
    client.history.extend(
        [
            {"role": "user", "content": "丙烯醯胺會致癌嗎？"},
            {"role": "assistant", "content": "上一個食品健康風險回答。"},
        ]
    )

    response = await client.ask("元智大學是私立學校嗎？")

    assert response.diagnostics["intent"] == "general_knowledge"
    assert response.answer == "是，元智大學是私立大學。"
    assert calls
    assert all(
        message.get("content") not in {"丙烯醯胺會致癌嗎？", "上一個食品健康風險回答。"}
        for message in calls[0]["messages"]
        if message.get("role") in {"user", "assistant"}
    )


@pytest.mark.anyio
async def test_general_food_question_bypasses_food_regulation_tools() -> None:
    class Completions:
        async def create(self, **kwargs: Any):
            assert "tools" not in kwargs
            message = type("Message", (), {"content": "熱狗通常呈紅褐色；外觀會依種類與烹調方式不同。"})()
            return type("Completion", (), {"choices": [type("Choice", (), {"message": message})()]})()

    fake_llm = type(
        "FakeLLM",
        (),
        {"chat": type("Chat", (), {"completions": Completions()})()},
    )()
    client = FoodGuardMCPClient(llm_client=fake_llm, require_api_key=False)
    client._mcp = object()

    response = await client.ask("熱狗是什麼顏色的？")

    assert response.tool_calls == []
    assert "紅褐色" in response.answer
    assert response.diagnostics["intent"] == "general_knowledge"
    assert response.diagnostics["answer_mode"] == "direct_llm"


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
