from __future__ import annotations

from typing import Any

import pytest

import mcp_server
from foodguard.context import parse_exposure_context
from foodguard.health_risk import detect_health_risk_topics
from mcp_client import FoodGuardMCPClient


def test_exposure_context_keeps_slots_across_followups():
    first = parse_exposure_context("我每天吃兩根，已經吃了兩年")
    second = parse_exposure_context("如果一個月吃一次呢？", first)

    assert first["amount"] == 2
    assert first["unit"] == "根"
    assert first["frequency"] == "daily"
    assert first["duration"] == "2 years"
    assert second["amount"] == 2
    assert second["frequency"] == "monthly_once"
    assert second["duration"] == "2 years"


def test_health_risk_structured_sources_distinguish_hazard_and_exposure():
    response = mcp_server.search_health_risk(
        "這個有致癌風險嗎？",
        {"product_name": "香腸", "ingredients": ["豬肉", "亞硝酸鈉"]},
        {"frequency": "daily"},
    )
    result = response["result"]

    assert result["health_risk_intent"] is True
    assert any(item["topic"] == "processed_meat" for item in result["risk_topics"])
    assert any(item["iarc_group"] == "1" for item in result["hazard_classifications"])
    assert result["exposure_context"]["frequency"] == "daily"
    assert all(source["knowledge_domain"] == "health_risk" for source in response["sources"])


def test_peanut_is_not_reported_as_proven_aflatoxin_contamination():
    result = detect_health_risk_topics(
        "花生會致癌嗎？", {"product_name": "花生", "ingredients": ["花生"]}
    )
    aflatoxin = next(item for item in result["risk_topics"] if item["topic"] == "aflatoxin")
    assert aflatoxin["detection_status"] == "possible"
    assert aflatoxin["product_signal"] is True


@pytest.mark.anyio
async def test_health_risk_followup_remembers_product_and_frequency(monkeypatch):
    client = FoodGuardMCPClient(llm_client=None, require_api_key=False)
    client._llm = None
    client._mcp = object()
    client.set_current_context({"product_name": "香腸", "ingredients": ["豬肉", "亞硝酸鈉"]})

    async def fake_call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        assert name == "search_health_risk"
        return mcp_server.search_health_risk(
            arguments["question"], arguments["product_context"], arguments["exposure_context"]
        )

    monkeypatch.setattr(client, "call_tool", fake_call_tool)
    first = await client.ask("這個有致癌風險嗎？")
    second = await client.ask("如果一個月吃一次呢？")

    assert first.tool_calls == ["search_health_risk"]
    assert second.tool_calls == ["search_health_risk"]
    assert second.diagnostics["health_risk_intent"] is True
    assert second.diagnostics["current_product"] == "香腸"
    assert second.diagnostics["exposure_context"]["frequency"] == "monthly_once"
    assert "一定會罹癌" not in second.answer
