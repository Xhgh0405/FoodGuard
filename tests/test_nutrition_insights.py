from __future__ import annotations

import pytest

import mcp_server
from foodguard.nutrition_insights import (
    build_nutrition_insights,
    lookup_dri,
    parse_portion,
)
from foodguard.parsing import parse_product_data
from mcp_client import FoodGuardMCPClient


def soy_product() -> dict:
    return parse_product_data(
        "高蛋白豆漿",
        "",
        "每 100 ml\n蛋白質 6 g\n糖 4 g\n鈉 45 mg",
        "",
    )


def cheesecake_product() -> dict:
    return {
        "product_name": "重乳酪蛋糕",
        "ingredients": ["牛奶", "雞蛋"],
        "claims": [],
        "nutrition": {
            "nutrition_basis": {"amount": 100, "unit": "g"},
            "values": {
                "calories_kcal": 345,
                "protein_g": 6,
                "fat_g": 26.5,
                "saturated_fat_g": 15.8,
                "carbohydrate_g": 25,
                "sugar_g": 17.2,
                "sodium_mg": 285,
            },
        },
    }


def test_qa31_dri_lookup_does_not_choose_a_demographic_without_profile() -> None:
    result = lookup_dri("蛋白質", {})
    assert result["status"] == "needs_profile"
    assert {item["reference_value"] for item in result["records"]} == {60, 70}
    assert {"age", "sex"}.issubset(result["missing_information"])


def test_qa32_and_qa33_dri_context_switches_sex() -> None:
    male = lookup_dri("蛋白質", {"age": 30, "sex": "male"})
    female = lookup_dri("蛋白質", {"age": 30, "sex": "female"})
    assert male["selected"]["reference_value"] == 70
    assert female["selected"]["reference_value"] == 60
    assert male["selected"]["reference_type"] == "RDA"


def test_qa34_and_qa35_dri_type_and_nutrient_are_structured() -> None:
    sodium = lookup_dri("鈉", {})
    potassium = lookup_dri("鉀", {"age": 30, "sex": "female"})
    assert sodium["selected"]["reference_type"] == "CDRR"
    assert sodium["selected"]["reference_value"] == 2300
    assert potassium["selected"]["reference_type"] == "AI"
    assert potassium["selected"]["reference_value"] == 2500


def test_qa36_to_qa38_scale_amount_and_keep_context() -> None:
    product = soy_product()
    insight = build_nutrition_insights(
        product,
        {"amount": 500, "unit": "ml"},
        {"age": 30, "sex": "male"},
        "protein",
    )
    assert insight["calculated_intake"] == {
        "protein_g": 30.0,
        "sugar_g": 20.0,
        "sodium_mg": 225.0,
    }
    protein = next(item for item in insight["reference_comparisons"] if item["nutrient"] == "protein")
    assert protein["percentage"] == 42.86
    assert parse_portion("改喝 250 ml", product["nutrition"])["amount"] == 250


def test_package_and_half_package_require_package_metadata() -> None:
    product = soy_product()
    product["nutrition"]["nutrition_basis"]["amount"] = 300
    product["nutrition"]["servings_per_package"] = 2
    assert parse_portion("喝一瓶", product["nutrition"])["amount"] == 600
    assert parse_portion("喝半瓶", product["nutrition"])["amount"] == 300
    del product["nutrition"]["servings_per_package"]
    assert parse_portion("喝一瓶", product["nutrition"]) is None


def test_label_punctuation_and_approximate_package_count_support_package_scaling() -> None:
    product = parse_product_data(
        "餅乾",
        "",
        "每一份量：30公克\n本包裝：約 5 份\n熱量：160 大卡\n鈉：150 毫克",
        "",
    )

    portion = parse_portion("吃一包", product["nutrition"])

    assert portion == {
        "amount": 150.0,
        "unit": "g",
        "raw": "一包",
        "source": "package",
    }


def test_qa39_proactive_analysis_reports_all_label_values_without_threshold_claims() -> None:
    insight = build_nutrition_insights(soy_product())
    assert {"protein_g", "sugar_g", "sodium_mg"}.issubset(insight["important_nutrients"])
    assert all(item["reason"] == "label_value_present_without_inventing_a_health_threshold" for item in insight["proactive_insights"])


def test_qa40_scales_every_available_cheesecake_nutrient() -> None:
    insight = build_nutrition_insights(cheesecake_product(), {"amount": 30, "unit": "g"})
    assert insight["calculated_intake"]["calories_kcal"] == 103.5
    assert insight["calculated_intake"]["saturated_fat_g"] == 4.74
    assert insight["calculated_intake"]["sodium_mg"] == 85.5


def test_qa41_large_portion_is_context_not_a_health_verdict() -> None:
    insight = build_nutrition_insights(soy_product(), {"amount": 2000, "unit": "ml"})
    messages = [item.get("message", "") for item in insight["proactive_insights"]]
    assert any("相對較大" in message for message in messages)
    assert not any("危險" in message or "過量" in message for message in messages)


@pytest.mark.anyio
async def test_context_memory_keeps_product_portion_and_profile(monkeypatch) -> None:
    client = FoodGuardMCPClient(llm_client=None, require_api_key=False)
    client._llm = None
    client._mcp = object()
    product = soy_product()
    client.set_current_context(product)

    async def fake_call_tool(name: str, arguments: dict):
        if name == "calculate_consumption_nutrients":
            return mcp_server.calculate_consumption_nutrients(**arguments)
        if name == "lookup_dri_reference":
            return mcp_server.lookup_dri_reference(**arguments)
        raise AssertionError(name)

    monkeypatch.setattr(client, "call_tool", fake_call_tool)
    first = await client.ask("喝500ml有多少蛋白質？")
    second = await client.ask("鈉呢？")
    third = await client.ask("這些大概占成人一天多少？")
    fourth = await client.ask("我是30歲男性。")

    assert "蛋白質：30" in first.answer
    assert "鈉：225" in second.answer
    assert third.diagnostics["consumption_context"]["consumption_amount"] == 500
    assert fourth.diagnostics["user_profile"]["age"] == 30
    assert fourth.diagnostics["user_profile"]["sex"] == "male"
    assert "42.86%" in fourth.answer
