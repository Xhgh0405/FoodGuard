r"""Manual stdio smoke test for the FoodGuard MCP tools.

Run from the project root with:
    py tests\mcp_stdio_smoke.py
"""

from __future__ import annotations

import anyio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from foodguard import parse_product_data
from mcp_client import FoodGuardMCPClient


async def main() -> None:
    # FoodGuardMCPClient uses official MCP stdio where Windows permits it and
    # the same registered tools in-process when the host rejects named pipes.
    async with FoodGuardMCPClient(require_api_key=False) as client:
        client._llm = None
        names = set(client.tool_names)
        expected = {
            "search_food_regulation",
            "check_allergens",
            "check_nutrition_label",
            "check_nutrition_claim",
            "search_disease_guideline",
            "calculate_consumption_nutrients",
            "search_health_risk",
            "web_search",
            "fetch_web_page",
        }
        assert names >= expected, f"Missing tools: {expected - names}"

        calls = [
            ("search_food_regulation", {"query": "食品標示"}),
            ("check_allergens", {"ingredients": "牛奶、大豆蛋白"}),
            ("check_nutrition_label", {"nutrition_data": {"熱量": "180 kcal"}}),
            (
                "check_nutrition_claim",
                {"claim": "高蛋白", "nutrition_data": {"蛋白質": "12 g"}},
            ),
            (
                "calculate_consumption_nutrients",
                {
                    "nutrition_data": {
                        "raw_text": "每一份量 100 毫升\n糖 4 公克",
                        "values": {"sugar_g": 4},
                        "provided_fields": ["糖"],
                        "serving_size": "100 毫升",
                    },
                    "consumption_amount": 2000,
                    "consumption_unit": "ml",
                },
            ),
            (
                "search_disease_guideline",
                {"disease": "糖尿病", "query": "飲料要注意什麼？"},
            ),
            (
                "search_health_risk",
                {
                    "question": "這個有致癌風險嗎？",
                    "product_context": {"product_name": "香腸", "ingredients": ["豬肉", "亞硝酸鈉"]},
                    "exposure_context": {},
                },
            ),
            ("web_search", {"query": "最新食品標示", "domains": ["fda.gov.tw"], "max_results": 5}),
            ("fetch_web_page", {"url": "file://not-allowed", "max_chars": 1000}),
        ]
        for name, arguments in calls:
            result = await client.call_tool(name, arguments)
            assert set(result) >= {"result", "sources"}
            print(f"{name}: passed")
        product = parse_product_data(
            "高蛋白豆漿",
            "黃豆蛋白、分離大豆蛋白",
            "每100毫升\n糖 4 公克\n碳水化合物 5 公克\n蛋白質 6 公克",
            "高蛋白",
        )
        client.set_current_context(product)
        first = await client.ask("糖尿病能喝嗎？")
        second = await client.ask("喝2000毫升")
        assert second.diagnostics["current_product"] == "高蛋白豆漿"
        assert second.diagnostics["parsed_follow_up"]["consumption_amount"] == 2000
        assert "缺少份量" not in second.answer
        print(f"follow-up: passed ({first.tool_calls} -> {second.tool_calls})")
        print(f"transport: {client.transport_mode}")


if __name__ == "__main__":
    anyio.run(main)
