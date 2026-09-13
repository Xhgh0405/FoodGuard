"""Manual stdio smoke test for the four FoodGuard MCP tools.

Run from the project root with:
    py tests\mcp_stdio_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import anyio
from mcp import Client
from mcp.client.stdio import StdioServerParameters


PROJECT_ROOT = Path(__file__).resolve().parents[1]


async def main() -> None:
    server = StdioServerParameters(
        command=sys.executable,
        args=[str(PROJECT_ROOT / "mcp_server.py")],
        cwd=PROJECT_ROOT,
    )
    async with Client(server) as client:
        listed = await client.list_tools()
        names = {tool.name for tool in listed.tools}
        expected = {
            "search_food_regulation",
            "check_allergens",
            "check_nutrition_label",
            "check_nutrition_claim",
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
        ]
        for name, arguments in calls:
            result = await client.call_tool(name, arguments)
            assert not result.is_error, f"{name} failed: {result.content}"
            assert result.structured_content is not None
            assert set(result.structured_content) >= {"result", "sources"}
            print(f"{name}: passed")


if __name__ == "__main__":
    anyio.run(main)
