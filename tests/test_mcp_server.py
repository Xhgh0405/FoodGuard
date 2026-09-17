from typing import Any

import mcp_server


def test_every_tool_returns_structured_result_and_relevant_sources(monkeypatch) -> None:
    def fake_search(query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if "過敏原" in query:
            source = "食品過敏原標示規定.pdf"
            text = "食品過敏原標示規定：牛奶、蛋及其製品。"
        elif "營養標示" in query:
            source = "包裝食品營養標示應遵行事項.pdf"
            text = "營養標示應列出熱量 蛋白質 脂肪 飽和脂肪 反式脂肪 碳水化合物 糖 鈉。"
        elif "營養宣稱" in query:
            source = "包裝食品營養宣稱應遵行事項.pdf"
            text = "營養宣稱應符合本規定的相關條件。"
        else:
            source = "食品安全衛生管理法_法務部官方重點.md"
            text = "食品安全衛生管理法食品標示與食品宣傳廣告相關規範。"
        return [{"source": source, "page": 2, "chunk_id": "x", "text": text, "score": 0.9}]

    monkeypatch.setattr(mcp_server, "rag_search", fake_search)
    calls = [
        mcp_server.search_food_regulation("食品法有哪些"),
        mcp_server.check_allergens("牛奶、雞蛋"),
        mcp_server.check_nutrition_label({"熱量": "180 kcal"}),
        mcp_server.check_nutrition_claim("高蛋白", {"蛋白質": "12 g"}),
    ]

    assert len(calls) == 4
    for response in calls:
        assert set(response) >= {"result", "sources"}
        assert response["result"]["status"] in {
            "pass",
            "warning",
            "fail",
            "info",
            "insufficient_evidence",
        }
        assert response["sources"]
        assert response["sources"][0].keys() >= {
            "document",
            "page",
            "text",
            "quote",
            "score",
        }


def test_tools_report_missing_knowledge_base(monkeypatch) -> None:
    def missing_search(query: str, top_k: int = 5) -> list[dict[str, Any]]:
        raise FileNotFoundError("Vector database not found")

    monkeypatch.setattr(mcp_server, "rag_search", missing_search)
    responses = [
        mcp_server.search_food_regulation("食品法"),
        mcp_server.check_allergens("牛奶"),
        mcp_server.check_nutrition_label({"熱量": "180 kcal"}),
        mcp_server.check_nutrition_claim("高蛋白", {"蛋白質": "12 g"}),
    ]

    assert all(response["sources"] == [] for response in responses)
    assert responses[0]["result"]["status"] == "insufficient_evidence"
    assert responses[2]["result"]["status"] == "insufficient_evidence"
    assert responses[3]["result"]["status"] == "insufficient_evidence"
    # Ingredient detection remains useful even when regulation retrieval is
    # unavailable; the two statuses are intentionally independent.
    assert responses[1]["result"]["status"] == "warning"
    assert responses[1]["result"]["detection_status"] == "detected"
    assert responses[1]["result"]["regulation_evidence_status"] == "insufficient"


def test_empty_claim_is_not_applicable_and_skips_rag(monkeypatch) -> None:
    def unexpected_search(query: str, top_k: int = 5) -> list[dict[str, Any]]:
        raise AssertionError("empty claim must not call RAG")

    monkeypatch.setattr(mcp_server, "rag_search", unexpected_search)
    response = mcp_server.check_nutrition_claim("無", {"蛋白質": "12 g"})

    assert response["sources"] == []
    assert response["result"]["status"] == "not_applicable"
    assert response["debug_evidence"]["skipped"] == "claim_not_provided"


def test_multiple_claims_return_one_finding_per_claim(monkeypatch) -> None:
    monkeypatch.setattr(mcp_server, "rag_search", lambda query, top_k=5: [])
    response = mcp_server.check_nutrition_claim(
        "低脂、無糖",
        {
            "raw_text": "每100毫升\n脂肪 1.5 g\n糖 0.4 g",
            "values": {"fat_g": 1.5, "sugar_g": 0.4},
            "nutrition_basis": {"amount": 100, "unit": "ml"},
        },
    )

    findings = response["result"]["findings"]
    assert [item["claim"] for item in findings] == ["低脂", "無糖"]
    assert all(item["evaluation"] is not None for item in findings)


def test_unmatched_claim_does_not_turn_tool_failure_into_missing_rag(monkeypatch) -> None:
    """An unsupported claim should return a guarded result, not crash the tool."""

    monkeypatch.setattr(mcp_server, "rag_search", lambda query, top_k=5: [])
    monkeypatch.setattr(
        mcp_server,
        "search_web",
        lambda *args, **kwargs: {"status": "unavailable", "results": []},
    )

    response = mcp_server.check_nutrition_claim(
        "健康美味",
        {"raw_text": "每100毫升\n熱量 48 kcal"},
    )

    assert response["result"]["status"] in {
        "insufficient_evidence",
        "not_applicable",
        "warning",
        "fail",
        "pass",
    }
    assert "UnboundLocalError" not in response["result"].get("message", "")


def test_consumption_calculator_is_deterministic() -> None:
    response = mcp_server.calculate_consumption_nutrients(
        {
            "raw_text": "每一份量 100 毫升\n糖 4 公克",
            "values": {"sugar_g": 4},
            "provided_fields": ["糖"],
            "serving_size": "100 毫升",
        },
        2000,
        "ml",
    )

    assert response["result"]["status"] == "calculated"
    assert response["result"]["scaled_values"]["sugar_g"] == 80
