from __future__ import annotations

from time import perf_counter

import pytest

import mcp_server


PRODUCT = {
    "product_name": "燕麥大豆蛋白飲",
    "ingredients": ["水", "燕麥", "大豆蛋白", "奶粉", "砂糖"],
    "nutrition": {
        "nutrition_basis": {"amount": 300, "unit": "ml"},
        "values": {"protein_g": 12, "sugar_g": 9, "sodium_mg": 135, "fat_g": 6},
    },
}


@pytest.mark.parametrize(
    ("disease", "question"),
    [
        ("糖尿病", "糖尿病患者喝這款燕麥大豆蛋白飲要注意什麼？"),
        ("高血壓", "高血壓患者喝這款飲品要注意什麼？"),
        ("腎臟病", "腎臟病患者需要注意這款飲品的蛋白質嗎？"),
        ("高血脂", "高血脂患者需要注意這款飲品的脂肪嗎？"),
    ],
)
def test_disease_qa_returns_official_guidance(
    disease: str, question: str, capsys: pytest.CaptureFixture[str]
) -> None:
    started = perf_counter()
    response = mcp_server.search_disease_guideline(disease, question, PRODUCT)
    elapsed = perf_counter() - started
    result = response["result"]
    print(f"{disease}: {elapsed:.2f}s, status={result.get('status')}, sources={len(response.get('sources', []))}")

    assert result["status"] == "pass"
    assert result["knowledge_domain"] == "disease_guidance"
    assert response["sources"]
    assert all(source["knowledge_domain"] == "disease_guidance" for source in response["sources"])
    assert "診斷" not in result["reasoning"]
    assert "個人醫療結論" in result["reasoning"]


@pytest.mark.parametrize(
    ("topic", "question", "product"),
    [
        ("processed_meat", "香腸有致癌風險嗎？", {"product_name": "香腸", "ingredients": ["豬肉", "亞硝酸鈉"]}),
        ("aflatoxin", "發霉花生要注意什麼？", {"product_name": "花生", "ingredients": ["發霉花生"]}),
        ("acrylamide", "洋芋片可能有什麼健康風險？", {"product_name": "洋芋片", "ingredients": ["馬鈴薯"]}),
        ("nitrosation", "亞硝酸鈉和亞硝胺有什麼差異？", {"product_name": "香腸", "ingredients": ["亞硝酸鈉"]}),
    ],
)
def test_health_risk_qa_separates_hazard_from_exposure(
    topic: str, question: str, product: dict[str, object]
) -> None:
    response = mcp_server.search_health_risk(question, product, {"frequency": "daily"})
    result = response["result"]

    assert result["status"] == "evidence_found"
    assert any(item["topic"] == topic for item in result["risk_topics"])
    assert result["exposure_context"]["frequency"] == "daily"
    assert all(source["knowledge_domain"] == "health_risk" for source in response["sources"])
    assert "一定會罹癌" not in str(result)
    assert "hazard" in result["reasoning"]
