import json

from foodguard.synthesis import NOT_ENOUGH_EVIDENCE, synthesize_evidence
from mcp_client import _llm_tool_payload


def test_synthesis_prioritises_formal_source_and_deduplicates_pages():
    evidence = [
        {
            "document": "official_QA.pdf",
            "page": 2,
            "text": "問答補充說明：產品標示應提供消費者可辨識的資訊。",
            "score": 0.99,
        },
        {
            "document": "food_label_regulation.pdf",
            "page": 8,
            "text": "包裝食品標示應以清楚、易懂的方式呈現，並依現行規定提供必要資訊。",
            "score": 0.72,
        },
        {
            "document": "food_label_regulation.pdf",
            "page": 8,
            "text": "包裝食品標示應以清楚、易懂的方式呈現，並依現行規定提供必要資訊。",
            "score": 0.70,
        },
    ]

    result = synthesize_evidence(evidence)

    assert result["sources"][0]["document"] == "food_label_regulation.pdf"
    assert len(result["sources"]) == 2
    assert all("text" not in source for source in result["sources"])
    assert result["key_points"]


def test_synthesis_does_not_invent_numeric_rules():
    result = synthesize_evidence(
        [
            {
                "document": "nutrition_guide.pdf",
                "page": 4,
                "text": "本資料建議依個人情況檢視飲食內容，未提供特定數值門檻。",
                "score": 0.8,
            }
        ]
    )

    assert result["numeric_rules"] == []
    assert "180" not in json.dumps(result, ensure_ascii=False)


def test_empty_synthesis_is_explicitly_insufficient():
    result = synthesize_evidence([])

    assert result["insufficient_information"] == [NOT_ENOUGH_EVIDENCE]
    assert result["sources"] == []


def test_llm_payload_excludes_raw_chunk_field():
    raw_text = (
        "這是一段很長的原始 PDF chunk，包含不應直接送給模型的完整段落與內部檢索資訊。"
        "這段文字只用來模擬完整文件內容，不能成為使用者看到的最終答案。"
        "文件後面還有其他規範段落、註解與版面解析結果，必須保留在開發者除錯資料中。"
        + "其他完整文件內容。" * 30
    )
    compact = _llm_tool_payload(
        {
            "result": {"status": "evidence_found"},
            "sources": [
                {
                    "document": "source.pdf",
                    "page": 1,
                    "text": raw_text,
                    "score": 0.91,
                    "chunk_id": "internal-1",
                }
            ],
            "debug_evidence": {"raw": True},
        }
    )

    payload_text = json.dumps(compact, ensure_ascii=False)
    assert "chunk_id" not in payload_text
    assert raw_text not in payload_text
    assert "evidence_summary" in compact
