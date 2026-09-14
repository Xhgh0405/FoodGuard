from pathlib import Path

from foodguard.context import calculate_consumption_nutrients, parse_consumption_amount
from foodguard.memory import load_session, save_session
from foodguard.parsing import parse_nutrition


def test_consumption_follow_up_is_parsed_and_scaled() -> None:
    nutrition = parse_nutrition("每一份量 100 毫升\n糖 4 公克\n鈉 5 毫克")

    assert parse_consumption_amount("我喝了 2000 ml") == {
        "amount": 2000.0,
        "unit": "ml",
        "raw": "2000 ml",
    }
    result = calculate_consumption_nutrients(nutrition, 2000, "ml")

    assert result["status"] == "calculated"
    assert result["multiplier"] == 20
    assert result["scaled_values"] == {"sugar_g": 80.0, "sodium_mg": 100.0}


def test_session_memory_round_trip() -> None:
    path = Path("data") / "_test_foodguard_memory.db"
    if path.exists():
        path.unlink()
    save_session(
        "session-1",
        product_profile={"product_name": "測試飲品"},
        analysis_results={"diagnostics": {"llm_used": False}},
        conversation_history=[{"role": "user", "content": "那糖呢？"}],
        path=path,
    )

    try:
        saved = load_session("session-1", path)
        assert saved is not None
        assert saved["product_profile"]["product_name"] == "測試飲品"
        assert saved["analysis_results"]["diagnostics"]["llm_used"] is False
        assert saved["conversation_history"][0]["content"] == "那糖呢？"
    finally:
        path.unlink(missing_ok=True)
