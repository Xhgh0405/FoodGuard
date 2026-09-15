from foodguard.parsing import parse_product_data
from foodguard.rules import analyse_allergens, analyse_nutrition_claim, analyse_nutrition_label


def test_sample_product_is_normalized_before_analysis() -> None:
    product = parse_product_data(
        "重乳酪蛋糕",
        "奶油乳酪、雞蛋、砂糖、鮮奶油、牛奶、小麥麵粉",
        "每一份量 100 公克\n熱量 345 大卡\n蛋白質 7.2 公克\n"
        "脂肪 26.5 公克\n飽和脂肪 15.8 公克\n反式脂肪 0.5 公克\n"
        "碳水化合物 21.5 公克\n糖 17.2 公克\n鈉 285 毫克",
        "無",
    )

    assert product["product_name"] == "重乳酪蛋糕"
    assert product["ingredients"][:2] == ["奶油乳酪", "雞蛋"]
    assert product["nutrition"]["values"]["protein_g"] == 7.2
    assert product["nutrition"]["provided_fields"]
    assert product["claims"] == []


def test_allergen_detection_is_deduplicated_and_keeps_input_evidence() -> None:
    result = analyse_allergens(["奶油乳酪", "雞蛋", "牛奶", "小麥麵粉"])

    categories = {item["category"] for item in result["detected_allergens"]}
    assert len(categories) >= 3
    assert any("雞蛋" in item["source_ingredients"] for item in result["detected_allergens"])


def test_nutrition_label_comparison_uses_required_fields_from_sources() -> None:
    evidence = [{
        "text": "營養標示應列出熱量、蛋白質、脂肪、飽和脂肪、反式脂肪、碳水化合物、糖、鈉。"
    }]
    nutrition = {
        "provided_fields": ["熱量", "蛋白質", "脂肪", "鈉"],
        "values": {},
    }

    result = analyse_nutrition_label(nutrition, evidence)

    assert result["status"] == "warning"
    assert "糖" in result["missing_fields"]


def test_structured_claim_rule_evaluates_high_protein_without_raw_chunk_regex() -> None:
    nutrition = {
        "serving_size": "100 毫升",
        "nutrition_basis": {"amount": 100, "unit": "ml"},
        "values": {"protein_g": 7.0},
        "provided_fields": ["蛋白質"],
    }
    result = analyse_nutrition_claim(
        "高蛋白",
        nutrition,
        [{"document": "official.pdf", "page": 7, "text": "官方營養宣稱來源", "score": 0.9}],
    )

    assert result["threshold"]["source_page"] == 7
    assert result["numeric_evaluation"]["actual"] == 7
    assert result["status"] == "pass"
