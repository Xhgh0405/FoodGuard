from foodguard.rules import analyse_allergens
from foodguard.claim_rules import evaluate_claim_rule


def test_soy_protein_does_not_leak_into_egg_category():
    result = analyse_allergens(["黃豆蛋白", "分離大豆蛋白", "雞蛋"])
    categories = {item["category"]: item["source_ingredients"] for item in result["detected_allergens"]}

    assert categories["蛋及其製品"] == ["雞蛋"]
    assert categories["大豆及其製品"] == ["黃豆蛋白", "分離大豆蛋白"]


def test_structured_zero_sugar_and_low_sodium_rules_use_label_basis():
    nutrition = {
        "values": {"sugar_g": 0.4, "sodium_mg": 100},
        "nutrition_basis": {"amount": 100, "unit": "ml"},
    }
    zero_sugar = evaluate_claim_rule("無糖", nutrition)
    low_sodium = evaluate_claim_rule("低鈉", nutrition)

    assert zero_sugar["evaluation"]["met"] is True
    assert zero_sugar["evaluation"]["source_page"] == 5
    assert low_sodium["evaluation"]["met"] is True
    assert low_sodium["evaluation"]["source_page"] == 6


def test_claim_rule_requires_explicit_nutrition_basis():
    result = evaluate_claim_rule("無糖", {"values": {"sugar_g": 0.1}})
    assert result["evaluation"] is None
