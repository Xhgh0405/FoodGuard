from foodguard.rules import analyse_allergens
import pytest

from foodguard.claim_rules import evaluate_claim_rule
from foodguard.parsing import parse_nutrition


def test_soy_protein_does_not_leak_into_egg_category():
    result = analyse_allergens(["黃豆蛋白", "分離大豆蛋白", "雞蛋"])
    categories = {item["category"]: item["source_ingredients"] for item in result["detected_allergens"]}

    assert categories["蛋及其製品"] == ["雞蛋"]
    assert categories["大豆及其製品"] == ["黃豆蛋白", "分離大豆蛋白"]


def test_milk_synonyms_and_allergen_label_are_classified_as_milk_products():
    result = analyse_allergens(["生乳", "維生素A", "維生素D", "過敏原：含乳製品"])

    milk = next(
        item
        for item in result["detected_allergens"]
        if item["category"] == "牛奶、羊奶及其製品"
    )
    assert milk["source_ingredients"] == ["生乳", "過敏原：含乳製品"]
    assert "生乳" in milk["matched_terms"]
    assert "乳製品" in milk["matched_terms"]


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


def test_reparse_keeps_normalized_nutrition_basis_for_claim_checks():
    nutrition = parse_nutrition(
        {
            "raw_text": "每一份量 100 公克\n蛋白質 10 公克",
            "values": {"protein_g": 10},
            "nutrition_basis": {"amount": 100, "unit": "g", "source": "serving_size"},
        }
    )

    result = evaluate_claim_rule("高蛋白", nutrition)

    assert result["evaluation"]["basis"] == "每100公克"
    assert result["evaluation"]["actual"] == 10
    assert result["evaluation"]["met"] is False


@pytest.mark.parametrize(
    ("claim", "field", "value", "expected_page"),
    [
        ("無糖", "sugar_g", 0.4, 5),
        ("低鈉", "sodium_mg", 120, 6),
        ("低糖", "sugar_g", 2.5, 6),
        ("低脂", "fat_g", 1.5, 6),
        ("高蛋白", "protein_g", 6, 7),
        ("高纖維", "fiber_g", 3, 7),
    ],
)
def test_all_supported_nutrition_claims_use_the_same_numeric_rule_engine(
    claim: str, field: str, value: float, expected_page: int
):
    result = evaluate_claim_rule(
        claim,
        {
            "values": {field: value},
            "nutrition_basis": {"amount": 100, "unit": "ml"},
        },
    )

    assert result["evaluation"] is not None
    assert result["evaluation"]["met"] is True
    assert result["evaluation"]["source_page"] == expected_page


def test_fiber_claim_can_be_evaluated_from_label_text():
    nutrition = parse_nutrition("每100毫升\n膳食纖維 3 公克")
    result = evaluate_claim_rule("高纖維", nutrition)

    assert result["evaluation"]["met"] is True
    assert result["evaluation"]["source_page"] == 7


def test_claims_are_normalized_from_a_non_100g_serving_basis():
    result = evaluate_claim_rule(
        "高蛋白",
        {
            "values": {"protein_g": 4},
            "nutrition_basis": {"amount": 30, "unit": "g"},
        },
    )

    assert result["evaluation"]["input_value"] == 4
    assert result["evaluation"]["input_amount"] == 30
    assert result["evaluation"]["actual"] == pytest.approx(13.333333)
    assert result["evaluation"]["met"] is True


def test_nutrition_parser_infers_field_units_when_units_are_missing():
    nutrition = parse_nutrition("每份 30\n蛋白質 4\n鈉 120")

    assert nutrition["nutrition_basis"] == {
        "amount": 30.0,
        "unit": "g",
        "source": "serving_size",
    }
    assert nutrition["values"]["protein_g"] == 4
    assert nutrition["values"]["sodium_mg"] == 120


def test_nutrition_parser_accepts_label_punctuation_and_approximate_package_count():
    nutrition = parse_nutrition(
        "每一份量：30公克\n本包裝含：約 5 份\n鈉：150 毫克"
    )

    assert nutrition["serving_size"] == "30 公克"
    assert nutrition["servings_per_package"] == 5
    assert nutrition["nutrition_basis"] == {
        "amount": 30.0,
        "unit": "g",
        "source": "serving_size",
    }
    result = evaluate_claim_rule("低鈉", nutrition)
    assert result["evaluation"]["actual"] == 500
    assert result["evaluation"]["met"] is False


def test_nutrition_parser_converts_units_written_after_values():
    nutrition = parse_nutrition("每100毫升\n蛋白質 3 g\n鈉 0.12 g")

    assert nutrition["values"]["protein_g"] == 3
    assert nutrition["values"]["sodium_mg"] == 120
