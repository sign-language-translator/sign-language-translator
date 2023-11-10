from sign_language_translator.languages.sign.mapping_rules import (
    CharacterByCharacterMappingRule,
    DirectMappingRule,
    LambdaMappingRule,
)


def test_lambda_mapping_rule():
    data_map = {
        "a": 97,
        "b": 98,
        "c": 99,
    }
    rule = LambdaMappingRule(
        is_applicable_function=lambda x, y, z: isinstance(x, str) and x in data_map,
        apply_function=lambda x: data_map[x] + 5,
        priority=1,
    )

    assert not rule.is_applicable("v")
    assert rule.is_applicable("a")

    assert rule.apply("b") == 103
    assert rule.priority == 1


def test_direct_mapping_rule():
    data_map = {
        "a": 97,
        "b": 98,
        "c": 99,
    }
    rule = DirectMappingRule(data_map, 5)
    assert not rule.is_applicable("hello")
    assert rule.is_applicable("c")

    assert rule.apply("a") == 97


def test_character_by_character_mapping_rule():
    data_map = {
        "a": 97,
        "b": 98,
        "c": 99,
    }
    rule = CharacterByCharacterMappingRule(data_map, {None}, 5)
    assert not rule.is_applicable("world!", tag=None)
    assert not rule.is_applicable("cab", tag="vehicle")
    assert rule.is_applicable("cab", tag=None)

    assert rule.apply("abc") == [97, 98, 99]
