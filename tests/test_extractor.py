from app.generation.extractor import TaskExtractor
from app.generation.formatter import OutputFormatter


def test_extractor_identifies_last_email_case():
    extractor = TaskExtractor()

    spec = extractor.extract(
        prompt="Верни последний адрес из массива wf.vars.emails; если массив пустой, верни nil.",
        context={"wf": {"vars": {"emails": ["a@example.com", "b@example.com"]}}},
    )

    assert spec.family == "last_array_item"
    assert spec.output_style == "lua_expression"
    assert spec.target_root == "wf.vars"
    assert spec.family_confidence == 1.0
    assert spec.generation_hints["source_path"] == "wf.vars.emails"


def test_extractor_detects_init_variables_root():
    extractor = TaskExtractor()

    spec = extractor.extract(
        prompt="Преобразуй ISO 8601 из wf.initVariables.recallTime в Unix timestamp.",
        context={"wf": {"initVariables": {"recallTime": "2023-10-15T15:30:00+00:00"}}},
    )

    assert spec.family == "iso8601_to_epoch"
    assert spec.target_root == "wf.initVariables"
    assert spec.generation_hints["iso_path"] == "wf.initVariables.recallTime"


def test_extractor_preserves_context_specific_paths():
    extractor = TaskExtractor()

    email_spec = extractor.extract(
        prompt="Из полученного списка emailsList получи последний.",
        context={"wf": {"vars": {"emailsList": ["a@example.com", "b@example.com"]}}},
    )
    counter_spec = extractor.extract(
        prompt="Увеличивай stats.try_count_n на каждой итерации.",
        context={"wf": {"vars": {"stats": {"try_count_n": 3}}}},
    )

    assert email_spec.generation_hints["source_path"] == "wf.vars.emailsList"
    assert counter_spec.generation_hints["counter_path"] == "wf.vars.stats.try_count_n"


def test_extractor_uses_prompt_path_without_context():
    extractor = TaskExtractor()
    spec = extractor.extract(
        prompt="Из списка wf.vars.emailsList получи последний email.",
        context=None,
    )

    assert spec.family == "last_array_item"
    assert spec.generation_hints["source_path"] == "wf.vars.emailsList"
    assert spec.family_confidence == 1.0


def test_extractor_does_not_route_last_symbol_prompt_to_last_array_item():
    extractor = TaskExtractor()
    spec = extractor.extract(
        prompt="Проверь email и верни последний символ строки email.",
        context=None,
    )

    assert spec.family != "last_array_item"


def test_extractor_prefers_pattern_fragment_after_pattern_intent():
    extractor = TaskExtractor()
    spec = extractor.extract(
        prompt='Из строки "message" извлеки значение по Lua pattern "ID:(%d+)".',
        context={"wf": {"vars": {"message": "Order ID:12345 ready"}}},
    )

    assert spec.family == "regex_extract"
    assert spec.generation_hints["pattern"] == "ID:(%d+)"


def test_extractor_marks_mixed_root_context_as_unknown_mixed():
    extractor = TaskExtractor()
    spec = extractor.extract(
        prompt="Сделай что-нибудь нестандартное.",
        context={"wf": {"vars": {"value": 1}, "initVariables": {"recallTime": "2023-10-15T15:30:00+00:00"}}},
    )

    assert spec.target_root == "unknown_mixed"


def test_extractor_identifies_conditional_array_projection_family():
    extractor = TaskExtractor()
    spec = extractor.extract(
        prompt="Из массива wf.vars.orders верни новый массив order_id только для заказов, где status равен paid и amount больше 1000.",
        context={"wf": {"vars": {"orders": [{"order_id": "A", "status": "paid", "amount": 1500}]}}},
    )

    assert spec.family == "conditional_array_projection"
    assert spec.generation_hints["source_path"] == "wf.vars.orders"
    assert spec.generation_hints["projection_field"] == "order_id"
    assert spec.generation_hints["conditions"] == [
        {"field": "status", "operator": "eq", "value": "paid"},
        {"field": "amount", "operator": "gt", "value": 1000},
    ]
    assert spec.safety_fallback is False


def test_formatter_pretty_prints_json_envelope():
    formatter = OutputFormatter()

    rendered = formatter.format('{"num":"lua{return 1}lua","squared":"lua{return 1}lua"}', "json_envelope")

    assert rendered.startswith("{\n")
    assert '"num": "lua{return 1}lua"' in rendered
