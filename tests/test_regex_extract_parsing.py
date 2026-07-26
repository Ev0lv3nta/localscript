from app.generation.extractor import TaskExtractor


def test_regex_extract_prefers_pattern_quote_over_field_quote():
    spec = TaskExtractor().extract(
        prompt='Из строки "message" извлеки значение по Lua pattern "ID:(%d+)".',
        context={"wf": {"vars": {"message": "Order ID:12345 ready"}}},
    )

    assert spec.family == "regex_extract"
    assert spec.generation_hints["pattern"] == "ID:(%d+)"
