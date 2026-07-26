from app.generation.extractor import TaskExtractor


def test_explicit_prompt_path_without_context_is_preserved():
    spec = TaskExtractor().extract(
        prompt="Из списка wf.vars.emailsList получи последний email.",
        context=None,
    )

    assert spec.generation_hints["source_path"] == "wf.vars.emailsList"


def test_bare_nested_prompt_path_without_context_is_materialized():
    spec = TaskExtractor().extract(
        prompt="Увеличивай stats.try_count_n на каждой итерации.",
        context=None,
    )

    assert spec.generation_hints["counter_path"] == "wf.vars.stats.try_count_n"


def test_mixed_root_context_stays_unknown_mixed():
    spec = TaskExtractor().extract(
        prompt="Сделай что-нибудь нестандартное.",
        context={"wf": {"vars": {"value": 1}, "initVariables": {"userEmail": "x@example.com"}}},
    )

    assert spec.target_root == "unknown_mixed"
