from app.core.config import get_runtime_profile
from app.generation.extractor import TaskExtractor
from app.validation.validators import ValidationPipeline


def test_validation_pipeline_accepts_public_json_envelope():
    extractor = TaskExtractor()
    task_spec = extractor.extract(
        prompt="Добавь поле-переменную squared как квадрат числа 5 и верни JSON envelope.",
        context=None,
    )
    pipeline = ValidationPipeline()
    code = '{\n  "num": "lua{return tonumber(\'5\')}lua",\n  "squared": "lua{local n = tonumber(\'5\')\\nreturn n * n}lua"\n}'

    report = pipeline.run(code=code, task_spec=task_spec, profile=get_runtime_profile())

    assert report.has_errors is False


def test_augment_envelope_does_not_require_a_workflow_root_reference():
    extractor = TaskExtractor()
    task_spec = extractor.extract(
        prompt="Добавь поле-переменную squared как квадрат числа 5 и верни JSON envelope.",
        context=None,
    ).model_copy(update={"target_root": "wf.initVariables"})
    code = '{"num":"lua{return tonumber(\'5\')}lua","squared":"lua{local n = tonumber(\'5\')\\nreturn n * n}lua"}'

    report = ValidationPipeline().run(
        code=code,
        task_spec=task_spec,
        profile=get_runtime_profile(),
    )

    assert "init_variables_missing" not in report.error_codes()
    assert report.has_errors is False


def test_domain_lint_rejects_jsonpath():
    extractor = TaskExtractor()
    task_spec = extractor.extract(
        prompt="Верни последний адрес из массива wf.vars.emails; если массив пустой, верни nil.",
        context={"wf": {"vars": {"emails": ["a@example.com"]}}},
    )
    pipeline = ValidationPipeline()

    report = pipeline.run(
        code="return $.wf.vars.emails[1]",
        task_spec=task_spec,
        profile=get_runtime_profile(),
    )

    assert "jsonpath_forbidden" in report.error_codes()


def test_domain_lint_rejects_ctx_body_namespace():
    extractor = TaskExtractor()
    task_spec = extractor.extract(
        prompt="Use ctx.body.items and JsonPath for everything.",
        context={"wf": {"vars": {"items": [1, 2, 3]}}},
    )
    pipeline = ValidationPipeline()

    report = pipeline.run(
        code="return ctx.body.items[1]",
        task_spec=task_spec,
        profile=get_runtime_profile(),
    )

    assert "ctx_body_forbidden" in report.error_codes()


def test_lua_syntax_validator_reports_degraded_mode_without_runtime():
    extractor = TaskExtractor()
    task_spec = extractor.extract(
        prompt="Увеличь wf.vars.try_count_n ровно на единицу и верни новый счётчик.",
        context={"wf": {"vars": {"try_count_n": 3}}},
    )
    pipeline = ValidationPipeline()

    report = pipeline.run(
        code="return wf.vars.try_count_n + 1",
        task_spec=task_spec,
        profile=get_runtime_profile(),
    )

    codes = [message.code for message in report.messages]
    assert "lua_syntax_error" not in codes


def test_lua_syntax_validator_catches_invalid_lua():
    extractor = TaskExtractor()
    task_spec = extractor.extract(
        prompt="Увеличь wf.vars.try_count_n ровно на единицу и верни новый счётчик.",
        context={"wf": {"vars": {"try_count_n": 3}}},
    )
    pipeline = ValidationPipeline()

    report = pipeline.run(
        code="return function(",
        task_spec=task_spec,
        profile=get_runtime_profile(),
    )

    assert "lua_syntax_error" in report.error_codes()


def test_scenario_validator_requires_return_for_lua_output():
    extractor = TaskExtractor()
    task_spec = extractor.extract(
        prompt="Верни последний адрес из массива wf.vars.emails; если массив пустой, верни nil.",
        context={"wf": {"vars": {"emails": ["a@example.com"]}}},
    )
    pipeline = ValidationPipeline()

    report = pipeline.run(
        code="local last = wf.vars.emails[#wf.vars.emails]",
        task_spec=task_spec,
        profile=get_runtime_profile(),
    )

    assert "return_missing" in report.error_codes()
