from app.core.public_eval import evaluate_case
from app.core.config import get_runtime_profile
from app.generation.extractor import TaskExtractor
from app.validation.runtime_executor import execute_output
from app.validation.validators import ValidationPipeline


def test_semantic_executor_accepts_valid_email_code():
    extractor = TaskExtractor()
    task_spec = extractor.extract(
        prompt="Проверь, что email в переменной email валиден, и верни boolean.",
        context={"wf": {"vars": {"email": "user@example.com"}}},
    )
    code = (
        'local email = wf.vars.email or ""\n'
        'return string.match(email, "^[%w%.%+%-_]+@[%w%.%-_]+%.[A-Za-z]+$") ~= nil'
    )
    pipeline = ValidationPipeline()

    report = pipeline.run(
        code=code,
        task_spec=task_spec,
        profile=get_runtime_profile(),
        source_context={"wf": {"vars": {"email": "user@example.com"}}},
        prompt="Проверь, что email в переменной email валиден, и верни boolean.",
    )

    assert "semantic_mismatch" not in report.error_codes()
    assert "lua_runtime_error" not in report.error_codes()
    assert execute_output(code, {"wf": {"vars": {"email": "user@example.com"}}}).value is True


def test_public_eval_catches_semantically_wrong_email_code():
    case = {
        "id": "semantic_email_regression",
        "family": "email_validation",
        "prompt": "Проверь, что email в переменной email валиден, и верни boolean.",
        "context": {"wf": {"vars": {"email": "user@example.com"}}},
        "expected_output_style": "lua_block",
        "assertions": [],
        "forbidden_patterns": [],
    }

    failures = evaluate_case("return false", case)

    assert "semantic_mismatch" in failures


def test_semantic_executor_accepts_conditional_array_projection_code():
    extractor = TaskExtractor()
    task_spec = extractor.extract(
        prompt="Из массива wf.vars.orders верни новый массив order_id только для заказов, где status равен paid и amount больше 1000.",
        context={"wf": {"vars": {"orders": [{"order_id": "A", "status": "paid", "amount": 1500}, {"order_id": "B", "status": "draft", "amount": 900}]}}},
    )
    code = (
        "local result = _utils.array.new()\n"
        "for _, item in ipairs(wf.vars.orders or {}) do\n"
        "  if item.status == \"paid\" and item.amount > 1000 then\n"
        "    table.insert(result, item.order_id)\n"
        "  end\n"
        "end\n"
        "return result"
    )

    execution = execute_output(
        code,
        {"wf": {"vars": {"orders": [{"order_id": "A", "status": "paid", "amount": 1500}, {"order_id": "B", "status": "draft", "amount": 900}]}}},
    )

    assert execution.ok is True
    assert execution.value == ["A"]


def test_semantic_executor_preserves_json_envelope_values():
    execution = execute_output(
        (
            '{"first":"lua{return wf.vars.items[1]}lua",'
            '"count":"lua{return #wf.vars.items}lua"}'
        ),
        {"wf": {"vars": {"items": [10, 20, 30]}}},
        output_style="json_envelope",
    )

    assert execution.ok is True
    assert execution.value == {"first": 10, "count": 3}


def test_generic_semantic_validator_catches_return_shape_mismatch():
    extractor = TaskExtractor()
    task_spec = extractor.extract(
        prompt="Из массива wf.vars.orders верни новый массив order_id.",
        context={"wf": {"vars": {"orders": [{"order_id": "A"}]}}},
    )
    pipeline = ValidationPipeline()

    report = pipeline.run(
        code="return 1",
        task_spec=task_spec,
        profile=get_runtime_profile(),
        source_context={"wf": {"vars": {"orders": [{"order_id": "A"}]}}},
        prompt="Из массива wf.vars.orders верни новый массив order_id.",
        planner_semantic_checks=[{"kind": "return_shape", "value": "array"}],
    )

    assert "generic_return_shape_array_mismatch" in report.error_codes()


def test_generic_semantic_validator_accepts_empty_array_behavior():
    extractor = TaskExtractor()
    task_spec = extractor.extract(
        prompt="Из массива wf.vars.orders верни новый массив order_id только для заказов, где status равен paid и amount больше 1000.",
        context={"wf": {"vars": {"orders": [{"order_id": "A", "status": "paid", "amount": 1500}]}}},
    )
    pipeline = ValidationPipeline()
    code = (
        "local result = _utils.array.new()\n"
        "for _, item in ipairs(wf.vars.orders or {}) do\n"
        "  if item.status == \"paid\" and item.amount > 1000 then\n"
        "    table.insert(result, item.order_id)\n"
        "  end\n"
        "end\n"
        "return result"
    )

    report = pipeline.run(
        code=code,
        task_spec=task_spec,
        profile=get_runtime_profile(),
        source_context={"wf": {"vars": {"orders": [{"order_id": "A", "status": "paid", "amount": 1500}]}}},
        prompt="Из массива wf.vars.orders верни новый массив order_id только для заказов, где status равен paid и amount больше 1000.",
        planner_semantic_checks=[
            {"kind": "return_shape", "value": "array"},
            {"kind": "empty_array_on_missing_source", "source_path": "wf.vars.orders"},
        ],
    )

    assert "generic_empty_array_behavior_mismatch" not in report.error_codes()
