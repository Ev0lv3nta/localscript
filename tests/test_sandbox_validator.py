from app.core.config import get_runtime_profile
from app.generation.extractor import TaskExtractor
from app.validation.runtime_executor import execute_output
from app.validation.validators import ValidationPipeline


def _task_spec():
    return TaskExtractor().extract(
        prompt="Верни последний адрес из массива wf.vars.emails; если массив пустой, верни nil.",
        context={"wf": {"vars": {"emails": ["a@example.com"]}}},
    )


def test_dangerous_stdlib_validator_rejects_os_namespace():
    report = ValidationPipeline().run(
        code="return os.execute('echo hi')",
        task_spec=_task_spec(),
        profile=get_runtime_profile(),
    )

    assert "dangerous_stdlib_os_forbidden" in report.error_codes()


def test_dangerous_stdlib_validator_rejects_io_namespace():
    report = ValidationPipeline().run(
        code="return io.open('/tmp/x', 'r')",
        task_spec=_task_spec(),
        profile=get_runtime_profile(),
    )

    assert "dangerous_stdlib_io_forbidden" in report.error_codes()


def test_dangerous_stdlib_validator_rejects_require():
    report = ValidationPipeline().run(
        code="local x = require('socket')\nreturn x",
        task_spec=_task_spec(),
        profile=get_runtime_profile(),
    )

    assert "dangerous_stdlib_require_forbidden" in report.error_codes()


def test_dangerous_stdlib_validator_rejects_debug_namespace():
    report = ValidationPipeline().run(
        code="return debug.getinfo(1)",
        task_spec=_task_spec(),
        profile=get_runtime_profile(),
    )

    assert "dangerous_stdlib_debug_forbidden" in report.error_codes()


def test_runtime_executor_denies_unsafe_namespace_even_without_pipeline():
    execution = execute_output(
        "return os.execute('echo hi')",
        {"wf": {"vars": {"emails": ["a@example.com"]}}},
    )

    assert execution.ok is False
    assert execution.error_code == "dangerous_stdlib_os_forbidden"
