import json

from typer.testing import CliRunner

from app.cli.main import cli
from app.core.benchmarks import QUALITY_EVAL_MANIFEST
from app.generation.backend_errors import BackendUnavailable
from app.generation.engine import SessionStateError
from app.generation.ollama import OllamaBackend
from app.generation.results import GenerationResult, SessionStatus, SessionSummary
from app.workflow.contracts import (
    CheckStatus,
    OutputContract,
    OutputFormat,
    OutputShape,
    ValidationCheck,
    ValidationResult,
    WorkflowResult,
    WorkflowRoot,
    WorkflowStatus,
)

runner = CliRunner()


def _completed_result() -> GenerationResult:
    return GenerationResult(
        workflow=WorkflowResult(
            status=WorkflowStatus.COMPLETED,
            code="return wf.vars.value",
            validation=ValidationResult(
                checks=(ValidationCheck(name="all", status=CheckStatus.PASSED),),
                observations=({"actual": 7},),
            ),
            output=OutputContract(format=OutputFormat.LUA_BLOCK, shape=OutputShape.SCALAR),
        ),
        trace_id="trace-1",
        session_id="session-1",
        session=SessionSummary(
            session_id="session-1",
            status=SessionStatus.COMPLETED,
            original_task="Return the value.",
        ),
    )


class CloseTracker:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class CapturingEngine:
    def __init__(self, result=None, error=None):
        self.backend = CloseTracker()
        self.result = result or _completed_result()
        self.error = error
        self.kwargs = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        if self.error is not None:
            raise self.error
        return self.result


def test_cli_forwards_structured_source_selection(monkeypatch):
    engine = CapturingEngine()
    monkeypatch.setattr("app.cli.main.build_engine", lambda: engine)
    result = runner.invoke(
        cli,
        [
            "generate",
            "--session-id",
            "session-1",
            "--source-root",
            "wf.vars",
            "--source-root",
            "wf.initVariables",
        ],
    )
    assert result.exit_code == 0, result.output
    assert engine.kwargs["source_roots"] == (WorkflowRoot.VARS, WorkflowRoot.INIT_VARIABLES)
    assert engine.kwargs["clarification_answer"] is None
    assert engine.backend.closed


def _green_quality_report():
    report = {
        "backend_type": "live_ollama",
        "eval_manifest": [dict(entry) for entry in QUALITY_EVAL_MANIFEST],
        "ok": True,
    }
    for entry in QUALITY_EVAL_MANIFEST:
        report[entry["name"]] = {
            "ok": True,
            "passed": 20,
            "supported_total": 16,
            "supported_passed": 16,
            "supported_pass_rate": 1.0,
            "clarification_total": 2,
            "clarification_passed": 2,
            "clarification_pass_rate": 1.0,
            "safety_total": 2,
            "safety_passed": 2,
            "safety_pass_rate": 1.0,
            "metrics": {
                "supported_total": 16,
                "supported_passed": 16,
                "supported_pass_rate": 1.0,
                "clarification_total": 2,
                "clarification_passed": 2,
                "clarification_pass_rate": 1.0,
                "safety_total": 2,
                "safety_passed": 2,
                "safety_pass_rate": 1.0,
                "invalid_success_count": 0,
            },
        }
    return report


def test_generate_command_returns_contract_json(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALSCRIPT_TRACE_DIR", str(tmp_path / "traces"))

    class DummyEngine:
        def generate(self, **_kwargs):
            return GenerationResult(
                workflow=WorkflowResult(
                    status=WorkflowStatus.COMPLETED,
                    code="return wf.vars.try_count_n + 1",
                    validation=ValidationResult(
                        checks=(ValidationCheck(name="all", status=CheckStatus.PASSED),)
                    ),
                ),
                trace_id="trace-1",
                session_id="session-1",
                session=SessionSummary(
                    session_id="session-1",
                    status=SessionStatus.COMPLETED,
                    original_task="Увеличь счётчик.",
                ),
            )

    monkeypatch.setattr("app.cli.main.build_engine", lambda: DummyEngine())

    result = runner.invoke(
        cli,
        [
            "generate",
            "--prompt",
            "Увеличь wf.vars.try_count_n ровно на единицу и верни новый счётчик.",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "completed"
    assert payload["code"] == "return wf.vars.try_count_n + 1"
    assert "question" not in payload


def test_validate_command_requires_explicit_contract(monkeypatch):
    class Policy:
        findings = ()

    class Execution:
        ok = True
        value = 1
        error_code = ""
        error_message = ""
        degraded = False

    monkeypatch.setattr(
        "app.workflow.validation._default_policy_analyzer",
        lambda _code, _style: Policy(),
    )
    monkeypatch.setattr(
        "app.workflow.validation._default_runtime_executor",
        lambda _code, _context, _style, _shape=None: Execution(),
    )
    monkeypatch.setattr(
        "app.workflow.validation._default_luac_locator",
        lambda: "/usr/bin/true",
    )

    result = runner.invoke(cli, ["validate", "--code", "return 1"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert [check["name"] for check in payload["validation"]["checks"]] == [
        "output_contract",
        "ast_policy",
        "luac",
        "sandbox",
    ]


def test_doctor_flag_parses_as_boolean(monkeypatch):
    quality_models = []
    monkeypatch.setattr(OllamaBackend, "ping", lambda self: True)
    monkeypatch.setattr("app.cli.main.find_lua_binary", lambda: "/usr/bin/lua")
    monkeypatch.setattr("app.cli.main.find_luac_binary", lambda: "/usr/bin/luac")
    monkeypatch.setattr(
        OllamaBackend,
        "list_tags",
        lambda self: ["hf.co/unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M"],
    )
    result = runner.invoke(cli, ["doctor"])
    assert result.exit_code == 0
    effective = json.loads(result.stdout)
    assert effective["profile"] == "local"
    assert effective["model"] == "hf.co/unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M"
    assert effective["num_ctx"] == 8192
    assert effective["num_predict"] == 2048

    def run_same_model_quality(profile=None, backend=None, mode="competition"):
        assert profile is backend.profile
        quality_models.append(profile.model)
        return _green_quality_report()

    monkeypatch.setattr("app.cli.main.run_quality_benchmark", run_same_model_quality)

    judge_result = runner.invoke(cli, ["doctor", "--eval"])
    assert judge_result.exit_code == 0
    payload = json.loads(judge_result.stdout)
    assert payload["judge_mode"] is True
    assert payload["quality_failures"] == []
    assert "selected_model" not in payload
    assert "vram_report" not in payload
    assert "runtime_snapshot_path" not in payload
    assert quality_models == ["hf.co/unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M"]


def test_doctor_fails_when_the_effective_model_is_unavailable(monkeypatch):
    def unavailable(_self):
        raise BackendUnavailable(reason="test_backend_unavailable")

    monkeypatch.setattr(OllamaBackend, "list_tags", unavailable)

    result = runner.invoke(cli, ["doctor"])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["ollama_reachable"] is False
    assert payload["model_present"] is False


def test_generate_command_continues_a_clarification_session(monkeypatch):
    class DummyEngine:
        def generate(self, **kwargs):
            assert kwargs["session_id"] == "session-1"
            assert kwargs["clarification_answer"] == "Use wf.vars."
            assert "context" not in kwargs
            return GenerationResult(
                workflow=WorkflowResult(
                    status=WorkflowStatus.CLARIFICATION_REQUIRED,
                    question="Use wf.vars or wf.initVariables?",
                ),
                trace_id="trace-1",
                session_id="session-1",
                session=SessionSummary(
                    session_id="session-1",
                    status=SessionStatus.CLARIFICATION_REQUIRED,
                    original_task="Normalize email.",
                    open_clarification_question="Use wf.vars or wf.initVariables?",
                ),
            )

    monkeypatch.setattr("app.cli.main.build_engine", lambda: DummyEngine())

    result = runner.invoke(
        cli,
        ["generate", "--session-id", "session-1", "--answer", "Use wf.vars."],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "clarification_required"
    assert payload["question"] == "Use wf.vars or wf.initVariables?"
    assert "code" not in payload


def test_generate_command_requires_a_prompt_or_a_session(monkeypatch):
    monkeypatch.setattr("app.cli.main.build_engine", lambda: None)

    result = runner.invoke(cli, ["generate"])

    assert result.exit_code != 0
    assert json.loads(result.stdout)["error"]["code"] == "prompt_or_session_required"


def test_generate_reports_output_and_validation_preview(monkeypatch):
    engine = CapturingEngine()
    monkeypatch.setattr("app.cli.main.build_engine", lambda: engine)

    result = runner.invoke(cli, ["generate", "--prompt", "Return wf.vars.value."])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["output"] == {
        "format": "lua_block",
        "shape": "scalar",
        "nullable": False,
    }
    assert payload["validation"]["observations"] == [{"actual": 7}]
    assert engine.backend.closed is True


def test_generate_failure_omits_code_and_question(monkeypatch):
    failed = GenerationResult(
        workflow=WorkflowResult(
            status=WorkflowStatus.VALIDATION_FAILED,
            validation=ValidationResult(
                checks=(
                    ValidationCheck(
                        name="sandbox",
                        status=CheckStatus.FAILED,
                        code="runtime_failed",
                        message="Candidate execution failed.",
                    ),
                )
            ),
            output=OutputContract(format=OutputFormat.LUA_BLOCK, shape=OutputShape.SCALAR),
        ),
        trace_id="trace-2",
        session_id="session-2",
        session=SessionSummary(
            session_id="session-2",
            status=SessionStatus.VALIDATION_FAILED,
            original_task="Return the value.",
        ),
    )
    engine = CapturingEngine(result=failed)
    monkeypatch.setattr("app.cli.main.build_engine", lambda: engine)

    result = runner.invoke(cli, ["generate", "--prompt", "Return the value."])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert "code" not in payload
    assert "question" not in payload
    assert payload["validation"]["checks"][0]["code"] == "runtime_failed"
    assert payload["output"]["shape"] == "scalar"


def test_generate_omitted_context_is_not_forwarded_but_explicit_null_is(monkeypatch):
    omitted_engine = CapturingEngine()
    monkeypatch.setattr("app.cli.main.build_engine", lambda: omitted_engine)

    omitted = runner.invoke(
        cli,
        ["generate", "--session-id", "session-1", "--feedback", "Retry."],
    )

    assert omitted.exit_code == 0
    assert "context" not in omitted_engine.kwargs

    null_engine = CapturingEngine()
    monkeypatch.setattr("app.cli.main.build_engine", lambda: null_engine)
    explicit_null = runner.invoke(
        cli,
        ["generate", "--session-id", "session-1", "--context", "null"],
    )

    assert explicit_null.exit_code == 0
    assert null_engine.kwargs["context"] is None


def test_generate_reads_prompt_and_context_files(tmp_path, monkeypatch):
    prompt_path = tmp_path / "prompt.txt"
    context_path = tmp_path / "context.json"
    prompt_path.write_text("Return wf.vars.value.", encoding="utf-8")
    context_path.write_text('{"wf":{"vars":{"value":7}}}', encoding="utf-8")
    engine = CapturingEngine()
    monkeypatch.setattr("app.cli.main.build_engine", lambda: engine)

    result = runner.invoke(
        cli,
        [
            "generate",
            "--prompt-file",
            str(prompt_path),
            "--context-file",
            str(context_path),
        ],
    )

    assert result.exit_code == 0
    assert engine.kwargs["prompt"] == "Return wf.vars.value."
    assert engine.kwargs["context"] == {"wf": {"vars": {"value": 7}}}


def test_generate_accepts_each_stdin_source_without_reading_stdin_twice(monkeypatch):
    prompt_engine = CapturingEngine()
    monkeypatch.setattr("app.cli.main.build_engine", lambda: prompt_engine)
    prompt_result = runner.invoke(
        cli,
        ["generate", "--prompt-file", "-", "--context", '{"wf":{"vars":{}}}'],
        input="Return 1.\n",
    )

    assert prompt_result.exit_code == 0
    assert prompt_engine.kwargs["prompt"] == "Return 1.\n"

    context_engine = CapturingEngine()
    monkeypatch.setattr("app.cli.main.build_engine", lambda: context_engine)
    context_result = runner.invoke(
        cli,
        ["generate", "--prompt", "Return 1.", "--context-file", "-"],
        input='{"wf":{"vars":{}}}',
    )

    assert context_result.exit_code == 0
    assert context_engine.kwargs["context"] == {"wf": {"vars": {}}}

    conflict = runner.invoke(
        cli,
        ["generate", "--prompt-file", "-", "--context-file", "-"],
        input="unused",
    )
    assert conflict.exit_code == 2
    assert json.loads(conflict.stdout)["error"]["code"] == "stdin_source_conflict"


def test_generate_typed_errors_are_safe_json_and_close_backend(monkeypatch):
    session_engine = CapturingEngine(
        error=SessionStateError(
            code="session_conflict",
            status_code=409,
            message="The session cannot accept this continuation.",
        )
    )
    monkeypatch.setattr("app.cli.main.build_engine", lambda: session_engine)

    session_result = runner.invoke(cli, ["generate", "--session-id", "session-1"])

    assert session_result.exit_code == 1
    assert json.loads(session_result.stdout) == {
        "ok": False,
        "error": {
            "code": "session_conflict",
            "message": "The session cannot accept this continuation.",
        },
    }
    assert "Traceback" not in session_result.stdout
    assert session_engine.backend.closed is True

    backend_engine = CapturingEngine(error=BackendUnavailable(reason="private-detail"))
    monkeypatch.setattr("app.cli.main.build_engine", lambda: backend_engine)
    backend_result = runner.invoke(cli, ["generate", "--prompt", "Return 1."])

    assert backend_result.exit_code == 1
    assert json.loads(backend_result.stdout)["error"] == {
        "code": "backend_unavailable",
        "message": "Backend is unavailable.",
    }
    assert "private-detail" not in backend_result.stdout
    assert backend_engine.backend.closed is True


def test_generate_rejects_inline_and_file_sources_together(tmp_path, monkeypatch):
    prompt_path = tmp_path / "prompt.txt"
    context_path = tmp_path / "context.json"
    prompt_path.write_text("Return 1.", encoding="utf-8")
    context_path.write_text('{"wf":{"vars":{}}}', encoding="utf-8")
    monkeypatch.setattr("app.cli.main.build_engine", lambda: None)

    result = runner.invoke(
        cli,
        ["generate", "--prompt", "Return 2.", "--prompt-file", str(prompt_path)],
    )

    assert result.exit_code == 2
    assert json.loads(result.stdout)["error"]["code"] == "prompt_source_conflict"

    context_result = runner.invoke(
        cli,
        [
            "generate",
            "--prompt",
            "Return 1.",
            "--context",
            '{"wf":{"vars":{}}}',
            "--context-file",
            str(context_path),
        ],
    )
    assert context_result.exit_code == 2
    assert json.loads(context_result.stdout)["error"]["code"] == "context_source_conflict"


def test_validate_rejects_empty_and_fenced_code_as_safe_json():
    empty = runner.invoke(cli, ["validate", "--code", ""])
    fenced = runner.invoke(cli, ["validate", "--code", "```lua\nreturn 1\n```"])

    assert empty.exit_code == 2
    assert json.loads(empty.stdout)["error"]["code"] == "code_empty"
    assert fenced.exit_code == 2
    assert json.loads(fenced.stdout)["error"]["code"] == "code_markdown_fence_forbidden"
    assert "Traceback" not in empty.stdout + fenced.stdout


def test_benchmark_closes_backend_after_typed_failure(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("", encoding="utf-8")

    class Backend:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    backend = Backend()
    monkeypatch.setattr("app.cli.main.OllamaBackend", lambda _profile: backend)

    def unavailable(*_args, **_kwargs):
        raise BackendUnavailable(reason="private-detail")

    monkeypatch.setattr("app.cli.main.run_dataset_benchmark", unavailable)

    result = runner.invoke(cli, ["benchmark", "--dataset", str(dataset)])

    assert result.exit_code == 1
    assert json.loads(result.stdout)["error"] == {
        "code": "backend_unavailable",
        "message": "Backend is unavailable.",
    }
    assert backend.closed is True


def test_doctor_closes_backend_when_live_evaluation_fails(monkeypatch):
    class Backend:
        def __init__(self, profile):
            self.profile = profile
            self.closed = False

        def list_tags(self):
            return [self.profile.model]

        def close(self):
            self.closed = True

    backend_holder = []

    def build_backend(profile):
        backend = Backend(profile)
        backend_holder.append(backend)
        return backend

    monkeypatch.setattr("app.cli.main.OllamaBackend", build_backend)
    monkeypatch.setattr("app.cli.main.find_lua_binary", lambda: "/usr/bin/lua")
    monkeypatch.setattr("app.cli.main.find_luac_binary", lambda: "/usr/bin/luac")

    def unavailable(*_args, **_kwargs):
        raise BackendUnavailable(reason="private-detail")

    monkeypatch.setattr("app.cli.main.run_quality_benchmark", unavailable)

    result = runner.invoke(cli, ["doctor", "--eval"])

    assert result.exit_code == 1
    assert json.loads(result.stdout)["error"]["code"] == "backend_unavailable"
    assert backend_holder[0].closed is True
