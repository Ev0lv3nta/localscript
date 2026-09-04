import json

from typer.testing import CliRunner

from app.cli.main import cli
from app.core import config as config_module
from app.core.benchmarks import QUALITY_EVAL_MANIFEST
from app.generation.ollama import OllamaBackend
from app.generation.results import GenerationResult, SessionStatus, SessionSummary
from app.workflow.contracts import (
    CheckStatus,
    ValidationCheck,
    ValidationResult,
    WorkflowResult,
    WorkflowStatus,
)

runner = CliRunner()


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
        lambda _code, _context, _style: Execution(),
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
    result = runner.invoke(cli, ["doctor"])
    assert result.exit_code == 0

    monkeypatch.setattr(OllamaBackend, "ping", lambda self: True)
    monkeypatch.setattr(
        "app.cli.main.run_quality_benchmark",
        lambda profile=None, backend=None, mode="competition": {
            "backend_type": "live_ollama",
            "eval_manifest": [
                {
                    "name": entry["name"],
                    "path": entry["path"],
                    "corpus": entry["corpus"],
                    "gate": entry["gate"],
                    "claim_scope": entry["claim_scope"],
                }
                for entry in QUALITY_EVAL_MANIFEST
            ],
            "live_v1": {"ok": True},
            "ok": True,
        },
    )

    class Completed:
        def __init__(self, stdout):
            self.stdout = stdout
            self.stderr = ""

    monkeypatch.setattr(
        "app.cli.main.subprocess.run",
        lambda args, capture_output=None, text=None: Completed(
            json.dumps({"status": "ok", "model": args[2]})
        ),
    )
    monkeypatch.setattr(
        OllamaBackend,
        "list_tags",
        lambda self: ["qwen3:8b-q4_K_M", "qwen3:4b-instruct-2507-q4_K_M"],
    )

    judge_result = runner.invoke(cli, ["doctor", "--judge"])
    assert judge_result.exit_code == 0
    payload = json.loads(judge_result.stdout)
    assert payload["judge_mode"] is True
    assert payload["selection_reason"] in {
        "primary_selected",
        "primary_selected_vram_skipped",
        "primary_within_vram_cap",
    }


def test_doctor_judge_switches_to_fallback_when_primary_over_cap(monkeypatch, tmp_path):
    lock_path = tmp_path / ".runtime_profile.lock.json"
    benchmark_models = []
    monkeypatch.setenv("LOCALSCRIPT_RUNTIME_LOCK_PATH", str(lock_path))
    monkeypatch.setenv("LOCALSCRIPT_TRACE_DIR", str(tmp_path / "traces"))
    config_module.get_runtime_profile.cache_clear()

    monkeypatch.setattr(OllamaBackend, "ping", lambda self: True)
    monkeypatch.setattr(
        OllamaBackend,
        "list_tags",
        lambda self: ["qwen3:8b-q4_K_M", "qwen3:4b-instruct-2507-q4_K_M"],
    )

    def fake_quality_benchmark(profile=None, backend=None, mode="competition"):
        benchmark_models.append(profile.model)
        assert backend.profile.model == profile.model
        return {
            "backend_type": "live_ollama",
            "eval_manifest": [
                {
                    "name": entry["name"],
                    "path": entry["path"],
                    "corpus": entry["corpus"],
                    "gate": entry["gate"],
                    "claim_scope": entry["claim_scope"],
                }
                for entry in QUALITY_EVAL_MANIFEST
            ],
            "live_v1": {"ok": True},
            "ok": True,
        }

    monkeypatch.setattr("app.cli.main.run_quality_benchmark", fake_quality_benchmark)

    class Completed:
        def __init__(self, stdout):
            self.stdout = stdout
            self.stderr = ""

    def fake_run(args, capture_output=None, text=None):
        assert args[0] == "bash"
        model = args[2]
        status = "over_cap" if model == "qwen3:8b-q4_K_M" else "ok"
        return Completed(json.dumps({"status": status, "model": model}))

    monkeypatch.setattr("app.cli.main.subprocess.run", fake_run)

    result = runner.invoke(cli, ["doctor", "--judge"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["selected_model"] == "qwen3:4b-instruct-2507-q4_K_M"
    assert payload["selection_reason"] == "primary_over_vram_cap"
    assert payload["hard_gate_failures"] == []
    assert benchmark_models == ["qwen3:4b-instruct-2507-q4_K_M"]
    assert (
        json.loads(lock_path.read_text(encoding="utf-8"))["selected_model"]
        == "qwen3:4b-instruct-2507-q4_K_M"
    )
    config_module.get_runtime_profile.cache_clear()


def test_generate_command_continues_a_clarification_session(monkeypatch):
    class DummyEngine:
        def generate(self, **kwargs):
            assert kwargs["session_id"] == "session-1"
            assert kwargs["clarification_answer"] == "Use wf.vars."
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
    assert payload["code"] is None


def test_generate_command_requires_a_prompt_or_a_session(monkeypatch):
    monkeypatch.setattr("app.cli.main.build_engine", lambda: None)

    result = runner.invoke(cli, ["generate"])

    assert result.exit_code != 0
