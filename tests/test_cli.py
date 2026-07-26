import json

from typer.testing import CliRunner

from app.cli.main import cli
from app.core import config as config_module
from app.core.benchmarks import QUALITY_EVAL_MANIFEST
from app.generation.engine import GenerationResult
from app.generation.ollama import OllamaBackend


runner = CliRunner()


def test_generate_command_returns_contract_json(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALSCRIPT_TRACE_DIR", str(tmp_path / "traces"))

    class DummyEngine:
        def generate(self, prompt=None, session_id=None, feedback=None):
            return GenerationResult(
                code="return wf.vars.try_count_n + 1",
                trace_id="trace-1",
                session_id="session-1",
                strategy="ollama_chain",
                verification_errors=[],
                validation_report={"has_errors": False, "has_warnings": False, "messages": []},
                repair_rounds=0,
                degraded_mode=False,
            )

    monkeypatch.setattr("app.cli.main.build_engine", lambda: DummyEngine())

    result = runner.invoke(
        cli,
        ["generate", "--prompt", "Увеличь wf.vars.try_count_n ровно на единицу и верни новый счётчик."],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["code"] == "return wf.vars.try_count_n + 1"


def test_verify_command_rejects_jsonpath():
    result = runner.invoke(cli, ["verify", "--code", "return $.wf.vars.email"])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert "jsonpath_forbidden" in payload["errors"]


def test_doctor_flag_parses_as_boolean(monkeypatch):
    result = runner.invoke(cli, ["doctor"])
    assert result.exit_code == 0

    monkeypatch.setattr(OllamaBackend, "ping", lambda self: True)
    monkeypatch.setattr(
        "app.cli.main.run_quality_benchmark",
        lambda profile=None, backend=None, mode="competition": {
            "backend_type": "live_ollama",
            "eval_manifest": [
                {"name": entry["name"], "role": entry["role"]}
                for entry in QUALITY_EVAL_MANIFEST
            ],
            "public_gold": {"ok": True},
            "stress_eval": {"ok": True},
            "showcase_eval": {"ok": True},
            "model_backed_eval": {"ok": True},
            "multilingual_eval": {"ok": True},
            "ambiguity_eval": {"ok": True},
            "clarification_eval": {"ok": True},
            "composition_eval": {"ok": True},
            "regression_eval": {"ok": True},
            "adversarial_eval": {"ok": True},
            "large_context_eval": {"ok": True},
            "adversarial_ok": True,
            "ok": True,
        },
    )

    class Completed:
        def __init__(self, stdout):
            self.stdout = stdout
            self.stderr = ""

    monkeypatch.setattr(
        "app.cli.main.subprocess.run",
        lambda args, stdout=None, stderr=None, text=None: Completed(
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
                {"name": entry["name"], "role": entry["role"]}
                for entry in QUALITY_EVAL_MANIFEST
            ],
            "public_gold": {"ok": True},
            "stress_eval": {"ok": True},
            "showcase_eval": {"ok": True},
            "model_backed_eval": {"ok": True},
            "multilingual_eval": {"ok": True},
            "ambiguity_eval": {"ok": True},
            "clarification_eval": {"ok": True},
            "composition_eval": {"ok": True},
            "regression_eval": {"ok": True},
            "large_context_eval": {"ok": True},
            "adversarial_eval": {"ok": True},
            "adversarial_ok": True,
            "ok": True,
        }

    monkeypatch.setattr("app.cli.main.run_quality_benchmark", fake_quality_benchmark)

    class Completed:
        def __init__(self, stdout):
            self.stdout = stdout
            self.stderr = ""

    def fake_run(args, stdout=None, stderr=None, text=None):
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
    assert json.loads(lock_path.read_text(encoding="utf-8"))["selected_model"] == "qwen3:4b-instruct-2507-q4_K_M"
    config_module.get_runtime_profile.cache_clear()


def test_interact_command_returns_rich_agent_payload(monkeypatch):
    class DummyEngine:
        def generate_rich(self, prompt=None, session_id=None, clarification_answer=None, feedback=None):
            return GenerationResult(
                code="",
                trace_id="trace-1",
                session_id="session-1",
                strategy="clarification",
                verification_errors=[],
                validation_report={"has_errors": False, "has_warnings": False, "messages": []},
                repair_rounds=0,
                degraded_mode=False,
                status="clarification_needed",
                question="Use wf.vars or wf.initVariables?",
                assumptions=["Ambiguous root detected."],
                session_summary={
                    "session_id": "session-1",
                    "status": "clarification_needed",
                    "original_task": "Normalize email.",
                    "latest_trace_id": "trace-1",
                    "last_strategy": "clarification",
                    "open_clarification_question": "Use wf.vars or wf.initVariables?",
                    "clarification_history": [],
                    "feedback_history": [],
                    "assumptions": ["Ambiguous root detected."],
                },
            )

    monkeypatch.setattr("app.cli.main.build_engine", lambda: DummyEngine())

    result = runner.invoke(cli, ["interact", "--prompt", "Normalize email."])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "clarification_needed"
    assert payload["question"] == "Use wf.vars or wf.initVariables?"
    assert payload["session"]["status"] == "clarification_needed"


def test_analyze_command_returns_route_inspection(monkeypatch):
    class DummyEngine:
        def analyze(self, prompt, context=None):
            return {
                "normalized_prompt": "normalize email",
                "suggested_strategy": "clarification",
                "clarification_question": "Use wf.vars or wf.initVariables?",
                "task_spec": {"family": None, "safety_fallback": False},
                "reduced_context": {"roots": ["wf.vars.email", "wf.initVariables.email"]},
                "available_paths": ["wf.vars.email", "wf.initVariables.email"],
                "assumptions": [],
                "ambiguity_notes": ["mixed root"],
            }

    monkeypatch.setattr("app.cli.main.build_engine", lambda: DummyEngine())

    result = runner.invoke(cli, ["analyze", "--prompt", "Normalize email."])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["suggested_strategy"] == "clarification"
    assert payload["clarification_question"] == "Use wf.vars or wf.initVariables?"
