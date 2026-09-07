import json
from types import SimpleNamespace

import pytest

from app.core import benchmarks
from app.core.benchmarks import (
    InstrumentedBackend,
    _aggregate_metrics,
    _backend_type,
    _stage_durations,
)
from app.core.public_eval import (
    EvaluationStatus,
    IndependentEvaluation,
    evaluate_case_detailed,
    validate_dataset_cases,
)
from app.generation.backend_errors import BackendTimeout
from app.workflow.contracts import CheckStatus
from tests.support_backends import FailIfCalledBackend


def public_case(**overrides):
    first = {"wf": {"vars": {"value": 1}}}
    second = {"wf": {"vars": {"value": 2}}}
    case = {
        "id": "public_value",
        "prompt": "Верни wf.vars.value.",
        "context": first,
        "output": {"format": "lua_block", "shape": "scalar", "nullable": False},
        "fixtures": [
            {"name": "one", "context": first, "expected_result": 1},
            {"name": "two", "context": second, "expected_result": 2},
        ],
        "expected_status": "completed",
        "case_type": "transformation",
        "category": "scalar",
        "scenario": "read_value",
        "safety": False,
        "source": "owner_synthetic_public_v2",
    }
    case.update(overrides)
    return case


def write_cases(path, cases):
    path.write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases),
        encoding="utf-8",
    )


def completed_generation():
    return SimpleNamespace(
        workflow=SimpleNamespace(
            status=SimpleNamespace(value="completed"),
            code="return wf.vars.value",
            diagnostics=(),
            revision_count=0,
            validation=SimpleNamespace(
                checks=(SimpleNamespace(name="luac", status=CheckStatus.PASSED),)
            ),
        ),
        session_id="session",
        trace_id="",
    )


@pytest.mark.parametrize(
    ("cases", "error_code"),
    [
        (
            [
                {
                    "id": "missing",
                    "prompt": "Return a value.",
                    "context": {"wf": {"vars": {"value": 1}}},
                    "expected_output_style": "lua_block",
                }
            ],
            "dataset_expected_result_missing",
        ),
        ([public_case(), public_case()], "dataset_case_id_duplicate"),
        ([public_case(expected_status="almost_done")], "dataset_expected_status_invalid"),
        (
            [public_case(output={"format": "json_envelope", "shape": "scalar"})],
            "dataset_output_contract_invalid",
        ),
        (
            [
                public_case(
                    context={"wf": {"vars": {"value": None}}},
                    fixtures=[
                        {
                            "name": "unsupported_null",
                            "context": {"wf": {"vars": {"value": None}}},
                            "expected_result": 1,
                        },
                        {
                            "name": "valid",
                            "context": {"wf": {"vars": {"value": 2}}},
                            "expected_result": 2,
                        },
                    ],
                )
            ],
            "dataset_context_invalid",
        ),
        (
            [
                public_case(
                    case_type="clarification",
                    expected_status="clarification_required",
                    expected_final_status="completed",
                    clarification_source_roots=["wf.unknown"],
                )
            ],
            "dataset_clarification_transition_invalid",
        ),
        (
            [public_case(source_roots=["wf.vars", "wf.vars"])],
            "dataset_source_roots_invalid",
        ),
    ],
)
def test_dataset_preflight_rejects_invalid_cases_before_backend_call(tmp_path, cases, error_code):
    dataset = tmp_path / "invalid.jsonl"
    write_cases(dataset, cases)

    with pytest.raises(ValueError, match=error_code):
        benchmarks.run_dataset_benchmark(dataset, backend=FailIfCalledBackend())


def test_explicit_null_expected_is_not_treated_as_missing():
    case = public_case(
        output={"format": "lua_block", "shape": "scalar", "nullable": True},
        fixtures=[
            {
                "name": "missing_one",
                "context": {"wf": {"vars": {"value": 1}}},
                "expected_result": None,
            },
            {
                "name": "missing_two",
                "context": {"wf": {"vars": {"value": 2}}},
                "expected_result": None,
            },
        ],
    )

    assert validate_dataset_cases([case]) == [case]


def test_independent_oracle_runs_same_code_on_every_fixture(monkeypatch):
    seen: list[tuple[str, int]] = []

    class Execution:
        ok = True
        degraded = False
        error_code = ""

        def __init__(self, value):
            self.value = value

    def execute(code, context, *_args):
        value = context["wf"]["vars"]["value"]
        seen.append((code, value))
        return Execution(value)

    monkeypatch.setattr("app.core.public_eval.execute_output", execute)

    result = evaluate_case_detailed("return wf.vars.value", public_case())

    assert result.status is EvaluationStatus.PASSED
    assert result.fixtures_attempted == 2
    assert seen == [("return wf.vars.value", 1), ("return wf.vars.value", 2)]


def test_runner_never_forwards_oracle_fixtures_as_model_examples(monkeypatch):
    calls = []

    class FakeEngine:
        def __init__(self, **_kwargs):
            pass

        def generate(self, **kwargs):
            calls.append(kwargs)
            return completed_generation()

    monkeypatch.setattr(benchmarks, "GenerationEngine", FakeEngine)
    monkeypatch.setattr(
        benchmarks,
        "evaluate_case_detailed",
        lambda *_args: IndependentEvaluation(
            status=EvaluationStatus.PASSED,
            errors=(),
            fixtures_total=2,
            fixtures_attempted=2,
            fixtures_passed=2,
        ),
    )

    results = benchmarks._run_cases(
        cases=[public_case()],
        runtime_profile=benchmarks.get_runtime_profile(),
        runtime_backend=FailIfCalledBackend(),
    )

    assert results[0]["passed"] is True
    assert len(calls) == 1
    assert set(calls[0]) == {"prompt", "context", "output"}
    assert calls[0]["context"] == public_case()["context"]
    assert "fixtures" not in str(calls[0])


def test_runner_forwards_explicit_source_roots_on_initial_generation(monkeypatch):
    calls = []

    class FakeEngine:
        def __init__(self, **_kwargs):
            pass

        def generate(self, **kwargs):
            calls.append(kwargs)
            return completed_generation()

    monkeypatch.setattr(benchmarks, "GenerationEngine", FakeEngine)
    monkeypatch.setattr(
        benchmarks,
        "evaluate_case_detailed",
        lambda *_args: IndependentEvaluation(
            status=EvaluationStatus.PASSED,
            errors=(),
            fixtures_total=2,
            fixtures_attempted=2,
            fixtures_passed=2,
        ),
    )
    case = public_case(source_roots=["wf.vars", "wf.initVariables"])

    results = benchmarks._run_cases(
        cases=[case],
        runtime_profile=benchmarks.get_runtime_profile(),
        runtime_backend=FailIfCalledBackend(),
    )

    assert results[0]["passed"] is True
    assert calls[0]["source_roots"] == (
        benchmarks.WorkflowRoot.VARS,
        benchmarks.WorkflowRoot.INIT_VARIABLES,
    )


def test_not_run_checks_do_not_become_perfect_rates():
    case_results = [
        {
            "passed": False,
            "status": "backend_unavailable",
            "revision_count": 0,
            "syntax_status": "not_run",
            "semantic_status": "not_run",
            "case_type": "transformation",
            "safety": False,
            "duration_ms": 2.0,
            "backend_calls": 1,
            "model_call_durations_ms": [1.5],
            "model_duration_ms": 1.5,
            "stage_durations_ms": {},
        }
    ]

    metrics = _aggregate_metrics(case_results)

    assert metrics["syntax_attempted_count"] == 0
    assert metrics["syntax_pass_rate"] is None
    assert metrics["semantic_attempted_count"] == 0
    assert metrics["semantic_pass_rate"] is None
    assert metrics["supported_success_rate"] == 0.0
    assert "cold_first" not in metrics["latency_ms"]


def test_backend_error_is_recorded_per_case_with_checks_not_run(tmp_path):
    dataset = tmp_path / "backend-error.jsonl"
    write_cases(dataset, [public_case()])

    class TimedOutBackend:
        def complete(self, *_args, **_kwargs):
            raise BackendTimeout()

    report = benchmarks.run_dataset_benchmark(dataset, backend=TimedOutBackend())

    assert report["failed"] == 1
    assert report["case_results"][0]["status"] == "backend_timeout"
    assert report["case_results"][0]["syntax_status"] == "not_run"
    assert report["case_results"][0]["semantic_status"] == "not_run"
    assert report["metrics"]["syntax_pass_rate"] is None
    assert report["metrics"]["semantic_pass_rate"] is None


def test_repeated_stage_durations_are_summed():
    trace = {
        "stage_events": [
            {"stage": "generating", "duration_ms": 10.5},
            {"stage": "validating", "duration_ms": 4},
            {"stage": "generating", "duration_ms": 2.25},
        ]
    }

    assert _stage_durations(trace) == {"generating": 12.75, "validating": 4.0}


def test_backend_type_cannot_be_spoofed_by_evidence_attribute():
    class FakeBackend:
        evidence_backend_type = "live_ollama"

    assert _backend_type(FakeBackend()) == "fake_backend"


def test_instrumentation_preserves_each_backend_call_metrics():
    class MetricsBackend:
        last_call_metrics = None

        def complete(self, _prompt, **_kwargs):
            self.last_call_metrics = {"eval_count": 7, "total_duration": 123}
            return "result"

    backend = InstrumentedBackend(MetricsBackend())

    assert backend.complete("first") == "result"
    assert backend.complete("second") == "result"
    assert len(backend.calls) == 2
    assert backend.calls[0]["backend_metrics"] == {
        "eval_count": 7,
        "total_duration": 123,
    }
    assert backend.calls[1]["backend_metrics"] == {
        "eval_count": 7,
        "total_duration": 123,
    }


def test_runner_closes_only_a_backend_it_constructs(tmp_path, monkeypatch):
    dataset = tmp_path / "valid.jsonl"
    legacy = {
        "id": "legacy",
        "prompt": "Return the value.",
        "context": {"wf": {"vars": {"value": 1}}},
        "expected_output_style": "lua_block",
        "expected_result": 1,
    }
    write_cases(dataset, [legacy])

    created = []

    class OwnedBackend:
        closed = False
        last_resolved_model = None
        base_url = "http://127.0.0.1:11434"

        def __init__(self, _profile):
            created.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setattr(benchmarks, "OllamaBackend", OwnedBackend)
    monkeypatch.setattr(benchmarks, "_run_cases", lambda **_kwargs: [])

    benchmarks.run_dataset_benchmark(dataset)

    assert created[0].closed is True

    class InjectedBackend:
        closed = False

        def close(self):
            self.closed = True

    injected = InjectedBackend()
    benchmarks.run_dataset_benchmark(dataset, backend=injected)
    assert injected.closed is False
