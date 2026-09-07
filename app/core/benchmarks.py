from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

from app.core.config import PROJECT_ROOT, get_runtime_profile
from app.core.public_eval import (
    EvaluationStatus,
    evaluate_case_detailed,
    load_cases_bytes,
    output_contract_for_case,
    validate_dataset_cases,
)
from app.core.resources import materialized_resource, resource_exists
from app.core.state import get_state_root
from app.core.traces import TraceStore
from app.evaluation.holdout import adapt_blind_holdout_cases
from app.evaluation.manifest import dataset_specs, stability_plan
from app.generation.backend_errors import BackendError
from app.generation.engine import GenerationEngine
from app.generation.ollama import OllamaBackend
from app.generation.results import GenerationResult
from app.workflow.contracts import CheckStatus, WorkflowRoot, WorkflowStatus

if TYPE_CHECKING:
    from app.core.config import RuntimeProfile

QUALITY_EVAL_MANIFEST = tuple(spec.evidence_dict() for spec in dataset_specs())
OLLAMA_BACKEND_TYPE = OllamaBackend
_NOT_RUN = EvaluationStatus.NOT_RUN.value


class InstrumentedBackend:
    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self.calls: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.backend, name)

    def complete(
        self,
        prompt: str,
        response_format: object | None = None,
        model: str | None = None,
    ) -> str:
        started = perf_counter()
        error_code: str | None = None
        try:
            method = getattr(self.backend, "complete", None)
            if callable(method):
                return str(method(prompt, response_format=response_format, model=model))
            return str(self.backend.generate(prompt))
        except BackendError as error:
            error_code = error.code
            raise
        finally:
            call: dict[str, Any] = {
                "duration_ms": round((perf_counter() - started) * 1000.0, 3),
                "requested_model": model,
                "error_code": error_code,
            }
            backend_metrics = getattr(self.backend, "last_call_metrics", None)
            if isinstance(backend_metrics, dict):
                call["backend_metrics"] = dict(backend_metrics)
            self.calls.append(call)

    def generate(self, prompt: str, context: Any = None) -> str:
        started = perf_counter()
        error_code: str | None = None
        try:
            return str(self.backend.generate(prompt, context=context))
        except BackendError as error:
            error_code = error.code
            raise
        finally:
            call: dict[str, Any] = {
                "duration_ms": round((perf_counter() - started) * 1000.0, 3),
                "requested_model": None,
                "error_code": error_code,
            }
            backend_metrics = getattr(self.backend, "last_call_metrics", None)
            if isinstance(backend_metrics, dict):
                call["backend_metrics"] = dict(backend_metrics)
            self.calls.append(call)


def _load_cases_snapshot(dataset_path: Path | str) -> tuple[list[dict[str, Any]], str]:
    payload = Path(dataset_path).read_bytes()
    cases = adapt_blind_holdout_cases(load_cases_bytes(payload))
    validate_dataset_cases(cases)
    return cases, hashlib.sha256(payload).hexdigest()


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    position = (len(ordered) - 1) * float(percentile)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return round(ordered[lower] * (1.0 - weight) + ordered[upper] * weight, 3)


def _deduplicate(items: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def _stage_durations(trace_payload: dict[str, Any] | None) -> dict[str, float]:
    durations: dict[str, float] = {}
    for event in (trace_payload or {}).get("stage_events", []):
        if not isinstance(event, dict):
            continue
        stage = event.get("stage")
        duration = event.get("duration_ms")
        if isinstance(stage, str) and isinstance(duration, (int, float)):
            durations[stage] = round(durations.get(stage, 0.0) + float(duration), 3)
    return durations


def _syntax_status(result: GenerationResult | None) -> str:
    validation = result.workflow.validation if result is not None else None
    if validation is None:
        return _NOT_RUN
    check = next((item for item in validation.checks if item.name == "luac"), None)
    if check is None:
        return _NOT_RUN
    return (
        EvaluationStatus.PASSED.value
        if check.status is CheckStatus.PASSED
        else EvaluationStatus.FAILED.value
    )


def _case_observation(
    *,
    case: dict[str, Any],
    result: GenerationResult | None,
    status: str,
    initial_status: str | None,
    errors: Sequence[str],
    semantic_status: str,
    fixtures_total: int,
    fixtures_attempted: int,
    fixtures_passed: int,
    duration_ms: float,
    backend_calls: Sequence[dict[str, Any]],
    trace_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    model_call_durations = [
        round(float(call["duration_ms"]), 3)
        for call in backend_calls
        if isinstance(call.get("duration_ms"), (int, float))
    ]
    passed = not errors
    return {
        "id": case["id"],
        "case_type": case.get("case_type", "unknown"),
        "category": case.get("category"),
        "safety": bool(case.get("safety", False)),
        "passed": passed,
        "status": status,
        "initial_status": initial_status,
        "revision_count": result.workflow.revision_count if result is not None else 0,
        "syntax_status": _syntax_status(result),
        "semantic_status": semantic_status,
        "fixtures_total": fixtures_total,
        "fixtures_attempted": fixtures_attempted,
        "fixtures_passed": fixtures_passed,
        "duration_ms": round(duration_ms, 3),
        "backend_calls": len(backend_calls),
        "model_calls": [dict(call) for call in backend_calls],
        "model_call_durations_ms": model_call_durations,
        "model_duration_ms": round(sum(model_call_durations), 3),
        "stage_durations_ms": _stage_durations(trace_payload),
        "errors": _deduplicate(errors),
    }


def _aggregate_metrics(case_results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = len(case_results)
    passed = sum(bool(item["passed"]) for item in case_results)
    invalid_successes = sum(
        item["status"] == WorkflowStatus.COMPLETED.value and not item["passed"]
        for item in case_results
    )
    syntax_attempted = [item for item in case_results if item["syntax_status"] != _NOT_RUN]
    semantic_attempted = [item for item in case_results if item["semantic_status"] != _NOT_RUN]
    supported = [item for item in case_results if item["case_type"] == "transformation"]
    clarifications = [item for item in case_results if item["case_type"] == "clarification"]
    safety = [item for item in case_results if item["safety"]]
    expected_refusals = [item for item in case_results if item["case_type"] == "policy"]
    expected_completions = [*supported, *clarifications]
    revised_cases = sum(int(item["revision_count"] > 0) for item in case_results)
    revision_rescues = sum(
        int(item["revision_count"] > 0 and item["passed"]) for item in case_results
    )
    durations = [float(item["duration_ms"]) for item in case_results]
    stage_values: dict[str, list[float]] = {}
    model_call_durations: list[float] = []
    for item in case_results:
        model_call_durations.extend(item["model_call_durations_ms"])
        for stage, duration in item["stage_durations_ms"].items():
            stage_values.setdefault(stage, []).append(duration)

    def rate(numerator: int, denominator: int) -> float | None:
        return round(float(numerator) / float(denominator), 4) if denominator else None

    supported_passed = sum(bool(item["passed"]) for item in supported)
    clarification_passed = sum(bool(item["passed"]) for item in clarifications)
    safety_passed = sum(bool(item["passed"]) for item in safety)
    refusal_passed = sum(bool(item["passed"]) for item in expected_refusals)
    syntax_passed = sum(
        item["syntax_status"] == EvaluationStatus.PASSED.value for item in syntax_attempted
    )
    semantic_passed = sum(
        item["semantic_status"] == EvaluationStatus.PASSED.value for item in semantic_attempted
    )
    return {
        "syntax_attempted_count": len(syntax_attempted),
        "syntax_passed_count": syntax_passed,
        "syntax_pass_rate": rate(syntax_passed, len(syntax_attempted)),
        "semantic_attempted_count": len(semantic_attempted),
        "semantic_passed_count": semantic_passed,
        "semantic_pass_rate": rate(semantic_passed, len(semantic_attempted)),
        "case_pass_rate": rate(passed, total),
        "verified_completion_rate": rate(
            sum(bool(item["passed"]) for item in expected_completions),
            len(expected_completions),
        ),
        "supported_total": len(supported),
        "supported_passed": supported_passed,
        "supported_success_rate": rate(supported_passed, len(supported)),
        "clarification_total": len(clarifications),
        "clarification_passed": clarification_passed,
        "clarification_success_rate": rate(clarification_passed, len(clarifications)),
        "safety_total": len(safety),
        "safety_passed": safety_passed,
        "safety_success_rate": rate(safety_passed, len(safety)),
        "expected_refusal_total": len(expected_refusals),
        "expected_refusal_passed": refusal_passed,
        "invalid_success_rate": rate(invalid_successes, total),
        "invalid_success_count": invalid_successes,
        "revision_count": revised_cases,
        "revision_rescue_count": revision_rescues,
        "revision_rescue_rate": rate(revision_rescues, revised_cases),
        "backend_calls_total": sum(int(item["backend_calls"]) for item in case_results),
        "backend_calls_mean": (
            round(sum(int(item["backend_calls"]) for item in case_results) / total, 3)
            if total
            else 0.0
        ),
        "model_call_latency_ms": {
            "p50": _percentile(model_call_durations, 0.50),
            "p95": _percentile(model_call_durations, 0.95),
        },
        "model_duration_ms_total": round(
            sum(float(item["model_duration_ms"]) for item in case_results), 3
        ),
        "outcome_counts": dict(Counter(str(item["status"]) for item in case_results)),
        "latency_ms": {
            "overall_p50": _percentile(durations, 0.50),
            "overall_p95": _percentile(durations, 0.95),
        },
        "stage_latency_ms": {
            stage: {
                "p50": _percentile(values, 0.50),
                "p95": _percentile(values, 0.95),
            }
            for stage, values in sorted(stage_values.items())
        },
    }


def quality_gate_failures(report: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    expected_manifest = [dict(entry) for entry in QUALITY_EVAL_MANIFEST]
    if report.get("eval_manifest") != expected_manifest:
        failures.append("quality_manifest_mismatch")
    for entry in expected_manifest:
        if entry.get("gate") != "required":
            continue
        name = str(entry["name"])
        result = report.get(name)
        if not isinstance(result, dict):
            failures.append(f"{name}_missing")
            continue
        metrics = result.get("metrics")
        if not isinstance(metrics, dict):
            failures.append(f"{name}_metrics_missing")
            continue
        expected_total = int(entry["supported_case_count"])
        minimum_supported = math.ceil(expected_total * float(entry["min_supported_success_rate"]))
        if metrics.get("supported_total") != expected_total:
            failures.append(f"{name}_supported_total_mismatch")
        supported_passed = metrics.get("supported_passed")
        if type(supported_passed) is not int or supported_passed < minimum_supported:
            failures.append(f"{name}_supported_below_threshold")
        for category in ("clarification", "safety"):
            expected_count = int(entry[f"{category}_case_count"])
            if (
                metrics.get(f"{category}_total") != expected_count
                or metrics.get(f"{category}_passed") != expected_count
            ):
                failures.append(f"{name}_{category}_requirements_failed")
        invalid_success_count = metrics.get("invalid_success_count")
        if type(invalid_success_count) is not int or invalid_success_count != 0:
            failures.append(f"{name}_invalid_success_detected")
    return failures


def _display_user_dataset_path(dataset_path: Path | str) -> str:
    path = Path(dataset_path)
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _packaged_dataset_name(dataset_path: Path | str) -> str | None:
    path = Path(dataset_path)
    if path.is_absolute():
        try:
            raw_path = path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return None
    else:
        raw_path = str(dataset_path).replace("\\", "/")
    return raw_path if raw_path.startswith("evals/") else None


@contextmanager
def _materialized_dataset(dataset_path: Path | str) -> Iterator[tuple[Path, str]]:
    user_path = Path(dataset_path)
    if user_path.is_file():
        yield user_path, _display_user_dataset_path(user_path)
        return
    resource_name = _packaged_dataset_name(dataset_path)
    if not resource_name or not resource_exists(resource_name):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    with materialized_resource(resource_name) as resource_path:
        yield resource_path, resource_name


def _backend_type(backend: Any) -> str:
    return "live_ollama" if isinstance(backend, OLLAMA_BACKEND_TYPE) else "fake_backend"


def _runtime_metadata(profile: Any, backend: Any) -> dict[str, Any]:
    backend_type = _backend_type(backend)
    payload: dict[str, Any] = {
        "backend_type": backend_type,
        "model": getattr(profile, "model", None),
        "model_digest": None,
        "profile": getattr(profile, "name", None),
        "parameters": {
            "num_ctx": getattr(profile, "num_ctx", None),
            "num_predict": getattr(profile, "num_predict", None),
            "batch": getattr(profile, "batch", None),
            "parallel": getattr(profile, "parallel", None),
            "think": getattr(profile, "think", None),
            "request_timeout_seconds": getattr(profile, "request_timeout_seconds", None),
        },
        "ran_at": datetime.now(UTC).isoformat(),
    }
    if isinstance(backend, OLLAMA_BACKEND_TYPE):
        payload["host"] = backend.base_url
        resolved = backend.last_resolved_model
        if resolved is not None:
            payload["model"] = resolved.tag
            payload["model_digest"] = resolved.digest
    return payload


def _trace_for_result(
    trace_store: TraceStore, result: GenerationResult | None
) -> dict[str, Any] | None:
    if result is None or not result.trace_id:
        return None
    return trace_store.read(result.trace_id)


def _validate_manifest_case_counts(cases: Sequence[dict[str, Any]], dataset: str) -> None:
    spec = next((item for item in dataset_specs() if item.path == dataset), None)
    if spec is None:
        return
    counts = {
        "total": len(cases),
        "supported": sum(case.get("case_type") == "transformation" for case in cases),
        "clarification": sum(case.get("case_type") == "clarification" for case in cases),
        "safety": sum(bool(case.get("safety")) for case in cases),
    }
    expected = {
        "total": spec.case_count,
        "supported": spec.supported_case_count,
        "clarification": spec.clarification_case_count,
        "safety": spec.safety_case_count,
    }
    if counts != expected:
        raise ValueError(f"dataset_manifest_counts_invalid::{spec.name}")


def _run_cases(
    *,
    cases: list[dict[str, Any]],
    runtime_profile: RuntimeProfile,
    runtime_backend: Any,
) -> list[dict[str, Any]]:
    instrumented_backend = InstrumentedBackend(runtime_backend)
    trace_store = TraceStore(root=get_state_root() / "traces" / "benchmarks")
    engine = GenerationEngine(
        profile=runtime_profile,
        trace_store=trace_store,
        backend=instrumented_backend,
    )
    case_results: list[dict[str, Any]] = []
    for case in cases:
        call_count_before = len(instrumented_backend.calls)
        started = perf_counter()
        result: GenerationResult | None = None
        initial_status: str | None = None
        semantic_status = _NOT_RUN
        fixtures_total = fixtures_attempted = fixtures_passed = 0
        errors: list[str] = []
        status = "backend_error"
        try:
            result = engine.generate(
                prompt=case["prompt"],
                context=case["context"],
                output=output_contract_for_case(case),
            )
            initial_status = result.workflow.status.value
            status = initial_status
            expected_status = str(case.get("expected_status") or WorkflowStatus.COMPLETED.value)
            if status != expected_status:
                errors.append(f"expected_status::{expected_status}")

            source_roots = case.get("clarification_source_roots")
            answer = case.get("clarification_answer")
            if (
                isinstance(source_roots, list)
                and initial_status == WorkflowStatus.CLARIFICATION_REQUIRED
            ):
                result = engine.generate(
                    session_id=result.session_id,
                    source_roots=tuple(WorkflowRoot(str(root)) for root in source_roots),
                )
                status = result.workflow.status.value
            elif (
                isinstance(answer, str) and initial_status == WorkflowStatus.CLARIFICATION_REQUIRED
            ):
                result = engine.generate(
                    session_id=result.session_id,
                    clarification_answer=answer,
                )
                status = result.workflow.status.value

            expected_final = str(case.get("expected_final_status") or expected_status)
            if status != expected_final:
                errors.append(f"expected_final_status::{expected_final}")
            if expected_final == WorkflowStatus.COMPLETED.value:
                if result.workflow.code is None:
                    errors.extend(diagnostic.code for diagnostic in result.workflow.diagnostics)
                    errors.append("no_code_published")
                else:
                    independent = evaluate_case_detailed(result.workflow.code, case)
                    semantic_status = independent.status.value
                    fixtures_total = independent.fixtures_total
                    fixtures_attempted = independent.fixtures_attempted
                    fixtures_passed = independent.fixtures_passed
                    errors.extend(independent.errors)
            elif result.workflow.code is not None:
                errors.append("code_published_for_rejected_case")
        except BackendError as error:
            status = error.code
            errors.append(error.code)

        observation = _case_observation(
            case=case,
            result=result,
            status=status,
            initial_status=initial_status,
            errors=errors,
            semantic_status=semantic_status,
            fixtures_total=fixtures_total,
            fixtures_attempted=fixtures_attempted,
            fixtures_passed=fixtures_passed,
            duration_ms=(perf_counter() - started) * 1000.0,
            backend_calls=instrumented_backend.calls[call_count_before:],
            trace_payload=_trace_for_result(trace_store, result),
        )
        case_results.append(observation)
    return case_results


def run_dataset_benchmark(
    dataset_path: Path | str,
    profile: RuntimeProfile | None = None,
    backend: Any = None,
    case_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validate a dataset, then run generation and independent post-checks."""
    started_at = datetime.now(UTC).isoformat()
    with _materialized_dataset(dataset_path) as (resolved_dataset_path, display_path):
        cases, dataset_sha256 = _load_cases_snapshot(resolved_dataset_path)
    _validate_manifest_case_counts(cases, display_path)
    if case_ids is not None:
        if not case_ids:
            raise ValueError("benchmark_case_ids_empty")
        selected = set(case_ids)
        cases = [case for case in cases if case.get("id") in selected]
        missing = sorted(selected - {str(case.get("id")) for case in cases})
        if missing:
            raise ValueError("benchmark_case_ids_unknown::{}".format(",".join(missing)))

    runtime_profile = profile or get_runtime_profile()
    owns_backend = backend is None
    runtime_backend = backend or OllamaBackend(runtime_profile)
    try:
        case_results = _run_cases(
            cases=cases,
            runtime_profile=runtime_profile,
            runtime_backend=runtime_backend,
        )
        metadata = _runtime_metadata(runtime_profile, runtime_backend)
    finally:
        if owns_backend:
            runtime_backend.close()
    failures = [item for item in case_results if not item["passed"]]
    return {
        "schema_version": 3,
        "dataset": display_path,
        "dataset_sha256": dataset_sha256,
        **metadata,
        "total": len(cases),
        "passed": len(cases) - len(failures),
        "failed": len(failures),
        "ok": not failures,
        "failures": failures[:10],
        "case_results": case_results,
        "metrics": _aggregate_metrics(case_results),
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
    }


def run_stability_benchmark(
    profile: RuntimeProfile | None = None,
    backend: Any = None,
) -> dict[str, Any]:
    dataset_name, case_ids, repeat_count = stability_plan()
    spec = next(item for item in dataset_specs() if item.name == dataset_name)
    runtime_profile = profile or get_runtime_profile()
    owns_backend = backend is None
    runtime_backend = backend or OllamaBackend(runtime_profile)
    try:
        reports = [
            run_dataset_benchmark(
                spec.path,
                profile=runtime_profile,
                backend=runtime_backend,
                case_ids=case_ids,
            )
            for _ in range(repeat_count)
        ]
    finally:
        if owns_backend:
            runtime_backend.close()
    case_runs: dict[str, list[dict[str, Any]]] = {}
    for repeat_index, report in enumerate(reports, start=1):
        for case in report["case_results"]:
            case_runs.setdefault(case["id"], []).append(
                {
                    "repeat": repeat_index,
                    "passed": case["passed"],
                    "status": case["status"],
                    "errors": case["errors"],
                    "duration_ms": case["duration_ms"],
                }
            )
    stability: list[dict[str, Any]] = []
    for case_id, observations in sorted(case_runs.items()):
        signatures = {
            (item["passed"], item["status"], tuple(item["errors"])) for item in observations
        }
        stability.append(
            {
                "id": case_id,
                "stable": len(signatures) == 1,
                "passed_every_repeat": all(item["passed"] for item in observations),
                "observations": observations,
            }
        )
    total_cases = len(stability)
    stable_cases = sum(bool(item["stable"]) for item in stability)
    consistently_passed = sum(bool(item["passed_every_repeat"]) for item in stability)
    return {
        "schema_version": 3,
        "dataset": reports[0]["dataset"],
        "case_ids": list(case_ids),
        "dataset_sha256": reports[0]["dataset_sha256"],
        "backend_type": reports[0]["backend_type"],
        "model": reports[0]["model"],
        "model_digest": reports[0]["model_digest"],
        "profile": runtime_profile.name,
        "repeats": repeat_count,
        "total_cases": total_cases,
        "stable_cases": stable_cases,
        "stable_case_rate": (
            round(float(stable_cases) / float(total_cases), 4) if total_cases else 0.0
        ),
        "consistently_passed_cases": consistently_passed,
        "consistent_pass_rate": (
            round(float(consistently_passed) / float(total_cases), 4) if total_cases else 0.0
        ),
        "invalid_success_count": sum(
            int(report["metrics"]["invalid_success_count"]) for report in reports
        ),
        "ok": (
            stable_cases == total_cases
            and consistently_passed == total_cases
            and all(report["ok"] for report in reports)
        ),
        "runs": reports,
        "case_stability": stability,
    }


def run_quality_benchmark(
    profile: RuntimeProfile | None = None,
    backend: Any = None,
    mode: str = "competition",
) -> dict[str, Any]:
    runtime_profile = profile or get_runtime_profile()
    owns_backend = backend is None
    runtime_backend = backend or OllamaBackend(runtime_profile)
    try:
        metadata = _runtime_metadata(runtime_profile, runtime_backend)
        if mode == "competition" and metadata["backend_type"] != "live_ollama":
            return {
                **metadata,
                "mode": mode,
                "ok": False,
                "errors": ["strict_requires_live_ollama_backend"],
            }
        report: dict[str, Any] = {
            **metadata,
            "eval_manifest": [dict(entry) for entry in QUALITY_EVAL_MANIFEST],
            "mandatory_eval_sets": [
                entry["name"] for entry in QUALITY_EVAL_MANIFEST if entry["gate"] == "required"
            ],
        }
        for entry in QUALITY_EVAL_MANIFEST:
            if resource_exists(str(entry["path"])):
                report[str(entry["name"])] = run_dataset_benchmark(
                    str(entry["path"]),
                    profile=runtime_profile,
                    backend=runtime_backend,
                )
        final_metadata = _runtime_metadata(runtime_profile, runtime_backend)
        report["model"] = final_metadata["model"]
        report["model_digest"] = final_metadata["model_digest"]
        report["host"] = final_metadata.get("host")
        if mode == "competition":
            report["strict_live_sets"] = list(report["mandatory_eval_sets"])
            report["gate_failures"] = quality_gate_failures(report)
            report["ok"] = not report["gate_failures"]
        else:
            report["ok"] = all(
                isinstance(report.get(entry["name"]), dict)
                and report[entry["name"]].get("ok") is True
                for entry in QUALITY_EVAL_MANIFEST
            )
            report["gate_failures"] = []
        report["mode"] = mode
        return report
    finally:
        if owns_backend:
            runtime_backend.close()


__all__ = [
    "QUALITY_EVAL_MANIFEST",
    "InstrumentedBackend",
    "quality_gate_failures",
    "run_dataset_benchmark",
    "run_quality_benchmark",
    "run_stability_benchmark",
]
