from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

from app.core.config import PROJECT_ROOT, get_runtime_profile
from app.core.public_eval import evaluate_case, load_cases_bytes
from app.core.resources import materialized_resource, resource_exists
from app.core.state import get_state_root
from app.core.traces import TraceStore
from app.evaluation.holdout import adapt_blind_holdout_cases
from app.evaluation.manifest import dataset_specs, stability_plan
from app.generation.engine import GenerationEngine
from app.generation.ollama import OllamaBackend
from app.generation.results import GenerationResult

if TYPE_CHECKING:
    from app.core.config import RuntimeProfile

QUALITY_EVAL_MANIFEST = tuple(spec.evidence_dict() for spec in dataset_specs())


class InstrumentedBackend:
    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self.calls: list[dict[str, Any]] = []
        self.evidence_backend_type = _backend_type(backend)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.backend, name)

    def complete(
        self,
        prompt: str,
        response_format: object | None = None,
        model: str | None = None,
    ) -> str:
        started = perf_counter()
        method = getattr(self.backend, "complete", None)
        try:
            if callable(method):
                return str(method(prompt, response_format=response_format, model=model))
            return str(self.backend.generate(prompt))
        finally:
            self.calls.append(
                {
                    "duration_ms": round((perf_counter() - started) * 1000.0, 3),
                    "model": model,
                }
            )

    def generate(self, prompt: str, context: Any = None) -> str:
        started = perf_counter()
        try:
            return str(self.backend.generate(prompt, context=context))
        finally:
            self.calls.append(
                {
                    "duration_ms": round((perf_counter() - started) * 1000.0, 3),
                    "model": None,
                }
            )


def _load_cases_snapshot(dataset_path: Path | str) -> tuple[list[dict[str, Any]], str]:
    payload = Path(dataset_path).read_bytes()
    cases = adapt_blind_holdout_cases(load_cases_bytes(payload))
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
        if stage and isinstance(duration, (int, float)):
            durations[stage] = round(float(duration), 3)
    return durations


def _case_observation(
    case: dict[str, Any],
    result: GenerationResult,
    errors: Sequence[str],
    duration_ms: float,
    backend_calls: Sequence[dict[str, Any]],
    trace_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    passed = not errors
    model_call_durations = [
        round(float(call["duration_ms"]), 3)
        for call in backend_calls
        if isinstance(call.get("duration_ms"), (int, float))
    ]
    return {
        "id": case["id"],
        "case_type": case.get("case_type", "unknown"),
        "category": case.get("category"),
        "safety": bool(case.get("safety", False)),
        "passed": passed,
        "status": result.workflow.status.value,
        "revision_count": result.workflow.revision_count,
        "duration_ms": round(duration_ms, 3),
        "backend_calls": len(backend_calls),
        "model_call_durations_ms": model_call_durations,
        "model_duration_ms": round(sum(model_call_durations), 3),
        "stage_durations_ms": _stage_durations(trace_payload),
        "errors": _deduplicate(errors),
    }


def _aggregate_metrics(case_results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = len(case_results)
    passed = sum(item["passed"] for item in case_results)
    syntax_failures = sum(
        any("syntax" in error or "lua_load_error" in error for error in item["errors"])
        for item in case_results
    )
    semantic_failures = sum(
        any("semantic" in error for error in item["errors"]) for item in case_results
    )
    invalid_successes = sum(
        item["status"] == "completed" and not item["passed"] for item in case_results
    )
    revised_cases = sum(item["revision_count"] > 0 for item in case_results)
    revision_rescues = sum(item["revision_count"] > 0 and item["passed"] for item in case_results)
    durations = [item["duration_ms"] for item in case_results]
    warm_durations = durations[1:]
    stage_values: dict[str, list[float]] = {}
    model_call_durations: list[float] = []
    for item in case_results:
        model_call_durations.extend(item["model_call_durations_ms"])
        for stage, duration in item["stage_durations_ms"].items():
            stage_values.setdefault(stage, []).append(duration)

    def rate(numerator: float) -> float:
        return round(float(numerator) / float(total), 4) if total else 0.0

    return {
        "syntax_pass_rate": rate(total - syntax_failures),
        "semantic_pass_rate": rate(total - semantic_failures),
        "verified_completion_rate": rate(passed),
        "invalid_success_rate": rate(invalid_successes),
        "invalid_success_count": invalid_successes,
        "revision_count": revised_cases,
        "revision_rescue_count": revision_rescues,
        "revision_rescue_rate": (
            round(float(revision_rescues) / float(revised_cases), 4) if revised_cases else None
        ),
        "backend_calls_total": sum(item["backend_calls"] for item in case_results),
        "backend_calls_mean": (
            round(
                float(sum(item["backend_calls"] for item in case_results)) / float(total),
                3,
            )
            if total
            else 0.0
        ),
        "model_call_latency_ms": {
            "p50": _percentile(model_call_durations, 0.50),
            "p95": _percentile(model_call_durations, 0.95),
        },
        "model_duration_ms_total": round(
            sum(item["model_duration_ms"] for item in case_results),
            3,
        ),
        "outcome_counts": dict(Counter(item["status"] for item in case_results)),
        "latency_ms": {
            "cold_first": round(durations[0], 3) if durations else None,
            "warm_p50": _percentile(warm_durations, 0.50),
            "warm_p95": _percentile(warm_durations, 0.95),
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
    manifest = report.get("eval_manifest")
    failures: list[str] = []
    expected_manifest = [
        {
            "name": entry["name"],
            "path": entry["path"],
            "corpus": entry["corpus"],
            "gate": entry["gate"],
            "claim_scope": entry["claim_scope"],
            "min_verified": entry["min_verified"],
        }
        for entry in QUALITY_EVAL_MANIFEST
    ]
    if manifest != expected_manifest:
        failures.append("quality_manifest_mismatch")

    for entry in expected_manifest:
        if entry.get("gate") != "required":
            continue
        name = entry["name"]
        result = report.get(name)
        if not isinstance(result, dict):
            failures.append(f"{name}_missing")
            continue
        # Порог объявлен в манифесте, а не выведен из результата: планка должна меняться
        # через ревью, а не подстраиваться под то, что модель показала сегодня.
        passed = result.get("passed")
        if not isinstance(passed, int) or passed < entry["min_verified"]:
            failures.append(f"{name}_below_min_verified")
        metrics = result.get("metrics")
        invalid = metrics.get("invalid_success_count") if isinstance(metrics, dict) else None
        if invalid != 0:
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
    if raw_path.startswith("evals/"):
        return raw_path
    return None


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
    explicit = getattr(backend, "evidence_backend_type", None)
    if explicit:
        return str(explicit)
    return "live_ollama" if backend.__class__.__name__ == "OllamaBackend" else "fake_backend"


def _runtime_metadata(profile: Any, backend: Any) -> dict[str, Any]:
    backend_type = _backend_type(backend)
    payload: dict[str, Any] = {
        "backend_type": backend_type,
        "model": getattr(profile, "model", None),
        "profile": getattr(profile, "name", None),
        "ran_at": datetime.now(UTC).isoformat(),
    }
    if backend_type == "live_ollama":
        payload["host"] = backend.base_url
    return payload


def run_dataset_benchmark(
    dataset_path: Path | str,
    profile: RuntimeProfile | None = None,
    backend: Any = None,
    case_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run one dataset through the single generation entry point.

    A case may declare `clarification_answer`; then the first response must be a question and the
    answer is sent back into the same session before the oracle runs.
    """
    runtime_profile = profile or get_runtime_profile()
    runtime_backend = backend or OllamaBackend(runtime_profile)
    instrumented_backend = InstrumentedBackend(runtime_backend)
    started_at = datetime.now(UTC).isoformat()
    with _materialized_dataset(dataset_path) as (resolved_dataset_path, display_path):
        cases, dataset_sha256 = _load_cases_snapshot(resolved_dataset_path)
        if case_ids is not None:
            selected = set(case_ids)
            cases = [case for case in cases if case.get("id") in selected]
            missing = sorted(selected - {str(case.get("id")) for case in cases})
            if missing:
                raise ValueError("benchmark_case_ids_unknown::{}".format(",".join(missing)))
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
            result = engine.generate(prompt=case.get("prompt"), context=case.get("context"))
            errors: list[str] = []

            expected_status = case.get("expected_status")
            if expected_status and result.workflow.status.value != expected_status:
                errors.append(f"expected_status::{expected_status}")

            initial_status = result.workflow.status.value
            if case.get("clarification_answer"):
                result = engine.generate(
                    session_id=result.session_id,
                    clarification_answer=case["clarification_answer"],
                )
                expected_final_status = case.get("expected_final_status")
                if expected_final_status and result.workflow.status.value != expected_final_status:
                    errors.append(f"expected_final_status::{expected_final_status}")

            # Кейс, который обязан завершиться отказом, проверяется зеркально успешному:
            # диагностика для него — ожидаемый результат, а опубликованный код — провал.
            final_expected = str(
                case.get("expected_final_status") or case.get("expected_status") or "completed"
            )
            if final_expected == "completed":
                errors.extend(diagnostic.code for diagnostic in result.workflow.diagnostics)
                if result.workflow.code:
                    errors.extend(evaluate_case(result.workflow.code, case))
                else:
                    errors.append("no_code_published")
            elif result.workflow.code is not None:
                errors.append("code_published_for_rejected_case")

            observation = _case_observation(
                case=case,
                result=result,
                errors=errors,
                duration_ms=(perf_counter() - started) * 1000.0,
                backend_calls=instrumented_backend.calls[call_count_before:],
                trace_payload=trace_store.read(result.trace_id),
            )
            observation["initial_status"] = initial_status
            case_results.append(observation)

    failures = [item for item in case_results if not item["passed"]]

    return {
        "schema_version": 2,
        "dataset": display_path,
        "dataset_sha256": dataset_sha256,
        "backend_type": _backend_type(runtime_backend),
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
    """Repeat the few scenarios named by the manifest and compare their outcomes.

    Repeating the whole corpus multiplies GPU time without telling us more than three
    representative scenarios do, so the manifest fixes both the cases and the repeat count.
    """
    dataset_name, case_ids, repeat_count = stability_plan()
    spec = next(item for item in dataset_specs() if item.name == dataset_name)
    runtime_profile = profile or get_runtime_profile()
    runtime_backend = backend or OllamaBackend(runtime_profile)
    reports = [
        run_dataset_benchmark(
            spec.path,
            profile=runtime_profile,
            backend=runtime_backend,
            case_ids=case_ids,
        )
        for _ in range(repeat_count)
    ]
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
            (
                item["passed"],
                item["status"],
                tuple(item["errors"]),
            )
            for item in observations
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
    stable_cases = sum(item["stable"] for item in stability)
    consistently_passed = sum(item["passed_every_repeat"] for item in stability)
    return {
        "schema_version": 2,
        "dataset": reports[0]["dataset"],
        "case_ids": list(case_ids),
        "dataset_sha256": reports[0]["dataset_sha256"],
        "backend_type": reports[0]["backend_type"],
        "model": runtime_profile.model,
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
            report["metrics"]["invalid_success_count"] for report in reports
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
    runtime_backend = backend or OllamaBackend(runtime_profile)
    metadata = _runtime_metadata(runtime_profile, runtime_backend)
    if mode == "competition" and metadata["backend_type"] != "live_ollama":
        return {
            "profile": runtime_profile.name,
            "mode": mode,
            "backend_type": metadata["backend_type"],
            "model": metadata["model"],
            "host": metadata.get("host"),
            "ran_at": metadata["ran_at"],
            "ok": False,
            "errors": ["strict_requires_live_ollama_backend"],
        }
    report: dict[str, Any] = {
        "profile": runtime_profile.name,
        "backend_type": metadata["backend_type"],
        "model": metadata["model"],
        "host": metadata.get("host"),
        "ran_at": metadata["ran_at"],
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
        "mandatory_eval_sets": [
            entry["name"] for entry in QUALITY_EVAL_MANIFEST if entry["gate"] == "required"
        ],
    }
    for entry in QUALITY_EVAL_MANIFEST:
        name = entry["name"]
        dataset_path = entry["path"]
        if not resource_exists(dataset_path):
            continue
        report[name] = run_dataset_benchmark(
            dataset_path,
            profile=runtime_profile,
            backend=runtime_backend,
        )

    if mode == "competition":
        report["strict_live_sets"] = list(report["mandatory_eval_sets"])
        report["gate_failures"] = quality_gate_failures(report)
        report["ok"] = not report["gate_failures"]
    else:
        report["ok"] = all(
            isinstance(report.get(entry["name"]), dict) and report[entry["name"]].get("ok") is True
            for entry in QUALITY_EVAL_MANIFEST
        )
        report["gate_failures"] = []
    report["mode"] = mode
    return report
