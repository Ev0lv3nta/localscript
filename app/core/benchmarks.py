from contextlib import contextmanager
from collections import Counter
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from time import perf_counter

from app.core.config import PROJECT_ROOT, get_runtime_profile
from app.core.public_eval import evaluate_case, load_cases
from app.core.resources import materialized_resource, resource_exists
from app.core.state import get_state_root
from app.core.traces import TraceStore
from app.evaluation.manifest import dataset_specs
from app.generation.engine import GenerationEngine
from app.generation.ollama import OllamaBackend


QUALITY_EVAL_MANIFEST = tuple(spec.evidence_dict() for spec in dataset_specs())


class InstrumentedBackend:
    def __init__(self, backend):
        self.backend = backend
        self.calls = []
        self.evidence_backend_type = _backend_type(backend)

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def complete(self, prompt, response_format=None, model=None):
        started = perf_counter()
        method = getattr(self.backend, "complete", None)
        try:
            if callable(method):
                return method(
                    prompt,
                    response_format=response_format,
                    model=model,
                )
            return self.backend.generate(prompt)
        finally:
            self.calls.append(
                {
                    "duration_ms": round((perf_counter() - started) * 1000.0, 3),
                    "model": model,
                }
            )

    def generate(self, prompt, context=None):
        started = perf_counter()
        try:
            return self.backend.generate(prompt, context=context)
        finally:
            self.calls.append(
                {
                    "duration_ms": round((perf_counter() - started) * 1000.0, 3),
                    "model": None,
                }
            )


def _sha256_path(dataset_path):
    digest = hashlib.sha256()
    with Path(dataset_path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values, percentile):
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


def _deduplicate(items):
    return list(dict.fromkeys(item for item in items if item))


def _stage_durations(trace_payload):
    durations = {}
    for event in (trace_payload or {}).get("stage_events", []):
        stage = event.get("stage") if isinstance(event, dict) else None
        duration = event.get("stage_duration_ms") if isinstance(event, dict) else None
        if stage and isinstance(duration, (int, float)):
            durations[stage] = round(float(duration), 3)
    return durations


def _case_observation(case, result, errors, duration_ms, backend_calls, trace_payload):
    passed = not errors
    return {
        "id": case["id"],
        "family": case.get("family"),
        "case_type": case.get("case_type", "unknown"),
        "passed": passed,
        "status": result.status,
        "strategy": result.strategy,
        "repair_rounds": result.repair_rounds,
        "degraded_mode": result.degraded_mode,
        "duration_ms": round(duration_ms, 3),
        "backend_calls": backend_calls,
        "stage_durations_ms": _stage_durations(trace_payload),
        "errors": _deduplicate(errors),
    }


def _aggregate_metrics(case_results):
    total = len(case_results)
    passed = sum(item["passed"] for item in case_results)
    syntax_failures = sum(
        any("syntax" in error or "lua_load_error" in error for error in item["errors"])
        for item in case_results
    )
    semantic_failures = sum(
        any("semantic" in error for error in item["errors"])
        for item in case_results
    )
    invalid_successes = sum(
        item["status"] == "completed" and not item["passed"]
        for item in case_results
    )
    repair_attempts = sum(item["repair_rounds"] > 0 for item in case_results)
    repair_rescues = sum(
        item["repair_rounds"] > 0 and item["passed"] for item in case_results
    )
    durations = [item["duration_ms"] for item in case_results]
    warm_durations = durations[1:]
    stage_values = {}
    for item in case_results:
        for stage, duration in item["stage_durations_ms"].items():
            stage_values.setdefault(stage, []).append(duration)
    rate = lambda numerator: round(float(numerator) / float(total), 4) if total else 0.0
    return {
        "syntax_pass_rate": rate(total - syntax_failures),
        "semantic_pass_rate": rate(total - semantic_failures),
        "verified_completion_rate": rate(passed),
        "invalid_success_rate": rate(invalid_successes),
        "invalid_success_count": invalid_successes,
        "repair_attempt_count": repair_attempts,
        "repair_rescue_count": repair_rescues,
        "repair_rescue_rate": (
            round(float(repair_rescues) / float(repair_attempts), 4)
            if repair_attempts
            else None
        ),
        "degraded_count": sum(item["degraded_mode"] for item in case_results),
        "backend_calls_total": sum(item["backend_calls"] for item in case_results),
        "backend_calls_mean": (
            round(
                float(sum(item["backend_calls"] for item in case_results))
                / float(total),
                3,
            )
            if total
            else 0.0
        ),
        "outcome_counts": dict(Counter(item["status"] for item in case_results)),
        "strategy_counts": dict(Counter(item["strategy"] for item in case_results)),
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


def quality_gate_failures(report):
    manifest = report.get("eval_manifest")
    failures = []
    expected_manifest = [
        {
            "name": entry["name"],
            "path": entry["path"],
            "corpus": entry["corpus"],
            "gate": entry["gate"],
            "claim_scope": entry["claim_scope"],
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
        if not isinstance(result, dict) or result.get("ok") is not True:
            failures.append("{0}_failed".format(name))
    return failures


def _display_user_dataset_path(dataset_path):
    path = Path(dataset_path)
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _packaged_dataset_name(dataset_path):
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
def _materialized_dataset(dataset_path):
    user_path = Path(dataset_path)
    if user_path.is_file():
        yield user_path, _display_user_dataset_path(user_path)
        return

    resource_name = _packaged_dataset_name(dataset_path)
    if not resource_name or not resource_exists(resource_name):
        raise FileNotFoundError("Dataset not found: {0}".format(dataset_path))
    with materialized_resource(resource_name) as resource_path:
        yield resource_path, resource_name


def _backend_type(backend):
    explicit = getattr(backend, "evidence_backend_type", None)
    if explicit:
        return explicit
    return "live_ollama" if backend.__class__.__name__ == "OllamaBackend" else "fake_backend"


def _runtime_metadata(profile, backend):
    backend_type = _backend_type(backend)
    payload = {
        "backend_type": backend_type,
        "model": getattr(profile, "model", None),
        "profile": getattr(profile, "name", None),
        "ran_at": datetime.now(timezone.utc).isoformat(),
    }
    if backend_type == "live_ollama":
        payload["host"] = backend.base_url
    return payload


def run_dataset_benchmark(dataset_path, profile=None, backend=None):
    runtime_profile = profile or get_runtime_profile()
    runtime_backend = backend or OllamaBackend(runtime_profile)
    instrumented_backend = InstrumentedBackend(runtime_backend)
    started_at = datetime.now(timezone.utc).isoformat()
    with _materialized_dataset(dataset_path) as (resolved_dataset_path, display_path):
        cases = load_cases(resolved_dataset_path)
        dataset_sha256 = _sha256_path(resolved_dataset_path)
        trace_store = TraceStore(root=get_state_root() / "traces" / "benchmarks")
        engine = GenerationEngine(
            profile=runtime_profile,
            trace_store=trace_store,
            backend=instrumented_backend,
        )

        case_results = []
        for case in cases:
            call_count_before = len(instrumented_backend.calls)
            started = perf_counter()
            result = engine.generate(prompt=case["prompt"], context=case.get("context"))
            duration_ms = (perf_counter() - started) * 1000.0
            errors = result.verification_errors + evaluate_case(result.code, case)
            expected_strategy = case.get("expected_strategy")
            if expected_strategy and result.strategy != expected_strategy:
                errors.append("expected_strategy::{0}".format(expected_strategy))
            forbidden_strategy = case.get("forbidden_strategy")
            if forbidden_strategy and result.strategy == forbidden_strategy:
                errors.append("forbidden_strategy::{0}".format(forbidden_strategy))
            case_results.append(
                _case_observation(
                    case=case,
                    result=result,
                    errors=errors,
                    duration_ms=duration_ms,
                    backend_calls=len(instrumented_backend.calls) - call_count_before,
                    trace_payload=trace_store.read(result.trace_id),
                )
            )

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
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }


def run_rich_dataset_benchmark(dataset_path, profile=None, backend=None):
    runtime_profile = profile or get_runtime_profile()
    runtime_backend = backend or OllamaBackend(runtime_profile)
    instrumented_backend = InstrumentedBackend(runtime_backend)
    started_at = datetime.now(timezone.utc).isoformat()
    with _materialized_dataset(dataset_path) as (resolved_dataset_path, display_path):
        cases = load_cases(resolved_dataset_path)
        dataset_sha256 = _sha256_path(resolved_dataset_path)
        trace_store = TraceStore(root=get_state_root() / "traces" / "benchmarks")
        engine = GenerationEngine(
            profile=runtime_profile,
            trace_store=trace_store,
            backend=instrumented_backend,
        )

        case_results = []
        for case in cases:
            call_count_before = len(instrumented_backend.calls)
            started = perf_counter()
            result = engine.generate_rich(prompt=case.get("prompt"), context=case.get("context"))
            errors = []
            expected_status = case.get("expected_status")
            if expected_status and result.status != expected_status:
                errors.append("expected_status::{0}".format(expected_status))
            expected_question_contains = case.get("expected_question_contains")
            if expected_question_contains and expected_question_contains not in (result.question or ""):
                errors.append("expected_question_contains::{0}".format(expected_question_contains))

            final_result = result
            if case.get("clarification_answer"):
                final_result = engine.generate_rich(
                    session_id=result.session_id,
                    clarification_answer=case["clarification_answer"],
                )
                expected_final_status = case.get("expected_final_status")
                if expected_final_status and final_result.status != expected_final_status:
                    errors.append("expected_final_status::{0}".format(expected_final_status))
                if final_result.code:
                    errors.extend(evaluate_case(final_result.code, case))

            observation = _case_observation(
                case=case,
                result=final_result,
                errors=errors,
                duration_ms=(perf_counter() - started) * 1000.0,
                backend_calls=len(instrumented_backend.calls) - call_count_before,
                trace_payload=trace_store.read(final_result.trace_id),
            )
            observation["initial_status"] = result.status
            observation["initial_strategy"] = result.strategy
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
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }


def run_repeated_dataset_benchmark(
    dataset_path,
    repeats=3,
    profile=None,
    backend=None,
):
    repeat_count = int(repeats)
    if repeat_count < 2:
        raise ValueError("repeats must be at least 2")
    runtime_profile = profile or get_runtime_profile()
    runtime_backend = backend or OllamaBackend(runtime_profile)
    reports = [
        run_dataset_benchmark(
            dataset_path,
            profile=runtime_profile,
            backend=runtime_backend,
        )
        for _ in range(repeat_count)
    ]
    case_runs = {}
    for repeat_index, report in enumerate(reports, start=1):
        for case in report["case_results"]:
            case_runs.setdefault(case["id"], []).append(
                {
                    "repeat": repeat_index,
                    "passed": case["passed"],
                    "status": case["status"],
                    "strategy": case["strategy"],
                    "errors": case["errors"],
                    "duration_ms": case["duration_ms"],
                }
            )

    stability = []
    for case_id, observations in sorted(case_runs.items()):
        signatures = {
            (
                item["passed"],
                item["status"],
                item["strategy"],
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
        "schema_version": 1,
        "dataset": reports[0]["dataset"],
        "dataset_sha256": reports[0]["dataset_sha256"],
        "backend_type": reports[0]["backend_type"],
        "model": runtime_profile.model,
        "profile": runtime_profile.name,
        "repeats": repeat_count,
        "total_cases": total_cases,
        "stable_cases": stable_cases,
        "stable_case_rate": (
            round(float(stable_cases) / float(total_cases), 4)
            if total_cases
            else 0.0
        ),
        "consistently_passed_cases": consistently_passed,
        "consistent_pass_rate": (
            round(float(consistently_passed) / float(total_cases), 4)
            if total_cases
            else 0.0
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


def run_quality_benchmark(profile=None, backend=None, mode="competition"):
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
    report = {
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
            entry["name"]
            for entry in QUALITY_EVAL_MANIFEST
            if entry["gate"] == "required"
        ],
        "diagnostic_eval_sets": [
            entry["name"]
            for entry in QUALITY_EVAL_MANIFEST
            if entry["gate"] == "diagnostic"
        ],
    }
    for entry in QUALITY_EVAL_MANIFEST:
        name = entry["name"]
        dataset_path = entry["path"]
        if not resource_exists(dataset_path):
            continue
        runner = (
            run_rich_dataset_benchmark
            if entry["runner"] == "rich"
            else run_dataset_benchmark
        )
        result = runner(
            dataset_path,
            profile=runtime_profile,
            backend=runtime_backend,
        )
        if name == "adversarial_eval" and mode != "competition":
            result = dict(result)
            result["ok"] = (
                result["total"] == 0
                or (float(result["passed"]) / float(result["total"])) >= 0.75
            )
        report[name] = result

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
    adversarial_report = report.get("adversarial_eval")
    if adversarial_report is not None:
        report["adversarial_ok"] = adversarial_report.get("ok") is True
    report["mode"] = mode
    return report
