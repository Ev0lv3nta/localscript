from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import PROJECT_ROOT, get_runtime_profile
from app.core.public_eval import evaluate_case, load_cases
from app.core.resources import materialized_resource, resource_exists
from app.core.state import get_state_root
from app.core.traces import TraceStore
from app.generation.engine import GenerationEngine
from app.generation.ollama import OllamaBackend


QUALITY_EVAL_MANIFEST = (
    {"name": "public_gold", "runner": "standard", "role": "mandatory"},
    {"name": "stress_eval", "runner": "standard", "role": "diagnostic"},
    {"name": "showcase_eval", "runner": "standard", "role": "diagnostic"},
    {"name": "model_backed_eval", "runner": "standard", "role": "mandatory"},
    {"name": "composition_eval", "runner": "standard", "role": "mandatory"},
    {"name": "regression_eval", "runner": "standard", "role": "mandatory"},
    {"name": "multilingual_eval", "runner": "standard", "role": "mandatory"},
    {"name": "ambiguity_eval", "runner": "rich", "role": "mandatory"},
    {"name": "clarification_eval", "runner": "rich", "role": "mandatory"},
    {"name": "adversarial_eval", "runner": "standard", "role": "mandatory"},
    {"name": "large_context_eval", "runner": "standard", "role": "mandatory"},
)


def quality_gate_failures(report):
    manifest = report.get("eval_manifest")
    if not manifest:
        return ["quality_manifest_missing"]

    failures = []
    for entry in manifest:
        if entry.get("role") != "mandatory":
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
    if raw_path.startswith("datasets/"):
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
    with _materialized_dataset(dataset_path) as (resolved_dataset_path, display_path):
        cases = load_cases(resolved_dataset_path)
        engine = GenerationEngine(
            profile=runtime_profile,
            trace_store=TraceStore(root=get_state_root() / "traces" / "benchmarks"),
            backend=runtime_backend,
        )

        failures = []
        for case in cases:
            result = engine.generate(prompt=case["prompt"], context=case.get("context"))
            errors = result.verification_errors + evaluate_case(result.code, case)
            expected_strategy = case.get("expected_strategy")
            if expected_strategy and result.strategy != expected_strategy:
                errors.append("expected_strategy::{0}".format(expected_strategy))
            forbidden_strategy = case.get("forbidden_strategy")
            if forbidden_strategy and result.strategy == forbidden_strategy:
                errors.append("forbidden_strategy::{0}".format(forbidden_strategy))
            if errors:
                failures.append(
                    {
                        "id": case["id"],
                        "family": case["family"],
                        "case_type": case.get("case_type", "unknown"),
                        "strategy": result.strategy,
                        "errors": errors,
                    }
                )

    return {
        "dataset": display_path,
        "backend_type": _backend_type(runtime_backend),
        "total": len(cases),
        "passed": len(cases) - len(failures),
        "failed": len(failures),
        "ok": not failures,
        "failures": failures[:10],
    }


def run_rich_dataset_benchmark(dataset_path, profile=None, backend=None):
    runtime_profile = profile or get_runtime_profile()
    runtime_backend = backend or OllamaBackend(runtime_profile)
    with _materialized_dataset(dataset_path) as (resolved_dataset_path, display_path):
        cases = load_cases(resolved_dataset_path)
        engine = GenerationEngine(
            profile=runtime_profile,
            trace_store=TraceStore(root=get_state_root() / "traces" / "benchmarks"),
            backend=runtime_backend,
        )

        failures = []
        for case in cases:
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

            if errors:
                failures.append(
                    {
                        "id": case["id"],
                        "strategy": result.strategy,
                        "status": result.status,
                        "final_status": final_result.status,
                        "errors": errors,
                    }
                )

    return {
        "dataset": display_path,
        "backend_type": _backend_type(runtime_backend),
        "total": len(cases),
        "passed": len(cases) - len(failures),
        "failed": len(failures),
        "ok": not failures,
        "failures": failures[:10],
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
            {"name": entry["name"], "role": entry["role"]}
            for entry in QUALITY_EVAL_MANIFEST
        ],
        "mandatory_eval_sets": [
            entry["name"]
            for entry in QUALITY_EVAL_MANIFEST
            if entry["role"] == "mandatory"
        ],
        "diagnostic_eval_sets": [
            entry["name"]
            for entry in QUALITY_EVAL_MANIFEST
            if entry["role"] == "diagnostic"
        ],
    }
    for entry in QUALITY_EVAL_MANIFEST:
        name = entry["name"]
        dataset_path = "datasets/{0}.jsonl".format(name)
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
            result.get("ok") is True
            for name, result in report.items()
            if name.endswith("_eval") or name == "public_gold"
        )
        report["gate_failures"] = []
    adversarial_report = report.get("adversarial_eval")
    if adversarial_report is not None:
        report["adversarial_ok"] = adversarial_report.get("ok") is True
    report["mode"] = mode
    return report
