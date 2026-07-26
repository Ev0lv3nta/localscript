from pathlib import Path
from datetime import datetime, timezone

from app.core.config import PROJECT_ROOT, get_runtime_profile
from app.core.public_eval import evaluate_case, load_cases
from app.core.traces import TraceStore
from app.generation.engine import GenerationEngine
from app.generation.ollama import OllamaBackend


def _display_dataset_path(dataset_path):
    path = Path(dataset_path)
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


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
    cases = load_cases(Path(dataset_path))
    engine = GenerationEngine(
        profile=runtime_profile,
        trace_store=TraceStore(root=PROJECT_ROOT / "traces" / "benchmarks"),
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
        "dataset": _display_dataset_path(dataset_path),
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
    cases = load_cases(Path(dataset_path))
    engine = GenerationEngine(
        profile=runtime_profile,
        trace_store=TraceStore(root=PROJECT_ROOT / "traces" / "benchmarks"),
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
        "dataset": _display_dataset_path(dataset_path),
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
    public_report = run_dataset_benchmark(
        PROJECT_ROOT / "datasets" / "public_gold.jsonl",
        profile=runtime_profile,
        backend=runtime_backend,
    )
    stress_report = run_dataset_benchmark(
        PROJECT_ROOT / "datasets" / "stress_eval.jsonl",
        profile=runtime_profile,
        backend=runtime_backend,
    )
    report = {
        "profile": runtime_profile.name,
        "backend_type": metadata["backend_type"],
        "model": metadata["model"],
        "host": metadata.get("host"),
        "ran_at": metadata["ran_at"],
        "public_gold": public_report,
        "stress_eval": stress_report,
        "ok": public_report["ok"] and stress_report["ok"],
    }
    showcase_path = PROJECT_ROOT / "datasets" / "showcase_eval.jsonl"
    if showcase_path.exists():
        showcase_report = run_dataset_benchmark(
            showcase_path,
            profile=runtime_profile,
            backend=runtime_backend,
        )
        report["showcase_eval"] = showcase_report
        if mode != "competition":
            report["ok"] = report["ok"] and showcase_report["ok"]
    model_backed_path = PROJECT_ROOT / "datasets" / "model_backed_eval.jsonl"
    if model_backed_path.exists():
        model_backed_report = run_dataset_benchmark(
            model_backed_path,
            profile=runtime_profile,
            backend=runtime_backend,
        )
        report["model_backed_eval"] = model_backed_report
        report["ok"] = report["ok"] and model_backed_report["ok"]
    composition_path = PROJECT_ROOT / "datasets" / "composition_eval.jsonl"
    if composition_path.exists():
        composition_report = run_dataset_benchmark(
            composition_path,
            profile=runtime_profile,
            backend=runtime_backend,
        )
        report["composition_eval"] = composition_report
        report["ok"] = report["ok"] and composition_report["ok"]
    regression_path = PROJECT_ROOT / "datasets" / "regression_eval.jsonl"
    if regression_path.exists():
        regression_report = run_dataset_benchmark(
            regression_path,
            profile=runtime_profile,
            backend=runtime_backend,
        )
        report["regression_eval"] = regression_report
        report["ok"] = report["ok"] and regression_report["ok"]
    multilingual_path = PROJECT_ROOT / "datasets" / "multilingual_eval.jsonl"
    if multilingual_path.exists():
        multilingual_report = run_dataset_benchmark(
            multilingual_path,
            profile=runtime_profile,
            backend=runtime_backend,
        )
        report["multilingual_eval"] = multilingual_report
        report["ok"] = report["ok"] and multilingual_report["ok"]
    ambiguity_path = PROJECT_ROOT / "datasets" / "ambiguity_eval.jsonl"
    if ambiguity_path.exists():
        ambiguity_report = run_rich_dataset_benchmark(
            ambiguity_path,
            profile=runtime_profile,
            backend=runtime_backend,
        )
        report["ambiguity_eval"] = ambiguity_report
        report["ok"] = report["ok"] and ambiguity_report["ok"]
    clarification_path = PROJECT_ROOT / "datasets" / "clarification_eval.jsonl"
    if clarification_path.exists():
        clarification_report = run_rich_dataset_benchmark(
            clarification_path,
            profile=runtime_profile,
            backend=runtime_backend,
        )
        report["clarification_eval"] = clarification_report
        report["ok"] = report["ok"] and clarification_report["ok"]
    adversarial_path = PROJECT_ROOT / "datasets" / "adversarial_eval.jsonl"
    if adversarial_path.exists():
        adversarial_report = run_dataset_benchmark(
            adversarial_path,
            profile=runtime_profile,
            backend=runtime_backend,
        )
        report["adversarial_eval"] = adversarial_report
        report["adversarial_ok"] = adversarial_report["failed"] == 0 if mode == "competition" else (
            adversarial_report["total"] == 0
            or (float(adversarial_report["passed"]) / float(adversarial_report["total"])) >= 0.75
        )
        report["ok"] = report["ok"] and report["adversarial_ok"]
    large_context_path = PROJECT_ROOT / "datasets" / "large_context_eval.jsonl"
    if large_context_path.exists():
        large_context_report = run_dataset_benchmark(
            large_context_path,
            profile=runtime_profile,
            backend=runtime_backend,
        )
        report["large_context_eval"] = large_context_report
        report["ok"] = report["ok"] and large_context_report["ok"]
    if mode == "competition":
        report["strict_live_sets"] = [
            "public_gold",
            "model_backed_eval",
            "composition_eval",
            "regression_eval",
            "multilingual_eval",
            "ambiguity_eval",
            "clarification_eval",
            "adversarial_eval",
            "large_context_eval",
        ]
        report["ok"] = (
            report["public_gold"]["ok"]
            and report.get("model_backed_eval", {}).get("ok") is True
            and report.get("composition_eval", {}).get("ok") is True
            and report.get("regression_eval", {}).get("ok") is True
            and report.get("multilingual_eval", {}).get("ok") is True
            and report.get("ambiguity_eval", {}).get("ok") is True
            and report.get("clarification_eval", {}).get("ok") is True
            and report.get("adversarial_ok") is True
            and report.get("large_context_eval", {}).get("ok") is True
        )
    report["mode"] = mode
    return report
