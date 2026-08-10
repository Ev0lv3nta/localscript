from datetime import datetime, timezone
from time import perf_counter

from app.core.benchmarks import (
    InstrumentedBackend,
    _backend_type,
    _load_cases_snapshot,
    _materialized_dataset,
    _percentile,
)
from app.core.config import get_runtime_profile
from app.core.kb import build_rule_lines, select_examples
from app.core.public_eval import evaluate_case
from app.core.state import get_state_root
from app.core.traces import TraceStore
from app.core.verifier import verify_code
from app.families import get_family_definition
from app.generation.context_reducer import ContextReducer
from app.generation.engine import GenerationEngine
from app.generation.extractor import TaskExtractor
from app.generation.formatter import OutputFormatter
from app.generation.model_chain import SameModelChain
from app.generation.ollama import OllamaBackend
from app.generation.prompts import build_writer_prompt
from app.generation.task_resolver import TaskResolver
from app.repair.loop import RepairLoop
from app.validation.validators import ValidationPipeline

ABLATION_PROFILES = (
    "one_shot_writer",
    "planner_writer",
    "planner_writer_validation",
    "deterministic_repair",
    "full_pipeline",
)


def _unique(items):
    return list(dict.fromkeys(item for item in items if item))


def _call_metrics(calls):
    durations = [
        round(float(call["duration_ms"]), 3)
        for call in calls
        if isinstance(call.get("duration_ms"), (int, float))
    ]
    return {
        "model_calls": len(calls),
        "model_call_durations_ms": durations,
        "model_duration_ms": round(sum(durations), 3),
    }


def _external_errors(code, case):
    return _unique(verify_code(code) + evaluate_case(code, case))


def _observation(
    *,
    case,
    code,
    accepted,
    errors,
    validation_errors,
    repair_rounds,
    calls,
    duration_ms,
    candidate_source,
):
    verified = not errors
    payload = {
        "id": case["id"],
        "family": case.get("family"),
        "accepted": bool(accepted),
        "verified": verified,
        "accepted_verified": bool(accepted) and verified,
        "invalid_success": bool(accepted) and not verified,
        "errors": _unique(errors),
        "validation_errors": _unique(validation_errors),
        "repair_rounds": int(repair_rounds),
        "duration_ms": round(float(duration_ms), 3),
        "candidate_source": candidate_source,
    }
    payload.update(_call_metrics(calls))
    return payload


def _aggregate(observations):
    total = len(observations)
    verified = sum(item["verified"] for item in observations)
    accepted_verified = sum(item["accepted_verified"] for item in observations)
    invalid_successes = sum(item["invalid_success"] for item in observations)
    model_calls = sum(item["model_calls"] for item in observations)
    call_durations = [
        duration
        for item in observations
        for duration in item["model_call_durations_ms"]
    ]
    def rate(value):
        return round(float(value) / float(total), 4) if total else 0.0

    return {
        "total": total,
        "verified_cases": verified,
        "verified_rate": rate(verified),
        "accepted_verified_cases": accepted_verified,
        "accepted_verified_rate": rate(accepted_verified),
        "invalid_success_count": invalid_successes,
        "invalid_success_rate": rate(invalid_successes),
        "repaired_cases": sum(item["repair_rounds"] > 0 for item in observations),
        "model_calls_total": model_calls,
        "model_calls_mean": (
            round(float(model_calls) / float(total), 3) if total else 0.0
        ),
        "model_call_latency_ms": {
            "p50": _percentile(call_durations, 0.50),
            "p95": _percentile(call_durations, 0.95),
        },
        "duration_ms": {
            "p50": _percentile(
                [item["duration_ms"] for item in observations],
                0.50,
            ),
            "p95": _percentile(
                [item["duration_ms"] for item in observations],
                0.95,
            ),
        },
    }


class AblationRunner:
    def __init__(self, profile, backend):
        self.profile = profile
        self.backend = InstrumentedBackend(backend)
        self.extractor = TaskExtractor()
        self.formatter = OutputFormatter()
        self.resolver = TaskResolver()
        self.context_reducer = ContextReducer()
        self.validation = ValidationPipeline()
        self.chain = SameModelChain(
            backend=self.backend,
            validation_pipeline=self.validation,
            formatter=self.formatter,
            task_resolver=self.resolver,
        )
        self.repair = RepairLoop(
            validation_pipeline=self.validation,
            formatter=self.formatter,
        )
        self.trace_store = TraceStore(
            root=get_state_root() / "traces" / "ablation"
        )
        self.engine = GenerationEngine(
            profile=self.profile,
            trace_store=self.trace_store,
            backend=self.backend,
        )

    def _calls_since(self, offset):
        return self.backend.calls[offset:]

    def _one_shot(self, case, extracted):
        task_spec = self.resolver.resolve(extracted)
        definition = get_family_definition(task_spec.family)
        return_shape = (
            definition.preferred_return_shape
            if definition and definition.preferred_return_shape
            else SameModelChain._default_return_shape(
                task_spec,
                case["prompt"],
                family=task_spec.family,
            )
        )
        planner_seed = {
            "family": task_spec.family,
            "root": task_spec.target_root,
            "source_paths": list(task_spec.prompt_paths),
            "return_shape": return_shape,
            "constraints": [],
            "assumptions": [],
            "clarification_needed": False,
            "clarification_question": "",
            "semantic_checks": [],
        }
        rules = build_rule_lines(task_spec)
        examples = select_examples(
            case["prompt"],
            family=task_spec.family,
            limit=1 if task_spec.family != "generic_lua" else 2,
        )
        prompt = build_writer_prompt(
            case["prompt"],
            self.context_reducer.reduce(case.get("context"), task_spec),
            task_spec,
            planner_seed,
            rules,
            examples,
        )
        raw = self.backend.complete(prompt, response_format=None)
        return (
            self.formatter.format(
                SameModelChain._clean_candidate(raw),
                task_spec.output_style,
            ),
            task_spec,
        )

    def run_case(self, case):
        observations = {}
        extracted = self.extractor.extract(
            prompt=case["prompt"],
            context=case.get("context"),
        )

        call_offset = len(self.backend.calls)
        started = perf_counter()
        one_shot_code, _ = self._one_shot(case, extracted)
        one_shot_errors = _external_errors(one_shot_code, case)
        observations["one_shot_writer"] = _observation(
            case=case,
            code=one_shot_code,
            accepted=True,
            errors=one_shot_errors,
            validation_errors=[],
            repair_rounds=0,
            calls=self._calls_since(call_offset),
            duration_ms=(perf_counter() - started) * 1000.0,
            candidate_source="independent_writer_call",
        )

        call_offset = len(self.backend.calls)
        started = perf_counter()
        chain_result = self.chain.run(
            prompt=case["prompt"],
            context=case.get("context"),
            task_spec=extracted,
            profile=self.profile,
            max_rounds=0,
            validate_candidate=False,
        )
        chain_duration_ms = (perf_counter() - started) * 1000.0
        chain_calls = self._calls_since(call_offset)
        chain_errors = _external_errors(chain_result.code, case)
        observations["planner_writer"] = _observation(
            case=case,
            code=chain_result.code,
            accepted=True,
            errors=chain_errors,
            validation_errors=[],
            repair_rounds=0,
            calls=chain_calls,
            duration_ms=chain_duration_ms,
            candidate_source="shared_planner_writer_candidate",
        )
        validation_started = perf_counter()
        validation_report = self.validation.run(
            code=chain_result.code,
            task_spec=chain_result.task_spec,
            profile=self.profile,
            source_context=case.get("context"),
            prompt=case["prompt"],
            planner_semantic_checks=chain_result.semantic_checks,
        )
        validation_duration_ms = (perf_counter() - validation_started) * 1000.0
        validation_errors = validation_report.error_codes()
        observations["planner_writer_validation"] = _observation(
            case=case,
            code=chain_result.code,
            accepted=not validation_report.has_errors,
            errors=chain_errors,
            validation_errors=validation_errors,
            repair_rounds=0,
            calls=chain_calls,
            duration_ms=chain_duration_ms + validation_duration_ms,
            candidate_source="shared_planner_writer_candidate",
        )

        started = perf_counter()
        repaired = self.repair.run(
            code=chain_result.code,
            task_spec=chain_result.task_spec,
            validation_report=validation_report,
            profile=self.profile,
            max_rounds=self.profile.max_repair_rounds,
            source_context=case.get("context"),
            prompt=case["prompt"],
            planner_semantic_checks=chain_result.semantic_checks,
        )
        repaired_errors = _external_errors(repaired.code, case)
        observations["deterministic_repair"] = _observation(
            case=case,
            code=repaired.code,
            accepted=not repaired.validation_report.has_errors,
            errors=repaired_errors,
            validation_errors=repaired.validation_report.error_codes(),
            repair_rounds=repaired.rounds,
            calls=chain_calls,
            duration_ms=(
                chain_duration_ms
                + validation_duration_ms
                + (perf_counter() - started) * 1000.0
            ),
            candidate_source="shared_candidate_after_deterministic_repair",
        )

        call_offset = len(self.backend.calls)
        started = perf_counter()
        full_result = self.engine.generate(
            prompt=case["prompt"],
            context=case.get("context"),
        )
        full_errors = _unique(
            full_result.verification_errors
            + evaluate_case(full_result.code, case)
        )
        observations["full_pipeline"] = _observation(
            case=case,
            code=full_result.code,
            accepted=full_result.status == "completed",
            errors=full_errors,
            validation_errors=full_result.verification_errors,
            repair_rounds=full_result.repair_rounds,
            calls=self._calls_since(call_offset),
            duration_ms=(perf_counter() - started) * 1000.0,
            candidate_source="independent_full_pipeline_run",
        )
        return observations


def run_ablation_benchmark(dataset_path, profile=None, backend=None):
    runtime_profile = profile or get_runtime_profile()
    runtime_backend = backend or OllamaBackend(runtime_profile)
    runner = AblationRunner(runtime_profile, runtime_backend)
    started_at = datetime.now(timezone.utc).isoformat()
    with _materialized_dataset(dataset_path) as (resolved_path, display_path):
        cases, dataset_sha256 = _load_cases_snapshot(resolved_path)
        case_matrix = [runner.run_case(case) for case in cases]

    profiles = {}
    previous = None
    baseline = None
    for profile_name in ABLATION_PROFILES:
        observations = [matrix[profile_name] for matrix in case_matrix]
        metrics = _aggregate(observations)
        if baseline is None:
            baseline = metrics
        metrics["uplift_vs_one_shot_verified_cases"] = (
            metrics["verified_cases"] - baseline["verified_cases"]
        )
        metrics["invalid_success_reduction_vs_one_shot"] = (
            baseline["invalid_success_count"] - metrics["invalid_success_count"]
        )
        metrics["uplift_vs_previous_verified_cases"] = (
            0
            if previous is None
            else metrics["verified_cases"] - previous["verified_cases"]
        )
        profiles[profile_name] = {
            "metrics": metrics,
            "cases": observations,
        }
        previous = metrics

    resolved_model = getattr(runtime_backend, "last_resolved_model", None)
    return {
        "schema_version": 1,
        "method": {
            "candidate_control": (
                "planner_writer, validation and deterministic repair share one "
                "planner/writer candidate per case; one-shot and full pipeline "
                "use independent calls"
            ),
            "profile_order": list(ABLATION_PROFILES),
        },
        "dataset": display_path,
        "dataset_sha256": dataset_sha256,
        "case_count": len(cases),
        "backend_type": _backend_type(runtime_backend),
        "model": runtime_profile.model,
        "model_digest": getattr(resolved_model, "digest", None),
        "profile": runtime_profile.name,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "profiles": profiles,
        "ok": (
            profiles["full_pipeline"]["metrics"]["verified_cases"] == len(cases)
            and profiles["full_pipeline"]["metrics"]["invalid_success_count"] == 0
        ),
    }
