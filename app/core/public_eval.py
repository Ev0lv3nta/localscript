import json
from pathlib import Path

from app.generation.extractor import TaskExtractor
from app.validation.oracles import UNSUPPORTED, build_expected_result, compare_expected_and_actual
from app.validation.runtime_executor import execute_output


def load_cases(path):
    dataset_path = Path(path)
    return [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalize_code(code):
    return "\n".join(
        line.strip()
        for line in (code or "").strip().splitlines()
        if line.strip()
    )


def _case_type(case):
    explicit_expected = case.get("expected_result")
    semantic_checks = case.get("semantic_checks") or []
    return case.get("case_type") or ("live_semantic" if explicit_expected is not None or semantic_checks else "unit_oracle")


def _is_live_semantic_case(case):
    return _case_type(case) in {
        "live_semantic",
        "model_backed",
        "composition",
        "regression",
        "multilingual",
        "adversarial",
        "large_context",
        "public_benchmark",
        "holdout",
    }


def evaluate_case(code, case):
    failures = []
    output_style = case.get("expected_output_style")
    semantic_supported = False

    explicit_expected = case.get("expected_result")
    semantic_checks = case.get("semantic_checks") or []
    if _is_live_semantic_case(case):
        if explicit_expected is None and not semantic_checks:
            failures.append("dataset_missing_expected_result")
            return failures
        semantic_supported = True
        execution = execute_output(code, case.get("context"), output_style or "lua_block")
        if execution.degraded:
            failures.append("semantic_degraded")
        elif not execution.ok:
            failures.append(execution.error_code or "semantic_runtime_error")
        elif explicit_expected is not None and not compare_expected_and_actual(explicit_expected, execution.value):
            failures.append("semantic_mismatch")
    else:
        extractor = TaskExtractor()
        task_spec = extractor.extract(prompt=case["prompt"], context=case.get("context"))
        expected = build_expected_result(task_spec, case.get("context"))
        if expected is not UNSUPPORTED:
            semantic_supported = True
            execution = execute_output(code, case.get("context"), output_style or task_spec.output_style)
            if execution.degraded:
                failures.append("semantic_degraded")
            elif not execution.ok:
                failures.append(execution.error_code or "semantic_runtime_error")
            elif not compare_expected_and_actual(expected, execution.value):
                failures.append("semantic_mismatch")

    if case.get("strict_code_match") and case.get("expected_code"):
        if normalize_code(code) != normalize_code(case["expected_code"]):
            failures.append("expected_code_mismatch")

    if output_style == "json_envelope":
        try:
            payload = json.loads(code)
        except json.JSONDecodeError:
            failures.append("json_envelope_invalid")
        else:
            for key in case.get("expected_json_keys", []):
                if key not in payload:
                    failures.append("missing_json_key::{0}".format(key))

    if not semantic_supported:
        for assertion in case.get("assertions", []):
            if assertion not in code:
                failures.append("missing_assertion::{0}".format(assertion))

    for pattern in case.get("forbidden_patterns", []):
        if pattern in code:
            failures.append("forbidden_pattern::{0}".format(pattern))

    return failures
