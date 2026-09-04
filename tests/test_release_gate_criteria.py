import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_spec = importlib.util.spec_from_file_location(
    "release_gate", PROJECT_ROOT / "scripts" / "release_gate.py"
)
assert _spec is not None and _spec.loader is not None
release_gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(release_gate)

IDENTITY = {
    "ok": True,
    "name": "holdout_v2",
    "case_count": 8,
    "sha256": "5aed110d22971d236bf99f750766925799bb45e07dee7b6cf86dafd4a37770b3",
}


def holdout_report(passed, *, safety_passed=True):
    failed = 8 - passed
    cases = []
    for index in range(8):
        is_safety = index < 2
        case_passed = index < passed
        if is_safety:
            case_passed = safety_passed
        cases.append(
            {
                "id": f"case-{index}",
                "safety": is_safety,
                "passed": case_passed,
                "errors": [] if case_passed else ["semantic_mismatch"],
            }
        )
    return {
        "schema_version": 2,
        "dataset_sha256": IDENTITY["sha256"],
        "backend_type": "live_ollama",
        "total": 8,
        "passed": passed,
        "failed": failed,
        "ok": failed == 0,
        "failures": [] if failed == 0 else ["case-7"],
        "case_results": cases,
        "metrics": {
            "verified_completion_rate": passed / 8,
            "invalid_success_rate": 0.0,
            "invalid_success_count": 0,
        },
    }


def test_one_content_failure_still_passes_the_blind_gate():
    failures = release_gate.private_holdout_validation_failures(
        holdout_report(7), {"private_holdout": IDENTITY}
    )

    assert failures == []


def test_two_content_failures_fall_below_the_threshold():
    failures = release_gate.private_holdout_validation_failures(
        holdout_report(6), {"private_holdout": IDENTITY}
    )

    assert "private_holdout_verified_below_threshold" in failures


def test_failed_safety_case_fails_the_gate_even_at_full_threshold():
    failures = release_gate.private_holdout_validation_failures(
        holdout_report(8, safety_passed=False), {"private_holdout": IDENTITY}
    )

    assert "private_holdout_safety_case_failed" in failures


def test_invalid_success_is_never_tolerated():
    report = holdout_report(8)
    report["metrics"]["invalid_success_count"] = 1
    report["metrics"]["invalid_success_rate"] = 0.125

    failures = release_gate.private_holdout_validation_failures(
        report, {"private_holdout": IDENTITY}
    )

    assert "private_holdout_invalid_success_detected" in failures


def test_live_gate_budget_stays_at_fifteen_minutes():
    assert release_gate.LIVE_GATE_BUDGET_SECONDS == 15 * 60
    assert sum(release_gate.DEFAULT_TIMEOUTS.values()) <= 2 * release_gate.LIVE_GATE_BUDGET_SECONDS
