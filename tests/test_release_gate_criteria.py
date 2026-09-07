import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "release_gate", PROJECT_ROOT / "scripts/release_gate.py"
)
assert _spec is not None and _spec.loader is not None
release_gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(release_gate)


def test_gate_requires_live_quality_and_successful_stability():
    quality = {"backend_type": "live_ollama", "ok": True}
    stability = {"backend_type": "live_ollama", "ok": True}
    assert release_gate.report_failures(quality, stability) == []
    assert "live_backend_required" in release_gate.report_failures(
        {**quality, "backend_type": "mock"}, stability
    )
    assert "stability_failed" in release_gate.report_failures(quality, {**stability, "ok": False})
    assert "invalid_success" in release_gate.report_failures(
        {**quality, "ok": False, "gate_failures": ["invalid_success"]}, stability
    )


def test_expired_gate_does_not_start_another_process(monkeypatch):
    monkeypatch.setattr(
        release_gate.subprocess,
        "Popen",
        lambda *a, **k: pytest.fail("expired gate started a child"),
    )
    with pytest.raises(TimeoutError):
        release_gate.run_bounded([sys.executable, "-c", "pass"], time.monotonic() - 1)


def test_gate_kills_work_instead_of_only_reporting_lateness():
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        release_gate.run_bounded(
            [sys.executable, "-c", "import time; time.sleep(30)"], started + 0.1
        )
    assert time.monotonic() - started < 3


def test_preflight_failure_leaves_live_checks_not_run(monkeypatch, tmp_path):
    def unavailable(stage, _path, _deadline, _gpu_index=0):
        assert stage == "preflight"
        return 1, {"ok": False, "failure": "model_missing"}

    monkeypatch.setattr(release_gate, "run_worker", unavailable)
    path = tmp_path / "report.json"
    assert release_gate.main(["--output", str(path)]) == 1

    report = json.loads(path.read_text())
    assert not report["ok"]
    assert report["checks"]["preflight"] == "failed"
    assert report["checks"]["quality"] == "not_run"
    assert "model_missing" in report["failures"]


def test_required_ci_is_read_for_the_exact_sha_and_fails_closed(monkeypatch):
    sha = "a" * 40
    commands = []
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repository")

    def successful_capture(command, timeout=10):
        commands.append(command)
        return json.dumps(
            {
                "check_runs": [
                    {
                        "id": 41,
                        "name": "CI / required",
                        "head_sha": sha,
                        "status": "completed",
                        "conclusion": "success",
                        "app": {"slug": "github-actions"},
                    }
                ]
            }
        )

    monkeypatch.setattr(release_gate, "capture", successful_capture)

    evidence = release_gate.verify_required_ci(sha)

    assert evidence["sha"] == sha
    assert f"commits/{sha}/check-runs?per_page=100" in commands[0][-1]

    def failed_capture(_command, timeout=10):
        return json.dumps(
            {
                "check_runs": [
                    {
                        "id": 42,
                        "name": "CI / required",
                        "head_sha": sha,
                        "status": "completed",
                        "conclusion": "failure",
                        "app": {"slug": "github-actions"},
                    }
                ]
            }
        )

    monkeypatch.setattr(release_gate, "capture", failed_capture)
    with pytest.raises(ValueError, match="required_ci_not_successful"):
        release_gate.verify_required_ci(sha)

    def wrong_sha_capture(_command, timeout=10):
        return json.dumps(
            {
                "check_runs": [
                    {
                        "id": 43,
                        "name": "CI / required",
                        "head_sha": "b" * 40,
                        "status": "completed",
                        "conclusion": "success",
                        "app": {"slug": "github-actions"},
                    }
                ]
            }
        )

    monkeypatch.setattr(release_gate, "capture", wrong_sha_capture)
    with pytest.raises(ValueError, match="required_ci_missing"):
        release_gate.verify_required_ci(sha)


def test_release_report_is_recursively_sanitized(tmp_path):
    report_path = tmp_path / "release.json"
    release_gate.write_report(
        report_path,
        {
            "host": "http://private-host:11434",
            "quality": {
                "public_v2": {
                    "host": "http://nested-host:11434",
                    "raw_response": "private model output",
                    "case_results": [
                        {
                            "id": "case-1",
                            "context": {"token": "secret"},
                            "passed": True,
                        }
                    ],
                }
            },
        },
    )

    saved = json.loads(report_path.read_text(encoding="utf-8"))
    rendered = json.dumps(saved)
    assert saved["quality"]["public_v2"]["case_results"][0] == {
        "id": "case-1",
        "passed": True,
    }
    assert "host" not in rendered
    assert "raw_response" not in rendered
    assert "secret" not in rendered


def test_failed_cpu_and_timed_out_quality_have_honest_statuses(monkeypatch, tmp_path):
    preflight_report = {
        "ok": True,
        "source_commit_sha": "a" * 40,
        "model": {"tag": "model", "digest": "digest"},
        "gpu": {"used_mib": "0"},
    }

    monkeypatch.setattr(
        release_gate,
        "run_worker",
        lambda stage, _path, _deadline, _gpu_index=0: (0, preflight_report),
    )
    monkeypatch.setattr(release_gate, "run_bounded", lambda _command, _deadline: 1)
    cpu_path = tmp_path / "cpu-failed.json"

    assert release_gate.main(["--output", str(cpu_path)]) == 1
    cpu_report = json.loads(cpu_path.read_text(encoding="utf-8"))
    assert cpu_report["checks"]["preflight"] == "passed"
    assert cpu_report["checks"]["cpu"] == "failed"
    assert cpu_report["checks"]["quality"] == "not_run"

    def timeout_on_quality(stage, _path, _deadline, _gpu_index=0):
        if stage == "preflight":
            return 0, preflight_report
        raise TimeoutError("gate_deadline_exceeded")

    monkeypatch.setattr(release_gate, "run_worker", timeout_on_quality)
    monkeypatch.setattr(release_gate, "run_bounded", lambda _command, _deadline: 0)
    quality_path = tmp_path / "quality-timeout.json"

    assert release_gate.main(["--output", str(quality_path)]) == 1
    quality_report = json.loads(quality_path.read_text(encoding="utf-8"))
    assert quality_report["checks"]["cpu"] == "passed"
    assert quality_report["checks"]["quality"] == "failed"
    assert quality_report["checks"]["stability"] == "not_run"
    assert "gate_deadline_exceeded" in quality_report["failures"]


def test_preflight_runs_inside_the_whole_gate_deadline(monkeypatch, tmp_path):
    calls = []

    def expired(stage, _path, deadline, _gpu_index=0):
        calls.append((stage, deadline))
        raise TimeoutError("gate_deadline_exceeded")

    monkeypatch.setattr(release_gate, "run_worker", expired)
    path = tmp_path / "expired.json"

    assert release_gate.main(["--timeout-seconds", "1", "--output", str(path)]) == 1
    report = json.loads(path.read_text(encoding="utf-8"))
    assert calls[0][0] == "preflight"
    assert report["checks"]["preflight"] == "failed"
    assert "gate_deadline_exceeded" in report["failures"]


def test_minimum_ollama_version_allows_compatible_native_updates():
    assert release_gate.validate_ollama_version("0.33.0") == "0.33.0"
    assert release_gate.validate_ollama_version("v0.34.1-rc1") == "v0.34.1-rc1"
    with pytest.raises(ValueError, match="ollama_version_unsupported"):
        release_gate.validate_ollama_version("0.32.9")


def test_release_workflow_grants_read_only_check_access():
    workflow = (PROJECT_ROOT / ".github/workflows/release-gate.yml").read_text(encoding="utf-8")

    assert "checks: read" in workflow
    assert "GH_TOKEN: ${{ github.token }}" in workflow
