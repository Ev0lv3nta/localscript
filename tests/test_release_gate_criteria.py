import importlib.util
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
    def unavailable(_index):
        raise ValueError("model_missing")

    monkeypatch.setattr(release_gate, "preflight", unavailable)
    path = tmp_path / "report.json"
    assert release_gate.main(["--output", str(path)]) == 1
    import json

    report = json.loads(path.read_text())
    assert not report["ok"]
    assert report["checks"]["quality"] == "not_run"
    assert "model_missing" in report["failures"]
