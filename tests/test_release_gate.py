import json
import subprocess
from types import SimpleNamespace

import pytest
from _pytest.outcomes import Failed

from app.core.benchmarks import QUALITY_EVAL_MANIFEST, quality_gate_failures
from app.core.runtime_lock import build_runtime_lock
from app.generation.ollama import OllamaBackend
from scripts import release_gate
from tests import conftest


def _quality_report(diagnostic_ok=False):
    report = {
        "backend_type": "live_ollama",
        "eval_manifest": [
            {"name": entry["name"], "role": entry["role"]}
            for entry in QUALITY_EVAL_MANIFEST
        ],
        "ok": True,
    }
    for entry in QUALITY_EVAL_MANIFEST:
        report[entry["name"]] = {
            "ok": True if entry["role"] == "mandatory" else diagnostic_ok
        }
    return report


def _command_report(name, payload=None, returncode=0):
    return {
        "name": name,
        "command": [name],
        "timeout_seconds": 10,
        "started_at": "2026-07-27T00:00:00+00:00",
        "finished_at": "2026-07-27T00:00:01+00:00",
        "duration_seconds": 1.0,
        "returncode": returncode,
        "timed_out": False,
        "stdout": json.dumps(payload or {}, ensure_ascii=False),
        "stderr": "",
    }


def test_live_fixture_fails_closed_when_backend_is_required(monkeypatch):
    monkeypatch.setenv("LOCALSCRIPT_REQUIRE_LIVE", "1")
    monkeypatch.setattr(OllamaBackend, "ping", lambda self: False)

    with pytest.raises(Failed, match="required but unavailable"):
        conftest.live_ollama_backend.__wrapped__()


def test_collection_rejects_mixed_unit_and_integration_markers():
    item = SimpleNamespace(
        nodeid="tests/test_example.py::test_mixed",
        iter_markers=lambda: [
            SimpleNamespace(name="unit"),
            SimpleNamespace(name="integration"),
        ],
    )

    with pytest.raises(pytest.UsageError, match="both unit and integration"):
        conftest.pytest_collection_modifyitems(None, [item])


def test_quality_manifest_gates_every_mandatory_set_but_not_diagnostics():
    report = _quality_report(diagnostic_ok=False)

    assert quality_gate_failures(report) == []
    assert "large_context_eval" in [
        entry["name"]
        for entry in QUALITY_EVAL_MANIFEST
        if entry["role"] == "mandatory"
    ]

    report["large_context_eval"]["ok"] = False
    assert quality_gate_failures(report) == ["large_context_eval_failed"]

    report = _quality_report(diagnostic_ok=False)
    report["eval_manifest"] = report["eval_manifest"][:-1]
    assert quality_gate_failures(report) == ["quality_manifest_mismatch"]


def test_runtime_lock_is_not_locked_when_gate_has_failures():
    profile = release_gate.get_runtime_profile()

    payload = build_runtime_lock(
        profile=profile,
        selected_model=profile.model,
        selection_reason="test",
        quality_report={"ok": False},
        vram_report={"status": "error"},
        hard_gate_failures=["large_context_eval_failed"],
    )

    assert payload["locked"] is False


def test_command_timeout_is_reported_without_hanging(monkeypatch):
    monkeypatch.setattr(release_gate, "timeout_for", lambda name: 1)

    def raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], timeout=1, output="partial")

    monkeypatch.setattr(release_gate.subprocess, "run", raise_timeout)

    report = release_gate.run_command("doctor", ["doctor"], release_gate.ROOT)

    assert report["returncode"] == 124
    assert report["timed_out"] is True
    assert report["stdout"] == "partial"


def test_release_gate_writes_sha_bound_artifact_and_runs_quality_once(
    monkeypatch, tmp_path
):
    quality_report = _quality_report(diagnostic_ok=False)
    doctor_report = {
        "ok": True,
        "selected_model": "qwen3:test",
        "quality_report": quality_report,
        "vram_report": {"status": "ok"},
    }
    calls = []

    def fake_run_command(name, command, cwd, extra_env=None):
        calls.append(name)
        if name == "doctor":
            lock_path = extra_env["LOCALSCRIPT_RUNTIME_LOCK_PATH"]
            with open(lock_path, "w", encoding="utf-8") as stream:
                json.dump(
                    {
                        "profile": "competition",
                        "selected_model": "qwen3:test",
                        "locked": True,
                    },
                    stream,
                )
            return _command_report(name, doctor_report)
        if name in {"smoke", "latency"}:
            return _command_report(name, {"ok": True})
        return _command_report(name)

    output = tmp_path / "release-gate.json"
    lock_path = tmp_path / "runtime-profile.lock.json"
    monkeypatch.setenv("LOCALSCRIPT_RUNTIME_LOCK_PATH", str(lock_path))
    monkeypatch.setattr(release_gate, "run_command", fake_run_command)
    monkeypatch.setattr(release_gate, "git_commit_sha", lambda: "deadbeef")
    monkeypatch.setattr(
        release_gate,
        "command_identity",
        lambda command: {
            "available": True,
            "version": "Lua 5.4.6",
            "sha256": "sha256:lua",
        },
    )
    monkeypatch.setattr(
        release_gate,
        "ollama_evidence",
        lambda profile, selected_model: {
            "reachable": True,
            "selected_model": selected_model,
            "version": "0.20.7",
            "model": {"digest": "sha256:model"},
        },
    )

    report = release_gate.main(
        ["--mode", "competition", "--output", str(output)]
    )

    assert report["ok"] is True
    assert report["commit_sha"] == "deadbeef"
    assert report["runtime"]["ollama"]["model"]["digest"] == "sha256:model"
    assert calls == ["pytest_live", "doctor", "smoke", "latency"]
    assert json.loads(output.read_text(encoding="utf-8"))["ok"] is True
    assert json.loads(lock_path.read_text(encoding="utf-8"))["locked"] is True


def test_release_gate_failure_writes_unlocked_runtime_snapshot(
    monkeypatch, tmp_path
):
    quality_report = _quality_report()
    quality_report["large_context_eval"]["ok"] = False
    quality_report["ok"] = False
    doctor_report = {
        "ok": False,
        "selected_model": "qwen3:test",
        "quality_report": quality_report,
        "vram_report": {"status": "ok"},
    }

    def fake_run_command(name, command, cwd, extra_env=None):
        if name == "doctor":
            with open(
                extra_env["LOCALSCRIPT_RUNTIME_LOCK_PATH"],
                "w",
                encoding="utf-8",
            ) as stream:
                json.dump({"profile": "competition", "locked": False}, stream)
            return _command_report(name, doctor_report, returncode=1)
        if name in {"smoke", "latency"}:
            return _command_report(name, {"ok": True})
        return _command_report(name)

    output = tmp_path / "failed-release-gate.json"
    lock_path = tmp_path / "runtime-profile.lock.json"
    monkeypatch.setenv("LOCALSCRIPT_RUNTIME_LOCK_PATH", str(lock_path))
    monkeypatch.setattr(release_gate, "run_command", fake_run_command)
    monkeypatch.setattr(release_gate, "git_commit_sha", lambda: "badc0de")
    monkeypatch.setattr(release_gate, "command_identity", lambda command: {})
    monkeypatch.setattr(release_gate, "ollama_evidence", lambda *args: {})

    with pytest.raises(SystemExit):
        release_gate.main(
            ["--mode", "competition", "--output", str(output)]
        )

    artifact = json.loads(output.read_text(encoding="utf-8"))
    runtime_lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert artifact["ok"] is False
    assert "large_context_eval_failed" in artifact["failures"]
    assert runtime_lock["locked"] is False
