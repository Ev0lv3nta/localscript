#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
PREFERRED_PYTHON = ROOT / ".venv" / "bin" / "python"
PREFERRED_VENV = PREFERRED_PYTHON.parent.parent.resolve()
if (
    __name__ == "__main__"
    and PREFERRED_PYTHON.exists()
    and Path(sys.prefix).resolve() != PREFERRED_VENV
):
    os.execv(
        str(PREFERRED_PYTHON),
        [str(PREFERRED_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
    )

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.benchmarks import QUALITY_EVAL_MANIFEST, quality_gate_failures
from app.core.config import get_runtime_profile
from app.core.resources import materialized_resource
from app.core.runtime_lock import write_runtime_lock
from app.validation.validators import _find_lua_binary, _find_luac_binary


DEFAULT_TIMEOUTS = {
    "pytest_live": 30 * 60,
    "doctor": 90 * 60,
    "smoke": 20 * 60,
    "latency": 30 * 60,
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def timeout_for(name):
    environment_name = "LOCALSCRIPT_RELEASE_{0}_TIMEOUT_SECONDS".format(
        name.upper()
    )
    return int(os.getenv(environment_name, str(DEFAULT_TIMEOUTS[name])))


def run_command(name, command, cwd, extra_env=None):
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    started_at = utc_now()
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            timeout=timeout_for(name),
        )
        return {
            "name": name,
            "command": [str(part) for part in command],
            "timeout_seconds": timeout_for(name),
            "started_at": started_at,
            "finished_at": utc_now(),
            "duration_seconds": round(time.monotonic() - started, 3),
            "returncode": completed.returncode,
            "timed_out": False,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else exc.stdout
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else exc.stderr
        return {
            "name": name,
            "command": [str(part) for part in command],
            "timeout_seconds": timeout_for(name),
            "started_at": started_at,
            "finished_at": utc_now(),
            "duration_seconds": round(time.monotonic() - started, 3),
            "returncode": 124,
            "timed_out": True,
            "stdout": (stdout or "").strip(),
            "stderr": (stderr or "").strip(),
        }


def json_payload(command_report):
    try:
        return json.loads(command_report["stdout"] or "{}")
    except json.JSONDecodeError:
        return {
            "ok": False,
            "status": "error",
            "reason": "invalid_json_output",
            "raw_stdout": command_report["stdout"],
        }


def sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_evidence():
    evidence = {}
    for entry in QUALITY_EVAL_MANIFEST:
        resource_name = "datasets/{0}.jsonl".format(entry["name"])
        with materialized_resource(resource_name) as dataset_path:
            evidence[entry["name"]] = {
                "resource": resource_name,
                "role": entry["role"],
                "sha256": sha256_path(dataset_path),
            }
    return evidence


def command_identity(command):
    if not command:
        return {"available": False}
    path = Path(command)
    try:
        completed = subprocess.run(
            [str(path), "-v"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
        version = (completed.stdout or completed.stderr).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        version = "unavailable: {0}".format(exc)
    return {
        "available": path.is_file(),
        "path": str(path),
        "version": version,
        "sha256": sha256_path(path) if path.is_file() else None,
    }


def ollama_evidence(profile, selected_model):
    host = os.getenv("LOCALSCRIPT_OLLAMA_HOST", profile.ollama_host).rstrip("/")
    evidence = {"host": host, "reachable": False, "selected_model": selected_model}
    try:
        with httpx.Client(timeout=5.0, trust_env=False) as client:
            version_response = client.get(host + "/api/version")
            version_response.raise_for_status()
            evidence["version"] = version_response.json().get("version")
            tags_response = client.get(host + "/api/tags")
            tags_response.raise_for_status()
            tags = tags_response.json().get("models", [])
        evidence["reachable"] = True
        model = next(
            (
                item
                for item in tags
                if item.get("name") == selected_model
                or item.get("model") == selected_model
            ),
            None,
        )
        if model:
            evidence["model"] = {
                "name": model.get("name") or model.get("model"),
                "digest": model.get("digest"),
                "size": model.get("size"),
                "modified_at": model.get("modified_at"),
                "details": model.get("details"),
            }
    except Exception as exc:
        evidence["error"] = "{0}: {1}".format(type(exc).__name__, exc)
    return evidence


def git_commit_sha():
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            check=True,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return os.getenv("GITHUB_SHA")


def write_json(path, payload):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Проверка готовности LocalScript к релизу.")
    parser.add_argument(
        "--mode",
        choices=("dev", "competition"),
        default="competition",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    started_at = utc_now()
    python_bin = PREFERRED_PYTHON if PREFERRED_PYTHON.exists() else Path(sys.executable)
    profile = get_runtime_profile()

    live_tests = run_command(
        "pytest_live",
        [
            str(python_bin),
            "-m",
            "pytest",
            "-q",
            "-m",
            "integration",
            "--strict-markers",
        ],
        ROOT,
        extra_env={"LOCALSCRIPT_REQUIRE_LIVE": "1"},
    )

    with tempfile.TemporaryDirectory(prefix="localscript-release-gate-") as temp_dir:
        doctor_lock_path = Path(temp_dir) / "runtime_profile.lock.json"
        doctor = run_command(
            "doctor",
            [str(python_bin), "-m", "app.cli.main", "doctor", "--judge"],
            ROOT,
            extra_env={
                "LOCALSCRIPT_IGNORE_LOCK": "1",
                "LOCALSCRIPT_RUNTIME_LOCK_PATH": str(doctor_lock_path),
            },
        )
        doctor_report = json_payload(doctor)
        doctor_lock = (
            json.loads(doctor_lock_path.read_text(encoding="utf-8"))
            if doctor_lock_path.is_file()
            else {}
        )

    smoke = run_command(
        "smoke",
        [str(ROOT / "scripts" / "judge_smoke.sh")],
        ROOT,
        extra_env={"LOCALSCRIPT_SKIP_INSTALL": "1"},
    )
    smoke_report = json_payload(smoke)
    latency = run_command(
        "latency",
        [str(ROOT / "scripts" / "bench_latency.py")],
        ROOT,
    )
    latency_report = json_payload(latency)

    quality_report = doctor_report.get("quality_report", {})
    vram_report = doctor_report.get("vram_report", {})
    selected_model = doctor_report.get("selected_model") or profile.model
    commit_sha = git_commit_sha()
    failures = []
    if live_tests["returncode"] != 0:
        failures.append("integration_tests_failed")
    if doctor["returncode"] != 0 or doctor_report.get("ok") is not True:
        failures.append("doctor_judge_failed")
    failures.extend(quality_gate_failures(quality_report))
    if quality_report.get("backend_type") != "live_ollama":
        failures.append("quality_backend_not_live_ollama")
    if args.mode == "competition" and vram_report.get("status") != "ok":
        failures.append("selected_model_vram_not_ok")
    if smoke["returncode"] != 0 or smoke_report.get("ok") is not True:
        failures.append("smoke_failed")
    if latency["returncode"] != 0 or latency_report.get("ok") is not True:
        failures.append("latency_failed")
    if not doctor_lock or doctor_lock.get("locked") is not True:
        failures.append("doctor_runtime_snapshot_invalid")
    if not commit_sha:
        failures.append("commit_sha_missing")

    try:
        datasets = dataset_evidence()
        lua = command_identity(_find_lua_binary())
        luac = command_identity(_find_luac_binary())
        ollama = ollama_evidence(profile, selected_model)
    except Exception as exc:
        datasets = {}
        lua = {"available": False}
        luac = {"available": False}
        ollama = {"reachable": False}
        failures.append(
            "evidence_collection_failed::{0}::{1}".format(type(exc).__name__, exc)
        )
    if not lua.get("available"):
        failures.append("lua_identity_missing")
    if not luac.get("available"):
        failures.append("luac_identity_missing")
    if not ollama.get("reachable"):
        failures.append("ollama_evidence_unreachable")
    if not ollama.get("model", {}).get("digest"):
        failures.append("selected_model_digest_missing")
    failures = list(dict.fromkeys(failures))

    runtime_lock = dict(doctor_lock)
    runtime_lock.update(
        {
            "locked": not failures,
            "artifact_role": "release_gate_runtime_snapshot",
            "generated_by": "scripts/release_gate.py --mode {0}".format(args.mode),
            "release_gate_commit_sha": commit_sha,
            "hard_gate_failures": failures,
        }
    )
    runtime_lock_path = write_runtime_lock(runtime_lock)

    report = {
        "schema_version": 1,
        "mode": args.mode,
        "ok": not failures,
        "failures": failures,
        "started_at": started_at,
        "finished_at": utc_now(),
        "commit_sha": commit_sha,
        "runtime_snapshot_path": str(runtime_lock_path),
        "selected_model": selected_model,
        "parameters": {
            "profile": profile.dict(),
            "timeouts_seconds": {
                name: timeout_for(name) for name in DEFAULT_TIMEOUTS
            },
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
        },
        "runtime": {
            "python": {
                "version": platform.python_version(),
                "implementation": platform.python_implementation(),
                "executable": sys.executable,
            },
            "lua": lua,
            "luac": luac,
            "ollama": ollama,
        },
        "datasets": datasets,
        "commands": {
            "pytest_live": live_tests,
            "doctor": doctor,
            "smoke": smoke,
            "latency": latency,
        },
        "quality_report": quality_report,
        "doctor_report": doctor_report,
        "smoke_report": smoke_report,
        "latency_report": latency_report,
        "vram_report": vram_report,
    }

    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False))
    if failures:
        raise SystemExit(1)
    return report


if __name__ == "__main__":
    main()
