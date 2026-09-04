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
from app.evaluation.integrity import run_integrity_check
from app.validation.runtime import find_lua_binary, find_luac_binary

DEFAULT_TIMEOUTS = {
    "pytest_live": 30 * 60,
    "doctor": 90 * 60,
    "smoke": 20 * 60,
    "latency": 30 * 60,
    "private_holdout": 30 * 60,
    "repeat_stability": 60 * 60,
}

PRIVATE_HOLDOUT_SCALAR_METRICS = frozenset(
    {
        "syntax_pass_rate",
        "semantic_pass_rate",
        "verified_completion_rate",
        "invalid_success_rate",
        "invalid_success_count",
        "repair_attempt_count",
        "repair_rescue_count",
        "repair_rescue_rate",
        "degraded_count",
        "backend_calls_total",
        "backend_calls_mean",
        "model_duration_ms_total",
    }
)
PRIVATE_HOLDOUT_NESTED_METRICS = {
    "model_call_latency_ms": frozenset({"p50", "p95"}),
    "latency_ms": frozenset(
        {"cold_first", "warm_p50", "warm_p95", "overall_p50", "overall_p95"}
    ),
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def timeout_for(name):
    environment_name = "LOCALSCRIPT_RELEASE_{0}_TIMEOUT_SECONDS".format(name.upper())
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
        payload = json.loads(command_report["stdout"] or "{}")
    except json.JSONDecodeError:
        return {
            "ok": False,
            "status": "error",
            "reason": "invalid_json_output",
        }
    if not isinstance(payload, dict):
        return {
            "ok": False,
            "status": "error",
            "reason": "invalid_json_shape",
        }
    return payload


def _redact_local_paths(value):
    if isinstance(value, str):
        redacted = value
        for path, replacement in (
            (str(ROOT), "<project>"),
            (str(Path.home()), "<home>"),
        ):
            if path:
                redacted = redacted.replace(path, replacement)
        return redacted
    if isinstance(value, list):
        return [_redact_local_paths(item) for item in value]
    if isinstance(value, dict):
        return {key: _redact_local_paths(item) for key, item in value.items()}
    return value


def command_public_evidence(command_report):
    """Keep execution metadata without publishing paths or raw command output."""
    allowed_fields = (
        "name",
        "timeout_seconds",
        "started_at",
        "finished_at",
        "duration_seconds",
        "returncode",
        "timed_out",
    )
    return {field: command_report.get(field) for field in allowed_fields}


def private_holdout_validation_failures(benchmark_report, integrity_report):
    identity = integrity_report.get("private_holdout") or {}
    failures = []
    if identity.get("ok") is not True:
        failures.append("private_holdout_integrity_not_verified")
    if benchmark_report.get("dataset_sha256") != identity.get("sha256"):
        failures.append("private_holdout_identity_mismatch")
    expected_count = identity.get("case_count")
    total = benchmark_report.get("total")
    passed = benchmark_report.get("passed")
    failed = benchmark_report.get("failed")
    if total != expected_count:
        failures.append("private_holdout_case_count_mismatch")
    if benchmark_report.get("backend_type") != "live_ollama":
        failures.append("private_holdout_backend_not_live_ollama")
    valid_counts = all(type(value) is int for value in (total, passed, failed))
    if (
        not valid_counts
        or passed + failed != total
        or passed != expected_count
        or failed != 0
    ):
        failures.append("private_holdout_result_counts_invalid")
    metrics = benchmark_report.get("metrics")
    invalid_success_count = (
        metrics.get("invalid_success_count") if isinstance(metrics, dict) else None
    )
    if type(invalid_success_count) is not int or invalid_success_count != 0:
        failures.append("private_holdout_invalid_success_detected")
    report_schema_valid = (
        benchmark_report.get("schema_version") == 2
        and benchmark_report.get("ok") is True
        and isinstance(benchmark_report.get("failures"), list)
        and not benchmark_report.get("failures")
        and isinstance(metrics, dict)
        and type(metrics.get("verified_completion_rate")) in {int, float}
        and metrics.get("verified_completion_rate") == 1.0
        and type(metrics.get("invalid_success_rate")) in {int, float}
        and metrics.get("invalid_success_rate") == 0.0
    )
    observations = benchmark_report.get("case_results")
    if not isinstance(observations, list) or not valid_counts or len(observations) != total:
        report_schema_valid = False
    else:
        case_ids = []
        for observation in observations:
            if not isinstance(observation, dict):
                report_schema_valid = False
                break
            case_id = observation.get("id")
            if (
                not isinstance(case_id, str)
                or not case_id
                or observation.get("passed") is not True
                or observation.get("errors") != []
            ):
                report_schema_valid = False
                break
            case_ids.append(case_id)
        if len(case_ids) != len(set(case_ids)):
            report_schema_valid = False
    if not report_schema_valid:
        failures.append("private_holdout_report_schema_invalid")
    return failures


def _private_holdout_metrics(metrics):
    if not isinstance(metrics, dict):
        return {}
    public_metrics = {
        key: value
        for key, value in metrics.items()
        if key in PRIVATE_HOLDOUT_SCALAR_METRICS
        and (value is None or type(value) in {int, float})
    }
    for section, allowed_fields in PRIVATE_HOLDOUT_NESTED_METRICS.items():
        values = metrics.get(section)
        if not isinstance(values, dict):
            continue
        public_metrics[section] = {
            key: value
            for key, value in values.items()
            if key in allowed_fields and (value is None or type(value) in {int, float})
        }
    return public_metrics


def _private_holdout_error_category(error):
    if not isinstance(error, str) or not error:
        return None
    category = error.split("::", 1)[0].lower()
    buckets = (
        ("syntax", ("syntax", "lua_load")),
        ("semantic", ("semantic",)),
        ("contract", ("contract", "shape", "structure", "format")),
        ("oracle", ("oracle", "expected_result")),
        ("strategy", ("strategy",)),
        ("backend", ("backend", "model", "ollama", "timeout")),
        ("policy", ("policy", "forbidden", "security")),
    )
    for bucket, markers in buckets:
        if any(marker in category for marker in markers):
            return bucket
    return "other"


def evaluation_integrity_public_report(integrity_report):
    private_identity = integrity_report.get("private_holdout")
    if isinstance(private_identity, dict):
        private_identity = {
            key: private_identity.get(key)
            for key in ("name", "case_count", "sha256", "ok")
        }
        error_category = _private_holdout_error_category(
            (integrity_report.get("private_holdout") or {}).get("error")
        )
        if error_category:
            private_identity["error_category"] = error_category

    error_categories = sorted(
        {
            category
            for error in integrity_report.get("errors") or []
            if (category := _private_holdout_error_category(error))
        }
    )
    overlap_kinds = {}
    for finding in integrity_report.get("overlaps") or []:
        kind = finding.get("kind") if isinstance(finding, dict) else None
        safe_kind = kind if kind in {"normalized_exact", "fuzzy"} else "other"
        overlap_kinds[safe_kind] = overlap_kinds.get(safe_kind, 0) + 1

    public_datasets = []
    for dataset in integrity_report.get("datasets") or []:
        if not isinstance(dataset, dict):
            continue
        logical_path = dataset.get("path")
        if (
            not isinstance(logical_path, str)
            or Path(logical_path).is_absolute()
            or ".." in Path(logical_path).parts
        ):
            logical_path = None
        public_datasets.append(
            {
                "name": dataset.get("name") if isinstance(dataset.get("name"), str) else None,
                "path": logical_path,
                "corpus": (
                    dataset.get("corpus")
                    if dataset.get("corpus") in {"public_benchmark", "regression"}
                    else None
                ),
                "runner": (
                    dataset.get("runner")
                    if dataset.get("runner") in {"standard", "rich"}
                    else None
                ),
                "gate": (
                    dataset.get("gate")
                    if dataset.get("gate") in {"required", "diagnostic"}
                    else None
                ),
                "claim_scope": (
                    dataset.get("claim_scope")
                    if isinstance(dataset.get("claim_scope"), str)
                    else None
                ),
                "case_count": (
                    dataset.get("case_count")
                    if type(dataset.get("case_count")) is int
                    else None
                ),
                "sha256": (
                    dataset.get("sha256")
                    if isinstance(dataset.get("sha256"), str)
                    and len(dataset.get("sha256")) == 64
                    else None
                ),
            }
        )

    return {
        "schema_version": integrity_report.get("schema_version"),
        "ok": integrity_report.get("ok") is True,
        "error_categories": error_categories,
        "datasets": public_datasets,
        "private_holdout": private_identity,
        "overlap_count": sum(overlap_kinds.values()),
        "overlap_kinds": overlap_kinds,
        "fuzzy_threshold": integrity_report.get("fuzzy_threshold"),
    }


def doctor_public_report(report):
    return {
        "profile": report.get("profile"),
        "model": report.get("model"),
        "fallback_model": report.get("fallback_model"),
        "trace_dir_writable": report.get("trace_dir_writable") is True,
        "ollama_reachable": report.get("ollama_reachable") is True,
        "judge_mode": report.get("judge_mode") is True,
        "selected_model": report.get("selected_model"),
        "selection_reason": report.get("selection_reason"),
        "available_tags": report.get("available_tags") or [],
        "hard_gate_failures": report.get("hard_gate_failures") or [],
        "ok": report.get("ok") is True,
        "runtime_snapshot": "runtime_profile.lock.json",
    }


def private_holdout_public_report(benchmark_report, integrity_report):
    """Reduce private benchmark evidence to identity and aggregate results."""
    identity = integrity_report.get("private_holdout") or {}
    error_categories = set()
    for observation in benchmark_report.get("case_results") or []:
        if not isinstance(observation, dict):
            continue
        for error in observation.get("errors") or []:
            category = _private_holdout_error_category(error)
            if category:
                error_categories.add(category)

    validation_failures = private_holdout_validation_failures(
        benchmark_report,
        integrity_report,
    )

    return {
        "name": identity.get("name"),
        "sha256": identity.get("sha256"),
        "case_count": identity.get("case_count"),
        "backend_type": (
            "live_ollama"
            if benchmark_report.get("backend_type") == "live_ollama"
            else "unexpected"
        ),
        "passed": (
            benchmark_report.get("passed")
            if type(benchmark_report.get("passed")) is int
            else None
        ),
        "failed": (
            benchmark_report.get("failed")
            if type(benchmark_report.get("failed")) is int
            else None
        ),
        "identity_verified": not {
            "private_holdout_integrity_not_verified",
            "private_holdout_identity_mismatch",
            "private_holdout_case_count_mismatch",
        }.intersection(validation_failures),
        "ok": benchmark_report.get("ok") is True and not validation_failures,
        "metrics": _private_holdout_metrics(benchmark_report.get("metrics")),
        "error_categories": sorted(error_categories),
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
        resource_name = entry["path"]
        with materialized_resource(resource_name) as dataset_path:
            evidence[entry["name"]] = {
                "resource": resource_name,
                "corpus": entry["corpus"],
                "gate": entry["gate"],
                "claim_scope": entry["claim_scope"],
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
        version = "unavailable: {0}".format(type(exc).__name__)
    return {
        "available": path.is_file(),
        "executable": path.name,
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
                if item.get("name") == selected_model or item.get("model") == selected_model
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
        evidence["error"] = type(exc).__name__
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


def git_evidence():
    commit_sha = git_commit_sha()
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            check=True,
        )
        dirty = bool(completed.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        dirty = None
    return {"commit_sha": commit_sha, "dirty": dirty}


def gpu_evidence():
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            check=True,
        )
        rows = []
        for line in completed.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) == 3:
                rows.append(
                    {
                        "name": parts[0],
                        "driver_version": parts[1],
                        "memory_total_mib": int(parts[2]),
                    }
                )
        return {"available": bool(rows), "gpus": rows}
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        return {
            "available": False,
            "gpus": [],
            "error": type(error).__name__,
        }


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
    parser.add_argument("--private-holdout", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    started_at = utc_now()
    write_runtime_lock(
        {
            "locked": False,
            "profile": None,
            "artifact_role": "release_gate_runtime_snapshot",
            "generated_by": "scripts/release_gate.py --mode {0}".format(args.mode),
            "release_gate_commit_sha": None,
            "hard_gate_failures": ["release_gate_initializing"],
        }
    )
    python_bin = PREFERRED_PYTHON if PREFERRED_PYTHON.exists() else Path(sys.executable)
    profile = get_runtime_profile()
    private_holdout_path = args.private_holdout or os.getenv("LOCALSCRIPT_PRIVATE_HOLDOUT_PATH")
    source_evidence = git_evidence()
    runtime_lock_path = write_runtime_lock(
        {
            "locked": False,
            "profile": profile.name,
            "artifact_role": "release_gate_runtime_snapshot",
            "generated_by": "scripts/release_gate.py --mode {0}".format(args.mode),
            "release_gate_commit_sha": source_evidence.get("commit_sha"),
            "hard_gate_failures": ["release_gate_in_progress"],
        }
    )
    try:
        integrity_report = run_integrity_check(private_holdout_path=private_holdout_path)
    except Exception as exc:
        failures = ["eval_integrity_error::{0}".format(type(exc).__name__)]
        runtime_lock_path = write_runtime_lock(
            {
                "locked": False,
                "profile": profile.name,
                "artifact_role": "release_gate_runtime_snapshot",
                "generated_by": "scripts/release_gate.py --mode {0}".format(args.mode),
                "release_gate_commit_sha": source_evidence.get("commit_sha"),
                "hard_gate_failures": failures,
            }
        )
        report = {
            "schema_version": 2,
            "mode": args.mode,
            "ok": False,
            "failures": failures,
            "started_at": started_at,
            "finished_at": utc_now(),
            "source": source_evidence,
            "runtime_snapshot_path": runtime_lock_path.name,
            "evaluation_integrity": None,
            "preflight_only": True,
        }
        if args.output:
            write_json(args.output, report)
        print(json.dumps(report, ensure_ascii=False))
        raise SystemExit(1)

    preflight_failures = []
    if args.mode == "competition" and private_holdout_path is None:
        preflight_failures.append("private_holdout_not_supplied")
    if integrity_report.get("ok") is not True:
        preflight_failures.append("eval_integrity_failed")
    if args.mode == "competition" and source_evidence.get("dirty") is not False:
        preflight_failures.append("release_worktree_not_clean")
    if not source_evidence.get("commit_sha"):
        preflight_failures.append("commit_sha_missing")
    if preflight_failures:
        runtime_lock_path = write_runtime_lock(
            {
                "locked": False,
                "profile": profile.name,
                "artifact_role": "release_gate_runtime_snapshot",
                "generated_by": "scripts/release_gate.py --mode {0}".format(args.mode),
                "release_gate_commit_sha": source_evidence.get("commit_sha"),
                "hard_gate_failures": preflight_failures,
            }
        )
        report = {
            "schema_version": 2,
            "mode": args.mode,
            "ok": False,
            "failures": preflight_failures,
            "started_at": started_at,
            "finished_at": utc_now(),
            "source": source_evidence,
            "runtime_snapshot_path": runtime_lock_path.name,
            "evaluation_integrity": evaluation_integrity_public_report(integrity_report),
            "preflight_only": True,
        }
        if args.output:
            write_json(args.output, report)
        print(json.dumps(report, ensure_ascii=False))
        raise SystemExit(1)

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

    quality_report = doctor_report.get("quality_report", {})
    vram_report = doctor_report.get("vram_report", {})
    selected_model = doctor_report.get("selected_model") or profile.model
    private_holdout = None
    private_holdout_report = None
    private_holdout_public_evidence = None
    private_holdout_validation_errors = []

    repeat_stability = run_command(
        "repeat_stability",
        [
            str(python_bin),
            str(ROOT / "scripts" / "bench_repeated.py"),
            "--dataset",
            "evals/public/v1.jsonl",
            "--repeats",
            "3",
        ],
        ROOT,
        extra_env={"LOCALSCRIPT_PRIMARY_MODEL": selected_model},
    )
    repeat_stability_report = json_payload(repeat_stability)

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

    private_holdout_preconditions_ok = (
        live_tests["returncode"] == 0
        and doctor["returncode"] == 0
        and doctor_report.get("ok") is True
        and not quality_gate_failures(quality_report)
        and quality_report.get("backend_type") == "live_ollama"
        and (args.mode != "competition" or vram_report.get("status") == "ok")
        and repeat_stability["returncode"] == 0
        and repeat_stability_report.get("ok") is True
        and smoke["returncode"] == 0
        and smoke_report.get("ok") is True
        and latency["returncode"] == 0
        and latency_report.get("ok") is True
        and doctor_lock.get("locked") is True
    )
    if private_holdout_path is not None and private_holdout_preconditions_ok:
        private_holdout = run_command(
            "private_holdout",
            [
                str(python_bin),
                "-m",
                "app.cli.main",
                "benchmark",
                "--dataset",
                str(private_holdout_path),
            ],
            ROOT,
            extra_env={"LOCALSCRIPT_PRIMARY_MODEL": selected_model},
        )
        private_holdout_report = json_payload(private_holdout)
        private_holdout_public_evidence = private_holdout_public_report(
            private_holdout_report,
            integrity_report,
        )
        private_holdout_validation_errors = private_holdout_validation_failures(
            private_holdout_report,
            integrity_report,
        )

    commit_sha = source_evidence["commit_sha"]
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
    if private_holdout_path is not None:
        if private_holdout is None:
            failures.append("private_holdout_not_run_due_to_public_failures")
        elif (
            private_holdout["returncode"] != 0
            or private_holdout_report.get("ok") is not True
        ):
            failures.append("private_holdout_failed")
    failures.extend(private_holdout_validation_errors)
    if repeat_stability["returncode"] != 0 or repeat_stability_report.get("ok") is not True:
        failures.append("repeat_stability_failed")
    if not doctor_lock or doctor_lock.get("locked") is not True:
        failures.append("doctor_runtime_snapshot_invalid")

    try:
        datasets = dataset_evidence()
        lua = command_identity(find_lua_binary())
        luac = command_identity(find_luac_binary())
        ollama = ollama_evidence(profile, selected_model)
        gpu = gpu_evidence()
    except Exception as exc:
        datasets = {}
        lua = {"available": False}
        luac = {"available": False}
        ollama = {"reachable": False}
        gpu = {"available": False, "gpus": []}
        failures.append("evidence_collection_failed::{0}".format(type(exc).__name__))
    if not lua.get("available"):
        failures.append("lua_identity_missing")
    if not luac.get("available"):
        failures.append("luac_identity_missing")
    if not ollama.get("reachable"):
        failures.append("ollama_evidence_unreachable")
    if not ollama.get("model", {}).get("digest"):
        failures.append("selected_model_digest_missing")
    if args.mode == "competition" and not gpu.get("available"):
        failures.append("gpu_evidence_missing")
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
        "schema_version": 2,
        "mode": args.mode,
        "ok": not failures,
        "failures": failures,
        "started_at": started_at,
        "finished_at": utc_now(),
        "commit_sha": commit_sha,
        "source": source_evidence,
        "runtime_snapshot_path": runtime_lock_path.name,
        "selected_model": selected_model,
        "parameters": {
            "profile": profile.model_dump(),
            "public_benchmark_repeats": 3,
            "private_holdout_repeats": 1,
            "timeouts_seconds": {name: timeout_for(name) for name in DEFAULT_TIMEOUTS},
            "mandatory_eval_sets": [
                entry["name"] for entry in QUALITY_EVAL_MANIFEST if entry["gate"] == "required"
            ],
            "diagnostic_eval_sets": [
                entry["name"] for entry in QUALITY_EVAL_MANIFEST if entry["gate"] == "diagnostic"
            ],
        },
        "runtime": {
            "python": {
                "version": platform.python_version(),
                "implementation": platform.python_implementation(),
                "executable": Path(sys.executable).name,
            },
            "lua": lua,
            "luac": luac,
            "ollama": ollama,
            "gpu": gpu,
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
            },
        },
        "datasets": datasets,
        "evaluation_integrity": evaluation_integrity_public_report(integrity_report),
        "commands": {
            "pytest_live": command_public_evidence(live_tests),
            "doctor": command_public_evidence(doctor),
            "smoke": command_public_evidence(smoke),
            "latency": command_public_evidence(latency),
            "private_holdout": (
                command_public_evidence(private_holdout)
                if private_holdout is not None
                else None
            ),
            "repeat_stability": command_public_evidence(repeat_stability),
        },
        "quality_report": _redact_local_paths(quality_report),
        "doctor_report": doctor_public_report(doctor_report),
        "smoke_report": _redact_local_paths(smoke_report),
        "latency_report": _redact_local_paths(latency_report),
        "private_holdout_report": private_holdout_public_evidence,
        "repeat_stability_report": _redact_local_paths(repeat_stability_report),
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
