#!/usr/bin/env python3
"""One bounded live gate: preflight, CPU checks, quality, stability, report."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import signal
import subprocess
import sys
import tempfile
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.benchmarks import run_quality_benchmark, run_stability_benchmark
from app.core.config import get_runtime_profile
from app.core.storage import TRACE_PRIVATE_KEYS
from app.evaluation.integrity import run_integrity_check
from app.generation.backend_errors import BackendError
from app.validation.runtime import find_lua_binary, find_luac_binary, runtime_version

LIVE_GATE_BUDGET_SECONDS = 20 * 60
MINIMUM_OLLAMA_VERSION = (0, 33, 0)
REQUIRED_CI_CHECK = "CI / required"
REPORT_PRIVATE_KEYS = TRACE_PRIVATE_KEYS | frozenset({"code", "host"})


def capture(command, *, timeout=10):
    return subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, timeout=timeout, check=True
    ).stdout.strip()


def sanitize_report(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: sanitize_report(nested)
            for key, nested in value.items()
            if isinstance(key, str)
            and key.casefold() not in REPORT_PRIVATE_KEYS
            and not key.casefold().startswith("raw_")
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_report(item) for item in value]
    return value


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(sanitize_report(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _ollama_version_tuple(version: object) -> tuple[int, int, int]:
    normalized = str(version or "").strip().removeprefix("v")
    parts = normalized.split(".")
    if len(parts) < 3:
        raise ValueError("ollama_version_invalid")
    try:
        return (
            int(parts[0]),
            int(parts[1]),
            int(parts[2].split("-", 1)[0]),
        )
    except ValueError:
        raise ValueError("ollama_version_invalid") from None


def validate_ollama_version(version: object) -> str:
    actual = str(version or "").strip()
    if _ollama_version_tuple(actual) < MINIMUM_OLLAMA_VERSION:
        raise ValueError("ollama_version_unsupported")
    return actual


def verify_required_ci(sha: str) -> dict[str, str]:
    repository = os.getenv("GITHUB_REPOSITORY") or capture(
        ["gh", "repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner"]
    )
    if repository.count("/") != 1:
        raise ValueError("github_repository_invalid")
    raw = capture(
        [
            "gh",
            "api",
            "-H",
            "Accept: application/vnd.github+json",
            f"repos/{repository}/commits/{sha}/check-runs?per_page=100",
        ],
        timeout=15,
    )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError("required_ci_response_invalid") from None
    check_runs = payload.get("check_runs") if isinstance(payload, dict) else None
    if not isinstance(check_runs, list):
        raise ValueError("required_ci_response_invalid")
    matches = [
        item
        for item in check_runs
        if isinstance(item, dict)
        and item.get("name") == REQUIRED_CI_CHECK
        and item.get("head_sha") == sha
        and isinstance(item.get("app"), dict)
        and item["app"].get("slug") == "github-actions"
    ]
    if not matches:
        raise ValueError("required_ci_missing")
    latest = max(matches, key=lambda item: int(item.get("id") or 0))
    if latest.get("status") != "completed" or latest.get("conclusion") != "success":
        raise ValueError("required_ci_not_successful")
    return {
        "name": REQUIRED_CI_CHECK,
        "sha": sha,
        "status": "completed",
        "conclusion": "success",
    }


def gpu_sample(index):
    row = capture(
        [
            "nvidia-smi",
            "-i",
            str(index),
            "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        timeout=3,
    )
    fields = [field.strip() for field in row.split(",")]
    if len(fields) != 6:
        raise ValueError("gpu_inventory_invalid")
    return dict(
        zip(("index", "uuid", "name", "driver", "total_mib", "used_mib"), fields, strict=True)
    )


def preflight(gpu_index):
    sha = capture(["git", "rev-parse", "HEAD"])
    if capture(["git", "status", "--porcelain"]):
        raise ValueError("dirty_checkout")
    capture(["git", "merge-base", "--is-ancestor", "HEAD", "origin/main"])
    required_ci = verify_required_ci(sha)
    if sys.version_info[:2] not in {(3, 11), (3, 12)}:
        raise ValueError("unsupported_python")
    lua, luac = find_lua_binary(), find_luac_binary()
    if (
        not lua
        or not luac
        or not runtime_version(lua)
        or runtime_version(lua) != runtime_version(luac)
    ):
        raise ValueError("lua54_pair_unavailable")
    profile = get_runtime_profile()
    if profile.ollama_host.startswith("https://ollama.com") or profile.model.endswith(":cloud"):
        raise ValueError("local_model_required")
    with httpx.Client(base_url=profile.ollama_host, timeout=3, trust_env=False) as client:
        version_response = client.get("/api/version")
        version_response.raise_for_status()
        version = validate_ollama_version(version_response.json()["version"])
        from app.generation.ollama import OllamaBackend

        with OllamaBackend(profile) as backend:
            resolved = backend.resolve_model()
        if not resolved.digest:
            raise ValueError("model_digest_missing")
    integrity = run_integrity_check()
    if not integrity.get("ok"):
        raise ValueError("corpus_integrity_failed")
    return {
        "ok": True,
        "source_commit_sha": sha,
        "dirty": False,
        "os": platform.system(),
        "python": platform.python_version(),
        "lua": runtime_version(lua),
        "luac": runtime_version(luac),
        "ollama": version,
        "ollama_minimum": ".".join(str(part) for part in MINIMUM_OLLAMA_VERSION),
        "required_ci": required_ci,
        "model": {"tag": resolved.tag, "digest": resolved.digest},
        "options": {
            name: getattr(profile, name)
            for name in (
                "think",
                "num_ctx",
                "num_predict",
                "batch",
                "parallel",
                "request_timeout_seconds",
            )
        },
        "gpu": gpu_sample(gpu_index),
        "integrity": integrity,
    }


def run_bounded(command, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("gate_deadline_exceeded")
    with (
        tempfile.TemporaryFile() as log,
        subprocess.Popen(
            command, cwd=ROOT, stdout=log, stderr=log, start_new_session=True
        ) as child,
    ):
        try:
            result = child.wait(timeout=remaining)
        except BaseException:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            raise
    return result


def run_worker(stage, output_path, deadline, gpu_index=0):
    code = run_bounded(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            stage,
            "--gpu-index",
            str(gpu_index),
            "--output",
            str(output_path),
        ],
        deadline,
    )
    if not output_path.is_file():
        raise ValueError(f"{stage}_report_missing")
    report = json.loads(output_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError(f"{stage}_report_invalid")
    return code, report


def _failure_name(error):
    if isinstance(error, BackendError):
        return error.reason
    if isinstance(error, (TimeoutError, subprocess.TimeoutExpired)):
        return "gate_deadline_exceeded"
    if isinstance(error, ValueError):
        return str(error)
    return type(error).__name__


def report_failures(quality, stability):
    failures = []
    if quality.get("backend_type") != "live_ollama":
        failures.append("live_backend_required")
    if quality.get("ok") is not True:
        failures.extend(quality.get("gate_failures") or ["quality_gate_failed"])
    if stability.get("backend_type") != "live_ollama" or stability.get("ok") is not True:
        failures.append("stability_failed")
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser(description="Один ограниченный прогон перед выпуском.")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/validation/release-gate.json"
    )
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=LIVE_GATE_BUDGET_SECONDS)
    parser.add_argument(
        "--worker",
        choices=("preflight", "quality", "stability"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--mode",
        choices=("local", "competition"),
        default="local",
        help="competition оставлен как совместимый alias",
    )
    args = parser.parse_args(argv)
    if args.worker:
        try:
            if args.worker == "preflight":
                report = preflight(args.gpu_index)
            elif args.worker == "quality":
                report = run_quality_benchmark(mode="competition")
            else:
                report = run_stability_benchmark()
        except (
            BackendError,
            OSError,
            ValueError,
            KeyError,
            httpx.HTTPError,
            subprocess.SubprocessError,
        ) as error:
            report = {"ok": False, "failure": _failure_name(error)}
        write_report(args.output, report)
        return 0 if report.get("ok") else 1
    if args.timeout_seconds <= 0 or args.timeout_seconds > 3600:
        parser.error("timeout должен быть от 1 до 3600 секунд")
    started = time.monotonic()
    deadline = started + args.timeout_seconds
    report = {
        "schema_version": 3,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "budget_seconds": args.timeout_seconds,
        "ok": False,
        "failures": [],
        "checks": {},
    }
    stop = threading.Event()
    memory_samples = []
    sampler_errors = []

    def sample_memory():
        while not stop.wait(0.5):
            try:
                memory_samples.append(int(gpu_sample(args.gpu_index)["used_mib"]))
            except (OSError, ValueError, subprocess.SubprocessError):
                sampler_errors.append("gpu_sampling_failed")
                return

    sampler = None
    try:
        with tempfile.TemporaryDirectory(prefix="localscript-live-") as temporary:
            temporary_root = Path(temporary)
            preflight_path = temporary_root / "preflight.json"
            report["checks"]["preflight"] = "failed"
            preflight_code, preflight_report = run_worker(
                "preflight",
                preflight_path,
                deadline,
                args.gpu_index,
            )
            report["preflight"] = preflight_report
            if preflight_code:
                raise ValueError(str(preflight_report.get("failure") or "preflight_failed"))
            report["checks"]["preflight"] = "passed"

            cpu_commands = [
                [sys.executable, "-m", "ruff", "check", "."],
                [sys.executable, "-m", "ruff", "format", "--check", "."],
                [sys.executable, "-m", "mypy", "--strict", "app"],
                [sys.executable, "-m", "pytest", "-q", "-m", "unit"],
            ]
            report["checks"]["cpu"] = "failed"
            for command in cpu_commands:
                if run_bounded(command, deadline):
                    raise ValueError("cpu_checks_failed")
            report["checks"]["cpu"] = "passed"

            sampler = threading.Thread(target=sample_memory, daemon=True)
            sampler.start()
            results = {}
            for stage in ("quality", "stability"):
                path = temporary_root / f"{stage}.json"
                report["checks"][stage] = "failed"
                code, stage_report = run_worker(
                    stage,
                    path,
                    deadline,
                    args.gpu_index,
                )
                results[stage] = stage_report
                report[stage] = results[stage]
                report["checks"][stage] = "passed" if code == 0 else "failed"
                if stage == "quality" and code:
                    report["failures"].extend(
                        results[stage].get("gate_failures") or ["quality_gate_failed"]
                    )
                    break
            if "stability" in results:
                report["failures"].extend(report_failures(results["quality"], results["stability"]))

            # Versions/model identity are checked again, but inference is not repeated.
            report["checks"]["preflight"] = "failed"
            postflight_code, after = run_worker(
                "preflight",
                temporary_root / "postflight.json",
                deadline,
                args.gpu_index,
            )
            if postflight_code:
                raise ValueError(str(after.get("failure") or "postflight_failed"))
            if (
                after["source_commit_sha"] != report["preflight"]["source_commit_sha"]
                or after["model"] != report["preflight"]["model"]
            ):
                report["failures"].append("identity_changed_during_gate")
            else:
                report["checks"]["preflight"] = "passed"
    except BackendError as error:
        report["failures"].append(error.reason)
    except (TimeoutError, subprocess.TimeoutExpired):
        report["failures"].append("gate_deadline_exceeded")
    except (OSError, ValueError, KeyError, httpx.HTTPError, subprocess.SubprocessError) as error:
        report["failures"].append(
            str(error) if isinstance(error, ValueError) else type(error).__name__
        )
    finally:
        stop.set()
        if sampler:
            sampler.join(timeout=max(0.0, min(4.0, deadline - time.monotonic())))
        report["failures"].extend(sampler_errors)
        preflight_evidence = report.get("preflight")
        gpu_evidence = (
            preflight_evidence.get("gpu") if isinstance(preflight_evidence, dict) else None
        )
        baseline_used_mib = gpu_evidence.get("used_mib") if isinstance(gpu_evidence, dict) else None
        report["gpu_memory"] = {
            "scope": "whole_selected_gpu_not_model_process",
            "baseline_used_mib": (
                int(baseline_used_mib) if baseline_used_mib is not None else None
            ),
            "peak_used_mib": max(memory_samples) if memory_samples else None,
            "sample_count": len(memory_samples),
        }
        for stage in ("preflight", "cpu", "quality", "stability"):
            report["checks"].setdefault(stage, "not_run")
        if not memory_samples:
            report["failures"].append("gpu_memory_not_measured")
        report["duration_seconds"] = round(time.monotonic() - started, 3)
        if report["duration_seconds"] > args.timeout_seconds:
            report["failures"].append("gate_deadline_exceeded")
        report["finished_at_utc"] = datetime.now(UTC).isoformat()
        report["ok"] = not report["failures"] and all(
            value == "passed" for value in report["checks"].values()
        )
        write_report(args.output, report)
    print(
        json.dumps(
            {"ok": report["ok"], "failures": report["failures"], "output": str(args.output)},
            ensure_ascii=False,
        )
    )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
