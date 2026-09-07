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

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.benchmarks import run_quality_benchmark, run_stability_benchmark
from app.core.config import get_runtime_profile
from app.evaluation.integrity import run_integrity_check
from app.generation.backend_errors import BackendError
from app.validation.runtime import find_lua_binary, find_luac_binary, runtime_version

LIVE_GATE_BUDGET_SECONDS = 20 * 60


def capture(command, *, timeout=10):
    return subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, timeout=timeout, check=True
    ).stdout.strip()


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


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
        version = version_response.json()["version"]
        from app.generation.ollama import OllamaBackend

        with OllamaBackend(profile) as backend:
            resolved = backend.resolve_model()
        if not resolved.digest:
            raise ValueError("model_digest_missing")
    integrity = run_integrity_check()
    if not integrity.get("ok"):
        raise ValueError("corpus_integrity_failed")
    return {
        "source_commit_sha": sha,
        "dirty": False,
        "os": platform.system(),
        "python": platform.python_version(),
        "lua": runtime_version(lua),
        "luac": runtime_version(luac),
        "ollama": version,
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
    parser.add_argument("--worker", choices=("quality", "stability"), help=argparse.SUPPRESS)
    parser.add_argument(
        "--mode",
        choices=("local", "competition"),
        default="local",
        help="competition оставлен как совместимый alias",
    )
    args = parser.parse_args(argv)
    if args.worker:
        report = (
            run_quality_benchmark(mode="competition")
            if args.worker == "quality"
            else run_stability_benchmark()
        )
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
        report["preflight"] = preflight(args.gpu_index)
        report["checks"]["preflight"] = "passed"
        cpu_commands = [
            [sys.executable, "-m", "ruff", "check", "."],
            [sys.executable, "-m", "ruff", "format", "--check", "."],
            [sys.executable, "-m", "mypy", "--strict", "app"],
            [sys.executable, "-m", "pytest", "-q", "-m", "unit"],
        ]
        for command in cpu_commands:
            if run_bounded(command, deadline):
                raise ValueError("cpu_checks_failed")
        report["checks"]["cpu"] = "passed"
        sampler = threading.Thread(target=sample_memory, daemon=True)
        sampler.start()
        with tempfile.TemporaryDirectory(prefix="localscript-live-") as temporary:
            results = {}
            for stage in ("quality", "stability"):
                path = Path(temporary) / f"{stage}.json"
                code = run_bounded(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        stage,
                        "--output",
                        str(path),
                    ],
                    deadline,
                )
                if not path.is_file():
                    raise ValueError(f"{stage}_report_missing")
                results[stage] = json.loads(path.read_text(encoding="utf-8"))
                results[stage].pop("host", None)
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
        after = preflight(args.gpu_index)
        if (
            after["source_commit_sha"] != report["preflight"]["source_commit_sha"]
            or after["model"] != report["preflight"]["model"]
        ):
            report["failures"].append("identity_changed_during_gate")
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
            sampler.join(timeout=4)
        report["failures"].extend(sampler_errors)
        report["gpu_memory"] = {
            "scope": "whole_selected_gpu_not_model_process",
            "baseline_used_mib": int(report["preflight"]["gpu"]["used_mib"])
            if "preflight" in report
            else None,
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
