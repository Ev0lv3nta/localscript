#!/usr/bin/env python3
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFERRED_PYTHON = ROOT / ".venv" / "bin" / "python"
PREFERRED_VENV = PREFERRED_PYTHON.parent.parent.resolve()
if PREFERRED_PYTHON.exists() and Path(sys.prefix).resolve() != PREFERRED_VENV:
    os.execv(str(PREFERRED_PYTHON), [str(PREFERRED_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def run_json_command(command, cwd, extra_env=None):
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    stdout = completed.stdout.strip()
    try:
        payload = json.loads(stdout or "{}")
    except json.JSONDecodeError:
        payload = {
            "ok": False,
            "status": "error",
            "reason": "invalid_json_output",
            "raw_stdout": stdout,
            "stderr": completed.stderr.strip(),
        }
    return completed.returncode, payload, completed.stderr.strip()


def run_text_command(command, cwd, extra_env=None):
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def parse_mode(argv):
    if "--mode" in argv:
        index = argv.index("--mode")
        try:
            mode = argv[index + 1]
        except IndexError:
            raise SystemExit("--mode requires a value")
    else:
        mode = "competition"
    if mode not in {"dev", "competition"}:
        raise SystemExit("--mode must be one of: dev, competition")
    return mode


def main():
    root = ROOT
    mode = parse_mode(sys.argv[1:])
    preferred_python = root / ".venv" / "bin" / "python"
    python_bin = preferred_python if preferred_python.exists() else Path(sys.executable)

    pytest_code, pytest_stdout, pytest_stderr = run_text_command(
        [str(python_bin), "-m", "pytest", "-q"],
        cwd=root,
    )
    quality_args = [str(root / "scripts" / "bench_quality.py")]
    if mode == "competition":
        quality_args.append("--strict")
    quality_code, quality_report, quality_stderr = run_json_command(
        quality_args,
        cwd=root,
    )
    doctor_code, doctor_report, doctor_stderr = run_json_command(
        [str(python_bin), "-m", "app.cli.main", "doctor", "--judge"],
        cwd=root,
        extra_env={"LOCALSCRIPT_IGNORE_LOCK": "1"},
    )
    smoke_code, smoke_report, smoke_stderr = run_json_command(
        [str(root / "scripts" / "judge_smoke.sh")],
        cwd=root,
        extra_env={"LOCALSCRIPT_SKIP_INSTALL": "1"},
    )
    latency_code, latency_report, latency_stderr = run_json_command(
        [str(root / "scripts" / "bench_latency.py")],
        cwd=root,
    )

    doctor_quality_report = doctor_report.get("quality_report", {})
    vram_report = doctor_report.get("vram_report", {})

    doctor_required = mode == "competition"
    report = {
        "mode": mode,
        "pytest_report": {
            "ok": pytest_code == 0,
            "stdout": pytest_stdout,
        },
        "quality_report": quality_report,
        "doctor_report": doctor_report,
        "smoke_report": smoke_report,
        "latency_report": latency_report,
        "runtime_snapshot_path": doctor_report.get("runtime_snapshot_path"),
        "selected_model": doctor_report.get("selected_model"),
        "doctor_quality_report": doctor_quality_report,
        "vram_report": vram_report,
        "ok": (
            pytest_code == 0
            and quality_code == 0
            and quality_report.get("ok") is True
            and quality_report.get("backend_type") == "live_ollama"
            and quality_report.get("public_gold", {}).get("ok") is True
            and quality_report.get("model_backed_eval", {}).get("ok") is True
            and quality_report.get("multilingual_eval", {}).get("ok") is True
            and quality_report.get("ambiguity_eval", {}).get("ok") is True
            and quality_report.get("clarification_eval", {}).get("ok") is True
            and quality_report.get("composition_eval", {}).get("ok") is True
            and quality_report.get("regression_eval", {}).get("ok") is True
            and quality_report.get("large_context_eval", {}).get("ok") is True
            and quality_report.get("adversarial_ok") is True
            and smoke_code == 0
            and smoke_report.get("ok") is True
            and latency_code == 0
            and latency_report.get("ok") is True
            and (
                not doctor_required
                or (
                    doctor_code == 0
                    and doctor_report.get("ok") is True
                    and vram_report.get("status") == "ok"
                )
            )
        ),
    }
    if not report["ok"]:
        report["errors"] = {
            "pytest_stderr": pytest_stderr,
            "quality_stderr": quality_stderr,
            "doctor_stderr": doctor_stderr,
            "smoke_stderr": smoke_stderr,
            "latency_stderr": latency_stderr,
        }

    print(json.dumps(report, ensure_ascii=False))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
