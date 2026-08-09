#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from email.parser import Parser
from pathlib import Path

WHEEL_REQUIRED = {
    "app/resources/config/profiles/competition.yaml",
    "app/resources/kb/rules.yaml",
    "app/resources/kb/examples.yaml",
    "app/resources/kb/templates.yaml",
    "app/resources/kb/critic_rules.yaml",
    "app/resources/evals/manifest.json",
    "app/resources/evals/public/v1.jsonl",
    "app/resources/evals/regression/adversarial_eval.jsonl",
    "app/resources/evals/regression/ambiguity_eval.jsonl",
    "app/resources/evals/regression/clarification_eval.jsonl",
    "app/resources/evals/regression/composition_eval.jsonl",
    "app/resources/evals/regression/large_context_eval.jsonl",
    "app/resources/evals/regression/model_backed_eval.jsonl",
    "app/resources/evals/regression/multilingual_eval.jsonl",
    "app/resources/evals/regression/public_gold.jsonl",
    "app/resources/evals/regression/regression_eval.jsonl",
    "app/resources/evals/regression/showcase_eval.jsonl",
    "app/resources/evals/regression/stress_eval.jsonl",
    "app/resources/scripts/bench_vram.sh",
    "app/ui/static/index.html",
    "app/ui/static/app.js",
    "app/ui/static/styles.css",
}

SDIST_REQUIRED_SUFFIXES = WHEEL_REQUIRED | {
    ".env.example",
    "Dockerfile",
    "LICENSE",
    "Makefile",
    "MANIFEST.in",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "docker-compose.yml",
    "pyproject.toml",
    "uv.lock",
    "scripts/bootstrap_lua54.sh",
    "scripts/bench_ablation.py",
    "scripts/bench_repeated.py",
    "scripts/release_gate.py",
    "tests/conftest.py",
    "tests/contracts/test_outcomes_contract.py",
    "tests/characterization/test_generation_fail_closed.py",
    "third_party/lua/lua-5.4.6.tar.gz",
}


def _run(command, cwd=None, env=None):
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _assert_wheel_manifest(wheel_path):
    with zipfile.ZipFile(wheel_path) as archive:
        names = set(archive.namelist())
        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata_names) != 1:
            raise SystemExit("wheel_invalid_metadata_count::{0}".format(len(metadata_names)))
        metadata = archive.read(metadata_names[0]).decode("utf-8")
        parsed_metadata = Parser().parsestr(metadata)
    missing = sorted(WHEEL_REQUIRED - names)
    if missing:
        raise SystemExit("wheel_missing_files::{0}".format(",".join(missing)))
    for license_name in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
        if not any(name.endswith(".dist-info/licenses/" + license_name) for name in names):
            raise SystemExit("wheel_missing_license::{0}".format(license_name))
    python_constraints = {
        constraint.strip()
        for constraint in (parsed_metadata.get("Requires-Python") or "").split(",")
        if constraint.strip()
    }
    if python_constraints != {">=3.11", "<3.13"}:
        raise SystemExit("wheel_invalid_requires_python")
    if parsed_metadata.get("License-Expression") != "MIT":
        raise SystemExit("wheel_invalid_license_expression")


def _assert_sdist_manifest(sdist_path):
    with tarfile.open(sdist_path, "r:gz") as archive:
        names = archive.getnames()
    missing = sorted(
        suffix
        for suffix in SDIST_REQUIRED_SUFFIXES
        if not any(name == suffix or name.endswith("/" + suffix) for name in names)
    )
    if missing:
        raise SystemExit("sdist_missing_files::{0}".format(",".join(missing)))


def _installed_wheel_smoke(wheel_path):
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("uv_not_found")

    with tempfile.TemporaryDirectory(prefix="localscript-wheel-smoke-") as temp_dir:
        temp_root = Path(temp_dir)
        venv = temp_root / "venv"
        empty_cwd = temp_root / "empty"
        state_dir = temp_root / "state"
        empty_cwd.mkdir()

        _run([uv, "venv", "--python", sys.executable, str(venv)])
        _run([uv, "pip", "install", "--python", str(venv / "bin" / "python"), str(wheel_path)])

        smoke_env = os.environ.copy()
        smoke_env.pop("PYTHONPATH", None)
        smoke_env.update(
            {
                "LOCALSCRIPT_STATE_DIR": str(state_dir),
                "LOCALSCRIPT_UI_ENABLED": "1",
                "VIRTUAL_ENV": str(venv),
                "PATH": "{0}{1}{2}".format(venv / "bin", os.pathsep, smoke_env.get("PATH", "")),
            }
        )
        _run([str(venv / "bin" / "localscript"), "--help"], cwd=empty_cwd, env=smoke_env)
        _run([str(venv / "bin" / "localscript"), "doctor"], cwd=empty_cwd, env=smoke_env)
        _run(
            [
                str(venv / "bin" / "python"),
                "-c",
                (
                    "import os;"
                    "from pathlib import Path;"
                    "from app.core.config import get_runtime_profile;"
                    "from app.core.kb import load_critic_rules,load_examples,load_rules;"
                    "from app.core.traces import TraceStore;"
                    "from app.main import create_app;"
                    "p=get_runtime_profile();"
                    "assert p.name=='competition';"
                    "assert load_rules() and load_examples() and load_critic_rules();"
                    "assert TraceStore().root.is_relative_to(Path(os.environ['LOCALSCRIPT_STATE_DIR']).resolve());"
                    "assert create_app().state.ui_enabled"
                ),
            ],
            cwd=empty_cwd,
            env=smoke_env,
        )


def main():
    parser = argparse.ArgumentParser(description="Validate LocalScript wheel and sdist contents.")
    parser.add_argument("--dist-dir", type=Path, required=True)
    args = parser.parse_args()

    dist_dir = args.dist_dir.resolve()
    wheels = sorted(dist_dir.glob("*.whl"))
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    if len(wheels) != 1:
        raise SystemExit("expected_one_wheel::found={0}".format(len(wheels)))
    if len(sdists) != 1:
        raise SystemExit("expected_one_sdist::found={0}".format(len(sdists)))
    wheel_path = wheels[0]
    sdist_path = sdists[0]

    _assert_wheel_manifest(wheel_path)
    _assert_sdist_manifest(sdist_path)
    _installed_wheel_smoke(wheel_path)
    print("package_artifacts_ok")


if __name__ == "__main__":
    main()
