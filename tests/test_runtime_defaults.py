import tomllib
from pathlib import Path

import yaml

from app.core import config as config_module


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIMARY_MODEL = "qwen3:8b-q4_K_M"
FALLBACK_MODEL = "qwen3:4b-instruct-2507-q4_K_M"


def test_supported_python_and_dependencies_are_explicit():
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert project["requires-python"] == ">=3.11,<3.13"
    assert "typer==0.27.0" in project["dependencies"]
    assert not any(dependency.lower().startswith("click") for dependency in project["dependencies"])


def test_runtime_profile_has_a_distinct_fallback(monkeypatch):
    monkeypatch.delenv("LOCALSCRIPT_PRIMARY_MODEL", raising=False)
    monkeypatch.delenv("LOCALSCRIPT_FALLBACK_MODEL", raising=False)
    config_module.get_runtime_profile.cache_clear()

    profile = config_module.get_runtime_profile()

    assert profile.model == PRIMARY_MODEL
    assert profile.fallback_model == FALLBACK_MODEL
    assert profile.fallback_model != profile.model
    config_module.get_runtime_profile.cache_clear()


def test_documented_and_script_defaults_match_runtime_profile():
    env_example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    rules = yaml.safe_load((PROJECT_ROOT / "kb" / "rules.yaml").read_text(encoding="utf-8"))
    shell_defaults = [
        PROJECT_ROOT / "scripts" / "bench_vram.sh",
        PROJECT_ROOT / "scripts" / "docker_entrypoint.sh",
        PROJECT_ROOT / "scripts" / "judge_up.sh",
        PROJECT_ROOT / "scripts" / "preflight_judge.sh",
    ]

    assert f"LOCALSCRIPT_PRIMARY_MODEL={PRIMARY_MODEL}" in env_example
    assert f"LOCALSCRIPT_FALLBACK_MODEL={FALLBACK_MODEL}" in env_example
    assert rules["runtime_constraints"]["local_ollama_profile"] == {
        "model": PRIMARY_MODEL,
        "fallback_model": FALLBACK_MODEL,
        "num_ctx": 4096,
        "num_predict": 256,
        "batch": 1,
        "parallel": 1,
    }
    for script in shell_defaults:
        assert FALLBACK_MODEL in script.read_text(encoding="utf-8")


def test_judge_up_defaults_to_supported_python_minors():
    script = (PROJECT_ROOT / "scripts" / "judge_up.sh").read_text(encoding="utf-8")

    assert 'SUPPORTED_PYTHON_MIN_MINOR="${LOCALSCRIPT_PYTHON_MIN_MINOR:-11}"' in script
    assert 'SUPPORTED_PYTHON_MAX_MINOR="${LOCALSCRIPT_PYTHON_MAX_MINOR:-12}"' in script
