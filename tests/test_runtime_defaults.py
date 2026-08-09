import tomllib
from pathlib import Path

import yaml

from app.core import config as config_module
from app.core.resources import read_resource_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIMARY_MODEL = "qwen3:8b-q4_K_M"
FALLBACK_MODEL = "qwen3:4b-instruct-2507-q4_K_M"


def test_supported_python_and_dependencies_are_explicit():
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    lock = tomllib.loads((PROJECT_ROOT / "uv.lock").read_text(encoding="utf-8"))
    locked_versions = {package["name"]: package["version"] for package in lock["package"]}

    assert project["requires-python"] == ">=3.11,<3.13"
    assert pyproject["build-system"]["requires"] == ["setuptools==80.9.0", "wheel==0.45.1"]
    assert "typer==0.27.0" in project["dependencies"]
    assert "fastapi==0.140.0" in project["dependencies"]
    assert "pydantic==2.13.4" in project["dependencies"]
    assert "pydantic-settings==2.14.2" in project["dependencies"]
    assert not any(dependency.lower().startswith("click") for dependency in project["dependencies"])
    assert lock["requires-python"] == ">=3.11, <3.13"
    assert locked_versions["typer"] == "0.27.0"
    assert locked_versions["click"] == "8.4.2"
    assert locked_versions["fastapi"] == "0.140.0"
    assert locked_versions["pydantic"] == "2.13.4"
    assert locked_versions["pydantic-settings"] == "2.14.2"


def test_runtime_profile_has_a_distinct_fallback(monkeypatch):
    monkeypatch.delenv("LOCALSCRIPT_PRIMARY_MODEL", raising=False)
    monkeypatch.delenv("LOCALSCRIPT_FALLBACK_MODEL", raising=False)
    config_module.get_runtime_profile.cache_clear()

    profile = config_module.get_runtime_profile()

    assert profile.model == PRIMARY_MODEL
    assert profile.fallback_model == FALLBACK_MODEL
    assert profile.fallback_model != profile.model
    config_module.get_runtime_profile.cache_clear()


def test_runtime_profile_applies_model_environment_overrides(monkeypatch):
    monkeypatch.setenv("LOCALSCRIPT_PRIMARY_MODEL", "custom-primary")
    monkeypatch.setenv("LOCALSCRIPT_FALLBACK_MODEL", "custom-fallback")
    config_module.get_runtime_profile.cache_clear()

    profile = config_module.get_runtime_profile()

    assert profile.model == "custom-primary"
    assert profile.fallback_model == "custom-fallback"
    config_module.get_runtime_profile.cache_clear()


def test_documented_and_script_defaults_match_runtime_profile():
    env_example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    rules = yaml.safe_load(read_resource_text("kb/rules.yaml"))
    shell_defaults = {
        PROJECT_ROOT
        / "scripts"
        / "docker_entrypoint.sh": f'FALLBACK_MODEL="${{LOCALSCRIPT_FALLBACK_MODEL:-{FALLBACK_MODEL}}}"',
        PROJECT_ROOT
        / "scripts"
        / "judge_up.sh": f'FALLBACK_MODEL="${{LOCALSCRIPT_FALLBACK_MODEL:-{FALLBACK_MODEL}}}"',
        PROJECT_ROOT
        / "scripts"
        / "preflight_judge.sh": f'FALLBACK_MODEL="${{LOCALSCRIPT_FALLBACK_MODEL:-{FALLBACK_MODEL}}}"',
    }

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
    assert (
        f'FALLBACK_MODEL="${{2:-{FALLBACK_MODEL}}}"'
        in read_resource_text("scripts/bench_vram.sh").splitlines()
    )
    for script, expected_assignment in shell_defaults.items():
        assert expected_assignment in script.read_text(encoding="utf-8").splitlines()


def test_judge_up_defaults_to_supported_python_minors():
    script = (PROJECT_ROOT / "scripts" / "judge_up.sh").read_text(encoding="utf-8")

    assert 'SUPPORTED_PYTHON_MIN_MINOR="${LOCALSCRIPT_PYTHON_MIN_MINOR:-11}"' in script
    assert 'SUPPORTED_PYTHON_MAX_MINOR="${LOCALSCRIPT_PYTHON_MAX_MINOR:-12}"' in script


def test_primary_install_paths_consume_the_lock():
    makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "$(UV) sync --frozen --all-extras" in makefile
    assert "uv sync --frozen --no-editable" in dockerfile
    assert "pip install ." not in dockerfile
