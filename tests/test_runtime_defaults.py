import tomllib
from pathlib import Path

import pytest
import yaml

from app.core import config as config_module

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIMARY_MODEL = "hf.co/unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M"


@pytest.fixture(autouse=True)
def clear_runtime_profile_cache():
    config_module.get_runtime_profile.cache_clear()
    yield
    config_module.get_runtime_profile.cache_clear()


def test_supported_python_and_dependencies_are_explicit():
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    lock = tomllib.loads((PROJECT_ROOT / "uv.lock").read_text(encoding="utf-8"))
    locked_versions = {package["name"]: package["version"] for package in lock["package"]}

    build_requires = pyproject["build-system"]["requires"]
    assert {requirement.split("==")[0] for requirement in build_requires} == {
        "setuptools",
        "wheel",
    }
    assert all("==" in requirement for requirement in build_requires)

    assert project["requires-python"] == ">=3.11,<3.13"
    assert lock["requires-python"] == ">=3.11, <3.13"

    pinned = {}
    for requirement in project["dependencies"]:
        assert "==" in requirement, requirement
        name, version = requirement.split("==", 1)
        pinned[name.split("[")[0].lower()] = version
    for name in ("typer", "fastapi", "starlette", "pydantic", "pydantic-settings"):
        assert name in pinned
        assert locked_versions[name] == pinned[name]

    assert "click" not in pinned
    assert "click" in locked_versions


def test_structured_output_budget_fits_a_full_task_plan():
    profile = config_module.get_runtime_profile()

    assert profile.num_predict >= 1024
    assert profile.num_ctx >= profile.num_predict * 2


def test_runtime_profile_uses_the_verified_model_without_a_required_fallback(monkeypatch):
    monkeypatch.delenv("LOCALSCRIPT_PRIMARY_MODEL", raising=False)
    monkeypatch.delenv("LOCALSCRIPT_FALLBACK_MODEL", raising=False)

    profile = config_module.get_runtime_profile()

    assert profile.name == "local"
    assert profile.model == PRIMARY_MODEL
    assert profile.fallback_model == profile.model


def test_runtime_profile_applies_model_environment_overrides(monkeypatch):
    monkeypatch.setenv("LOCALSCRIPT_PRIMARY_MODEL", "custom-primary")
    monkeypatch.setenv("LOCALSCRIPT_FALLBACK_MODEL", "custom-fallback")

    profile = config_module.get_runtime_profile()

    assert profile.model == "custom-primary"
    assert profile.fallback_model == "custom-fallback"


def test_compose_and_startup_scripts_defer_to_the_runtime_profile():
    env_example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    compose = yaml.safe_load((PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    service_environment = compose["services"]["localscript"]["environment"]

    assert f"# LOCALSCRIPT_PRIMARY_MODEL={PRIMARY_MODEL}" in env_example
    assert "LOCALSCRIPT_PRIMARY_MODEL" not in service_environment
    assert "LOCALSCRIPT_FALLBACK_MODEL" not in service_environment
    assert service_environment["LOCALSCRIPT_OLLAMA_HOST"] == "http://ollama:11434"
    for script_name in ("start.sh", "docker_entrypoint.sh", "preflight_judge.sh"):
        script = (PROJECT_ROOT / "scripts" / script_name).read_text(encoding="utf-8")
        assert PRIMARY_MODEL not in script
        assert "get_runtime_profile" in script


def test_start_defaults_to_supported_python_minors():
    script = (PROJECT_ROOT / "scripts" / "start.sh").read_text(encoding="utf-8")

    assert 'SUPPORTED_PYTHON_MIN_MINOR="${LOCALSCRIPT_PYTHON_MIN_MINOR:-11}"' in script
    assert 'SUPPORTED_PYTHON_MAX_MINOR="${LOCALSCRIPT_PYTHON_MAX_MINOR:-12}"' in script


def test_startup_probes_have_per_request_and_wall_clock_timeouts():
    for script_name in ("start.sh", "docker_entrypoint.sh"):
        script = (PROJECT_ROOT / "scripts" / script_name).read_text(encoding="utf-8")
        assert "--connect-timeout" in script
        assert "--max-time" in script
        assert "SECONDS=0" in script


def test_model_setup_targets_the_effective_ollama_host():
    script = (PROJECT_ROOT / "scripts" / "setup_model.sh").read_text(encoding="utf-8")

    assert "profile.model" in script
    assert "profile.ollama_host" in script
    assert 'OLLAMA_HOST="${OLLAMA_ENDPOINT}" ollama pull "${MODEL}"' in script


def test_primary_install_paths_consume_the_lock():
    makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "$(UV) sync --frozen --all-extras" in makefile
    assert "uv sync --frozen --no-editable" in dockerfile
    assert "pip install ." not in dockerfile


def test_batch_size_admits_a_whole_prompt():
    profile = config_module.get_runtime_profile()

    assert profile.batch >= 512


def test_dead_candidate_chain_settings_are_not_part_of_effective_config():
    fields = config_module.RuntimeProfile.model_fields

    assert "max_candidates" not in fields
    assert "model_chain_rounds" not in fields
    assert "primary_launch" not in fields


def test_compose_pins_verified_ollama_and_persists_separate_state():
    compose = yaml.safe_load((PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))

    assert compose["services"]["ollama"]["image"] == "ollama/ollama:0.33.3"
    assert compose["services"]["ollama"]["environment"]["OLLAMA_NO_CLOUD"] == "1"
    assert compose["services"]["ollama"]["volumes"] == ["ollama_models:/root/.ollama"]
    assert compose["services"]["localscript"]["volumes"] == [
        "localscript_state:/var/lib/localscript"
    ]
    assert set(compose["volumes"]) == {"localscript_state", "ollama_models"}


def test_container_image_does_not_copy_the_build_workspace():
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "COPY --from=build /workspace /workspace" not in dockerfile
    assert "COPY --from=build /opt/venv /opt/venv" in dockerfile
    assert "LOCALSCRIPT_STATE_DIR=/var/lib/localscript" in dockerfile
