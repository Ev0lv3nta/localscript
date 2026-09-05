import tomllib
from pathlib import Path

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

    # Инвариант здесь — «всё закреплено точной версией и совпадает с локом», а не конкретные
    # номера: дублировать их в тесте значило бы править его при каждом обновлении зависимостей.
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

    # Typer тянет click транзитивно; прямой зависимостью он быть не должен, иначе две
    # несогласованные версии CLI-слоя разъедутся молча.
    assert "click" not in pinned
    assert "click" in locked_versions


def test_structured_output_budget_fits_a_full_task_plan():
    """Планировщик возвращает JSON-план целиком, а не первые несколько сотен токенов.

    С прежним бюджетом в 256 токенов план обрывался на полуслове и приходил как невалидный
    структурированный ответ, поэтому нижняя граница здесь — часть контракта, а не вкусовщина.
    """
    profile = config_module.get_runtime_profile()

    assert profile.num_predict >= 1024
    assert profile.num_ctx >= profile.num_predict * 2
    config_module.get_runtime_profile.cache_clear()


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


def test_batch_size_admits_a_whole_prompt():
    """`num_batch: 1` означает промпт в один токен и валит свежий llama.cpp жёстким assert'ом.

    Старый рантайм это молча переживал, поэтому настройка дожила до сюда из архитектуры,
    где модель получала короткий сниппет.
    """
    profile = config_module.get_runtime_profile()

    assert profile.batch >= 512
    config_module.get_runtime_profile.cache_clear()
