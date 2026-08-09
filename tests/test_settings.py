import json

import pytest
from pydantic import ValidationError

from app.api.schemas import (
    MAX_SCHEMA_CLARIFICATION_ANSWER_CHARS,
    MAX_SCHEMA_CODE_CHARS,
    MAX_SCHEMA_FEEDBACK_CHARS,
    MAX_SCHEMA_PROMPT_CHARS,
    MAX_SCHEMA_SESSION_ID_CHARS,
    GenerateRequest,
    GenerateRichRequest,
    ValidateRequest,
)
from app.core import config as config_module


@pytest.fixture(autouse=True)
def clear_settings_cache():
    config_module.get_runtime_profile.cache_clear()
    yield
    config_module.get_runtime_profile.cache_clear()


def _use_profile(monkeypatch, tmp_path, text):
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(config_module, "get_profile_path", lambda profile_name=None: profile_path)
    return profile_path


def test_runtime_profile_is_an_immutable_validated_snapshot():
    profile = config_module.get_runtime_profile()

    with pytest.raises(ValidationError):
        profile.num_ctx = 1


def test_profile_rejects_unknown_keys_without_echoing_values(monkeypatch, tmp_path):
    secret = "do-not-echo-this-value"
    _use_profile(
        monkeypatch,
        tmp_path,
        "name: competition\nmodel: qwen\nfallback_model: fallback\nunknown_key: {0}\n".format(secret),
    )

    with pytest.raises(config_module.ConfigurationError) as raised:
        config_module.get_runtime_profile()

    assert raised.value.code == "configuration_profile_invalid"
    assert "unknown_key:extra_forbidden" in str(raised.value)
    assert secret not in str(raised.value)


@pytest.mark.parametrize(
    "profile_value,error_location",
    [
        ("num_ctx: 0", "num_ctx:greater_than"),
        ('think: "false"', "think:bool_type"),
    ],
)
def test_profile_enforces_positive_bounds_and_strict_yaml_types(
    monkeypatch,
    tmp_path,
    profile_value,
    error_location,
):
    _use_profile(monkeypatch, tmp_path, "name: competition\n{0}\n".format(profile_value))

    with pytest.raises(config_module.ConfigurationError, match=error_location):
        config_module.get_runtime_profile()


def test_environment_is_the_highest_precedence_layer(monkeypatch, tmp_path):
    lock_path = tmp_path / "runtime.lock.json"
    lock_path.write_text(
        json.dumps(
            {
                "locked": True,
                "profile": "competition",
                "selected_model": "locked-primary",
                "fallback_model": "locked-fallback",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("LOCALSCRIPT_RUNTIME_LOCK_PATH", str(lock_path))
    monkeypatch.setenv("LOCALSCRIPT_USE_RUNTIME_LOCK", "true")
    monkeypatch.setenv("LOCALSCRIPT_PRIMARY_MODEL", "environment-primary")
    monkeypatch.setenv("LOCALSCRIPT_FALLBACK_MODEL", "environment-fallback")
    monkeypatch.setenv("LOCALSCRIPT_NUM_CTX", "8192")

    profile = config_module.get_runtime_profile()

    assert profile.model == "environment-primary"
    assert profile.fallback_model == "environment-fallback"
    assert profile.num_ctx == 8192


def test_runtime_lock_overrides_profile_only_after_explicit_opt_in(monkeypatch, tmp_path):
    lock_path = tmp_path / "runtime.lock.json"
    lock_path.write_text(
        json.dumps(
            {
                "locked": True,
                "profile": "competition",
                "selected_model": "locked-primary",
                "fallback_model": "locked-fallback",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("LOCALSCRIPT_RUNTIME_LOCK_PATH", str(lock_path))
    monkeypatch.delenv("LOCALSCRIPT_USE_RUNTIME_LOCK", raising=False)

    assert config_module.get_runtime_profile().model != "locked-primary"

    config_module.get_runtime_profile.cache_clear()
    monkeypatch.setenv("LOCALSCRIPT_USE_RUNTIME_LOCK", "1")
    assert config_module.get_runtime_profile().model == "locked-primary"


def test_unlocked_runtime_artifact_does_not_require_model_selection(monkeypatch, tmp_path):
    lock_path = tmp_path / "runtime.lock.json"
    lock_path.write_text(json.dumps({"locked": False}), encoding="utf-8")
    monkeypatch.setenv("LOCALSCRIPT_RUNTIME_LOCK_PATH", str(lock_path))
    monkeypatch.setenv("LOCALSCRIPT_USE_RUNTIME_LOCK", "true")

    assert config_module.get_runtime_profile().name == "competition"


def test_invalid_environment_boolean_is_a_safe_configuration_error(monkeypatch):
    monkeypatch.setenv("LOCALSCRIPT_USE_RUNTIME_LOCK", "yes-please")

    with pytest.raises(config_module.ConfigurationError) as raised:
        config_module.get_runtime_profile()

    assert raised.value.code == "configuration_environment_invalid"
    assert "yes-please" not in str(raised.value)


def test_public_request_models_keep_extra_field_compatibility():
    request = GenerateRequest.model_validate({"prompt": "return 1", "future": "field"})

    assert request.prompt == "return 1"
    assert not hasattr(request, "future")


def test_public_text_fields_publish_explicit_positive_schema_limits():
    limits = [
        (
            GenerateRequest.model_json_schema()["properties"]["prompt"]["maxLength"],
            MAX_SCHEMA_PROMPT_CHARS,
        ),
        (
            GenerateRequest.model_json_schema()["properties"]["session_id"]["anyOf"][0]["maxLength"],
            MAX_SCHEMA_SESSION_ID_CHARS,
        ),
        (
            GenerateRichRequest.model_json_schema()["properties"]["feedback"]["anyOf"][0]["maxLength"],
            MAX_SCHEMA_FEEDBACK_CHARS,
        ),
        (
            GenerateRichRequest.model_json_schema()["properties"]["clarification_answer"]["anyOf"][0]["maxLength"],
            MAX_SCHEMA_CLARIFICATION_ANSWER_CHARS,
        ),
        (
            ValidateRequest.model_json_schema()["properties"]["code"]["maxLength"],
            MAX_SCHEMA_CODE_CHARS,
        ),
    ]

    assert limits
    assert all(actual == expected and expected > 0 for actual, expected in limits)
