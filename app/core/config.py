from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

import yaml
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.resources import get_resource

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE = "competition"

PositiveInt = Annotated[int, Field(gt=0)]


class ConfigurationError(RuntimeError):
    """Safe, typed startup failure for invalid runtime configuration."""

    def __init__(self, code: str, details: tuple[str, ...] = ()):
        self.code = code
        self.details = details
        suffix = "::{0}".format(",".join(details)) if details else ""
        super().__init__("{0}{1}".format(code, suffix))


def _strict_environment_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true"}:
            return True
        if normalized in {"0", "false"}:
            return False
    raise ValueError("expected one of: 0, 1, false, true")


EnvironmentBool = Annotated[bool, BeforeValidator(_strict_environment_bool)]


class RuntimeProfile(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
    )

    name: str = Field(default=DEFAULT_PROFILE, min_length=1)
    model: str = Field(default="qwen3:8b-q4_K_M", min_length=1)
    fallback_model: str = Field(
        default="qwen3:4b-instruct-2507-q4_K_M",
        min_length=1,
    )
    think: bool = False
    stream: bool = False
    num_ctx: PositiveInt = 4096
    num_predict: PositiveInt = 256
    batch: PositiveInt = 1
    parallel: PositiveInt = 1
    max_candidates: PositiveInt = 2
    model_chain_rounds: PositiveInt = 1
    max_repair_rounds: PositiveInt = 2
    runtime_lua: str = Field(default="lua5.4_subprocess", min_length=1)
    primary_launch: str = Field(default="./scripts/judge_up.sh", min_length=1)
    ollama_host: str = Field(default="http://127.0.0.1:11434", min_length=1)
    request_timeout_seconds: PositiveInt = 45
    max_request_body_bytes: PositiveInt = 131072
    max_prompt_chars: PositiveInt = 6000
    max_context_bytes: PositiveInt = 64000
    max_context_depth: PositiveInt = 16
    max_context_nodes: PositiveInt = 2000


class _EnvironmentSettings(BaseSettings):
    """Only the environment layer; YAML and runtime lock are loaded separately."""

    model_config = SettingsConfigDict(
        env_prefix="LOCALSCRIPT_",
        case_sensitive=False,
        extra="ignore",
    )

    profile: str | None = None
    use_runtime_lock: EnvironmentBool = False
    model: str | None = Field(
        default=None,
        validation_alias="LOCALSCRIPT_PRIMARY_MODEL",
    )
    fallback_model: str | None = None
    think: EnvironmentBool | None = None
    stream: EnvironmentBool | None = None
    num_ctx: PositiveInt | None = None
    num_predict: PositiveInt | None = None
    batch: PositiveInt | None = None
    parallel: PositiveInt | None = None
    max_candidates: PositiveInt | None = None
    model_chain_rounds: PositiveInt | None = None
    max_repair_rounds: PositiveInt | None = None
    runtime_lua: str | None = None
    primary_launch: str | None = None
    ollama_host: str | None = None
    request_timeout_seconds: PositiveInt | None = None
    max_request_body_bytes: PositiveInt | None = None
    max_prompt_chars: PositiveInt | None = None
    max_context_bytes: PositiveInt | None = None
    max_context_depth: PositiveInt | None = None
    max_context_nodes: PositiveInt | None = None


class _RuntimeLockOverlay(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    locked: bool
    profile: str | None = None
    selected_model: str | None = Field(default=None, min_length=1)
    fallback_model: str | None = Field(default=None, min_length=1)


def _validation_details(error: ValidationError) -> tuple[str, ...]:
    return tuple(
        "{0}:{1}".format(
            ".".join(str(part) for part in item["loc"]),
            item["type"],
        )
        for item in error.errors(include_url=False, include_context=False, include_input=False)
    )


def _load_environment() -> _EnvironmentSettings:
    try:
        return _EnvironmentSettings()
    except ValidationError as error:
        raise ConfigurationError(
            "configuration_environment_invalid",
            _validation_details(error),
        ) from error


def get_profile_path(profile_name=None):
    environment = _load_environment()
    selected = profile_name or environment.profile or DEFAULT_PROFILE
    try:
        return get_resource("config/profiles/{0}.yaml".format(selected))
    except (FileNotFoundError, ModuleNotFoundError, ValueError) as error:
        raise ConfigurationError("configuration_profile_not_found") from error


def _load_profile(profile_path: Path) -> RuntimeProfile:
    try:
        raw = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigurationError("configuration_profile_unreadable") from error
    if not isinstance(raw, dict):
        raise ConfigurationError("configuration_profile_not_mapping")
    try:
        return RuntimeProfile.model_validate(raw)
    except ValidationError as error:
        raise ConfigurationError(
            "configuration_profile_invalid",
            _validation_details(error),
        ) from error


def _load_runtime_lock(profile: RuntimeProfile) -> dict[str, str]:
    from app.core.runtime_lock import load_runtime_lock

    try:
        raw_lock = load_runtime_lock()
    except (OSError, ValueError) as error:
        raise ConfigurationError("configuration_runtime_lock_unreadable") from error
    if raw_lock is None:
        return {}
    try:
        lock = _RuntimeLockOverlay.model_validate(raw_lock)
    except ValidationError as error:
        raise ConfigurationError(
            "configuration_runtime_lock_invalid",
            _validation_details(error),
        ) from error
    if not lock.locked or lock.profile != profile.name:
        return {}
    if lock.selected_model is None:
        raise ConfigurationError(
            "configuration_runtime_lock_invalid",
            ("selected_model:missing",),
        )
    values = {"model": lock.selected_model}
    if lock.fallback_model is not None:
        values["fallback_model"] = lock.fallback_model
    return values


@lru_cache(maxsize=4)
def get_runtime_profile(profile_name=None):
    environment = _load_environment()
    profile_path = get_profile_path(profile_name or environment.profile)
    profile = _load_profile(profile_path)
    merged = profile.model_dump()

    if environment.use_runtime_lock:
        merged.update(_load_runtime_lock(profile))

    environment_values = environment.model_dump(
        exclude={"profile", "use_runtime_lock"},
        exclude_none=True,
    )
    merged.update(environment_values)
    try:
        return RuntimeProfile.model_validate(merged)
    except ValidationError as error:
        raise ConfigurationError(
            "configuration_snapshot_invalid",
            _validation_details(error),
        ) from error
