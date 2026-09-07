from functools import lru_cache
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Annotated, Any, Literal

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
DEFAULT_PROFILE = "local"

PositiveInt = Annotated[int, Field(gt=0)]
Port = Annotated[int, Field(gt=0, le=65535)]


class ConfigurationError(RuntimeError):
    """Safe, typed startup failure for invalid runtime configuration."""

    def __init__(self, code: str, details: tuple[str, ...] = ()):
        self.code = code
        self.details = details
        suffix = "::{}".format(",".join(details)) if details else ""
        super().__init__(f"{code}{suffix}")


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
    model: str = Field(
        default="hf.co/unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M",
        min_length=1,
    )
    # Kept in the public snapshot for compatibility. Startup and readiness do not
    # require or download a second model.
    fallback_model: str = Field(
        default="hf.co/unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M",
        min_length=1,
    )
    think: bool = False
    stream: bool = False
    num_ctx: PositiveInt = 4096
    num_predict: PositiveInt = 256
    batch: PositiveInt = 1
    parallel: PositiveInt = 1
    runtime_lua: str = Field(default="lua5.4_subprocess", min_length=1)
    ollama_host: str = Field(default="http://127.0.0.1:11434", min_length=1)
    ollama_mode: Literal["auto", "local_cli", "remote_api"] = "auto"
    request_timeout_seconds: PositiveInt = 45
    max_request_body_bytes: PositiveInt = 131072
    max_prompt_chars: PositiveInt = 6000
    max_context_bytes: PositiveInt = 64000
    max_context_depth: PositiveInt = 16
    max_context_nodes: PositiveInt = 2000
    port: Port = 8080
    bind_host: str = Field(default="127.0.0.1", min_length=1)
    ui_enabled: bool = True
    remote_mode: bool = False
    remote_token: str = ""
    startup_timeout_seconds: PositiveInt = 120
    ollama_poll_interval_seconds: PositiveInt = 2


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
    runtime_lua: str | None = None
    ollama_host: str | None = None
    ollama_mode: Literal["auto", "local_cli", "remote_api"] | None = None
    request_timeout_seconds: PositiveInt | None = None
    max_request_body_bytes: PositiveInt | None = None
    max_prompt_chars: PositiveInt | None = None
    max_context_bytes: PositiveInt | None = None
    max_context_depth: PositiveInt | None = None
    max_context_nodes: PositiveInt | None = None
    port: Port | None = None
    bind_host: str | None = None
    ui_enabled: EnvironmentBool | None = None
    remote_mode: EnvironmentBool | None = None
    remote_token: str | None = None
    startup_timeout_seconds: PositiveInt | None = None
    ollama_poll_interval_seconds: PositiveInt | None = None


class _RuntimeLockOverlay(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    locked: bool
    profile: str | None = None
    selected_model: str | None = Field(default=None, min_length=1)
    fallback_model: str | None = Field(default=None, min_length=1)


def _validation_details(error: ValidationError) -> tuple[str, ...]:
    return tuple(
        "{}:{}".format(
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


def get_profile_path(profile_name: str | None = None) -> Traversable:
    environment = _load_environment()
    selected = profile_name or environment.profile or DEFAULT_PROFILE
    try:
        return get_resource(f"config/profiles/{selected}.yaml")
    except (FileNotFoundError, ModuleNotFoundError, ValueError) as error:
        raise ConfigurationError("configuration_profile_not_found") from error


def _load_profile(
    profile_path: Traversable,
    *,
    requested_name: str,
    aliases: tuple[str, ...] = (),
) -> RuntimeProfile:
    try:
        raw = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigurationError("configuration_profile_unreadable") from error
    if not isinstance(raw, dict):
        raise ConfigurationError("configuration_profile_not_mapping")
    if set(raw) == {"alias"}:
        alias = raw.get("alias")
        if not isinstance(alias, str) or not alias.strip():
            raise ConfigurationError("configuration_profile_invalid", ("alias:string_type",))
        alias = alias.strip()
        if alias == requested_name or alias in aliases:
            raise ConfigurationError("configuration_profile_alias_cycle")
        target = _load_profile(
            get_profile_path(alias),
            requested_name=alias,
            aliases=(*aliases, requested_name),
        )
        return target.model_copy(update={"name": requested_name})
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
def get_runtime_profile(profile_name: str | None = None) -> RuntimeProfile:
    environment = _load_environment()
    selected_profile = profile_name or environment.profile or DEFAULT_PROFILE
    profile_path = get_profile_path(selected_profile)
    profile = _load_profile(profile_path, requested_name=selected_profile)
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
