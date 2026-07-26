from functools import lru_cache
from pathlib import Path

import os
import yaml
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE = "competition"


class RuntimeProfile(BaseModel):
    name: str
    model: str
    fallback_model: str
    think: bool = False
    stream: bool = False
    num_ctx: int
    num_predict: int
    batch: int
    parallel: int
    max_candidates: int
    model_chain_rounds: int = 1
    max_repair_rounds: int
    runtime_lua: str
    primary_launch: str
    ollama_host: str = "http://127.0.0.1:11434"
    request_timeout_seconds: int = 45
    max_request_body_bytes: int = 131072
    max_prompt_chars: int = 6000
    max_context_bytes: int = 64000
    max_context_depth: int = 16
    max_context_nodes: int = 2000

    class Config:
        extra = "ignore"


def get_profile_path(profile_name=None):
    selected = profile_name or os.getenv("LOCALSCRIPT_PROFILE", DEFAULT_PROFILE)
    return PROJECT_ROOT / "config" / "profiles" / "{0}.yaml".format(selected)


@lru_cache(maxsize=4)
def get_runtime_profile(profile_name=None):
    profile_path = get_profile_path(profile_name)
    raw = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    primary_model = os.getenv("LOCALSCRIPT_PRIMARY_MODEL")
    fallback_model = os.getenv("LOCALSCRIPT_FALLBACK_MODEL")
    if primary_model:
        raw["model"] = primary_model
    if fallback_model:
        raw["fallback_model"] = fallback_model
    if os.getenv("LOCALSCRIPT_USE_RUNTIME_LOCK", "0") == "1":
        from app.core.runtime_lock import load_runtime_lock

        runtime_lock = load_runtime_lock()
        if runtime_lock and runtime_lock.get("locked") and runtime_lock.get("profile") == raw.get("name"):
            raw["model"] = runtime_lock.get("selected_model", raw["model"])
            raw["fallback_model"] = runtime_lock.get("fallback_model", raw["fallback_model"])
    return RuntimeProfile(**raw)
