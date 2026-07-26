import json
import ipaddress
import math
import os
import threading
from contextlib import contextmanager
from typing import Mapping
from urllib.parse import urlparse

import httpx

from app.generation.backend_errors import (
    BackendError,
    BackendModel,
    BackendProtocol,
    BackendTimeout,
    BackendUnavailable,
)
from app.generation.model_resolver import (
    model_identities_match,
    parse_model_tags,
    resolve_model,
)


class OllamaBackend:
    def __init__(self, profile):
        self.profile = profile
        self.host = os.getenv("LOCALSCRIPT_OLLAMA_HOST", profile.ollama_host).rstrip("/")
        self.base_url = self.host
        self._validate_host_policy()
        self._parallel = _safe_parallel(profile.parallel)
        self._semaphore = threading.BoundedSemaphore(self._parallel)
        self._state = threading.Condition()
        self._active_requests = 0
        self._closing = False
        self._closed = False
        self._model_lock = threading.RLock()
        self._resolved_models = {}
        self._last_resolved_model = None
        self._client = httpx.Client(
            base_url=self.host,
            timeout=_request_timeout(profile.request_timeout_seconds),
            limits=httpx.Limits(
                max_connections=self._parallel,
                max_keepalive_connections=self._parallel,
                keepalive_expiry=30.0,
            ),
            trust_env=False,
        )

    def _validate_host_policy(self):
        parsed = urlparse(self.host)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise BackendUnavailable("ollama_host_not_local", reason="host_not_permitted")

        hostname = (parsed.hostname or "").strip().lower()
        if hostname == "localhost":
            return
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            address = None
        if address is not None and address.is_loopback:
            return

        if os.getenv("LOCALSCRIPT_ALLOW_REMOTE_OLLAMA", "0") == "1":
            configured_alias = os.getenv("LOCALSCRIPT_OLLAMA_CONTAINER_ALIAS", "ollama")
            if hostname == configured_alias.strip().lower():
                return
            if address is not None and address.is_private:
                return

        raise BackendUnavailable("ollama_host_not_local", reason="host_not_permitted")

    def ping(self):
        try:
            self._fetch_tag_details()
            return True
        except BackendError:
            return False

    def list_tags(self):
        return [item.tag for item in self._fetch_tag_details()]

    def list_tag_details(self):
        return [
            {"name": item.tag, "digest": item.digest, "details": dict(item.details)}
            for item in self._fetch_tag_details()
        ]

    def resolve_model(self, model=None, *, refresh=False):
        selected = model or self.profile.model
        cache_key = (selected or "").strip()
        with self._model_lock:
            if not refresh and cache_key in self._resolved_models:
                resolved = self._resolved_models[cache_key]
                self._last_resolved_model = resolved
                return resolved
            resolved = resolve_model(selected, self._fetch_tag_details())
            self._resolved_models[cache_key] = resolved
            self._last_resolved_model = resolved
        return resolved

    def refresh_model(self, model=None):
        return self.resolve_model(model, refresh=True)

    @property
    def last_resolved_model(self):
        with self._model_lock:
            return self._last_resolved_model

    def complete(self, prompt, model=None, response_format=None):
        resolved = self.resolve_model(model)
        payload = {
            "model": resolved.tag,
            "think": self.profile.think,
            "stream": False,
            "prompt": prompt or "",
            "options": {
                "num_ctx": self.profile.num_ctx,
                "num_predict": self.profile.num_predict,
                "num_batch": self.profile.batch,
            },
        }
        if response_format:
            payload["format"] = response_format

        data = self._request_json("POST", "/api/generate", json=payload)
        if not model_identities_match(resolved, data.get("model")):
            raise BackendModel(reason="model_identity_mismatch")

        raw_candidate = data.get("response")
        if not isinstance(raw_candidate, str):
            raise BackendProtocol(reason="invalid_response_text")
        candidate = raw_candidate.strip()
        if not candidate:
            raise BackendProtocol(reason="empty_response")
        return candidate

    def generate(self, prompt, context=None):
        return self.complete(self._build_prompt(prompt, context))

    def close(self):
        with self._state:
            if self._closed:
                return
            if self._closing:
                while not self._closed:
                    self._state.wait()
                return
            self._closing = True
            while self._active_requests:
                self._state.wait()

        try:
            self._client.close()
        finally:
            with self._state:
                self._closed = True
                self._closing = False
                self._state.notify_all()

    def __enter__(self):
        with self._state:
            if self._closed or self._closing:
                raise BackendUnavailable(reason="backend_closed")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def _fetch_tag_details(self):
        payload = self._request_json("GET", "/api/tags")
        return parse_model_tags(payload)

    def _request_json(self, method, path, **kwargs):
        try:
            with self._request_slot():
                response = self._client.request(method, path, **kwargs)
                response.raise_for_status()
        except httpx.TimeoutException:
            raise BackendTimeout(reason="request_timeout") from None
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            raise BackendProtocol(
                reason="bad_status",
                status_code=status_code,
            ) from None
        except httpx.HTTPError:
            raise BackendUnavailable(reason="transport_error") from None

        try:
            payload = response.json()
        except (TypeError, ValueError):
            raise BackendProtocol(reason="invalid_json") from None
        if not isinstance(payload, Mapping):
            raise BackendProtocol(reason="invalid_json_shape")
        return payload

    @contextmanager
    def _request_slot(self):
        self._semaphore.acquire()
        entered = False
        try:
            with self._state:
                if self._closed or self._closing:
                    raise BackendUnavailable(reason="backend_closed")
                self._active_requests += 1
                entered = True
            yield
        finally:
            if entered:
                with self._state:
                    self._active_requests -= 1
                    if not self._active_requests:
                        self._state.notify_all()
            self._semaphore.release()

    @staticmethod
    def _build_prompt(prompt, context):
        sections = [
            "You generate only LocalScript/Lua code.",
            "Do not use JsonPath.",
            "Use wf.vars or wf.initVariables for direct data access.",
            "Return code only without markdown fences.",
            "User prompt:",
            prompt or "",
        ]
        if context is not None:
            sections.extend(["Context JSON:", json.dumps(context, ensure_ascii=False, sort_keys=True)])
        return "\n".join(sections)


def _safe_parallel(value):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = 1
    return max(1, min(parsed, 64))


def _safe_timeout_seconds(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = 45.0
    if not math.isfinite(parsed) or parsed <= 0:
        parsed = 45.0
    return max(0.1, min(parsed, 3600.0))


def _request_timeout(value):
    total = _safe_timeout_seconds(value)
    return httpx.Timeout(
        connect=min(total, 5.0),
        read=total,
        write=min(total, 10.0),
        pool=min(total, 5.0),
    )
