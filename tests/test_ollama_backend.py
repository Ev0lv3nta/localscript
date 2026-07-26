import json
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from app.core.config import RuntimeProfile
from app.generation.backend_errors import (
    BackendModel,
    BackendProtocol,
    BackendTimeout,
    BackendUnavailable,
)
from app.generation.ollama import OllamaBackend


def make_profile(**overrides):
    values = {
        "name": "competition",
        "model": "qwen3:8b-q4_K_M",
        "fallback_model": "qwen3:4b-instruct-2507-q4_K_M",
        "think": False,
        "stream": False,
        "num_ctx": 4096,
        "num_predict": 256,
        "batch": 1,
        "parallel": 1,
        "max_candidates": 2,
        "max_repair_rounds": 2,
        "runtime_lua": "lua5.4_subprocess",
        "primary_launch": "./scripts/judge_up.sh",
        "request_timeout_seconds": 45,
    }
    values.update(overrides)
    return RuntimeProfile(**values)


class FakeResponse:
    def __init__(self, payload=None, *, status_code=200, json_error=None):
        self._payload = payload
        self.status_code = status_code
        self._json_error = json_error
        self.request = httpx.Request("GET", "http://127.0.0.1/api/test")

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "unsafe upstream status text",
                request=self.request,
                response=httpx.Response(self.status_code, request=self.request),
            )

    def json(self):
        if self._json_error:
            raise self._json_error
        return self._payload


class FakeClient:
    def __init__(self, responses=None, request_hook=None):
        self.responses = list(responses or [])
        self.request_hook = request_hook
        self.requests = []
        self.close_calls = 0

    def request(self, method, path, **kwargs):
        self.requests.append((method, path, kwargs))
        if self.request_hook:
            return self.request_hook(method, path, kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def close(self):
        self.close_calls += 1


def install_fake_client(monkeypatch, fake):
    constructed = []

    def factory(**kwargs):
        constructed.append(kwargs)
        return fake

    monkeypatch.setattr("app.generation.ollama.httpx.Client", factory)
    return constructed


def tags_payload(name="qwen3:8b-q4_K_M", digest="sha256:abc123"):
    return {
        "models": [
            {
                "name": name,
                "digest": digest,
                "details": {"parameter_size": "8B", "quantization_level": "Q4_K_M"},
            }
        ]
    }


def test_backend_constructs_one_isolated_bounded_client(monkeypatch):
    fake = FakeClient()
    constructed = install_fake_client(monkeypatch, fake)

    backend = OllamaBackend(make_profile(parallel=3, request_timeout_seconds=12))

    assert len(constructed) == 1
    options = constructed[0]
    assert options["base_url"] == "http://127.0.0.1:11434"
    assert options["trust_env"] is False
    assert options["timeout"].connect == 5.0
    assert options["timeout"].read == 12.0
    assert options["timeout"].write == 10.0
    assert options["timeout"].pool == 5.0
    assert options["limits"].max_connections == 3
    assert options["limits"].max_keepalive_connections == 3
    backend.close()


def test_ollama_generate_resolves_model_and_sends_profile_flags(monkeypatch):
    fake = FakeClient(
        [
            FakeResponse(tags_payload()),
            FakeResponse({"model": "qwen3:8b-q4_K_M", "response": "return 1"}),
        ]
    )
    constructed = install_fake_client(monkeypatch, fake)

    backend = OllamaBackend(make_profile())
    result = backend.generate("Return Lua code only: return 1")

    assert result == "return 1"
    assert len(constructed) == 1
    method, path, request = fake.requests[1]
    assert (method, path) == ("POST", "/api/generate")
    assert request["json"]["think"] is False
    assert request["json"]["stream"] is False
    assert request["json"]["model"] == "qwen3:8b-q4_K_M"
    assert request["json"]["options"]["num_batch"] == 1
    assert backend.last_resolved_model.tag == "qwen3:8b-q4_K_M"
    assert backend.last_resolved_model.digest == "sha256:abc123"
    backend.close()


def test_tag_details_expose_digest_and_model_metadata(monkeypatch):
    fake = FakeClient([FakeResponse(tags_payload())])
    install_fake_client(monkeypatch, fake)
    backend = OllamaBackend(make_profile())

    assert backend.list_tag_details() == [
        {
            "name": "qwen3:8b-q4_K_M",
            "digest": "sha256:abc123",
            "details": {"parameter_size": "8B", "quantization_level": "Q4_K_M"},
        }
    ]
    backend.close()


def test_resolved_model_is_cached_until_explicit_refresh(monkeypatch):
    fake = FakeClient(
        [
            FakeResponse(tags_payload(digest="sha256:first")),
            FakeResponse({"model": "qwen3:8b-q4_K_M", "response": "return 1"}),
            FakeResponse({"model": "qwen3:8b-q4_K_M", "response": "return 2"}),
            FakeResponse(tags_payload(digest="sha256:second")),
        ]
    )
    install_fake_client(monkeypatch, fake)
    backend = OllamaBackend(make_profile())

    assert backend.complete("first") == "return 1"
    assert backend.complete("second") == "return 2"
    assert [path for _, path, _ in fake.requests].count("/api/tags") == 1

    refreshed = backend.refresh_model()

    assert refreshed.digest == "sha256:second"
    assert [path for _, path, _ in fake.requests].count("/api/tags") == 2
    backend.close()


def test_close_is_idempotent_and_context_managed(monkeypatch):
    fake = FakeClient()
    install_fake_client(monkeypatch, fake)

    with OllamaBackend(make_profile()) as backend:
        assert backend is not None
    backend.close()

    assert fake.close_calls == 1
    with pytest.raises(BackendUnavailable) as exc_info:
        backend.list_tags()
    assert exc_info.value.reason == "backend_closed"


def test_parallel_profile_caps_all_backend_io(monkeypatch):
    lock = threading.Lock()
    active = 0
    maximum = 0

    def request_hook(method, path, kwargs):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        try:
            time.sleep(0.03)
            if path == "/api/tags":
                return FakeResponse(tags_payload())
            return FakeResponse({"model": "qwen3:8b-q4_K_M", "response": "return 1"})
        finally:
            with lock:
                active -= 1

    fake = FakeClient(request_hook=request_hook)
    install_fake_client(monkeypatch, fake)
    backend = OllamaBackend(make_profile(parallel=1))
    barrier = threading.Barrier(3)
    failures = []

    def run():
        barrier.wait()
        try:
            backend.complete("return 1")
        except Exception as exc:  # pragma: no cover - asserted through failures
            failures.append(exc)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert failures == []
    assert maximum == 1
    backend.close()


@pytest.mark.parametrize(
    ("failure", "error_type", "reason"),
    [
        (
            httpx.ConnectError(
                "token=secret-password",
                request=httpx.Request("GET", "http://user:secret@127.0.0.1/api/tags"),
            ),
            BackendUnavailable,
            "transport_error",
        ),
        (
            httpx.ReadTimeout(
                "token=secret-password",
                request=httpx.Request("GET", "http://user:secret@127.0.0.1/api/tags"),
            ),
            BackendTimeout,
            "request_timeout",
        ),
    ],
)
def test_transport_failures_are_typed_and_sanitized(
    monkeypatch, failure, error_type, reason
):
    fake = FakeClient([failure])
    install_fake_client(monkeypatch, fake)
    backend = OllamaBackend(make_profile())

    with pytest.raises(error_type) as exc_info:
        backend.list_tags()

    assert exc_info.value.code.startswith("backend_")
    assert exc_info.value.reason == reason
    assert "secret" not in str(exc_info.value)
    assert "user" not in str(exc_info.value)
    rendered = "".join(
        traceback.format_exception(
            type(exc_info.value), exc_info.value, exc_info.value.__traceback__
        )
    )
    assert "secret" not in rendered
    assert "user" not in rendered
    backend.close()


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        (FakeResponse({}, status_code=503), "bad_status"),
        (FakeResponse(json_error=ValueError("secret invalid JSON")), "invalid_json"),
        (FakeResponse(["not", "an", "object"]), "invalid_json_shape"),
    ],
)
def test_status_and_json_failures_are_protocol_errors(monkeypatch, response, reason):
    fake = FakeClient([response])
    install_fake_client(monkeypatch, fake)
    backend = OllamaBackend(make_profile())

    with pytest.raises(BackendProtocol) as exc_info:
        backend.list_tags()

    assert exc_info.value.code == "backend_protocol_error"
    assert exc_info.value.reason == reason
    assert "secret" not in str(exc_info.value)
    backend.close()


def test_empty_generation_is_protocol_error(monkeypatch):
    fake = FakeClient(
        [
            FakeResponse(tags_payload()),
            FakeResponse({"model": "qwen3:8b-q4_K_M", "response": "  "}),
        ]
    )
    install_fake_client(monkeypatch, fake)
    backend = OllamaBackend(make_profile())

    with pytest.raises(BackendProtocol) as exc_info:
        backend.complete("prompt")

    assert str(exc_info.value) == "Backend returned an invalid response."
    assert exc_info.value.reason == "empty_response"
    backend.close()


def test_generation_rejects_response_from_different_model(monkeypatch):
    fake = FakeClient(
        [
            FakeResponse(tags_payload()),
            FakeResponse({"model": "other:latest", "response": "return 1"}),
        ]
    )
    install_fake_client(monkeypatch, fake)
    backend = OllamaBackend(make_profile())

    with pytest.raises(BackendModel) as exc_info:
        backend.complete("prompt")

    assert exc_info.value.reason == "model_identity_mismatch"
    backend.close()


def test_generation_preserves_model_resolution_failure(monkeypatch):
    fake = FakeClient([FakeResponse(tags_payload(name="other:latest"))])
    install_fake_client(monkeypatch, fake)
    backend = OllamaBackend(make_profile())

    with pytest.raises(BackendModel) as exc_info:
        backend.complete("prompt")

    assert exc_info.value.reason == "model_not_found"
    assert len(fake.requests) == 1
    backend.close()


def test_competition_backend_rejects_public_and_private_hosts_without_opt_in(
    monkeypatch,
):
    for host in ("http://8.8.8.8:11434", "http://10.0.0.2:11434", "http://ollama:11434"):
        monkeypatch.setenv("LOCALSCRIPT_OLLAMA_HOST", host)
        with pytest.raises(BackendUnavailable) as exc_info:
            OllamaBackend(make_profile())
        assert str(exc_info.value) == "ollama_host_not_local"


def test_remote_opt_in_allows_private_host_and_explicit_container_alias(monkeypatch):
    fake = FakeClient()
    install_fake_client(monkeypatch, fake)
    monkeypatch.setenv("LOCALSCRIPT_ALLOW_REMOTE_OLLAMA", "1")

    for host in ("http://10.0.0.2:11434", "http://ollama:11434"):
        monkeypatch.setenv("LOCALSCRIPT_OLLAMA_HOST", host)
        backend = OllamaBackend(make_profile())
        backend.close()


class OllamaHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        self._send(tags_payload())

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(content_length)
        self._send({"model": "qwen3:8b-q4_K_M", "response": "return 1"})

    def _send(self, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


def test_loopback_backend_ignores_broken_proxy_environment(monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), OllamaHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv(
        "LOCALSCRIPT_OLLAMA_HOST", "http://127.0.0.1:{0}".format(server.server_port)
    )

    try:
        with OllamaBackend(make_profile()) as backend:
            assert backend.complete("prompt") == "return 1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
