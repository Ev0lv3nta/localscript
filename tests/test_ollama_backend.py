from app.core.config import RuntimeProfile
from app.generation.ollama import OllamaBackend


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class DummyClient:
    def __init__(self, sink):
        self.sink = sink

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, url, json):
        self.sink["url"] = url
        self.sink["json"] = json
        return DummyResponse({"response": "return 1"})


def test_ollama_generate_sends_profile_think_flag(monkeypatch):
    captured = {}

    def fake_client(*args, **kwargs):
        return DummyClient(captured)

    monkeypatch.setattr("app.generation.ollama.httpx.Client", fake_client)

    profile = RuntimeProfile(
        name="competition",
        model="qwen3:8b-q4_K_M",
        fallback_model="qwen3:4b-instruct-2507-q4_K_M",
        think=False,
        stream=False,
        num_ctx=4096,
        num_predict=256,
        batch=1,
        parallel=1,
        max_candidates=2,
        max_repair_rounds=2,
        runtime_lua="lua5.4_subprocess",
        primary_launch="./scripts/judge_up.sh",
    )

    backend = OllamaBackend(profile)
    result = backend.generate("Return Lua code only: return 1")

    assert result == "return 1"
    assert captured["json"]["think"] is False
    assert captured["json"]["stream"] is False
    assert captured["json"]["model"] == "qwen3:8b-q4_K_M"
    assert captured["json"]["options"]["num_batch"] == 1


def test_competition_backend_rejects_public_ollama_host(monkeypatch):
    monkeypatch.setenv("LOCALSCRIPT_OLLAMA_HOST", "http://8.8.8.8:11434")
    profile = RuntimeProfile(
        name="competition",
        model="qwen3:8b-q4_K_M",
        fallback_model="qwen3:4b-instruct-2507-q4_K_M",
        think=False,
        stream=False,
        num_ctx=4096,
        num_predict=256,
        batch=1,
        parallel=1,
        max_candidates=2,
        max_repair_rounds=2,
        runtime_lua="lua5.4_subprocess",
        primary_launch="./scripts/judge_up.sh",
    )

    try:
        OllamaBackend(profile)
    except RuntimeError as exc:
        assert str(exc) == "ollama_host_not_local"
    else:
        raise AssertionError("competition backend should reject public Ollama hosts")
