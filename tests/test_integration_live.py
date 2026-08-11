import pytest
from fastapi.testclient import TestClient

from app.core.benchmarks import run_dataset_benchmark
from app.core.config import get_runtime_profile
from app.core.traces import TraceStore
from app.main import create_app

pytestmark = [pytest.mark.integration, pytest.mark.ollama, pytest.mark.gpu]


def test_model_backed_eval_live_backend_passes(live_ollama_backend):
    report = run_dataset_benchmark(
        "evals/regression/model_backed_eval.jsonl",
        profile=get_runtime_profile(),
        backend=live_ollama_backend,
    )

    assert report["ok"] is True
    assert report["passed"] == report["total"]


def test_richest_current_live_generate_path_uses_ollama_chain(tmp_path, live_ollama_backend):
    app = create_app(
        profile=get_runtime_profile(),
        trace_store=TraceStore(root=tmp_path / "traces"),
        backend=live_ollama_backend,
    )
    client = TestClient(app)

    response = client.post(
        "/generate",
        json={
            "prompt": "Из массива wf.vars.subscribers собери новый массив уникальных доменов почты через _utils.array.new(). Бери только записи, где email не nil и содержит символ @. Домен нужно взять после @, привести к нижнему регистру и добавить в результат только один раз. Если subscribers равен nil или пустой, верни пустой массив.",
            "context": {
                "wf": {
                    "vars": {
                        "subscribers": [
                            {"email": "User@Example.com"},
                            {"email": "admin@example.com"},
                            {"email": "root@test.local"},
                        ]
                    }
                }
            },
        },
    )

    assert response.status_code == 200
    assert response.headers["X-Strategy"] == "ollama_chain"
    assert "example.com" in response.json()["code"] or "subscribers" in response.json()["code"]
