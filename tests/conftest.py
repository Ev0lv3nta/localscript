import pytest

from app.core.config import get_runtime_profile
from app.generation.ollama import OllamaBackend


def pytest_collection_modifyitems(config, items):
    for item in items:
        marker_names = {mark.name for mark in item.iter_markers()}
        if "integration" not in marker_names:
            item.add_marker(pytest.mark.unit)


@pytest.fixture
def live_ollama_backend():
    backend = OllamaBackend(get_runtime_profile())
    if not backend.ping():
        pytest.skip("live Ollama backend unavailable")
    return backend
