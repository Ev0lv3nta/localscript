import os

import pytest

from app.core.config import get_runtime_profile
from app.generation.ollama import OllamaBackend


def pytest_collection_modifyitems(config, items):
    for item in items:
        marker_names = {mark.name for mark in item.iter_markers()}
        if "integration" in marker_names:
            if "unit" in marker_names:
                raise pytest.UsageError(
                    f"{item.nodeid} cannot be marked as both unit and integration"
                )
            continue
        item.add_marker(pytest.mark.unit)


@pytest.fixture
def live_ollama_backend():
    backend = OllamaBackend(get_runtime_profile())
    if not backend.ping():
        if os.getenv("LOCALSCRIPT_REQUIRE_LIVE", "0") == "1":
            pytest.fail("live Ollama backend is required but unavailable")
        pytest.skip("live Ollama backend unavailable")
    return backend
