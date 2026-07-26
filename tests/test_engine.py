from pathlib import Path

import pytest

from app.core.config import get_runtime_profile
from app.core.traces import TraceStore
from app.generation.engine import BackendUnavailableError, GenerationEngine
from app.generation.extractor import TaskExtractor
from app.generation.model_chain import SameModelChain
from tests.support_backends import FailIfCalledBackend, UnavailableBackend


def test_engine_raises_when_backend_unreachable(tmp_path, monkeypatch):
    trace_store = TraceStore(root=tmp_path / "traces")
    engine = GenerationEngine(
        profile=get_runtime_profile(),
        trace_store=trace_store,
        backend=UnavailableBackend(),
    )

    with pytest.raises(BackendUnavailableError):
        engine.generate(prompt="Сделай что-нибудь нестандартное без готового шаблона")
    trace_files = list(Path(trace_store.root).glob("**/*.json"))
    assert len(trace_files) == 0


def test_engine_routes_broken_envelope_prompt_to_safety_guard(tmp_path):
    trace_store = TraceStore(root=tmp_path / "traces")
    engine = GenerationEngine(
        profile=get_runtime_profile(),
        trace_store=trace_store,
        backend=FailIfCalledBackend(),
    )

    result = engine.generate(prompt="Broken envelope: {num: lua{return 1}lua}.")

    assert result.strategy == "safe_fallback"
    assert result.code == "-- judged-safe fallback\nreturn nil"
    assert result.verification_errors == []


def test_parse_planner_downgrades_inconsistent_array_family_to_generic_scalar():
    task_spec = TaskExtractor().extract(
        prompt="Для wf.vars.orders сложи amount только у записей со status shipped.",
        context={"wf": {"vars": {"orders": [{"status": "shipped", "amount": 10}]}}},
    )

    planner = SameModelChain._parse_planner(
        SameModelChain,
        '{"family":"conditional_array_projection","root":"wf.vars","source_paths":["wf.vars.orders"],"return_shape":"scalar","constraints":[],"assumptions":[],"clarification_needed":false,"clarification_question":"","semantic_checks":["must return sum of shipped amounts"]}',
        task_spec,
        prompt="Для wf.vars.orders сложи amount только у записей со status shipped.",
    )

    assert planner["family"] == "generic_lua"
    assert planner["return_shape"] == "scalar"
