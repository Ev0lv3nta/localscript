from pathlib import Path
import threading
import time

import pytest

from app.core.config import get_runtime_profile
from app.core.traces import TraceStore
from app.domain.outcomes import GenerationStatus, ValidationStatus
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


def test_engine_serializes_updates_for_the_same_session(tmp_path, monkeypatch):
    engine = GenerationEngine(
        profile=get_runtime_profile(),
        trace_store=TraceStore(root=tmp_path / "traces"),
        backend=FailIfCalledBackend(),
    )

    def increment_session(**kwargs):
        session_state = kwargs["session_state"]
        current = session_state.get("count", 0)
        time.sleep(0.01)
        session_state["count"] = current + 1

    monkeypatch.setattr(engine, "_run_generation_locked", increment_session)
    threads = [
        threading.Thread(
            target=engine._run_generation,
            kwargs={
                "prompt": "prompt",
                "context": None,
                "session_id": "shared-session",
                "feedback": None,
                "clarification_answer": None,
                "rich_mode": False,
            },
        )
        for _ in range(8)
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert engine.session_store.read("shared-session")["count"] == 8


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
    assert result.outcome.status is GenerationStatus.POLICY_REJECTED
    assert result.outcome.code is None


def test_engine_clarification_has_not_run_validation(tmp_path):
    trace_store = TraceStore(root=tmp_path / "traces")
    engine = GenerationEngine(
        profile=get_runtime_profile(),
        trace_store=trace_store,
        backend=FailIfCalledBackend(),
    )

    result = engine.generate_rich(
        prompt="Нормализуй email и верни его в lower-case.",
        context={
            "wf": {
                "vars": {"email": "A@EXAMPLE.COM"},
                "initVariables": {"email": "B@EXAMPLE.COM"},
            }
        },
    )

    assert result.outcome.status is GenerationStatus.CLARIFICATION_REQUIRED
    assert result.outcome.validation.status is ValidationStatus.NOT_RUN
    assert result.outcome.question == result.question
    assert result.outcome.code is None


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
