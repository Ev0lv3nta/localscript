from fastapi.testclient import TestClient

from app.core.config import get_runtime_profile
from app.core.traces import TraceStore
from app.generation.engine import GenerationEngine
from app.main import create_app
from tests.support_backends import DeterministicTestBackend, UnavailableBackend


def _make_engine(tmp_path):
    return GenerationEngine(
        profile=get_runtime_profile(),
        trace_store=TraceStore(root=tmp_path / "traces"),
        backend=UnavailableBackend(),
    )


def _make_client(tmp_path):
    app = create_app(
        profile=get_runtime_profile(),
        trace_store=TraceStore(root=tmp_path / "traces"),
        backend=UnavailableBackend(),
    )
    return TestClient(app)


def test_analyze_email_root_ambiguity_requests_clarification(tmp_path):
    engine = _make_engine(tmp_path)

    payload = engine.analyze(
        prompt="Нормализуй email и верни его в lower-case.",
        context={"wf": {"vars": {"email": "A@EXAMPLE.COM"}, "initVariables": {"email": "B@EXAMPLE.COM"}}},
    )

    assert payload["suggested_strategy"] == "clarification"
    assert payload["clarification_question"] == "Use wf.vars or wf.initVariables for email root?"


def test_analyze_non_email_root_ambiguity_requests_generic_clarification(tmp_path):
    engine = _make_engine(tmp_path)

    payload = engine.analyze(
        prompt="Посчитай количество элементов в массиве items и верни число.",
        context={"wf": {"vars": {"items": [1, 2, 3]}, "initVariables": {"items": [1]}}},
    )

    assert payload["suggested_strategy"] == "clarification"
    assert payload["clarification_question"] == "Use wf.vars or wf.initVariables for this task?"


def test_rich_generate_returns_clarification_required_for_customer_city_ambiguity(tmp_path):
    client = _make_client(tmp_path)

    response = client.post(
        "/api/generate",
        json={
            "prompt": "Верни city из customer.address.city.",
            "context": {
                "wf": {
                    "vars": {"customer": {"address": {"city": "Moscow"}}},
                    "initVariables": {"customer": {"address": {"city": "Kazan"}}},
                }
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "clarification_required"
    assert payload["question"] == "Use wf.vars or wf.initVariables for this task?"


def test_analyze_mutate_vs_return_ambiguity_requests_clarification(tmp_path):
    engine = _make_engine(tmp_path)

    payload = engine.analyze(
        prompt="Очисти payload от лишних полей и верни результат.",
        context={"wf": {"vars": {"payload": {"id": "1", "debug": "x"}}}},
    )

    assert payload["suggested_strategy"] == "clarification"
    assert "mutate" in payload["clarification_question"].lower()


def test_analyze_output_key_ambiguity_requests_clarification(tmp_path):
    engine = _make_engine(tmp_path)

    payload = engine.analyze(
        prompt="Верни результат в JSON envelope. Какой key использовать внутри envelope для итогового значения, не уточнено.",
        context={"wf": {"vars": {"value": "demo"}}},
    )

    assert payload["suggested_strategy"] == "clarification"
    assert "key" in payload["clarification_question"].lower()


def test_prepare_session_state_sets_vars_root_from_clarification_answer(tmp_path):
    engine = _make_engine(tmp_path)

    initial = engine._prepare_session_state(
        session_id="session-vars",
        prompt="Верни city из customer.address.city.",
        context={
            "wf": {
                "vars": {"customer": {"address": {"city": "Moscow"}}},
                "initVariables": {"customer": {"address": {"city": "Kazan"}}},
            }
        },
        feedback=None,
        clarification_answer=None,
    )
    initial["open_clarification_question"] = "Use wf.vars or wf.initVariables for this task?"
    engine.session_store.write("session-vars", initial)

    continued = engine._prepare_session_state(
        session_id="session-vars",
        prompt=None,
        context=None,
        feedback=None,
        clarification_answer="Use wf.vars for this task.",
    )

    assert continued["clarified_root"] == "wf.vars"
    assert continued["clarification_history"][-1]["answer"] == "Use wf.vars for this task."


def test_prepare_session_state_sets_init_root_from_clarification_answer(tmp_path):
    engine = _make_engine(tmp_path)

    initial = engine._prepare_session_state(
        session_id="session-init",
        prompt="Верни city из customer.address.city.",
        context={
            "wf": {
                "vars": {"customer": {"address": {"city": "Moscow"}}},
                "initVariables": {"customer": {"address": {"city": "Kazan"}}},
            }
        },
        feedback=None,
        clarification_answer=None,
    )
    initial["open_clarification_question"] = "Use wf.vars or wf.initVariables for this task?"
    engine.session_store.write("session-init", initial)

    continued = engine._prepare_session_state(
        session_id="session-init",
        prompt=None,
        context=None,
        feedback=None,
        clarification_answer="Use wf.initVariables for this task.",
    )

    assert continued["clarified_root"] == "wf.initVariables"
    assert continued["clarification_history"][-1]["answer"] == "Use wf.initVariables for this task."


def test_clarified_root_retargets_semantic_oracle_hints(tmp_path):
    engine = GenerationEngine(
        profile=get_runtime_profile(),
        trace_store=TraceStore(root=tmp_path / "traces"),
        backend=DeterministicTestBackend(),
    )
    context = {
        "wf": {
            "vars": {"email": "A@EXAMPLE.COM"},
            "initVariables": {"email": "B@EXAMPLE.COM"},
        }
    }

    first = engine.generate_rich(
        prompt="Нормализуй email и верни его в lower-case.",
        context=context,
    )
    second = engine.generate_rich(
        session_id=first.session_id,
        clarification_answer="Use wf.initVariables for email root.",
    )
    session = engine.session_store.read(first.session_id)

    assert second.status == "completed"
    assert second.verification_errors == []
    assert "wf.initVariables.email" in second.code
    assert session["extracted_slots"]["generation_hints"]["email_path"] == (
        "wf.initVariables.email"
    )
