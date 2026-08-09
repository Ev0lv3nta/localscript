import json
from pathlib import Path

from app.core.config import get_runtime_profile
from app.core.storage import REDACTED
from app.core.traces import TraceStore
from app.generation.engine import GenerationEngine


class ScriptedChainBackend:
    def complete(self, prompt, response_format=None, model=None):
        if "You are the planner for a LocalScript/Lua generation pipeline." in prompt:
            return json.dumps(
                {
                    "family": "generic_lua",
                    "root": "wf.vars",
                    "source_paths": ["wf.vars.value"],
                    "return_shape": "scalar",
                    "constraints": ["Do not use JsonPath"],
                    "assumptions": ["Prefer the direct wf.vars root."],
                    "clarification_needed": False,
                    "clarification_question": "",
                    "semantic_checks": ["must return wf.vars.value"],
                },
                ensure_ascii=False,
            )
        if "You are the critic for a LocalScript/Lua generation pipeline." in prompt:
            return json.dumps(
                {
                    "repairable": True,
                    "issues": ["jsonpath_forbidden"],
                    "minimal_actions": ["switch_from_jsonpath_to_direct_root"],
                },
                ensure_ascii=False,
            )
        if "You are the fixer for a LocalScript/Lua generation pipeline." in prompt:
            return "return wf.vars.value"
        return "return $.wf.vars.value"


def test_engine_uses_same_model_chain_and_redacts_private_trace_artifacts(tmp_path):
    trace_store = TraceStore(root=tmp_path / "traces")
    engine = GenerationEngine(
        profile=get_runtime_profile(),
        trace_store=trace_store,
        backend=ScriptedChainBackend(),
    )

    result = engine.generate(
        prompt="Верни значение из wf.vars.value без дополнительных преобразований.",
        context={"wf": {"vars": {"value": 7}}},
    )

    assert result.strategy == "ollama_chain"
    assert result.code == "return wf.vars.value"
    assert result.repair_rounds == 1
    assert result.verification_errors == []

    trace_file = next(Path(trace_store.root).glob("**/*.json"))
    trace_payload = json.loads(trace_file.read_text(encoding="utf-8"))
    assert trace_payload["planner"] == REDACTED
    assert trace_payload["critic"] == REDACTED
    assert trace_payload["rules_applied"]
    assert trace_payload["repair_trace"] == REDACTED
    assert trace_payload["code"] == REDACTED
    assert trace_payload["task_spec"]["family"] == "generic_lua"
    assert trace_payload["task_spec"]["resolution_source"] == "planner"
    assert [event["stage"] for event in trace_payload["stage_events"]] == [
        "session_ready",
        "task_resolved",
        "candidate_generated",
        "outcome_finalized",
    ]
    assert all(
        event["interval_since_previous_ms"] >= 0
        for event in trace_payload["stage_events"]
    )
