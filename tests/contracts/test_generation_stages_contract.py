import pytest

from app.generation.stages import GenerationExecution, GenerationStage


def test_success_path_has_explicit_ordered_stages():
    execution = GenerationExecution()

    execution.transition(GenerationStage.SESSION_READY)
    execution.transition(GenerationStage.TASK_RESOLVED)
    execution.transition(GenerationStage.CANDIDATE_GENERATED)
    execution.transition(GenerationStage.CANDIDATE_REPAIRED)
    execution.transition(GenerationStage.OUTCOME_FINALIZED)

    events = execution.snapshot()

    assert [event["stage"] for event in events] == [
        "session_ready",
        "task_resolved",
        "candidate_generated",
        "candidate_repaired",
        "outcome_finalized",
    ]
    assert all(event["elapsed_ms"] >= 0 for event in events)
    assert all(event["stage_duration_ms"] >= 0 for event in events)
    assert [event["elapsed_ms"] for event in events] == sorted(
        event["elapsed_ms"] for event in events
    )


def test_clarification_path_cannot_generate_a_candidate_after_decision():
    execution = GenerationExecution()
    execution.transition(GenerationStage.SESSION_READY)
    execution.transition(GenerationStage.TASK_RESOLVED)
    execution.transition(GenerationStage.CLARIFICATION_REQUIRED)

    with pytest.raises(ValueError, match="clarification_required -> candidate_generated"):
        execution.transition(GenerationStage.CANDIDATE_GENERATED)


def test_finalized_execution_rejects_further_transitions():
    execution = GenerationExecution()
    execution.transition(GenerationStage.SESSION_READY)
    execution.transition(GenerationStage.TASK_RESOLVED)
    execution.transition(GenerationStage.CANDIDATE_GENERATED)
    execution.transition(GenerationStage.OUTCOME_FINALIZED)

    with pytest.raises(ValueError, match="outcome_finalized -> candidate_repaired"):
        execution.transition(GenerationStage.CANDIDATE_REPAIRED)
