import pytest

from app.generation.stages import GenerationExecution, GenerationStage


def test_success_path_has_explicit_ordered_stages():
    execution = GenerationExecution()

    execution.transition(GenerationStage.SESSION_READY)
    execution.transition(GenerationStage.TASK_RESOLVED)
    execution.transition(GenerationStage.CANDIDATE_GENERATED)
    execution.transition(GenerationStage.CANDIDATE_REPAIRED)
    execution.transition(GenerationStage.OUTCOME_FINALIZED)

    assert execution.snapshot() == [
        {"stage": "session_ready"},
        {"stage": "task_resolved"},
        {"stage": "candidate_generated"},
        {"stage": "candidate_repaired"},
        {"stage": "outcome_finalized"},
    ]


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
