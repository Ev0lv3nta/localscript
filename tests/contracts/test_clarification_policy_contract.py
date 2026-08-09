from app.generation.clarification import ClarificationKind, ClarificationPolicy
from app.generation.taskspec import TaskSpec


def task_spec(**updates):
    values = {
        "normalized_prompt": "return value",
        "target_root": "wf.vars",
    }
    values.update(updates)
    return TaskSpec(**values)


def test_explicit_root_history_prevents_repeated_clarification():
    decision = ClarificationPolicy().evaluate(
        task_spec(target_root="unknown_mixed"),
        {"clarified_root": "wf.vars"},
    )

    assert decision.required is False


def test_root_ambiguity_has_a_typed_decision():
    decision = ClarificationPolicy().evaluate(
        task_spec(
            normalized_prompt="normalize email",
            target_root="unknown_mixed",
        ),
        {},
    )

    assert decision.required is True
    assert decision.kind is ClarificationKind.ROOT
    assert "email root" in decision.question


def test_policy_does_not_reclassify_an_unambiguous_task():
    decision = ClarificationPolicy().evaluate(task_spec(), {})

    assert decision.required is False
    assert decision.kind is None
