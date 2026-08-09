import pytest
from pydantic import ValidationError

from app.generation.task_resolver import TaskResolver
from app.generation.taskspec import TaskResolutionSource, TaskSpec


def candidate(**updates):
    values = {
        "normalized_prompt": "return the first item",
        "family": None,
        "target_root": "unknown",
    }
    values.update(updates)
    return TaskSpec(**values)


def test_extractor_family_has_priority_over_planner_family():
    resolved = TaskResolver().resolve(
        candidate(family="last_array_item", target_root="wf.vars"),
        planner={"family": "generic_lua", "root": "wf.initVariables"},
    )

    assert resolved.family == "last_array_item"
    assert resolved.target_root == "wf.vars"
    assert resolved.resolution_source is TaskResolutionSource.EXTRACTOR
    assert resolved.planner_family == "generic_lua"


def test_planner_resolves_family_and_root_when_extractor_has_no_decision():
    resolved = TaskResolver().resolve(
        candidate(),
        planner={"family": "conditional_array_projection", "root": "wf.vars"},
    )

    assert resolved.family == "conditional_array_projection"
    assert resolved.target_root == "wf.vars"
    assert resolved.resolution_source is TaskResolutionSource.PLANNER


def test_missing_family_resolves_to_explicit_generic_family():
    resolved = TaskResolver().resolve(candidate(), planner={})

    assert resolved.family == "generic_lua"
    assert resolved.resolution_source is TaskResolutionSource.GENERIC


def test_resolving_an_existing_spec_is_idempotent():
    resolver = TaskResolver()
    resolved = resolver.resolve(candidate(), planner={"family": "generic_lua"})

    repeated = resolver.resolve(resolved, planner={"family": "other"})

    assert repeated is resolved


def test_task_specs_are_immutable_and_reject_unknown_fields():
    extracted = candidate()
    resolved = TaskResolver().resolve(extracted)

    with pytest.raises(ValidationError):
        extracted.target_root = "wf.vars"
    with pytest.raises(ValidationError):
        resolved.family = "other"
    with pytest.raises(ValidationError):
        TaskSpec(normalized_prompt="x", hidden_switch=True)
