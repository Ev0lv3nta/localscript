from dataclasses import FrozenInstanceError

import pytest

from app.families import (
    UNSUPPORTED,
    FamilyModule,
    all_family_definitions,
    get_family_definition,
    is_known_family,
)
from app.generation.task_resolver import TaskResolver
from app.generation.taskspec import TaskResolutionSource, TaskSpec


EXPECTED_FAMILIES = {
    "augment_existing_code",
    "conditional_array_projection",
    "counter_increment",
    "datum_time_to_iso8601",
    "email_validation",
    "ensure_items_array",
    "field_mapping",
    "filter_discount_markdown",
    "generic_lua",
    "iso8601_to_epoch",
    "last_array_item",
    "normalize_email_string",
    "regex_extract",
    "rest_cleanup",
    "safety_guard",
    "table_transform",
}


def test_registry_contains_each_supported_family_once():
    definitions = all_family_definitions()
    names = [definition.name for definition in definitions]

    assert set(names) == EXPECTED_FAMILIES
    assert len(names) == len(set(names))
    assert all(isinstance(definition, FamilyModule) for definition in definitions)
    assert all(get_family_definition(name) is not None for name in names)


def test_family_definitions_are_immutable():
    definition = get_family_definition("email_validation")

    with pytest.raises(FrozenInstanceError):
        definition.name = "changed"


def test_non_semantic_families_explicitly_return_unsupported():
    assert get_family_definition("generic_lua").build_expected_result({}, {}) is UNSUPPORTED
    assert get_family_definition("safety_guard").build_expected_result({}, {}) is UNSUPPORTED


def test_unknown_planner_family_fails_closed_to_generic():
    candidate = TaskSpec(normalized_prompt="верни значение", target_root="wf.vars")

    resolved = TaskResolver().resolve(
        candidate,
        planner={"family": "invented_family", "root": "wf.vars"},
    )

    assert resolved.family == "generic_lua"
    assert resolved.resolution_source is TaskResolutionSource.GENERIC
    assert resolved.planner_family == "invented_family"
    assert is_known_family(resolved.family) is True
    assert is_known_family(resolved.planner_family) is False


def test_regex_and_email_families_accept_method_match_syntax():
    regex = get_family_definition("regex_extract")
    email = get_family_definition("email_validation")

    assert regex.validate_structure('return value:match("ID:(%d+)")', "lua_block", {}) == ()
    assert email.validate_structure(
        'return email:match("^[^@]+@[^@]+$") ~= nil',
        "lua_block",
        {},
    ) == ()
    assert email.validate_structure(
        'return string.match(email, "^[^@]+@[^@]+$")',
        "lua_block",
        {},
    )[0].code == "email_validation_boolean_missing"
