from types import SimpleNamespace

import pytest

from app.generation.formatter import OutputFormatter
from app.repair.canonical import CANONICAL_REPAIR_ACTIONS, CanonicalFamilyRepairer
from app.repair.critic import RepairAction, RepairPlan, ValidationCritic
from app.repair.loop import DeterministicRepairer


def test_canonical_repair_actions_have_an_explicit_boundary():
    repairer = CanonicalFamilyRepairer()

    assert CANONICAL_REPAIR_ACTIONS == {
        "rewrite_augment_existing_code_envelope",
        "rewrite_datum_time_to_iso8601",
        "rewrite_email_validation",
        "rewrite_ensure_items_array",
        "rewrite_iso8601_to_epoch",
        "rewrite_normalize_email_string",
        "rewrite_rest_cleanup_keep_only",
    }
    assert all(repairer.supports(action) for action in CANONICAL_REPAIR_ACTIONS)
    assert not repairer.supports("strip_markdown_fences")


def test_unknown_canonical_action_fails_closed():
    with pytest.raises(ValueError, match="unsupported_canonical_repair"):
        CanonicalFamilyRepairer().apply("rewrite_unknown_family", SimpleNamespace())


def test_deterministic_repair_delegates_full_family_replacement():
    class RecordingCanonicalRepairer:
        def __init__(self):
            self.calls = []

        def supports(self, action_name):
            return action_name == "rewrite_iso8601_to_epoch"

        def apply(self, action_name, task_spec):
            self.calls.append((action_name, task_spec))
            return "return 42"

    canonical = RecordingCanonicalRepairer()
    task_spec = SimpleNamespace(output_style="lua_block")
    plan = RepairPlan(
        repairable=True,
        summary="canonical",
        actions=[RepairAction("rewrite_iso8601_to_epoch", "test")],
    )

    code = DeterministicRepairer(
        OutputFormatter(),
        canonical_repairer=canonical,
    ).apply("return nil", plan, task_spec)

    assert code == "return 42"
    assert canonical.calls == [("rewrite_iso8601_to_epoch", task_spec)]


def test_normalize_email_repair_uses_the_clarified_root():
    task_spec = SimpleNamespace(
        family="normalize_email_string",
        generation_hints={"email_path": "wf.vars.email"},
        target_root="wf.initVariables",
        output_style="lua_block",
    )
    validation_report = SimpleNamespace(error_codes=lambda: ["semantic_mismatch"])

    plan = ValidationCritic().build_plan(
        task_spec,
        validation_report,
        'return string.lower(string.gsub(wf.initVariables.email, "%W", ""))',
    )
    code = DeterministicRepairer(OutputFormatter()).apply(
        "return nil",
        plan,
        task_spec,
    )

    assert [action.name for action in plan.actions] == ["rewrite_normalize_email_string"]
    assert "wf.initVariables.email" in code
    assert "wf.vars.email" not in code
    assert 'string.gsub(email, "^%s*(.-)%s*$", "%1")' in code
    assert "return string.lower(email)" in code


def test_normalize_email_lower_missing_uses_canonical_repair():
    task_spec = SimpleNamespace(family="normalize_email_string")
    validation_report = SimpleNamespace(error_codes=lambda: ["normalize_email_lower_missing"])

    plan = ValidationCritic().build_plan(task_spec, validation_report)

    assert [action.name for action in plan.actions] == ["rewrite_normalize_email_string"]
