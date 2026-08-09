from types import SimpleNamespace

import pytest

from app.generation.formatter import OutputFormatter
from app.repair.canonical import CANONICAL_REPAIR_ACTIONS, CanonicalFamilyRepairer
from app.repair.critic import RepairAction, RepairPlan
from app.repair.loop import DeterministicRepairer


def test_canonical_repair_actions_have_an_explicit_boundary():
    repairer = CanonicalFamilyRepairer()

    assert CANONICAL_REPAIR_ACTIONS == {
        "rewrite_augment_existing_code_envelope",
        "rewrite_datum_time_to_iso8601",
        "rewrite_email_validation",
        "rewrite_ensure_items_array",
        "rewrite_iso8601_to_epoch",
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
