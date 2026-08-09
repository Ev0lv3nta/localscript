from dataclasses import dataclass, field
from typing import List


@dataclass
class RepairAction:
    name: str
    reason: str


@dataclass
class RepairPlan:
    repairable: bool
    summary: str
    actions: List[RepairAction] = field(default_factory=list)
    source_errors: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "repairable": self.repairable,
            "summary": self.summary,
            "source_errors": self.source_errors,
            "actions": [
                {"name": action.name, "reason": action.reason}
                for action in self.actions
            ],
        }


class ValidationCritic:
    def build_plan(self, task_spec, validation_report, code=""):
        error_codes = validation_report.error_codes()
        actions = []
        has_prefix = lambda prefix: any(error.startswith(prefix) for error in error_codes)

        if "markdown_fence_forbidden" in error_codes:
            actions.append(
                RepairAction(
                    name="strip_markdown_fences",
                    reason="Remove markdown fences from the generated code.",
                )
            )

        if "jsonpath_forbidden" in error_codes:
            actions.append(
                RepairAction(
                    name="rewrite_jsonpath_roots",
                    reason="Replace JsonPath access with direct wf roots.",
                )
            )

        if "ctx_body_forbidden" in error_codes or "workflow_variables_forbidden" in error_codes:
            actions.append(
                RepairAction(
                    name="rewrite_unsupported_roots",
                    reason="Replace unsupported workflow namespaces with the allowed wf roots.",
                )
            )
        if "init_variables_missing" in error_codes or has_prefix("unexpected_root_reference::"):
            actions.append(
                RepairAction(
                    name="rewrite_unsupported_roots",
                    reason="Rewrite references so the final code uses the selected wf root consistently.",
                )
            )

        if has_prefix("shadowed_stdlib_local::"):
            actions.append(
                RepairAction(
                    name="rename_shadowed_stdlib_locals",
                    reason="Rename locals that shadow protected runtime identifiers such as table/string/math/_utils/wf.",
                )
            )

        if "return_missing" in error_codes:
            actions.append(
                RepairAction(
                    name="append_return_nil",
                    reason="Restore an explicit return path for Lua output.",
                )
            )

        if "generic_empty_array_behavior_mismatch" in error_codes:
            actions.append(
                RepairAction(
                    name="add_nil_safe_array_iteration",
                    reason="Ensure array iteration handles nil and empty input safely.",
                )
            )
            actions.append(
                RepairAction(
                    name="normalize_empty_array_return",
                    reason="Return an explicit empty array for empty input branches.",
                )
            )

        if (
            "generic_return_shape_array_mismatch" in error_codes
            or "semantic_return_shape_array_mismatch" in error_codes
        ):
            actions.append(
                RepairAction(
                    name="normalize_array_return_shape",
                    reason="Restore the expected array return shape.",
                )
            )

        if (
            "generic_return_shape_scalar_mismatch" in error_codes
            or "semantic_return_shape_scalar_mismatch" in error_codes
        ):
            actions.append(
                RepairAction(
                    name="normalize_scalar_return_shape",
                    reason="Unwrap accidental array/table wrappers when a scalar result is required.",
                )
            )

        if (
            ("lua_runtime_error" in error_codes and "trim" in (code or ""))
            or "normalize_email_trim_missing" in error_codes
        ):
            actions.append(
                RepairAction(
                    name="rewrite_string_trim",
                    reason="Replace unsupported string.trim() with a canonical gsub-based trim.",
                )
            )

        if "lua_runtime_error" in error_codes and ":contains(" in (code or ""):
            actions.append(
                RepairAction(
                    name="rewrite_array_contains",
                    reason="Replace unsupported array:contains(value) with a portable Lua membership scan.",
                )
            )

        if task_spec.family == "iso8601_to_epoch" and (
            "dangerous_stdlib_os_forbidden" in error_codes
            or "lua_syntax_error" in error_codes
            or "lua_runtime_error" in error_codes
            or "init_variables_missing" in error_codes
        ):
            actions.append(
                RepairAction(
                    name="rewrite_iso8601_to_epoch",
                    reason="Rewrite the ISO-8601-to-epoch conversion into a compact pure-Lua arithmetic implementation.",
                )
            )

        if task_spec.family == "datum_time_to_iso8601" and (
            "semantic_mismatch" in error_codes
            or "lua_syntax_error" in error_codes
            or "lua_runtime_error" in error_codes
            or "return_missing" in error_codes
            or "init_variables_missing" in error_codes
        ):
            actions.append(
                RepairAction(
                    name="rewrite_datum_time_to_iso8601",
                    reason="Rewrite DATUM/TIME formatting into a stable pure-Lua ISO-8601 implementation with short-TIME normalization.",
                )
            )

        if task_spec.family == "ensure_items_array" and (
            "lua_runtime_error" in error_codes
            or "semantic_mismatch" in error_codes
            or "generic_return_shape_array_mismatch" in error_codes
        ):
            actions.append(
                RepairAction(
                    name="rewrite_ensure_items_array",
                    reason="Rewrite package item normalization into the canonical ensureArray in-place loop without unsupported helper calls.",
                )
            )

        if task_spec.family == "email_validation" and (
            "semantic_mismatch" in error_codes
            or "lua_runtime_error" in error_codes
            or "email_validation_boolean_missing" in error_codes
        ):
            actions.append(
                RepairAction(
                    name="rewrite_email_validation",
                    reason="Rewrite email validation into the canonical boolean-return pattern over the extracted email path.",
                )
            )

        if task_spec.family == "rest_cleanup" and has_prefix("rest_cleanup_excluded_key_reference::"):
            actions.append(
                RepairAction(
                    name="rewrite_rest_cleanup_keep_only",
                    reason="Rewrite REST cleanup into a generic keep-only loop that preserves only the requested keys.",
                )
            )

        if task_spec.family == "augment_existing_code" and (
            "augment_existing_code_invalid_json" in error_codes
            or "augment_existing_code_forbidden_workflow_state" in error_codes
            or "lua_syntax_error" in error_codes
            or "lua_runtime_error" in error_codes
            or has_prefix("augment_existing_code_missing_key::")
            or has_prefix("augment_existing_code_unexpected_key::")
        ):
            actions.append(
                RepairAction(
                    name="rewrite_augment_existing_code_envelope",
                    reason="Rewrite the JSON envelope into the canonical num/squared shape without extra keys or workflow-state reads.",
                )
            )

        deduped = []
        seen = set()
        for action in actions:
            if action.name in seen:
                continue
            deduped.append(action)
            seen.add(action.name)

        summary = "No repair actions were produced."
        if deduped:
            summary = "Apply actions: {0}".format(", ".join(action.name for action in deduped))

        return RepairPlan(
            repairable=bool(deduped),
            summary=summary,
            actions=deduped,
            source_errors=error_codes,
        )
