import json
import os
import re
import subprocess
import tempfile
from copy import deepcopy

from app.validation.oracles import UNSUPPORTED, build_expected_result, compare_expected_and_actual
from app.validation.base import BaseValidator, ValidationReport, ValidatorContext
from app.validation.runtime import find_lua_binary, find_luac_binary
from app.validation.runtime_executor import DANGEROUS_LUA_PATTERNS, execute_output
from app.generation.taskspec import TaskResolutionSource


SHADOW_PROTECTED_GLOBALS = ("table", "string", "math", "utf8", "_utils", "wf")


def _strip_lua_wrapper(code):
    if not isinstance(code, str):
        return ""
    stripped = code.strip()
    if stripped.startswith("lua{") and stripped.endswith("}lua"):
        return stripped[4:-4]
    return stripped


def _extract_lua_chunks(code, output_style):
    if output_style != "json_envelope":
        if not isinstance(code, str) or not code.strip():
            return []
        return [_strip_lua_wrapper(code)]

    if not isinstance(code, str):
        return []

    try:
        payload = json.loads(code)
    except (ValueError, TypeError, RecursionError):
        return []

    if not isinstance(payload, dict) or not payload:
        return []

    chunks = []
    for value in payload.values():
        if not (
            isinstance(value, str)
            and value.startswith("lua{")
            and value.endswith("}lua")
        ):
            return []
        chunks.append(_strip_lua_wrapper(value))
    return chunks


def _find_lua_binary():
    return find_lua_binary()


def _find_luac_binary():
    return find_luac_binary()


class ContractValidator(BaseValidator):
    name = "contract"

    def validate(self, code, context):
        report = ValidationReport()
        if not isinstance(code, str):
            report.add(self.name, "error", "contract_not_string", "Generated output must be a string.")
            return report
        if not code.strip():
            report.add(self.name, "error", "contract_empty_code", "Generated code is empty.")
        return report


class JsonEnvelopeValidator(BaseValidator):
    name = "json_envelope"

    def validate(self, code, context):
        report = ValidationReport()
        if context.task_spec.output_style != "json_envelope":
            return report

        if not isinstance(code, str):
            report.add(
                self.name,
                "error",
                "json_envelope_invalid",
                "Envelope must be provided as a JSON string.",
            )
            return report

        try:
            payload = json.loads(code)
        except (ValueError, TypeError, RecursionError) as exc:
            report.add(self.name, "error", "json_envelope_invalid", "Envelope is not valid JSON: {0}".format(exc))
            return report

        if not isinstance(payload, dict):
            report.add(self.name, "error", "json_envelope_not_object", "Envelope must be a JSON object.")
            return report

        if not payload:
            report.add(self.name, "error", "json_envelope_empty", "Envelope must contain at least one Lua value.")
            return report

        for key, value in payload.items():
            if not isinstance(value, str):
                report.add(
                    self.name,
                    "error",
                    "json_envelope_value_not_string",
                    "Envelope value for `{0}` must be a string.".format(key),
                )
                continue
            if not value.startswith("lua{") or not value.endswith("}lua"):
                report.add(
                    self.name,
                    "error",
                    "json_envelope_value_not_lua_wrapper",
                    "Envelope value for `{0}` must use `lua{{...}}lua`.".format(key),
                )
        return report


class DomainLintValidator(BaseValidator):
    name = "domain_lint"

    def validate(self, code, context):
        report = ValidationReport()
        if "$." in code or "$[" in code:
            report.add(self.name, "error", "jsonpath_forbidden", "JsonPath is forbidden in LocalScript.")
        if "```" in code:
            report.add(self.name, "error", "markdown_fence_forbidden", "Markdown fences are not allowed.")
        if "ctx.body" in code:
            report.add(
                self.name,
                "error",
                "ctx_body_forbidden",
                "ctx.body is not an allowed workflow namespace; use wf.vars or wf.initVariables.",
            )
        if "workflow.variables" in code:
            report.add(
                self.name,
                "error",
                "workflow_variables_forbidden",
                "workflow.variables is not an allowed workflow namespace; use wf.vars.",
            )

        if context.task_spec.target_root == "wf.initVariables" and "wf.initVariables" not in code:
            report.add(
                self.name,
                "error",
                "init_variables_missing",
                "Task expects launch variables through wf.initVariables.",
            )
        if context.task_spec.target_root == "wf.initVariables" and "wf.vars" in code:
            report.add(
                self.name,
                "error",
                "unexpected_root_reference::wf.vars",
                "Task target root is wf.initVariables, but generated code still references wf.vars.",
            )
        if context.task_spec.target_root == "wf.vars" and "wf.initVariables" in code:
            report.add(
                self.name,
                "error",
                "unexpected_root_reference::wf.initVariables",
                "Task target root is wf.vars, but generated code still references wf.initVariables.",
            )

        if context.task_spec.target_root == "wf.vars" and "wf.vars" not in code and context.task_spec.output_style != "json_envelope":
            report.add(
                self.name,
                "warning",
                "wf_vars_missing",
                "Task context points to wf.vars, but generated code does not reference it directly.",
            )
        return report


class DangerousStdlibValidator(BaseValidator):
    name = "dangerous_stdlib"

    def validate(self, code, context):
        report = ValidationReport()
        for token, error_code, message in DANGEROUS_LUA_PATTERNS:
            if token in code:
                report.add(
                    self.name,
                    "error",
                    error_code,
                    message,
                )
        return report


class ShadowedStdlibValidator(BaseValidator):
    name = "shadowed_stdlib"

    def validate(self, code, context):
        report = ValidationReport()
        for identifier in SHADOW_PROTECTED_GLOBALS:
            if re.search(r"\blocal\s+{0}\b".format(re.escape(identifier)), code or ""):
                report.add(
                    self.name,
                    "error",
                    "shadowed_stdlib_local::{0}".format(identifier),
                    "Local variable `{0}` shadows a protected runtime/global identifier.".format(identifier),
                )
        return report


class LengthBudgetValidator(BaseValidator):
    name = "length_budget"

    def validate(self, code, context):
        report = ValidationReport()
        max_chars = max(1024, int(context.profile.num_predict) * 16)
        current_size = len(code)

        if current_size > max_chars:
            report.add(
                self.name,
                "error",
                "code_too_long",
                "Generated code length {0} exceeds budget {1}.".format(current_size, max_chars),
            )
        elif current_size > int(max_chars * 0.75):
            report.add(
                self.name,
                "warning",
                "code_near_budget",
                "Generated code length {0} is close to budget {1}.".format(current_size, max_chars),
            )
        return report


class LuaSyntaxValidator(BaseValidator):
    name = "lua_syntax"

    def validate(self, code, context):
        report = ValidationReport()
        lua_binary = _find_lua_binary()
        luac_binary = _find_luac_binary()
        if not lua_binary and not luac_binary:
            report.add(
                self.name,
                "warning",
                "lua_runtime_missing",
                "Lua runtime is unavailable; syntax validation ran in degraded mode.",
            )
            return report

        for index, chunk in enumerate(_extract_lua_chunks(code, context.task_spec.output_style), start=1):
            if not chunk.strip():
                report.add(
                    self.name,
                    "error",
                    "lua_chunk_empty",
                    "Lua chunk #{0} is empty.".format(index),
                )
                continue

            with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".lua", delete=False) as handle:
                handle.write(chunk)
                temp_path = handle.name

            try:
                command = (
                    [luac_binary, "-p", temp_path]
                    if luac_binary
                    else [lua_binary, "-e", "assert(loadfile(arg[1]))", "--", temp_path]
                )
                completed = subprocess.run(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=5,
                )
            finally:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

            if completed.returncode != 0:
                report.add(
                    self.name,
                    "error",
                    "lua_syntax_error",
                    "Lua chunk #{0} failed to compile: {1}".format(index, completed.stderr.strip()),
                )
        return report


class ScenarioValidator(BaseValidator):
    name = "scenario"

    def validate(self, code, context):
        report = ValidationReport()
        if context.task_spec.output_style != "json_envelope" and "return" not in code:
            report.add(self.name, "error", "return_missing", "Lua output must contain a return statement.")

        if context.task_spec.family == "filter_discount_markdown" and "_utils.array.new()" not in code:
            report.add(
                self.name,
                "error",
                "array_constructor_missing",
                "Filtering case must initialize result via _utils.array.new().",
            )

        if context.task_spec.family == "augment_existing_code" and context.task_spec.output_style != "json_envelope":
            report.add(
                self.name,
                "error",
                "expected_json_envelope",
                "Augment-existing-code task must return a JSON envelope.",
            )

        if context.task_spec.family == "regex_extract" and "string.match" not in code:
            report.add(
                self.name,
                "error",
                "regex_extract_missing_string_match",
                "Regex extraction tasks must use string.match.",
            )

        if context.task_spec.family == "email_validation":
            if "string.match" not in code or "~= nil" not in code:
                report.add(
                    self.name,
                    "error",
                    "email_validation_boolean_missing",
                    "Email validation must return an explicit boolean via string.match(... ) ~= nil.",
                )

        if context.task_spec.family == "normalize_email_string":
            if "string.lower" not in code and ":lower()" not in code:
                report.add(
                    self.name,
                    "error",
                    "normalize_email_lower_missing",
                    "Email normalization must convert the final scalar to lower case.",
                )
            if "string.gsub" not in code:
                report.add(
                    self.name,
                    "error",
                    "normalize_email_trim_missing",
                    "Email normalization must trim surrounding whitespace via string.gsub.",
                )

        if context.task_spec.family == "table_transform" and "_utils.array.new()" not in code:
            report.add(
                self.name,
                "error",
                "table_transform_missing_array_constructor",
                "Table transform tasks must build arrays via _utils.array.new().",
            )

        if context.task_spec.family == "conditional_array_projection" and "_utils.array.new()" not in code:
            report.add(
                self.name,
                "error",
                "conditional_projection_missing_array_constructor",
                "Conditional array projection must build arrays via _utils.array.new().",
            )

        if context.task_spec.family == "rest_cleanup":
            hints = context.task_spec.generation_hints or {}
            keep_keys = {str(key) for key in hints.get("keep_keys", []) if key}
            for key in hints.get("available_keys", []):
                key = str(key)
                if key in keep_keys:
                    continue
                if (
                    '"{0}"'.format(key) in code
                    or "'{0}'".format(key) in code
                    or ".{0}".format(key) in code
                ):
                    report.add(
                        self.name,
                        "error",
                        "rest_cleanup_excluded_key_reference::{0}".format(key),
                        "REST cleanup must not reference excluded key `{0}` directly; preserve only requested keys generically.".format(key),
                    )

        if context.task_spec.family == "augment_existing_code":
            if "wf.vars" in code or "wf.initVariables" in code:
                report.add(
                    self.name,
                    "error",
                    "augment_existing_code_forbidden_workflow_state",
                    "Augment-existing-code must use the extracted literal and must not read workflow state.",
                )
            try:
                payload = json.loads(code)
            except json.JSONDecodeError:
                report.add(
                    self.name,
                    "error",
                    "augment_existing_code_invalid_json",
                    "Augment-existing-code tasks must return a valid JSON envelope.",
                )
            else:
                for key in ["num", "squared"]:
                    if key not in payload:
                        report.add(
                            self.name,
                            "error",
                            "augment_existing_code_missing_key::{0}".format(key),
                            "Augment-existing-code tasks must preserve `{0}` in the JSON envelope.".format(key),
                        )
                for key in payload.keys():
                    if key not in {"num", "squared"}:
                        report.add(
                            self.name,
                            "error",
                            "augment_existing_code_unexpected_key::{0}".format(key),
                            "Augment-existing-code tasks must not add unexpected envelope key `{0}`.".format(key),
                        )
        return report


class SemanticScenarioValidator(BaseValidator):
    name = "semantic"

    def validate(self, code, context):
        report = ValidationReport()
        if context.source_context is None:
            return report
        if (
            getattr(context.task_spec, "resolution_source", None)
            is TaskResolutionSource.PLANNER
        ):
            return report
        if getattr(context.task_spec, "ambiguity_notes", None) and context.task_spec.family is None:
            return report
        expected = build_expected_result(context.task_spec, context.source_context)
        if expected is UNSUPPORTED:
            return report

        execution = execute_output(
            code=code,
            context=context.source_context,
            output_style=context.task_spec.output_style,
        )
        if execution.degraded:
            report.add(
                self.name,
                "warning",
                execution.error_code or "semantic_runtime_missing",
                execution.error_message or "Semantic execution ran in degraded mode.",
            )
            return report

        if not execution.ok:
            report.add(
                self.name,
                "error",
                execution.error_code or "semantic_runtime_error",
                execution.error_message or "Generated Lua failed during semantic execution.",
            )
            return report

        if not compare_expected_and_actual(expected, execution.value):
            report.add(
                self.name,
                "error",
                "semantic_mismatch",
                "Generated Lua result does not match the semantic oracle.",
            )
        return report


class GenericSemanticValidator(BaseValidator):
    name = "generic_semantic"

    def validate(self, code, context):
        report = ValidationReport()
        semantic_checks = context.planner_semantic_checks or []
        if context.source_context is None or not semantic_checks:
            return report

        execution = execute_output(
            code=code,
            context=context.source_context,
            output_style=context.task_spec.output_style,
        )
        if execution.degraded:
            report.add(
                self.name,
                "warning",
                execution.error_code or "generic_semantic_runtime_missing",
                execution.error_message or "Generic semantic execution ran in degraded mode.",
            )
            return report
        if not execution.ok:
            report.add(
                self.name,
                "error",
                execution.error_code or "generic_semantic_runtime_error",
                execution.error_message or "Generated Lua failed during generic semantic execution.",
            )
            return report

        for check in semantic_checks:
            if not isinstance(check, dict):
                continue
            kind = check.get("kind")
            if kind == "return_shape":
                self._validate_return_shape(report, execution.value, check.get("value"))
            elif kind == "scalar_equals":
                self._validate_scalar_equals(report, execution.value, check.get("value"))
            elif kind == "object_subset":
                self._validate_object_subset(report, execution.value, check.get("value"))
            elif kind == "array_equals":
                self._validate_array_equals(report, execution.value, check.get("value"))
            elif kind == "unordered_set_equals":
                self._validate_unordered_set_equals(report, execution.value, check.get("value"))
            elif kind == "empty_array_on_missing_source":
                self._validate_empty_array_behavior(report, code, context, check.get("source_path"))
            elif kind == "contains_fields":
                self._validate_contains_fields(report, execution.value, check.get("fields", []))
            elif kind == "numeric_tolerance":
                self._validate_numeric_tolerance(report, execution.value, check.get("value"), check.get("tolerance"))
        return report

    @staticmethod
    def _validate_return_shape(report, value, expected_shape):
        if expected_shape == "array" and not isinstance(value, list):
            report.add(
                "generic_semantic",
                "error",
                "generic_return_shape_array_mismatch",
                "Planner expected an array result.",
            )
        elif expected_shape == "object" and not isinstance(value, dict):
            report.add(
                "generic_semantic",
                "error",
                "generic_return_shape_object_mismatch",
                "Planner expected an object result.",
            )
        elif expected_shape == "scalar" and isinstance(value, (list, dict)):
            report.add(
                "generic_semantic",
                "error",
                "generic_return_shape_scalar_mismatch",
                "Planner expected a scalar result.",
            )

    def _validate_empty_array_behavior(self, report, code, context, source_path):
        if not source_path:
            return
        for variant_name, replacement in [("missing", None), ("empty", [])]:
            variant_context = deepcopy(context.source_context)
            self._set_path_value(variant_context, source_path, replacement)
            execution = execute_output(
                code=code,
                context=variant_context,
                output_style=context.task_spec.output_style,
            )
            if not execution.ok or execution.degraded or execution.value != []:
                report.add(
                    self.name,
                    "error",
                    "generic_empty_array_behavior_mismatch",
                    "Expected an empty array for {0} input at `{1}`.".format(variant_name, source_path),
                )
                return

    @staticmethod
    def _validate_contains_fields(report, value, fields):
        if not fields:
            return
        if isinstance(value, dict):
            values = [value]
        elif isinstance(value, list):
            values = [item for item in value if isinstance(item, dict)]
        else:
            values = []
        for item in values:
            for field in fields:
                if field not in item:
                    report.add(
                        "generic_semantic",
                        "error",
                        "generic_missing_field",
                        "Expected field `{0}` is missing in the semantic result.".format(field),
                    )
                    return

    @staticmethod
    def _validate_scalar_equals(report, value, expected):
        if value != expected:
            report.add(
                "generic_semantic",
                "error",
                "generic_scalar_equals_mismatch",
                "Scalar result does not match the expected value.",
            )

    @staticmethod
    def _validate_object_subset(report, value, expected):
        if not isinstance(value, dict) or not isinstance(expected, dict):
            report.add(
                "generic_semantic",
                "error",
                "generic_object_subset_mismatch",
                "Expected an object containing the requested subset.",
            )
            return
        for key, expected_value in expected.items():
            if value.get(key) != expected_value:
                report.add(
                    "generic_semantic",
                    "error",
                    "generic_object_subset_mismatch",
                    "Object result is missing the expected key subset.",
                )
                return

    @staticmethod
    def _validate_array_equals(report, value, expected):
        if value != expected:
            report.add(
                "generic_semantic",
                "error",
                "generic_array_equals_mismatch",
                "Array result does not match the expected ordered values.",
            )

    @staticmethod
    def _validate_unordered_set_equals(report, value, expected):
        if not isinstance(value, list) or not isinstance(expected, list):
            report.add(
                "generic_semantic",
                "error",
                "generic_unordered_set_mismatch",
                "Expected an array result for unordered-set comparison.",
            )
            return
        if sorted(value) != sorted(expected):
            report.add(
                "generic_semantic",
                "error",
                "generic_unordered_set_mismatch",
                "Array result does not match the expected unordered values.",
            )

    @staticmethod
    def _validate_numeric_tolerance(report, value, expected, tolerance):
        if not isinstance(value, (int, float)) or not isinstance(expected, (int, float)):
            report.add(
                "generic_semantic",
                "error",
                "generic_numeric_tolerance_mismatch",
                "Expected a numeric result for tolerance comparison.",
            )
            return
        tolerance = float(tolerance or 0.0)
        if abs(float(value) - float(expected)) > tolerance:
            report.add(
                "generic_semantic",
                "error",
                "generic_numeric_tolerance_mismatch",
                "Numeric result is outside the allowed tolerance.",
            )

    @staticmethod
    def _set_path_value(payload, path, replacement):
        parts = (path or "").split(".")
        current = payload
        for part in parts[:-1]:
            if not isinstance(current, dict):
                return
            current = current.get(part)
            if current is None:
                return
        if isinstance(current, dict) and parts:
            current[parts[-1]] = replacement


class ValidationPipeline:
    def __init__(self, validators=None):
        self.validators = validators or [
            ContractValidator(),
            JsonEnvelopeValidator(),
            DomainLintValidator(),
            ShadowedStdlibValidator(),
            DangerousStdlibValidator(),
            LengthBudgetValidator(),
            LuaSyntaxValidator(),
            ScenarioValidator(),
            GenericSemanticValidator(),
            SemanticScenarioValidator(),
        ]

    def run(self, code, task_spec, profile, source_context=None, prompt="", planner_semantic_checks=None):
        context = ValidatorContext(
            profile=profile,
            task_spec=task_spec,
            source_context=source_context,
            prompt=prompt,
            planner_semantic_checks=planner_semantic_checks,
        )
        report = ValidationReport()
        for validator in self.validators:
            if isinstance(validator, (GenericSemanticValidator, SemanticScenarioValidator)) and report.has_errors:
                continue
            validator_report = validator.validate(code, context)
            report.messages.extend(validator_report.messages)
            if validator_report.has_errors and isinstance(
                validator,
                (ContractValidator, JsonEnvelopeValidator),
            ):
                break
        return report
