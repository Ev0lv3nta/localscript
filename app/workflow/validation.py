from __future__ import annotations

import importlib
import json
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

from pydantic import TypeAdapter, ValidationError

from app.validation.runtime import find_luac_binary
from app.workflow.contracts import (
    CheckStatus,
    CodeCandidate,
    JsonValue,
    OutputContract,
    OutputFormat,
    OutputShape,
    TaskPlan,
    ValidationCheck,
    ValidationResult,
)


class PolicyFinding(Protocol):
    code: str
    message: str


class PolicyResult(Protocol):
    findings: tuple[PolicyFinding, ...]


class RuntimeResult(Protocol):
    ok: bool
    value: object
    error_code: str
    error_message: str
    degraded: bool


PolicyAnalyzer = Callable[[str, str], PolicyResult]
RuntimeExecutor = Callable[..., RuntimeResult]
LuacLocator = Callable[[], str | None]
JSON_ADAPTER: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)


def _default_policy_analyzer(code: str, output_style: str) -> PolicyResult:
    # PR1 owns the parser and its policy. A dynamic import keeps this adapter independently
    # testable while PR1 and PR2 are developed on separate branches.
    module = importlib.import_module("app.validation.lua_ast")
    analyzer = cast(Callable[..., PolicyResult], module.analyze_lua_output)
    return analyzer(code, output_style=output_style)


def _default_runtime_executor(
    code: str,
    context: object,
    output_style: str,
    output_shape: str | None = None,
) -> RuntimeResult:
    module = importlib.import_module("app.validation.runtime_executor")
    executor = cast(Callable[..., RuntimeResult], module.execute_output)
    return executor(
        code=code,
        context=context,
        output_style=output_style,
        output_shape=output_shape,
    )


def _default_luac_locator() -> str | None:
    return find_luac_binary()


class _DuplicateEnvelopeKey(ValueError):
    pass


def _json_object_without_duplicates(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateEnvelopeKey(key)
        result[key] = value
    return result


class DeterministicCandidateValidator:
    """Fail-closed AST, luac, and executable acceptance validation."""

    def __init__(
        self,
        *,
        policy_analyzer: PolicyAnalyzer | None = None,
        runtime_executor: RuntimeExecutor | None = None,
        luac_locator: LuacLocator | None = None,
        compile_timeout_seconds: float = 5.0,
    ) -> None:
        self._policy_analyzer = policy_analyzer or _default_policy_analyzer
        self._runtime_executor = runtime_executor or _default_runtime_executor
        self._luac_locator = luac_locator or _default_luac_locator
        self._compile_timeout_seconds = compile_timeout_seconds

    def validate(self, *, candidate: CodeCandidate, plan: TaskPlan) -> ValidationResult:
        checks: list[ValidationCheck] = []
        observations: list[JsonValue] = []

        contract_check, chunks = self._validate_format(candidate.code, plan.output.format)
        checks.append(contract_check)
        if contract_check.status is CheckStatus.FAILED:
            return ValidationResult(checks=tuple(checks))

        try:
            policy = self._policy_analyzer(candidate.code, plan.output.format.value)
        except Exception:
            checks.append(
                self._failed(
                    "ast_policy",
                    "policy_internal_error",
                    "Lua AST policy could not analyze the candidate.",
                )
            )
            return ValidationResult(checks=tuple(checks))

        if policy.findings:
            checks.extend(
                self._failed("ast_policy", finding.code, finding.message)
                for finding in policy.findings
            )
            return ValidationResult(checks=tuple(checks))
        checks.append(self._passed("ast_policy"))

        luac_check = self._compile_chunks(chunks)
        checks.append(luac_check)
        if luac_check.status is CheckStatus.FAILED:
            return ValidationResult(checks=tuple(checks))

        for case in plan.acceptance_cases:
            try:
                execution = self._runtime_executor(
                    candidate.code,
                    case.context,
                    plan.output.format.value,
                    plan.output.shape.value,
                )
            except Exception:
                checks.append(
                    self._failed(
                        f"acceptance:{case.name}",
                        "sandbox_internal_error",
                        "The restricted runtime could not execute this acceptance case.",
                    )
                )
                continue

            if execution.degraded:
                checks.append(
                    self._failed(
                        f"acceptance:{case.name}",
                        execution.error_code or "sandbox_runtime_missing",
                        execution.error_message
                        or "The required restricted Lua runtime is unavailable.",
                    )
                )
                continue
            if not execution.ok:
                checks.append(
                    self._failed(
                        f"acceptance:{case.name}",
                        execution.error_code or "sandbox_execution_failed",
                        execution.error_message
                        or "The candidate failed in the restricted runtime.",
                    )
                )
                continue

            try:
                actual = JSON_ADAPTER.validate_python(execution.value, strict=True)
            except ValidationError:
                checks.append(
                    self._failed(
                        f"acceptance:{case.name}",
                        "sandbox_non_json_result",
                        "The restricted runtime returned a non-JSON value.",
                    )
                )
                continue
            observations.append({"case": case.name, "actual": actual})
            shape_error = self._shape_error(
                actual,
                expected=plan.output.shape,
                nullable=plan.output.nullable,
            )
            if shape_error is not None:
                checks.append(
                    self._failed(
                        f"acceptance:{case.name}",
                        shape_error,
                        "The candidate result does not satisfy the declared output shape.",
                    )
                )
            elif not self._json_equal(actual, case.expected):
                checks.append(
                    self._failed(
                        f"acceptance:{case.name}",
                        "acceptance_result_mismatch",
                        "The candidate result does not match the expected JSON value.",
                    )
                )
            else:
                checks.append(self._passed(f"acceptance:{case.name}"))

        return ValidationResult(checks=tuple(checks), observations=tuple(observations))

    def validate_existing(
        self,
        *,
        candidate: CodeCandidate,
        output: OutputContract,
        context: dict[str, JsonValue],
    ) -> ValidationResult:
        """Validate and execute caller-supplied code without inventing expected semantics."""
        checks: list[ValidationCheck] = []
        contract_check, chunks = self._validate_format(candidate.code, output.format)
        checks.append(contract_check)
        if contract_check.status is CheckStatus.FAILED:
            return ValidationResult(checks=tuple(checks))

        try:
            policy = self._policy_analyzer(candidate.code, output.format.value)
        except Exception:
            checks.append(
                self._failed(
                    "ast_policy",
                    "policy_internal_error",
                    "Lua AST policy could not analyze the candidate.",
                )
            )
            return ValidationResult(checks=tuple(checks))
        if policy.findings:
            checks.extend(
                self._failed("ast_policy", finding.code, finding.message)
                for finding in policy.findings
            )
            return ValidationResult(checks=tuple(checks))
        checks.append(self._passed("ast_policy"))

        luac_check = self._compile_chunks(chunks)
        checks.append(luac_check)
        if luac_check.status is CheckStatus.FAILED:
            return ValidationResult(checks=tuple(checks))

        try:
            execution = self._runtime_executor(
                candidate.code,
                context,
                output.format.value,
                output.shape.value,
            )
        except Exception:
            checks.append(
                self._failed(
                    "sandbox",
                    "sandbox_internal_error",
                    "The restricted runtime could not execute the candidate.",
                )
            )
            return ValidationResult(checks=tuple(checks))
        if execution.degraded:
            checks.append(
                self._failed(
                    "sandbox",
                    execution.error_code or "sandbox_runtime_missing",
                    execution.error_message or "The restricted Lua runtime is unavailable.",
                )
            )
            return ValidationResult(checks=tuple(checks))
        if not execution.ok:
            checks.append(
                self._failed(
                    "sandbox",
                    execution.error_code or "sandbox_execution_failed",
                    execution.error_message or "The candidate failed in the restricted runtime.",
                )
            )
            return ValidationResult(checks=tuple(checks))

        try:
            actual = JSON_ADAPTER.validate_python(execution.value, strict=True)
        except ValidationError:
            checks.append(
                self._failed(
                    "sandbox",
                    "sandbox_non_json_result",
                    "The restricted runtime returned a non-JSON value.",
                )
            )
            return ValidationResult(checks=tuple(checks))
        shape_error = self._shape_error(
            actual,
            expected=output.shape,
            nullable=output.nullable,
        )
        if shape_error is not None:
            checks.append(
                self._failed(
                    "sandbox",
                    shape_error,
                    "The candidate result does not satisfy the declared output shape.",
                )
            )
        else:
            checks.append(self._passed("sandbox"))
        return ValidationResult(
            checks=tuple(checks),
            observations=({"actual": actual},),
        )

    @staticmethod
    def _validate_format(
        code: str,
        output_format: OutputFormat,
    ) -> tuple[ValidationCheck, tuple[str, ...]]:
        if output_format is OutputFormat.LUA_BLOCK:
            if code.startswith("lua{") and code.endswith("}lua"):
                return (
                    DeterministicCandidateValidator._failed(
                        "output_contract",
                        "lua_block_wrapper_forbidden",
                        "A raw Lua block must not use a lua{...}lua wrapper.",
                    ),
                    (),
                )
            return DeterministicCandidateValidator._passed("output_contract"), (code,)

        try:
            payload = json.loads(code, object_pairs_hook=_json_object_without_duplicates)
        except _DuplicateEnvelopeKey:
            return (
                DeterministicCandidateValidator._failed(
                    "output_contract",
                    "json_envelope_duplicate_key",
                    "JSON envelope keys must be unique.",
                ),
                (),
            )
        except (json.JSONDecodeError, TypeError, RecursionError):
            return (
                DeterministicCandidateValidator._failed(
                    "output_contract",
                    "json_envelope_invalid",
                    "The candidate is not a valid JSON envelope.",
                ),
                (),
            )
        if not isinstance(payload, dict) or not payload:
            return (
                DeterministicCandidateValidator._failed(
                    "output_contract",
                    "json_envelope_not_object",
                    "A JSON envelope must be a non-empty object.",
                ),
                (),
            )

        chunks: list[str] = []
        for key, value in payload.items():
            if not key or not isinstance(value, str):
                return (
                    DeterministicCandidateValidator._failed(
                        "output_contract",
                        "json_envelope_value_invalid",
                        "Every envelope key must contain one wrapped Lua chunk.",
                    ),
                    (),
                )
            if not value.startswith("lua{") or not value.endswith("}lua"):
                return (
                    DeterministicCandidateValidator._failed(
                        "output_contract",
                        "json_envelope_wrapper_invalid",
                        "Every envelope value must use a lua{...}lua wrapper.",
                    ),
                    (),
                )
            chunk = value[4:-4]
            if not chunk.strip():
                return (
                    DeterministicCandidateValidator._failed(
                        "output_contract",
                        "lua_chunk_empty",
                        "Envelope Lua chunks must not be empty.",
                    ),
                    (),
                )
            chunks.append(chunk)
        return DeterministicCandidateValidator._passed("output_contract"), tuple(chunks)

    def _compile_chunks(self, chunks: tuple[str, ...]) -> ValidationCheck:
        try:
            luac = self._luac_locator()
        except Exception:
            return self._failed(
                "luac",
                "luac_lookup_failed",
                "The required luac syntax checker could not be resolved.",
            )
        if not luac:
            return self._failed(
                "luac",
                "luac_runtime_missing",
                "The required luac syntax checker is unavailable.",
            )

        for index, chunk in enumerate(chunks, start=1):
            temp_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    "w",
                    encoding="utf-8",
                    suffix=".lua",
                    delete=False,
                ) as handle:
                    handle.write(chunk)
                    temp_path = Path(handle.name)
                completed = subprocess.run(
                    [luac, "-p", str(temp_path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=self._compile_timeout_seconds,
                    check=False,
                    close_fds=True,
                )
            except subprocess.TimeoutExpired:
                return self._failed(
                    "luac",
                    "luac_timeout",
                    "luac exceeded the syntax-check timeout.",
                )
            except (OSError, ValueError):
                return self._failed(
                    "luac",
                    "luac_execution_failed",
                    "luac could not check the candidate.",
                )
            except Exception:
                return self._failed(
                    "luac",
                    "luac_internal_error",
                    "luac failed unexpectedly while checking the candidate.",
                )
            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)

            if completed.returncode != 0:
                return self._failed(
                    "luac",
                    "lua_syntax_error",
                    f"Lua chunk #{index} failed the luac syntax check.",
                )
        return self._passed("luac")

    @staticmethod
    def _shape_error(
        value: JsonValue,
        *,
        expected: OutputShape,
        nullable: bool,
    ) -> str | None:
        if value is None:
            return None if nullable else "output_null_forbidden"
        if expected is OutputShape.ARRAY and not isinstance(value, list):
            return "output_shape_array_mismatch"
        if expected is OutputShape.OBJECT and not isinstance(value, dict):
            return "output_shape_object_mismatch"
        if expected is OutputShape.SCALAR and isinstance(value, (list, dict)):
            return "output_shape_scalar_mismatch"
        return None

    @classmethod
    def _json_equal(cls, actual: JsonValue, expected: JsonValue) -> bool:
        if isinstance(actual, bool) or isinstance(expected, bool):
            return type(actual) is type(expected) and actual == expected
        if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
            return actual == expected
        if type(actual) is not type(expected):
            return False
        if isinstance(actual, list) and isinstance(expected, list):
            return len(actual) == len(expected) and all(
                cls._json_equal(left, right) for left, right in zip(actual, expected, strict=True)
            )
        if isinstance(actual, dict) and isinstance(expected, dict):
            return actual.keys() == expected.keys() and all(
                cls._json_equal(actual[key], expected[key]) for key in actual
            )
        return actual == expected

    @staticmethod
    def _passed(name: str) -> ValidationCheck:
        return ValidationCheck(name=name, status=CheckStatus.PASSED)

    @staticmethod
    def _failed(name: str, code: str, message: str) -> ValidationCheck:
        return ValidationCheck(
            name=name,
            status=CheckStatus.FAILED,
            code=code,
            message=message,
        )
