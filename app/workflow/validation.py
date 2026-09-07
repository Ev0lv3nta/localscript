from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from pydantic import TypeAdapter, ValidationError

from app.core.budgets import remaining_seconds, workflow_budget
from app.generation.backend_errors import BackendTimeout
from app.validation.lua_ast import analyze_lua_output
from app.validation.output import OutputParseError, parse_output
from app.validation.runtime import find_lua_binary, find_luac_binary, runtime_version
from app.validation.runtime_executor import execute_output
from app.workflow.contracts import (
    AcceptanceCase,
    CheckStatus,
    CodeCandidate,
    JsonValue,
    OutputContract,
    OutputFormat,
    OutputShape,
    TaskPlan,
    ValidationCheck,
    ValidationResult,
    WorkflowRoot,
)


class PolicyFinding(Protocol):
    @property
    def code(self) -> str: ...

    @property
    def message(self) -> str: ...


class PolicyResult(Protocol):
    @property
    def findings(self) -> tuple[PolicyFinding, ...]: ...


class RuntimeResult(Protocol):
    ok: bool
    value: object
    error_code: str
    error_message: str
    degraded: bool
    read_roots: tuple[str, ...]


PolicyAnalyzer = Callable[[str, str], PolicyResult]
RuntimeExecutor = Callable[..., RuntimeResult]
LuacLocator = Callable[[], str | None]
JSON_ADAPTER: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)


def _default_policy_analyzer(code: str, output_style: str) -> PolicyResult:
    return analyze_lua_output(code, output_style=output_style)


def _default_runtime_executor(
    code: str,
    context: object,
    output_style: str,
    output_shape: str | None = None,
) -> RuntimeResult:
    return execute_output(
        code=code,
        context=context,
        output_style=output_style,
        output_shape=output_shape,
    )


def _default_luac_locator() -> str | None:
    lua, luac = find_lua_binary(), find_luac_binary()
    if (
        lua
        and luac
        and runtime_version(lua) is not None
        and runtime_version(lua) == runtime_version(luac)
    ):
        return luac
    return None


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

    def validate(
        self,
        *,
        candidate: CodeCandidate,
        plan: TaskPlan,
        context: dict[str, JsonValue],
        examples: tuple[AcceptanceCase, ...] = (),
        source_roots: tuple[WorkflowRoot, ...] | None = None,
    ) -> ValidationResult:
        with workflow_budget(20):
            return self._validate_all(
                candidate=candidate,
                plan=plan,
                context=context,
                examples=examples,
                source_roots=source_roots,
            )

    def _validate_all(
        self,
        *,
        candidate: CodeCandidate,
        plan: TaskPlan,
        context: dict[str, JsonValue],
        examples: tuple[AcceptanceCase, ...],
        source_roots: tuple[WorkflowRoot, ...] | None,
    ) -> ValidationResult:
        request_result = self.validate_existing(
            candidate=candidate,
            output=plan.output,
            context=context,
            source_roots=source_roots,
        )
        checks = list(request_result.checks)
        observations: list[JsonValue] = [
            {"source": "request", **observation}
            for observation in request_result.observations
            if isinstance(observation, dict)
        ]
        if not request_result.ok:
            return ValidationResult(checks=tuple(checks), observations=tuple(observations))
        cases = [("caller", case) for case in examples] + [
            ("model", case) for case in plan.acceptance_cases
        ]
        for source, case in cases:
            remaining_seconds(20)
            try:
                execution = self._runtime_executor(
                    candidate.code,
                    case.context,
                    plan.output.format.value,
                    plan.output.shape.value,
                )
            except BackendTimeout:
                raise
            except Exception:
                checks.append(
                    self._failed(
                        f"{source}:{case.name}",
                        "sandbox_internal_error",
                        "The restricted runtime could not execute this acceptance case.",
                    )
                )
                continue

            if execution.degraded:
                checks.append(
                    self._failed(
                        f"{source}:{case.name}",
                        execution.error_code or "sandbox_runtime_missing",
                        execution.error_message
                        or "The required restricted Lua runtime is unavailable.",
                    )
                )
                continue
            if not execution.ok:
                checks.append(
                    self._failed(
                        f"{source}:{case.name}",
                        execution.error_code or "sandbox_execution_failed",
                        execution.error_message
                        or "The candidate failed in the restricted runtime.",
                    )
                )
                continue

            read_roots = self._execution_read_roots(execution)
            source_check = self._source_root_check(
                f"{source}:{case.name}", read_roots, source_roots
            )
            if source_check is not None:
                checks.append(source_check)
                continue

            try:
                actual = JSON_ADAPTER.validate_python(execution.value, strict=True)
            except ValidationError:
                checks.append(
                    self._failed(
                        f"{source}:{case.name}",
                        "sandbox_non_json_result",
                        "The restricted runtime returned a non-JSON value.",
                    )
                )
                continue
            # Наблюдение несёт и ожидание: без него ни ревизия, ни человек в трассе не видят,
            # чем именно ответ разошёлся с планом.
            observation: dict[str, JsonValue] = {
                "source": source,
                "case": case.name,
                "actual": actual,
                "expected": case.expected,
            }
            if read_roots:
                observation["read_roots"] = list(read_roots)
            observations.append(observation)
            shape_error = self._shape_error(
                actual,
                expected=plan.output.shape,
                nullable=plan.output.nullable,
            )
            if shape_error is not None:
                checks.append(
                    self._failed(
                        f"{source}:{case.name}",
                        shape_error,
                        "The candidate result does not satisfy the declared output shape.",
                    )
                )
            elif not self._json_equal(actual, case.expected):
                checks.append(
                    self._failed(
                        f"{source}:{case.name}",
                        "acceptance_result_mismatch",
                        "The candidate result does not match the expected JSON value.",
                    )
                )
            else:
                checks.append(self._passed(f"{source}:{case.name}"))

        return ValidationResult(checks=tuple(checks), observations=tuple(observations))

    def validate_existing(
        self,
        *,
        candidate: CodeCandidate,
        output: OutputContract,
        context: dict[str, JsonValue],
        source_roots: tuple[WorkflowRoot, ...] | None = None,
    ) -> ValidationResult:
        with workflow_budget(20):
            return self._validate_existing(
                candidate=candidate,
                output=output,
                context=context,
                source_roots=source_roots,
            )

    def _validate_existing(
        self,
        *,
        candidate: CodeCandidate,
        output: OutputContract,
        context: dict[str, JsonValue],
        source_roots: tuple[WorkflowRoot, ...] | None,
    ) -> ValidationResult:
        """Validate and execute caller-supplied code without inventing expected semantics."""
        checks: list[ValidationCheck] = []
        contract_check, chunks = self._validate_format(candidate.code, output.format)
        checks.append(contract_check)
        if contract_check.status is CheckStatus.FAILED:
            return ValidationResult(checks=tuple(checks))

        try:
            policy = self._policy_analyzer(candidate.code, output.format.value)
        except BackendTimeout:
            raise
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
        except BackendTimeout:
            raise
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

        read_roots = self._execution_read_roots(execution)
        source_check = self._source_root_check("source_roots", read_roots, source_roots)
        if source_check is not None:
            checks.append(source_check)
            return ValidationResult(
                checks=tuple(checks),
                observations=({"read_roots": list(read_roots)},),
            )

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
        observation: dict[str, JsonValue] = {"actual": actual}
        if read_roots:
            observation["read_roots"] = list(read_roots)
        return ValidationResult(checks=tuple(checks), observations=(observation,))

    @staticmethod
    def _execution_read_roots(execution: RuntimeResult) -> tuple[str, ...]:
        raw_roots = getattr(execution, "read_roots", ())
        return tuple(
            root
            for root in (WorkflowRoot.VARS.value, WorkflowRoot.INIT_VARIABLES.value)
            if root in raw_roots
        )

    @staticmethod
    def _source_root_check(
        name: str,
        read_roots: tuple[str, ...],
        source_roots: tuple[WorkflowRoot, ...] | None,
    ) -> ValidationCheck | None:
        if source_roots is None:
            return None
        selected = {root.value for root in source_roots}
        outside = [root for root in read_roots if root not in selected]
        if not outside:
            return None
        return DeterministicCandidateValidator._failed(
            name,
            "source_root_not_selected",
            "Candidate read an unselected workflow root: " + ", ".join(outside),
        )

    @staticmethod
    def _validate_format(
        code: str, output_format: OutputFormat
    ) -> tuple[ValidationCheck, tuple[str, ...]]:
        try:
            parsed = parse_output(code, output_format.value)
            return DeterministicCandidateValidator._passed("output_contract"), parsed.chunks
        except OutputParseError as error:
            return DeterministicCandidateValidator._failed(
                "output_contract", error.code, str(error)
            ), ()

    def _compile_chunks(self, chunks: tuple[str, ...]) -> ValidationCheck:
        try:
            luac = self._luac_locator()
        except BackendTimeout:
            raise
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
                    timeout=remaining_seconds(self._compile_timeout_seconds),
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
            except BackendTimeout:
                raise
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
