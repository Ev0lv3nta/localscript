from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from pydantic import TypeAdapter, ValidationError

from app.generation.backend_errors import (
    BackendBusy,
    BackendError,
    BackendModel,
    BackendTimeout,
    BackendUnavailable,
)
from app.workflow.context import ContextInspector
from app.workflow.contracts import (
    AcceptanceCase,
    CheckStatus,
    ClarificationRequest,
    CodeCandidate,
    ContextInventory,
    JsonValue,
    OutputContract,
    ReviewDecision,
    ReviewRejected,
    TaskPlan,
    ValidationCheck,
    ValidationResult,
    WorkflowDiagnostic,
    WorkflowResult,
    WorkflowRoot,
    WorkflowStage,
    WorkflowState,
    WorkflowStatus,
)
from app.workflow.roles import GeneratorRole, PlannerRole, ReviewerRole

JSON_ADAPTER: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)


class _InvalidJsonContext(Exception):
    """The caller-supplied workflow context is not a JSON value.

    Kept separate from other contract violations so a defect inside a workflow stage is never
    reported to the caller as a malformed request.
    """


class CandidateValidator(Protocol):
    def validate(
        self,
        *,
        candidate: CodeCandidate,
        plan: TaskPlan,
        context: dict[str, JsonValue],
        examples: tuple[AcceptanceCase, ...] = (),
        source_roots: tuple[WorkflowRoot, ...] | None = None,
    ) -> ValidationResult: ...


StageObserver = Callable[[WorkflowStage], None]


class WorkflowCoordinator:
    def __init__(
        self,
        *,
        planner: PlannerRole,
        generator: GeneratorRole,
        reviewer: ReviewerRole,
        validator: CandidateValidator,
        context_inspector: ContextInspector | None = None,
    ) -> None:
        self.planner = planner
        self.generator = generator
        self.reviewer = reviewer
        self.validator = validator
        self.context_inspector = context_inspector or ContextInspector()

    def run(
        self,
        *,
        prompt: str,
        context: object,
        clarification_answer: str | None = None,
        feedback: str | None = None,
        output: OutputContract | None = None,
        examples: tuple[AcceptanceCase, ...] = (),
        source_roots: tuple[WorkflowRoot, ...] | None = None,
        observe: StageObserver | None = None,
    ) -> WorkflowResult:
        state = WorkflowState()
        self._observe(observe, state.stage)
        try:
            try:
                json_context = JSON_ADAPTER.validate_python(context, strict=True)
            except ValidationError as error:
                raise _InvalidJsonContext from error
            if json_context is None:
                self._observe(observe, WorkflowStage.CLARIFICATION_REQUIRED)
                return WorkflowResult(
                    status=WorkflowStatus.CLARIFICATION_REQUIRED,
                    question=(
                        "Передайте JSON-контекст с исходными данными. Для задачи без входных "
                        "данных передайте явно пустой wf.vars или wf.initVariables."
                    ),
                )
            if not isinstance(json_context, dict):
                raise _InvalidJsonContext
            inventory = self.context_inspector.inventory(json_context)
            context_sample = self.context_inspector.sample(json_context)
            # Every role sees the confirmed task, including choices made after clarification.
            prompt = "\n".join(
                part
                for part in (
                    prompt,
                    f"Confirmed clarification: {clarification_answer}"
                    if clarification_answer
                    else "",
                    f"Requested changes: {feedback}" if feedback else "",
                )
                if part
            )
            self._observe(observe, WorkflowStage.PLANNING)
            decision = self.planner.run(
                prompt=prompt,
                context_sample=context_sample,
                inventory=inventory,
                clarification_answer=clarification_answer,
                feedback=feedback,
                output=output,
                examples=examples,
                source_roots=source_roots,
            )
            if isinstance(decision, ClarificationRequest):
                self._observe(observe, WorkflowStage.CLARIFICATION_REQUIRED)
                return WorkflowResult(
                    status=WorkflowStatus.CLARIFICATION_REQUIRED,
                    question=decision.question,
                )

            plan = decision
            actual_context: dict[str, JsonValue] = json_context
            state = WorkflowState(stage=WorkflowStage.PLANNED, plan=plan)
            self._observe(observe, state.stage)
            plan_check = self._validate_plan(
                plan,
                output=output,
                source_roots=source_roots,
            )
            if not plan_check.ok:
                # Противоречивый план восстановим ровно так же, как невалидный код: планировщик
                # получает свои же замечания и одну попытку. Отказывать сразу было асимметрично —
                # коду правка полагалась, а плану нет.
                self._observe(observe, WorkflowStage.PLANNING)
                decision = self.planner.run(
                    prompt=prompt,
                    context_sample=context_sample,
                    inventory=inventory,
                    clarification_answer=clarification_answer,
                    feedback=feedback,
                    output=output,
                    examples=examples,
                    source_roots=source_roots,
                    rejected_plan_findings=tuple(
                        f"{check.code}: {check.message}"
                        for check in plan_check.checks
                        if check.status is CheckStatus.FAILED and check.code and check.message
                    ),
                )
                if isinstance(decision, ClarificationRequest):
                    return self._failure(plan_check, WorkflowStage.PLANNED)
                plan = decision
                state = WorkflowState(stage=WorkflowStage.PLANNED, plan=plan)
                self._observe(observe, state.stage)
                plan_check = self._validate_plan(
                    plan,
                    output=output,
                    source_roots=source_roots,
                )
                if not plan_check.ok:
                    return self._failure(plan_check, WorkflowStage.PLANNED)

            self._observe(observe, WorkflowStage.GENERATING)
            candidate = self.generator.run(prompt=prompt, plan=plan)
            state = WorkflowState(
                stage=WorkflowStage.GENERATED,
                plan=plan,
                candidate=candidate,
            )
            self._observe(observe, state.stage)
            self._observe(observe, WorkflowStage.VALIDATING)
            validation = self.validator.validate(
                candidate=candidate,
                plan=plan,
                context=actual_context,
                examples=examples,
                source_roots=source_roots,
            )
            state = WorkflowState(
                stage=WorkflowStage.VALIDATED,
                plan=plan,
                candidate=candidate,
                validation=validation,
            )
            self._observe(observe, state.stage)
            if self._source_clarification_required(validation, inventory, source_roots):
                self._observe(observe, WorkflowStage.CLARIFICATION_REQUIRED)
                return self._source_clarification()
            review: ReviewDecision | None = None
            if validation.ok:
                self._observe(observe, WorkflowStage.REVIEWING)
                review = self.reviewer.run(
                    prompt=prompt,
                    plan=plan,
                    candidate=candidate,
                    validation=validation,
                )
                state = WorkflowState(
                    stage=WorkflowStage.REVIEWED,
                    plan=plan,
                    candidate=candidate,
                    validation=validation,
                    review=review,
                )
                self._observe(observe, state.stage)
                if not isinstance(review, ReviewRejected):
                    self._observe(observe, WorkflowStage.COMPLETED)
                    return WorkflowResult(
                        status=WorkflowStatus.COMPLETED,
                        code=candidate.code,
                        validation=validation,
                        output=plan.output,
                    )

            self._observe(observe, WorkflowStage.REVISING)
            revised = self.generator.revise(
                prompt=prompt,
                plan=plan,
                candidate=candidate,
                validation=validation,
                review=review,
            )
            state = WorkflowState(
                stage=WorkflowStage.REVISED,
                plan=plan,
                candidate=revised,
                revision_count=1,
            )
            self._observe(observe, state.stage)
            self._observe(observe, WorkflowStage.VALIDATING)
            revised_validation = self.validator.validate(
                candidate=revised,
                plan=plan,
                context=actual_context,
                examples=examples,
                source_roots=source_roots,
            )
            state = WorkflowState(
                stage=WorkflowStage.VALIDATED,
                plan=plan,
                candidate=revised,
                validation=revised_validation,
                revision_count=1,
            )
            self._observe(observe, WorkflowStage.VALIDATED)
            if self._source_clarification_required(revised_validation, inventory, source_roots):
                self._observe(observe, WorkflowStage.CLARIFICATION_REQUIRED)
                return self._source_clarification()
            if not revised_validation.ok:
                return self._failure(
                    revised_validation,
                    WorkflowStage.VALIDATED,
                    revision_count=1,
                )
            self._observe(observe, WorkflowStage.REVIEWING)
            revised_review = self.reviewer.run(
                prompt=prompt,
                plan=plan,
                candidate=revised,
                validation=revised_validation,
            )
            state = WorkflowState(
                stage=WorkflowStage.REVIEWED,
                plan=plan,
                candidate=revised,
                validation=revised_validation,
                review=revised_review,
                revision_count=1,
            )
            self._observe(observe, WorkflowStage.REVIEWED)
            if isinstance(revised_review, ReviewRejected):
                diagnostics = tuple(
                    WorkflowDiagnostic(
                        code=finding.code,
                        message=finding.message,
                        stage=WorkflowStage.REVIEWED,
                    )
                    for finding in revised_review.findings
                )
                self._observe(observe, WorkflowStage.FAILED)
                return WorkflowResult(
                    status=WorkflowStatus.VALIDATION_FAILED,
                    diagnostics=diagnostics,
                    validation=revised_validation,
                    revision_count=1,
                )
            self._observe(observe, WorkflowStage.COMPLETED)
            return WorkflowResult(
                status=WorkflowStatus.COMPLETED,
                code=revised.code,
                validation=revised_validation,
                revision_count=1,
                output=plan.output,
            )
        except (BackendBusy, BackendTimeout, BackendModel):
            self._observe(observe, WorkflowStage.FAILED)
            raise
        except BackendUnavailable as error:
            self._observe(observe, WorkflowStage.FAILED)
            return WorkflowResult(
                status=WorkflowStatus.BACKEND_UNAVAILABLE,
                diagnostics=(
                    WorkflowDiagnostic(
                        code=error.reason or "backend_unavailable",
                        message="The local model backend is unavailable.",
                        stage=state.stage,
                    ),
                ),
            )
        except BackendError as error:
            self._observe(observe, WorkflowStage.FAILED)
            return WorkflowResult(
                status=WorkflowStatus.VALIDATION_FAILED,
                diagnostics=(
                    WorkflowDiagnostic(
                        code=error.reason or "model_protocol_error",
                        message="A model role returned an invalid structured response.",
                        stage=state.stage,
                    ),
                ),
            )
        except _InvalidJsonContext:
            self._observe(observe, WorkflowStage.FAILED)
            return WorkflowResult(
                status=WorkflowStatus.VALIDATION_FAILED,
                diagnostics=(
                    WorkflowDiagnostic(
                        code="invalid_json_context",
                        message="Workflow context must be a valid JSON value.",
                        stage=state.stage,
                    ),
                ),
            )
        except ValidationError:
            self._observe(observe, WorkflowStage.FAILED)
            return WorkflowResult(
                status=WorkflowStatus.VALIDATION_FAILED,
                diagnostics=(
                    WorkflowDiagnostic(
                        code="workflow_contract_violation",
                        message="A workflow stage produced a value that violates its contract.",
                        stage=state.stage,
                    ),
                ),
            )

    @staticmethod
    def _validate_plan(
        plan: TaskPlan,
        *,
        output: OutputContract | None = None,
        source_roots: tuple[WorkflowRoot, ...] | None = None,
    ) -> ValidationResult:
        checks: list[ValidationCheck] = []
        if output is not None and plan.output != output:
            checks.append(
                ValidationCheck(
                    name="plan_contract",
                    status=CheckStatus.FAILED,
                    code="caller_output_contract_changed",
                    message="The plan must preserve the caller's output contract.",
                )
            )
        if source_roots is not None:
            selected = set(source_roots)
            outside = sorted(
                {path.root for path in plan.inputs if path.root not in selected},
                key=lambda root: root.value,
            )
            if outside:
                checks.append(
                    ValidationCheck(
                        name="plan_contract",
                        status=CheckStatus.FAILED,
                        code="plan_source_root_not_selected",
                        message=(
                            "Plan declares an unselected workflow root: "
                            + ", ".join(root.value for root in outside)
                        ),
                    )
                )
        case_names = [case.name for case in plan.acceptance_cases]
        if len(case_names) != len(set(case_names)):
            checks.append(
                ValidationCheck(
                    name="plan_contract",
                    status=CheckStatus.FAILED,
                    code="duplicate_acceptance_case",
                    message="Acceptance case names must be unique.",
                )
            )
        for case in plan.acceptance_cases:
            if not WorkflowCoordinator._matches_output_contract(
                case.expected,
                shape=plan.output.shape.value,
                nullable=plan.output.nullable,
            ):
                checks.append(
                    ValidationCheck(
                        name="plan_contract",
                        status=CheckStatus.FAILED,
                        code="acceptance_output_contract_mismatch",
                        message=(
                            f"Acceptance case `{case.name}` contradicts the declared output contract."
                        ),
                    )
                )
        if not checks:
            checks.append(ValidationCheck(name="plan_contract", status=CheckStatus.PASSED))
        return ValidationResult(checks=tuple(checks))

    @staticmethod
    def _source_clarification_required(
        validation: ValidationResult,
        inventory: ContextInventory,
        source_roots: tuple[WorkflowRoot, ...] | None,
    ) -> bool:
        if source_roots is not None or not ContextInspector.ambiguous_paths(inventory):
            return False
        for observation in validation.observations:
            if not isinstance(observation, dict) or observation.get("source") != "request":
                continue
            read_roots = observation.get("read_roots")
            return isinstance(read_roots, list) and len(set(read_roots)) == 1
        return False

    @staticmethod
    def _source_clarification() -> WorkflowResult:
        return WorkflowResult(
            status=WorkflowStatus.CLARIFICATION_REQUIRED,
            question="Какой источник использовать для совпадающих путей?",
            source_choices=(WorkflowRoot.VARS, WorkflowRoot.INIT_VARIABLES),
        )

    @staticmethod
    def _matches_output_contract(value: JsonValue, *, shape: str, nullable: bool) -> bool:
        if value is None:
            return nullable
        if shape == "array":
            return isinstance(value, list)
        if shape == "object":
            return isinstance(value, dict)
        return not isinstance(value, (list, dict))

    @staticmethod
    def _failure(
        validation: ValidationResult,
        stage: WorkflowStage,
        *,
        revision_count: int = 0,
    ) -> WorkflowResult:
        diagnostics = tuple(
            WorkflowDiagnostic(
                code=check.code or "validation_failed",
                message=check.message or "Validation failed.",
                stage=stage,
            )
            for check in validation.checks
            if check.status is CheckStatus.FAILED
        )
        status = (
            WorkflowStatus.POLICY_REJECTED
            if any(
                check.name == "ast_policy" and check.status is CheckStatus.FAILED
                for check in validation.checks
            )
            else WorkflowStatus.VALIDATION_FAILED
        )
        return WorkflowResult(
            status=status,
            diagnostics=diagnostics,
            validation=validation,
            revision_count=revision_count,
        )

    @staticmethod
    def _observe(observer: StageObserver | None, stage: WorkflowStage) -> None:
        if observer is not None and stage not in {
            WorkflowStage.PLANNED,
            WorkflowStage.GENERATED,
            WorkflowStage.VALIDATED,
            WorkflowStage.REVIEWED,
            WorkflowStage.REVISED,
        }:
            observer(stage)
