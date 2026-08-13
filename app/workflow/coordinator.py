from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from pydantic import TypeAdapter, ValidationError

from app.generation.backend_errors import BackendError, BackendUnavailable
from app.workflow.context import ContextInspector
from app.workflow.contracts import (
    CheckStatus,
    ClarificationRequest,
    CodeCandidate,
    JsonValue,
    ReviewDecision,
    ReviewRejected,
    TaskPlan,
    ValidationCheck,
    ValidationResult,
    WorkflowDiagnostic,
    WorkflowResult,
    WorkflowStage,
    WorkflowState,
    WorkflowStatus,
)
from app.workflow.roles import GeneratorRole, PlannerRole, ReviewerRole

JSON_ADAPTER: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)


class CandidateValidator(Protocol):
    def validate(self, *, candidate: CodeCandidate, plan: TaskPlan) -> ValidationResult: ...


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
        observe: StageObserver | None = None,
    ) -> WorkflowResult:
        state = WorkflowState()
        self._observe(observe, state.stage)
        try:
            json_context = JSON_ADAPTER.validate_python(context, strict=True)
            inventory = self.context_inspector.inventory(json_context)
            context_sample = self.context_inspector.sample(json_context)
            decision = self.planner.run(
                prompt=prompt,
                context_sample=context_sample,
                inventory=inventory,
                clarification_answer=clarification_answer,
                feedback=feedback,
            )
            if isinstance(decision, ClarificationRequest):
                self._observe(observe, WorkflowStage.CLARIFICATION_REQUIRED)
                return WorkflowResult(
                    status=WorkflowStatus.CLARIFICATION_REQUIRED,
                    question=decision.question,
                )

            plan = decision
            state = WorkflowState(stage=WorkflowStage.PLANNED, plan=plan)
            self._observe(observe, state.stage)
            plan_check = self._validate_plan(plan)
            if not plan_check.ok:
                return self._failure(plan_check, WorkflowStage.PLANNED)

            candidate = self.generator.run(prompt=prompt, plan=plan)
            state = WorkflowState(
                stage=WorkflowStage.GENERATED,
                plan=plan,
                candidate=candidate,
            )
            self._observe(observe, state.stage)
            validation = self.validator.validate(candidate=candidate, plan=plan)
            state = WorkflowState(
                stage=WorkflowStage.VALIDATED,
                plan=plan,
                candidate=candidate,
                validation=validation,
            )
            self._observe(observe, state.stage)
            review: ReviewDecision | None = None
            if validation.ok:
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
                    )

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
            revised_validation = self.validator.validate(candidate=revised, plan=plan)
            state = WorkflowState(
                stage=WorkflowStage.VALIDATED,
                plan=plan,
                candidate=revised,
                validation=revised_validation,
                revision_count=1,
            )
            self._observe(observe, WorkflowStage.VALIDATED)
            if not revised_validation.ok:
                return self._failure(
                    revised_validation,
                    WorkflowStage.VALIDATED,
                    revision_count=1,
                )
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
            )
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
        except ValidationError:
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

    @staticmethod
    def _validate_plan(plan: TaskPlan) -> ValidationResult:
        checks: list[ValidationCheck] = []
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
                            "Acceptance case `{0}` contradicts the declared output contract."
                        ).format(case.name),
                    )
                )
        if not checks:
            checks.append(ValidationCheck(name="plan_contract", status=CheckStatus.PASSED))
        return ValidationResult(checks=tuple(checks))

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
            if any(diagnostic.code.startswith("policy_") for diagnostic in diagnostics)
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
        if observer is not None:
            observer(stage)
