import uuid
from dataclasses import dataclass, field, replace
from typing import List, Optional

from app.core.sessions import SessionStore
from app.core.verifier import verify_code
from app.domain.outcomes import (
    Diagnostic,
    DiagnosticSeverity,
    GenerationOutcome,
    GenerationStatus,
    ValidationOutcome,
    ValidationStatus,
)
from app.generation.backend_errors import BackendUnavailable
from app.generation.candidates import GeneratedCandidate, PlannerClarification
from app.generation.clarification import ClarificationPolicy
from app.generation.context_reducer import ContextReducer
from app.generation.extractor import TaskExtractor
from app.generation.formatter import OutputFormatter
from app.generation.model_chain import SameModelChain
from app.generation.stages import GenerationExecution, GenerationStage
from app.generation.task_resolver import TaskResolver
from app.repair.loop import RepairLoop
from app.validation.validators import ValidationPipeline

SAFE_FALLBACK_CODE = "-- judged-safe fallback\nreturn nil"


# Backward-compatible import name for callers of the pre-0.2 engine.
BackendUnavailableError = BackendUnavailable


@dataclass
class GenerationResult:
    code: str = ""
    trace_id: str = ""
    session_id: str = ""
    strategy: str = ""
    verification_errors: List[str] = field(default_factory=list)
    validation_report: dict = field(default_factory=dict)
    repair_rounds: int = 0
    degraded_mode: bool = False
    status: str = "completed"
    question: str = ""
    assumptions: List[str] = field(default_factory=list)
    session_summary: dict = field(default_factory=dict)
    clarification_suggested: bool = False
    assumption_risk: str = "low"
    outcome: Optional[GenerationOutcome] = None


class GenerationEngine:
    def __init__(
        self,
        profile,
        trace_store,
        backend,
        session_store=None,
        extractor=None,
        formatter=None,
    ):
        self.profile = profile
        self.trace_store = trace_store
        self.backend = backend
        self.session_store = session_store or SessionStore(root=self.trace_store.root.parent / "sessions")
        self.extractor = extractor or TaskExtractor()
        self.formatter = formatter or OutputFormatter()
        self.validation_pipeline = ValidationPipeline()
        self.context_reducer = ContextReducer()
        self.task_resolver = TaskResolver()
        self.clarification_policy = ClarificationPolicy()
        self.model_chain = SameModelChain(
            backend=self.backend,
            validation_pipeline=self.validation_pipeline,
            formatter=self.formatter,
            task_resolver=self.task_resolver,
        )
        self.repair_loop = RepairLoop(
            validation_pipeline=self.validation_pipeline,
            formatter=self.formatter,
        )

    def generate(self, prompt, context=None, session_id=None, feedback=None):
        return self._run_generation(
            prompt=prompt,
            context=context,
            session_id=session_id,
            feedback=feedback,
            clarification_answer=None,
            rich_mode=False,
        )

    def generate_rich(
        self,
        prompt=None,
        context=None,
        session_id=None,
        feedback=None,
        clarification_answer=None,
    ):
        return self._run_generation(
            prompt=prompt,
            context=context,
            session_id=session_id,
            feedback=feedback,
            clarification_answer=clarification_answer,
            rich_mode=True,
        )

    def _run_generation(self, prompt, context, session_id, feedback, clarification_answer, rich_mode):
        session_id = session_id or uuid.uuid4().hex
        with self.session_store.transaction(session_id) as session_state:
            return self._run_generation_locked(
                prompt=prompt,
                context=context,
                session_id=session_id,
                feedback=feedback,
                clarification_answer=clarification_answer,
                rich_mode=rich_mode,
                session_state=session_state,
            )

    def _run_generation_locked(
        self,
        prompt,
        context,
        session_id,
        feedback,
        clarification_answer,
        rich_mode,
        session_state,
    ):
        session_state = self._prepare_session_state(
            session_id=session_id,
            prompt=prompt,
            context=context,
            feedback=feedback,
            clarification_answer=clarification_answer,
            session_state=session_state,
        )
        execution = GenerationExecution()
        execution.transition(GenerationStage.SESSION_READY)
        if rich_mode and session_state.get("open_clarification_question") and not clarification_answer:
            execution.transition(GenerationStage.CLARIFICATION_REQUIRED)
            execution.transition(GenerationStage.OUTCOME_FINALIZED)
            return self._build_clarification_result(session_state)

        effective_prompt = session_state["original_task"]
        effective_context = session_state.get("context")
        task_spec = self.extractor.extract(prompt=effective_prompt, context=effective_context)
        if session_state.get("clarified_root"):
            task_spec = task_spec.model_copy(
                update={"target_root": session_state["clarified_root"]}
            )
        session_state["normalized_task"] = task_spec.normalized_prompt
        session_state["extracted_slots"] = task_spec.model_dump()
        clarification_decision = self.clarification_policy.evaluate(
            task_spec,
            session_state,
        )
        assumption_risk = self.clarification_policy.assumption_risk(
            task_spec,
            clarification_decision,
        )

        rule_based_question = clarification_decision.question
        if rich_mode and rule_based_question:
            task_spec = self.task_resolver.resolve(task_spec)
            session_state["extracted_slots"] = task_spec.model_dump(mode="json")
            execution.transition(GenerationStage.TASK_RESOLVED)
            assumptions = list(task_spec.assumptions) + [
                "Rule-based ambiguity policy requested a clarification before generation.",
            ]
            assumptions.extend(clarification_decision.assumptions)
            return self._complete_clarification(
                prompt=prompt,
                effective_prompt=effective_prompt,
                effective_context=effective_context,
                feedback=feedback,
                clarification_answer=clarification_answer,
                task_spec=task_spec,
                session_state=session_state,
                execution=execution,
                question=rule_based_question,
                assumptions=assumptions,
                assumption_risk=assumption_risk,
            )

        generation_step = self._generate_candidate(
            effective_prompt=effective_prompt,
            effective_context=effective_context,
            task_spec=task_spec,
            session_state=session_state,
            rich_mode=rich_mode,
            feedback=feedback,
        )
        task_spec = generation_step.task_spec
        session_state["extracted_slots"] = task_spec.model_dump(mode="json")
        execution.transition(GenerationStage.TASK_RESOLVED)
        if isinstance(generation_step, PlannerClarification):
            return self._complete_clarification(
                prompt=prompt,
                effective_prompt=effective_prompt,
                effective_context=effective_context,
                feedback=feedback,
                clarification_answer=clarification_answer,
                task_spec=task_spec,
                session_state=session_state,
                execution=execution,
                question=generation_step.question,
                assumptions=list(generation_step.assumptions),
                assumption_risk=assumption_risk,
                planner_payload=generation_step.planner_payload,
                repair_trace=generation_step.repair_trace,
                rules_applied=generation_step.rules_applied,
                semantic_checks=generation_step.semantic_checks,
            )

        candidate = generation_step
        execution.transition(GenerationStage.CANDIDATE_GENERATED)
        candidate, repaired = self._repair_candidate(
            candidate,
            effective_prompt=effective_prompt,
            effective_context=effective_context,
        )
        if repaired:
            execution.transition(GenerationStage.CANDIDATE_REPAIRED)
        verification_errors = []
        for error_code in candidate.validation_report.error_codes() + verify_code(candidate.code):
            if error_code not in verification_errors:
                verification_errors.append(error_code)
        degraded_mode = self._is_degraded(
            strategy=candidate.strategy,
            validation_report=candidate.validation_report,
        )
        status = self._build_status(strategy=candidate.strategy, degraded_mode=degraded_mode)
        session_state["status"] = status
        session_state["planner_state"] = candidate.planner_payload
        session_state["previous_candidate_code"] = candidate.code
        session_state["previous_validation_report"] = candidate.validation_report.to_dict()
        session_state["assumptions"] = candidate.assumptions
        session_state["last_strategy"] = candidate.strategy
        session_state["open_clarification_question"] = ""
        execution.transition(GenerationStage.OUTCOME_FINALIZED)
        trace_payload = {
            "prompt": prompt,
            "effective_prompt": effective_prompt,
            "context": effective_context,
            "feedback": feedback,
            "clarification_answer": clarification_answer,
            "status": status,
            "strategy": candidate.strategy,
            "task_spec": task_spec.model_dump(),
            "assumptions": candidate.assumptions,
            "verification_errors": verification_errors,
            "validation_report": candidate.validation_report.to_dict(),
            "repair_rounds": candidate.repair_rounds,
            "repair_trace": candidate.repair_trace,
            "planner": candidate.planner_payload,
            "critic": candidate.critic_payload,
            "rules_applied": candidate.rules_applied,
            "examples_used": candidate.examples_used,
            "critic_rules_used": candidate.critic_rules_used,
            "semantic_checks": candidate.semantic_checks,
            "backend_error": None,
            "model": self.profile.model,
            "fallback_model": self.profile.fallback_model,
            "code": candidate.code,
            "session_id": session_id,
            "degraded_mode": degraded_mode,
            "clarification_suggested": bool(rule_based_question),
            "assumption_risk": assumption_risk,
            "stage_events": execution.snapshot(),
        }
        trace_id = self.trace_store.write(trace_payload)
        session_state["latest_trace_id"] = trace_id
        session_state["trace_ids"].append(trace_id)
        outcome = self._build_generation_outcome(
            strategy=candidate.strategy,
            code=candidate.code,
            validation_report=candidate.validation_report,
        )
        return GenerationResult(
            code=candidate.code,
            trace_id=trace_id,
            session_id=session_id,
            strategy=candidate.strategy,
            verification_errors=verification_errors,
            validation_report=candidate.validation_report.to_dict(),
            repair_rounds=candidate.repair_rounds,
            degraded_mode=degraded_mode,
            status=status,
            assumptions=candidate.assumptions,
            session_summary=self.build_session_summary(session_state),
            clarification_suggested=bool(rule_based_question),
            assumption_risk=assumption_risk,
            outcome=outcome,
        )

    def _generate_candidate(
        self,
        *,
        effective_prompt,
        effective_context,
        task_spec,
        session_state,
        rich_mode,
        feedback,
    ):
        if task_spec.safety_fallback:
            task_spec = self.task_resolver.resolve(task_spec)
            code = self.formatter.format(SAFE_FALLBACK_CODE, task_spec.output_style)
            validation_report = self.validation_pipeline.run(
                code=code,
                task_spec=task_spec,
                profile=self.profile,
                source_context=effective_context,
                prompt=effective_prompt,
                planner_semantic_checks=None,
            )
            assumptions = list(task_spec.assumptions)
            assumptions.append("Unsafe or malformed task was denied by the safety guardrail.")
            return GeneratedCandidate(
                task_spec=task_spec,
                code=code,
                validation_report=validation_report,
                strategy="safe_fallback",
                assumptions=assumptions,
            )

        chain_result = self.model_chain.run(
            prompt=effective_prompt,
            context=effective_context,
            task_spec=task_spec,
            profile=self.profile,
            max_rounds=max(1, int(getattr(self.profile, "model_chain_rounds", 1))),
            session_state=session_state,
            stop_on_clarification=rich_mode,
        )
        task_spec = getattr(chain_result, "task_spec", None) or self.task_resolver.resolve(
            task_spec,
            planner=getattr(chain_result, "planner", None),
        )
        planner_payload = chain_result.planner
        if chain_result.status == "clarification_needed":
            assumptions = list(task_spec.assumptions) + [
                "Planner detected an ambiguity and requested one clarification before code generation.",
            ]
            assumptions.extend(planner_payload.get("assumptions", []))
            return PlannerClarification(
                task_spec=task_spec,
                question=chain_result.question,
                assumptions=tuple(assumptions),
                planner_payload=planner_payload,
                repair_trace=chain_result.history,
                rules_applied=chain_result.rules_applied,
                semantic_checks=chain_result.semantic_checks,
            )

        assumptions = list(task_spec.assumptions) + [
            "The same-model planner/writer chain handled the request.",
        ]
        assumptions.extend(planner_payload.get("assumptions", []))
        return GeneratedCandidate(
            task_spec=task_spec,
            code=chain_result.code,
            validation_report=chain_result.validation_report,
            strategy="feedback_revision" if feedback else "ollama_chain",
            assumptions=assumptions,
            repair_trace=chain_result.history,
            repair_rounds=chain_result.rounds,
            rules_applied=chain_result.rules_applied,
            examples_used=chain_result.examples_used,
            critic_rules_used=chain_result.critic_rules_used,
            planner_payload=planner_payload,
            critic_payload=chain_result.critic,
            semantic_checks=chain_result.semantic_checks,
        )

    def _repair_candidate(self, candidate, *, effective_prompt, effective_context):
        if not candidate.validation_report.has_errors or candidate.strategy == "safe_fallback":
            return candidate, False

        remaining_rounds = max(
            int(self.profile.max_repair_rounds) - int(candidate.repair_rounds),
            1,
        )
        repair_result = self.repair_loop.run(
            code=candidate.code,
            task_spec=candidate.task_spec,
            validation_report=candidate.validation_report,
            profile=self.profile,
            max_rounds=remaining_rounds,
            source_context=effective_context,
            prompt=effective_prompt,
            planner_semantic_checks=candidate.semantic_checks,
        )
        return (
            replace(
                candidate,
                code=repair_result.code,
                validation_report=repair_result.validation_report,
                repair_trace=candidate.repair_trace
                + [{"stage": "deterministic_repair", "history": repair_result.history}],
                repair_rounds=candidate.repair_rounds + repair_result.rounds,
            ),
            True,
        )

    def _complete_clarification(
        self,
        *,
        prompt,
        effective_prompt,
        effective_context,
        feedback,
        clarification_answer,
        task_spec,
        session_state,
        execution,
        question,
        assumptions,
        assumption_risk,
        planner_payload=None,
        repair_trace=None,
        rules_applied=None,
        semantic_checks=None,
    ):
        planner_payload = planner_payload or {}
        repair_trace = repair_trace or []
        rules_applied = rules_applied or []
        semantic_checks = semantic_checks or []
        session_id = session_state["session_id"]
        validation_report = {
            "has_errors": False,
            "has_warnings": False,
            "messages": [],
        }

        execution.transition(GenerationStage.CLARIFICATION_REQUIRED)
        session_state["status"] = "clarification_needed"
        session_state["planner_state"] = planner_payload
        session_state["open_clarification_question"] = question
        session_state["assumptions"] = assumptions
        execution.transition(GenerationStage.OUTCOME_FINALIZED)

        trace_id = self.trace_store.write(
            {
                "prompt": prompt,
                "effective_prompt": effective_prompt,
                "context": effective_context,
                "feedback": feedback,
                "clarification_answer": clarification_answer,
                "status": "clarification_needed",
                "strategy": "clarification",
                "task_spec": task_spec.model_dump(mode="json"),
                "assumptions": assumptions,
                "verification_errors": [],
                "validation_report": validation_report,
                "repair_rounds": 0,
                "repair_trace": repair_trace,
                "planner": planner_payload,
                "critic": {},
                "rules_applied": rules_applied,
                "examples_used": [],
                "critic_rules_used": [],
                "backend_error": None,
                "model": self.profile.model,
                "fallback_model": self.profile.fallback_model,
                "code": "",
                "question": question,
                "semantic_checks": semantic_checks,
                "session_id": session_id,
                "degraded_mode": False,
                "clarification_suggested": True,
                "assumption_risk": assumption_risk,
                "stage_events": execution.snapshot(),
            }
        )
        session_state["latest_trace_id"] = trace_id
        session_state["trace_ids"].append(trace_id)
        session_state["last_strategy"] = "clarification"
        return GenerationResult(
            code="",
            trace_id=trace_id,
            session_id=session_id,
            strategy="clarification",
            verification_errors=[],
            validation_report=validation_report,
            repair_rounds=0,
            degraded_mode=False,
            status="clarification_needed",
            question=question,
            assumptions=assumptions,
            session_summary=self.build_session_summary(session_state),
            clarification_suggested=True,
            assumption_risk=assumption_risk,
            outcome=self._build_clarification_outcome(question),
        )

    def _prepare_session_state(
        self,
        session_id,
        prompt,
        context,
        feedback,
        clarification_answer,
        session_state=None,
    ):
        if session_state is None:
            session_state = self.session_store.read(session_id) or {}
        if not session_state:
            session_state.update(
                {
                    "session_id": session_id,
                    "status": "pending",
                    "original_task": prompt or "",
                    "latest_prompt": prompt or "",
                    "context": context,
                    "normalized_task": "",
                    "context_summary": {},
                    "extracted_slots": {},
                    "planner_state": {},
                    "open_clarification_question": "",
                    "clarification_history": [],
                    "feedback_history": [],
                    "previous_candidate_code": "",
                    "previous_validation_report": {},
                    "latest_trace_id": None,
                    "trace_ids": [],
                    "last_strategy": None,
                    "assumptions": [],
                }
            )
        if prompt:
            if not session_state.get("original_task"):
                session_state["original_task"] = prompt
            session_state["latest_prompt"] = prompt
        if context is not None:
            session_state["context"] = context
        if feedback:
            session_state.setdefault("feedback_history", []).append(feedback)
        if clarification_answer:
            open_question = session_state.get("open_clarification_question") or ""
            session_state.setdefault("clarification_history", []).append(
                {"question": open_question, "answer": clarification_answer}
            )
            session_state["open_clarification_question"] = ""
            lowered_answer = clarification_answer.lower()
            if "wf.initvariables" in lowered_answer:
                session_state["clarified_root"] = "wf.initVariables"
            elif "wf.vars" in lowered_answer:
                session_state["clarified_root"] = "wf.vars"
        if not session_state.get("original_task"):
            raise ValueError("original task is required to initialize a session")
        return session_state

    def _build_clarification_result(self, session_state):
        return GenerationResult(
            code="",
            trace_id=session_state.get("latest_trace_id", ""),
            session_id=session_state["session_id"],
            strategy="clarification",
            verification_errors=[],
            validation_report={"has_errors": False, "has_warnings": False, "messages": []},
            repair_rounds=0,
            degraded_mode=False,
            status="clarification_needed",
            question=session_state.get("open_clarification_question", ""),
            assumptions=session_state.get("assumptions", []),
            session_summary=self.build_session_summary(session_state),
            clarification_suggested=True,
            assumption_risk="high",
            outcome=self._build_clarification_outcome(
                session_state.get("open_clarification_question", "")
            ),
        )

    def analyze(self, prompt, context=None):
        task_spec = self.extractor.extract(prompt=prompt, context=context)
        reduced_context = self.context_reducer.reduce(context, task_spec)
        clarification_question = self.clarification_policy.evaluate(
            task_spec,
            {},
        ).question or None
        return {
            "normalized_prompt": task_spec.normalized_prompt,
            "suggested_strategy": (
                "clarification"
                if clarification_question
                else ("safe_fallback" if task_spec.safety_fallback else "ollama_chain")
            ),
            "clarification_question": clarification_question,
            "task_spec": task_spec.model_dump(),
            "reduced_context": reduced_context,
            "available_paths": task_spec.context_paths,
            "assumptions": task_spec.assumptions,
            "ambiguity_notes": task_spec.ambiguity_notes,
        }

    @staticmethod
    def _is_degraded(strategy, validation_report):
        if strategy == "safe_fallback":
            return True
        degraded_codes = {
            "lua_runtime_missing",
            "semantic_runtime_missing",
        }
        for message in validation_report.messages:
            if message.code in degraded_codes:
                return True
        return False

    @staticmethod
    def _build_status(strategy, degraded_mode):
        if strategy == "safe_fallback":
            return "failed_safe"
        if degraded_mode:
            return "degraded_completed"
        return "completed"

    @staticmethod
    def _build_clarification_outcome(question):
        return GenerationOutcome(
            status=GenerationStatus.CLARIFICATION_REQUIRED,
            validation=ValidationOutcome(status=ValidationStatus.NOT_RUN),
            question=question,
        )

    @staticmethod
    def _build_generation_outcome(strategy, code, validation_report):
        findings = tuple(
            Diagnostic(
                code=message.code,
                message=message.message,
                severity=DiagnosticSeverity(message.level),
                stage=message.validator,
            )
            for message in validation_report.messages
        )
        incomplete_codes = {
            "lua_runtime_missing",
            "semantic_runtime_missing",
        }
        if validation_report.has_errors:
            validation_status = ValidationStatus.FAILED
        elif any(message.code in incomplete_codes for message in validation_report.messages):
            validation_status = ValidationStatus.INCOMPLETE
        else:
            validation_status = ValidationStatus.PASSED

        validation = ValidationOutcome(
            status=validation_status,
            findings=findings,
        )
        if strategy == "safe_fallback":
            generation_status = GenerationStatus.POLICY_REJECTED
            outcome_code = None
        elif validation_status is ValidationStatus.PASSED:
            generation_status = GenerationStatus.COMPLETED
            outcome_code = code
        else:
            generation_status = GenerationStatus.VALIDATION_FAILED
            outcome_code = None

        return GenerationOutcome(
            status=generation_status,
            validation=validation,
            code=outcome_code,
            diagnostics=findings,
        )

    @staticmethod
    def build_session_summary(session_state):
        return {
            "session_id": session_state["session_id"],
            "status": session_state.get("status", "pending"),
            "original_task": session_state.get("original_task"),
            "latest_trace_id": session_state.get("latest_trace_id"),
            "last_strategy": session_state.get("last_strategy"),
            "open_clarification_question": session_state.get("open_clarification_question") or None,
            "clarification_history": session_state.get("clarification_history", []),
            "feedback_history": session_state.get("feedback_history", []),
            "assumptions": session_state.get("assumptions", []),
        }
