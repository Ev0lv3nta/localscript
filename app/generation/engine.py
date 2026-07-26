from dataclasses import dataclass, field
from typing import List
import uuid

from app.core.sessions import SessionStore
from app.core.verifier import verify_code
from app.generation.extractor import TaskExtractor
from app.generation.formatter import OutputFormatter
from app.generation.model_chain import SameModelChain
from app.generation.context_reducer import ContextReducer
from app.repair.loop import RepairLoop
from app.validation.validators import ValidationPipeline


SAFE_FALLBACK_CODE = "-- judged-safe fallback\nreturn nil"


class BackendUnavailableError(RuntimeError):
    pass


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
        self.model_chain = SameModelChain(
            backend=self.backend,
            validation_pipeline=self.validation_pipeline,
            formatter=self.formatter,
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
        session_state = self._prepare_session_state(
            session_id=session_id,
            prompt=prompt,
            context=context,
            feedback=feedback,
            clarification_answer=clarification_answer,
        )
        if rich_mode and session_state.get("open_clarification_question") and not clarification_answer:
            return self._build_clarification_result(session_state)

        effective_prompt = session_state["original_task"]
        effective_context = session_state.get("context")
        task_spec = self.extractor.extract(prompt=effective_prompt, context=effective_context)
        if session_state.get("clarified_root"):
            task_spec.target_root = session_state["clarified_root"]
        session_state["normalized_task"] = task_spec.normalized_prompt
        session_state["extracted_slots"] = task_spec.dict()
        backend_error = None
        rules_applied = []
        examples_used = []
        critic_rules_used = []
        planner_payload = {}
        critic_payload = {}
        repair_trace = []
        repair_rounds = 0
        semantic_checks = []
        clarification_policy = self._rule_based_clarification(task_spec, session_state)
        assumption_risk = self._assumption_risk(task_spec, clarification_policy)

        rule_based_question = clarification_policy.get("question", "")
        if rich_mode and rule_based_question:
            assumptions = list(task_spec.assumptions) + [
                "Rule-based ambiguity policy requested a clarification before generation.",
            ]
            assumptions.extend(clarification_policy.get("assumptions", []))
            session_state["status"] = "clarification_needed"
            session_state["open_clarification_question"] = rule_based_question
            session_state["assumptions"] = assumptions
            trace_payload = {
                "prompt": prompt,
                "effective_prompt": effective_prompt,
                "context": effective_context,
                "feedback": feedback,
                "clarification_answer": clarification_answer,
                "status": "clarification_needed",
                "strategy": "clarification",
                "task_spec": task_spec.dict(),
                "assumptions": assumptions,
                "verification_errors": [],
                "validation_report": {"has_errors": False, "has_warnings": False, "messages": []},
                "repair_rounds": 0,
                "repair_trace": [],
                "planner": {},
                "critic": {},
                "rules_applied": [],
                "examples_used": [],
                "critic_rules_used": [],
                "backend_error": None,
                "model": self.profile.model,
                "fallback_model": self.profile.fallback_model,
                "code": "",
                "question": rule_based_question,
                "semantic_checks": [],
                "session_id": session_id,
                "degraded_mode": False,
                "clarification_suggested": True,
                "assumption_risk": assumption_risk,
            }
            trace_id = self.trace_store.write(trace_payload)
            session_state["latest_trace_id"] = trace_id
            session_state["trace_ids"].append(trace_id)
            session_state["last_strategy"] = "clarification"
            self.session_store.write(session_id, session_state)
            return GenerationResult(
                code="",
                trace_id=trace_id,
                session_id=session_id,
                strategy="clarification",
                verification_errors=[],
                validation_report={"has_errors": False, "has_warnings": False, "messages": []},
                repair_rounds=0,
                degraded_mode=False,
                status="clarification_needed",
                question=rule_based_question,
                assumptions=assumptions,
                session_summary=self.build_session_summary(session_state),
                clarification_suggested=True,
                assumption_risk=assumption_risk,
            )

        if task_spec.safety_fallback:
            raw_code = SAFE_FALLBACK_CODE
            strategy = "safe_fallback"
            assumptions = list(task_spec.assumptions)
            assumptions.append("Unsafe or malformed task was denied by the safety guardrail.")
            code = self.formatter.format(raw_code, task_spec.output_style)
            validation_report = self.validation_pipeline.run(
                code=code,
                task_spec=task_spec,
                profile=self.profile,
                source_context=effective_context,
                prompt=effective_prompt,
                planner_semantic_checks=None,
            )
        else:
            try:
                chain_result = self.model_chain.run(
                    prompt=effective_prompt,
                    context=effective_context,
                    task_spec=task_spec,
                    profile=self.profile,
                    max_rounds=max(1, int(getattr(self.profile, "model_chain_rounds", 1))),
                    session_state=session_state,
                    stop_on_clarification=rich_mode,
                )
                if chain_result.status == "clarification_needed":
                    assumptions = list(task_spec.assumptions) + [
                        "Planner detected an ambiguity and requested one clarification before code generation.",
                    ]
                    assumptions.extend(chain_result.planner.get("assumptions", []))
                    planner_payload = chain_result.planner
                    semantic_checks = chain_result.semantic_checks
                    session_state["status"] = "clarification_needed"
                    session_state["planner_state"] = planner_payload
                    session_state["open_clarification_question"] = chain_result.question
                    session_state["assumptions"] = assumptions
                    trace_payload = {
                        "prompt": prompt,
                        "effective_prompt": effective_prompt,
                        "context": effective_context,
                        "feedback": feedback,
                        "clarification_answer": clarification_answer,
                        "status": "clarification_needed",
                        "strategy": "clarification",
                        "task_spec": task_spec.dict(),
                        "assumptions": assumptions,
                        "verification_errors": [],
                        "validation_report": {"has_errors": False, "has_warnings": False, "messages": []},
                        "repair_rounds": 0,
                        "repair_trace": chain_result.history,
                        "planner": planner_payload,
                        "critic": {},
                        "rules_applied": chain_result.rules_applied,
                        "examples_used": [],
                        "critic_rules_used": [],
                        "backend_error": None,
                        "model": self.profile.model,
                        "fallback_model": self.profile.fallback_model,
                        "code": "",
                        "question": chain_result.question,
                        "semantic_checks": semantic_checks,
                        "session_id": session_id,
                        "degraded_mode": False,
                        "clarification_suggested": True,
                        "assumption_risk": assumption_risk,
                    }
                    trace_id = self.trace_store.write(trace_payload)
                    session_state["latest_trace_id"] = trace_id
                    session_state["trace_ids"].append(trace_id)
                    session_state["last_strategy"] = "clarification"
                    self.session_store.write(session_id, session_state)
                    return GenerationResult(
                        code="",
                        trace_id=trace_id,
                        session_id=session_id,
                        strategy="clarification",
                        verification_errors=[],
                        validation_report={"has_errors": False, "has_warnings": False, "messages": []},
                        repair_rounds=0,
                        degraded_mode=False,
                        status="clarification_needed",
                        question=chain_result.question,
                        assumptions=assumptions,
                        session_summary=self.build_session_summary(session_state),
                        clarification_suggested=True,
                        assumption_risk=assumption_risk,
                    )
                strategy = "feedback_revision" if feedback else "ollama_chain"
                assumptions = list(task_spec.assumptions) + [
                    "The same-model planner/writer chain handled the request.",
                ]
                assumptions.extend(chain_result.planner.get("assumptions", []))
                code = chain_result.code
                validation_report = chain_result.validation_report
                repair_trace = chain_result.history
                repair_rounds = chain_result.rounds
                rules_applied = chain_result.rules_applied
                examples_used = chain_result.examples_used
                critic_rules_used = chain_result.critic_rules_used
                planner_payload = chain_result.planner
                critic_payload = chain_result.critic
                semantic_checks = chain_result.semantic_checks
            except Exception as exc:
                backend_error = str(exc)
                raise BackendUnavailableError(backend_error) from exc

        if validation_report.has_errors and strategy != "safe_fallback":
            remaining_rounds = max(int(self.profile.max_repair_rounds) - int(repair_rounds), 1)
            repair_result = self.repair_loop.run(
                code=code,
                task_spec=task_spec,
                validation_report=validation_report,
                profile=self.profile,
                max_rounds=remaining_rounds,
                source_context=effective_context,
                prompt=effective_prompt,
                planner_semantic_checks=semantic_checks,
            )
            code = repair_result.code
            validation_report = repair_result.validation_report
            repair_trace = repair_trace + [
                {"stage": "deterministic_repair", "history": repair_result.history}
            ]
            repair_rounds = repair_rounds + repair_result.rounds
        verification_errors = []
        for error_code in validation_report.error_codes() + verify_code(code):
            if error_code not in verification_errors:
                verification_errors.append(error_code)
        degraded_mode = self._is_degraded(strategy=strategy, validation_report=validation_report)
        status = self._build_status(strategy=strategy, degraded_mode=degraded_mode)
        session_state["status"] = status
        session_state["planner_state"] = planner_payload
        session_state["previous_candidate_code"] = code
        session_state["previous_validation_report"] = validation_report.to_dict()
        session_state["assumptions"] = assumptions
        session_state["last_strategy"] = strategy
        session_state["open_clarification_question"] = ""
        trace_payload = {
            "prompt": prompt,
            "effective_prompt": effective_prompt,
            "context": effective_context,
            "feedback": feedback,
            "clarification_answer": clarification_answer,
            "status": status,
            "strategy": strategy,
            "task_spec": task_spec.dict(),
            "assumptions": assumptions,
            "verification_errors": verification_errors,
            "validation_report": validation_report.to_dict(),
            "repair_rounds": repair_rounds,
            "repair_trace": repair_trace,
            "planner": planner_payload,
            "critic": critic_payload,
            "rules_applied": rules_applied,
            "examples_used": examples_used,
            "critic_rules_used": critic_rules_used,
            "semantic_checks": semantic_checks,
            "backend_error": backend_error,
            "model": self.profile.model,
            "fallback_model": self.profile.fallback_model,
            "code": code,
            "session_id": session_id,
            "degraded_mode": degraded_mode,
            "clarification_suggested": bool(rule_based_question),
            "assumption_risk": assumption_risk,
        }
        trace_id = self.trace_store.write(trace_payload)
        session_state["latest_trace_id"] = trace_id
        session_state["trace_ids"].append(trace_id)
        self.session_store.write(session_id, session_state)
        return GenerationResult(
            code=code,
            trace_id=trace_id,
            session_id=session_id,
            strategy=strategy,
            verification_errors=verification_errors,
            validation_report=validation_report.to_dict(),
            repair_rounds=repair_rounds,
            degraded_mode=degraded_mode,
            status=status,
            assumptions=assumptions,
            session_summary=self.build_session_summary(session_state),
            clarification_suggested=bool(rule_based_question),
            assumption_risk=assumption_risk,
        )

    def _prepare_session_state(self, session_id, prompt, context, feedback, clarification_answer):
        session_state = self.session_store.read(session_id) or {
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
        )

    def analyze(self, prompt, context=None):
        task_spec = self.extractor.extract(prompt=prompt, context=context)
        reduced_context = self.context_reducer.reduce(context, task_spec)
        clarification_question = self._rule_based_clarification(task_spec, {}).get("question") or None
        return {
            "normalized_prompt": task_spec.normalized_prompt,
            "suggested_strategy": (
                "clarification"
                if clarification_question
                else ("safe_fallback" if task_spec.safety_fallback else "ollama_chain")
            ),
            "clarification_question": clarification_question,
            "task_spec": task_spec.dict(),
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

    @staticmethod
    def _rule_based_clarification(task_spec, session_state):
        if session_state.get("clarified_root"):
            return {"question": "", "assumptions": [], "kind": None}
        if session_state.get("clarification_history"):
            return {"question": "", "assumptions": [], "kind": None}
        prompt = task_spec.normalized_prompt
        if task_spec.target_root == "unknown_mixed":
            if "email" in prompt:
                return {
                    "question": "Use wf.vars or wf.initVariables for email root?",
                    "assumptions": ["Root ambiguity detected between wf.vars and wf.initVariables."],
                    "kind": "root_ambiguity",
                }
            return {
                "question": "Use wf.vars or wf.initVariables for this task?",
                "assumptions": ["Root ambiguity detected between wf.vars and wf.initVariables."],
                "kind": "root_ambiguity",
            }

        if "json envelope" in prompt or "json_envelope" in prompt:
            if "ключ" in prompt or "key" in prompt:
                return {
                    "question": "Which JSON envelope key should contain the generated result?",
                    "assumptions": ["Output-key ambiguity detected for JSON envelope output."],
                    "kind": "output_key_ambiguity",
                }

        if ("очист" in prompt or "mutate" in prompt or "обнов" in prompt) and ("верни" in prompt or "return" in prompt):
            return {
                "question": "Should the code mutate the source data in place or return a new cleaned value?",
                "assumptions": ["Mutate-vs-return ambiguity detected."],
                "kind": "mutate_return_ambiguity",
            }

        if task_spec.ambiguity_score >= 0.65 and task_spec.composition_score <= 0.4:
            return {
                "question": "Should the result be a single value, an object, or an array?",
                "assumptions": ["Return-shape ambiguity detected."],
                "kind": "return_shape_ambiguity",
            }

        return {"question": "", "assumptions": [], "kind": None}

    @staticmethod
    def _assumption_risk(task_spec, clarification_policy):
        if clarification_policy.get("question"):
            return "high"
        if task_spec.ambiguity_score >= 0.5 or len(task_spec.ambiguity_notes) >= 2:
            return "high"
        if task_spec.ambiguity_score >= 0.25 or task_spec.ambiguity_notes:
            return "medium"
        return "low"
