from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple


class ClarificationKind(str, Enum):
    ROOT = "root_ambiguity"
    OUTPUT_KEY = "output_key_ambiguity"
    MUTATE_OR_RETURN = "mutate_return_ambiguity"
    RETURN_SHAPE = "return_shape_ambiguity"


@dataclass(frozen=True)
class ClarificationDecision:
    question: str = ""
    assumptions: Tuple[str, ...] = field(default_factory=tuple)
    kind: Optional[ClarificationKind] = None

    @property
    def required(self):
        return bool(self.question)


class ClarificationPolicy:
    def evaluate(self, task_spec, session_state) -> ClarificationDecision:
        if session_state.get("clarified_root") or session_state.get(
            "clarification_history"
        ):
            return ClarificationDecision()

        prompt = task_spec.normalized_prompt
        if task_spec.target_root == "unknown_mixed":
            question = "Use wf.vars or wf.initVariables for this task?"
            if "email" in prompt:
                question = "Use wf.vars or wf.initVariables for email root?"
            return ClarificationDecision(
                question=question,
                assumptions=(
                    "Root ambiguity detected between wf.vars and wf.initVariables.",
                ),
                kind=ClarificationKind.ROOT,
            )

        if (
            ("json envelope" in prompt or "json_envelope" in prompt)
            and ("ключ" in prompt or "key" in prompt)
        ):
            return ClarificationDecision(
                question="Which JSON envelope key should contain the generated result?",
                assumptions=(
                    "Output-key ambiguity detected for JSON envelope output.",
                ),
                kind=ClarificationKind.OUTPUT_KEY,
            )

        mutate = "очист" in prompt or "mutate" in prompt or "обнов" in prompt
        returns_value = "верни" in prompt or "return" in prompt
        if mutate and returns_value:
            return ClarificationDecision(
                question=(
                    "Should the code mutate the source data in place or return a new "
                    "cleaned value?"
                ),
                assumptions=("Mutate-vs-return ambiguity detected.",),
                kind=ClarificationKind.MUTATE_OR_RETURN,
            )

        if task_spec.ambiguity_score >= 0.65 and task_spec.composition_score <= 0.4:
            return ClarificationDecision(
                question="Should the result be a single value, an object, or an array?",
                assumptions=("Return-shape ambiguity detected.",),
                kind=ClarificationKind.RETURN_SHAPE,
            )

        return ClarificationDecision()

    @staticmethod
    def assumption_risk(task_spec, decision: ClarificationDecision):
        if decision.required:
            return "high"
        if task_spec.ambiguity_score >= 0.5 or len(task_spec.ambiguity_notes) >= 2:
            return "high"
        if task_spec.ambiguity_score >= 0.25 or task_spec.ambiguity_notes:
            return "medium"
        return "low"
