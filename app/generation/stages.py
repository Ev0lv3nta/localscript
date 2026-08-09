from dataclasses import dataclass, field
from enum import Enum
from time import perf_counter
from typing import Optional


class GenerationStage(str, Enum):
    SESSION_READY = "session_ready"
    TASK_RESOLVED = "task_resolved"
    CLARIFICATION_REQUIRED = "clarification_required"
    CANDIDATE_GENERATED = "candidate_generated"
    CANDIDATE_REPAIRED = "candidate_repaired"
    OUTCOME_FINALIZED = "outcome_finalized"


_ALLOWED_TRANSITIONS = {
    None: {GenerationStage.SESSION_READY},
    GenerationStage.SESSION_READY: {
        GenerationStage.TASK_RESOLVED,
        GenerationStage.CLARIFICATION_REQUIRED,
    },
    GenerationStage.TASK_RESOLVED: {
        GenerationStage.CLARIFICATION_REQUIRED,
        GenerationStage.CANDIDATE_GENERATED,
    },
    GenerationStage.CLARIFICATION_REQUIRED: {
        GenerationStage.OUTCOME_FINALIZED,
    },
    GenerationStage.CANDIDATE_GENERATED: {
        GenerationStage.CANDIDATE_REPAIRED,
        GenerationStage.OUTCOME_FINALIZED,
    },
    GenerationStage.CANDIDATE_REPAIRED: {
        GenerationStage.OUTCOME_FINALIZED,
    },
    GenerationStage.OUTCOME_FINALIZED: set(),
}


@dataclass
class GenerationExecution:
    stage: Optional[GenerationStage] = None
    events: list = field(default_factory=list)
    started_at: float = field(default_factory=perf_counter, repr=False)

    def transition(self, next_stage: GenerationStage):
        if not isinstance(next_stage, GenerationStage):
            raise TypeError("generation stage must be GenerationStage")
        if next_stage not in _ALLOWED_TRANSITIONS[self.stage]:
            current = self.stage.value if self.stage is not None else "not_started"
            raise ValueError(
                "invalid generation stage transition: {0} -> {1}".format(
                    current,
                    next_stage.value,
                )
            )
        self.stage = next_stage
        elapsed_ms = round((perf_counter() - self.started_at) * 1000.0, 3)
        previous_elapsed_ms = self.events[-1]["elapsed_ms"] if self.events else 0.0
        self.events.append(
            {
                "stage": next_stage.value,
                "elapsed_ms": elapsed_ms,
                "stage_duration_ms": round(elapsed_ms - previous_elapsed_ms, 3),
            }
        )

    def snapshot(self):
        return [dict(event) for event in self.events]
