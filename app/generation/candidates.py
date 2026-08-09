from dataclasses import dataclass, field
from typing import Any

from app.generation.taskspec import ResolvedTaskSpec
from app.validation.base import ValidationReport


@dataclass(frozen=True)
class GeneratedCandidate:
    task_spec: ResolvedTaskSpec
    code: str
    validation_report: ValidationReport
    strategy: str
    assumptions: list[str]
    repair_trace: list[dict[str, Any]] = field(default_factory=list)
    repair_rounds: int = 0
    rules_applied: list[str] = field(default_factory=list)
    examples_used: list[str] = field(default_factory=list)
    critic_rules_used: list[str] = field(default_factory=list)
    planner_payload: dict[str, Any] = field(default_factory=dict)
    critic_payload: dict[str, Any] = field(default_factory=dict)
    semantic_checks: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PlannerClarification:
    task_spec: ResolvedTaskSpec
    question: str
    assumptions: tuple[str, ...]
    planner_payload: dict[str, Any]
    repair_trace: list[dict[str, Any]]
    rules_applied: list[str]
    semantic_checks: list[str]
