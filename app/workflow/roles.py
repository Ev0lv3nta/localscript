from __future__ import annotations

import json
from collections.abc import Callable
from typing import TypeVar

from pydantic import TypeAdapter, ValidationError

from app.generation.backend_errors import BackendProtocol
from app.workflow.contracts import (
    CodeCandidate,
    ContextInventory,
    JsonValue,
    PlanningDecision,
    ReviewDecision,
    TaskPlan,
    ValidationResult,
)

SchemaValue = TypeVar("SchemaValue")

PLANNING_ADAPTER: TypeAdapter[PlanningDecision] = TypeAdapter(PlanningDecision)
CODE_ADAPTER: TypeAdapter[CodeCandidate] = TypeAdapter(CodeCandidate)
REVIEW_ADAPTER: TypeAdapter[ReviewDecision] = TypeAdapter(ReviewDecision)

DOMAIN_SPECIFICATION = """LocalScript generates small Lua 5.4 transformations for a workflow runtime.
The only workflow data roots are wf.vars and wf.initVariables. Treat both as read-only.
Return either one raw Lua block or a JSON object whose values are lua{...}lua chunks.
Do not use operating-system, file, network, package, debug, dynamic-loading, or process APIs.
Prefer a direct returned value over workflow mutation. Ask one concrete question when the requested
source, result shape, or mutate-versus-return intent cannot be determined safely.
"""


class StructuredModelClient:
    def __init__(self, complete: Callable[..., str]) -> None:
        self._complete = complete

    def request(
        self,
        prompt: str,
        adapter: TypeAdapter[SchemaValue],
    ) -> SchemaValue:
        json_schema = adapter.json_schema()
        raw = self._complete(prompt, response_format=json_schema)
        try:
            return adapter.validate_json(raw, strict=True)
        except ValidationError as first_error:
            correction_prompt = self._correction_prompt(prompt, first_error)
            corrected = self._complete(correction_prompt, response_format=json_schema)
            try:
                return adapter.validate_json(corrected, strict=True)
            except ValidationError as second_error:
                raise BackendProtocol(reason="structured_response_invalid") from second_error

    @staticmethod
    def _correction_prompt(prompt: str, error: ValidationError) -> str:
        safe_errors = [
            {
                "type": item["type"],
                "loc": [str(part) for part in item["loc"]],
            }
            for item in error.errors(include_input=False, include_url=False)
        ]
        return "\n".join(
            (
                prompt,
                "The previous response did not satisfy the JSON schema.",
                "Return one corrected JSON value only. Do not add markdown or explanations.",
                "Validation errors: {0}".format(json.dumps(safe_errors, sort_keys=True)),
            )
        )


class PlannerRole:
    def __init__(self, model: StructuredModelClient) -> None:
        self.model = model

    def run(
        self,
        *,
        prompt: str,
        context_sample: JsonValue,
        inventory: ContextInventory,
        clarification_answer: str | None = None,
        feedback: str | None = None,
    ) -> PlanningDecision:
        payload = {
            "request": prompt,
            "context_sample": context_sample,
            "context_inventory": inventory.model_dump(mode="json"),
            "clarification_answer": clarification_answer,
            "feedback": feedback,
        }
        role_prompt = """You are the planner in a local code-generation workflow.
Interpret the request; do not write Lua. Return exactly one JSON PlanningDecision.

For a plan:
- express workflow paths as a root enum plus path segments, never as an invented dotted string;
- describe ordered implementation steps without choosing a predefined task family;
- preserve the requested output format and shape;
- provide 1 to 3 small executable acceptance cases with complete workflow contexts and exact JSON results;
- acceptance cases must test the requested behavior, not a preferred source-code spelling.

For unresolved ambiguity return a clarification with one short, concrete question.
Never guess between wf.vars and wf.initVariables when both are plausible.

Domain specification:
{domain}

Input:
{payload}
""".format(domain=DOMAIN_SPECIFICATION, payload=json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return self.model.request(role_prompt, PLANNING_ADAPTER)


class GeneratorRole:
    def __init__(self, model: StructuredModelClient) -> None:
        self.model = model

    def run(self, *, prompt: str, plan: TaskPlan) -> CodeCandidate:
        role_prompt = """You are the generator in a local code-generation workflow.
Return exactly one JSON CodeCandidate. Its code field must contain the requested raw Lua block or
the complete JSON envelope. Do not use markdown. Implement the structured plan, not a remembered
example. Workflow inputs are read-only and the code must return its result.

Domain specification:
{domain}

Original request:
{request}

Task plan:
{plan}
""".format(
            domain=DOMAIN_SPECIFICATION,
            request=prompt,
            plan=plan.model_dump_json(),
        )
        return self.model.request(role_prompt, CODE_ADAPTER)

    def revise(
        self,
        *,
        prompt: str,
        plan: TaskPlan,
        candidate: CodeCandidate,
        validation: ValidationResult,
        review: ReviewDecision | None,
    ) -> CodeCandidate:
        failure_payload = {
            "validation": validation.model_dump(mode="json"),
            "review": REVIEW_ADAPTER.dump_python(review, mode="json") if review else None,
        }
        role_prompt = """You are revising one rejected LocalScript candidate.
Return exactly one JSON CodeCandidate. Replace the candidate with a minimal semantic correction
that addresses every structured finding. Do not use markdown and do not change the requested result.

Domain specification:
{domain}

Original request:
{request}

Task plan:
{plan}

Rejected code:
{code}

Structured findings:
{findings}
""".format(
            domain=DOMAIN_SPECIFICATION,
            request=prompt,
            plan=plan.model_dump_json(),
            code=candidate.code,
            findings=json.dumps(failure_payload, ensure_ascii=False, sort_keys=True),
        )
        return self.model.request(role_prompt, CODE_ADAPTER)


class ReviewerRole:
    def __init__(self, model: StructuredModelClient) -> None:
        self.model = model

    def run(
        self,
        *,
        prompt: str,
        plan: TaskPlan,
        candidate: CodeCandidate,
        validation: ValidationResult,
    ) -> ReviewDecision:
        role_prompt = """You are the reviewer in a fresh context. Return exactly one JSON
ReviewDecision. Compare the original request, structured plan, executable acceptance observations,
and candidate. Approve only when the code implements the requested semantics and respects the plan.
Do not infer approval merely from syntax success. Findings must use short stable snake_case codes.

Domain specification:
{domain}

Original request:
{request}

Task plan:
{plan}

Candidate code:
{code}

Deterministic validation:
{validation}
""".format(
            domain=DOMAIN_SPECIFICATION,
            request=prompt,
            plan=plan.model_dump_json(),
            code=candidate.code,
            validation=validation.model_dump_json(),
        )
        return self.model.request(role_prompt, REVIEW_ADAPTER)
