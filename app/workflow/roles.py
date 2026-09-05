from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Generic, TypeVar

from pydantic import TypeAdapter, ValidationError

from app.generation.backend_errors import BackendProtocol
from app.workflow.context import ContextInspector
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
Two output formats exist and they are not interchangeable. `lua_block` is one raw Lua chunk that
returns the result, and it is the default: choose it unless the request itself asks for several
named workflow variables at once. `json_envelope` is a JSON object whose every value is a
lua{...}lua chunk, and it belongs only to that multi-variable case. One returned value is a
`lua_block`, whatever the shape of that value.
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
        response: StructuredResponse[SchemaValue],
    ) -> SchemaValue:
        adapter = response.adapter
        json_schema = response.schema
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
                f"Validation errors: {json.dumps(safe_errors, sort_keys=True)}",
            )
        )


def _strip_string_length_bounds(node: object) -> None:
    """Remove `maxLength` from every string in the schema handed to the model.

    Ollama compiles the schema into a sampling grammar, and an upper bound becomes an explicit
    repetition rule over the vocabulary. Our code field allows 131072 characters, and that grammar
    fails to build at all: the request comes back as HTTP 500 `failed to load model vocabulary
    required for format`. The bound stays in the Pydantic contract, so an oversized response is
    still rejected — it is enforced after generation instead of during it.
    """
    if isinstance(node, dict):
        if node.get("type") == "string":
            node.pop("maxLength", None)
        for value in node.values():
            _strip_string_length_bounds(value)
    elif isinstance(node, list):
        for value in node:
            _strip_string_length_bounds(value)


def _model_facing_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Adapt the Pydantic schema to what the local model can actually be constrained by.

    Two adjustments, both learned from live failures rather than from the specification.
    """
    # Ollama treats an optional property as one the model may skip, and Pydantic marks a
    # discriminator with a Python default as optional. The model kept omitting `kind`, and without
    # the tag the union cannot be resolved at all — every planner answer came back invalid with a
    # message that pointed nowhere.
    for definition in (schema, *(schema.get("$defs") or {}).values()):
        if not isinstance(definition, dict):
            continue
        properties = definition.get("properties")
        if not isinstance(properties, dict):
            continue
        required = set(definition.get("required") or ())
        for name, spec in properties.items():
            if isinstance(spec, dict) and "const" in spec:
                required.add(name)
        if required:
            definition["required"] = sorted(required)

    # Pydantic отдаёт JsonValue как пустую схему `{}`, то есть «что угодно», и грамматика Ollama
    # перестаёт что-либо ограничивать. Модель в этом месте по умолчанию строила объект и заворачивала
    # в него скалярный ожидаемый результат, из-за чего план противоречил собственному output-контракту.
    # Явное перечисление вариантов возвращает грамматике смысл, а скаляры ставит первыми.
    definitions = schema.get("$defs")
    if isinstance(definitions, dict) and definitions.get("JsonValue") == {}:
        definitions["JsonValue"] = {
            "anyOf": [
                {"type": "string"},
                {"type": "number"},
                {"type": "boolean"},
                {"type": "null"},
                {"type": "array", "items": {"$ref": "#/$defs/JsonValue"}},
                {"type": "object", "additionalProperties": {"$ref": "#/$defs/JsonValue"}},
            ]
        }

    _strip_string_length_bounds(schema)
    return schema


class StructuredResponse(Generic[SchemaValue]):
    def __init__(self, adapter: TypeAdapter[SchemaValue]) -> None:
        self.adapter = adapter
        self.schema = _model_facing_schema(adapter.json_schema())


PLANNING_RESPONSE: StructuredResponse[PlanningDecision] = StructuredResponse(PLANNING_ADAPTER)
CODE_RESPONSE: StructuredResponse[CodeCandidate] = StructuredResponse(CODE_ADAPTER)
REVIEW_RESPONSE: StructuredResponse[ReviewDecision] = StructuredResponse(REVIEW_ADAPTER)


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
        rejected_plan_findings: tuple[str, ...] = (),
    ) -> PlanningDecision:
        payload = {
            "request": prompt,
            "context_sample": context_sample,
            "context_inventory": inventory.model_dump(mode="json"),
            "paths_present_under_both_roots": list(ContextInspector.ambiguous_paths(inventory)),
            "clarification_answer": clarification_answer,
            "feedback": feedback,
            "rejected_plan_findings": list(rejected_plan_findings),
        }
        role_prompt = f"""You are the planner in a local code-generation workflow.
Interpret the request; do not write Lua. Return exactly one JSON PlanningDecision.

For a plan:
- express workflow paths as a root enum plus path segments, never as an invented dotted string;
- describe ordered implementation steps without choosing a predefined task category;
- preserve the requested output format and shape;
- provide 1 to 3 small executable acceptance cases with complete workflow contexts;
- acceptance cases must test the requested behavior, not a preferred source-code spelling.

Each acceptance case field `expected` holds the exact JSON value the generated code returns for
that context, and it must match the declared output shape: `scalar` is a bare number, string,
boolean or null; `array` is a JSON array; `object` is a JSON object. A request that returns a
lower-cased address has `expected` equal to "user@example.com".

This holds for `json_envelope` too: `expected` is the object the envelope evaluates to, never the
Lua sources it is written from. An envelope returning a number and its square has `expected` equal
to {{"num": 5, "squared": 25}}, not {{"num": "return 5", "squared": "return 5 * 5"}}.

`nullable` describes the same contract from the other side: set it to true whenever the request
names a case that yields nothing, and whenever any acceptance case has `expected` equal to null.
A plan that declares `nullable` false and then expects null for the empty input contradicts
itself and is rejected before any code is written.

When `rejected_plan_findings` is not empty, your previous plan was rejected for exactly those
reasons. Fix them; do not repeat the same plan and do not fall back to a clarification.

`paths_present_under_both_roots` lists paths that exist under wf.vars and wf.initVariables at the
same time. If the request needs one of them and does not itself name the root, that is the
ambiguity you must ask about: choosing a root yourself produces confidently wrong code.

Return a clarification only when the request leaves you choosing between two concrete alternatives
that are both present in the context, most often wf.vars versus wf.initVariables. A request that
names its own source and its transformation is not ambiguous; plan it. Read-only workflow context
is a rule you already know, not an ambiguity worth asking about.

Domain specification:
{DOMAIN_SPECIFICATION}

Input:
{json.dumps(payload, ensure_ascii=False, sort_keys=True)}
"""
        return self.model.request(role_prompt, PLANNING_RESPONSE)


class GeneratorRole:
    def __init__(self, model: StructuredModelClient) -> None:
        self.model = model

    def run(self, *, prompt: str, plan: TaskPlan) -> CodeCandidate:
        role_prompt = f"""You are the generator in a local code-generation workflow.
Return exactly one JSON CodeCandidate. Do not use markdown. Implement the structured plan, not a
remembered example. Workflow inputs are read-only and the code must return its result.

The plan's output format decides the exact shape of the code field, and the two are not
interchangeable:
- `lua_block`: raw Lua source that returns the result. It must not be wrapped: code starting with
  lua{{ or ending with }}lua is rejected.
- `json_envelope`: a JSON object written as text, non-empty, where every value is a string of the
  form lua{{ ... }}lua holding one non-empty Lua chunk that returns that key's value.

Domain specification:
{DOMAIN_SPECIFICATION}

Original request:
{prompt}

Task plan:
{plan.model_dump_json()}
"""
        return self.model.request(role_prompt, CODE_RESPONSE)

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
        role_prompt = f"""You are revising one rejected LocalScript candidate.
Return exactly one JSON CodeCandidate. Replace the candidate with a minimal semantic correction
that addresses every structured finding. Do not use markdown and do not change the requested result.

The plan's output format still decides the shape: `lua_block` is raw Lua that must not be wrapped
in lua{{ }}lua, and `json_envelope` is a JSON object whose every value is a lua{{ ... }}lua string.

Domain specification:
{DOMAIN_SPECIFICATION}

Original request:
{prompt}

Task plan:
{plan.model_dump_json()}

Rejected code:
{candidate.code}

Structured findings:
{json.dumps(failure_payload, ensure_ascii=False, sort_keys=True)}
"""
        return self.model.request(role_prompt, CODE_RESPONSE)


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
        role_prompt = f"""You are the reviewer in a fresh context. Return exactly one JSON
ReviewDecision. Compare the original request, structured plan, executable acceptance observations,
and candidate. Approve only when the code implements the requested semantics and respects the plan.
Do not infer approval merely from syntax success. Findings must use short stable snake_case codes.

Domain specification:
{DOMAIN_SPECIFICATION}

Original request:
{prompt}

Task plan:
{plan.model_dump_json()}

Candidate code:
{candidate.code}

Deterministic validation:
{validation.model_dump_json()}
"""
        return self.model.request(role_prompt, REVIEW_RESPONSE)
