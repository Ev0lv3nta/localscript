import json
from dataclasses import dataclass, field

from app.families import get_family_definition
from app.core.kb import build_rule_lines, select_critic_rules, select_examples
from app.generation.context_reducer import ContextReducer
from app.generation.task_resolver import TaskResolver
from app.validation.base import ValidationReport
from app.generation.prompts import (
    build_critic_prompt,
    build_fixer_prompt,
    build_planner_prompt,
    build_writer_prompt,
)


@dataclass
class ModelChainResult:
    code: str
    validation_report: object
    rounds: int
    planner: dict
    critic: dict
    task_spec: object
    rules_applied: list = field(default_factory=list)
    examples_used: list = field(default_factory=list)
    critic_rules_used: list = field(default_factory=list)
    history: list = field(default_factory=list)
    status: str = "completed"
    question: str = ""
    semantic_checks: list = field(default_factory=list)


class SameModelChain:
    def __init__(self, backend, validation_pipeline, formatter, task_resolver=None):
        self.backend = backend
        self.validation_pipeline = validation_pipeline
        self.formatter = formatter
        self.context_reducer = ContextReducer()
        self.task_resolver = task_resolver or TaskResolver()

    def run(self, prompt, context, task_spec, profile, max_rounds=1, session_state=None, stop_on_clarification=False):
        rules = build_rule_lines(task_spec)
        reduced_context = self.context_reducer.reduce(context, task_spec)

        planner_text = self._complete(
            build_planner_prompt(prompt, reduced_context, task_spec, rules, session_state=session_state),
            response_format="json",
        )
        planner = self._parse_planner(planner_text, task_spec, prompt=prompt)
        task_spec = self.task_resolver.resolve(task_spec, planner=planner)
        rules = build_rule_lines(task_spec)
        if stop_on_clarification and planner.get("clarification_needed"):
            return ModelChainResult(
                code="",
                validation_report=ValidationReport(),
                rounds=0,
                planner=planner,
                critic={},
                task_spec=task_spec,
                rules_applied=rules,
                history=[
                    {
                        "stage": "planner",
                        "payload": planner,
                        "raw": planner_text,
                    }
                ],
                status="clarification_needed",
                question=planner.get("clarification_question", ""),
                semantic_checks=self._normalize_semantic_checks(planner),
            )
        example_limit = 1 if planner.get("family") and planner.get("family") != "generic_lua" else 2
        examples = select_examples(prompt, family=planner.get("family"), limit=example_limit)
        examples_used = [example.get("id") for example in examples]
        planner_checks = self._normalize_semantic_checks(planner)

        writer_text = self._complete(
            build_writer_prompt(
                prompt,
                reduced_context,
                task_spec,
                planner,
                rules,
                examples,
                session_state=session_state,
            ),
            response_format=None,
        )
        code = self.formatter.format(self._clean_candidate(writer_text), task_spec.output_style)
        validation_report = self.validation_pipeline.run(
            code=code,
            task_spec=task_spec,
            profile=profile,
            source_context=context,
            prompt=prompt,
            planner_semantic_checks=planner_checks,
        )

        history = [
            {
                "stage": "planner",
                "payload": planner,
                "raw": planner_text,
            },
            {
                "stage": "writer",
                "raw": writer_text,
                "code": code,
                "validation_report": validation_report.to_dict(),
            },
        ]
        critic_payload = {"repairable": False, "issues": [], "minimal_actions": []}
        critic_rules_used = []
        rounds = 0

        for round_index in range(1, int(max_rounds) + 1):
            if not validation_report.has_errors:
                break

            validation_errors = validation_report.error_codes()
            critic_rules = select_critic_rules(prompt, validation_errors=validation_errors, limit=4)
            critic_rules_used = [rule.get("id") for rule in critic_rules]
            critic_text = self._complete(
                build_critic_prompt(prompt, reduced_context, planner, code, validation_errors, critic_rules, session_state=session_state),
                response_format="json",
            )
            critic_payload = self._parse_critic(critic_text, validation_errors)
            history.append(
                {
                    "stage": "critic",
                    "round": round_index,
                    "payload": critic_payload,
                    "raw": critic_text,
                }
            )
            if not critic_payload.get("repairable"):
                break

            fixer_text = self._complete(
                build_fixer_prompt(
                    prompt,
                    reduced_context,
                    task_spec,
                    planner,
                    code,
                    critic_payload,
                    validation_errors,
                    rules,
                    examples,
                    session_state=session_state,
                ),
                response_format=None,
            )
            fixed_code = self.formatter.format(self._clean_candidate(fixer_text), task_spec.output_style)
            if fixed_code == code:
                history.append(
                    {
                        "stage": "fixer",
                        "round": round_index,
                        "status": "no_change",
                        "raw": fixer_text,
                    }
                )
                break

            code = fixed_code
            rounds = round_index
            validation_report = self.validation_pipeline.run(
                code=code,
                task_spec=task_spec,
                profile=profile,
                source_context=context,
                prompt=prompt,
                planner_semantic_checks=planner_checks,
            )
            history.append(
                {
                    "stage": "fixer",
                    "round": round_index,
                    "raw": fixer_text,
                    "code": code,
                    "validation_report": validation_report.to_dict(),
                }
            )
            if not validation_report.has_errors:
                break

        return ModelChainResult(
            code=code,
            validation_report=validation_report,
            rounds=rounds,
            planner=planner,
            critic=critic_payload,
            task_spec=task_spec,
            rules_applied=rules,
            examples_used=examples_used,
            critic_rules_used=critic_rules_used,
            history=history,
            semantic_checks=planner_checks,
        )

    def _complete(self, prompt, response_format=None):
        if hasattr(self.backend, "complete"):
            return self.backend.complete(prompt=prompt, response_format=response_format)
        return self.backend.generate(prompt=prompt, context=None)

    @staticmethod
    def _clean_candidate(raw_candidate):
        text = (raw_candidate or "").strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return text

    @staticmethod
    def _extract_json_object(text):
        if not text:
            return None
        candidate = text.strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        start_positions = [index for index, char in enumerate(candidate) if char == "{"]
        end_positions = [index for index, char in enumerate(candidate) if char == "}"]
        for start in start_positions:
            for end in reversed(end_positions):
                if end <= start:
                    continue
                chunk = candidate[start : end + 1]
                try:
                    return json.loads(chunk)
                except json.JSONDecodeError:
                    continue
        return None

    def _parse_planner(self, text, task_spec, prompt=""):
        payload = self._extract_json_object(text) or {}
        family = payload.get("family")
        if not isinstance(family, str) or not family:
            family = task_spec.family or "generic_lua"
        elif family == "generic_lua" and task_spec.family:
            family = task_spec.family
        inferred_return_shape = self._default_return_shape(task_spec, prompt, family=task_spec.family)
        explicit_return_shape = payload.get("return_shape") if isinstance(payload.get("return_shape"), str) else None
        root = payload.get("root")
        if not isinstance(root, str) or not root:
            root = task_spec.target_root or "unknown"
        source_paths = payload.get("source_paths")
        if not isinstance(source_paths, list):
            source_paths = []
        return_shape = explicit_return_shape
        if not isinstance(return_shape, str) or not return_shape:
            return_shape = inferred_return_shape
        elif return_shape == "scalar" and inferred_return_shape in {"array", "object", "json_envelope"}:
            return_shape = inferred_return_shape
        preferred_shape = self._preferred_return_shape(family)
        if (
            preferred_shape
            and return_shape != preferred_shape
            and task_spec.family is None
            and family != "generic_lua"
        ):
            family = "generic_lua"
            preferred_shape = self._preferred_return_shape(family)
        if preferred_shape:
            return_shape = preferred_shape
        constraints = payload.get("constraints")
        assumptions = payload.get("assumptions")
        semantic_checks = payload.get("semantic_checks")
        return {
            "family": family,
            "root": root,
            "source_paths": [item for item in source_paths if isinstance(item, str)],
            "return_shape": return_shape,
            "constraints": constraints if isinstance(constraints, list) else [],
            "assumptions": assumptions if isinstance(assumptions, list) else [],
            "clarification_needed": bool(payload.get("clarification_needed")),
            "clarification_question": payload.get("clarification_question") or "",
            "semantic_checks": semantic_checks if isinstance(semantic_checks, list) else [],
        }

    @staticmethod
    def _parse_critic(text, validation_errors):
        payload = SameModelChain._extract_json_object(text) or {}
        issues = payload.get("issues") if isinstance(payload.get("issues"), list) else validation_errors
        minimal_actions = payload.get("minimal_actions") if isinstance(payload.get("minimal_actions"), list) else []
        repairable = payload.get("repairable")
        if not isinstance(repairable, bool):
            repairable = bool(minimal_actions or issues)
        return {
            "repairable": repairable,
            "issues": [item for item in issues if isinstance(item, str)],
            "minimal_actions": [item for item in minimal_actions if isinstance(item, str)],
        }

    @staticmethod
    def _preferred_return_shape(family):
        definition = get_family_definition(family)
        return definition.preferred_return_shape if definition else None

    @classmethod
    def _default_return_shape(cls, task_spec, prompt, family=None):
        preferred_shape = cls._preferred_return_shape(family or task_spec.family)
        if preferred_shape:
            return preferred_shape
        if task_spec.output_style == "json_envelope":
            return "json_envelope"
        normalized_prompt = " ".join((prompt or "").lower().split())
        if any(
            token in normalized_prompt
            for token in [
                "новый массив",
                "верни массив",
                "return array",
                "list of",
                "_utils.array.new",
                "список",
                "таблиц",
                "list ",
            ]
        ):
            return "array"
        if any(token in normalized_prompt for token in ["новый объект", "верни объект", "return object", "payload"]):
            return "object"
        scalar_cues = [
            "сколько",
            "count",
            "sum",
            "сумм",
            "сложи",
            "нормализ",
            "normalize",
            "lower",
            "lower-case",
            "trim",
            "перв",
            "first",
            "last",
            "boolean",
            "bool",
            "верни nil",
            "return nil",
        ]
        array_roots = []
        for path in getattr(task_spec, "context_paths", []) or []:
            if not isinstance(path, str) or not path.endswith("[]"):
                continue
            leaf = path[:-2].split(".")[-1].lower()
            if leaf and leaf in normalized_prompt:
                array_roots.append(leaf)
        if array_roots and not any(token in normalized_prompt for token in scalar_cues):
            if any(
                token in normalized_prompt
                for token in [
                    "только",
                    "where",
                    "где",
                    "для ",
                    "phone",
                    "email",
                    "order_id",
                    "sku",
                    "список",
                    "list",
                    "таблиц",
                ]
            ):
                return "array"
        return "scalar"

    @staticmethod
    def _normalize_semantic_checks(planner):
        checks = []
        return_shape = planner.get("return_shape")
        if isinstance(return_shape, str) and return_shape in {"array", "object", "scalar"}:
            checks.append({"kind": "return_shape", "value": return_shape})
        for item in planner.get("semantic_checks", []):
            if isinstance(item, dict):
                checks.append(item)
                continue
            if not isinstance(item, str):
                continue
            lowered = item.lower()
            if "empty array" in lowered and planner.get("source_paths"):
                checks.append(
                    {
                        "kind": "empty_array_on_missing_source",
                        "source_path": planner["source_paths"][0],
                    }
                )
            elif "preserve" in lowered and "id" in lowered:
                checks.append({"kind": "contains_fields", "fields": ["id"]})
        return checks
