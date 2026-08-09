import re
from functools import lru_cache

import yaml

from app.core.resources import read_resource_text


def _load_yaml(relative_path):
    try:
        content = read_resource_text(relative_path)
    except FileNotFoundError:
        return {}
    return yaml.safe_load(content) or {}


@lru_cache(maxsize=1)
def load_rules():
    return _load_yaml("kb/rules.yaml")


@lru_cache(maxsize=1)
def load_examples():
    return _load_yaml("kb/examples.yaml").get("examples", [])


@lru_cache(maxsize=1)
def load_templates():
    return _load_yaml("kb/templates.yaml").get("templates", [])


@lru_cache(maxsize=1)
def load_critic_rules():
    return _load_yaml("kb/critic_rules.yaml").get("critic_rules", [])


def _tokenize(text):
    return {
        token.lower()
        for token in re.findall(r"[A-Za-zА-Яа-я_][A-Za-zА-Яа-я0-9_\-]*", text or "")
    }


def build_rule_lines(task_spec=None):
    rules = load_rules()
    lines = []
    lines.extend(rules.get("domain_invariants", []))

    lowcode = rules.get("lowcode_lua", {})
    compatibility_note = lowcode.get("compatibility_note")
    if compatibility_note:
        lines.append(compatibility_note)

    if task_spec and task_spec.output_style == "json_envelope":
        lines.append("Return a valid JSON object whose values are wrapped as lua{...}lua strings.")
    else:
        lines.append("Return raw LocalScript/Lua code only without markdown fences or explanations.")

    if task_spec and task_spec.target_root == "wf.initVariables":
        lines.append("Prefer wf.initVariables for launch variables referenced by the task.")
    elif task_spec and task_spec.target_root == "wf.vars":
        lines.append("Prefer wf.vars for workflow variables referenced by the task.")

    seen = set()
    ordered = []
    for line in lines:
        normalized = (line or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def select_examples(prompt, family=None, limit=2):
    prompt_tokens = _tokenize(prompt)
    scored = []
    for example in load_examples():
        score = 0
        if family and example.get("family") == family:
            score += 3
        prompt_match = len(prompt_tokens & _tokenize(example.get("prompt", "")))
        constraint_match = sum(
            1 for constraint in example.get("key_constraints", []) if _tokenize(constraint) & prompt_tokens
        )
        score += prompt_match + constraint_match
        if score <= 0:
            continue
        scored.append((score, example))

    scored.sort(key=lambda item: (-item[0], item[1].get("id", "")))
    return [example for _, example in scored[:limit]]


def select_critic_rules(prompt, validation_errors=None, limit=4):
    prompt_lower = (prompt or "").lower()
    errors = {error.lower() for error in (validation_errors or [])}
    scored = []
    for rule in load_critic_rules():
        score = 0
        detect = rule.get("detect", {})
        for hint in detect.get("prompt_hints", []):
            if hint.lower() in prompt_lower:
                score += 2
        rule_id = (rule.get("id") or "").lower()
        if rule_id and any(rule_id in error for error in errors):
            score += 3
        if "jsonpath" in errors and "jsonpath" in rule_id:
            score += 3
        if "markdown_fence_forbidden" in errors and "markdown" in rule_id:
            score += 3
        if score <= 0:
            continue
        scored.append((score, rule))

    scored.sort(key=lambda item: (-item[0], item[1].get("id", "")))
    return [rule for _, rule in scored[:limit]]
