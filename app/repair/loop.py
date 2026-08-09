import re
from dataclasses import dataclass, field

from app.repair.canonical import CanonicalFamilyRepairer
from app.repair.critic import ValidationCritic

SHADOW_RENAME_RULES = {
    "table": {"replacement": "entry", "allowed_methods": ("insert", "concat", "sort", "remove", "unpack")},
    "string": {
        "replacement": "text_value",
        "allowed_methods": ("byte", "char", "find", "format", "gmatch", "gsub", "len", "lower", "match", "rep", "reverse", "sub", "upper"),
    },
    "math": {
        "replacement": "numeric_value",
        "allowed_methods": ("abs", "ceil", "floor", "max", "min", "modf", "random", "randomseed", "sqrt", "huge"),
    },
    "utf8": {"replacement": "utf8_value", "allowed_methods": ("char", "codes", "codepoint", "len", "offset")},
    "package": {"replacement": "package_item", "allowed_methods": ()},
    "_utils": {"replacement": "utils_value", "allowed_methods": ()},
    "wf": {"replacement": "wf_value", "allowed_methods": ()},
}


@dataclass
class RepairResult:
    code: str
    validation_report: object
    rounds: int
    history: list = field(default_factory=list)


class DeterministicRepairer:
    def __init__(self, formatter, canonical_repairer=None):
        self.formatter = formatter
        self.canonical_repairer = canonical_repairer or CanonicalFamilyRepairer()

    def apply(self, code, plan, task_spec):
        repaired = code
        for action in plan.actions:
            if self.canonical_repairer.supports(action.name):
                repaired = self.canonical_repairer.apply(action.name, task_spec)
                continue
            if action.name == "strip_markdown_fences":
                repaired = self._strip_markdown_fences(repaired)
            elif action.name == "rewrite_jsonpath_roots":
                repaired = self._rewrite_jsonpath_roots(repaired)
            elif action.name == "rewrite_unsupported_roots":
                repaired = self._rewrite_unsupported_roots(repaired, task_spec.target_root)
            elif action.name == "rename_shadowed_stdlib_locals":
                repaired = self._rename_shadowed_stdlib_locals(repaired)
            elif action.name == "append_return_nil":
                repaired = self._append_return_nil(repaired)
            elif action.name == "add_nil_safe_array_iteration":
                repaired = self._add_nil_safe_array_iteration(repaired)
            elif action.name == "normalize_empty_array_return":
                repaired = self._normalize_empty_array_return(repaired)
            elif action.name == "normalize_array_return_shape":
                repaired = self._normalize_array_return_shape(repaired)
            elif action.name == "normalize_scalar_return_shape":
                repaired = self._normalize_scalar_return_shape(repaired)
            elif action.name == "rewrite_string_trim":
                repaired = self._rewrite_string_trim(repaired)
            elif action.name == "rewrite_array_contains":
                repaired = self._rewrite_array_contains(repaired)

        return self.formatter.format(repaired, task_spec.output_style)

    @staticmethod
    def _strip_markdown_fences(code):
        text = (code or "").strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            return "\n".join(lines).strip()
        return text

    @staticmethod
    def _rewrite_jsonpath_roots(code):
        repaired = code
        repaired = re.sub(r"\$\.wf\.", "wf.", repaired)
        repaired = re.sub(r"\$\.vars\.", "wf.vars.", repaired)
        repaired = re.sub(r"\$\.initVariables\.", "wf.initVariables.", repaired)
        repaired = repaired.replace("$.", "")
        return repaired

    @staticmethod
    def _rewrite_unsupported_roots(code, target_root):
        repaired = code
        target = target_root or "wf.vars"
        repaired = repaired.replace("ctx.body.", "{0}.".format(target))
        repaired = repaired.replace("ctx.body", target)
        repaired = repaired.replace("workflow.variables.", "wf.vars.")
        repaired = repaired.replace("workflow.variables", "wf.vars")
        if target == "wf.initVariables":
            repaired = repaired.replace("wf.vars.", "wf.initVariables.")
            repaired = repaired.replace("wf.vars", "wf.initVariables")
        elif target == "wf.vars":
            repaired = repaired.replace("wf.initVariables.", "wf.vars.")
            repaired = repaired.replace("wf.initVariables", "wf.vars")
        return repaired

    @staticmethod
    def _rename_shadowed_stdlib_locals(code):
        repaired = code
        for identifier, rule in SHADOW_RENAME_RULES.items():
            escaped = re.escape(identifier)
            local_declaration = r"\blocal\s+(?:[A-Za-z_][A-Za-z0-9_]*\s*,\s*)*{0}\b".format(escaped)
            loop_declaration = r"\bfor\s+(?:[A-Za-z_][A-Za-z0-9_]*\s*,\s*)*{0}\s*(?:,|\bin\b)".format(escaped)
            if not (
                re.search(local_declaration, repaired)
                or re.search(loop_declaration, repaired)
            ):
                continue

            replacement = rule["replacement"]
            while re.search(r"\b{0}\b".format(re.escape(replacement)), repaired):
                replacement = replacement + "_value"

            repaired = re.sub(
                r"\blocal\s+{0}\b".format(re.escape(identifier)),
                "local {0}".format(replacement),
                repaired,
            )

            allowed_methods = rule["allowed_methods"]
            if allowed_methods:
                allowed_pattern = "|".join(re.escape(item) for item in allowed_methods)
                repaired = re.sub(
                    r"\b{0}\b(?!\s*\.(?:{1})\b)".format(re.escape(identifier), allowed_pattern),
                    replacement,
                    repaired,
                )
            else:
                repaired = re.sub(
                    r"\b{0}\b".format(re.escape(identifier)),
                    replacement,
                    repaired,
                )
        return repaired

    @staticmethod
    def _append_return_nil(code):
        if "return" in code:
            return code
        return "{0}\nreturn nil".format(code.rstrip())

    @staticmethod
    def _add_nil_safe_array_iteration(code):
        return re.sub(
            r"ipairs\(([^)\n]+)\)",
            lambda match: "ipairs({0} or {{}})".format(match.group(1).strip())
            if " or {}" not in match.group(1) and " or { }" not in match.group(1)
            else match.group(0),
            code,
        )

    @staticmethod
    def _normalize_empty_array_return(code):
        repaired = re.sub(r"return\s+nil\b", "return _utils.array.new()", code)
        repaired = re.sub(r"return\s+\{\s*\}\b", "return _utils.array.new()", repaired)
        return repaired

    @staticmethod
    def _normalize_array_return_shape(code):
        stripped = (code or "").strip()
        if stripped == "return nil":
            return "return _utils.array.new()"
        if stripped == "return {}":
            return "return _utils.array.new()"
        return code

    @staticmethod
    def _normalize_scalar_return_shape(code):
        repaired = re.sub(
            r"return\s+_utils\.array\.new\(\{\s*([^{}\n]+?)\s*\}\)",
            lambda match: "return {0}".format(match.group(1).strip()),
            code,
        )
        repaired = re.sub(
            r"return\s+\{\s*([^{}\n]+?)\s*\}",
            lambda match: "return {0}".format(match.group(1).strip()),
            repaired,
        )
        return repaired

    @staticmethod
    def _rewrite_string_trim(code):
        repaired = re.sub(
            r"string\.trim\(([^()\n]+)\)",
            lambda match: 'string.gsub(({0} or ""), "^%s*(.-)%s*$", "%1")'.format(match.group(1).strip()),
            code,
        )
        repaired = re.sub(
            r"([A-Za-z_][A-Za-z0-9_\.]*)\:trim\(\)",
            lambda match: 'string.gsub(({0} or ""), "^%s*(.-)%s*$", "%1")'.format(match.group(1).strip()),
            repaired,
        )
        repaired = re.sub(
            r"\btrim\(([^()\n]+)\)",
            lambda match: 'string.gsub(({0} or ""), "^%s*(.-)%s*$", "%1")'.format(match.group(1).strip()),
            repaired,
        )
        return DeterministicRepairer._normalize_scalar_return_shape(repaired)

    @staticmethod
    def _rewrite_array_contains(code):
        return re.sub(
            r"\b([A-Za-z_][A-Za-z0-9_]*)\s*:\s*contains\(([^()\n]+)\)",
            lambda match: (
                "(function() for _, candidate_value in ipairs({array_name} or {{}}) do "
                "if candidate_value == {needle} then return true end end return false end)()"
            ).format(
                array_name=match.group(1),
                needle=match.group(2).strip(),
            ),
            code,
        )

class RepairLoop:
    def __init__(self, validation_pipeline, formatter, critic=None, repairer=None):
        self.validation_pipeline = validation_pipeline
        self.formatter = formatter
        self.critic = critic or ValidationCritic()
        self.repairer = repairer or DeterministicRepairer(formatter)

    def run(self, code, task_spec, validation_report, profile, max_rounds, source_context=None, prompt="", planner_semantic_checks=None):
        current_code = code
        current_report = validation_report
        history = []

        for round_index in range(1, int(max_rounds) + 1):
            if not current_report.has_errors:
                break

            plan = self.critic.build_plan(task_spec, current_report, code=current_code)
            history_entry = {"round": round_index, "plan": plan.to_dict()}
            if not plan.repairable:
                history_entry["status"] = "no_actions"
                history.append(history_entry)
                break

            repaired_code = self.repairer.apply(current_code, plan, task_spec)
            if repaired_code == current_code:
                history_entry["status"] = "no_change"
                history.append(history_entry)
                break

            current_code = repaired_code
            current_report = self.validation_pipeline.run(
                code=current_code,
                task_spec=task_spec,
                profile=profile,
                source_context=source_context,
                prompt=prompt,
                planner_semantic_checks=planner_semantic_checks,
            )
            history_entry["status"] = "revalidated"
            history_entry["validation_report"] = current_report.to_dict()
            history.append(history_entry)

            if not current_report.has_errors:
                break

        return RepairResult(
            code=current_code,
            validation_report=current_report,
            rounds=len(history),
            history=history,
        )
