import json
import re
from dataclasses import dataclass, field

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
    "_utils": {"replacement": "utils_value", "allowed_methods": ()},
    "wf": {"replacement": "wf_value", "allowed_methods": ()},
}


@dataclass
class RepairResult:
    code: str
    validation_report: object
    rounds: int
    history: list = field(default_factory=list)


class MinimalRepairer:
    def __init__(self, formatter):
        self.formatter = formatter

    def apply(self, code, plan, task_spec):
        repaired = code
        for action in plan.actions:
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
            elif action.name == "rewrite_datum_time_to_iso8601":
                repaired = self._rewrite_datum_time_to_iso8601(task_spec)
            elif action.name == "rewrite_iso8601_to_epoch":
                repaired = self._rewrite_iso8601_to_epoch(task_spec)
            elif action.name == "rewrite_rest_cleanup_keep_only":
                repaired = self._rewrite_rest_cleanup_keep_only(task_spec)
            elif action.name == "rewrite_augment_existing_code_envelope":
                repaired = self._rewrite_augment_existing_code_envelope(task_spec)
            elif action.name == "rewrite_ensure_items_array":
                repaired = self._rewrite_ensure_items_array(task_spec)
            elif action.name == "rewrite_email_validation":
                repaired = self._rewrite_email_validation(task_spec)

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
            if not re.search(r"\blocal\s+{0}\b".format(re.escape(identifier)), repaired):
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
        return MinimalRepairer._normalize_scalar_return_shape(repaired)

    @staticmethod
    def _rewrite_iso8601_to_epoch(task_spec):
        iso_path = (task_spec.generation_hints or {}).get("iso_path") or "wf.initVariables.recallTime"
        return (
            "local iso = {iso_path}\n"
            "if not iso then\n"
            "  return nil\n"
            "end\n"
            "local y, m, d, h, mi, s, sign, oh, om = iso:match(\"^(%d%d%d%d)%-(%d%d)%-(%d%d)T(%d%d):(%d%d):(%d%d)([+-])(%d%d):(%d%d)$\")\n"
            "if not y then\n"
            "  return nil\n"
            "end\n"
            "y, m, d, h, mi, s, oh, om = tonumber(y), tonumber(m), tonumber(d), tonumber(h), tonumber(mi), tonumber(s), tonumber(oh), tonumber(om)\n"
            "local function leap(year)\n"
            "  return (year % 4 == 0 and year % 100 ~= 0) or (year % 400 == 0)\n"
            "end\n"
            "local md = {{31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}}\n"
            "local days = d - 1\n"
            "for year = 1970, y - 1 do\n"
            "  days = days + (leap(year) and 366 or 365)\n"
            "end\n"
            "for month = 1, m - 1 do\n"
            "  days = days + md[month] + (month == 2 and leap(y) and 1 or 0)\n"
            "end\n"
            "local offset = oh * 3600 + om * 60\n"
            "if sign == \"-\" then\n"
            "  offset = -offset\n"
            "end\n"
            "return days * 86400 + h * 3600 + mi * 60 + s - offset"
        ).format(iso_path=iso_path)

    @staticmethod
    def _rewrite_datum_time_to_iso8601(task_spec):
        hints = task_spec.generation_hints or {}
        datum_path = hints.get("datum_path") or "wf.vars.json.IDOC.ZCDF_HEAD.DATUM"
        time_path = hints.get("time_path") or "wf.vars.json.IDOC.ZCDF_HEAD.TIME"
        return (
            "local DATUM = {datum_path}\n"
            "local TIME = {time_path}\n"
            "local function safe_sub(str, start, finish)\n"
            "  local s = string.sub(str or \"\", start, math.min(finish, #(str or \"\")))\n"
            "  return s ~= \"\" and s or \"00\"\n"
            "end\n"
            "local digits = string.gsub(TIME or \"\", \"%D\", \"\")\n"
            "if #digits <= 2 then\n"
            "  TIME = string.rep(\"0\", math.max(0, 2 - #digits)) .. digits .. \"0000\"\n"
            "elseif #digits <= 4 then\n"
            "  TIME = string.rep(\"0\", math.max(0, 4 - #digits)) .. digits .. \"00\"\n"
            "else\n"
            "  TIME = string.sub((digits .. \"000000\"), 1, 6)\n"
            "end\n"
            "local year = safe_sub(DATUM, 1, 4)\n"
            "local month = safe_sub(DATUM, 5, 6)\n"
            "local day = safe_sub(DATUM, 7, 8)\n"
            "local hour = safe_sub(TIME, 1, 2)\n"
            "local minute = safe_sub(TIME, 3, 4)\n"
            "local second = safe_sub(TIME, 5, 6)\n"
            "return string.format('%s-%s-%sT%s:%s:%s.00000Z', year, month, day, hour, minute, second)"
        ).format(datum_path=datum_path, time_path=time_path)

    @staticmethod
    def _rewrite_rest_cleanup_keep_only(task_spec):
        hints = task_spec.generation_hints or {}
        result_path = hints.get("result_path") or "wf.vars.RESTbody.result"
        keep_keys = [str(key) for key in hints.get("keep_keys", []) if key]
        if not keep_keys:
            keep_keys = ["ID", "ENTITY_ID", "CALL"]
        guard = " and ".join('key ~= "{0}"'.format(key) for key in keep_keys)
        return (
            "local result = {result_path}\n"
            "for _, filteredEntry in pairs(result or {{}}) do\n"
            "  for key, _ in pairs(filteredEntry) do\n"
            "    if {guard} then\n"
            "      filteredEntry[key] = nil\n"
            "    end\n"
            "  end\n"
            "end\n"
            "return result"
        ).format(result_path=result_path, guard=guard)

    @staticmethod
    def _rewrite_augment_existing_code_envelope(task_spec):
        hints = task_spec.generation_hints or {}
        raw_literal = str(hints.get("number_literal") or "5")
        match = re.search(r"-?\d+(?:\.\d+)?", raw_literal)
        literal = match.group(0) if match else "5"
        payload = {
            "num": "lua{return tonumber('" + literal + "')}lua",
            "squared": "lua{local n = tonumber('" + literal + "')\nreturn n * n}lua",
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @staticmethod
    def _rewrite_ensure_items_array(task_spec):
        hints = task_spec.generation_hints or {}
        packages_path = hints.get("packages_path") or "wf.vars.json.IDOC.ZCDF_HEAD.ZCDF_PACKAGES"
        item_field = hints.get("item_field") or "items"
        return (
            "local function ensureArray(value)\n"
            "  if type(value) ~= \"table\" then\n"
            "    return {{value}}\n"
            "  end\n"
            "  local isArray = true\n"
            "  for key, _ in pairs(value) do\n"
            "    if type(key) ~= \"number\" or math.floor(key) ~= key then\n"
            "      isArray = false\n"
            "      break\n"
            "    end\n"
            "  end\n"
            "  if isArray then\n"
            "    return value\n"
            "  end\n"
            "  return {{value}}\n"
            "end\n"
            "local packages = {packages_path} or {{}}\n"
            "for _, pkg in ipairs(packages) do\n"
            "  if type(pkg) == \"table\" and pkg.{item_field} ~= nil then\n"
            "    pkg.{item_field} = ensureArray(pkg.{item_field})\n"
            "  end\n"
            "end\n"
            "return packages"
        ).format(packages_path=packages_path, item_field=item_field)

    @staticmethod
    def _rewrite_email_validation(task_spec):
        hints = task_spec.generation_hints or {}
        email_path = hints.get("email_path") or "wf.vars.email"
        return (
            "local email = {email_path}\n"
            "if not email or email == \"\" then\n"
            "  return false\n"
            "end\n"
            "return string.match(email, \"^[A-Za-z0-9._%+%-]+@[A-Za-z0-9.%-]+%.[A-Za-z]+$\") ~= nil"
        ).format(email_path=email_path)


class RepairLoop:
    def __init__(self, validation_pipeline, formatter, critic=None, repairer=None):
        self.validation_pipeline = validation_pipeline
        self.formatter = formatter
        self.critic = critic or ValidationCritic()
        self.repairer = repairer or MinimalRepairer(formatter)

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
