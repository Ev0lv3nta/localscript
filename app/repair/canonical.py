import json
import re

CANONICAL_REPAIR_ACTIONS = frozenset(
    {
        "rewrite_augment_existing_code_envelope",
        "rewrite_datum_time_to_iso8601",
        "rewrite_email_validation",
        "rewrite_ensure_items_array",
        "rewrite_iso8601_to_epoch",
        "rewrite_rest_cleanup_keep_only",
    }
)


class CanonicalFamilyRepairer:
    """Replace an invalid family candidate with its trusted canonical form."""

    def supports(self, action_name):
        return action_name in CANONICAL_REPAIR_ACTIONS

    def apply(self, action_name, task_spec):
        renderer = getattr(self, "_{0}".format(action_name), None)
        if renderer is None:
            raise ValueError("unsupported_canonical_repair::{0}".format(action_name))
        return renderer(task_spec)

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
