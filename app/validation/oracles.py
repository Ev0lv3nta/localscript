import copy
import re
from datetime import datetime


UNSUPPORTED = object()


def _normalize_path(path):
    return (path or "").replace("[]", "")


def resolve_path(value, path):
    if not path:
        return value
    current = value
    for chunk in _normalize_path(path).split("."):
        if not chunk:
            continue
        if not isinstance(current, dict) or chunk not in current:
            return None
        current = current[chunk]
    return current


def _deep_copy(value):
    return copy.deepcopy(value)


def _ensure_array(value):
    if isinstance(value, list):
        return _deep_copy(value)
    return [_deep_copy(value)]


def _translate_lua_pattern(pattern):
    result = []
    index = 0
    length = len(pattern or "")
    classes = {
        "d": r"\d",
        "u": r"[A-Z]",
        "l": r"[a-z]",
        "a": r"[A-Za-z]",
        "w": r"[A-Za-z0-9]",
        "s": r"\s",
    }
    while index < length:
        char = pattern[index]
        if char == "%" and index + 1 < length:
            token = pattern[index + 1]
            if token in classes:
                result.append(classes[token])
            else:
                result.append(re.escape(token))
            index += 2
            continue
        result.append(char)
        index += 1
    return "".join(result)


def _expected_regex_extract(source_value, pattern):
    translated = _translate_lua_pattern(pattern)
    match = re.search(translated, source_value or "")
    if not match:
        return None
    if match.groups():
        return match.group(1)
    return match.group(0)


def _expected_iso8601_to_epoch(value):
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    return int(datetime.fromisoformat(normalized).timestamp())


def _normalize_short_time(value):
    digits = re.sub(r"\D", "", str(value or ""))
    if len(digits) <= 2:
        return digits.zfill(2) + "0000"
    if len(digits) <= 4:
        return digits.zfill(4) + "00"
    return digits[:6].zfill(6)


def _safe_nested(source, path):
    current = source
    for part in (path or "").split("."):
        if current is None:
            return None
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _find_first_by_predicate(value, predicate, prefix=""):
    if isinstance(value, dict):
        for key, nested in value.items():
            next_prefix = "{0}.{1}".format(prefix, key) if prefix else key
            if predicate(key, nested):
                return next_prefix
            found = _find_first_by_predicate(nested, predicate, next_prefix)
            if found:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _find_first_by_predicate(nested, predicate, prefix)
            if found:
                return found
    return None


def _coerce_condition_value(value):
    if isinstance(value, str):
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value
    return value


def _matches_condition(item, condition):
    actual = _safe_nested(item, condition.get("field"))
    operator = condition.get("operator")
    expected = _coerce_condition_value(condition.get("value"))
    if operator == "eq":
        return actual == expected
    if operator == "gt":
        return isinstance(actual, (int, float)) and actual > expected
    if operator == "gte":
        return isinstance(actual, (int, float)) and actual >= expected
    if operator == "lt":
        return isinstance(actual, (int, float)) and actual < expected
    if operator == "lte":
        return isinstance(actual, (int, float)) and actual <= expected
    if operator == "not_nil":
        return actual is not None
    return False


def build_expected_result(task_spec, context):
    family = task_spec.family
    params = task_spec.generation_hints or {}

    if family == "last_array_item":
        path = params.get("source_path") or _find_first_by_predicate(context, lambda _key, nested: isinstance(nested, list))
        items = resolve_path(context, path)
        return items[-1] if isinstance(items, list) and items else None

    if family == "counter_increment":
        path = params.get("counter_path") or _find_first_by_predicate(context, lambda _key, nested: isinstance(nested, (int, float)))
        value = resolve_path(context, path)
        return value + 1 if isinstance(value, (int, float)) else None

    if family == "rest_cleanup":
        rows = resolve_path(context, params.get("result_path"))
        keep_keys = params.get("keep_keys") or []
        if not isinstance(rows, list):
            return None
        cleaned = []
        for row in rows:
            if not isinstance(row, dict):
                cleaned.append(row)
                continue
            cleaned.append({key: _deep_copy(value) for key, value in row.items() if key in keep_keys})
        return cleaned

    if family == "datum_time_to_iso8601":
        datum_path = params.get("datum_path") or _find_first_by_predicate(context, lambda key, _nested: key.lower() == "datum")
        time_path = params.get("time_path") or _find_first_by_predicate(context, lambda key, _nested: key.lower() == "time")
        datum = resolve_path(context, datum_path) or ""
        time_value = _normalize_short_time(resolve_path(context, time_path) or "")
        if not datum:
            return None
        year = (datum[0:4] or "0000").ljust(4, "0")
        month = (datum[4:6] or "00").ljust(2, "0")
        day = (datum[6:8] or "00").ljust(2, "0")
        hour = (time_value[0:2] or "00").ljust(2, "0")
        minute = (time_value[2:4] or "00").ljust(2, "0")
        second = (time_value[4:6] or "00").ljust(2, "0")
        return "{0}-{1}-{2}T{3}:{4}:{5}.00000Z".format(year, month, day, hour, minute, second)

    if family == "ensure_items_array":
        packages = resolve_path(context, params.get("packages_path"))
        item_field = params.get("item_field", "items")
        if not isinstance(packages, list):
            return packages
        result = _deep_copy(packages)
        for package in result:
            if isinstance(package, dict) and item_field in package:
                package[item_field] = _ensure_array(package[item_field])
        return result

    if family == "filter_discount_markdown":
        rows = resolve_path(context, params.get("items_path")) or []
        discount_field = params.get("discount_field", "Discount")
        markdown_field = params.get("markdown_field", "Markdown")
        filtered = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            discount = row.get(discount_field)
            markdown = row.get(markdown_field)
            if (discount not in ("", None)) or (markdown not in ("", None)):
                filtered.append(_deep_copy(row))
        return filtered

    if family == "augment_existing_code":
        number_literal = int(params.get("number_literal", "5"))
        return {"num": number_literal, "squared": number_literal * number_literal}

    if family == "iso8601_to_epoch":
        return _expected_iso8601_to_epoch(resolve_path(context, params.get("iso_path")))

    if family == "email_validation":
        email_path = params.get("email_path") or _find_first_by_predicate(context, lambda key, _nested: key.lower() == "email")
        email = resolve_path(context, email_path)
        if not email:
            return False
        return re.fullmatch(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]+", email) is not None

    if family == "normalize_email_string":
        email_path = params.get("email_path") or _find_first_by_predicate(
            context,
            lambda key, _nested: key.lower() in {"email", "useremail"},
        )
        value = resolve_path(context, email_path) or ""
        return re.sub(r"^\s*(.*?)\s*$", r"\1", str(value)).lower()

    if family == "regex_extract":
        value = resolve_path(context, params.get("source_path")) or ""
        return _expected_regex_extract(value, params.get("pattern"))

    if family == "field_mapping":
        source = resolve_path(context, params.get("source_path"))
        if source is None:
            return None
        result = {}
        for pair in params.get("mapping_pairs") or []:
            result[pair["target"]] = _safe_nested(source, pair["source"])
        return result

    if family == "table_transform":
        source = resolve_path(context, params.get("source_path")) or []
        result = []
        for item in source:
            mapped = {}
            for pair in params.get("mapping_pairs") or []:
                mapped[pair["target"]] = _safe_nested(item, pair["source"])
            result.append(mapped)
        return result

    if family == "conditional_array_projection":
        source = resolve_path(context, params.get("source_path")) or []
        result = []
        field_name = params.get("projection_field")
        conditions = params.get("conditions") or []
        for item in source:
            if not isinstance(item, dict):
                continue
            if all(_matches_condition(item, condition) for condition in conditions):
                result.append(_safe_nested(item, field_name))
        return result

    return UNSUPPORTED


def compare_expected_and_actual(expected, actual):
    return expected == actual
