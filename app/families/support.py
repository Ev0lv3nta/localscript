import copy
from collections.abc import Callable
from typing import Any


def normalize_path(path: str | None) -> str:
    return (path or "").replace("[]", "")


def resolve_path(value: Any, path: str | None) -> Any:
    if not path:
        return value
    current = value
    for chunk in normalize_path(path).split("."):
        if not chunk:
            continue
        if not isinstance(current, dict) or chunk not in current:
            return None
        current = current[chunk]
    return current


def deep_copy(value: Any) -> Any:
    return copy.deepcopy(value)


def safe_nested(source: Any, path: str | None) -> Any:
    current = source
    for part in (path or "").split("."):
        if current is None or not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def find_first_by_predicate(
    value: Any,
    predicate: Callable[[str, Any], bool],
    prefix: str = "",
) -> str | None:
    if isinstance(value, dict):
        for key, nested in value.items():
            next_prefix = "{0}.{1}".format(prefix, key) if prefix else key
            if predicate(key, nested):
                return next_prefix
            found = find_first_by_predicate(nested, predicate, next_prefix)
            if found:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = find_first_by_predicate(nested, predicate, prefix)
            if found:
                return found
    return None


def coerce_condition_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return float(value) if "." in value else int(value)
    except ValueError:
        return value


def matches_condition(item: Any, condition: dict[str, Any]) -> bool:
    actual = safe_nested(item, condition.get("field"))
    operator = condition.get("operator")
    expected = coerce_condition_value(condition.get("value"))
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
