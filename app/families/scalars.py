import re
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from app.families.base import FamilyDefinition, FamilyFinding
from app.families.support import find_first_by_predicate, resolve_path


def _counter_increment(hints: Mapping[str, Any], context: Any) -> Any:
    path = hints.get("counter_path") or find_first_by_predicate(
        context,
        lambda _key, nested: isinstance(nested, (int, float)),
    )
    value = resolve_path(context, path)
    return value + 1 if isinstance(value, (int, float)) else None


def _normalize_short_time(value: Any) -> str:
    digits = re.sub(r"\D", "", str(value or ""))
    if len(digits) <= 2:
        return digits.zfill(2) + "0000"
    if len(digits) <= 4:
        return digits.zfill(4) + "00"
    return digits[:6].zfill(6)


def _datum_time_to_iso8601(hints: Mapping[str, Any], context: Any) -> Any:
    datum_path = hints.get("datum_path") or find_first_by_predicate(
        context,
        lambda key, _nested: key.lower() == "datum",
    )
    time_path = hints.get("time_path") or find_first_by_predicate(
        context,
        lambda key, _nested: key.lower() == "time",
    )
    datum = resolve_path(context, datum_path) or ""
    time_value = _normalize_short_time(resolve_path(context, time_path) or "")
    if not datum:
        return None
    return "{0}-{1}-{2}T{3}:{4}:{5}.00000Z".format(
        (datum[0:4] or "0000").ljust(4, "0"),
        (datum[4:6] or "00").ljust(2, "0"),
        (datum[6:8] or "00").ljust(2, "0"),
        (time_value[0:2] or "00").ljust(2, "0"),
        (time_value[2:4] or "00").ljust(2, "0"),
        (time_value[4:6] or "00").ljust(2, "0"),
    )


def _iso8601_to_epoch(hints: Mapping[str, Any], context: Any) -> Any:
    value = resolve_path(context, hints.get("iso_path"))
    if not value:
        return None
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())


def _email_validation(hints: Mapping[str, Any], context: Any) -> bool:
    email_path = hints.get("email_path") or find_first_by_predicate(
        context,
        lambda key, _nested: key.lower() == "email",
    )
    email = resolve_path(context, email_path)
    if not email:
        return False
    return re.fullmatch(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]+", email) is not None


def _validate_email(code: str, _style: str, _hints: Mapping[str, Any]):
    if ("string.match" in code or ":match(" in code) and "~= nil" in code:
        return ()
    return (
        FamilyFinding(
            "email_validation_boolean_missing",
            "Email validation must return an explicit boolean via string.match(... ) ~= nil.",
        ),
    )


def _normalize_email(hints: Mapping[str, Any], context: Any) -> str:
    email_path = hints.get("email_path") or find_first_by_predicate(
        context,
        lambda key, _nested: key.lower() in {"email", "useremail"},
    )
    value = resolve_path(context, email_path) or ""
    return re.sub(r"^\s*(.*?)\s*$", r"\1", str(value)).lower()


def _validate_normalize_email(code: str, _style: str, _hints: Mapping[str, Any]):
    findings = []
    if "string.lower" not in code and ":lower()" not in code:
        findings.append(
            FamilyFinding(
                "normalize_email_lower_missing",
                "Email normalization must convert the final scalar to lower case.",
            )
        )
    return tuple(findings)


def _translate_lua_pattern(pattern: str | None) -> str:
    result = []
    classes = {
        "d": r"\d",
        "u": r"[A-Z]",
        "l": r"[a-z]",
        "a": r"[A-Za-z]",
        "w": r"[A-Za-z0-9]",
        "s": r"\s",
    }
    index = 0
    pattern = pattern or ""
    while index < len(pattern):
        char = pattern[index]
        if char == "%" and index + 1 < len(pattern):
            token = pattern[index + 1]
            result.append(classes.get(token, re.escape(token)))
            index += 2
            continue
        result.append(char)
        index += 1
    return "".join(result)


def _regex_extract(hints: Mapping[str, Any], context: Any) -> Any:
    source = resolve_path(context, hints.get("source_path")) or ""
    match = re.search(_translate_lua_pattern(hints.get("pattern")), source)
    if not match:
        return None
    return match.group(1) if match.groups() else match.group(0)


def _validate_regex(code: str, _style: str, _hints: Mapping[str, Any]):
    if "string.match" in code or ":match(" in code:
        return ()
    return (
        FamilyFinding(
            "regex_extract_missing_string_match",
            "Regex extraction tasks must use string.match.",
        ),
    )


SCALAR_FAMILIES = (
    FamilyDefinition(
        "counter_increment",
        preferred_return_shape="scalar",
        expected_result_builder=_counter_increment,
    ),
    FamilyDefinition(
        "datum_time_to_iso8601",
        preferred_return_shape="scalar",
        expected_result_builder=_datum_time_to_iso8601,
    ),
    FamilyDefinition(
        "iso8601_to_epoch",
        preferred_return_shape="scalar",
        expected_result_builder=_iso8601_to_epoch,
    ),
    FamilyDefinition(
        "email_validation",
        preferred_return_shape="scalar",
        expected_result_builder=_email_validation,
        structural_validator=_validate_email,
    ),
    FamilyDefinition(
        "normalize_email_string",
        preferred_return_shape="scalar",
        expected_result_builder=_normalize_email,
        structural_validator=_validate_normalize_email,
    ),
    FamilyDefinition(
        "regex_extract",
        preferred_return_shape="scalar",
        expected_result_builder=_regex_extract,
        structural_validator=_validate_regex,
    ),
)
