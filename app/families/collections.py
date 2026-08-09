from collections.abc import Mapping
from typing import Any

from app.families.base import FamilyDefinition, FamilyFinding
from app.families.support import (
    deep_copy,
    find_first_by_predicate,
    matches_condition,
    resolve_path,
    safe_nested,
)


def _last_array_item(hints: Mapping[str, Any], context: Any) -> Any:
    path = hints.get("source_path") or find_first_by_predicate(
        context,
        lambda _key, nested: isinstance(nested, list),
    )
    items = resolve_path(context, path)
    return items[-1] if isinstance(items, list) and items else None


def _rest_cleanup(hints: Mapping[str, Any], context: Any) -> Any:
    rows = resolve_path(context, hints.get("result_path"))
    keep_keys = hints.get("keep_keys") or []
    if not isinstance(rows, list):
        return None
    cleaned = []
    for row in rows:
        if not isinstance(row, dict):
            cleaned.append(row)
            continue
        cleaned.append(
            {key: deep_copy(value) for key, value in row.items() if key in keep_keys}
        )
    return cleaned


def _validate_rest_cleanup(
    code: str,
    _output_style: str,
    hints: Mapping[str, Any],
) -> tuple[FamilyFinding, ...]:
    keep_keys = {str(key) for key in hints.get("keep_keys", []) if key}
    findings = []
    for raw_key in hints.get("available_keys", []):
        key = str(raw_key)
        if key in keep_keys:
            continue
        if (
            '"{0}"'.format(key) in code
            or "'{0}'".format(key) in code
            or ".{0}".format(key) in code
        ):
            findings.append(
                FamilyFinding(
                    code="rest_cleanup_excluded_key_reference::{0}".format(key),
                    message=(
                        "REST cleanup must not reference excluded key `{0}` directly; "
                        "preserve only requested keys generically."
                    ).format(key),
                )
            )
    return tuple(findings)


def _ensure_items_array(hints: Mapping[str, Any], context: Any) -> Any:
    packages = resolve_path(context, hints.get("packages_path"))
    item_field = hints.get("item_field", "items")
    if not isinstance(packages, list):
        return packages
    result = deep_copy(packages)
    for package in result:
        if isinstance(package, dict) and item_field in package:
            value = package[item_field]
            package[item_field] = deep_copy(value) if isinstance(value, list) else [deep_copy(value)]
    return result


def _filter_discount_markdown(hints: Mapping[str, Any], context: Any) -> Any:
    rows = resolve_path(context, hints.get("items_path")) or []
    discount_field = hints.get("discount_field", "Discount")
    markdown_field = hints.get("markdown_field", "Markdown")
    return [
        deep_copy(row)
        for row in rows
        if isinstance(row, dict)
        and (
            row.get(discount_field) not in ("", None)
            or row.get(markdown_field) not in ("", None)
        )
    ]


def _field_mapping(hints: Mapping[str, Any], context: Any) -> Any:
    source = resolve_path(context, hints.get("source_path"))
    if source is None:
        return None
    return {
        pair["target"]: safe_nested(source, pair["source"])
        for pair in hints.get("mapping_pairs") or []
    }


def _table_transform(hints: Mapping[str, Any], context: Any) -> Any:
    source = resolve_path(context, hints.get("source_path")) or []
    return [
        {
            pair["target"]: safe_nested(item, pair["source"])
            for pair in hints.get("mapping_pairs") or []
        }
        for item in source
    ]


def _conditional_projection(hints: Mapping[str, Any], context: Any) -> Any:
    source = resolve_path(context, hints.get("source_path")) or []
    field_name = hints.get("projection_field")
    conditions = hints.get("conditions") or []
    return [
        safe_nested(item, field_name)
        for item in source
        if isinstance(item, dict)
        and all(matches_condition(item, condition) for condition in conditions)
    ]


def _requires_array_constructor(code: str, error_code: str, message: str):
    if "_utils.array.new()" in code:
        return ()
    return (FamilyFinding(code=error_code, message=message),)


COLLECTION_FAMILIES = (
    FamilyDefinition(
        "last_array_item",
        preferred_return_shape="scalar",
        expected_result_builder=_last_array_item,
    ),
    FamilyDefinition(
        "rest_cleanup",
        preferred_return_shape="array",
        expected_result_builder=_rest_cleanup,
        structural_validator=_validate_rest_cleanup,
    ),
    FamilyDefinition(
        "ensure_items_array",
        preferred_return_shape="array",
        expected_result_builder=_ensure_items_array,
    ),
    FamilyDefinition(
        "filter_discount_markdown",
        preferred_return_shape="array",
        expected_result_builder=_filter_discount_markdown,
        structural_validator=lambda code, _style, _hints: _requires_array_constructor(
            code,
            "array_constructor_missing",
            "Filtering case must initialize result via _utils.array.new().",
        ),
    ),
    FamilyDefinition(
        "field_mapping",
        preferred_return_shape="object",
        expected_result_builder=_field_mapping,
    ),
    FamilyDefinition(
        "table_transform",
        preferred_return_shape="array",
        expected_result_builder=_table_transform,
        structural_validator=lambda code, _style, _hints: _requires_array_constructor(
            code,
            "table_transform_missing_array_constructor",
            "Table transform tasks must build arrays via _utils.array.new().",
        ),
    ),
    FamilyDefinition(
        "conditional_array_projection",
        preferred_return_shape="array",
        expected_result_builder=_conditional_projection,
        structural_validator=lambda code, _style, _hints: _requires_array_constructor(
            code,
            "conditional_projection_missing_array_constructor",
            "Conditional array projection must build arrays via _utils.array.new().",
        ),
    ),
)
