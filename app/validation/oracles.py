from app.families import UNSUPPORTED, get_family_definition
from app.families.support import resolve_path


def build_expected_result(task_spec, context):
    definition = get_family_definition(task_spec.family)
    if definition is None:
        return UNSUPPORTED
    return definition.build_expected_result(
        task_spec.generation_hints or {},
        context,
    )


def compare_expected_and_actual(expected, actual):
    if isinstance(expected, dict) and isinstance(actual, dict):
        keys = set(expected) | set(actual)
        return all(
            compare_expected_and_actual(expected.get(key), actual.get(key))
            for key in keys
        )
    if isinstance(expected, list) and isinstance(actual, list):
        return len(expected) == len(actual) and all(
            compare_expected_and_actual(expected_item, actual_item)
            for expected_item, actual_item in zip(expected, actual)
        )
    return expected == actual


__all__ = [
    "UNSUPPORTED",
    "build_expected_result",
    "compare_expected_and_actual",
    "resolve_path",
]
