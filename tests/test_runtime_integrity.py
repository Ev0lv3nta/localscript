import pytest

from app.workflow.contracts import CodeCandidate, OutputContract, OutputFormat, OutputShape
from app.workflow.validation import DeterministicCandidateValidator


def validate(code, data, shape="scalar"):
    return DeterministicCandidateValidator().validate_existing(
        candidate=CodeCandidate(code=code),
        output=OutputContract(format=OutputFormat.LUA_BLOCK, shape=OutputShape(shape)),
        context={"wf": {"vars": {"data": data}}},
    )


@pytest.mark.parametrize(
    "library,member",
    [("string", "gsub"), ("math", "floor"), ("table", "concat"), ("_utils.array", "new")],
)
def test_library_alias_cannot_modify_trusted_observer(library, member):
    result = validate(
        f'local alias = {library}; alias.{member} = function() return "tampered" end; return "original"',
        {},
    )
    assert not result.ok


@pytest.mark.parametrize(
    "data,shape",
    [({"x": None}, "object"), ([1, None], "array"), (2**53, "scalar"), (float("inf"), "scalar")],
)
def test_unsupported_input_is_rejected_without_loss(data, shape):
    result = validate("return wf.vars.data", data, shape)
    assert not result.ok
    assert result.checks[-1].code == "unsupported_json_context"


def test_exported_plain_lua_and_array_helper_preserve_false():
    result = validate(
        "return _utils.array.new(function(x) return x end, wf.vars.data)", [False, 0, True], "array"
    )
    assert result.ok
    assert result.observations == ({"actual": [False, 0, True], "read_roots": ["wf.vars"]},)


def test_exhausted_cpu_does_not_break_following_execution():
    assert not validate("while true do end", {}).ok
    assert validate("return 1", {}).ok


@pytest.mark.parametrize(
    "code", ['{"x":"lua{return 1}lua","x":"lua{return 2}lua"}', '{"x":"lua{}lua"}']
)
def test_envelope_parser_rejects_ambiguous_or_empty_input(code):
    from app.validation.lua_ast import analyze_lua_output
    from app.validation.runtime_executor import execute_output

    assert not analyze_lua_output(code, "json_envelope").ok
    assert not execute_output(code, {}, "json_envelope").ok


def test_nonempty_readonly_object_is_not_coerced_to_array():
    assert not validate("return wf.vars.data", {"a": 1}, "array").ok


@pytest.mark.parametrize(
    "code",
    [
        "return function() end",
        'return {[2] = "two", label = "ok"}',
        "local t = {}; t.self = t; return t",
        "return 0/0",
        "return 9007199254740992",
        'return string.rep("x", 70000)',
    ],
)
def test_unsupported_or_oversized_result_fails(code):
    assert not validate(code, {}).ok


@pytest.mark.parametrize(
    "data,shape",
    [
        (False, "scalar"),
        (0, "scalar"),
        ("Привет\n'", "scalar"),
        ([], "array"),
        ({}, "object"),
        ({"items": [], "obj": {}}, "object"),
    ],
)
def test_supported_values_round_trip(data, shape):
    result = validate("return wf.vars.data", data, shape)
    assert result.ok
    assert result.observations == ({"actual": data, "read_roots": ["wf.vars"]},)
