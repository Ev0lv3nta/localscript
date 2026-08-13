import subprocess

import pytest

from app.validation.lua_ast import analyze_lua_chunk, analyze_lua_output
from app.validation.runtime_executor import execute_output


def _codes(code: str) -> list[str]:
    return [finding.code for finding in analyze_lua_chunk(code).findings]


def test_comments_and_string_literals_are_not_treated_as_executable_code():
    code = '-- os.execute("ignored")\nreturn "require(\\"also ignored\\")"'

    assert analyze_lua_chunk(code).ok


def test_forbidden_global_is_found_in_dead_branch():
    codes = _codes('if false then os.execute("still forbidden") end\nreturn true')

    assert "dangerous_stdlib_os_forbidden" in codes


@pytest.mark.parametrize(
    ("expression", "expected_code"),
    [
        ('require("socket")', "dangerous_stdlib_require_forbidden"),
        ('load("return 1")', "dangerous_stdlib_load_forbidden"),
        ('loadfile("payload.lua")', "dangerous_stdlib_loadfile_forbidden"),
        ('dofile("payload.lua")', "dangerous_stdlib_dofile_forbidden"),
    ],
)
def test_dynamic_loading_is_forbidden(expression: str, expected_code: str):
    assert expected_code in _codes(f"return {expression}")


def test_global_assignment_is_forbidden():
    assert "lua_global_assignment_forbidden::answer" in _codes("answer = 42\nreturn answer")


@pytest.mark.parametrize(
    "target",
    [
        "wf.vars.customer.name",
        'wf["vars"]["customer"]["name"]',
    ],
)
def test_nested_workflow_assignment_is_forbidden(target: str):
    assert "lua_wf_mutation_forbidden" in _codes(f'{target} = "changed"\nreturn true')


def test_workflow_alias_assignment_is_forbidden():
    code = "local customer = wf.vars.customer\ncustomer.name = 'changed'\nreturn customer"

    assert "lua_wf_mutation_forbidden" in _codes(code)


def test_multiple_assignment_tracks_workflow_aliases_before_updating_locals():
    code = (
        "local first = wf.vars\nlocal second = {}\nfirst, second = second, first\nsecond.value = 1"
    )

    assert "lua_wf_mutation_forbidden" in _codes(code)


def test_local_declaration_without_initializer_is_in_scope_for_later_assignment():
    code = "local project\nproject = function(value) return value end\nreturn project(1)"

    assert analyze_lua_chunk(code).ok


def test_local_attributes_do_not_shift_workflow_alias_tracking():
    code = "local first <const>, second <close> = {}, wf.vars\nif false then second.value = 1 end"

    assert "lua_wf_mutation_forbidden" in _codes(code)


def test_reserved_workflow_identifier_cannot_be_shadowed():
    code = "local wf = {}\nwf.value = 1\nreturn wf"

    assert "lua_reserved_identifier_shadowed::wf" in _codes(code)


@pytest.mark.parametrize("method", ["insert", "remove", "sort"])
def test_table_mutator_call_on_workflow_input_is_forbidden(method: str):
    code = f"local items = wf.vars.items\ntable.{method}(items, 1)\nreturn items"

    assert "lua_wf_mutation_forbidden" in _codes(code)


def test_table_mutator_call_is_forbidden_even_in_dead_branch():
    code = "if false then table.insert(wf.vars.items, 1) end\nreturn true"

    assert "lua_wf_mutation_forbidden" in _codes(code)


@pytest.mark.parametrize(
    "code",
    [
        "function wf.change() return true end",
        "function wf:change() return true end",
        "local target = wf.vars\nfunction target.change() return true end",
    ],
)
def test_function_declaration_cannot_write_to_workflow_input(code: str):
    assert "lua_wf_mutation_forbidden" in _codes(code)


def test_locals_parameters_loops_local_functions_and_shadowing_are_allowed():
    code = """
local package = {}
local function project(items)
  local result = _utils.array.new()
  for index, value in ipairs(items or {}) do
    package[index] = value
    table.insert(result, { index = index, value = value })
  end
  return result
end
return project(wf.vars.items)
"""

    assert analyze_lua_chunk(code).ok


def test_local_declaration_rhs_uses_outer_scope():
    codes = _codes("local os = os\nreturn os")

    assert "dangerous_stdlib_os_forbidden" in codes


def test_syntax_error_nodes_are_reported_by_ast_policy():
    assert "lua_ast_syntax_error" in _codes("return function(")


def test_every_json_envelope_chunk_is_analyzed():
    code = '{"safe":"lua{return wf.vars.value}lua","unsafe":"lua{if false then os.execute(\\"x\\") end return 1}lua"}'

    result = analyze_lua_output(code, "json_envelope")

    assert [finding.code for finding in result.findings] == ["dangerous_stdlib_os_forbidden"]
    assert result.findings[0].chunk_index == 2


def test_runtime_readonly_boundary_rejects_mutation_hidden_behind_parameter():
    code = """
local function mutate(value)
  value.name = "changed"
end
mutate(wf.vars.customer)
return wf.vars.customer.name
"""

    assert analyze_lua_chunk(code).ok
    execution = execute_output(
        code,
        {"wf": {"vars": {"customer": {"name": "original"}}}},
    )

    assert execution.ok is False
    assert execution.error_code == "lua_runtime_error"
    assert "workflow input is read-only" in execution.error_message


def test_normal_workflow_transform_passes_static_and_runtime_boundaries():
    code = """
local result = _utils.array.new()
for _, value in ipairs(wf.vars.items or {}) do
  table.insert(result, value * 2)
end
return result
"""

    assert analyze_lua_chunk(code).ok
    execution = execute_output(code, {"wf": {"vars": {"items": [1, 2, 3]}}})

    assert execution.ok is True
    assert execution.value == [2, 4, 6]


def test_pairs_and_ipairs_work_but_raw_next_is_not_part_of_runtime_contract():
    code = """
local total = 0
for _, value in ipairs(wf.vars.items) do
  total = total + value
end
for _, value in pairs(wf.vars.extra) do
  total = total + value
end
return total
"""

    assert analyze_lua_chunk(code).ok
    assert "lua_global_not_allowed::next" in _codes("return next(wf.vars.items)")
    execution = execute_output(
        code,
        {"wf": {"vars": {"items": [1, 2], "extra": {"value": 3}}}},
    )

    assert execution.ok is True
    assert execution.value == 6


def test_array_helper_copies_readonly_workflow_array_before_marking_it():
    code = "local result = _utils.array.new(wf.vars.items)\ntable.insert(result, 4)\nreturn result"

    execution = execute_output(code, {"wf": {"vars": {"items": [1, 2, 3]}}})

    assert execution.ok is True
    assert execution.value == [1, 2, 3, 4]


def test_runtime_timeout_is_a_stable_fail_closed_result(monkeypatch):
    def time_out(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], timeout=5)

    monkeypatch.setattr("app.validation.runtime_executor.subprocess.run", time_out)

    execution = execute_output("return true")

    assert execution.ok is False
    assert execution.error_code == "lua_runtime_timeout"
    assert execution.value is None


def test_non_utf8_lua_stdout_is_a_stable_fail_closed_result():
    execution = execute_output("return string.char(255)")

    assert execution.ok is False
    assert execution.error_code == "lua_runtime_invalid_utf8"
    assert execution.error_message == "Lua subprocess stdout is not valid UTF-8."


def test_non_utf8_lua_stderr_is_a_stable_fail_closed_result(monkeypatch):
    def invalid_stderr(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=1, stdout=b"", stderr=b"\xff")

    monkeypatch.setattr("app.validation.runtime_executor.subprocess.run", invalid_stderr)

    execution = execute_output("return true")

    assert execution.ok is False
    assert execution.error_code == "lua_runtime_invalid_utf8"
    assert execution.error_message == "Lua subprocess stderr is not valid UTF-8."


def test_mixed_numeric_and_string_result_keys_preserve_their_values():
    execution = execute_output('return {[2] = "two", label = "ok"}')

    assert execution.ok is True
    assert execution.value == {"2": "two", "label": "ok"}


def test_unsupported_result_key_shape_has_stable_runtime_diagnostic():
    execution = execute_output('return {[true] = "unsupported"}')

    assert execution.ok is False
    assert execution.error_code == "lua_result_serialization_error"
    assert "JSON object keys must be strings or numbers" in execution.error_message
