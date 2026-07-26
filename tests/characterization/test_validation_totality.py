import pytest

from app.validation.runtime_executor import execute_output
from app.validation.validators import _extract_lua_chunks


@pytest.mark.xfail(
    strict=True,
    raises=AttributeError,
    reason="JSON envelope extraction is not total for non-object JSON values",
)
@pytest.mark.parametrize("code", ["[]", "null", "42", '"text"'])
def test_envelope_chunk_extraction_is_total(code):
    assert _extract_lua_chunks(code, "json_envelope") == []


@pytest.mark.xfail(
    strict=True,
    raises=IndexError,
    reason="runtime execution assumes every envelope value is a Lua wrapper",
)
def test_execute_output_is_total_for_structurally_invalid_envelope():
    result = execute_output('{"value": 1}', output_style="json_envelope")

    assert result.ok is False
    assert result.error_code == "json_envelope_value_not_string"
