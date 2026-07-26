import pytest

from app.validation.base import BaseValidator
from app.validation.runtime_executor import execute_output
from app.validation.validators import (
    ContractValidator,
    JsonEnvelopeValidator,
    ValidationPipeline,
    _extract_lua_chunks,
)


@pytest.mark.parametrize(
    "code",
    [None, [], {}, 42, b"{}", "not-json", "[]", "null", "42", '"text"', "{}"],
)
def test_envelope_chunk_extraction_is_total(code):
    assert _extract_lua_chunks(code, "json_envelope") == []


@pytest.mark.parametrize(
    ("code", "error_code"),
    [
        (None, "contract_not_string"),
        ("not-json", "json_envelope_invalid"),
        ("[]", "json_envelope_not_object"),
        ("null", "json_envelope_not_object"),
        ("42", "json_envelope_not_object"),
        ('"text"', "json_envelope_not_object"),
        ("{}", "json_envelope_empty"),
        ('{"value": 1}', "json_envelope_value_not_string"),
        ('{"value": "return 1"}', "json_envelope_value_not_lua_wrapper"),
    ],
)
def test_execute_output_is_total_for_structurally_invalid_envelope(code, error_code):
    result = execute_output(code, output_style="json_envelope")

    assert result.ok is False
    assert result.error_code == error_code


def test_execute_output_reports_missing_plain_lua_chunk():
    result = execute_output("", output_style="lua_block")

    assert result.ok is False
    assert result.error_code == "lua_chunk_missing"


def test_execute_output_does_not_mask_programmer_errors(monkeypatch):
    def fail_with_programmer_error(chunk, context):
        raise AssertionError("unexpected executor defect")

    monkeypatch.setattr(
        "app.validation.runtime_executor._run_chunk",
        fail_with_programmer_error,
    )

    with pytest.raises(AssertionError, match="unexpected executor defect"):
        execute_output("return 1", output_style="lua_block")


class UnexpectedValidator(BaseValidator):
    name = "unexpected"

    def validate(self, code, context):
        raise AssertionError("later validation stage must not run")


@pytest.mark.parametrize(
    ("code", "validators", "error_code"),
    [
        (None, [ContractValidator(), UnexpectedValidator()], "contract_not_string"),
        (
            "[]",
            [ContractValidator(), JsonEnvelopeValidator(), UnexpectedValidator()],
            "json_envelope_not_object",
        ),
        (
            "not-json",
            [ContractValidator(), JsonEnvelopeValidator(), UnexpectedValidator()],
            "json_envelope_invalid",
        ),
        (
            "{}",
            [ContractValidator(), JsonEnvelopeValidator(), UnexpectedValidator()],
            "json_envelope_empty",
        ),
        (
            '{"value": 1}',
            [ContractValidator(), JsonEnvelopeValidator(), UnexpectedValidator()],
            "json_envelope_value_not_string",
        ),
        (
            '{"value": "return 1"}',
            [ContractValidator(), JsonEnvelopeValidator(), UnexpectedValidator()],
            "json_envelope_value_not_lua_wrapper",
        ),
    ],
)
def test_pipeline_stops_after_structural_error(code, validators, error_code):
    class TaskSpec:
        output_style = "json_envelope"

    report = ValidationPipeline(validators=validators).run(
        code=code,
        task_spec=TaskSpec(),
        profile=object(),
    )

    assert report.error_codes() == [error_code]
