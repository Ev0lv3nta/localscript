import json

import pytest

from app.generation.backend_errors import BackendProtocol
from app.workflow.contracts import (
    CheckStatus,
    CodeCandidate,
    ValidationCheck,
    ValidationResult,
)
from app.workflow.roles import CODE_ADAPTER, StructuredModelClient


class SequenceBackend:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, prompt, response_format=None):
        self.calls.append((prompt, response_format))
        return self.responses.pop(0)


def test_structured_model_passes_json_schema_and_parses_strictly():
    backend = SequenceBackend([json.dumps({"code": "return 1"})])

    candidate = StructuredModelClient(backend.complete).request("write code", CODE_ADAPTER)

    assert candidate == CodeCandidate(code="return 1")
    assert backend.calls[0][1]["type"] == "object"


def test_structured_model_allows_exactly_one_schema_correction():
    backend = SequenceBackend(["not json", json.dumps({"code": "return 2"})])

    candidate = StructuredModelClient(backend.complete).request("write code", CODE_ADAPTER)

    assert candidate.code == "return 2"
    assert len(backend.calls) == 2
    assert "did not satisfy the JSON schema" in backend.calls[1][0]


def test_structured_model_fails_closed_after_second_invalid_response():
    backend = SequenceBackend(["not json", "still not json"])

    with pytest.raises(BackendProtocol) as error:
        StructuredModelClient(backend.complete).request("write code", CODE_ADAPTER)

    assert error.value.reason == "structured_response_invalid"
    assert len(backend.calls) == 2


def test_validation_result_requires_every_check_to_pass():
    result = ValidationResult(
        checks=(
            ValidationCheck(name="ast", status=CheckStatus.PASSED),
            ValidationCheck(
                name="runtime",
                status=CheckStatus.FAILED,
                code="runtime_failed",
                message="The code failed.",
            ),
        )
    )

    assert result.ok is False
