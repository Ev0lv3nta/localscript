import json

import pytest
from pydantic import ValidationError

from app.generation.backend_errors import BackendProtocol
from app.workflow.contracts import (
    CheckStatus,
    CodeCandidate,
    ValidationCheck,
    ValidationResult,
)
from app.workflow.roles import (
    CODE_RESPONSE,
    PLANNING_RESPONSE,
    REVIEW_RESPONSE,
    StructuredModelClient,
)


class SequenceBackend:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, prompt, response_format=None):
        self.calls.append((prompt, response_format))
        return self.responses.pop(0)


def test_structured_model_passes_json_schema_and_parses_strictly():
    backend = SequenceBackend([json.dumps({"code": "return 1"})])

    candidate = StructuredModelClient(backend.complete).request("write code", CODE_RESPONSE)

    assert candidate == CodeCandidate(code="return 1")
    assert backend.calls[0][1]["type"] == "object"


def test_structured_model_allows_exactly_one_schema_correction():
    backend = SequenceBackend(["not json", json.dumps({"code": "return 2"})])

    candidate = StructuredModelClient(backend.complete).request("write code", CODE_RESPONSE)

    assert candidate.code == "return 2"
    assert len(backend.calls) == 2
    assert "did not satisfy the JSON schema" in backend.calls[1][0]


def test_structured_model_fails_closed_after_second_invalid_response():
    backend = SequenceBackend(["not json", "still not json"])

    with pytest.raises(BackendProtocol) as error:
        StructuredModelClient(backend.complete).request("write code", CODE_RESPONSE)

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


def test_model_facing_schema_requires_the_union_discriminator():
    """Модель пропускает поле, которое схема не требует, и union перестаёт разрешаться.

    Pydantic помечает дискриминатор со значением по умолчанию как необязательный, поэтому схему
    для модели приходится чинить явно: без этого каждый ответ planner'а приходил без `kind` и
    отвергался как невалидный, а сообщение об ошибке ни на что не указывало.
    """
    for response in (PLANNING_RESPONSE, REVIEW_RESPONSE):
        definitions = response.schema["$defs"]
        tagged = [
            definition
            for definition in definitions.values()
            if "kind" in (definition.get("properties") or {})
        ]
        assert tagged
        for definition in tagged:
            assert "kind" in definition["required"]


def test_model_facing_schema_drops_unbuildable_string_bounds():
    """Верхняя граница длины строки превращается в грамматику по всему словарю.

    Для поля кода с лимитом в 131072 символа Ollama просто не собирает грамматику и отвечает
    HTTP 500. Ограничение остаётся в контракте Pydantic и проверяется после генерации.
    """
    assert "maxLength" not in json.dumps(CODE_RESPONSE.schema)
    assert "maxLength" not in json.dumps(PLANNING_RESPONSE.schema)
    assert CODE_RESPONSE.schema["properties"]["code"]["minLength"] == 1
    with pytest.raises(ValidationError):
        CODE_RESPONSE.adapter.validate_python({"code": "x" * 200_000})


def test_model_facing_schema_gives_json_values_a_real_grammar():
    """Пустая схема `{}` не ограничивает грамматику, и модель заполняла её объектом.

    Из-за этого скалярный ожидаемый результат приезжал завёрнутым в объект, и план противоречил
    собственному output-контракту. Явное перечисление вариантов со скалярами впереди это снимает.
    """
    json_value = PLANNING_RESPONSE.schema["$defs"]["JsonValue"]

    assert json_value != {}
    assert [variant.get("type") for variant in json_value["anyOf"][:4]] == [
        "string",
        "number",
        "boolean",
        "null",
    ]


def test_domain_specification_names_the_default_output_format():
    """Спецификация перечисляла оба формата, не говоря, когда какой.

    Планировщик из-за этого выбирал json_envelope для обычной задачи с одним результатом, и
    кандидат отвергался как невалидный конверт ещё до выполнения.
    """
    from app.workflow.roles import DOMAIN_SPECIFICATION

    assert "it is the default" in DOMAIN_SPECIFICATION
    assert "json_envelope" in DOMAIN_SPECIFICATION
    assert "lua_block" in DOMAIN_SPECIFICATION
