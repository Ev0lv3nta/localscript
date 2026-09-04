from __future__ import annotations

import os
from pathlib import Path
from typing import NoReturn

from fastapi import APIRouter, HTTPException, Request, Response, status
from fastapi.responses import FileResponse

from app.api.limits import APIConstraintError, validate_context, validate_prompt
from app.api.schemas import (
    ExampleEntry,
    ExamplesResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ProfileResponse,
    ReadyResponse,
    TraceResponse,
    ValidateRequest,
    ValidateResponse,
)
from app.core.storage import InvalidIdentifierError
from app.generation.backend_errors import (
    BackendError,
    BackendModel,
    BackendProtocol,
    BackendTimeout,
    BackendUnavailable,
)
from app.generation.results import GenerationResult, SessionSummary
from app.validation.runtime import find_lua_binary, find_luac_binary
from app.workflow.contracts import CodeCandidate, JsonValue, WorkflowStatus
from app.workflow.validation import DeterministicCandidateValidator

router = APIRouter()

UI_EXAMPLES = [
    ExampleEntry(
        id="unique_domains",
        title="Композиция обработки массива",
        prompt=(
            "Из массива wf.vars.subscribers собери новый массив уникальных доменов почты. "
            "Бери только записи, где email не nil и содержит символ @. Домен нужно взять "
            "после @, привести к нижнему регистру и добавить в результат только один раз. "
            "Если subscribers равен nil или пустой, верни пустой массив."
        ),
        context={
            "wf": {"vars": {"subscribers": [{"email": "A@Example.com"}, {"email": "b@example.com"}]}}
        },
        description="Задача, которую нельзя решить одним выражением: фильтрация, преобразование и дедупликация.",
    ),
    ExampleEntry(
        id="clarification_root",
        title="Сценарий уточнения",
        prompt="Нормализуй email и верни его в lower-case.",
        context={
            "wf": {
                "vars": {"email": "A@EXAMPLE.COM"},
                "initVariables": {"email": "B@EXAMPLE.COM"},
            }
        },
        description="Источник неоднозначен, поэтому planner должен задать один вопрос вместо догадки.",
    ),
    ExampleEntry(
        id="json_envelope",
        title="JSON-конверт",
        prompt=(
            "Добавь переменную squared как квадрат числа wf.vars.num и верни результат "
            "как JSON envelope."
        ),
        context={"wf": {"vars": {"num": 5}}},
        description="Проверяет второй поддерживаемый формат вывода.",
    ),
]


def _raise_constraint_error(exc: APIConstraintError) -> NoReturn:
    raise HTTPException(
        status_code=exc.status_code,
        detail={"code": exc.code, "message": exc.message},
    )


def _validate_payload_limits(request: Request, prompt: str | None, context: JsonValue) -> None:
    profile = request.app.state.profile
    max_prompt_chars = int(os.getenv("LOCALSCRIPT_MAX_PROMPT_CHARS", profile.max_prompt_chars))
    max_context_bytes = int(os.getenv("LOCALSCRIPT_MAX_CONTEXT_BYTES", profile.max_context_bytes))
    max_context_depth = int(os.getenv("LOCALSCRIPT_MAX_CONTEXT_DEPTH", profile.max_context_depth))
    max_context_nodes = int(os.getenv("LOCALSCRIPT_MAX_CONTEXT_NODES", profile.max_context_nodes))
    try:
        validate_prompt(prompt, max_prompt_chars)
        validate_context(
            context,
            max_bytes=max_context_bytes,
            max_depth=max_context_depth,
            max_nodes=max_context_nodes,
        )
    except APIConstraintError as exc:
        _raise_constraint_error(exc)


def _generation_headers(result: GenerationResult) -> dict[str, str]:
    return {"X-Trace-Id": result.trace_id, "X-Session-Id": result.session_id}


def _handle_identifier_error(exc: InvalidIdentifierError) -> NoReturn:
    raise HTTPException(status_code=400, detail={"code": exc.code, "message": exc.message})


def _handle_backend_error(exc: BackendError) -> NoReturn:
    if isinstance(exc, BackendTimeout):
        status_code = status.HTTP_504_GATEWAY_TIMEOUT
    elif isinstance(exc, BackendUnavailable):
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    elif isinstance(exc, (BackendProtocol, BackendModel)):
        status_code = status.HTTP_502_BAD_GATEWAY
    else:
        status_code = status.HTTP_502_BAD_GATEWAY
    raise HTTPException(
        status_code=status_code,
        detail={"code": exc.code, "message": exc.public_message},
    )


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    profile = request.app.state.profile
    return HealthResponse(status="ok", profile=profile.name)


@router.get("/ready", response_model=ReadyResponse)
def ready(request: Request, response: Response) -> ReadyResponse:
    profile = request.app.state.profile
    backend = request.app.state.engine.backend
    try:
        if hasattr(backend, "list_tag_details"):
            tag_details = backend.list_tag_details()
            tags = [item["name"] for item in tag_details]
            backend_reachable = True
        else:
            backend_reachable = backend.ping()
            tags = backend.list_tags() if backend_reachable else []
    except BackendError:
        tags = []
        backend_reachable = False
    checks = {
        "backend_reachable": backend_reachable,
        "primary_model_present": profile.model in tags,
        "fallback_model_present": profile.fallback_model in tags,
        "lua_runtime_present": bool(find_lua_binary()),
        "luac_runtime_present": bool(find_luac_binary()),
    }
    errors = [name for name, ok in checks.items() if not ok]
    if errors:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadyResponse(status="not_ready", profile=profile.name, checks=checks, errors=errors)
    return ReadyResponse(status="ready", profile=profile.name, checks=checks, errors=[])


@router.post("/api/generate", response_model=GenerateResponse)
def generate(
    payload: GenerateRequest,
    request: Request,
    response: Response,
) -> GenerateResponse:
    if not payload.prompt and not payload.session_id:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "prompt_or_session_required",
                "message": "Provide a prompt for a new session or a session_id to continue one.",
            },
        )
    _validate_payload_limits(request, payload.prompt, payload.context)

    try:
        result = request.app.state.engine.generate(
            prompt=payload.prompt,
            context=payload.context,
            session_id=payload.session_id,
            feedback=payload.feedback,
            clarification_answer=payload.clarification_answer,
        )
    except InvalidIdentifierError as exc:
        _handle_identifier_error(exc)
    except BackendError as exc:
        _handle_backend_error(exc)

    workflow = result.workflow
    if workflow.status is WorkflowStatus.BACKEND_UNAVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": workflow.status.value,
                "message": "The local model backend is unavailable.",
            },
            headers=_generation_headers(result),
        )
    response.headers.update(_generation_headers(result))
    return GenerateResponse(
        status=workflow.status,
        session_id=result.session_id,
        trace_id=result.trace_id,
        code=workflow.code,
        question=workflow.question,
        diagnostics=workflow.diagnostics,
        validation=workflow.validation,
        revision_count=workflow.revision_count,
    )


@router.get("/api/sessions/{session_id}", response_model=SessionSummary)
def get_session(session_id: str, request: Request) -> SessionSummary:
    try:
        payload = request.app.state.session_store.read(session_id)
    except InvalidIdentifierError as exc:
        _handle_identifier_error(exc)
    if payload is None:
        raise HTTPException(status_code=404, detail="session_not_found")
    summary: SessionSummary = request.app.state.engine.build_session_summary(payload)
    return summary


@router.get("/api/traces/{trace_id}", response_model=TraceResponse)
def get_trace(trace_id: str, request: Request) -> TraceResponse:
    try:
        payload = request.app.state.trace_store.read(trace_id)
    except InvalidIdentifierError as exc:
        _handle_identifier_error(exc)
    if payload is None:
        raise HTTPException(status_code=404, detail="trace_not_found")
    return TraceResponse(**request.app.state.trace_store.sanitize_trace(payload))


@router.get("/api/profile", response_model=ProfileResponse)
def get_profile(request: Request) -> ProfileResponse:
    profile = request.app.state.profile
    return ProfileResponse(
        profile=profile.name,
        model=profile.model,
        fallback_model=profile.fallback_model,
        num_ctx=profile.num_ctx,
        num_predict=profile.num_predict,
        ui_enabled=bool(request.app.state.ui_enabled),
    )


@router.get("/api/examples", response_model=ExamplesResponse)
def get_examples() -> ExamplesResponse:
    return ExamplesResponse(examples=UI_EXAMPLES)


@router.post("/api/validate", response_model=ValidateResponse)
def validate_code(payload: ValidateRequest, request: Request) -> ValidateResponse:
    _validate_payload_limits(request, "validate existing code", payload.context)
    report = DeterministicCandidateValidator().validate_existing(
        candidate=CodeCandidate(code=payload.code),
        output=payload.output,
        context=payload.context,
    )
    return ValidateResponse(ok=report.ok, validation=report)


@router.get("/")
def get_ui_root(request: Request) -> FileResponse:
    if not request.app.state.ui_enabled:
        raise HTTPException(status_code=404, detail="ui_disabled")
    return FileResponse(Path(request.app.state.ui_index_path))


@router.get("/ui")
def get_ui_alias(request: Request) -> FileResponse:
    if not request.app.state.ui_enabled:
        raise HTTPException(status_code=404, detail="ui_disabled")
    return FileResponse(Path(request.app.state.ui_index_path))
