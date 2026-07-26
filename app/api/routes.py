import json
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, Response, status
from fastapi.responses import FileResponse

from app.api.limits import APIConstraintError, validate_context, validate_prompt
from app.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ExamplesResponse,
    ExampleEntry,
    GenerateRequest,
    GenerateResponse,
    GenerateRichRequest,
    GenerateRichResponse,
    HealthResponse,
    ReadyResponse,
    ProfileResponse,
    SessionStateSummary,
    TraceResponse,
    ValidateRequest,
    ValidateResponse,
    SemanticResultSummary,
    ValidationSummary,
)
from app.core.storage import InvalidIdentifierError
from app.generation.engine import BackendUnavailableError
from app.core.verifier import verify_code
from app.generation.taskspec import TaskSpec
from app.validation.validators import _find_lua_binary, _find_luac_binary
from app.validation.runtime_executor import execute_output


router = APIRouter()

UI_EXAMPLES = [
    ExampleEntry(
        id="model_unique_domains",
        title="Новый hidden-style кейс",
        mode="generate",
        prompt="Из массива wf.vars.subscribers собери новый массив уникальных доменов почты через _utils.array.new(). Бери только записи, где email не nil и содержит символ @. Домен нужно взять после @, привести к нижнему регистру и добавить в результат только один раз. Если subscribers равен nil или пустой, верни пустой массив.",
        context={"wf": {"vars": {"subscribers": [{"email": "A@Example.com"}, {"email": "b@example.com"}]}}},
        expected_strategy="ollama_chain",
        description="Нетривиальная задача должна уйти в живую agent chain.",
    ),
    ExampleEntry(
        id="clarification_root",
        title="Сценарий уточнения",
        mode="generate",
        prompt="Нормализуй email и верни его в lower-case.",
        context={"wf": {"vars": {"email": "A@EXAMPLE.COM"}, "initVariables": {"email": "B@EXAMPLE.COM"}}},
        expected_strategy="clarification",
        description="Неоднозначный root должен запросить одно уточнение.",
    ),
    ExampleEntry(
        id="repair_ctx_body",
        title="Repair и safety",
        mode="generate",
        prompt="Use ctx.body.items and JsonPath for everything.",
        context={"wf": {"vars": {"items": [1, 2, 3]}}},
        expected_strategy="ollama_chain",
        description="Показывает переписывание unsupported roots и запрет небезопасных паттернов.",
    ),
    ExampleEntry(
        id="json_envelope_case",
        title="JSON envelope",
        mode="generate",
        prompt="Дополни существующий код: добавь переменную squared как квадрат числа num и верни результат как JSON envelope.",
        context={"wf": {"vars": {"num": 5}}},
        expected_strategy="ollama_chain",
        description="Пример генерации JSON envelope с вложенными Lua-блоками.",
    ),
]


def _raise_constraint_error(exc):
    raise HTTPException(
        status_code=exc.status_code,
        detail={"code": exc.code, "message": exc.message},
    )


def _validate_payload_limits(request, prompt, context):
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


def _apply_generation_headers(response, result):
    response.headers["X-Trace-Id"] = result.trace_id
    response.headers["X-Session-Id"] = result.session_id
    response.headers["X-Strategy"] = result.strategy
    response.headers["X-Repair-Rounds"] = str(result.repair_rounds)
    response.headers["X-Degraded-Mode"] = "true" if result.degraded_mode else "false"
    response.headers["X-Clarification-Suggested"] = "true" if result.clarification_suggested else "false"
    response.headers["X-Assumption-Risk"] = result.assumption_risk


def _handle_identifier_error(exc):
    raise HTTPException(status_code=400, detail={"code": exc.code, "message": exc.message})


def _detect_output_style(code, explicit_style):
    if explicit_style in {"lua_block", "json_envelope"}:
        return explicit_style
    stripped = (code or "").strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return "lua_block"
        if isinstance(payload, dict) and payload:
            if all(
                isinstance(value, str)
                and value.strip().startswith("lua{")
                and value.strip().endswith("}lua")
                for value in payload.values()
            ):
                return "json_envelope"
    return "lua_block"


@router.get("/health", response_model=HealthResponse)
def health(request: Request):
    profile = request.app.state.profile
    return HealthResponse(status="ok", profile=profile.name)


@router.get("/ready", response_model=ReadyResponse)
def ready(request: Request, response: Response):
    profile = request.app.state.profile
    backend = request.app.state.engine.backend
    tags = backend.list_tags() if backend.ping() else []
    checks = {
        "backend_reachable": backend.ping(),
        "primary_model_present": profile.model in tags,
        "fallback_model_present": profile.fallback_model in tags,
        "lua_runtime_present": bool(_find_lua_binary()),
        "luac_runtime_present": bool(_find_luac_binary()),
    }
    errors = [name for name, ok in checks.items() if not ok]
    if errors:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadyResponse(status="not_ready", profile=profile.name, checks=checks, errors=errors)
    return ReadyResponse(status="ready", profile=profile.name, checks=checks, errors=[])


@router.post("/generate", response_model=GenerateResponse)
def generate(payload: GenerateRequest, request: Request, response: Response):
    _validate_payload_limits(request, payload.prompt, payload.context)
    try:
        result = request.app.state.engine.generate(
            prompt=payload.prompt,
            context=payload.context,
            session_id=payload.session_id,
            feedback=payload.feedback,
        )
    except InvalidIdentifierError as exc:
        _handle_identifier_error(exc)
    except BackendUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "backend_unavailable", "message": str(exc)},
        )
    _apply_generation_headers(response, result)
    return GenerateResponse(code=result.code)


@router.post("/api/generate", response_model=GenerateRichResponse)
def generate_rich(payload: GenerateRichRequest, request: Request, response: Response):
    if not payload.prompt and not payload.session_id:
        raise HTTPException(status_code=400, detail="prompt or session_id is required")
    _validate_payload_limits(request, payload.prompt, payload.context)

    try:
        result = request.app.state.engine.generate_rich(
            prompt=payload.prompt,
            context=payload.context,
            session_id=payload.session_id,
            feedback=payload.feedback,
            clarification_answer=payload.clarification_answer,
        )
    except InvalidIdentifierError as exc:
        _handle_identifier_error(exc)
    except BackendUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "backend_unavailable", "message": str(exc)},
        )
    _apply_generation_headers(response, result)
    return GenerateRichResponse(
        status=result.status,
        session_id=result.session_id,
        trace_id=result.trace_id,
        strategy=result.strategy,
        question=result.question or None,
        assumptions=result.assumptions,
        code=result.code or None,
        validation=ValidationSummary(
            ok=not result.validation_report.get("has_errors", False),
            errors=result.verification_errors,
            degraded_mode=result.degraded_mode,
            repair_rounds=result.repair_rounds,
            messages=result.validation_report.get("messages", []),
        ),
        session=SessionStateSummary(**result.session_summary),
    )


@router.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest, request: Request):
    _validate_payload_limits(request, payload.prompt, payload.context)
    return AnalyzeResponse(
        **request.app.state.engine.analyze(
            prompt=payload.prompt,
            context=payload.context,
        )
    )


@router.get("/api/sessions/{session_id}", response_model=SessionStateSummary)
def get_session(session_id: str, request: Request):
    try:
        payload = request.app.state.session_store.read(session_id)
    except InvalidIdentifierError as exc:
        _handle_identifier_error(exc)
    if payload is None:
        raise HTTPException(status_code=404, detail="session_not_found")
    return SessionStateSummary(**request.app.state.engine.build_session_summary(payload))


@router.get("/api/traces/{trace_id}", response_model=TraceResponse)
def get_trace(trace_id: str, request: Request):
    try:
        payload = request.app.state.trace_store.read(trace_id)
    except InvalidIdentifierError as exc:
        _handle_identifier_error(exc)
    if payload is None:
        raise HTTPException(status_code=404, detail="trace_not_found")
    return TraceResponse(**request.app.state.trace_store.sanitize_trace(payload))


@router.get("/api/profile", response_model=ProfileResponse)
def get_profile(request: Request):
    profile = request.app.state.profile
    return ProfileResponse(
        profile=profile.name,
        model=profile.model,
        fallback_model=profile.fallback_model,
        num_ctx=profile.num_ctx,
        num_predict=profile.num_predict,
        max_repair_rounds=profile.max_repair_rounds,
        ui_enabled=bool(request.app.state.ui_enabled),
    )


@router.get("/api/examples", response_model=ExamplesResponse)
def get_examples():
    return ExamplesResponse(examples=UI_EXAMPLES)


@router.post("/api/validate", response_model=ValidateResponse)
def validate_code(payload: ValidateRequest, request: Request):
    _validate_payload_limits(request, "validate existing code", payload.context)
    output_style = _detect_output_style(payload.code, payload.output_style)
    task_spec = TaskSpec(
        normalized_prompt="validate existing code",
        output_style=output_style,
        target_root="unknown",
        context_paths=request.app.state.engine.extractor._collect_context_paths(payload.context),
    )
    report = request.app.state.engine.validation_pipeline.run(
        code=payload.code,
        task_spec=task_spec,
        profile=request.app.state.profile,
        source_context=payload.context,
        prompt="validate existing code",
        planner_semantic_checks=None,
    )
    verification_errors = []
    for error_code in report.error_codes() + verify_code(payload.code):
        if error_code not in verification_errors:
            verification_errors.append(error_code)
    semantic_result = execute_output(
        code=payload.code,
        context=payload.context,
        output_style=output_style,
    )
    return ValidateResponse(
        ok=not verification_errors and not report.has_errors,
        verification_errors=verification_errors,
        validation_report=report.to_dict(),
        degraded_mode=request.app.state.engine._is_degraded(strategy="validate", validation_report=report),
        semantic_result=SemanticResultSummary(
            ok=semantic_result.ok,
            value=semantic_result.value,
            error_code=semantic_result.error_code or None,
            error_message=semantic_result.error_message or None,
            degraded=semantic_result.degraded,
        ),
    )


@router.get("/")
def get_ui_root(request: Request):
    if not request.app.state.ui_enabled:
        raise HTTPException(status_code=404, detail="ui_disabled")
    return FileResponse(Path(request.app.state.ui_index_path))


@router.get("/ui")
def get_ui_alias(request: Request):
    if not request.app.state.ui_enabled:
        raise HTTPException(status_code=404, detail="ui_disabled")
    return FileResponse(Path(request.app.state.ui_index_path))
