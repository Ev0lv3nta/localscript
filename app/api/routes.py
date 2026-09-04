import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, Response, status
from fastapi.responses import FileResponse

from app.api.limits import APIConstraintError, validate_context, validate_prompt
from app.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ExampleEntry,
    ExamplesResponse,
    GenerateRequest,
    GenerateResponse,
    GenerateRichRequest,
    GenerateRichResponse,
    HealthResponse,
    ProfileResponse,
    ReadyResponse,
    SemanticResultSummary,
    SessionStateSummary,
    TraceResponse,
    ValidateRequest,
    ValidateResponse,
    ValidationSummary,
)
from app.core.storage import InvalidIdentifierError
from app.domain.outcomes import DiagnosticSeverity, GenerationStatus, ValidationStatus
from app.generation.backend_errors import (
    BackendError,
    BackendModel,
    BackendProtocol,
    BackendTimeout,
    BackendUnavailable,
)
from app.validation.runtime import find_lua_binary, find_luac_binary
from app.workflow.contracts import CheckStatus, CodeCandidate
from app.workflow.validation import DeterministicCandidateValidator

router = APIRouter()

LEGACY_SESSION_STATUS_MAP = {
    "clarification_needed": GenerationStatus.CLARIFICATION_REQUIRED.value,
    "degraded_completed": GenerationStatus.VALIDATION_FAILED.value,
    "failed_safe": GenerationStatus.POLICY_REJECTED.value,
}

UI_EXAMPLES = [
    ExampleEntry(
        id="model_unique_domains",
        title="Композиция обработки массива",
        mode="generate",
        prompt="Из массива wf.vars.subscribers собери новый массив уникальных доменов почты через _utils.array.new(). Бери только записи, где email не nil и содержит символ @. Домен нужно взять после @, привести к нижнему регистру и добавить в результат только один раз. Если subscribers равен nil или пустой, верни пустой массив.",
        context={"wf": {"vars": {"subscribers": [{"email": "A@Example.com"}, {"email": "b@example.com"}]}}},
        expected_strategy="ollama_chain",
        description="Нетривиальная задача обрабатывается модельной цепочкой.",
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
        title="Безопасное исправление",
        mode="generate",
        prompt="Use ctx.body.items and JsonPath for everything.",
        context={"wf": {"vars": {"items": [1, 2, 3]}}},
        expected_strategy="ollama_chain",
        description="Показывает переписывание unsupported roots и запрет небезопасных паттернов.",
    ),
    ExampleEntry(
        id="json_envelope_case",
        title="JSON-конверт",
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


def _generation_headers(result):
    return {
        "X-Trace-Id": result.trace_id,
        "X-Session-Id": result.session_id,
        "X-Strategy": result.strategy,
        "X-Repair-Rounds": str(result.repair_rounds),
        "X-Degraded-Mode": "true" if result.degraded_mode else "false",
        "X-Clarification-Suggested": (
            "true" if result.clarification_suggested else "false"
        ),
        "X-Assumption-Risk": result.assumption_risk,
    }


def _apply_generation_headers(response, result):
    response.headers.update(_generation_headers(result))


def _require_generation_outcome(result):
    if result.outcome is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "generation_outcome_missing",
                "message": "Generation engine returned no typed outcome.",
            },
        )
    return result.outcome


def _outcome_error_detail(outcome):
    if outcome.status is GenerationStatus.POLICY_REJECTED:
        return {
            "code": outcome.status.value,
            "message": "Generation request was rejected by policy.",
        }
    if outcome.status is GenerationStatus.CLARIFICATION_REQUIRED:
        return {
            "code": outcome.status.value,
            "message": outcome.question,
        }
    if outcome.status is GenerationStatus.BACKEND_UNAVAILABLE:
        return {
            "code": outcome.status.value,
            "message": "Generation backend is unavailable.",
        }
    findings = outcome.validation.findings + outcome.diagnostics
    message = next(
        (finding.message for finding in findings if finding.message),
        "Generation did not produce validated code.",
    )
    return {"code": outcome.status.value, "message": message}


def _outcome_error_codes(outcome):
    codes = []
    for finding in outcome.validation.findings:
        if (
            finding.severity is DiagnosticSeverity.ERROR
            or outcome.validation.status is ValidationStatus.INCOMPLETE
        ) and finding.code not in codes:
            codes.append(finding.code)
    return codes


def _outcome_messages(outcome):
    return [
        {
            "validator": finding.stage,
            "level": finding.severity.value,
            "code": finding.code,
            "message": finding.message,
        }
        for finding in outcome.validation.findings
    ]


def _raise_for_backend_outcome(outcome, result):
    if outcome.status is GenerationStatus.BACKEND_UNAVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=_outcome_error_detail(outcome),
            headers=_generation_headers(result),
        )


def _public_session_summary(summary, outcome=None, validation_report=None):
    summary = dict(summary)
    if outcome is not None:
        summary["status"] = outcome.status.value
    elif (validation_report or {}).get("has_errors"):
        summary["status"] = GenerationStatus.VALIDATION_FAILED.value
    else:
        summary["status"] = LEGACY_SESSION_STATUS_MAP.get(
            summary["status"],
            summary["status"],
        )
    return summary


def _handle_identifier_error(exc):
    raise HTTPException(status_code=400, detail={"code": exc.code, "message": exc.message})


def _handle_backend_error(exc):
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
def health(request: Request):
    profile = request.app.state.profile
    return HealthResponse(status="ok", profile=profile.name)


@router.get("/ready", response_model=ReadyResponse)
def ready(request: Request, response: Response):
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
    except BackendError as exc:
        _handle_backend_error(exc)
    outcome = _require_generation_outcome(result)
    _raise_for_backend_outcome(outcome, result)
    if outcome.status is not GenerationStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=_outcome_error_detail(outcome),
            headers=_generation_headers(result),
        )
    _apply_generation_headers(response, result)
    return GenerateResponse(code=outcome.code)


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
    except BackendError as exc:
        _handle_backend_error(exc)
    outcome = _require_generation_outcome(result)
    _raise_for_backend_outcome(outcome, result)
    _apply_generation_headers(response, result)
    session_summary = _public_session_summary(
        result.session_summary,
        outcome=outcome,
    )
    return GenerateRichResponse(
        status=outcome.status,
        session_id=result.session_id,
        trace_id=result.trace_id,
        strategy=result.strategy,
        question=outcome.question,
        assumptions=result.assumptions,
        code=outcome.code,
        validation=ValidationSummary(
            status=outcome.validation.status,
            ok=outcome.validation.ok,
            errors=_outcome_error_codes(outcome),
            degraded_mode=result.degraded_mode,
            repair_rounds=result.repair_rounds,
            messages=_outcome_messages(outcome),
        ),
        session=SessionStateSummary(**session_summary),
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
    return SessionStateSummary(
        **_public_session_summary(
            request.app.state.engine.build_session_summary(payload),
            validation_report=payload.get("previous_validation_report"),
        )
    )


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
    report = DeterministicCandidateValidator().validate_existing(
        candidate=CodeCandidate(code=payload.code),
        output=payload.output,
        context=payload.context,
    )
    failed = [check for check in report.checks if check.status is CheckStatus.FAILED]
    first_error = failed[0] if failed else None
    actual = None
    if report.observations:
        observation = report.observations[-1]
        if isinstance(observation, dict):
            actual = observation.get("actual")
    semantic_summary = SemanticResultSummary(
        ok=report.ok,
        value=actual,
        error_code=first_error.code if first_error else None,
        error_message=first_error.message if first_error else None,
        degraded=(first_error is not None and first_error.code == "sandbox_runtime_missing"),
    )
    verification_errors = [check.code for check in failed if check.code]
    validation_report = {
        "has_errors": bool(failed),
        "has_warnings": False,
        "messages": [
            {
                "validator": check.name,
                "level": "error" if check.status is CheckStatus.FAILED else "info",
                "code": check.code,
                "message": check.message,
            }
            for check in report.checks
        ],
    }
    return ValidateResponse(
        ok=report.ok,
        verification_errors=verification_errors,
        validation_report=validation_report,
        degraded_mode=semantic_summary.degraded,
        semantic_result=semantic_summary,
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
