from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.middleware import RemoteBearerAuthMiddleware, RequestBodyLimitMiddleware
from app.api.routes import router
from app.core.config import RuntimeProfile, get_runtime_profile
from app.core.sessions import SessionStore
from app.core.storage import InvalidIdentifierError
from app.core.traces import TraceStore
from app.generation.engine import CompletionBackend, GenerationEngine
from app.generation.ollama import OllamaBackend


def create_app(
    profile: RuntimeProfile | None = None,
    trace_store: TraceStore | None = None,
    backend: CompletionBackend | None = None,
    session_store: SessionStore | None = None,
) -> FastAPI:
    runtime_profile = profile or get_runtime_profile()
    store = trace_store or TraceStore()
    sessions = session_store or SessionStore(root=store.root.parent / "sessions")
    model_backend = backend or OllamaBackend(runtime_profile)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            close = getattr(model_backend, "close", None)
            if callable(close):
                close()

    app = FastAPI(
        title="LocalScript API",
        version="0.3.0",
        description="Локальная генерация и fail-closed проверка LocalScript/Lua через Ollama.",
        lifespan=lifespan,
    )
    ui_static_dir = os.path.join(os.path.dirname(__file__), "ui", "static")
    ui_enabled = os.getenv("LOCALSCRIPT_UI_ENABLED", "0") != "0" and os.path.exists(ui_static_dir)
    app.state.profile = runtime_profile
    app.state.trace_store = store
    app.state.session_store = sessions
    app.state.ui_enabled = ui_enabled
    app.state.ui_index_path = os.path.join(ui_static_dir, "index.html")
    app.state.engine = GenerationEngine(
        profile=runtime_profile,
        trace_store=store,
        session_store=sessions,
        backend=model_backend,
    )
    app.add_middleware(
        RequestBodyLimitMiddleware,
        max_bytes=runtime_profile.max_request_body_bytes,
    )
    remote_mode = os.getenv("LOCALSCRIPT_REMOTE_MODE", "0") == "1"
    if remote_mode:
        remote_token = os.getenv("LOCALSCRIPT_REMOTE_TOKEN", "")
        if len(remote_token) < 32:
            raise RuntimeError("remote_token_missing_or_too_short")
        app.add_middleware(RemoteBearerAuthMiddleware, token=remote_token)

    @app.exception_handler(InvalidIdentifierError)
    async def invalid_identifier_handler(
        request: Request,
        exc: InvalidIdentifierError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={"detail": {"code": exc.code, "message": exc.message}},
        )

    if ui_enabled:
        app.mount("/static", StaticFiles(directory=ui_static_dir), name="static")

    app.include_router(router)
    return app


app = create_app()
