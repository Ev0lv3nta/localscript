import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import RuntimeProfile, get_runtime_profile
from app.core.storage import InvalidIdentifierError
from app.core.sessions import SessionStore
from app.core.traces import TraceStore
from app.generation.engine import GenerationEngine
from app.generation.ollama import OllamaBackend


def create_app(profile=None, trace_store=None, backend=None, session_store=None):
    runtime_profile = profile or get_runtime_profile()
    store = trace_store or TraceStore()
    sessions = session_store or SessionStore(root=store.root.parent / "sessions")
    model_backend = backend or OllamaBackend(runtime_profile)

    app = FastAPI(
        title="LocalScript API",
        version="0.1.0",
        description="Judged-safe local generator for LocalScript/Lua.",
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

    @app.exception_handler(InvalidIdentifierError)
    async def invalid_identifier_handler(request: Request, exc: InvalidIdentifierError):
        return JSONResponse(
            status_code=400,
            content={"detail": {"code": exc.code, "message": exc.message}},
        )

    @app.middleware("http")
    async def request_body_limit_middleware(request: Request, call_next):
        if request.method in {"POST", "PUT", "PATCH"}:
            body = await request.body()
            max_request_body_bytes = int(
                os.getenv("LOCALSCRIPT_MAX_REQUEST_BODY_BYTES", runtime_profile.max_request_body_bytes)
            )
            if len(body) > max_request_body_bytes:
                return JSONResponse(
                    status_code=413,
                    content={
                        "detail": {
                            "code": "request_too_large",
                            "message": "request body exceeds maximum allowed size",
                        }
                    },
                )

            async def receive():
                return {"type": "http.request", "body": body, "more_body": False}

            request = Request(request.scope, receive)
        return await call_next(request)

    if ui_enabled:
        app.mount("/static", StaticFiles(directory=ui_static_dir), name="static")

    app.include_router(router)
    return app


app = create_app()
