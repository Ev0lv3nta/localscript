from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NoReturn

import typer
from pydantic import ValidationError

from app.core.benchmarks import (
    quality_gate_failures,
    run_dataset_benchmark,
    run_quality_benchmark,
)
from app.core.config import RuntimeProfile, get_runtime_profile
from app.core.resources import resource_exists
from app.core.traces import TraceStore
from app.generation.backend_errors import BackendError
from app.generation.engine import GenerationEngine, SessionStateError
from app.generation.ollama import OllamaBackend
from app.generation.results import GenerationResult
from app.validation.runtime import find_lua_binary, find_luac_binary
from app.workflow.contracts import (
    CheckStatus,
    CodeCandidate,
    OutputContract,
    OutputFormat,
    OutputShape,
    WorkflowStatus,
)
from app.workflow.validation import DeterministicCandidateValidator

cli = typer.Typer(help="LocalScript local CLI")


def _emit_error(code: str, message: str, *, exit_code: int = 1) -> NoReturn:
    typer.echo(
        json.dumps(
            {"ok": False, "error": {"code": code, "message": message}},
            ensure_ascii=False,
        )
    )
    raise typer.Exit(code=exit_code)


def _read_text_file(path: str, *, label: str) -> str:
    if path == "-":
        return typer.get_text_stream("stdin").read()
    try:
        return Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        _emit_error(
            f"{label}_file_unreadable",
            f"The {label} file could not be read as UTF-8 text.",
            exit_code=2,
        )


def _resolve_text_input(
    inline: str | None,
    file_path: str | None,
    *,
    label: str,
) -> str | None:
    if inline is not None and file_path is not None:
        _emit_error(
            f"{label}_source_conflict",
            f"Use either --{label} or --{label}-file, not both.",
            exit_code=2,
        )
    return _read_text_file(file_path, label=label) if file_path is not None else inline


def _close_backend(owner: object) -> None:
    backend = getattr(owner, "backend", None)
    close = getattr(backend, "close", None)
    if callable(close):
        close()


def _generation_payload(result: GenerationResult) -> dict[str, Any]:
    workflow = result.workflow
    payload: dict[str, Any] = {
        "status": workflow.status.value,
        "session_id": result.session_id,
        "trace_id": result.trace_id,
        "diagnostics": [diagnostic.model_dump(mode="json") for diagnostic in workflow.diagnostics],
        "revision_count": workflow.revision_count,
    }
    if workflow.code is not None:
        payload["code"] = workflow.code
    if workflow.question is not None:
        payload["question"] = workflow.question
    if workflow.validation is not None:
        payload["validation"] = workflow.validation.model_dump(mode="json")
    if workflow.output is not None:
        payload["output"] = workflow.output.model_dump(mode="json")
    return payload


def build_engine() -> GenerationEngine:
    profile = get_runtime_profile()
    backend = OllamaBackend(profile)
    try:
        return GenerationEngine(
            profile=profile,
            trace_store=TraceStore(),
            backend=backend,
        )
    except Exception:
        backend.close()
        raise


@cli.command()
def generate(
    prompt: str | None = typer.Option(None, help="Natural-language generation request."),
    prompt_file: str | None = typer.Option(
        None,
        help="Read the generation request from a UTF-8 file; use '-' for stdin.",
    ),
    context: str | None = typer.Option(None, help="Workflow context as JSON."),
    context_file: str | None = typer.Option(
        None,
        help="Read workflow context JSON from a UTF-8 file; use '-' for stdin.",
    ),
    session_id: str | None = typer.Option(None, help="Existing session id to continue."),
    answer: str | None = typer.Option(None, help="Answer to the open clarification question."),
    feedback: str | None = typer.Option(None, help="Feedback for revising the previous result."),
) -> None:
    """Generate LocalScript code, or continue an existing session."""
    if prompt_file == "-" and context_file == "-":
        _emit_error(
            "stdin_source_conflict",
            "Standard input can provide either the prompt or the context, not both.",
            exit_code=2,
        )
    resolved_prompt = _resolve_text_input(prompt, prompt_file, label="prompt")
    if not resolved_prompt and not session_id:
        _emit_error(
            "prompt_or_session_required",
            "Provide a prompt for a new session or a session_id to continue one.",
            exit_code=2,
        )
    raw_context = _resolve_text_input(context, context_file, label="context")
    context_was_provided = context is not None or context_file is not None
    parsed_context: object = None
    if context_was_provided:
        try:
            parsed_context = json.loads(raw_context or "")
        except json.JSONDecodeError:
            _emit_error(
                "invalid_json_context",
                "Context must be valid JSON.",
                exit_code=2,
            )

    try:
        engine = build_engine()
    except BackendError as error:
        _emit_error(error.code, error.public_message)
    try:
        if context_was_provided:
            result = engine.generate(
                prompt=resolved_prompt,
                context=parsed_context,
                session_id=session_id,
                clarification_answer=answer,
                feedback=feedback,
            )
        else:
            result = engine.generate(
                prompt=resolved_prompt,
                session_id=session_id,
                clarification_answer=answer,
                feedback=feedback,
            )
    except SessionStateError as error:
        _emit_error(error.code, error.message)
    except BackendError as error:
        _emit_error(error.code, error.public_message)
    finally:
        _close_backend(engine)

    typer.echo(json.dumps(_generation_payload(result), ensure_ascii=False))
    raise typer.Exit(code=0 if result.workflow.status is WorkflowStatus.COMPLETED else 1)


@cli.command()
def validate(
    code: str | None = typer.Option(None, help="Inline LocalScript/Lua code."),
    code_file: str | None = typer.Option(None, help="Path to a file with LocalScript/Lua code."),
    context: str = typer.Option('{"wf":{"vars":{}}}', help="Workflow context as JSON."),
    output_format: str = typer.Option("lua_block", help="lua_block or json_envelope."),
    output_shape: str = typer.Option("scalar", help="scalar, array, or object."),
    nullable: bool = typer.Option(False, help="Allow a null result."),
) -> None:
    content = _resolve_text_input(code, code_file, label="code")
    if content is None:
        _emit_error(
            "code_required",
            "Provide --code or --code-file.",
            exit_code=2,
        )

    try:
        parsed_context = json.loads(context)
    except json.JSONDecodeError:
        _emit_error(
            "invalid_json_context",
            "Context must be valid JSON.",
            exit_code=2,
        )
    if not isinstance(parsed_context, dict):
        _emit_error(
            "invalid_json_context",
            "Context must contain a JSON object.",
            exit_code=2,
        )
    try:
        resolved_output_format = OutputFormat(output_format)
        resolved_output_shape = OutputShape(output_shape)
    except ValueError:
        _emit_error(
            "invalid_output_contract",
            "The output format or shape is not supported.",
            exit_code=2,
        )
    try:
        output = OutputContract(
            format=resolved_output_format,
            shape=resolved_output_shape,
            nullable=nullable,
        )
    except ValidationError:
        _emit_error(
            "invalid_output_contract",
            "The output format, shape, and nullable options are inconsistent.",
            exit_code=2,
        )

    try:
        candidate = CodeCandidate(code=content)
    except ValidationError:
        rendered = content.strip()
        if not rendered:
            error_code = "code_empty"
            message = "Code must not be empty."
        elif rendered.startswith("```") or rendered.endswith("```"):
            error_code = "code_markdown_fence_forbidden"
            message = "Code must not contain Markdown fences."
        else:
            error_code = "invalid_code"
            message = "Code does not satisfy the supported validation input contract."
        _emit_error(error_code, message, exit_code=2)

    report = DeterministicCandidateValidator().validate_existing(
        candidate=candidate,
        output=output,
        context=parsed_context,
    )
    failed = [check for check in report.checks if check.status is CheckStatus.FAILED]
    errors = [check.code for check in failed if check.code]
    payload = {
        "ok": report.ok,
        "errors": errors,
        "validation": report.model_dump(mode="json"),
    }
    typer.echo(json.dumps(payload, ensure_ascii=False))
    raise typer.Exit(code=0 if report.ok else 1)


@cli.command()
def benchmark(
    dataset: str = typer.Option("evals/public/v2.jsonl", help="JSONL dataset path."),
) -> None:
    dataset_path = Path(dataset)
    packaged_dataset = dataset.replace("\\", "/")
    if not dataset_path.is_file() and not (
        packaged_dataset.startswith("evals/") and resource_exists(packaged_dataset)
    ):
        _emit_error(
            "dataset_not_found",
            "The requested benchmark dataset was not found.",
        )

    profile = get_runtime_profile()
    try:
        backend = OllamaBackend(profile)
    except BackendError as error:
        _emit_error(error.code, error.public_message)
    try:
        report = run_dataset_benchmark(dataset_path, profile=profile, backend=backend)
    except BackendError as error:
        _emit_error(error.code, error.public_message)
    finally:
        backend.close()
    typer.echo(json.dumps(report, ensure_ascii=False))
    raise typer.Exit(code=0 if report["ok"] else 1)


def _doctor_report(
    profile: RuntimeProfile,
    backend: OllamaBackend,
    evaluation: bool,
) -> dict[str, Any]:
    trace_dir_writable = TraceStore().root.exists()
    lua_runtime_present = bool(find_lua_binary())
    luac_runtime_present = bool(find_luac_binary())
    try:
        available_tags = backend.list_tags()
        ollama_reachable = True
    except BackendError:
        available_tags = []
        ollama_reachable = False
    model_present = profile.model in available_tags
    report = {
        "profile": profile.name,
        "model": profile.model,
        "ollama_host": profile.ollama_host,
        "num_ctx": profile.num_ctx,
        "num_predict": profile.num_predict,
        "batch": profile.batch,
        "parallel": profile.parallel,
        "request_timeout_seconds": profile.request_timeout_seconds,
        "trace_dir_writable": trace_dir_writable,
        "ollama_reachable": ollama_reachable,
        "model_present": model_present,
        "lua_runtime_present": lua_runtime_present,
        "luac_runtime_present": luac_runtime_present,
        "eval_mode": evaluation,
        "judge_mode": evaluation,
        "ok": all(
            (
                trace_dir_writable,
                ollama_reachable,
                model_present,
                lua_runtime_present,
                luac_runtime_present,
            )
        ),
    }
    if evaluation:
        quality_report = (
            run_quality_benchmark(profile=profile, backend=backend, mode="competition")
            if report["ok"]
            else {"ok": False, "backend_type": "not_run", "reason": "runtime_not_ready"}
        )
        quality_failures = []
        if quality_report.get("backend_type") != "live_ollama":
            quality_failures.append("quality_backend_not_live_ollama")
        quality_failures.extend(quality_gate_failures(quality_report))
        report.update(
            {
                "quality_report": quality_report,
                "available_tags": available_tags,
                "quality_failures": quality_failures,
                "ok": bool(report["ok"]) and not quality_failures,
            }
        )

    return report


@cli.command()
def doctor(
    evaluation: bool = typer.Option(
        False,
        "--eval",
        "--judge",
        help="Run the slow live quality evaluation with the effective model.",
    ),
) -> None:
    profile = get_runtime_profile()
    try:
        backend = OllamaBackend(profile)
    except BackendError as error:
        _emit_error(error.code, error.public_message)
    try:
        report = _doctor_report(profile, backend, evaluation)
    except BackendError as error:
        _emit_error(error.code, error.public_message)
    finally:
        backend.close()

    typer.echo(json.dumps(report, ensure_ascii=False))
    if not report["ok"]:
        raise typer.Exit(code=1)


def run() -> None:
    cli()


if __name__ == "__main__":
    run()
