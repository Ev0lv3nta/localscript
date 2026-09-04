from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import typer

from app.core.benchmarks import (
    quality_gate_failures,
    run_dataset_benchmark,
    run_quality_benchmark,
)
from app.core.config import get_runtime_profile
from app.core.resources import materialized_resource, resource_exists
from app.core.runtime_lock import build_runtime_lock, write_runtime_lock
from app.core.traces import TraceStore
from app.generation.engine import GenerationEngine
from app.generation.ollama import OllamaBackend
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


def _display_path(path: Path | str) -> str:
    target = Path(path).resolve()
    project_root = Path(__file__).resolve().parents[2]
    try:
        return str(target.relative_to(project_root))
    except ValueError:
        return str(target)


def build_engine() -> GenerationEngine:
    profile = get_runtime_profile()
    return GenerationEngine(
        profile=profile,
        trace_store=TraceStore(),
        backend=OllamaBackend(profile),
    )


def run_vram_probe(model: str, fallback_model: str) -> dict[str, Any]:
    with materialized_resource("scripts/bench_vram.sh") as vram_script:
        completed = subprocess.run(
            ["bash", str(vram_script), model, fallback_model, "judged_probe"],
            capture_output=True,
            text=True,
        )
    try:
        report = json.loads(completed.stdout.strip() or "{}")
    except json.JSONDecodeError:
        report = {
            "status": "error",
            "reason": "invalid_vram_report",
            "raw_stdout": completed.stdout.strip(),
        }
    report.setdefault("model", model)
    report.setdefault("fallback_model", fallback_model)
    typed_report: dict[str, Any] = report
    return typed_report


@cli.command()
def generate(
    prompt: str | None = typer.Option(None, help="Natural-language generation request."),
    context: str | None = typer.Option(None, help="Workflow context as JSON."),
    session_id: str | None = typer.Option(None, help="Existing session id to continue."),
    answer: str | None = typer.Option(None, help="Answer to the open clarification question."),
    feedback: str | None = typer.Option(None, help="Feedback for revising the previous result."),
) -> None:
    """Generate LocalScript code, or continue an existing session."""
    if not prompt and not session_id:
        raise typer.BadParameter("Provide --prompt for a new session or --session-id to continue.")
    parsed_context = None
    if context is not None:
        try:
            parsed_context = json.loads(context)
        except json.JSONDecodeError as error:
            raise typer.BadParameter("--context must be valid JSON") from error

    engine = build_engine()
    result = engine.generate(
        prompt=prompt,
        context=parsed_context,
        session_id=session_id,
        clarification_answer=answer,
        feedback=feedback,
    )
    workflow = result.workflow
    typer.echo(
        json.dumps(
            {
                "status": workflow.status.value,
                "session_id": result.session_id,
                "trace_id": result.trace_id,
                "code": workflow.code,
                "question": workflow.question,
                "diagnostics": [
                    diagnostic.model_dump(mode="json") for diagnostic in workflow.diagnostics
                ],
                "revision_count": workflow.revision_count,
            },
            ensure_ascii=False,
        )
    )
    raise typer.Exit(code=0 if workflow.status is WorkflowStatus.COMPLETED else 1)


@cli.command()
def validate(
    code: str | None = typer.Option(None, help="Inline LocalScript/Lua code."),
    code_file: str | None = typer.Option(None, help="Path to a file with LocalScript/Lua code."),
    context: str = typer.Option('{"wf":{"vars":{}}}', help="Workflow context as JSON."),
    output_format: str = typer.Option("lua_block", help="lua_block or json_envelope."),
    output_shape: str = typer.Option("scalar", help="scalar, array, or object."),
    nullable: bool = typer.Option(False, help="Allow a null result."),
) -> None:
    if not code and not code_file:
        raise typer.BadParameter("Provide --code or --code-file.")

    content = code or Path(str(code_file)).read_text(encoding="utf-8")
    try:
        parsed_context = json.loads(context)
    except json.JSONDecodeError as error:
        raise typer.BadParameter("--context must be valid JSON") from error
    if not isinstance(parsed_context, dict):
        raise typer.BadParameter("--context must contain a JSON object")
    try:
        output = OutputContract(
            format=OutputFormat(output_format),
            shape=OutputShape(output_shape),
            nullable=nullable,
        )
    except ValueError as error:
        raise typer.BadParameter("invalid output contract") from error

    report = DeterministicCandidateValidator().validate_existing(
        candidate=CodeCandidate(code=content),
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
    dataset: str = typer.Option("evals/live/v1.jsonl", help="JSONL dataset path."),
) -> None:
    dataset_path = Path(dataset)
    packaged_dataset = dataset.replace("\\", "/")
    if not dataset_path.is_file() and not (
        packaged_dataset.startswith("evals/") and resource_exists(packaged_dataset)
    ):
        typer.echo(
            json.dumps(
                {"ok": False, "error": "dataset_not_found", "dataset": str(dataset_path)},
                ensure_ascii=False,
            )
        )
        raise typer.Exit(code=1)

    report = run_dataset_benchmark(
        dataset_path, profile=get_runtime_profile(), backend=OllamaBackend(get_runtime_profile())
    )
    typer.echo(json.dumps(report, ensure_ascii=False))
    raise typer.Exit(code=0 if report["ok"] else 1)


@cli.command()
def doctor(judge: bool = typer.Option(False, "--judge", help="Run judged-path checks.")) -> None:
    profile = get_runtime_profile()
    backend = OllamaBackend(profile)
    report = {
        "profile": profile.name,
        "model": profile.model,
        "fallback_model": profile.fallback_model,
        "trace_dir_writable": TraceStore().root.exists(),
        "ollama_reachable": backend.ping(),
        "judge_mode": judge,
    }
    if judge:
        available_tags = backend.list_tags()
        primary_vram_report = run_vram_probe(profile.model, profile.fallback_model)
        fallback_vram_report = None
        selected_model = profile.model
        selection_reason = "primary_selected"

        if (
            available_tags
            and profile.model not in available_tags
            and profile.fallback_model in available_tags
        ):
            selected_model = profile.fallback_model
            selection_reason = "primary_tag_missing"
            fallback_vram_report = run_vram_probe(profile.fallback_model, profile.fallback_model)
        elif primary_vram_report.get("status") == "over_cap":
            selected_model = profile.fallback_model
            selection_reason = "primary_over_vram_cap"
            fallback_vram_report = run_vram_probe(profile.fallback_model, profile.fallback_model)
        elif primary_vram_report.get("status") == "ok":
            selected_model = profile.model
            selection_reason = "primary_within_vram_cap"
        elif primary_vram_report.get("status") == "skipped":
            selection_reason = "primary_selected_vram_skipped"

        selected_vram_report = primary_vram_report
        if selected_model == profile.fallback_model and fallback_vram_report is not None:
            selected_vram_report = fallback_vram_report

        selected_profile = profile
        selected_backend = backend
        if selected_model != profile.model:
            selected_profile = profile.model_copy(update={"model": selected_model})
            selected_backend = OllamaBackend(selected_profile)
        quality_report = run_quality_benchmark(
            profile=selected_profile,
            backend=selected_backend,
            mode="competition",
        )

        hard_gate_failures = []
        if not report["ollama_reachable"]:
            hard_gate_failures.append("ollama_unreachable")
        if quality_report.get("backend_type") != "live_ollama":
            hard_gate_failures.append("quality_backend_not_live_ollama")
        hard_gate_failures.extend(quality_gate_failures(quality_report))
        if available_tags and selected_model not in available_tags:
            hard_gate_failures.append("selected_model_tag_missing")
        if selected_vram_report.get("status") != "ok":
            hard_gate_failures.append("selected_model_vram_not_ok")

        lock_payload = build_runtime_lock(
            profile=profile,
            selected_model=selected_model,
            selection_reason=selection_reason,
            quality_report=quality_report,
            vram_report=selected_vram_report,
            primary_vram_report=primary_vram_report,
            fallback_vram_report=fallback_vram_report,
            available_tags=available_tags,
            hard_gate_failures=hard_gate_failures,
        )
        lock_path = write_runtime_lock(lock_payload)
        report.update(
            {
                "quality_report": quality_report,
                "vram_report": selected_vram_report,
                "primary_vram_report": primary_vram_report,
                "fallback_vram_report": fallback_vram_report,
                "available_tags": available_tags,
                "selected_model": selected_model,
                "selection_reason": selection_reason,
                "hard_gate_failures": hard_gate_failures,
                "ok": not hard_gate_failures,
                "runtime_snapshot_path": _display_path(lock_path),
            }
        )

    typer.echo(json.dumps(report, ensure_ascii=False))
    if judge and not report.get("ok", report["ollama_reachable"]):
        raise typer.Exit(code=1)


def run() -> None:
    cli()


if __name__ == "__main__":
    run()
