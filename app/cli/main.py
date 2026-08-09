import json
import subprocess
from pathlib import Path

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
from app.core.verifier import verify_code
from app.generation.engine import GenerationEngine
from app.generation.ollama import OllamaBackend
from app.generation.taskspec import TaskSpec
from app.validation.validators import ValidationPipeline

cli = typer.Typer(help="LocalScript local CLI")


def _display_path(path):
    target = Path(path).resolve()
    project_root = Path(__file__).resolve().parents[2]
    try:
        return str(target.relative_to(project_root))
    except ValueError:
        return str(target)


def build_engine():
    profile = get_runtime_profile()
    return GenerationEngine(
        profile=profile,
        trace_store=TraceStore(),
        backend=OllamaBackend(profile),
    )


def run_vram_probe(model, fallback_model):
    with materialized_resource("scripts/bench_vram.sh") as vram_script:
        completed = subprocess.run(
            ["bash", str(vram_script), model, fallback_model, "judged_probe"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
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
    return report


@cli.command()
def generate(
    prompt=typer.Option(..., help="Natural-language generation request."),
    session_id=typer.Option(None, help="Optional iterative generation session id."),
    feedback=typer.Option(None, help="Optional feedback for revising the previous session output."),
):
    engine = build_engine()
    result = engine.generate(prompt=prompt, session_id=session_id, feedback=feedback)
    typer.echo(
        json.dumps(
            {
                "code": result.code,
                "trace_id": result.trace_id,
                "session_id": result.session_id,
                "strategy": result.strategy,
                "repair_rounds": result.repair_rounds,
                "degraded_mode": result.degraded_mode,
            },
            ensure_ascii=False,
        )
    )


@cli.command()
def interact(
    prompt=typer.Option(None, help="Natural-language generation request."),
    session_id=typer.Option(None, help="Existing session id for clarification/continuation."),
    answer=typer.Option(None, help="Clarification answer for an existing session."),
    feedback=typer.Option(None, help="Revision feedback for an existing session."),
):
    engine = build_engine()
    result = engine.generate_rich(
        prompt=prompt,
        session_id=session_id,
        clarification_answer=answer,
        feedback=feedback,
    )
    typer.echo(
        json.dumps(
            {
                "status": result.status,
                "code": result.code,
                "question": result.question,
                "trace_id": result.trace_id,
                "session_id": result.session_id,
                "strategy": result.strategy,
                "repair_rounds": result.repair_rounds,
                "degraded_mode": result.degraded_mode,
                "assumptions": result.assumptions,
                "session": result.session_summary,
            },
            ensure_ascii=False,
        )
    )


@cli.command()
def analyze(
    prompt=typer.Option(..., help="Natural-language generation request to inspect."),
):
    engine = build_engine()
    typer.echo(json.dumps(engine.analyze(prompt=prompt), ensure_ascii=False))


@cli.command()
def verify(
    code=typer.Option(None, help="Inline LocalScript/Lua code."),
    code_file=typer.Option(None, help="Path to a file with LocalScript/Lua code."),
):
    if not code and not code_file:
        raise typer.BadParameter("Provide --code or --code-file.")

    content = code or Path(code_file).read_text(encoding="utf-8")
    stripped = (content or "").strip()
    output_style = "json_envelope"
    if not (stripped.startswith("{") and stripped.endswith("}")):
        output_style = "lua_block"

    task_spec = TaskSpec(
        normalized_prompt="generic_lua_verification",
        family="generic_lua",
        output_style=output_style,
        target_root="unknown",
        context_paths=[],
        family_confidence=0.0,
        generation_hints={},
        assumptions=["Standalone CLI verify path."],
        ambiguity_notes=[],
    )
    pipeline = ValidationPipeline()
    report = pipeline.run(
        code=content,
        task_spec=task_spec,
        profile=get_runtime_profile(),
        source_context=None,
        prompt="generic_lua_verification",
    )
    errors = []
    for error_code in report.error_codes() + verify_code(content):
        if error_code not in errors:
            errors.append(error_code)
    degraded_codes = [
        message.code
        for message in report.messages
        if message.level == "warning" and "degraded" in message.message.lower()
    ]
    payload = {
        "ok": not errors,
        "errors": errors,
        "validation_report": report.to_dict(),
        "degraded_mode": bool(degraded_codes),
        "degraded_codes": degraded_codes,
    }
    typer.echo(json.dumps(payload, ensure_ascii=False))
    raise typer.Exit(code=0 if not errors else 1)


@cli.command()
def benchmark(dataset=typer.Option("evals/public/v1.jsonl", help="JSONL dataset path.")):
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

    report = run_dataset_benchmark(dataset_path, profile=get_runtime_profile(), backend=OllamaBackend(get_runtime_profile()))
    typer.echo(json.dumps(report, ensure_ascii=False))
    raise typer.Exit(code=0 if report["ok"] else 1)


@cli.command()
def doctor(judge: bool = typer.Option(False, "--judge", help="Run judged-path checks.")):
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

        if available_tags and profile.model not in available_tags and profile.fallback_model in available_tags:
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


def run():
    cli()


if __name__ == "__main__":
    run()
