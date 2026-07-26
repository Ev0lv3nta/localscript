#!/usr/bin/env python3
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFERRED_PYTHON = ROOT / ".venv" / "bin" / "python"
PREFERRED_VENV = PREFERRED_PYTHON.parent.parent.resolve()
if PREFERRED_PYTHON.exists() and Path(sys.prefix).resolve() != PREFERRED_VENV:
    os.execv(str(PREFERRED_PYTHON), [str(PREFERRED_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_runtime_profile
from app.core.state import get_state_root
from app.core.traces import TraceStore
from app.generation.engine import GenerationEngine
from app.generation.ollama import OllamaBackend


class UnsupportedRootBackend:
    def generate(self, prompt, context=None):
        return "local items = ctx.body.items\nreturn items[1]"


def measure_case(name, engine, prompt, context=None, expected_strategy=None, min_repair_rounds=None):
    started = time.perf_counter()
    result = engine.generate(prompt=prompt, context=context)
    elapsed = round(time.perf_counter() - started, 3)

    failures = []
    if expected_strategy and result.strategy != expected_strategy:
        failures.append("unexpected_strategy::{0}".format(result.strategy))
    if min_repair_rounds is not None and result.repair_rounds < min_repair_rounds:
        failures.append("repair_rounds_below_expected::{0}".format(result.repair_rounds))
    if result.verification_errors:
        failures.extend(result.verification_errors)

    return {
        "name": name,
        "elapsed_seconds": elapsed,
        "strategy": result.strategy,
        "repair_rounds": result.repair_rounds,
        "selected_model": engine.profile.model,
        "ok": not failures,
        "failures": failures,
        "trace_id": result.trace_id,
    }


def main():
    profile = get_runtime_profile()
    trace_root = get_state_root() / "traces" / "benchmarks" / "latency"

    public_engine = GenerationEngine(
        profile=profile,
        trace_store=TraceStore(root=trace_root),
        backend=OllamaBackend(profile),
    )
    model_engine = GenerationEngine(
        profile=profile,
        trace_store=TraceStore(root=trace_root),
        backend=OllamaBackend(profile),
    )
    repair_engine = GenerationEngine(
        profile=profile,
        trace_store=TraceStore(root=trace_root),
        backend=UnsupportedRootBackend(),
    )

    report = {
        "profile": profile.name,
        "cases": [
            measure_case(
                "public_case",
                public_engine,
                "Верни последний адрес из массива wf.vars.emails; если массив пустой, верни nil.",
                {"wf": {"vars": {"emails": ["a@example.com", "b@example.com"]}}},
                expected_strategy="ollama_chain",
            ),
            measure_case(
                "model_backed_case",
                model_engine,
                "Возьми wf.vars.contacts и подготовь список таблиц для активных контактов с email: поле id оставь как есть, а email переведи в lower case. Если вход пустой, нужен пустой список через _utils.array.new().",
                {
                    "wf": {
                        "vars": {
                            "contacts": [
                                {"id": "C1", "active": True, "email": "ADMIN@EXAMPLE.COM"},
                                {"id": "C2", "active": False, "email": "skip@example.com"},
                                {"id": "C3", "active": True, "email": "Owner@Example.com"},
                            ]
                        }
                    }
                },
                expected_strategy="ollama_chain",
            ),
            measure_case(
                "repair_case",
                repair_engine,
                "Iterate over the incoming items collection and return the first value.",
                {"wf": {"vars": {"items": [1, 2, 3]}}},
                expected_strategy="ollama_chain",
                min_repair_rounds=1,
            ),
        ],
    }
    report["ok"] = all(case["ok"] for case in report["cases"])
    print(json.dumps(report, ensure_ascii=False))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
