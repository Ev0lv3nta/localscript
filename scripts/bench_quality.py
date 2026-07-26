#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFERRED_PYTHON = ROOT / ".venv" / "bin" / "python"
PREFERRED_VENV = PREFERRED_PYTHON.parent.parent.resolve()
if PREFERRED_PYTHON.exists() and Path(sys.prefix).resolve() != PREFERRED_VENV:
    os.execv(str(PREFERRED_PYTHON), [str(PREFERRED_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.benchmarks import run_quality_benchmark


def main():
    strict = "--strict" in sys.argv
    mode = "competition" if strict else "dev"
    report = run_quality_benchmark(mode=mode)
    print(json.dumps(report, ensure_ascii=False))
    if strict and not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
