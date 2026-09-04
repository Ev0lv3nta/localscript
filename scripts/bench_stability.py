#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFERRED_PYTHON = ROOT / ".venv" / "bin" / "python"
PREFERRED_VENV = PREFERRED_PYTHON.parent.parent.resolve()
if PREFERRED_PYTHON.exists() and Path(sys.prefix).resolve() != PREFERRED_VENV:
    os.execv(
        str(PREFERRED_PYTHON),
        [str(PREFERRED_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
    )
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.benchmarks import run_stability_benchmark


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Проверка стабильности LocalScript на сценариях из манифеста."
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    report = run_stability_benchmark()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False))
    if not report["ok"]:
        raise SystemExit(1)
    return report


if __name__ == "__main__":
    main()
