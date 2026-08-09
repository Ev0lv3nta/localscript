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

from app.evaluation.ablation import run_ablation_benchmark  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Compare LocalScript pipeline layers on one dataset."
    )
    parser.add_argument(
        "--dataset",
        default="evals/public/v1.jsonl",
        help="JSONL dataset path or packaged eval resource.",
    )
    parser.add_argument(
        "--require-full-pass",
        action="store_true",
        help="Exit non-zero unless full pipeline verifies every case.",
    )
    args = parser.parse_args()

    report = run_ablation_benchmark(args.dataset)
    print(json.dumps(report, ensure_ascii=False))
    full_metrics = report["profiles"]["full_pipeline"]["metrics"]
    if args.require_full_pass and (
        full_metrics["verified_cases"] != report["case_count"]
        or full_metrics["invalid_success_count"] != 0
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
