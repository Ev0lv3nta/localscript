from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Sequence
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from app.core.public_eval import load_cases
from app.core.resources import materialized_resource
from app.evaluation.manifest import dataset_specs, load_evaluation_manifest

FUZZY_THRESHOLD = 0.85
OUTPUT_STYLES = frozenset({"lua_block", "lua_expression", "json_envelope"})


def normalize_prompt(prompt: object) -> str:
    normalized = unicodedata.normalize("NFKC", str(prompt or "")).casefold()
    return " ".join(re.sub(r"[^\w]+", " ", normalized).split())


def _sha256_path(dataset_path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(dataset_path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _case_input_fingerprint(case: dict[str, Any]) -> str:
    payload = {
        "prompt": normalize_prompt(case.get("prompt")),
        "context": case.get("context"),
        "clarification_answer": case.get("clarification_answer"),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _validate_case(
    case: object,
    corpus: str,
    seen_ids: set[str],
    seen_inputs: dict[str, str],
) -> list[str]:
    errors: list[str] = []
    if not isinstance(case, dict):
        return ["case_not_object"]
    case_id = case.get("id")
    if not isinstance(case_id, str) or not case_id.strip():
        errors.append("case_id_missing")
    elif case_id in seen_ids:
        errors.append(f"case_id_duplicate::{case_id}")
    else:
        seen_ids.add(case_id)
    if not isinstance(case.get("prompt"), str) or not case["prompt"].strip():
        errors.append(f"case_prompt_missing::{case_id}")
    if case.get("context") is not None and not isinstance(case.get("context"), dict):
        errors.append(f"case_context_not_object::{case_id}")
    if (
        case.get("expected_output_style") is not None
        and case.get("expected_output_style") not in OUTPUT_STYLES
    ):
        errors.append(f"case_output_style_invalid::{case_id}")

    fingerprint = _case_input_fingerprint(case)
    previous = seen_inputs.get(fingerprint)
    if previous is not None:
        errors.append(f"case_input_duplicate::{previous}::{case_id}")
    else:
        seen_inputs[fingerprint] = str(case_id)

    if corpus == "public_benchmark":
        if case.get("source") != "owner_synthetic_public_v1":
            errors.append(f"public_case_source_invalid::{case_id}")
        if case.get("case_type") != "public_benchmark":
            errors.append(f"public_case_type_invalid::{case_id}")
        if case.get("expected_output_style") not in OUTPUT_STYLES:
            errors.append(f"public_case_output_style_missing::{case_id}")
        if "expected_result" not in case and not case.get("semantic_checks"):
            errors.append(f"public_case_oracle_missing::{case_id}")
        if "expected_code" in case or "reference_code" in case:
            errors.append(f"public_case_reference_injection::{case_id}")
    return errors


def _find_cross_corpus_overlaps(
    protected_records: Sequence[dict[str, Any]],
    comparison_records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for protected in protected_records:
        protected_prompt = normalize_prompt(protected["prompt"])
        protected_tokens = set(protected_prompt.split())
        for other in comparison_records:
            other_prompt = normalize_prompt(other["prompt"])
            if protected_prompt == other_prompt:
                findings.append(
                    {
                        "kind": "normalized_exact",
                        "protected_source": protected["source"],
                        "protected_id": protected["id"],
                        "other_source": other["source"],
                        "other_id": other["id"],
                        "score": 1.0,
                    }
                )
                continue
            sequence_score = SequenceMatcher(None, protected_prompt, other_prompt).ratio()
            other_tokens = set(other_prompt.split())
            token_score = (
                float(len(protected_tokens & other_tokens))
                / float(len(protected_tokens | other_tokens))
                if protected_tokens or other_tokens
                else 1.0
            )
            score = max(sequence_score, token_score)
            if score >= FUZZY_THRESHOLD:
                findings.append(
                    {
                        "kind": "fuzzy",
                        "protected_source": protected["source"],
                        "protected_id": protected["id"],
                        "other_source": other["source"],
                        "other_id": other["id"],
                        "score": round(score, 4),
                    }
                )
    return findings


def _load_external_holdout(
    dataset_path: Path | str,
    expected: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = Path(dataset_path)
    evidence: dict[str, Any] = {
        "name": expected["name"],
        "case_count": 0,
        "sha256": None,
        "ok": False,
    }
    if not path.is_file():
        evidence["error"] = "private_holdout_missing"
        return evidence, []
    cases = load_cases(path)
    evidence.update(
        {
            "case_count": len(cases),
            "sha256": _sha256_path(path),
        }
    )
    evidence["ok"] = (
        evidence["case_count"] == expected["case_count"]
        and evidence["sha256"] == expected["sha256"]
    )
    if not evidence["ok"]:
        evidence["error"] = "private_holdout_identity_mismatch"
    return evidence, cases


def run_integrity_check(private_holdout_path: Path | str | None = None) -> dict[str, Any]:
    specs = dataset_specs()
    seen_ids: set[str] = set()
    seen_inputs: dict[str, str] = {}
    schema_errors: list[str] = []
    datasets: list[dict[str, Any]] = []
    public_records: list[dict[str, Any]] = []
    comparison_records: list[dict[str, Any]] = []

    for spec in specs:
        with materialized_resource(spec.path) as dataset_path:
            cases = load_cases(dataset_path)
            digest = _sha256_path(dataset_path)
        for case in cases:
            schema_errors.extend(
                _validate_case(case, spec.corpus, seen_ids, seen_inputs)
            )
            record = {
                "source": spec.name,
                "id": str(case.get("id")),
                "prompt": str(case.get("prompt") or ""),
            }
            if spec.corpus == "public_benchmark":
                public_records.append(record)
            else:
                comparison_records.append(record)
        datasets.append(
            {
                **spec.evidence_dict(),
                "case_count": len(cases),
                "sha256": digest,
            }
        )

    manifest = load_evaluation_manifest()
    holdout_evidence: dict[str, Any] | None = None
    holdout_cases: list[dict[str, Any]] = []
    holdout_records: list[dict[str, Any]] = []
    if private_holdout_path is not None:
        expected_holdouts = manifest.get("private_holdouts") or []
        if len(expected_holdouts) != 1:
            schema_errors.append("private_holdout_manifest_invalid")
        else:
            holdout_evidence, holdout_cases = _load_external_holdout(
                private_holdout_path,
                expected_holdouts[0],
            )
            if not holdout_evidence["ok"]:
                schema_errors.append(holdout_evidence["error"])
            for case in holdout_cases:
                schema_errors.extend(
                    _validate_case(
                        case,
                        "private_holdout",
                        seen_ids,
                        seen_inputs,
                    )
                )
                holdout_records.append(
                    {
                        "source": "private_holdout",
                        "id": str(case.get("id")),
                        "prompt": str(case.get("prompt") or ""),
                    }
                )

    overlaps = _find_cross_corpus_overlaps(public_records, comparison_records)
    if holdout_records:
        overlaps.extend(
            _find_cross_corpus_overlaps(
                holdout_records,
                comparison_records + public_records,
            )
        )
    errors = list(schema_errors)
    errors.extend(
        "corpus_overlap::{}::{}::{}".format(
            finding["protected_id"],
            finding["other_source"],
            finding["other_id"],
        )
        for finding in overlaps
    )
    return {
        "schema_version": 1,
        "ok": not errors,
        "errors": errors,
        "datasets": datasets,
        "private_holdout": holdout_evidence,
        "overlaps": overlaps,
        "fuzzy_threshold": FUZZY_THRESHOLD,
    }
