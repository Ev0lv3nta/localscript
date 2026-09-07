from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

from app.core.resources import read_resource_text, resource_exists

MANIFEST_RESOURCE = "evals/manifest.json"
ALLOWED_CORPORA = frozenset({"public"})
ALLOWED_GATES = frozenset({"required"})


@dataclass(frozen=True)
class EvaluationDataset:
    name: str
    path: str
    corpus: str
    gate: str
    case_count: int
    supported_case_count: int
    clarification_case_count: int
    safety_case_count: int
    min_supported_success_rate: float
    claim_scope: str

    @property
    def required(self) -> bool:
        return self.gate == "required"

    @property
    def min_verified(self) -> int:
        return (
            math.ceil(self.supported_case_count * self.min_supported_success_rate)
            + self.clarification_case_count
            + self.safety_case_count
        )

    def evidence_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "corpus": self.corpus,
            "gate": self.gate,
            "case_count": self.case_count,
            "supported_case_count": self.supported_case_count,
            "clarification_case_count": self.clarification_case_count,
            "safety_case_count": self.safety_case_count,
            "min_supported_success_rate": self.min_supported_success_rate,
            "min_verified": self.min_verified,
            "claim_scope": self.claim_scope,
        }


def load_evaluation_manifest() -> dict[str, Any]:
    try:
        payload = json.loads(read_resource_text(MANIFEST_RESOURCE))
    except (json.JSONDecodeError, OSError, ValueError) as error:
        raise ValueError("evaluation_manifest_invalid") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 3:
        raise ValueError("evaluation_manifest_schema_unsupported")
    datasets = payload.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("evaluation_manifest_datasets_missing")
    holdouts = payload.get("private_holdouts")
    if not isinstance(holdouts, list) or len(holdouts) != 1:
        raise ValueError("evaluation_manifest_private_holdout_invalid")
    holdout = holdouts[0]
    if (
        not isinstance(holdout, dict)
        or holdout.get("external") is not True
        or not isinstance(holdout.get("case_count"), int)
        or holdout["case_count"] <= 0
        or re.fullmatch(r"[0-9a-f]{64}", str(holdout.get("sha256") or "")) is None
        or "path" in holdout
    ):
        raise ValueError("evaluation_manifest_private_holdout_invalid")
    return payload


def dataset_specs() -> tuple[EvaluationDataset, ...]:
    payload = load_evaluation_manifest()
    specs: list[EvaluationDataset] = []
    names: set[str] = set()
    paths: set[str] = set()
    for raw in payload["datasets"]:
        if not isinstance(raw, dict):
            raise ValueError("evaluation_manifest_dataset_invalid")
        expected_fields = {
            "name",
            "path",
            "corpus",
            "gate",
            "case_count",
            "supported_case_count",
            "clarification_case_count",
            "safety_case_count",
            "min_supported_success_rate",
            "claim_scope",
        }
        if set(raw) != expected_fields:
            raise ValueError("evaluation_manifest_dataset_fields_invalid")
        count_fields = (
            "case_count",
            "supported_case_count",
            "clarification_case_count",
            "safety_case_count",
        )
        if any(type(raw[field]) is not int for field in count_fields):
            raise ValueError("evaluation_manifest_dataset_counts_invalid")
        if type(raw["min_supported_success_rate"]) not in {int, float}:
            raise ValueError("evaluation_manifest_dataset_threshold_invalid")
        try:
            spec = EvaluationDataset(
                name=str(raw["name"]),
                path=str(raw["path"]),
                corpus=str(raw["corpus"]),
                gate=str(raw["gate"]),
                case_count=int(raw["case_count"]),
                supported_case_count=int(raw["supported_case_count"]),
                clarification_case_count=int(raw["clarification_case_count"]),
                safety_case_count=int(raw["safety_case_count"]),
                min_supported_success_rate=float(raw["min_supported_success_rate"]),
                claim_scope=str(raw["claim_scope"]),
            )
        except KeyError as error:
            raise ValueError("evaluation_manifest_dataset_field_missing") from error
        if not spec.name or spec.name in names:
            raise ValueError("evaluation_manifest_dataset_name_duplicate")
        if not spec.path or spec.path in paths or not resource_exists(spec.path):
            raise ValueError("evaluation_manifest_dataset_path_invalid")
        if spec.corpus not in ALLOWED_CORPORA:
            raise ValueError("evaluation_manifest_corpus_invalid")
        if spec.gate not in ALLOWED_GATES:
            raise ValueError("evaluation_manifest_gate_invalid")
        if (
            spec.case_count <= 0
            or spec.supported_case_count <= 0
            or spec.clarification_case_count <= 0
            or spec.safety_case_count <= 0
            or spec.case_count
            != spec.supported_case_count + spec.clarification_case_count + spec.safety_case_count
            or not 0.0 < spec.min_supported_success_rate <= 1.0
        ):
            raise ValueError("evaluation_manifest_min_verified_invalid")
        names.add(spec.name)
        paths.add(spec.path)
        specs.append(spec)
    if len(specs) != 1:
        raise ValueError("evaluation_manifest_public_corpus_invalid")
    return tuple(specs)


def stability_plan() -> tuple[str, tuple[str, ...], int]:
    """Return the dataset, cases and repeat count of the stability check.

    Stability is deliberately narrow: repeating the whole corpus multiplies GPU time without
    telling us anything the three representative scenarios do not.
    """
    payload = load_evaluation_manifest()
    plan = payload.get("stability")
    if not isinstance(plan, dict):
        raise ValueError("evaluation_manifest_stability_missing")
    dataset = plan.get("dataset")
    case_ids = plan.get("case_ids")
    repeats = plan.get("repeats")
    if (
        not isinstance(dataset, str)
        or not isinstance(case_ids, list)
        or not case_ids
        or not all(isinstance(item, str) and item for item in case_ids)
        or not isinstance(repeats, int)
        or repeats < 2
    ):
        raise ValueError("evaluation_manifest_stability_invalid")
    known = {spec.name for spec in dataset_specs()}
    if dataset not in known:
        raise ValueError("evaluation_manifest_stability_dataset_unknown")
    return dataset, tuple(case_ids), repeats
