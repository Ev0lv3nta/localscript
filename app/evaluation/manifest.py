from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from app.core.resources import read_resource_text, resource_exists

MANIFEST_RESOURCE = "evals/manifest.json"
ALLOWED_CORPORA = frozenset({"live"})
ALLOWED_GATES = frozenset({"required"})


@dataclass(frozen=True)
class EvaluationDataset:
    name: str
    path: str
    corpus: str
    gate: str
    claim_scope: str

    @property
    def required(self) -> bool:
        return self.gate == "required"

    def evidence_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "path": self.path,
            "corpus": self.corpus,
            "gate": self.gate,
            "claim_scope": self.claim_scope,
        }


def load_evaluation_manifest() -> dict[str, Any]:
    try:
        payload = json.loads(read_resource_text(MANIFEST_RESOURCE))
    except (json.JSONDecodeError, OSError, ValueError) as error:
        raise ValueError("evaluation_manifest_invalid") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 2:
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
        try:
            spec = EvaluationDataset(
                name=str(raw["name"]),
                path=str(raw["path"]),
                corpus=str(raw["corpus"]),
                gate=str(raw["gate"]),
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
        names.add(spec.name)
        paths.add(spec.path)
        specs.append(spec)
    if len(specs) != 1:
        raise ValueError("evaluation_manifest_live_corpus_invalid")
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
