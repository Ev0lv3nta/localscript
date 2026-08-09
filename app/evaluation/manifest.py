import json
import re
from dataclasses import dataclass

from app.core.resources import read_resource_text, resource_exists

MANIFEST_RESOURCE = "evals/manifest.json"
ALLOWED_CORPORA = frozenset({"public_benchmark", "regression"})
ALLOWED_RUNNERS = frozenset({"standard", "rich"})
ALLOWED_GATES = frozenset({"required", "diagnostic"})


@dataclass(frozen=True)
class EvaluationDataset:
    name: str
    path: str
    corpus: str
    runner: str
    gate: str
    claim_scope: str

    @property
    def required(self):
        return self.gate == "required"

    def evidence_dict(self):
        return {
            "name": self.name,
            "path": self.path,
            "corpus": self.corpus,
            "runner": self.runner,
            "gate": self.gate,
            "claim_scope": self.claim_scope,
        }


def load_evaluation_manifest():
    try:
        payload = json.loads(read_resource_text(MANIFEST_RESOURCE))
    except (json.JSONDecodeError, OSError, ValueError) as error:
        raise ValueError("evaluation_manifest_invalid") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
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


def dataset_specs():
    payload = load_evaluation_manifest()
    specs = []
    names = set()
    paths = set()
    for raw in payload["datasets"]:
        if not isinstance(raw, dict):
            raise ValueError("evaluation_manifest_dataset_invalid")
        try:
            spec = EvaluationDataset(
                name=str(raw["name"]),
                path=str(raw["path"]),
                corpus=str(raw["corpus"]),
                runner=str(raw["runner"]),
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
        if spec.runner not in ALLOWED_RUNNERS:
            raise ValueError("evaluation_manifest_runner_invalid")
        if spec.gate not in ALLOWED_GATES:
            raise ValueError("evaluation_manifest_gate_invalid")
        names.add(spec.name)
        paths.add(spec.path)
        specs.append(spec)
    if sum(spec.corpus == "public_benchmark" for spec in specs) != 1:
        raise ValueError("evaluation_manifest_public_corpus_invalid")
    return tuple(specs)
