import json

import yaml

from app.core.resources import get_resource, read_resource_text
from app.evaluation.manifest import dataset_specs


def _normalize_prompt(value):
    return " ".join((value or "").casefold().split())


def _load_eval_prompts():
    prompts = {}
    for spec in dataset_specs():
        dataset_path = get_resource(spec.path)
        lines = dataset_path.read_text(encoding="utf-8").splitlines()
        for line_number, raw_line in enumerate(lines, start=1):
            if not raw_line.strip():
                continue
            case = json.loads(raw_line)
            prompt = _normalize_prompt(case.get("prompt"))
            if prompt:
                prompts.setdefault(prompt, []).append(
                    "{0}:{1}".format(dataset_path.name, line_number)
                )
    return prompts


def test_kb_examples_have_explicit_synthetic_provenance():
    payload = yaml.safe_load(read_resource_text("kb/examples.yaml"))

    assert payload["schema_version"] == 1
    assert payload["provenance"] == {
        "origin": "synthetic",
        "author": "repository_owner",
        "license": "MIT",
    }

    for example in payload["examples"]:
        assert example["id"].startswith("synthetic_")
        assert "dataset_ref" not in example
        assert "reference_id" not in example


def test_kb_prompts_do_not_duplicate_eval_prompts():
    payload = yaml.safe_load(read_resource_text("kb/examples.yaml"))
    kb_prompts = {
        _normalize_prompt(example["prompt"]): example["id"]
        for example in payload["examples"]
    }
    eval_prompts = _load_eval_prompts()

    overlap = sorted(set(kb_prompts) & set(eval_prompts))

    assert overlap == [], {
        prompt: {
            "kb": kb_prompts[prompt],
            "eval": eval_prompts[prompt],
        }
        for prompt in overlap
    }
