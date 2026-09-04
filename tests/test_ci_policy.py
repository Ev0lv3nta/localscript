import re
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = PROJECT_ROOT / ".github/workflows/ci.yml"
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")


def load_workflow():
    return yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))


def test_ci_has_stable_required_aggregator_and_least_permissions():
    workflow = load_workflow()

    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["concurrency"]["cancel-in-progress"] is True
    assert workflow["jobs"]["required"]["name"] == "CI / required"
    assert workflow["jobs"]["required"]["if"] == "${{ always() }}"
    assert workflow["jobs"]["required"]["permissions"] == {}


def test_all_github_actions_are_pinned_to_full_commit_shas():
    workflow = load_workflow()
    action_refs = []
    for job in workflow["jobs"].values():
        for step in job.get("steps", []):
            if "uses" in step:
                action_refs.append(step["uses"].rsplit("@", 1)[1])

    assert action_refs
    assert all(FULL_SHA.fullmatch(ref) for ref in action_refs)


def test_ci_uses_frozen_matrix_lua_package_container_and_secret_checks():
    workflow = load_workflow()
    workflow_text = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert workflow["jobs"]["unit"]["strategy"]["matrix"]["python-version"] == ["3.11", "3.12"]
    assert "uv sync --frozen --all-extras" in workflow_text
    assert "make test-unit" in workflow_text
    assert "make policy-check" in workflow_text
    assert "make build-check" in workflow_text
    assert "scripts/check_package_artifacts.py" in (PROJECT_ROOT / "Makefile").read_text(
        encoding="utf-8"
    )
    assert "make container-check" in workflow_text
    assert "make dependency-audit" in workflow_text
    assert "pip-audit==2.10.0" in (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "aquasecurity/trivy-action@" in workflow_text
    assert "severity: HIGH,CRITICAL" in workflow_text
    assert "gitleaks git" in workflow_text

    required_needs = set(workflow["jobs"]["required"]["needs"])
    assert "dependency-audit" in required_needs
