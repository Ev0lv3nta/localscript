import json

from app.generation.backend_errors import BackendUnavailable


def _plan_response():
    return json.dumps(
        {
            "kind": "plan",
            "objective": "Return the workflow value.",
            "inputs": [{"root": "wf.vars", "segments": ["value"]}],
            "output": {"format": "lua_block", "shape": "scalar", "nullable": False},
            "steps": [
                {
                    "description": "Read and return the value.",
                    "reads": [{"root": "wf.vars", "segments": ["value"]}],
                }
            ],
            "constraints": [],
            "acceptance_cases": [
                {
                    "name": "value",
                    "context": {"wf": {"vars": {"value": 4}}},
                    "expected": 4,
                }
            ],
        }
    )


class FailIfCalledBackend:
    def complete(self, prompt, response_format=None, model=None):
        raise AssertionError("backend should not be called for this test")


class UnavailableBackend:
    def complete(self, prompt, response_format=None, model=None):
        raise BackendUnavailable(reason="test_backend_unavailable")


class DeterministicTestBackend:
    def complete(self, prompt, response_format=None, model=None):
        if "You are the planner" in prompt:
            return _plan_response()
        if "You are the generator" in prompt or "You are revising" in prompt:
            return json.dumps({"code": "return wf.vars.value"})
        if "You are the reviewer" in prompt:
            return json.dumps({"kind": "approved"})
        raise AssertionError("unexpected model role")

    def ping(self):
        return True

    def list_tags(self):
        return []

    def close(self):
        return None
