#!/usr/bin/env python3
"""Check the running API and real Lua execution without calling a model."""

import json
import os

import httpx


def main():
    base_url = os.getenv("LOCALSCRIPT_API_URL", "http://127.0.0.1:8080")
    try:
        with httpx.Client(base_url=base_url, timeout=10, trust_env=False) as client:
            response = client.get("/health")
            response.raise_for_status()
            if response.json().get("status") != "ok":
                raise ValueError("health_failed")
            response = client.post(
                "/api/validate",
                json={
                    "code": "return wf.vars.value",
                    "context": {"wf": {"vars": {"value": 7}}},
                    "output": {"format": "lua_block", "shape": "scalar", "nullable": False},
                },
            )
            response.raise_for_status()
            result = response.json()
            if not result.get("ok") or result.get("validation", {}).get("observations") != [
                {"actual": 7}
            ]:
                raise ValueError("runtime_smoke_failed")
    except (ValueError, httpx.HTTPError) as error:
        print(json.dumps({"ok": False, "failure": type(error).__name__}))
        return 1
    print(json.dumps({"ok": True, "api": "passed", "lua_execution": "passed", "model": "not_run"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
