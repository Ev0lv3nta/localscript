import os
import json
import ipaddress
from urllib.parse import urlparse

import httpx


class OllamaBackend:
    def __init__(self, profile):
        self.profile = profile
        self.host = os.getenv("LOCALSCRIPT_OLLAMA_HOST", profile.ollama_host).rstrip("/")
        self.base_url = self.host
        self._validate_host_policy()

    def _validate_host_policy(self):
        if self.profile.name != "competition":
            return
        if os.getenv("LOCALSCRIPT_ALLOW_REMOTE_OLLAMA", "0") == "1":
            return

        parsed = urlparse(self.host)
        hostname = (parsed.hostname or "").strip().lower()
        if hostname in {"localhost", "127.0.0.1", "::1", "ollama"}:
            return
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            raise RuntimeError("ollama_host_not_local")
        if address.is_loopback or address.is_private:
            return
        raise RuntimeError("ollama_host_not_local")

    def ping(self):
        try:
            with httpx.Client(timeout=2.0) as client:
                response = client.get("{0}/api/tags".format(self.host))
                return response.status_code == 200
        except httpx.HTTPError:
            return False

    def list_tags(self):
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get("{0}/api/tags".format(self.host))
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPError:
            return []
        return [item.get("name") for item in payload.get("models", []) if item.get("name")]

    def complete(self, prompt, model=None, response_format=None):
        payload = {
            "model": model or self.profile.model,
            "think": self.profile.think,
            "stream": False,
            "prompt": prompt or "",
            "options": {
                "num_ctx": self.profile.num_ctx,
                "num_predict": self.profile.num_predict,
                "num_batch": self.profile.batch,
            },
        }
        if response_format:
            payload["format"] = response_format
        timeout = float(self.profile.request_timeout_seconds)
        with httpx.Client(timeout=timeout) as client:
            response = client.post("{0}/api/generate".format(self.host), json=payload)
            response.raise_for_status()
            data = response.json()

        candidate = (data.get("response") or "").strip()
        if not candidate:
            raise RuntimeError("ollama_empty_response")
        return candidate

    def generate(self, prompt, context=None):
        return self.complete(self._build_prompt(prompt, context))

    @staticmethod
    def _build_prompt(prompt, context):
        sections = [
            "You generate only LocalScript/Lua code.",
            "Do not use JsonPath.",
            "Use wf.vars or wf.initVariables for direct data access.",
            "Return code only without markdown fences.",
            "User prompt:",
            prompt or "",
        ]
        if context is not None:
            sections.extend(["Context JSON:", json.dumps(context, ensure_ascii=False, sort_keys=True)])
        return "\n".join(sections)
