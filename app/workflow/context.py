from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from pydantic import ValidationError

from app.workflow.contracts import (
    ContextEntry,
    ContextInventory,
    ContextValueType,
    JsonValue,
    WorkflowPath,
    WorkflowRoot,
)


class ContextInspector:
    def __init__(self, *, max_entries: int = 256, sample_chars: int = 6000) -> None:
        self.max_entries = max_entries
        self.sample_chars = sample_chars

    def inventory(self, context: JsonValue) -> ContextInventory:
        entries: list[ContextEntry] = []
        truncated = False
        workflow = context.get("wf") if isinstance(context, Mapping) else None
        roots: list[tuple[WorkflowRoot, JsonValue]]
        if isinstance(workflow, Mapping):
            roots = []
            if "vars" in workflow:
                roots.append((WorkflowRoot.VARS, workflow["vars"]))
            if "initVariables" in workflow:
                roots.append((WorkflowRoot.INIT_VARIABLES, workflow["initVariables"]))
        else:
            roots = []

        def walk(value: JsonValue, path: WorkflowPath) -> None:
            nonlocal truncated
            if len(entries) >= self.max_entries:
                truncated = True
                return
            entries.append(ContextEntry(path=path, value_type=self._value_type(value)))
            if isinstance(value, Mapping):
                for key in sorted(value):
                    child = self._child(path, key)
                    if child is None:
                        truncated = True
                        continue
                    walk(value[key], child)
            elif isinstance(value, Sequence) and not isinstance(value, str) and value:
                child = self._child(path, "[]")
                if child is None:
                    truncated = True
                    return
                walk(value[0], child)

        for root, value in roots:
            walk(value, WorkflowPath(root=root))
        return ContextInventory(entries=tuple(entries), truncated=truncated)

    def sample(self, context: JsonValue) -> JsonValue:
        encoded = json.dumps(context, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if len(encoded) <= self.sample_chars:
            return context
        inventory = self.inventory(context)
        return {
            "truncated": True,
            "paths": [
                {
                    "root": entry.path.root.value,
                    "segments": list(entry.path.segments),
                    "type": entry.value_type.value,
                }
                for entry in inventory.entries
            ],
        }

    @staticmethod
    def _child(path: WorkflowPath, segment: str) -> WorkflowPath | None:
        """Extend a workflow path, or report that the contract cannot represent the key.

        Workflow contexts come from user data, so a key may be longer than the contract allows or
        may contain control characters. Such a subtree is omitted from the inventory and marks it
        truncated instead of failing the whole request.
        """
        try:
            return WorkflowPath(root=path.root, segments=(*path.segments, segment))
        except ValidationError:
            return None

    @staticmethod
    def _value_type(value: JsonValue) -> ContextValueType:
        if value is None:
            return ContextValueType.NULL
        if isinstance(value, bool):
            return ContextValueType.BOOLEAN
        if isinstance(value, (int, float)):
            return ContextValueType.NUMBER
        if isinstance(value, str):
            return ContextValueType.STRING
        if isinstance(value, list):
            return ContextValueType.ARRAY
        return ContextValueType.OBJECT
