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
        seen: set[tuple[WorkflowPath, ContextValueType]] = set()
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
            value_type = self._value_type(value)
            if (path, value_type) not in seen:
                entries.append(ContextEntry(path=path, value_type=value_type))
                seen.add((path, value_type))
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
                # Inspect every allowed input item: heterogeneous records must not disappear.
                for item in value:
                    walk(item, child)
                    if truncated:
                        break

        for root, value in roots:
            walk(value, WorkflowPath(root=root))
        return ContextInventory(entries=tuple(entries), truncated=truncated)

    @staticmethod
    def roots_overlap(context: dict[str, JsonValue]) -> bool:
        """Check the full input, independently of the model-facing inventory budget.

        A shared nested path always has a shared first segment. We need only this boolean
        for the publication boundary, not an unbounded list of paths for the model.
        """
        workflow = context.get("wf")
        if not isinstance(workflow, dict):
            return False
        left, right = workflow.get("vars"), workflow.get("initVariables")
        if isinstance(left, dict) and isinstance(right, dict):
            return bool(left.keys() & right.keys())
        return isinstance(left, list) and isinstance(right, list) and bool(left) and bool(right)

    @staticmethod
    def ambiguous_paths(inventory: ContextInventory) -> tuple[str, ...]:
        """Return the paths that exist under both workflow roots.

        Whether a request means wf.vars or wf.initVariables is the one ambiguity the product
        promises to ask about, and it is a fact about the context rather than about the wording.
        Computing it here keeps the planner from having to notice it on its own.
        """
        by_root: dict[WorkflowRoot, set[tuple[str, ...]]] = {}
        for entry in inventory.entries:
            if not entry.path.segments:
                continue
            by_root.setdefault(entry.path.root, set()).add(entry.path.segments)
        if len(by_root) < 2:
            return ()
        shared = set.intersection(*by_root.values())
        return tuple(".".join(path) for path in sorted(shared))

    def sample(self, context: JsonValue) -> JsonValue:
        encoded = json.dumps(context, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if len(encoded) <= self.sample_chars:
            return context
        remaining = max(0, self.sample_chars - 1500)
        omitted: list[JsonValue] = []

        def select(value: JsonValue, path: str) -> JsonValue:
            nonlocal remaining
            if isinstance(value, dict):
                result: dict[str, JsonValue] = {}
                for key in sorted(value):
                    cost = len(json.dumps(key)) + 4
                    if remaining < cost + 20:
                        if len(omitted) < 12:
                            omitted.append(f"{path}[{key!r}]"[:80])
                        continue
                    remaining -= cost
                    result[key] = select(value[key], f"{path}[{key!r}]")
                return result
            if isinstance(value, list):
                indices = sorted({0, len(value) // 2, len(value) - 1}) if value else []
                result_list = [
                    select(value[index], f"{path}[{index}]") for index in indices if remaining > 40
                ]
                if len(result_list) < len(value) and len(omitted) < 12:
                    omitted.append(f"{path}: {len(value)} items, representative subset shown"[:80])
                return result_list
            cost = len(json.dumps(value, ensure_ascii=False)) + 4
            if cost > remaining or cost > 300:
                if len(omitted) < 12:
                    omitted.append(path[:80])
                return {"omitted": True, "type": self._value_type(value).value}
            remaining -= cost
            return value

        return {
            "truncated": True,
            "representative_context": select(context, "$"),
            "omitted_paths": omitted,
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
