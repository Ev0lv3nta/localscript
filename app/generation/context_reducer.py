import json


class ContextReducer:
    def __init__(self, max_serialized_chars=1800, max_paths=24, max_array_items=2):
        self.max_serialized_chars = max_serialized_chars
        self.max_paths = max_paths
        self.max_array_items = max_array_items

    def reduce(self, context, task_spec):
        if context is None:
            return None

        serialized = json.dumps(context, ensure_ascii=False, sort_keys=True)
        context_paths = list(task_spec.context_paths or [])
        if len(serialized) <= self.max_serialized_chars and len(context_paths) <= self.max_paths:
            return context

        highlighted_paths = self._highlighted_paths(task_spec)
        kept_paths = []
        for path in list(task_spec.prompt_paths or []) + highlighted_paths + list(task_spec.root_candidates or []):
            if isinstance(path, str) and path and path not in kept_paths:
                kept_paths.append(path)
        return {
            "reduced": True,
            "__localscript_reduced__": True,
            "__omitted_paths_count__": max(0, len(context_paths) - len(kept_paths)),
            "__kept_paths__": kept_paths[: self.max_paths],
            "roots": self._roots_summary(context),
            "root_candidates": list(task_spec.root_candidates or []),
            "highlighted_paths": highlighted_paths,
            "available_paths_sample": context_paths[: self.max_paths],
            "context_density": task_spec.context_density,
            "shape_summary": self._shape_summary(context),
            "sampled_fragments": self._sample_fragments(context, kept_paths),
        }

    @staticmethod
    def _roots_summary(context):
        wf = (context or {}).get("wf", {}) if isinstance(context, dict) else {}
        summary = {}
        for root in ["vars", "initVariables"]:
            value = wf.get(root)
            if isinstance(value, dict):
                summary["wf.{0}".format(root)] = sorted(value.keys())
        return summary

    @staticmethod
    def _highlighted_paths(task_spec):
        params = task_spec.generation_hints or {}
        highlighted = []
        for key in ["source_path", "datum_path", "time_path", "packages_path", "result_path", "counter_path", "iso_path", "email_path"]:
            value = params.get(key)
            if isinstance(value, str):
                highlighted.append(value)
        return highlighted[:8]

    def _shape_summary(self, context):
        def summarize(value):
            if isinstance(value, dict):
                return {
                    "type": "object",
                    "keys": sorted(list(value.keys()))[:8],
                }
            if isinstance(value, list):
                summary = {"type": "array", "length": len(value)}
                if value:
                    summary["item_shape"] = summarize(value[0])
                return summary
            return {"type": type(value).__name__}

        return summarize(context)

    def _sample_fragments(self, context, highlighted_paths):
        fragments = {}
        for path in highlighted_paths:
            value = self._lookup_path(context, path)
            if isinstance(value, list):
                fragments[path] = value[: self.max_array_items]
            elif value is not None:
                fragments[path] = value
            parent_path = ".".join(path.split(".")[:-1])
            if parent_path and parent_path not in fragments:
                parent_value = self._lookup_path(context, parent_path)
                if isinstance(parent_value, list):
                    fragments[parent_path] = parent_value[: self.max_array_items]
                elif parent_value is not None:
                    fragments[parent_path] = parent_value
        return fragments

    @staticmethod
    def _lookup_path(context, path):
        current = context
        for part in (path or "").split("."):
            if not part:
                continue
            if isinstance(current, dict) and part in current:
                current = current[part]
                continue
            return None
        return current
