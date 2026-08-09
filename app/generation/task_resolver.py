from app.families import is_known_family
from app.generation.taskspec import (
    ResolvedTaskSpec,
    TaskResolutionSource,
    TaskSpec,
)


class TaskResolver:
    """Resolve extractor and planner evidence into one immutable task spec."""

    ROOTS = frozenset({"wf.vars", "wf.initVariables"})

    def resolve(self, candidate: TaskSpec, planner=None) -> ResolvedTaskSpec:
        if isinstance(candidate, ResolvedTaskSpec):
            return candidate
        planner = planner if isinstance(planner, dict) else {}
        planner_family = self._non_empty_string(planner.get("family"))
        registered_planner_family = (
            planner_family if is_known_family(planner_family) else None
        )

        if candidate.family and is_known_family(candidate.family):
            family = candidate.family
            source = TaskResolutionSource.EXTRACTOR
        elif registered_planner_family:
            family = registered_planner_family
            source = TaskResolutionSource.PLANNER
        else:
            family = "generic_lua"
            source = TaskResolutionSource.GENERIC

        target_root = candidate.target_root
        planner_root = self._non_empty_string(planner.get("root"))
        if target_root in {"unknown", "unknown_mixed"} and planner_root in self.ROOTS:
            target_root = planner_root

        return ResolvedTaskSpec(
            **candidate.model_dump(
                exclude={
                    "family",
                    "target_root",
                    "resolution_source",
                    "planner_family",
                },
            ),
            family=family,
            target_root=target_root,
            resolution_source=source,
            planner_family=planner_family,
        )

    @staticmethod
    def _non_empty_string(value):
        if not isinstance(value, str):
            return None
        normalized = value.strip()
        return normalized or None
