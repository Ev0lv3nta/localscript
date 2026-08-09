from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskResolutionSource(str, Enum):
    EXTRACTOR = "extractor"
    PLANNER = "planner"
    GENERIC = "generic"


class TaskSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    normalized_prompt: str
    family: Optional[str] = None
    output_style: str = "lua_block"
    target_root: str = "unknown"
    context_paths: List[str] = Field(default_factory=list)
    prompt_paths: List[str] = Field(default_factory=list)
    root_candidates: List[str] = Field(default_factory=list)
    family_confidence: float = 0.0
    ambiguity_score: float = 0.0
    composition_score: float = 0.0
    context_density: float = 0.0
    generation_hints: Dict[str, object] = Field(default_factory=dict)
    assumptions: List[str] = Field(default_factory=list)
    ambiguity_notes: List[str] = Field(default_factory=list)
    safety_fallback: bool = False


class ResolvedTaskSpec(TaskSpec):
    family: str
    resolution_source: TaskResolutionSource
    planner_family: Optional[str] = None
