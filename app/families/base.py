from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable


UNSUPPORTED = object()

ExpectedResultBuilder = Callable[[Mapping[str, Any], Any], Any]
StructuralValidator = Callable[
    [str, str, Mapping[str, Any]],
    tuple["FamilyFinding", ...],
]


@dataclass(frozen=True)
class FamilyFinding:
    code: str
    message: str


@runtime_checkable
class FamilyModule(Protocol):
    name: str
    preferred_return_shape: Optional[str]

    def build_expected_result(self, hints: Mapping[str, Any], context: Any) -> Any: ...

    def validate_structure(
        self,
        code: str,
        output_style: str,
        hints: Mapping[str, Any],
    ) -> tuple[FamilyFinding, ...]: ...


@dataclass(frozen=True)
class FamilyDefinition:
    name: str
    preferred_return_shape: Optional[str] = None
    expected_result_builder: Optional[ExpectedResultBuilder] = None
    structural_validator: Optional[StructuralValidator] = None

    def build_expected_result(self, hints: Mapping[str, Any], context: Any) -> Any:
        if self.expected_result_builder is None:
            return UNSUPPORTED
        return self.expected_result_builder(hints, context)

    def validate_structure(
        self,
        code: str,
        output_style: str,
        hints: Mapping[str, Any],
    ) -> tuple[FamilyFinding, ...]:
        if self.structural_validator is None:
            return ()
        return self.structural_validator(code, output_style, hints)
