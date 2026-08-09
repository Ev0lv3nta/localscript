from types import MappingProxyType

from app.families.base import FamilyDefinition
from app.families.collections import COLLECTION_FAMILIES
from app.families.records import RECORD_FAMILIES
from app.families.scalars import SCALAR_FAMILIES

_NON_SEMANTIC_FAMILIES = (
    FamilyDefinition("generic_lua"),
    FamilyDefinition("safety_guard", preferred_return_shape="scalar"),
)


def _build_registry() -> MappingProxyType:
    definitions = (
        *COLLECTION_FAMILIES,
        *SCALAR_FAMILIES,
        *RECORD_FAMILIES,
        *_NON_SEMANTIC_FAMILIES,
    )
    registry = {}
    for definition in definitions:
        if definition.name in registry:
            raise ValueError("duplicate family definition: {0}".format(definition.name))
        registry[definition.name] = definition
    return MappingProxyType(registry)


_REGISTRY = _build_registry()


def get_family_definition(name: str | None) -> FamilyDefinition | None:
    return _REGISTRY.get(name)


def is_known_family(name: str | None) -> bool:
    return name in _REGISTRY


def all_family_definitions() -> tuple[FamilyDefinition, ...]:
    return tuple(_REGISTRY.values())
