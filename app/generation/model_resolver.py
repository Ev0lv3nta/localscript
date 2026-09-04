from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from app.generation.backend_errors import BackendModelError, BackendProtocolError


@dataclass(frozen=True)
class ResolvedModel:
    tag: str
    digest: str


@dataclass(frozen=True)
class ModelTag:
    tag: str
    digest: str
    details: Mapping[str, Any]


def parse_model_tags(payload: object) -> tuple[ModelTag, ...]:
    if not isinstance(payload, Mapping):
        raise BackendProtocolError(reason="invalid_tags_payload")

    raw_models = payload.get("models")
    if not isinstance(raw_models, list):
        raise BackendProtocolError(reason="invalid_tags_payload")

    parsed = []
    for item in raw_models:
        if not isinstance(item, Mapping):
            continue
        tag = item.get("name") or item.get("model")
        if not isinstance(tag, str) or not tag.strip():
            continue
        digest = item.get("digest")
        parsed.append(
            ModelTag(
                tag=tag.strip(),
                digest=digest.strip() if isinstance(digest, str) else "",
                details=(
                    dict(details) if isinstance(details := item.get("details"), Mapping) else {}
                ),
            )
        )
    return tuple(parsed)


def resolve_model(requested: str, available: Iterable[ModelTag]) -> ResolvedModel:
    requested = (requested or "").strip()
    if not requested:
        raise BackendModelError(reason="model_not_configured")

    tags = tuple(available)
    requested_tag, separator, requested_digest = requested.partition("@")
    exact = [item for item in tags if item.tag == requested_tag]
    if requested_digest:
        exact = [item for item in exact if _digest_matches(requested_digest, item.digest)]
    if len(exact) == 1:
        item = exact[0]
        return ResolvedModel(tag=item.tag, digest=item.digest)

    if not separator and ":" not in requested_tag:
        latest = [item for item in tags if item.tag == requested_tag + ":latest"]
        if len(latest) == 1:
            item = latest[0]
            return ResolvedModel(tag=item.tag, digest=item.digest)

        matching_tags = [item for item in tags if item.tag.split(":", 1)[0] == requested_tag]
        if len(matching_tags) == 1:
            item = matching_tags[0]
            return ResolvedModel(tag=item.tag, digest=item.digest)

    raise BackendModelError(reason="model_not_found")


def model_identities_match(expected: ResolvedModel, response_model: object) -> bool:
    if response_model is None:
        return True
    if not isinstance(response_model, str) or not response_model.strip():
        return False

    actual_tag, separator, actual_digest = response_model.strip().partition("@")
    if not _tags_match(expected.tag, actual_tag):
        return False
    return not separator or not expected.digest or _digest_matches(actual_digest, expected.digest)


def _tags_match(left: str, right: str) -> bool:
    if left == right:
        return True
    return left.removesuffix(":latest") == right.removesuffix(":latest")


def _digest_matches(left: str, right: str) -> bool:
    if not left or not right:
        return False
    return left == right or left.startswith(right) or right.startswith(left)
