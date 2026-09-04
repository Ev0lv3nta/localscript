from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from importlib import resources as importlib_resources
from importlib.resources.abc import Traversable
from pathlib import Path, PurePosixPath

RESOURCE_PACKAGE = "app.resources"


def _resource_parts(relative_path: Path | str) -> tuple[str, ...]:
    raw_path = str(relative_path)
    resource_path = PurePosixPath(raw_path)
    if (
        not raw_path
        or "\\" in raw_path
        or resource_path.is_absolute()
        or ".." in resource_path.parts
    ):
        raise ValueError("Resource path must be a safe package-relative path.")
    return resource_path.parts


def get_resource(relative_path: Path | str) -> Traversable:
    return importlib_resources.files(RESOURCE_PACKAGE).joinpath(*_resource_parts(relative_path))


def resource_exists(relative_path: Path | str) -> bool:
    return get_resource(relative_path).is_file()


def read_resource_text(relative_path: Path | str) -> str:
    resource = get_resource(relative_path)
    if not resource.is_file():
        raise FileNotFoundError(f"Packaged resource not found: {relative_path}")
    return resource.read_text(encoding="utf-8")


@contextmanager
def materialized_resource(relative_path: Path | str) -> Iterator[Path]:
    resource = get_resource(relative_path)
    if not resource.is_file():
        raise FileNotFoundError(f"Packaged resource not found: {relative_path}")
    with importlib_resources.as_file(resource) as path:
        yield path
