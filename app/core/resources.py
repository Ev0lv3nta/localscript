from contextlib import contextmanager
from importlib import resources as importlib_resources
from pathlib import PurePosixPath

RESOURCE_PACKAGE = "app.resources"


def _resource_parts(relative_path):
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


def get_resource(relative_path):
    return importlib_resources.files(RESOURCE_PACKAGE).joinpath(
        *_resource_parts(relative_path)
    )


def resource_exists(relative_path):
    return get_resource(relative_path).is_file()


def read_resource_text(relative_path):
    resource = get_resource(relative_path)
    if not resource.is_file():
        raise FileNotFoundError("Packaged resource not found: {0}".format(relative_path))
    return resource.read_text(encoding="utf-8")


@contextmanager
def materialized_resource(relative_path):
    resource = get_resource(relative_path)
    if not resource.is_file():
        raise FileNotFoundError("Packaged resource not found: {0}".format(relative_path))
    with importlib_resources.as_file(resource) as path:
        yield path
