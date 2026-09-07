from pathlib import Path

import pytest

from app.core.config import get_profile_path
from app.core.resources import (
    materialized_resource,
    read_resource_text,
    resource_exists,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_profile_loads_from_canonical_package_resources():
    assert get_profile_path().is_file()
    assert resource_exists("config/profiles/local.yaml")
    assert resource_exists("config/profiles/competition.yaml")


def test_resource_api_rejects_paths_outside_package():
    with pytest.raises(ValueError):
        read_resource_text("../pyproject.toml")


def test_profile_can_be_materialized_outside_source_layout():
    with materialized_resource("config/profiles/local.yaml") as path:
        assert "name: local" in path.read_text(encoding="utf-8")
