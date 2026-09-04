import os
import subprocess
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
    assert resource_exists("config/profiles/competition.yaml")


def test_resource_api_rejects_paths_outside_package():
    with pytest.raises(ValueError):
        read_resource_text("../pyproject.toml")


def test_packaged_script_is_materialized_within_context():
    with materialized_resource("scripts/bench_vram.sh") as script_path:
        assert script_path.is_file()
        assert script_path.read_text(encoding="utf-8").startswith("#!/usr/bin/env bash")


def test_source_vram_wrapper_resolves_supported_python_in_order():
    wrapper = (PROJECT_ROOT / "scripts" / "bench_vram.sh").read_text(encoding="utf-8")

    positions = [
        wrapper.index("LOCALSCRIPT_PYTHON_BIN"),
        wrapper.index("${PROJECT_ROOT}/.venv/bin/python"),
        wrapper.index("/opt/venv/bin/python"),
        wrapper.index("command -v python3"),
    ]

    assert positions == sorted(positions)


def test_source_vram_wrapper_uses_explicit_python(tmp_path):
    selected_python = tmp_path / "selected-python"
    capture_path = tmp_path / "selected.txt"
    selected_python.write_text(
        '#!/usr/bin/env sh\nprintf "%s\\n" "$0" > "$CAPTURE_PATH"\n',
        encoding="utf-8",
    )
    selected_python.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        {
            "CAPTURE_PATH": str(capture_path),
            "LOCALSCRIPT_PYTHON_BIN": str(selected_python),
        }
    )

    completed = subprocess.run(
        ["bash", str(PROJECT_ROOT / "scripts" / "bench_vram.sh")],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0
    assert capture_path.read_text(encoding="utf-8").strip() == str(selected_python)
