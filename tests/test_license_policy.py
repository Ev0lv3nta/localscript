import subprocess
import sys
from pathlib import Path

from scripts import check_licenses


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_locked_dependency_licenses_use_explicit_spdx_allowlist():
    lock_packages = check_licenses.load_lock(PROJECT_ROOT / "uv.lock")

    assert set(check_licenses.LOCKED_LICENSES) == set(lock_packages) - {"localscript"}
    assert set(check_licenses.LOCKED_LICENSES.values()) <= check_licenses.ALLOWED_SPDX


def test_license_scanner_accepts_current_frozen_environment():
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/check_licenses.py",
            "--lock",
            "uv.lock",
            "--notices",
            "THIRD_PARTY_NOTICES.md",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.startswith("license_check_ok::packages=")


def test_vendored_lua_is_covered_by_notices_and_hash():
    check_licenses.check_notices(PROJECT_ROOT, PROJECT_ROOT / "THIRD_PARTY_NOTICES.md")
