#!/usr/bin/env python3
"""Проверка лицензий строго по uv.lock и установленному окружению."""

import argparse
import hashlib
import re
import sys
import tomllib
from importlib.metadata import distributions
from pathlib import Path

from packaging.markers import Marker

ALLOWED_SPDX = {
    "Apache-2.0 OR BSD-2-Clause",
    "Apache-2.0 OR MIT",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "ISC",
    "MIT",
    "MPL-2.0",
    "PSF-2.0",
}

# Любая новая транзитивная зависимость должна получить явное решение здесь.
LOCKED_LICENSES = {
    "annotated-doc": "MIT",
    "annotated-types": "MIT",
    "ast-serialize": "MIT",
    "anyio": "MIT",
    "certifi": "MPL-2.0",
    "click": "BSD-3-Clause",
    "colorama": "BSD-3-Clause",
    "fastapi": "MIT",
    "h11": "MIT",
    "httpcore": "BSD-3-Clause",
    "httptools": "MIT",
    "httpx": "BSD-3-Clause",
    "idna": "BSD-3-Clause",
    "iniconfig": "MIT",
    "librt": "MIT",
    "markdown-it-py": "MIT",
    "mdurl": "MIT",
    "mypy": "MIT",
    "mypy-extensions": "MIT",
    "packaging": "Apache-2.0 OR BSD-2-Clause",
    "pathspec": "MPL-2.0",
    "pluggy": "MIT",
    "pydantic": "MIT",
    "pydantic-core": "MIT",
    "pydantic-settings": "MIT",
    "pygments": "BSD-2-Clause",
    "pytest": "MIT",
    "python-dotenv": "BSD-3-Clause",
    "pyyaml": "MIT",
    "rich": "MIT",
    "ruff": "MIT",
    "shellingham": "ISC",
    "sniffio": "Apache-2.0 OR MIT",
    "starlette": "BSD-3-Clause",
    "typer": "MIT",
    "tree-sitter": "MIT",
    "tree-sitter-lua": "MIT",
    "typing-extensions": "PSF-2.0",
    "typing-inspection": "MIT",
    "uvicorn": "BSD-3-Clause",
    "uvloop": "MIT",
    "watchfiles": "MIT",
    "websockets": "BSD-3-Clause",
}

CLASSIFIER_TO_SPDX = {
    "License :: OSI Approved :: Apache Software License": "Apache-2.0",
    "License :: OSI Approved :: BSD License": "BSD-3-Clause",
    "License :: OSI Approved :: ISC License (ISCL)": "ISC",
    "License :: OSI Approved :: MIT License": "MIT",
    "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
    "License :: OSI Approved :: Python Software Foundation License": "PSF-2.0",
}

LICENSE_ALIASES = {
    "ISC License": "ISC",
    "MIT License": "MIT",
}

LUA_ARCHIVE = Path("third_party/lua/lua-5.4.6.tar.gz")
LUA_SHA256 = "7d5ea1b9cb6aa0b59ca3dde1c6adcb57ef83a1ba8e5432c0ecd06bf439b3ad88"


def normalize_name(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def normalize_expression(expression):
    expression = LICENSE_ALIASES.get(expression.strip(), expression.strip())
    if " OR " in expression:
        return " OR ".join(sorted(part.strip() for part in expression.split(" OR ")))
    return expression


def metadata_license(metadata):
    expression = metadata.get("License-Expression") or metadata.get("License")
    if expression:
        return normalize_expression(expression)

    classifiers = {
        CLASSIFIER_TO_SPDX[classifier]
        for classifier in metadata.get_all("Classifier", [])
        if classifier in CLASSIFIER_TO_SPDX
    }
    if classifiers:
        return " OR ".join(sorted(classifiers))
    return None


def load_lock(lock_path):
    with lock_path.open("rb") as handle:
        packages = tomllib.load(handle)["package"]
    return {normalize_name(package["name"]): package for package in packages}


def resolved_packages(lock_packages):
    root = lock_packages.get("localscript")
    if root is None:
        raise ValueError("lock_missing_project::localscript")

    pending = []
    for dependency in root.get("dependencies", []):
        pending.append((dependency, set(dependency.get("extra", []))))
    for dependencies in root.get("optional-dependencies", {}).values():
        pending.extend((dependency, set(dependency.get("extra", []))) for dependency in dependencies)

    resolved = set()
    while pending:
        dependency, extras = pending.pop()
        marker = dependency.get("marker")
        if marker and not Marker(marker).evaluate():
            continue
        name = normalize_name(dependency["name"])
        if name in resolved:
            continue
        resolved.add(name)
        package = lock_packages[name]
        pending.extend((child, set(child.get("extra", []))) for child in package.get("dependencies", []))
        for extra in extras:
            pending.extend(
                (child, set(child.get("extra", [])))
                for child in package.get("optional-dependencies", {}).get(extra, [])
            )
    return resolved


def check_notices(project_root, notices_path):
    notices = notices_path.read_text(encoding="utf-8")
    for required in (str(LUA_ARCHIVE), LUA_SHA256, "MIT"):
        if required not in notices:
            raise ValueError("third_party_notice_missing::{0}".format(required))

    archive_path = project_root / LUA_ARCHIVE
    archive_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    if archive_sha256 != LUA_SHA256:
        raise ValueError("vendored_lua_hash_mismatch::{0}".format(archive_sha256))


def check_licenses(lock_path, notices_path):
    lock_packages = load_lock(lock_path)
    locked_third_party = set(lock_packages) - {"localscript"}
    mapped = set(LOCKED_LICENSES)
    if locked_third_party != mapped:
        missing = sorted(locked_third_party - mapped)
        stale = sorted(mapped - locked_third_party)
        raise ValueError("license_map_lock_mismatch::missing={0}::stale={1}".format(missing, stale))

    disallowed = {
        package: license_expression
        for package, license_expression in LOCKED_LICENSES.items()
        if license_expression not in ALLOWED_SPDX
    }
    if disallowed:
        raise ValueError("spdx_not_allowed::{0}".format(disallowed))

    installed = {}
    for distribution in distributions():
        name = normalize_name(distribution.metadata["Name"])
        if name == "localscript":
            continue
        installed[name] = distribution

    expected_installed = resolved_packages(lock_packages)
    if set(installed) != expected_installed:
        missing = sorted(expected_installed - set(installed))
        unexpected = sorted(set(installed) - expected_installed)
        raise ValueError("installed_lock_mismatch::missing={0}::unexpected={1}".format(missing, unexpected))

    for name in sorted(installed):
        distribution = installed[name]
        locked_version = str(lock_packages[name]["version"])
        if distribution.version != locked_version:
            raise ValueError(
                "installed_version_mismatch::{0}::installed={1}::locked={2}".format(
                    name, distribution.version, locked_version
                )
            )
        declared = metadata_license(distribution.metadata)
        expected = normalize_expression(LOCKED_LICENSES[name])
        if declared != expected:
            raise ValueError(
                "license_metadata_mismatch::{0}::declared={1}::expected={2}".format(
                    name, declared, expected
                )
            )

    check_notices(lock_path.parent, notices_path)
    return len(installed)


def main():
    parser = argparse.ArgumentParser(description="Проверить locked-зависимости и SPDX-лицензии.")
    parser.add_argument("--lock", type=Path, default=Path("uv.lock"))
    parser.add_argument("--notices", type=Path, default=Path("THIRD_PARTY_NOTICES.md"))
    args = parser.parse_args()

    try:
        scanned = check_licenses(args.lock.resolve(), args.notices.resolve())
    except (KeyError, OSError, ValueError) as error:
        print("license_check_failed::{0}".format(error), file=sys.stderr)
        raise SystemExit(1) from error
    print("license_check_ok::packages={0}".format(scanned))


if __name__ == "__main__":
    main()
