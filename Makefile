.RECIPEPREFIX := >

UV ?= uv

.PHONY: install lock-check test build package-check smoke run

install:
>$(UV) sync --frozen --all-extras

lock-check:
>$(UV) lock --check

test:
>.venv/bin/python -m pytest -q

build:
>$(UV) build --clear

package-check: build
>.venv/bin/python scripts/check_package_artifacts.py --dist-dir dist

smoke:
>./scripts/judge_smoke.sh

run:
>./scripts/judge_up.sh
