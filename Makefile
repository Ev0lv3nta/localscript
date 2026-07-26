.RECIPEPREFIX := >

UV ?= uv

.PHONY: install lock-check test smoke run

install:
>$(UV) sync --frozen --all-extras

lock-check:
>$(UV) lock --check

test:
>.venv/bin/python -m pytest -q

smoke:
>./scripts/judge_smoke.sh

run:
>./scripts/judge_up.sh
