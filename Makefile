.RECIPEPREFIX := >

.PHONY: install test smoke run

install:
>if [ ! -d .venv ]; then if ! python3 -m venv .venv; then python3 -m pip install --user virtualenv && python3 -m virtualenv .venv; fi; fi
>. .venv/bin/activate && pip install --upgrade pip
>. .venv/bin/activate && pip install -e .[dev]

test:
>. .venv/bin/activate && pytest -q

smoke:
>./scripts/judge_smoke.sh

run:
>./scripts/judge_up.sh
