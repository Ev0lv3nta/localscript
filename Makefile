UV ?= uv

.PHONY: install lua-bootstrap lock-check license-check policy-check test test-unit \
	build package-check build-check container-check check smoke run

install:
	$(UV) sync --frozen --all-extras

lua-bootstrap:
	./scripts/bootstrap_lua54.sh

lock-check:
	$(UV) lock --check

license-check:
	.venv/bin/python scripts/check_licenses.py --lock uv.lock --notices THIRD_PARTY_NOTICES.md

policy-check: lock-check license-check

test: test-unit

test-unit: lua-bootstrap
	LOCALSCRIPT_UI_ENABLED=1 .venv/bin/python -m pytest -q -m unit

build:
	$(UV) build --clear

package-check: build
	.venv/bin/python scripts/check_package_artifacts.py --dist-dir dist

build-check: lock-check package-check

container-check:
	docker build --tag localscript:ci .
	./scripts/check_container.sh localscript:ci

check: install policy-check test-unit build-check

smoke:
	./scripts/judge_smoke.sh

run:
	./scripts/judge_up.sh
