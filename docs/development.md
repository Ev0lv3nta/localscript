# Разработка

## Окружение

Поддерживаются CPython 3.11 и 3.12. Канонический менеджер зависимостей — `uv 0.11.21`; lock-файл обязателен. Для полной проверки нужны Docker и локально собираемый Lua 5.4.6.

```bash
uv sync --frozen --all-extras --python 3.12
./scripts/bootstrap_lua54.sh
make check
```

Writable runtime state не должен попадать в checkout. Для изолированного прогона:

```bash
export LOCALSCRIPT_STATE_DIR="$(mktemp -d)"
```

## Проверки

| Команда | Назначение |
|---|---|
| `make quality-check` | Ruff и mypy на публичных typed boundaries |
| `make policy-check` | неизменность lock и allowlist лицензий |
| `make eval-integrity` | schema, provenance и отсутствие overlap |
| `make test-unit` | герметичные unit/contract/characterization tests с Lua |
| `make build-check` | wheel/sdist manifest и smoke из пустого каталога |
| `make container-check` | сборка и runtime smoke non-root контейнера |
| `make check` | все локальные проверки, кроме контейнера и live GPU |

Integration tests запускаются отдельно и требуют Ollama:

```bash
LOCALSCRIPT_REQUIRE_LIVE=1 .venv/bin/python -m pytest -q -m integration --strict-markers
```

## Правило изменения контрактов

Изменение `GenerationOutcome`, контрактов workflow, границ валидации или evaluation claim сначала описывается в ADR. Затем добавляется contract test, реализация и негативный тест fail-closed поведения.

Новая возможность генерации не добавляется веткой в Python: она выражается через план, который planner способен построить, и через acceptance cases, которые валидация умеет исполнить. Если для запроса нужен новый Python-путь, это признак того, что меняется сам контракт, и решение идёт через ADR.

## Git workflow

Работа ведётся короткими ветками от `main`: `feat/...`, `fix/...`, `refactor/...`, `docs/...`, `release/...`. Один PR решает одну проверяемую задачу. Merge выполняется squash после зелёного `CI / required`; прямые push и force push в `main` запрещаются ruleset.

Не добавляйте generated state, model weights, benchmark artifacts и private holdout в Git. Перед push проверяйте `git diff --check` и `git status`.

## Релиз

Release candidate должен быть чистым commit из истории `main`. Full gate запускается владельцем на GPU-машине с внешним holdout:

```bash
LOCALSCRIPT_STATE_DIR=/safe/external/state \
LOCALSCRIPT_PRIVATE_HOLDOUT_PATH=/safe/external/holdout-v1.jsonl \
.venv/bin/python scripts/release_gate.py \
  --mode competition \
  --output /safe/external/release-gate.json
```

Tag и GitHub Release создаются только если report содержит `ok: true`, пустой `failures`, точный SHA кандидата и `locked: true` runtime snapshot.
Закрытый holdout запускается последним, только после успешных публичных live-проверок. Gate повторно сверяет SHA-256 и число фактически обработанных кейсов с manifest; публичный JSON не содержит сырые результаты или локальные пути.
