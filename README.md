# LocalScript

[![CI](https://github.com/Ev0lv3nta/localscript/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Ev0lv3nta/localscript/actions/workflows/ci.yml)
[![Python 3.11 | 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Локальный сервис, который генерирует небольшие Lua-скрипты через Ollama, проверяет их и явно отделяет успешный результат от уточнения или ошибки.

![Интерфейс LocalScript](docs/assets/localscript-ui.jpg)

LocalScript предназначен для задач преобразования данных в окружении workflow: прочитать значения из `wf.vars` или `wf.initVariables`, отфильтровать массив, нормализовать строку, собрать объект или вернуть JSON envelope с Lua-блоками. Модель предлагает решение, а детерминированный pipeline проверяет форму ответа, ограничения среды, синтаксис Lua 5.4 и доступные семантические свойства.

Главный контракт проекта: **непроверенный код не возвращается как успешная генерация**. Если данных недостаточно, сервис задаёт один вопрос. Если модель недоступна или проверка не пройдена после ограниченного repair-цикла, API возвращает явный неуспешный исход без поля `code`.

## Зачем это нужно

Обычная генерация кода заканчивается текстом модели. LocalScript добавляет вокруг неё управляемый локальный контур:

- запрос и контекст остаются на машине с Ollama после загрузки модели;
- planner формирует типизированное описание задачи;
- неоднозначный root приводит к уточнению, а не к молчаливой догадке;
- кандидат проверяется несколькими независимыми слоями;
- repair ограничен двумя раундами и не маскирует неуспех;
- сессия, проверка и безопасная трассировка доступны через API и UI;
- один application service используется из HTTP API, CLI и браузерного интерфейса.

Это исследовательский локальный инструмент, а не среда исполнения недоверенного кода и не универсальный генератор Lua. Поддерживаемая область и ограничения описаны ниже.

## Быстрый запуск через Docker Compose

Понадобятся Docker с Compose v2, доступная NVIDIA GPU и настроенный NVIDIA Container Toolkit. Образы привязаны к конкретным версиям, сервис публикуется только на loopback-интерфейсе.

```bash
git clone https://github.com/Ev0lv3nta/localscript.git
cd localscript
docker compose up -d ollama
docker compose exec ollama ollama pull hf.co/unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M
docker compose exec ollama ollama pull qwen3:8b-q4_K_M
docker compose up --build localscript
```

После готовности контейнеров:

- UI: <http://127.0.0.1:8080/>
- Swagger: <http://127.0.0.1:8080/docs>
- readiness: <http://127.0.0.1:8080/ready>

Остановка:

```bash
docker compose down
```

Модель хранится в `artifacts/ollama_home` и не коммитится. Первый `pull` загружает несколько гигабайт. Для другого порта задайте `LOCALSCRIPT_PORT` перед `docker compose up`.

## Локальный запуск без контейнера

Поддерживаются Python 3.11 и 3.12. Нужны `uv`, Ollama версии 0.33 или новее и модель `Qwen3.8-27B` в кванте UD-Q4_K_M — около 17 ГБ видеопамяти.

```bash
uv sync --frozen --all-extras --python 3.12
./scripts/bootstrap_lua54.sh
ollama pull hf.co/unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M
ollama pull qwen3:8b-q4_K_M
make run
```

`make run` проверяет версию Python, доступность Ollama, наличие primary и fallback tags, политику bind-адреса и свободный порт. По умолчанию сервис слушает только `127.0.0.1:8080`, UI включён.

Ключевые настройки можно передать переменными окружения:

```bash
LOCALSCRIPT_PORT=8090 \
LOCALSCRIPT_PRIMARY_MODEL=qwen3:8b-q4_K_M \
LOCALSCRIPT_FALLBACK_MODEL=qwen3:4b-instruct-2507-q4_K_M \
make run
```

Полный список безопасных defaults находится в [`competition.yaml`](app/resources/config/profiles/competition.yaml) и [`.env.example`](.env.example).

## HTTP API

Генерация — один endpoint:

```bash
curl -s http://127.0.0.1:8080/api/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Нормализуй wf.vars.email и верни строку в нижнем регистре",
    "context": {"wf": {"vars": {"email": "USER@EXAMPLE.COM"}}}
  }'
```

Ответ типизирован: `status`, `session_id`, `trace_id`, `diagnostics`, `validation` и число правок. Поле `code` появляется только у `completed`, `question` — только у `clarification_required`; собрать противоречивый ответ нельзя, это запрещает сама модель ответа.

```json
{
  "status": "completed",
  "session_id": "<uuid>",
  "trace_id": "<uuid>",
  "code": "local value = wf.vars.email or \"\"\nreturn string.lower(value)",
  "question": null,
  "diagnostics": [],
  "revision_count": 0
}
```

Возможные исходы:

| Статус | Значение | Публикуется `code` |
|---|---|---:|
| `completed` | Все обязательные проверки пройдены | да |
| `clarification_required` | Нужен один ответ пользователя | нет |
| `validation_failed` | Кандидат не прошёл проверку | нет |
| `policy_rejected` | Запрос выходит за разрешённую границу | нет |
| `backend_unavailable` | Ollama или модель недоступны | нет |

Продолжение уточнения отправляется в ту же сессию:

```json
{
  "session_id": "<uuid>",
  "clarification_answer": "Используй wf.vars.email"
}
```

Самостоятельная проверка готового кода доступна через `POST /api/validate`: он принимает код, контекст и явный контракт вывода — формат, форму результата и допустимость `nil`. Полная и всегда актуальная схема — в Swagger и `/openapi.json`.

## CLI

После `uv sync` команда `localscript` доступна из виртуального окружения:

```bash
.venv/bin/localscript generate --prompt "Верни последний элемент wf.vars.items"
.venv/bin/localscript generate --session-id <uuid> --answer "Используй wf.vars"
.venv/bin/localscript validate --code-file example.lua
.venv/bin/localscript doctor
```

`generate` начинает сессию по `--prompt` и продолжает её по `--session-id` с `--answer` или `--feedback`; идентификатор сессии печатается в ответе. `doctor --judge` — дорогая GPU-проверка выбранной модели; это релизный инструмент, а не обычная healthcheck-команда.

## Как устроена генерация

```mermaid
flowchart LR
    C["CLI · HTTP API · UI"] --> A["Application service"]
    A --> P["Planner"]
    P -->|неоднозначно| Q["Уточнение"]
    P -->|план с acceptance cases| G["Generator"]
    G --> V["AST-policy · luac · ограниченный runtime"]
    V -->|проверено| R["Reviewer"]
    V -->|не пройдено| REV["Одна revision"]
    R -->|отклонено| REV
    REV --> V
    R -->|одобрено| O["completed + code"]
    V -->|повторный отказ| F["явный неуспешный исход"]
```

Основные границы ответственности:

- `app/workflow` — контракты плана, роли planner/generator/reviewer и координатор стадий;
- `app/generation` — обращения к Ollama, разрешение модели и типизированные ошибки backend;
- `app/validation` — Lua AST-policy, `luac` и ограниченный runtime;
- `app/api`, `app/cli`, `app/ui` — транспортные адаптеры без собственной бизнес-логики;
- `app/evaluation` — целостность корпуса, метрики и стабильность.

Подробнее: [архитектура](docs/architecture.md) и принятые [ADR](docs/adr/).

## Что именно проверяется

Pipeline последовательно проверяет:

1. тип и размер входа, глубину и число узлов контекста;
2. форму Lua block или JSON envelope;
3. запрещённые roots, глобальные мутации и неподдерживаемые конструкции по AST, а не по строкам;
4. синтаксис через `luac`;
5. выполнение acceptance cases плана в ограниченном Lua-subprocess с лимитами;
6. совпадение результата с ожидаемым JSON и объявленной формой вывода;
7. согласованность финального typed outcome.

Проверки не доказывают корректность произвольной программы: они проверяют кандидата ровно на тех acceptance cases, которые planner вывел из запроса. Результат стоит считать проверенным в рамках заявленного pipeline, а не математически доказанным.

## Оценка качества

Живых контура два:

- `live-v1` — 6 сценариев в репозитории, по одному на каждую заявленную форму задачи: скалярное преобразование, фильтрация с проекцией, агрегация, вложенный объект, JSON-конверт и уточнение;
- synthetic blind holdout — 8 замороженных кейсов вне Git, из них 2 по безопасности; запускаются один раз перед релизом.

Безопасность в живой корпус не входит: её проверяют детерминированные тесты AST-политики и ограниченного runtime, без обращений к модели.

Integrity check ищет точные, нормализованные и нечёткие пересечения слепого набора с живым корпусом. Release gate фиксирует SHA коммита, хеш корпуса, версии Python/Lua/Ollama, digest модели, GPU, метрики стадий, стабильность трёх сценариев в двух прогонах и укладывается в 15 минут. Живой корпус требует не менее 5 из 6 при нуле невалидных успехов, слепой набор — не менее 7 из 8 и все пройденные кейсы безопасности. Пороги объявлены в манифесте корпусов и меняются только через ревью.

Результаты `v0.3.0` публикуются как JSON-артефакт GitHub Release. Они подтверждают работу конкретной ревизии в зафиксированном окружении и не являются заявлением об обобщающей способности на любые Lua-задачи. Методика: [docs/evaluation.md](docs/evaluation.md).

## Разработка

Основная локальная проверка совпадает с CI:

```bash
make check
```

Она устанавливает зависимости из lock-файла, запускает Ruff и mypy, проверяет лицензии и целостность eval-корпусов, подготавливает Lua, выполняет unit suite и проверяет wheel/sdist из чистого окружения.

Отдельные команды:

```bash
make quality-check
make test-unit
make build-check
make container-check
```

CI работает на Python 3.11 и 3.12. Required aggregator становится зелёным только после unit, static analysis, package, container, policy, eval integrity и full-history secret scan. Правила участия и структура PR описаны в [CONTRIBUTING.md](CONTRIBUTING.md), локальная среда — в [docs/development.md](docs/development.md).

## Ограничения и безопасность

- Это не production sandbox. Lua выполняется в отдельном subprocess с лимитами, но без VM-изоляции уровня gVisor или Firecracker.
- Сервис по умолчанию доступен только через loopback. Внешний bind разрешается лишь при явном remote mode и bearer token длиной не менее 32 символов.
- Локальность относится к inference после загрузки модели. Docker/Ollama и модель нужно получить из внешних registry один раз.
- Трассировки редактируют код и приватные model artifacts, но prompt и рабочий контекст всё равно следует считать чувствительными локальными данными.
- Поддерживается ограниченный диалект LocalScript/Lua и известные workflow-roots, а не произвольная системная автоматизация.
- При отсутствии Lua runtime семантическая проверка не объявляется полной.

Полная модель угроз, сетевые defaults и порядок сообщения об уязвимостях: [SECURITY.md](SECURITY.md) и [docs/security.md](docs/security.md).

## Происхождение

Проект начался как личный прототип для задачи MWS Octapi на MTS True Tech Hack 2026. Автор — Николай Никитенко; соавторов нет, проект не занял призового места. Репозиторий не является официальным продуктом или проектом МТС/MWS.

Хакатонный снимок сохранён отдельно и не переписывается. Новый репозиторий получил чистую документированную историю: курируемый импорт, небольшие PR, обязательный CI и воспроизводимый release gate. Подробнее — в [истории проекта](docs/history/hackathon-origin.md) и [манифесте импорта](docs/history/import-manifest.md).

## Лицензия

Код распространяется по лицензии [MIT](LICENSE). Лицензии runtime-зависимостей и vendored Lua перечислены в [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

История релизов: [CHANGELOG.md](CHANGELOG.md). Демонстрационный сценарий: [docs/demo.md](docs/demo.md).
