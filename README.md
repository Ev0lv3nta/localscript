# LocalScript

[![CI](https://github.com/Ev0lv3nta/localscript/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Ev0lv3nta/localscript/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

LocalScript превращает описание небольшого преобразования данных в Lua-код. Перед выдачей он запускает код на переданном JSON-контексте и проверяет результат. Работает с локальной моделью через Ollama; при недостатке данных запрашивает уточнение.

Подходит для нормализации строк, обработки массивов, агрегации и построения объектов в окружении `wf.vars` / `wf.initVariables`. Есть обычный Lua-блок и JSON-конверт с несколькими именованными Lua-блоками. Это локальный инструмент одного пользователя, не универсальная IDE.

**Статус:** доработка 0.3.0 до выпуска. Новый живой прогон после исправлений ещё не выполнен. Предыдущая проверка нашла ошибки; результаты сохранены ниже.

## Пример

Входной контекст:

```json
{"wf":{"vars":{"amounts":[10,20,5]}}}
```

Задача: «Сложи значения wf.vars.amounts; для пустого массива верни 0».

Один из корректных вариантов такого преобразования:

```lua
local total = 0
for _, amount in ipairs(wf.vars.amounts or {}) do
  total = total + amount
end
return total
```

На этом входе результат — `35`. В интерфейсе отдельно показываются код, фактическое значение и объём проверки. Пробный запуск не доказывает правильность на всех возможных входах; для важных условий можно передать свои examples с ожидаемыми результатами.

## Запуск

Базовый профиль использует `Qwen3.8-27B` UD-Q4_K_M. Предыдущий GPU-прогон выполнялся на RTX 3090 24 ГБ; веса и runtime занимали около 17 ГБ видеопамяти. Эти измерения относятся к предыдущей ревизии. Для новой версии скорость и качество ещё предстоит измерить.

Нужны Python 3.11/3.12, uv, Lua 5.4 и Ollama 0.33.3. Для Compose — Docker с NVIDIA Container Toolkit.

### Docker Compose

Первоначальная подготовка — отдельно от обычного запуска:

```bash
git clone https://github.com/Ev0lv3nta/localscript.git
cd localscript
docker compose up -d ollama
docker compose exec ollama ollama pull hf.co/unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M
docker compose up --build -d localscript
```

UI: <http://127.0.0.1:8080/>. API: <http://127.0.0.1:8080/docs>.

После установки достаточно `docker compose up -d`. Остановка — `docker compose down`. Модели и сессии находятся в отдельных named volumes и переживают пересоздание контейнеров. `down -v` удаляет эти данные; для обычной остановки он не нужен.

Ollama не публикует порт на хост, API доступен через loopback. В Compose отключены cloud features Ollama.

### Без контейнера

Запустите отдельный локальный Ollama с `OLLAMA_NO_CLOUD=1`, затем:

```bash
uv sync --frozen --all-extras --python 3.12
make lua-bootstrap
make model-setup
make run
```

`model-setup` скачивает выбранную модель. Обычный запуск ничего не скачивает и не требует запасной модели. При недоступном backend или неправильном окружении `localscript doctor` завершится с ненулевым кодом.

Настройки передаются через окружение; список — в [.env.example](.env.example). Канонический профиль — [local.yaml](app/resources/config/profiles/local.yaml). Другая модель требует своей проверки качества: изменение имени не переносит на неё результаты baseline.

Для удалённой GPU используйте SSH-туннель к loopback API. Не открывайте Ollama и неаутентифицированный интерфейс в интернет.

## API и CLI

```bash
curl -sS http://127.0.0.1:8080/api/generate \
  -H 'Content-Type: application/json' \
  --data '{"prompt":"Сложи wf.vars.amounts; для пустого массива верни 0","context":{"wf":{"vars":{"amounts":[10,20,5]}}}}'
```

Ответ содержит status, идентификаторы session/trace, output contract и validation. Code есть только у completed, question — только при уточнении.

При генерации можно передать `output` и до трёх `examples` с полями `name`, `context`, `expected`. Это требования пользователя; planner не вправе их менять. Независимые evaluation-ожидания в prompt не передаются.

`source_roots` явно выбирает `wf.vars`, `wf.initVariables` или оба корня. При совпадающих путях сервис запрашивает этот выбор, если код обращается к workflow-данным. В CLI для обоих корней повторите `--source-root`; в UI есть соответствующий список. Подтверждённый выбор нельзя менять внутри сессии.

`POST /api/validate` принимает уже готовые code, context, output и выполняет проверку без модели:

```bash
curl -sS http://127.0.0.1:8080/api/validate \
  -H 'Content-Type: application/json' \
  --data '{"code":"return wf.vars.value","context":{"wf":{"vars":{"value":4}}},"output":{"format":"lua_block","shape":"scalar","nullable":false}}'
```

CLI использует то же ядро:

```bash
.venv/bin/localscript generate --prompt-file task.txt --context-file context.json
.venv/bin/localscript generate --session-id ID --source-root wf.vars
.venv/bin/localscript generate --session-id ID --feedback 'Сохрани исходный порядок'
```

Новый prompt начинает новую сессию. Существующая задача меняется через feedback; передача нового prompt с прежним session_id отклоняется. Подробности — в [сценариях использования](docs/demo.md) и OpenAPI.

## Как устроено

`planner → generator → validation → reviewer → не более одной code revision`

Модель возвращает структурированные ответы. Validator проверяет формат, Tree-sitter AST, синтаксис через luac, выполняет код на реальном контексте и сравнивает результаты на предоставленных примерах. Reviewer работает в новом контексте, но на той же модели, поэтому остаётся эвристикой.

Нет роутера по формулировкам задач, канонических Lua-ответов и строкового «ремонта» кода. [Архитектура](docs/architecture.md) описывает роли, сессии и ограничения времени.

## Ограничения и данные

- Поддерживаются строки, bool, конечные числа в пределах ±(2^53−1), объекты со строковыми ключами и плотные массивы.
- Вложенный JSON null отклоняется явно. Верхнеуровневый Lua nil допустим при nullable=true.
- Sparse/mixed tables, циклы, функции и результаты больше 64 КиБ отклоняются.
- Lua string.lower/upper не обеспечивают Unicode-регистр. Совместимость helpers с внешними workflow-платформами отдельно не подтверждена.
- Проверка subprocess не является production sandbox для публичного исполнения чужих программ.
- Сессии сохраняют исходные данные локально: по умолчанию 100 сессий / 7 дней. Трассировки не содержат prompt, context или код. Уточнения и правки ограничены десятью ходами без молчаливого удаления ранних условий.

Подробнее: [безопасность](docs/security.md), [сообщить о проблеме](SECURITY.md).

## Оценка качества

Исторический прогон от 5 сентября: знакомые сценарии — **6/6**, synthetic blind — **3/8**, **один ошибочный completed**, итоговый gate не пройден. [Исходный отчёт](docs/evidence/release-gate-3cbd432.json) сохранён без изменений.

Для исправленной версии подготовлен отдельный публичный regression corpus. Он проверяет заданную область продукта; результат на нём не следует выдавать за независимый benchmark или гарантию обобщения. Новых GPU-цифр пока нет.

[Методика и команды проверки](docs/evaluation.md) фиксируют корпус, критерии, модель, SHA и способ измерения. Релиз создаётся только после нового успешного прогона.

## Разработка

```bash
make check
make container-check
```

CI проверяет Python 3.11/3.12, mypy --strict для app, Ruff, runtime-тесты, пакет, контейнер, зависимости, лицензии и секреты. Изменения идут через PR с обязательным CI. [Руководство разработчика](docs/development.md), [правила PR](CONTRIBUTING.md).

Проект вырос из личного хакатонного прототипа. [Происхождение](docs/history/hackathon-origin.md). Автор — Николай Никитенко. Код распространяется под MIT; лицензия модели указана в её карточке.
