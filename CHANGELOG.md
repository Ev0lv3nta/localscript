# История изменений

Формат основан на Keep a Changelog, версии следуют Semantic Versioning.

## [Не выпущено]

### Добавлено

- ролевой agentic workflow: planner, generator, детерминированная проверка, reviewer и не более одной правки;
- typed-контракты плана, кандидата, ревью и результата, запрещающие противоречивые состояния;
- Lua-политика по AST вместо строкового поиска и выполнение acceptance cases в ограниченном runtime;
- `mypy --strict` для всего `app/`, расширенный Ruff и проверка форматирования в CI.

### Изменено

- одна точка входа генерации `POST /api/generate` вместо трёх endpoint'ов, одна CLI-команда вместо трёх;
- UI показывает реальные стадии workflow вместо «способа» и «риска допущений»;
- трассировка хранит только стадии, длительности, модель и коды диагностик.

### Удалено

- rule-based `TaskExtractor`, `TaskResolver`, реестр семейств задач и канонические замены кода;
- legacy `POST /generate`, `/api/analyze` и дублирующий доменный слой исходов;
- отдельный latency-прогон: длительности стадий теперь метрика внутри общего прогона.

## [0.2.0] — 2026-08-10

### Добавлено

- типизированные исходы генерации с fail-closed контрактом;
- структурная, policy-, syntax-, runtime- и semantic-проверка Lua;
- безопасные session/trace stores, retention и runtime snapshot;
- HTTP API, CLI, локальный двухколоночный UI и clarification flow;
- публичный semantic benchmark, repeat stability и внешний private holdout gate;
- воспроизводимые wheel, sdist и non-root Docker image;
- CI для Python 3.11/3.12, статический анализ, supply-chain и secret checks;
- русская документация по архитектуре, evaluation, безопасности, разработке и demo.

### Изменено

- зависимости и build toolchain зафиксированы в `uv.lock`;
- writable state вынесен из checkout;
- backend Ollama получил typed errors, digest evidence, bounded concurrency и безопасную сетевую политику;
- исторический хакатонный прототип импортирован в новый репозиторий с чистой документированной историей.

### Исправлено

- невалидный или непроверенный код больше не может возвращаться как успешный результат;
- произвольный ввод `/validate` не приводит к необработанному исключению;
- отсутствие backend, модели, Lua или полной проверки не маскируется fallback-заглушкой;
- eval reference data не попадает в model prompt, а пересечения корпусов блокируются до benchmark.
- публичный release evidence не раскрывает путь, идентификаторы кейсов или сырые
  результаты закрытого holdout и сверяет хеш фактически проверенного набора.

[0.2.0]: https://github.com/Ev0lv3nta/localscript/releases/tag/v0.2.0
