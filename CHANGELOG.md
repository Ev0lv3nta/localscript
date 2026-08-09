# История изменений

Формат основан на Keep a Changelog, версии следуют Semantic Versioning.

## [0.2.0] — 2026-08-09

### Добавлено

- типизированные исходы генерации с fail-closed контрактом;
- единый task resolver, generation state machine и family registry;
- структурная, policy-, syntax-, runtime- и semantic-проверка Lua;
- ограниченный repair с отдельными каноническими и локальными действиями;
- безопасные session/trace stores, retention и runtime snapshot;
- расширенный HTTP API, CLI, локальный двухколоночный UI и clarification flow;
- публичный semantic benchmark, repeat stability, ablation и внешний private holdout gate;
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

[0.2.0]: https://github.com/Ev0lv3nta/localscript/releases/tag/v0.2.0
