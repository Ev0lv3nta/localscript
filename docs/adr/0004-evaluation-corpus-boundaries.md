# ADR 0004: границы evaluation-корпусов

- Статус: заменено [ADR 0008](0008-compact-live-evaluation.md)
- Дата: 9 августа 2026 года

## Контекст

Исторические 140 eval-кейсов использовались одновременно как regression fixtures, демонстрация и quality gate. Часть формулировок повторялась между наборами и была близка к few-shot knowledge base. Поэтому общий процент прохождения нельзя честно интерпретировать как качество на независимых задачах.

## Решение

Evaluation разделяется на три физически и семантически разные области:

- `evals/regression` — все 140 исторических кейсов; они защищают уже известное поведение, но не подтверждают обобщение;
- `evals/public/v1.jsonl` — новый публичный синтетический benchmark без `expected_code` и `reference_code`; его результат можно воспроизвести, но после публикации он перестаёт быть holdout для будущих версий;
- `holdout_v1` — внешний закрытый файл вне продуктового Git; публичный manifest содержит только SHA-256 и число кейсов.

Machine-readable manifest независимо задаёт corpus, runner, gate и допустимую область утверждений (`claim_scope`). Required check проверяет schema, уникальность полных входов и exact/normalized/fuzzy пересечения public/holdout с regression и knowledge base. Изменение eval-файла без синхронного manifest/evidence становится видимым в CI.

Первый результат закрытого holdout фиксируется без настройки pipeline по найденным ошибкам. Содержимое, абсолютный путь и сырые prompts закрытого набора не входят в публичный evidence.

## Последствия

Regression pass rate публикуется только с явной пометкой `regression`. Публичный benchmark и private holdout получают отдельные абсолютные результаты и dataset hashes. Release gate обязан проверить identity внешнего holdout до дорогого GPU-прогона. Любой новый публичный корпус должен пройти тот же overlap gate и получить новый versioned identifier.
