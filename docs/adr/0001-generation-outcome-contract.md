# ADR 0001: типизированный результат генерации

- Статус: принято
- Дата: 27 июля 2026 года

## Контекст

Текущий pipeline хранит результат генерации в наборе строковых полей. Статус вычисляется отдельно от validation report, поэтому ответ может одновременно содержать `completed`, ошибки проверки и отклонённый Lua-код. Отсутствие обязательного runtime также представляется как degraded success.

Этот контракт нужен до изменения generation, validation и HTTP-слоёв: независимые изменения должны одинаково понимать успех, неполную проверку и отказ.

## Решение

Доменный модуль `app.domain.outcomes` вводит:

- `GenerationStatus`: `completed`, `clarification_required`, `validation_failed`, `policy_rejected`, `backend_unavailable`;
- `ValidationStatus`: `not_run`, `passed`, `failed`, `incomplete`;
- структурированный `Diagnostic`;
- неизменяемые `ValidationOutcome` и `GenerationOutcome`.

`completed` допустим только при непустом коде и полностью пройденной проверке. Любой другой generation status запрещает публикацию кода через публичное поле `code`. Отклонённый candidate может оставаться во внутреннем trace для диагностики, но не должен попадать в обычный API response.

`incomplete` означает, что обязательная проверка не могла завершиться, например из-за отсутствия Lua runtime. Это не доказательство ошибки в коде, но и не основание объявлять генерацию успешной.

## Границы внедрения

Контракт внедряется в три последовательных изменения:

1. generation core преобразует validation errors и incomplete validation в неуспешный outcome;
2. validation core становится total function для любого входа и возвращает структурированную ошибку вместо исключения;
3. HTTP adapter перестаёт публиковать rejected candidate, сохраняя успешный code-only ответ совместимого `/generate`.

Для rich API `validation_failed` остаётся доменным ответом с HTTP 200 и `code: null`. Совместимый `/generate`, который не может выразить typed outcome в своём успешном schema, отвечает HTTP 422. Backend outage продолжает отвечать HTTP 503.

Legacy-значения `clarification_needed`, `degraded_completed` и `failed_safe` не входят в новый enum. Переходный adapter преобразует их на границе старого результата, пока `GenerationResult` не будет удалён.

## Последствия

Инварианты проверяются при создании outcome, поэтому противоречивое состояние нельзя случайно передать дальше по pipeline. При этом этот ADR не меняет внешний API сам по себе: ожидаемые дефекты зафиксированы строгими `xfail`-тестами, которые станут обязательными regression tests в следующих изменениях.
