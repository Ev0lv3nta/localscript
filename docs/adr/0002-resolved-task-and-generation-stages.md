# ADR 0002: единая спецификация задачи и этапы генерации

- Статус: принято
- Дата: 9 августа 2026 года

## Контекст

Extractor и planner независимо определяют семейство задачи, а mutable `TaskSpec` передаётся в prompt, validation и repair. Из-за этого downstream-слои могут работать с разными представлениями одной задачи, а большой метод orchestration не показывает допустимые переходы между уточнением, генерацией, repair и финальным outcome.

## Решение

Extractor возвращает неизменяемый кандидат `TaskSpec`. После ответа planner `TaskResolver` один раз создаёт `ResolvedTaskSpec` по фиксированному приоритету:

1. уверенное семейство extractor;
2. семейство planner;
3. явное `generic_lua`.

Явно указанный root не перезаписывается planner. Planner может уточнить root только когда extractor оставил его неизвестным или смешанным. Полученный `ResolvedTaskSpec` используется writer, validation, repair и trace без повторной классификации.

Orchestration фиксирует переходы `session_ready → task_resolved → candidate_generated → candidate_repaired → outcome_finalized`. Ветка уточнения проходит через `clarification_required` и после этого не может генерировать candidate. Repair-этап необязателен, но после `outcome_finalized` переходы запрещены.

Результат model-chain преобразуется во внутренний тип `GeneratedCandidate` либо `PlannerClarification`. Генерация, deterministic repair и финализация session/trace разделены на самостоятельные операции; orchestration связывает их, но не хранит россыпь промежуточных переменных.

Family, предложенная planner, может выбирать prompt guidance и общие structural validators, но не активирует deterministic family oracle: для такого oracle нужны проверенные `generation_hints`, которые формирует extractor. Planner-derived задача проверяется generic semantic checks. Источник resolution хранится в spec, поэтому это одна явная политика доверия, а не повторная классификация.

## Последствия

Неподдерживаемое состояние завершается ошибкой рядом с источником, а trace содержит упорядоченный список этапов без prompt, context или model response. Внешний HTTP/CLI-контракт не меняется. Family registry и разделение validators смогут опираться на один resolved contract, не создавая второй classifier.
