# ADR 0003: единый реестр семейств задач

- Статус: принято
- Дата: 9 августа 2026 года

## Контекст

Знание о family было продублировано в `TaskExtractor`, таблице return shape, semantic oracles и `ScenarioValidator`. Добавление семейства требовало синхронно менять несколько цепочек `if family == ...`; неизвестное имя от planner могло попасть в resolved spec без зарегистрированного контракта.

## Решение

Вводится неизменяемый `FamilyDefinition` и один fail-closed registry. Definition задаёт:

- стабильное имя family;
- предпочтительную форму результата;
- deterministic oracle, если он существует;
- family-specific structural validation.

Реализации сгруппированы по предметной ответственности: коллекции и отображения, скалярные преобразования, record/envelope-задачи. `oracles.py` и `ScenarioValidator` становятся адаптерами registry и больше не классифицируют задачу повторно.

Extractor остаётся единственным deterministic router. Planner может выбрать только зарегистрированную family; неизвестное имя сохраняется как evidence в `planner_family`, но эффективная family становится `generic_lua`. Deterministic family oracle по-прежнему запускается только для extractor-derived spec с проверенными hints.

## Последствия

Новая family должна получить definition и contract fixtures. Дубликаты имён делают импорт registry ошибочным. Общие validators, runtime executor и generic semantic checks не зависят от количества family. Matcher extraction и canonical repair будут переноситься отдельными reviewable шагами, не меняя этот registry contract.
