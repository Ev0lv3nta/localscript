# Оценка качества

## Цель

Evaluation отвечает на узкий вопрос: может ли конкретная ревизия LocalScript в зафиксированном окружении вернуть проверенный результат для заявленного набора задач, не выдавая невалидный код за успех. Она не измеряет универсальную способность модели писать Lua.

## Корпусы

| Набор | Размер | Роль | Публикация |
|---|---:|---|---|
| Regression | 140 кейсов | разработка и предотвращение известных регрессий | в репозитории |
| Public v1 | 12 кейсов | воспроизводимый semantic benchmark | в репозитории |
| Private holdout v1 | 8 кейсов | однократная проверка замороженного release candidate | только хеш и итоговый evidence |

В runtime prompt не передаётся reference code. Public и private cases содержат запрос, контекст и executable/structural expectations. Manifest задаёт роль каждого файла, runner, gate и допустимый scope утверждений.

До запуска integrity check сравнивает prompt и существенные поля корпусов между собой и с knowledge base. Проверяются точное совпадение, нормализованное совпадение и нечёткое сходство выше установленного порога. Holdout загружается по внешнему пути и не входит в wheel, sdist или Git history.

## Метрики

- `verified_cases` — кейсы с исходом `completed` и пройденными oracle/validation checks;
- `invalid_success_count` — случаи, объявленные успешными вопреки проваленной проверке; обязательное значение — ноль;
- `clarification_count` — ожидаемые и неожиданные уточнения;
- `degraded_count` — результаты с неполной доступностью проверок;
- latency — длительность end-to-end и фактических model calls;
- repair distribution — число раундов и доля результатов, потребовавших repair;
- repeat stability — совпадение исходов в трёх прогонах public-v1.

Агрегат не скрывает проваленные cases: evidence сохраняет результат каждого кейса, validation codes, стадии и время.

## Ablation

Контролируемый ablation сравнивает пять профилей:

1. one-shot candidate;
2. planner + writer;
3. planner + writer + validation;
4. предыдущий профиль + deterministic repair;
5. полный pipeline.

Средние профили используют один и тот же заранее полученный candidate, что явно отмечается в отчёте. Release gate требует полного прохождения `full_pipeline` и нулевого `invalid_success_count`; остальные профили служат объяснением вклада стадий, а не отдельными release gates.

## Release evidence

`scripts/release_gate.py --mode competition` работает только на чистой ревизии из `main` и требует внешний holdout. JSON-отчёт включает:

- точный commit SHA и признак чистого worktree;
- SHA-256 каждого корпуса и integrity report;
- версии Python, Lua, `luac`, Ollama и digest выбранной модели;
- GPU и доступную память;
- параметры runtime-профиля;
- результаты integration tests, doctor, smoke, latency, public repeats, ablation и private holdout;
- stdout/stderr, return code и длительность каждой команды;
- полный список hard-gate failures.

Успешный runtime snapshot получает `locked: true` только после всех проверок. Evidence прикладывается к GitHub Release с именем, содержащим релиз и короткий SHA.

## Воспроизведение публичного контура

При запущенной Ollama и установленной модели:

```bash
.venv/bin/localscript benchmark --dataset evals/public/v1.jsonl
.venv/bin/python scripts/bench_repeated.py --dataset evals/public/v1.jsonl --repeats 3
.venv/bin/python scripts/bench_ablation.py --dataset evals/public/v1.jsonl --require-full-pass
```

Полный competition gate требует владельца private holdout и подходящую GPU-машину. Публичный набор после выпуска считается regression benchmark для будущих версий; новые обобщающие утверждения требуют нового внешнего набора.

## Интерпретация

Корректная формулировка: «ревизия X прошла N/N кейсов набора Y в окружении Z; невалидных успешных исходов не обнаружено». Некорректная формулировка: «система гарантированно решает любые Lua-задачи».
