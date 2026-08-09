# Модель безопасности

## Активы и доверенная граница

LocalScript обрабатывает prompt, workflow context, model output, сессии и диагностические traces. Предполагается один оператор, контролирующий host и Ollama. Код приложения и локальные конфиги находятся внутри доверенной границы; model output и вход пользователя считаются недоверенными.

## Защитные меры

- HTTP bind по умолчанию ограничен `127.0.0.1`.
- Non-loopback bind требует `LOCALSCRIPT_REMOTE_MODE=1` и bearer token не короче 32 символов.
- Произвольный удалённый Ollama host запрещён; Compose разрешает только service alias `ollama`.
- Request body, prompt, context size, nesting depth и node count ограничены.
- Session/trace paths строятся только из проверенных UUID; записи атомарны и защищены блокировками.
- Запрещены опасные Lua roots, глобальные мутации и неподдерживаемые вызовы.
- Lua запускается отдельным процессом с timeout и resource limits.
- Model candidate публикуется только после обязательной проверки; неуспех не содержит `code`.
- Trace storage не сохраняет полный model transcript и редактирует приватные artifacts.
- Dependencies зафиксированы lock-файлом; CI проверяет лицензии, package contents, container smoke и полную Git-историю на секреты.

## Что не гарантируется

Subprocess с resource limits — это не security sandbox. Он не даёт границу уровня VM/container sandbox и не рассчитан на исполнение намеренно вредоносного Lua от недоверенного удалённого пользователя. Не публикуйте сервис в интернет и не используйте его как multi-tenant execution platform.

Семантическая проверка полна только для известных families и доступного Lua runtime. Внешние side effects не исполняются и не моделируются. Local inference не делает сам host конфиденциальным: локальный администратор и процессы с доступом к каталогу state могут читать рабочие данные.

## Рекомендации оператору

- Оставляйте loopback bind и подключайтесь к удалённой машине через SSH tunnel.
- Храните `LOCALSCRIPT_STATE_DIR` на локальном зашифрованном диске с правами текущего пользователя.
- Не передавайте в prompt/context секреты, не нужные для генерации.
- Обновляйте model image и зависимости только через отдельный проверяемый PR.
- Для недоверенного кода добавляйте отдельную изоляцию уровня gVisor, Firecracker или одноразовой VM.
- После работы удаляйте ненужные state/evidence artifacts согласно своей политике хранения.

## Supply chain

Python dependencies и build tools pinned в `uv.lock`; Docker base, service images и GitHub Actions используют версии или полные commit SHA. Vendored Lua 5.4.6 сопровождается исходной лицензией и проверяется при сборке. Перед релизом выполняются secret scan, dependency audit и container vulnerability scan; результаты релизной runtime-проверки прикладываются к GitHub Release.

## Сообщение об уязвимости

Не создавайте публичный issue с exploit, чувствительными данными или токенами. Используйте GitHub Private Vulnerability Reporting в разделе Security репозитория. Укажите затронутую версию/SHA, окружение, минимальные шаги воспроизведения и оценку воздействия.
