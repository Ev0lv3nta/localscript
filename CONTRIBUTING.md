# Как внести изменение

Спасибо за интерес к LocalScript. Перед началом ознакомьтесь с [архитектурой](docs/architecture.md), [средой разработки](docs/development.md) и существующими [ADR](docs/adr/).

1. Создайте короткую ветку от актуального `main`.
2. Добавьте тест, который фиксирует новое поведение или воспроизводит ошибку.
3. Не ослабляйте fail-closed outcome, validation, security defaults или eval integrity.
4. Запустите `make check`; для изменений контейнера — также `make container-check`.
5. Откройте небольшой PR с описанием причины, решения, проверок и оставшихся рисков.

Новый dependency требует обоснования, pinned lock, совместимой лицензии и обновления `THIRD_PARTY_NOTICES.md`. Private eval data, model weights, runtime state и generated artifacts не принимаются в Git.

Проект использует squash merge. `main` должен оставаться выпускаемым; required CI обязателен для каждого PR.
