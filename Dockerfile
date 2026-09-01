FROM python:3.12.14-slim AS build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash build-essential ca-certificates curl make \
    && rm -rf /var/lib/apt/lists/*

COPY . /workspace

ENV PATH="/opt/venv/bin:${PATH}" \
    LOCALSCRIPT_PYTHON_BIN=/opt/venv/bin/python

RUN python -m pip install uv==0.11.21 \
    && uv sync --frozen --no-editable

RUN ./scripts/bootstrap_lua54.sh


FROM python:3.12.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    LOCALSCRIPT_PYTHON_BIN=/opt/venv/bin/python \
    LOCALSCRIPT_OLLAMA_MODE=remote_api \
    LOCALSCRIPT_OLLAMA_HOST=http://ollama:11434 \
    LOCALSCRIPT_ALLOW_REMOTE_OLLAMA=1 \
    LOCALSCRIPT_OLLAMA_CONTAINER_ALIAS=ollama \
    LOCALSCRIPT_UI_ENABLED=1

WORKDIR /workspace

# Базовый образ пересобирается реже, чем Debian выпускает обновления безопасности,
# поэтому пакеты рантайма обновляются при сборке: иначе в финальном образе остаются
# устранимые high-уязвимости базовых пакетов, которые ловит trivy в CI.
RUN apt-get update \
    && apt-get upgrade -y --no-install-recommends \
    && apt-get install -y --no-install-recommends bash ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash appuser

COPY --from=build /opt/venv /opt/venv
COPY --from=build /workspace /workspace

RUN chown -R appuser:appuser /workspace /opt/venv

USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=5 \
  CMD curl -fsS http://127.0.0.1:8080/health || exit 1

CMD ["bash", "./scripts/docker_entrypoint.sh"]
