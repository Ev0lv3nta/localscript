const state = {
  examples: [],
  sessionId: null,
  traceId: null,
  latestSession: {},
  latestValidation: {},
  latestTrace: {},
  timeline: [],
  busy: false,
};

const statusLabels = {
  completed: "Готово и проверено",
  clarification_required: "Нужно уточнение",
  validation_failed: "Проверка не пройдена",
  backend_unavailable: "Модель недоступна",
  policy_rejected: "Запрос отклонён",
  failed: "Ошибка",
};

const stageLabels = {
  received: "запрос принят",
  planned: "план",
  generated: "генерация",
  validated: "проверка",
  reviewed: "ревью",
  revised: "правка",
  completed: "готово",
  failed: "отказ",
  clarification_required: "уточнение",
};

const elements = {
  exampleSelect: document.getElementById("exampleSelect"),
  promptInput: document.getElementById("promptInput"),
  contextInput: document.getElementById("contextInput"),
  clarificationInput: document.getElementById("clarificationInput"),
  feedbackInput: document.getElementById("feedbackInput"),
  codeOutput: document.getElementById("codeOutput"),
  jsonStatus: document.getElementById("jsonStatus"),
  statusHealth: document.getElementById("statusHealth"),
  statusProfile: document.getElementById("statusProfile"),
  statusModel: document.getElementById("statusModel"),
  statusBadge: document.getElementById("statusBadge"),
  revisionBadge: document.getElementById("revisionBadge"),
  stageStrip: document.getElementById("stageStrip"),
  outputShape: document.getElementById("outputShape"),
  outputNullable: document.getElementById("outputNullable"),
  validationSummary: document.getElementById("validationSummary"),
  clarificationBox: document.getElementById("clarificationBox"),
  clarificationQuestion: document.getElementById("clarificationQuestion"),
  sessionBadge: document.getElementById("sessionBadge"),
  traceBadge: document.getElementById("traceBadge"),
  sessionPanel: document.getElementById("sessionPanel"),
  validationPanel: document.getElementById("validationPanel"),
  tracePanel: document.getElementById("tracePanel"),
  timeline: document.getElementById("timeline"),
};

function pretty(value) {
  return JSON.stringify(value ?? {}, null, 2);
}

function parseContext() {
  const raw = elements.contextInput.value.trim();
  if (!raw) {
    elements.jsonStatus.textContent = "Контекст пустой.";
    return null;
  }
  try {
    const parsed = JSON.parse(raw);
    elements.jsonStatus.textContent = "JSON корректен.";
    return parsed;
  } catch (error) {
    elements.jsonStatus.textContent = `Ошибка JSON: ${error.message}`;
    throw error;
  }
}

function detectOutputFormat(code) {
  const stripped = (code || "").trim();
  if (!stripped.startsWith("{") || !stripped.endsWith("}")) {
    return "lua_block";
  }
  try {
    const payload = JSON.parse(stripped);
    if (payload && typeof payload === "object" && !Array.isArray(payload)) {
      return "json_envelope";
    }
  } catch (_error) {
    return "lua_block";
  }
  return "lua_block";
}

function outputContract(code) {
  return {
    format: detectOutputFormat(code),
    shape: elements.outputShape.value,
    nullable: elements.outputNullable.checked,
  };
}

function failedChecks(validation) {
  return (validation?.checks || []).filter((check) => check.status === "failed");
}

function renderStages(trace) {
  elements.stageStrip.innerHTML = "";
  const events = trace?.stage_events || [];
  if (events.length === 0) {
    return;
  }
  for (const event of events) {
    const chip = document.createElement("span");
    chip.className = "meta-chip";
    const label = document.createElement("span");
    label.className = "meta-label";
    label.textContent = stageLabels[event.stage] || event.stage;
    const value = document.createElement("span");
    value.textContent = `${Math.round(event.duration_ms)} мс`;
    chip.appendChild(label);
    chip.appendChild(value);
    elements.stageStrip.appendChild(chip);
  }
}

async function apiFetch(url, options = {}) {
  const response = await fetch(url, options);
  const contentType = response.headers.get("content-type") || "";
  const body = contentType.includes("application/json") ? await response.json() : await response.text();
  if (!response.ok) {
    const detail = body && body.detail ? body.detail : body;
    throw new Error(typeof detail === "string" ? detail : pretty(detail));
  }
  return { response, body };
}

function setBadge(element, text, kind = "neutral") {
  element.textContent = text;
  element.className = `state-badge ${kind}`;
}

function refreshMetaBadges() {
  elements.sessionBadge.textContent = state.sessionId || "—";
  elements.traceBadge.textContent = state.traceId || "—";
}

function setBusy(busy) {
  state.busy = busy;
  for (const id of ["generateBtn", "continueBtn", "feedbackBtn", "validateBtn"]) {
    document.getElementById(id).disabled = busy;
  }
  document.querySelector("#generateBtn span").textContent = busy
    ? "Обработка…"
    : "Сгенерировать и проверить";
}

async function runAction(action) {
  if (state.busy) {
    return;
  }
  setBusy(true);
  try {
    await action();
  } catch (error) {
    showError(error);
  } finally {
    setBusy(false);
  }
}

function renderDiagnostics() {
  elements.sessionPanel.textContent = pretty(state.latestSession);
  elements.validationPanel.textContent = pretty(state.latestValidation);
  elements.tracePanel.textContent = pretty(state.latestTrace);
}

function renderTimeline() {
  elements.timeline.innerHTML = "";
  if (state.timeline.length === 0) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "История появится после первого запроса.";
    elements.timeline.appendChild(empty);
    return;
  }

  for (const entry of state.timeline) {
    const row = document.createElement("article");
    row.className = "timeline-item";
    const title = document.createElement("div");
    title.className = "timeline-title";
    title.textContent = entry.title;
    const meta = document.createElement("div");
    meta.className = "timeline-meta";
    meta.textContent = entry.meta || "";
    row.appendChild(title);
    if (entry.detail) {
      const detail = document.createElement("div");
      detail.className = "timeline-detail";
      detail.textContent = entry.detail;
      row.appendChild(detail);
    }
    if (entry.meta) {
      row.appendChild(meta);
    }
    elements.timeline.appendChild(row);
  }
}

function pushTimeline(title, detail = "", meta = "") {
  state.timeline.unshift({ title, detail, meta });
  renderTimeline();
}

async function refreshSession() {
  if (!state.sessionId) {
    return;
  }
  const { body } = await apiFetch(`/api/sessions/${state.sessionId}`);
  state.latestSession = body;
  renderDiagnostics();
}

async function refreshTrace() {
  if (!state.traceId) {
    return;
  }
  const { body } = await apiFetch(`/api/traces/${state.traceId}`);
  state.latestTrace = body;
  renderStages(body);
  renderDiagnostics();
}

function renderClarification(question) {
  if (!question) {
    elements.clarificationBox.classList.add("hidden");
    elements.clarificationQuestion.textContent = "";
    return;
  }
  elements.clarificationBox.classList.remove("hidden");
  elements.clarificationQuestion.textContent = question;
}

function renderResult(body) {
  state.sessionId = body.session_id;
  state.traceId = body.trace_id;
  state.latestValidation = body.validation || {};

  elements.codeOutput.value = body.code || "";
  renderClarification(body.question || "");
  refreshMetaBadges();
  renderDiagnostics();

  const statusKind =
    body.status === "completed" ? "ok" : body.status === "clarification_required" ? "warn" : "error";
  setBadge(elements.statusBadge, statusLabels[body.status] || body.status, statusKind);
  setBadge(
    elements.revisionBadge,
    `Правок: ${body.revision_count || 0}`,
    body.revision_count ? "warn" : "neutral",
  );

  if (body.status === "clarification_required") {
    elements.validationSummary.textContent = "Ответьте на уточнение — код пока не публикуется.";
    elements.validationSummary.className = "result-note warn";
    return;
  }
  if (body.status === "completed") {
    elements.validationSummary.textContent = body.revision_count
      ? "Проверка и ревью пройдены после одной правки."
      : "Проверка и ревью пройдены без правок.";
    elements.validationSummary.className = "result-note ok";
    return;
  }
  const codes = (body.diagnostics || []).map((item) => item.code);
  elements.validationSummary.textContent = `Код не опубликован: ${codes.join(", ") || "см. диагностику"}`;
  elements.validationSummary.className = "result-note warn";
}

async function loadStatus() {
  const [{ body: health }, { body: profile }] = await Promise.all([
    apiFetch("/health"),
    apiFetch("/api/profile"),
  ]);
  elements.statusHealth.textContent = health.status === "ok" ? "работает" : health.status;
  elements.statusProfile.textContent = profile.profile;
  elements.statusModel.textContent = profile.model;
}

async function loadExamples() {
  const { body } = await apiFetch("/api/examples");
  state.examples = body.examples || [];
  elements.exampleSelect.innerHTML = "";
  for (const example of state.examples) {
    const option = document.createElement("option");
    option.value = example.id;
    option.textContent = example.title;
    elements.exampleSelect.appendChild(option);
  }
}

function loadSelectedExample() {
  const selected = state.examples.find((item) => item.id === elements.exampleSelect.value);
  if (!selected) {
    return;
  }
  elements.promptInput.value = selected.prompt || "";
  elements.contextInput.value = selected.context ? pretty(selected.context) : "";
  elements.feedbackInput.value = "";
  elements.clarificationInput.value = "";
  elements.codeOutput.value = "";
  renderClarification("");
  elements.validationSummary.textContent = selected.description || "Пример загружен.";
  elements.validationSummary.className = "result-note";
  pushTimeline("Запрос подготовлен", selected.description || selected.title, selected.id);
}

async function generate() {
  const prompt = elements.promptInput.value.trim();
  const context = parseContext();
  pushTimeline("Запрос принят", prompt, "POST /api/generate");
  const { body } = await apiFetch("/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      prompt,
      context,
      session_id: state.sessionId,
    }),
  });
  renderResult(body);
  const timelineTitle =
    body.status === "clarification_required" ? "Найдено уточнение" : "Ответ получен";
  pushTimeline(timelineTitle, body.question || body.code || "Код не опубликован.", body.status);
  await refreshSession();
  await refreshTrace();
}

async function continueSession() {
  if (!state.sessionId) {
    throw new Error("Нет активной сессии для продолжения.");
  }
  const answer = elements.clarificationInput.value.trim();
  const { body } = await apiFetch("/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: state.sessionId,
      clarification_answer: answer,
    }),
  });
  renderResult(body);
  pushTimeline("Уточнение учтено", answer || "Пустой ответ", body.status);
  await refreshSession();
  await refreshTrace();
}

async function sendFeedback() {
  if (!state.sessionId) {
    throw new Error("Нет активной сессии для правки результата.");
  }
  const feedback = elements.feedbackInput.value.trim();
  const { body } = await apiFetch("/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: state.sessionId,
      feedback,
    }),
  });
  renderResult(body);
  pushTimeline("Правка применена", feedback || "Пустое замечание", body.status);
  await refreshSession();
  await refreshTrace();
}

async function validateCode() {
  const code = elements.codeOutput.value.trim();
  if (!code) {
    throw new Error("Нет кода для проверки.");
  }
  const { body } = await apiFetch("/api/validate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      code,
      context: parseContext() ?? { wf: { vars: {} } },
      output: outputContract(code),
    }),
  });
  state.latestValidation = body;
  renderDiagnostics();
  if (body.ok) {
    elements.validationSummary.textContent = "Проверка кода прошла успешно.";
    elements.validationSummary.className = "result-note ok";
    pushTimeline("Проверка пройдена", "Код корректен по validation API.", "validated");
  } else {
    const codes = failedChecks(body.validation).map((check) => check.code);
    elements.validationSummary.textContent = `Проверка нашла проблемы: ${codes.join(", ") || "см. диагностику"}`;
    elements.validationSummary.className = "result-note warn";
    pushTimeline("Проверка нашла проблемы", elements.validationSummary.textContent, "validation");
  }
}

async function copyCode() {
  await navigator.clipboard.writeText(elements.codeOutput.value || "");
}

async function copyCurl() {
  const payload = {
    prompt: elements.promptInput.value,
    context: elements.contextInput.value.trim() ? JSON.parse(elements.contextInput.value) : null,
  };
  const curl = `curl -s http://127.0.0.1:8080/api/generate -H 'Content-Type: application/json' -d '${JSON.stringify(payload)}'`;
  await navigator.clipboard.writeText(curl);
  pushTimeline("cURL скопирован", "Команда для POST /api/generate готова.", "буфер обмена");
}

function resetSession() {
  state.sessionId = null;
  state.traceId = null;
  state.latestSession = {};
  state.latestValidation = {};
  state.latestTrace = {};
  state.timeline = [];
  elements.clarificationInput.value = "";
  elements.feedbackInput.value = "";
  elements.codeOutput.value = "";
  renderClarification("");
  setBadge(elements.statusBadge, "Ожидание", "neutral");
  setBadge(elements.revisionBadge, "Правок: 0", "neutral");
  elements.stageStrip.innerHTML = "";
  elements.validationSummary.textContent = "Сессия сброшена.";
  elements.validationSummary.className = "result-note";
  refreshMetaBadges();
  renderDiagnostics();
  renderTimeline();
}

function bindEvents() {
  document.getElementById("loadExampleBtn").addEventListener("click", loadSelectedExample);
  document.getElementById("formatJsonBtn").addEventListener("click", () => {
    const parsed = parseContext();
    elements.contextInput.value = parsed ? pretty(parsed) : "";
  });
  document.getElementById("generateBtn").addEventListener("click", () => runAction(generate));
  document.getElementById("continueBtn").addEventListener("click", () => runAction(continueSession));
  document.getElementById("feedbackBtn").addEventListener("click", () => runAction(sendFeedback));
  document.getElementById("validateBtn").addEventListener("click", () => runAction(validateCode));
  document.getElementById("copyCodeBtn").addEventListener("click", () => copyCode().catch(showError));
  document.getElementById("copyCurlBtn").addEventListener("click", () => copyCurl().catch(showError));
  document.getElementById("resetBtn").addEventListener("click", resetSession);
  document.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      runAction(generate);
    }
  });
}

function showError(error) {
  setBadge(elements.statusBadge, "Ошибка", "error");
  elements.validationSummary.textContent = error.message || String(error);
  elements.validationSummary.className = "result-note error";
  pushTimeline("Ошибка", error.message || String(error), "request");
}

async function boot() {
  bindEvents();
  setBadge(elements.statusBadge, "Ожидание", "neutral");
  setBadge(elements.revisionBadge, "Правок: 0", "neutral");
  refreshMetaBadges();
  renderDiagnostics();
  renderTimeline();
  await loadStatus();
  await loadExamples();
  loadSelectedExample();
}

boot().catch(showError);
