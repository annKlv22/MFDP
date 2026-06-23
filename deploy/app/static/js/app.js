"use strict";

const form = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const dropzone = document.getElementById("dropzone");
const fileInfo = document.getElementById("file-info");
const fileName = document.getElementById("file-name");
const audioPreview = document.getElementById("audio-preview");
const submitBtn = document.getElementById("submit-btn");
const resetBtn = document.getElementById("reset-btn");
const result = document.getElementById("result");

const PLACEHOLDER = "/static/img/placeholder.svg";
const POLL_INTERVAL_MS = 1500;
const POLL_MAX_ATTEMPTS = 80; // ~2 минуты

let selectedFile = null;
let pollTimer = null;

function onFile(file) {
  if (!file) return;
  selectedFile = file;
  fileName.textContent = file.name;
  audioPreview.src = URL.createObjectURL(file);
  fileInfo.classList.remove("hidden");
  submitBtn.disabled = false;
  dropzone.classList.add("has-file");
  result.classList.add("hidden");
}

fileInput.addEventListener("change", (e) => onFile(e.target.files[0]));

["dragenter", "dragover"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.add("drag");
  })
);
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag");
  })
);
dropzone.addEventListener("drop", (e) => onFile(e.dataTransfer.files[0]));

resetBtn.addEventListener("click", () => {
  selectedFile = null;
  fileInput.value = "";
  fileInfo.classList.add("hidden");
  submitBtn.disabled = true;
  dropzone.classList.remove("has-file");
  result.classList.add("hidden");
  if (pollTimer) clearInterval(pollTimer);
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!selectedFile) return;
  submitBtn.disabled = true;
  showStatus("Загрузка записи…");

  const fd = new FormData();
  fd.append("file", selectedFile);

  let res;
  try {
    res = await fetch("/api/predictions/", { method: "POST", body: fd });
  } catch (err) {
    showError("Сеть недоступна. Проверьте соединение.");
    submitBtn.disabled = false;
    return;
  }
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    showError(data.detail || "Не удалось загрузить файл.");
    submitBtn.disabled = false;
    return;
  }
  const { id } = await res.json();
  poll(id);
});

function poll(id) {
  showStatus("Анализируем запись…");
  if (pollTimer) clearInterval(pollTimer);
  let attempts = 0;
  pollTimer = setInterval(async () => {
    attempts += 1;
    let res;
    try {
      res = await fetch(`/api/predictions/${id}`);
    } catch {
      return;
    }
    if (!res.ok) return;
    const p = await res.json();
    if (p.status === "DONE") {
      clearInterval(pollTimer);
      renderResult(p);
      submitBtn.disabled = false;
    } else if (p.status === "FAILED") {
      clearInterval(pollTimer);
      showError(p.error || "Не удалось обработать запись.");
      submitBtn.disabled = false;
    } else if (attempts > POLL_MAX_ATTEMPTS) {
      clearInterval(pollTimer);
      showError("Превышено время ожидания. Попробуйте ещё раз.");
      submitBtn.disabled = false;
    }
  }, POLL_INTERVAL_MS);
}

function showStatus(text) {
  result.classList.remove("hidden");
  result.innerHTML = `<div class="status"><div class="spinner"></div><div>${text}</div></div>`;
}

function showError(text) {
  result.classList.remove("hidden");
  result.innerHTML = `<div class="error">⚠️ ${escapeHtml(text)}</div>`;
}

function renderResult(p) {
  if (!p.scores || !p.scores.length) {
    showError("Модель не вернула результат.");
    return;
  }
  const top = p.scores[0];
  const top3 = p.scores.slice(0, 3);

  const badge = p.is_confident
    ? `<span class="badge ok">Уверенное определение</span>`
    : `<span class="badge warn">Низкая уверенность — попробуйте запись почище или подлиннее</span>`;

  const bars = top3
    .map(
      (s) => `
      <div class="bar-row">
        <div class="bar-label"><span>${escapeHtml(s.name)}</span><span class="bar-pct">${s.percent}%</span></div>
        <div class="bar"><div class="bar-fill" style="width:${s.percent}%"></div></div>
      </div>`
    )
    .join("");

  result.classList.remove("hidden");
  result.innerHTML = `
    <div class="hero">
      <img class="hero-photo" src="${top.photo_url || PLACEHOLDER}" alt="${escapeHtml(top.name)}"
           onerror="this.onerror=null;this.src='${PLACEHOLDER}'" />
      <div class="hero-body">
        ${badge}
        <h2>${escapeHtml(top.name)} <span class="pct">${top.percent}%</span></h2>
        <div class="latin">${escapeHtml(top.latin)}</div>
        <p class="desc">${escapeHtml(top.description)}</p>
      </div>
    </div>
    <div class="bars">
      <div class="bars-title">Топ-3 вероятных вида</div>
      ${bars}
    </div>
    <div class="meta-line">
      Длительность: ${p.duration_sec != null ? p.duration_sec.toFixed(1) + " с" : "—"} ·
      Обработка: ${p.processing_ms != null ? p.processing_ms + " мс" : "—"}
    </div>`;
}

function escapeHtml(s) {
  return (s || "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}
