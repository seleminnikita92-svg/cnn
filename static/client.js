const uploadForm = document.getElementById("classifier-form");
const pickerInput = document.getElementById("garment-picker");
const invertCheckbox = document.getElementById("invert-toggle");
const analyzeBtn = document.getElementById("analyze-btn");

const outputPanel = document.getElementById("output-panel");
const errorBanner = document.getElementById("error-banner");
const verdictCard = document.getElementById("verdict-card");

const topLabel = document.getElementById("top-label");
const topScore = document.getElementById("top-score");
const classCode = document.getElementById("class-code");
const srcPreview = document.getElementById("src-preview");
const cnnPreview = document.getElementById("cnn-preview");
const scoresBody = document.getElementById("scores-body");

const LIMIT_BYTES = 5 * 1024 * 1024;

function readFileAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(new Error("read failed"));
    reader.readAsDataURL(file);
  });
}

function renderFailure(text) {
  outputPanel.hidden = false;
  verdictCard.hidden = true;
  errorBanner.hidden = false;
  errorBanner.textContent = `Ошибка: ${text}`;
}

function resetValidityState() {
  pickerInput.setCustomValidity("");
  pickerInput.removeAttribute("aria-invalid");
}

pickerInput.addEventListener("change", () => {
  resetValidityState();
  const chosen = pickerInput.files?.[0];
  if (chosen && chosen.size > LIMIT_BYTES) {
    pickerInput.setCustomValidity("Превышен максимальный размер файла 5 МБ.");
    pickerInput.setAttribute("aria-invalid", "true");
  }
});

pickerInput.addEventListener("invalid", () => {
  if (!pickerInput.files || pickerInput.files.length === 0) {
    pickerInput.setCustomValidity("Сначала выберите изображение.");
  }
  pickerInput.setAttribute("aria-invalid", "true");
});

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (!uploadForm.reportValidity()) {
    pickerInput.focus();
    return;
  }

  const chosen = pickerInput.files?.[0];
  if (!chosen) {
    renderFailure("Выберите файл для проверки.");
    pickerInput.focus();
    return;
  }

  const payload = new FormData();
  payload.append("image", chosen);
  payload.append("invert", invertCheckbox.checked ? "true" : "false");

  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Проверка...";

  try {
    const res = await fetch("/api/classify", {
      method: "POST",
      body: payload,
    });
    const body = await res.json();

    if (!res.ok) {
      renderFailure(body.error || "Не удалось обработать файл.");
      return;
    }

    srcPreview.src = await readFileAsDataUrl(chosen);
    cnnPreview.src = body.input_preview_uri;
    topLabel.textContent = body.label_title;
    topScore.textContent = `${(body.score * 100).toFixed(2)}%`;
    classCode.textContent = `(класс ${body.label_id})`;

    scoresBody.replaceChildren();
    for (const entry of body.distribution) {
      const tr = document.createElement("tr");

      const th = document.createElement("th");
      th.scope = "row";
      th.textContent = entry.label_title;

      const td = document.createElement("td");
      td.textContent = `${(entry.score * 100).toFixed(1)}%`;

      tr.append(th, td);
      scoresBody.appendChild(tr);
    }

    outputPanel.hidden = false;
    errorBanner.hidden = true;
    verdictCard.hidden = false;
  } catch {
    renderFailure("Ошибка соединения с сервером.");
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Проверить";
  }
});
