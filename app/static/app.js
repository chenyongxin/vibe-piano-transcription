/* piano2 本地 Web 应用（M4.1 MVP） */

const el = (id) => document.getElementById(id);
const STAGES = ["排队中", "提取特征（CQT）", "模型推理", "解码成音符", "生成乐谱",
                "渲染乐谱图片", "导出 PDF", "导出 MIDI", "合成回放音频", "完成"];

let selectedFile = null;
let pollTimer = null;

function stagePercent(stage) {
  const i = STAGES.indexOf(stage);
  if (i < 0) return 5;
  return Math.round(((i + 1) / STAGES.length) * 100);
}

async function checkHealth() {
  try {
    const r = await fetch("/api/health");
    const d = await r.json();
    const label = d.device === "cuda" ? "GPU（CUDA）" : (d.device === "mps" ? "Apple MPS" : "CPU");
    el("device").textContent = `推理设备：${label}`;
    if (!d.checkpoint_exists) {
      el("device").textContent += " · ⚠ 模型缺失";
      el("device").classList.add("warn");
    }
  } catch (e) {
    el("device").textContent = "后端未连接";
  }
}

function bindDropzone() {
  const dz = el("dropzone");
  ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => {
    e.preventDefault();
    dz.classList.add("over");
  }));
  ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => {
    e.preventDefault();
    dz.classList.remove("over");
  }));
  dz.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) selectFile(f);
  });
  el("pick-btn").addEventListener("click", () => el("file-input").click());
  el("file-input").addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) selectFile(e.target.files[0]);
  });
  el("start-btn").addEventListener("click", startJob);
}

function selectFile(file) {
  const ok = /\.(wav|mp3|m4a|flac|ogg|aac|webm)$/i.test(file.name);
  if (!ok) {
    showError("不支持的格式，请用 wav / mp3 / m4a / flac。");
    return;
  }
  selectedFile = file;
  el("file-name").textContent = `${file.name}（${(file.size / 1e6).toFixed(1)} MB）`;
  el("file-info").hidden = false;
  el("result-card").hidden = true;
  el("status-card").hidden = true;
}

function showError(msg) {
  const box = el("error");
  box.textContent = msg;
  box.hidden = false;
}

async function startJob() {
  if (!selectedFile) return;
  el("start-btn").disabled = true;
  el("result-card").hidden = true;
  el("status-card").hidden = false;
  el("error").hidden = true;
  el("stage").textContent = "上传中…";
  try {
    const fd = new FormData();
    fd.append("file", selectedFile, selectedFile.name);
    const r = await fetch("/api/transcribe", { method: "POST", body: fd });
    if (!r.ok) {
      const d = await r.json().catch(() => ({}));
      throw new Error(d.detail || `上传失败（HTTP ${r.status}）`);
    }
    const d = await r.json();
    pollJob(d.job_id);
  } catch (e) {
    showError(e.message);
    el("start-btn").disabled = false;
  }
}

async function pollJob(jobId) {
  if (pollTimer) clearInterval(pollTimer);
  const tick = async () => {
    try {
      const r = await fetch(`/api/jobs/${jobId}`);
      if (!r.ok) throw new Error("任务查询失败");
      const d = await r.json();
      el("stage").textContent = d.stage || "处理中…";
      el("elapsed").textContent = `${d.elapsed_s || 0} 秒`;
      el("progress-bar").style.width = `${stagePercent(d.stage)}%`;
      if (d.status === "done") {
        clearInterval(pollTimer);
        el("device").textContent = `推理设备：${d.device === "cuda" ? "GPU（CUDA）" : (d.device === "mps" ? "Apple MPS" : "CPU")}`;
        renderResult(jobId, d);
      } else if (d.status === "failed") {
        clearInterval(pollTimer);
        showError(d.error || "转录失败");
        el("start-btn").disabled = false;
      }
    } catch (e) {
      clearInterval(pollTimer);
      showError(e.message);
    }
  };
  tick();
  pollTimer = setInterval(tick, 800);
}

function renderResult(jobId, d) {
  el("status-card").hidden = true;
  el("result-card").hidden = false;
  el("start-btn").disabled = false;
  el("stats").textContent =
    `${d.n_events} 个音符 · ${d.n_pages} 页 · 推理 ${d.elapsed_s} 秒` +
    (d.audio_name ? ` · ${d.audio_name}` : "");

  const pages = el("pages");
  pages.innerHTML = "";
  (d.pages || []).forEach((src, i) => {
    const img = document.createElement("img");
    img.src = src;
    img.alt = `第 ${i + 1} 页`;
    img.loading = "lazy";
    pages.appendChild(img);
  });

  el("dl-xml").href = d.files.musicxml;
  el("dl-pdf").href = d.files.pdf;
  el("dl-mid").href = d.files.midi;
  el("player").src = d.files.wav;

  // 把 job id 写进地址栏，方便刷新/分享（同一台机器）
  const url = new URL(window.location.href);
  url.searchParams.set("job", jobId);
  window.history.replaceState(null, "", url);
}

async function loadExisting(jobId) {
  el("status-card").hidden = false;
  el("stage").textContent = "加载已有任务…";
  try {
    const r = await fetch(`/api/jobs/${jobId}`);
    if (!r.ok) throw new Error("任务不存在或已清理");
    const d = await r.json();
    if (d.status === "done") {
      renderResult(jobId, d);
    } else {
      pollJob(jobId);
    }
  } catch (e) {
    showError(e.message);
  }
}

window.addEventListener("DOMContentLoaded", () => {
  bindDropzone();
  checkHealth();
  const job = new URL(window.location.href).searchParams.get("job");
  if (job) loadExisting(job);
});
