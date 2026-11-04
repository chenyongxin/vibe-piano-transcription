/* piano2 本地 Web 应用（M4.1 MVP） */

const el = (id) => document.getElementById(id);
const STAGES = ["排队中", "提取特征（CQT）", "模型推理", "解码成音符", "生成乐谱",
                "导出 MIDI（与谱面同一时间轴）", "合成回放音频", "计算高亮时间轴", "完成"];

let selectedFile = null;
let pollTimer = null;
let rafId = null;

// 乐谱跟随视图状态
const view = {
  audio: null,
  timing: null,        // {duration, samples:[{t,page,ids}]}
  els: new Map(),      // id -> {el, page}
  samples: [],
  sampleY: [],         // 每个采样点在当前页里的纵向比例（0-1），用于拖动定位
  cur: -1,
  dragging: false,
  resumeAfterDrag: false,
};

function stagePercent(stage) {
  const i = STAGES.indexOf(stage);
  if (i < 0) return 5;
  return Math.round(((i + 1) / STAGES.length) * 100);
}

function fmtTime(sec) {
  sec = Math.max(0, Number(sec) || 0);
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
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
        await renderResult(jobId, d);
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

async function renderResult(jobId, d) {
  el("status-card").hidden = true;
  el("result-card").hidden = false;
  el("start-btn").disabled = false;
  el("stats").textContent =
    `${d.n_events} 个音符 · ${d.n_pages} 页 · ${(d.duration_s || 0).toFixed(1)} 秒` +
    (d.audio_name ? ` · ${d.audio_name}` : "");

  el("dl-xml").href = d.files.musicxml;
  el("dl-pdf").href = d.files.pdf;
  el("dl-mid").href = d.files.midi;

  const audio = el("player");
  audio.src = d.files.wav;

  const url = new URL(window.location.href);
  url.searchParams.set("job", jobId);
  window.history.replaceState(null, "", url);

  await setupScoreView(audio, d);
}

async function setupScoreView(audio, d) {
  const pages = el("pages");
  pages.innerHTML = "";
  view.els.clear();
  view.samples = [];
  view.sampleY = [];
  view.cur = -1;
  view.audio = audio;

  // 1) 内联 SVG 分页（需要 DOM 才能高亮）
  if (d.svg_pages && d.svg_pages.length) {
    for (const url of d.svg_pages) {
      const text = await (await fetch(url)).text();
      const wrap = document.createElement("div");
      wrap.className = "score-page";
      wrap.innerHTML = text;
      pages.appendChild(wrap);
    }
  } else {
    // 兜底：老任务只有 PNG
    (d.pages || []).forEach((src, i) => {
      const img = document.createElement("img");
      img.src = src;
      img.alt = `第 ${i + 1} 页`;
      pages.appendChild(img);
    });
  }

  // 2) 建立 id → 元素 + 页码索引
  const pageEls = [...pages.querySelectorAll(".score-page")];
  pageEls.forEach((wrap, idx) => {
    wrap.dataset.page = String(idx + 1);
    wrap.querySelectorAll("[id]").forEach((node) => {
      view.els.set(node.id, { el: node, page: idx + 1 });
    });
  });

  // 3) 时间轴 + 每个采样点的纵向位置（用于拖动定位）
  if (d.timing) {
    view.timing = await (await fetch(d.timing)).json();
    view.samples = view.timing.samples || [];
    view.sampleY = view.samples.map((s) => {
      const rec = s.ids.length ? view.els.get(s.ids[0]) : null;
      if (!rec) return 0;
      const svg = rec.el.ownerSVGElement;
      const vb = svg && svg.viewBox ? svg.viewBox.baseVal : null;
      if (!vb || !vb.height) return 0;
      const box = rec.el.getBBox();
      return (box.y + box.height / 2) / vb.height;
    });
    bindScoreSeek(pageEls);
  }

  // 4) 高亮 + 进度条同步循环
  if (rafId) cancelAnimationFrame(rafId);
  const loop = () => {
    const dur = audio.duration || (view.timing && view.timing.duration) || 0;
    const cur = audio.currentTime || 0;
    if (dur > 0) {
      el("seek").value = String((cur / dur) * 100);
      el("time-cur").textContent = fmtTime(cur);
      el("time-total").textContent = fmtTime(dur);
    }
    if (!view.dragging) highlightAt(cur);
    rafId = requestAnimationFrame(loop);
  };
  rafId = requestAnimationFrame(loop);

  // 5) 进度条拖动：播放中拖动继续播放、暂停时拖动保持暂停（浏览器原生行为）
  el("seek").addEventListener("input", (e) => {
    const dur = audio.duration || (view.timing && view.timing.duration) || 0;
    if (dur > 0) audio.currentTime = (Number(e.target.value) / 100) * dur;
  });

  // 6) ?t=10 这样的参数用于演示/截图：定位到指定秒
  const t = Number(new URL(window.location.href).searchParams.get("t") || 0);
  if (t > 0) {
    const seek = () => { audio.currentTime = Math.min(t, (audio.duration || 1e9)); };
    audio.addEventListener("loadedmetadata", seek, { once: true });
    if (audio.readyState >= 1) seek();
  }
}

function sampleNear(pageNo, yFrac) {
  let best = -1;
  let bestDist = Infinity;
  for (let i = 0; i < view.samples.length; i++) {
    const s = view.samples[i];
    if (s.page !== pageNo) continue;
    const d = Math.abs(view.sampleY[i] - yFrac);
    if (d < bestDist) { bestDist = d; best = i; }
  }
  return best;
}

function bindScoreSeek(pageEls) {
  pageEls.forEach((wrap) => {
    const pageNo = Number(wrap.dataset.page);
    const locate = (evt) => {
      const rect = wrap.getBoundingClientRect();
      const yFrac = Math.min(1, Math.max(0, (evt.clientY - rect.top) / rect.height));
      return sampleNear(pageNo, yFrac);
    };
    wrap.addEventListener("pointerdown", (evt) => {
      if (!view.samples.length) return;
      view.dragging = true;
      view.resumeAfterDrag = !view.audio.paused;
      if (view.resumeAfterDrag) view.audio.pause();
      wrap.setPointerCapture(evt.pointerId);
      const idx = locate(evt);
      if (idx >= 0) previewSample(idx);
      evt.preventDefault();
    });
    wrap.addEventListener("pointermove", (evt) => {
      if (!view.dragging) return;
      const idx = locate(evt);
      if (idx >= 0) previewSample(idx);
    });
    const finish = (evt) => {
      if (!view.dragging) return;
      view.dragging = false;
      const idx = locate(evt);
      if (idx >= 0) {
        view.audio.currentTime = view.samples[idx].t;
        if (view.resumeAfterDrag) view.audio.play();
      }
      view.resumeAfterDrag = false;
    };
    wrap.addEventListener("pointerup", finish);
    wrap.addEventListener("pointercancel", finish);
  });
}

function clearActive() {
  document.querySelectorAll(".note-active").forEach((n) => n.classList.remove("note-active"));
}

function previewSample(idx) {
  const s = view.samples[idx];
  if (!s) return;
  clearActive();
  s.ids.forEach((id) => {
    const rec = view.els.get(id);
    if (rec) rec.el.classList.add("note-active");
  });
  el("time-cur").textContent = fmtTime(s.t);
}

function highlightAt(t) {
  if (!view.samples.length) return;
  let lo = 0, hi = view.samples.length - 1, idx = 0;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (view.samples[mid].t <= t) { idx = mid; lo = mid + 1; } else { hi = mid - 1; }
  }
  if (idx === view.cur) return;
  view.cur = idx;
  clearActive();
  const s = view.samples[idx];
  s.ids.forEach((id) => {
    const rec = view.els.get(id);
    if (rec) {
      rec.el.classList.add("note-active");
      if (!view.dragging) ensureVisible(rec.el);
    }
  });
}

function ensureVisible(node) {
  const rect = node.getBoundingClientRect();
  const margin = 120;
  if (rect.top < margin || rect.bottom > window.innerHeight - margin) {
    node.scrollIntoView({ block: "center", behavior: "smooth" });
  }
}

async function loadExisting(jobId) {
  el("status-card").hidden = false;
  el("stage").textContent = "加载已有任务…";
  try {
    const r = await fetch(`/api/jobs/${jobId}`);
    if (!r.ok) throw new Error("任务不存在或已清理");
    const d = await r.json();
    if (d.status === "done") {
      await renderResult(jobId, d);
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
