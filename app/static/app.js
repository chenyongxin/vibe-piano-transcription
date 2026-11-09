/* piano2 M4.2：多曲目队列 + 单页翻页查看器 + 播放跟随 */

const el = (id) => document.getElementById(id);
const STAGES = ["排队中", "提取特征（CQT）", "模型推理", "解码成音符", "生成乐谱",
                "导出 MIDI（与谱面同一时间轴）", "渲染乐谱", "合成回放音频", "完成"];

const MAX_TRACKS = 10;
const MAX_TOTAL_BYTES = 100 * 1024 * 1024;
const ALLOWED = /\.(wav|mp3|m4a|flac|ogg|aac|webm)$/i;

const state = {
  mode: "idle",            // idle | tracks | score
  tracks: [],
  nextId: 1,
  processing: false,
  activeTrackId: null,
};

const view = {
  audio: null,
  els: new Map(),          // SVG 元素 id -> {el, page}
  pageWraps: [],
  bands: [],
  samples: [],
  sampleY: [],
  pageIndex: 0,
  follow: true,
  playing: false,
  currentSample: -1,
  rafId: null,
  loadToken: 0,
};

let pollTimer = null;

function stagePercent(stage) {
  const i = STAGES.indexOf(stage);
  if (i < 0) return 5;
  return Math.round(((i + 1) / STAGES.length) * 100);
}

function formatSize(bytes) {
  return `${(bytes / 1e6).toFixed(1)} MB`;
}

function showError(msg) {
  const box = state.tracks.length ? el("tracks-error") : el("upload-error");
  box.textContent = msg;
  box.hidden = false;
}

function hideError() {
  el("tracks-error").hidden = true;
  el("upload-error").hidden = true;
}

/* ---------- 健康检查 ---------- */

async function checkHealth() {
  const node = el("device");
  if (!node) return;
  try {
    const r = await fetch("/api/health");
    const d = await r.json();
    const label = d.device === "cuda" ? "GPU（CUDA）" : (d.device === "mps" ? "Apple MPS" : "CPU");
    node.textContent = `推理设备：${label}`;
    if (!d.checkpoint_exists) {
      node.textContent += " · ⚠ 模型缺失";
      node.classList.add("warn");
    }
  } catch (e) {
    node.textContent = "后端未连接";
  }
}

/* ---------- 文件选择与队列 ---------- */

function trackStatusText(t) {
  if (t.status === "ready") return "待转录";
  if (t.status === "uploading") return "上传中…";
  if (t.status === "processing") return t.stage ? `${t.stage}（${stagePercent(t.stage)}%）` : "转录中…";
  if (t.status === "done") return "已完成";
  if (t.status === "failed") return t.error || "失败";
  return t.status;
}

function totalBytes() {
  return state.tracks.reduce((s, t) => s + (t.file ? t.file.size : 0), 0);
}

function addFiles(fileList) {
  hideError();
  const files = Array.from(fileList || []);
  if (!files.length) return;
  const errors = [];
  let added = 0;
  for (const f of files) {
    if (!ALLOWED.test(f.name)) {
      errors.push(`${f.name}：不支持的格式`);
      continue;
    }
    if (!f.size) {
      errors.push(`${f.name}：空文件`);
      continue;
    }
    if (state.tracks.some((t) => t.name === f.name && t.size === f.size)) {
      errors.push(`${f.name}：已在列表中`);
      continue;
    }
    if (state.tracks.length >= MAX_TRACKS) {
      errors.push(`最多 ${MAX_TRACKS} 首，已忽略 ${f.name}`);
      continue;
    }
    if (totalBytes() + f.size > MAX_TOTAL_BYTES) {
      errors.push(`总大小不能超过 100 MB，已忽略 ${f.name}`);
      continue;
    }
    state.tracks.push({
      id: state.nextId++,
      file: f,
      name: f.name,
      size: f.size,
      status: "ready",
      stage: "",
      jobId: null,
      result: null,
      error: "",
    });
    added += 1;
  }
  if (errors.length) showError(errors.join("；"));
  if (added) showTracks();
  renderTracks();
}

function deleteTrack(id) {
  if (state.processing) return;
  const t = state.tracks.find((x) => x.id === id);
  if (!t) return;
  if (t.jobId && t.status === "done") {
    fetch(`/api/jobs/${t.jobId}`, { method: "DELETE" }).catch(() => {});
  }
  state.tracks = state.tracks.filter((x) => x.id !== id);
  if (state.tracks.length === 0) {
    showUpload();
  } else {
    renderTracks();
  }
}

function showUpload() {
  state.mode = "idle";
  el("upload-card").hidden = false;
  el("tracks-card").hidden = true;
  el("score-card").hidden = true;
  pauseAudio();
}

function showTracks() {
  state.mode = "tracks";
  el("upload-card").hidden = true;
  el("tracks-card").hidden = false;
  el("score-card").hidden = true;
  pauseAudio();
  renderTracks();
}

function showScore() {
  state.mode = "score";
  el("upload-card").hidden = true;
  el("tracks-card").hidden = true;
  el("score-card").hidden = false;
}

/* ---------- 队列列表渲染 ---------- */

function renderTracks() {
  if (!el("tracks-card").hidden) {
    const list = el("track-list");
    list.innerHTML = "";
    const nDone = state.tracks.filter((t) => t.status === "done").length;
    el("tracks-title").textContent = state.processing ? "正在转录…" : "曲目列表";
    el("tracks-summary").textContent =
      `${state.tracks.length} / ${MAX_TRACKS} 首 · ${formatSize(totalBytes())} / 100 MB` +
      ` · 已完成 ${nDone} 首`;
    el("add-btn").disabled = state.processing || state.tracks.length >= MAX_TRACKS;
    el("start-btn").disabled = state.processing ||
      !state.tracks.some((t) => t.status === "ready");
    el("start-btn").textContent = state.processing ? "转录中…" : "开始转录";

    state.tracks.forEach((t) => {
      const row = document.createElement("div");
      row.className = "track-row";
      row.dataset.id = String(t.id);

      const main = document.createElement("div");
      main.className = "track-main";
      const name = document.createElement("div");
      name.className = "track-name";
      name.textContent = t.name;
      const meta = document.createElement("div");
      meta.className = "track-meta";
      meta.textContent = `${formatSize(t.size)} · ${trackStatusText(t)}`;
      main.appendChild(name);
      main.appendChild(meta);
      row.appendChild(main);

      const actions = document.createElement("div");
      actions.className = "track-actions";
      if (t.status === "done") {
        const open = document.createElement("button");
        open.className = "btn primary small";
        open.textContent = "打开乐谱";
        open.addEventListener("click", () => openScore(t));
        actions.appendChild(open);
      }
      if (!state.processing) {
        const del = document.createElement("button");
        del.className = "btn ghost small danger";
        del.textContent = "删除";
        del.addEventListener("click", () => deleteTrack(t.id));
        actions.appendChild(del);
      }
      row.appendChild(actions);
      list.appendChild(row);
    });
  }
}

/* ---------- 转录流程：逐首串行 ---------- */

async function uploadTrack(t) {
  const fd = new FormData();
  fd.append("file", t.file, t.name);
  const r = await fetch("/api/transcribe", { method: "POST", body: fd });
  const d = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(d.detail || `上传失败（HTTP ${r.status}）`);
  return d.job_id;
}

function pollTrack(t, jobId) {
  return new Promise((resolve, reject) => {
    const tick = async () => {
      try {
        const r = await fetch(`/api/jobs/${jobId}`);
        if (!r.ok) throw new Error("任务查询失败");
        const d = await r.json();
        if (d.status === "done") {
          resolve(d);
        } else if (d.status === "failed") {
          reject(new Error(d.error || "转录失败"));
        } else {
          t.stage = d.stage || "转录中";
          t.status = "processing";
          renderTracks();
          pollTimer = setTimeout(tick, 800);
        }
      } catch (e) {
        reject(e);
      }
    };
    tick();
  });
}

async function startTranscription() {
  if (state.processing) return;
  const pending = state.tracks.filter((t) => t.status === "ready");
  if (!pending.length) return;
  hideError();
  state.processing = true;
  renderTracks();

  for (const t of pending) {
    try {
      t.status = "uploading";
      t.stage = "";
      renderTracks();
      const jobId = await uploadTrack(t);
      t.jobId = jobId;
      t.status = "processing";
      renderTracks();
      const d = await pollTrack(t, jobId);
      t.result = d;
      t.status = "done";
      t.stage = "";
      renderTracks();
    } catch (e) {
      t.status = "failed";
      t.error = e.message || "失败";
      renderTracks();
    }
  }

  state.processing = false;
  if (pollTimer) { clearTimeout(pollTimer); pollTimer = null; }
  const done = state.tracks.filter((t) => t.status === "done");
  if (done.length === 1 && state.tracks.length === 1) {
    openScore(done[0]);
  } else {
    showTracks();
  }
}

/* ---------- 谱面查看器 ---------- */

function sampleIndexAt(t) {
  if (!view.samples.length) return -1;
  let lo = 0, hi = view.samples.length - 1, idx = 0;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (view.samples[mid].t <= t) { idx = mid; lo = mid + 1; } else { hi = mid - 1; }
  }
  return idx;
}

function playbackPageIndex() {
  const idx = sampleIndexAt(view.audio ? view.audio.currentTime : 0);
  if (idx < 0) return 0;
  return Math.max(0, Number(view.samples[idx].page) - 1);
}

function openScore(track) {
  state.activeTrackId = track.id;
  showScore();
  el("score-error").hidden = true;
  el("score-title").textContent = track.name;
  const r = track.result || {};
  el("score-stats").textContent =
    `${r.n_events || 0} 个音符 · ${r.n_pages || 0} 页 · ${(r.duration_s || 0).toFixed(1)} 秒`;
  el("dl-xml").href = r.files ? r.files.musicxml : "#";
  el("dl-pdf").href = r.files ? r.files.pdf : "#";
  el("dl-mid").href = r.files ? r.files.midi : "#";
  const audio = el("player");
  audio.pause();
  audio.src = r.files ? r.files.wav : "";
  audio.currentTime = 0;
  loadScore(track);
}

async function loadScore(track) {
  const token = ++view.loadToken;
  const r = track.result || {};
  const pagesEl = el("pages");
  pagesEl.innerHTML = "";
  view.els.clear();
  view.pageWraps = [];
  view.bands = [];
  view.samples = [];
  view.sampleY = [];
  view.pageIndex = 0;
  view.follow = true;
  view.playing = false;
  view.currentSample = -1;
  if (view.rafId) { cancelAnimationFrame(view.rafId); view.rafId = null; }
  el("follow-btn").hidden = true;

  const svgUrls = r.svg_pages || [];
  for (let i = 0; i < svgUrls.length; i++) {
    const text = await (await fetch(svgUrls[i])).text();
    if (token !== view.loadToken) return;
    const wrap = document.createElement("div");
    wrap.className = "score-page";
    wrap.dataset.page = String(i + 1);
    wrap.innerHTML = text;
    const band = document.createElement("div");
    band.className = "band";
    wrap.appendChild(band);
    pagesEl.appendChild(wrap);
    view.pageWraps.push(wrap);
    view.bands.push(band);
  }

  view.pageWraps.forEach((wrap, idx) => {
    wrap.querySelectorAll("[id]").forEach((node) => {
      view.els.set(node.id, { el: node, page: idx + 1 });
    });
    wrap.addEventListener("click", (evt) => seekFromClick(wrap, evt));
  });

  if (r.timing) {
    const timing = await (await fetch(r.timing)).json();
    if (token !== view.loadToken) return;
    view.samples = timing.samples || [];
    view.sampleY = view.samples.map((s) => {
      const rec = s.ids && s.ids.length ? view.els.get(s.ids[0]) : null;
      if (!rec) return 0;
      const svg = rec.el.ownerSVGElement;
      const vb = svg && svg.viewBox ? svg.viewBox.baseVal : null;
      if (!vb || !vb.height) return 0;
      const box = rec.el.getBBox();
      return (box.y + box.height / 2) / vb.height;
    });
  }

  updatePageControls();
  renderPage();
  if (!view.rafId) {
    const loop = () => {
      if (state.mode === "score") updateHighlight();
      view.rafId = requestAnimationFrame(loop);
    };
    view.rafId = requestAnimationFrame(loop);
  }
}

function renderPage() {
  view.pageWraps.forEach((wrap, i) => {
    wrap.classList.toggle("page-active", i === view.pageIndex);
  });
  updatePageControls();
  clearActive();
  const idx = view.currentSample >= 0 ? view.currentSample : sampleIndexAt(view.audio.currentTime || 0);
  if (idx >= 0 && view.samples[idx] && Number(view.samples[idx].page) - 1 === view.pageIndex) {
    paintSample(idx);
    view.currentSample = idx;
  } else {
    view.currentSample = -1;
  }
}

function updatePageControls() {
  const total = view.pageWraps.length;
  const single = total <= 1;
  el("prev-page").hidden = single;
  el("next-page").hidden = single;
  el("page-indicator").textContent = single ? "" : `第 ${view.pageIndex + 1} / ${total} 页`;
  el("prev-page").disabled = view.pageIndex <= 0;
  el("next-page").disabled = view.pageIndex >= total - 1;
  updateFollowButton();
}

function setPage(target, manual) {
  const total = view.pageWraps.length;
  if (!total) return;
  const next = Math.max(0, Math.min(total - 1, target));
  if (next === view.pageIndex) return;
  view.pageIndex = next;
  if (manual) {
    view.follow = (next === playbackPageIndex());
  }
  renderPage();
  updateFollowButton();
}

function updateFollowButton() {
  const btn = el("follow-btn");
  const total = view.pageWraps.length;
  if (total <= 1 || !view.playing || view.follow) {
    btn.hidden = true;
    return;
  }
  const pp = playbackPageIndex();
  btn.hidden = (pp === view.pageIndex);
}

function returnToCurrent() {
  if (!view.playing) return;
  view.follow = true;
  const pp = playbackPageIndex();
  if (pp !== view.pageIndex) {
    view.pageIndex = pp;
    renderPage();
  }
  el("follow-btn").hidden = true;
}

function clearActive() {
  document.querySelectorAll(".note-active").forEach((n) => n.classList.remove("note-active"));
  view.bands.forEach((b) => { b.style.display = "none"; });
  view.currentSample = -1;
}

function paintSample(idx) {
  const s = view.samples[idx];
  if (!s) return;
  if (Number(s.page) - 1 !== view.pageIndex) return;
  if (s.ids) {
    s.ids.forEach((id) => {
      const rec = view.els.get(id);
      if (rec) rec.el.classList.add("note-active");
    });
  }
  if (s.measures && s.measures.length) {
    let rec = view.els.get(s.measures[0]);
    let measureEl = rec && rec.el.closest ? rec.el.closest("g.measure") : null;
    if (!measureEl && s.ids && s.ids.length) {
      const noteRec = view.els.get(s.ids[0]);
      measureEl = noteRec && noteRec.el.closest ? noteRec.el.closest("g.measure") : null;
    }
    const band = view.bands[view.pageIndex];
    const wrap = view.pageWraps[view.pageIndex];
    if (measureEl && band && wrap) {
      // 用屏幕坐标差定位：verovio 的 SVG 是嵌套的（外层 svg + 内层 viewBox），
      // 直接用 getBBox()*scale 会被嵌套 viewBox 和缩放带偏，高亮块就对不准小节。
      // 纵向只用 staff（五线谱）的包围盒，避免把 tempo 记号和空白也框进去。
      const wr = wrap.getBoundingClientRect();
      const mr = measureEl.getBoundingClientRect();
      const staffRects = Array.from(measureEl.querySelectorAll(".staff"))
        .map((node) => node.getBoundingClientRect())
        .filter((r) => r.width > 0 && r.height >= 0);
      let top = mr.top;
      let height = mr.height;
      if (staffRects.length) {
        top = Math.min(...staffRects.map((r) => r.top));
        const bottom = Math.max(...staffRects.map((r) => r.bottom));
        height = bottom - top;
      }
      band.style.display = "block";
      band.style.left = `${mr.left - wr.left}px`;
      band.style.top = `${top - wr.top}px`;
      band.style.width = `${mr.width}px`;
      band.style.height = `${height}px`;
    }
  }
}

function updateHighlight() {
  if (!view.samples.length || !view.audio) return;
  const idx = sampleIndexAt(view.audio.currentTime || 0);
  if (idx < 0) return;
  const s = view.samples[idx];
  if (view.playing && view.follow) {
    const target = Math.max(0, Number(s.page) - 1);
    if (target !== view.pageIndex) {
      view.pageIndex = target;
      renderPage();
    }
  } else if (view.playing) {
    updateFollowButton();
  }
  if (Number(s.page) - 1 !== view.pageIndex) return;
  if (idx !== view.currentSample) {
    clearActive();
    paintSample(idx);
    view.currentSample = idx;
  }
}

function seekFromClick(wrap, evt) {
  if (!view.samples.length) return;
  const pageNo = Number(wrap.dataset.page);
  const rect = wrap.getBoundingClientRect();
  const yFrac = Math.min(1, Math.max(0, (evt.clientY - rect.top) / rect.height));
  let best = -1;
  let bestDist = Infinity;
  for (let i = 0; i < view.samples.length; i++) {
    if (Number(view.samples[i].page) !== pageNo) continue;
    const d = Math.abs(view.sampleY[i] - yFrac);
    if (d < bestDist) { bestDist = d; best = i; }
  }
  if (best < 0) return;
  const wasPlaying = !view.audio.paused;
  view.audio.currentTime = view.samples[best].t;
  if (wasPlaying) view.audio.play();
}

/* ---------- 播放器事件 ---------- */

function bindPlayer() {
  const audio = el("player");
  view.audio = audio;
  audio.addEventListener("play", () => {
    view.playing = true;
    if (!view.follow) {
      view.follow = (playbackPageIndex() === view.pageIndex);
    }
    updateFollowButton();
  });
  audio.addEventListener("pause", () => {
    view.playing = false;
    updateFollowButton();
  });
  audio.addEventListener("ended", () => {
    view.playing = false;
    updateFollowButton();
  });
}

function pauseAudio() {
  if (view.audio) view.audio.pause();
  view.playing = false;
  el("follow-btn").hidden = true;
}

/* ---------- 事件绑定 ---------- */

function bindEvents() {
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
    if (e.dataTransfer.files && e.dataTransfer.files.length) addFiles(e.dataTransfer.files);
  });
  el("pick-btn").addEventListener("click", () => el("file-input").click());
  el("add-btn").addEventListener("click", () => el("file-input").click());
  el("score-add-btn").addEventListener("click", () => el("file-input").click());
  el("file-input").addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length) addFiles(e.target.files);
    e.target.value = "";
  });
  el("start-btn").addEventListener("click", startTranscription);
  el("prev-page").addEventListener("click", () => setPage(view.pageIndex - 1, true));
  el("next-page").addEventListener("click", () => setPage(view.pageIndex + 1, true));
  el("follow-btn").addEventListener("click", returnToCurrent);
  el("back-btn").addEventListener("click", () => {
    pauseAudio();
    showTracks();
  });
  document.addEventListener("keydown", (e) => {
    if (state.mode !== "score") return;
    const tag = (e.target && e.target.tagName) || "";
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "AUDIO") return;
    if (e.key === "ArrowLeft") setPage(view.pageIndex - 1, true);
    if (e.key === "ArrowRight") setPage(view.pageIndex + 1, true);
  });
}

/* ---------- 兼容：?job= 直接打开单曲 ---------- */

async function loadExisting(jobId) {
  try {
    const r = await fetch(`/api/jobs/${jobId}`);
    if (!r.ok) throw new Error("任务不存在或已清理");
    const d = await r.json();
    if (d.status !== "done") throw new Error("该任务尚未完成");
    const track = {
      id: state.nextId++,
      file: null,
      name: d.audio_name || "已有任务",
      size: 0,
      status: "done",
      stage: "",
      jobId,
      result: d,
      error: "",
    };
    state.tracks = [track];
    openScore(track);
  } catch (e) {
    showUpload();
    showError(e.message);
  }
}

window.addEventListener("DOMContentLoaded", () => {
  bindEvents();
  bindPlayer();
  checkHealth();
  const job = new URL(window.location.href).searchParams.get("job");
  if (job) loadExisting(job);
});
