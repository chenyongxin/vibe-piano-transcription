"""M4.1 FastAPI 后端：上传音频 → 后台转录 → 返回乐谱 / 下载 / 回放。

启动（在仓库根目录）：
    python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
浏览器打开 http://127.0.0.1:8000
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from piano2.constants import CKPT, OUTPUT_ROOT
from piano2.inference import pick_device

from .jobs import JobStore

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
MEDIA_ROOT = Path(OUTPUT_ROOT) / "app"
MEDIA_ROOT.mkdir(parents=True, exist_ok=True)

ALLOWED_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".webm"}
MAX_UPLOAD_MB = 100

app = FastAPI(title="piano2 钢琴转录", version="0.1.0")
store = JobStore(out_root=MEDIA_ROOT)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/media", StaticFiles(directory=str(MEDIA_ROOT)), name="media")


@app.get("/", include_in_schema=False)
def index():
    """返回单页前端。"""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    """健康检查：设备、模型、队列。"""
    checkpoint = Path(CKPT)
    return {
        "status": "ok",
        "device": pick_device(),
        "checkpoint": str(checkpoint),
        "checkpoint_exists": checkpoint.exists(),
        "checkpoint_mb": round(checkpoint.stat().st_size / 1e6, 1) if checkpoint.exists() else 0,
        "queue_workers": 1,
    }


@app.post("/api/transcribe")
async def transcribe_endpoint(file: UploadFile = File(...)):
    """上传音频并创建转录任务，返回 job_id。"""
    name = file.filename or "audio.wav"
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400,
                            detail=f"不支持的格式 {suffix}，可用：{sorted(ALLOWED_SUFFIXES)}")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="空文件")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413,
                            detail=f"文件超过 {MAX_UPLOAD_MB} MB 限制")
    job = store.submit(data, name)
    return {"job_id": job.id, "size_mb": round(len(data) / 1e6, 2)}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    """查询任务状态与结果。"""
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在（可能已过期清理）")
    return store.to_dict(job)
