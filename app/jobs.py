"""任务队列：单 worker 串行推理，避免多个任务同时抢显存。"""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from piano2.constants import OUTPUT_ROOT

from .pipeline import ModelHolder, transcribe


@dataclass
class Job:
    """一个转录任务。"""

    id: str
    status: str = "queued"          # queued / running / done / failed
    stage: str = "排队中"
    error: str = ""
    created: float = field(default_factory=time.time)
    audio_name: str = ""
    result: dict = field(default_factory=dict)
    out_dir: Path = Path()


class JobStore:
    """内存任务表 + 单线程执行器。"""

    def __init__(self, out_root=None, checkpoint=None, ttl_seconds=7200):
        """初始化。

        参数:
            out_root (str | Path | None): 产物根目录，默认 OUTPUT_ROOT/app。
            checkpoint (str | None): 模型检查点。
            ttl_seconds (int): 任务产物保留时长（秒），超时在下次提交时清理。
        """
        self.out_root = Path(out_root or (OUTPUT_ROOT / "app"))
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds
        self.holder = ModelHolder(checkpoint)
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="piano2-job")

    def submit(self, audio_bytes, original_name):
        """保存上传文件并提交任务，返回 Job。

        参数:
            audio_bytes (bytes): 音频内容。
            original_name (str): 原始文件名（用于显示）。
        """
        self._cleanup()
        job_id = uuid.uuid4().hex[:12]
        job = Job(id=job_id, audio_name=original_name,
                  out_dir=self.out_root / job_id)
        job.out_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(original_name).suffix or ".wav"
        audio_path = job.out_dir / f"input{suffix}"
        audio_path.write_bytes(audio_bytes)
        with self.lock:
            self.jobs[job_id] = job
        self.pool.submit(self._run, job, str(audio_path))
        return job

    def get(self, job_id):
        """查询任务；不存在返回 None。"""
        with self.lock:
            return self.jobs.get(job_id)

    def _run(self, job, audio_path):
        """后台执行转录，更新任务状态。"""
        job.status = "running"
        try:
            result = transcribe(
                audio_path, job.out_dir, self.holder,
                title=f"转录：{job.audio_name}",
                progress=lambda name: setattr(job, "stage", name),
            )
            job.result = result
            job.status = "done"
            job.stage = "完成"
        except Exception as exc:  # 任何异常都转成任务失败，而不是让服务崩溃
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            job.stage = "失败"

    def to_dict(self, job):
        """转成给前端用的 JSON。"""
        data = {
            "job_id": job.id,
            "status": job.status,
            "stage": job.stage,
            "error": job.error,
            "audio_name": job.audio_name,
            "elapsed_s": round(time.time() - job.created, 1),
        }
        if job.status == "done":
            media = f"/media/{job.id}"
            data.update({
                "n_events": job.result["n_events"],
                "n_pages": job.result["n_pages"],
                "device": job.result["device"],
                "pages": [f"{media}/{name}" for name in job.result["pages"]],
                "files": {
                    "musicxml": f"{media}/{job.result['musicxml']}",
                    "midi": f"{media}/{job.result['midi']}",
                    "pdf": f"{media}/{job.result['pdf']}",
                    "wav": f"{media}/{job.result['wav']}",
                },
            })
        return data

    def _cleanup(self):
        """清理超过 TTL 的任务与目录。"""
        now = time.time()
        with self.lock:
            expired = [jid for jid, j in self.jobs.items()
                       if now - j.created > self.ttl_seconds]
            for jid in expired:
                job = self.jobs.pop(jid)
                self._remove_dir(job.out_dir)
        # 顺便清理磁盘上不在任务表里的旧目录
        if self.out_root.exists():
            for child in self.out_root.iterdir():
                if child.is_dir() and now - child.stat().st_mtime > self.ttl_seconds:
                    self._remove_dir(child)

    @staticmethod
    def _remove_dir(path):
        """递归删除目录（失败忽略）。"""
        import shutil
        shutil.rmtree(path, ignore_errors=True)
