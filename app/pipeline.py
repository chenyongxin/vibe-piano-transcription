"""M4.1 应用后端：把一段音频变成网页需要的全部产物。

产物：
    score.musicxml    乐谱源文件
    page_*.png        分页图片（用于 PDF）
    page_*.svg        分页矢量谱（浏览器内联显示 + 跟随高亮）
    timing.json       播放时间 → 谱面元素 id 的时间轴
    score.pdf         合并后的 PDF
    score.mid         与谱面同一时间轴的 MIDI
    preview.wav       浏览器内回放用的钢琴音色预览

用法（由 jobs.py 调用）：
    holder = ModelHolder(checkpoint)
    result = transcribe(audio_path, out_dir, holder, title=..., progress=fn)
"""

from __future__ import annotations

import json
import os
import re
import time

import numpy as np
import pretty_midi
import pymupdf
import soundfile as sf
import verovio

from piano2.constants import CKPT, DECODE_DEFAULTS
from piano2.decode import DecodeConfig, decode
from piano2.inference import compute_cqt, infer_sliding_window, load_model, pick_device
from piano2.notation import write_musicxml
from . import render_service


class ModelHolder:
    """只加载一次模型，多个任务复用（避免每个任务重复读 135 MB 权重）。"""

    def __init__(self, checkpoint=None, device="auto"):
        """初始化。

        参数:
            checkpoint (str | None): 检查点路径；None 用 constants.CKPT。
            device (str): 设备偏好，见 inference.pick_device。
        """
        self.checkpoint = str(checkpoint) if checkpoint else str(CKPT)
        self.device = pick_device(device)
        self.model = None

    def get(self):
        """懒加载模型并返回（首次调用会真正加载权重）。"""
        if self.model is None:
            self.model = load_model(self.checkpoint, self.device)
        return self.model



def write_midi(events, midi_path, bpm=120.0):
    """把解码出的音符事件写成标准 MIDI（下载或外部播放用）。

    参数:
        events (list): 音符事件（pitch/start/end/velocity）。
        midi_path (str): 输出路径。
        bpm (float): 速度标记（不影响绝对时间，只影响小节网格）。
    """
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    inst = pretty_midi.Instrument(program=0, name="Piano")
    for e in events:
        inst.notes.append(pretty_midi.Note(
            velocity=int(max(1, min(127, e.velocity))),
            pitch=int(e.pitch),
            start=float(e.start),
            end=float(max(e.end, e.start + 0.02)),
        ))
    pm.instruments.append(inst)
    pm.write(str(midi_path))


def synthesize_preview(events, wav_path, sr=22050):
    """用简单加法合成生成钢琴音色预览（不依赖外部音源，离线可用）。

    参数:
        events (list): 音符事件。
        wav_path (str): 输出 wav 路径。
        sr (int): 采样率。
    """
    if not events:
        sf.write(str(wav_path), np.zeros(sr // 2, dtype=np.float32), sr, subtype="PCM_16")
        return
    duration = max(e.end for e in events) + 0.8
    audio = np.zeros(int(duration * sr), dtype=np.float32)
    harmonics = ((1, 1.0), (2, 0.45), (3, 0.22), (4, 0.12))
    for e in events:
        start = int(e.start * sr)
        n_key = max(int((e.end - e.start) * sr), int(0.05 * sr))
        n = n_key + int(0.18 * sr)  # 多留一点释放尾巴，避免爆音
        idx = np.arange(n)
        t = idx / sr
        freq = 440.0 * 2.0 ** ((int(e.pitch) - 69) / 12.0)
        tau = max(0.10, (e.end - e.start) * 0.8)
        env = np.exp(-t / tau) * (1.0 - np.exp(-t / 0.005))
        amp = (float(e.velocity) / 127.0) ** 1.5
        wave = np.zeros(n, dtype=np.float32)
        for h, a in harmonics:
            wave += a * np.sin(2 * np.pi * freq * h * t).astype(np.float32)
        end = min(len(audio), start + n)
        audio[start:end] += (wave[: end - start] * env[: end - start] * amp * 0.28)
    peak = float(np.max(np.abs(audio))) or 1.0
    audio = (audio / peak * 0.85).astype(np.float32)
    sf.write(str(wav_path), audio, sr, subtype="PCM_16")


def pngs_to_pdf(png_paths, pdf_path, max_width=1600, quality=82):
    """把分页 PNG 合并成一个 PDF（先缩图再编码，避免 PDF 几十 MB）。

    参数:
        png_paths (list[str]): 分页 PNG。
        pdf_path (str): 输出 PDF。
        max_width (int): 插入 PDF 的最宽像素（超过会被等比缩小）。
        quality (int): JPEG 质量。
    """
    import io

    from PIL import Image

    doc = pymupdf.open()
    for png in png_paths:
        img = Image.open(png)
        # 乐谱 PNG 常带透明背景；直接 convert("RGB") 会把透明区域变成黑色。
        # 先把 alpha 合成到白底，保证 PDF 是白纸黑谱。
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            img = img.convert("RGBA")
            bg = Image.new("RGB", img.size, "white")
            bg.paste(img, mask=img.getchannel("A"))
            img = bg
        else:
            img = img.convert("RGB")
        if img.width > max_width:
            ratio = max_width / float(img.width)
            img = img.resize((max_width, int(img.height * ratio)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        w_pt, h_pt = img.width * 0.75, img.height * 0.75
        page = doc.new_page(width=w_pt, height=h_pt)
        page.insert_image(pymupdf.Rect(0, 0, w_pt, h_pt), stream=buf.getvalue())
    doc.save(str(pdf_path), deflate=True)
    doc.close()


def _build_timing(xml_path, page_svgs, end_seconds, step=0.1):
    """生成「播放时间 → 谱面元素 id」时间轴，供前端做跟随高亮。

    参数:
        xml_path (str): MusicXML 路径。
        page_svgs (list[tuple[int, str]]): [(页码, SVG 文本)]，用于建立 id → 页码索引。
        end_seconds (float): 播放总时长（秒）。
        step (float): 采样间隔（秒）。

    返回:
        dict: {"step", "duration", "samples": [{"t", "page", "ids"}]}。
    """
    id_page = {}
    for page, svg in page_svgs:
        for match in re.finditer(r'id="([^"]+)"', svg):
            id_page.setdefault(match.group(1), page)
    times = []
    t = 0.0
    while t <= end_seconds + 0.05:
        times.append(t * 1000.0)
        t += step
    raw = render_service.elements_at_times(xml_path, times)
    samples = []
    for ms, data in zip(times, raw):
        ids = [i for i in data.get("notes", []) if i in id_page]
        if ids:
            samples.append({"t": round(ms / 1000.0, 2),
                            "page": id_page[ids[0]],
                            "ids": ids})
    return {"step": step, "duration": round(end_seconds, 2), "samples": samples}


def transcribe(audio_path, out_dir, holder, title="piano2 transcription", progress=None,
               use_offset_end=False):
    """音频 → 全部产物，返回结果字典。

    参数:
        audio_path (str): 输入音频路径。
        out_dir (str): 产物目录。
        holder (ModelHolder): 模型持有者（复用权重）。
        title (str): 乐谱标题。
        progress (callable | None): 进度回调，参数是中文阶段名。

    返回:
        dict: 产物文件名、音符数、耗时、设备。

    异常:
        RuntimeError: 推理/渲染/导出任一环节失败。
    """
    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    def stage(name):
        if progress:
            progress(name)

    stage("提取特征（CQT）")
    cqt = compute_cqt(audio_path)
    stage("模型推理")
    model = holder.get()
    if use_offset_end and getattr(model, "has_offset", False):
        probs = infer_sliding_window(model, cqt, holder.device, with_offset=True)
        stage("解码成音符")
        cfg = DecodeConfig(**DECODE_DEFAULTS, use_offset_end=True)
        events = decode(probs[0], probs[1], probs[2], cfg, offset_probs=probs[3])
    else:
        onset, frame, vel = infer_sliding_window(model, cqt, holder.device)
        stage("解码成音符")
        events = decode(onset, frame, vel, DecodeConfig(**DECODE_DEFAULTS))
    stage("生成乐谱")
    xml_path = os.path.join(out_dir, "score.musicxml")
    write_musicxml([(e.pitch, e.start, e.end, e.velocity) for e in events],
                   audio_path, xml_path, title=title)
    stage("导出 MIDI（与谱面同一时间轴）")
    midi_path = os.path.join(out_dir, "score.mid")
    try:
        midi_bytes = render_service.export_midi(xml_path)
        with open(midi_path, "wb") as fh:
            fh.write(midi_bytes)
    except Exception:
        write_midi(events, midi_path)  # 兜底：verovio 导出失败时用事件写 MIDI
    pmid = pretty_midi.PrettyMIDI(midi_path)
    midi_notes = sorted((n for inst in pmid.instruments for n in inst.notes),
                        key=lambda n: n.start)
    duration = float(pmid.get_end_time())

    stage("渲染乐谱")
    n_pages = render_service.page_count(xml_path)
    n_pages = max(1, min(int(n_pages), 20))  # 防御：异常文件最多渲染 20 页
    step = 0.1
    times_ms, t = [], 0.0
    while t <= duration + 0.05:
        times_ms.append(t * 1000.0)
        t += step
    svg_texts, samples = render_service.render_svg_with_timing(xml_path, n_pages, times_ms)

    svg_names = []
    for pg, svg in enumerate(svg_texts, 1):
        svg_name = f"page_{pg}.svg"
        with open(os.path.join(out_dir, svg_name), "w", encoding="utf-8") as fh:
            fh.write(svg)
        svg_names.append(svg_name)
    timing = {"step": step, "duration": round(duration, 2), "samples": samples}
    with open(os.path.join(out_dir, "timing.json"), "w", encoding="utf-8") as fh:
        json.dump(timing, fh, ensure_ascii=False)

    page_paths = []
    for pg in range(1, n_pages + 1):
        png = os.path.join(out_dir, f"page_{pg}.png")
        render_service.render_png(xml_path, png, page=pg)
        page_paths.append(png)
    stage("导出 PDF")
    pdf_path = os.path.join(out_dir, "score.pdf")
    pngs_to_pdf(page_paths, pdf_path)

    stage("合成回放音频")
    wav_path = os.path.join(out_dir, "preview.wav")
    synthesize_preview(midi_notes, wav_path)

    return {
        "musicxml": "score.musicxml",
        "midi": "score.mid",
        "pdf": "score.pdf",
        "wav": "preview.wav",
        "timing": "timing.json",
        "pages": [os.path.basename(p) for p in page_paths],
        "svg_pages": svg_names,
        "duration_s": round(duration, 2),
        "n_events": len(events),
        "n_pages": len(page_paths),
        "elapsed_s": round(time.time() - t0, 1),
        "device": holder.device,
    }
