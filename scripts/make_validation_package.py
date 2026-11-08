"""准备人工校核包：3 首不同难度的真实曲目 → 模型谱 + 回放音频。

产物目录（容器内 /Docker/piano_validation，Windows 侧 E:\\Docker\\piano_validation）：
    README.md                        校核说明与评分表
    index.json                       曲目清单与指标
    01_<slug>/original.wav           20 秒原始音频截段
    01_<slug>/score.musicxml         模型生成的乐谱
    01_<slug>/score.pdf              乐谱 PDF
    01_<slug>/page_1.png ...         乐谱页面图
    01_<slug>/model_playback.wav     模型谱的合成回放
    01_<slug>/gt_playback.wav        GT 的合成回放（参考）
    01_<slug>/score.mid              模型谱导出的 MIDI
    01_<slug>/info.json              曲目信息
"""

from __future__ import annotations

import json
import os
import shutil
import sys

import librosa
import numpy as np
import pandas as pd
import pretty_midi
import soundfile as sf

sys.path.insert(0, "/Docker/git/piano2")
from app.pipeline import ModelHolder, synthesize_preview, transcribe, write_midi  # noqa: E402
from piano2.constants import MAESTRO_ROOT  # noqa: E402
from piano2.core import NoteEvent  # noqa: E402
from piano2.notation import events_from_midi  # noqa: E402

OUT_ROOT = "/Docker/piano_validation"
MODEL = "/Docker/git/piano2/outputs/M1/combo_offset_loss2/checkpoints/best.pt"
EXCERPT_S = 20.0


def choose_pieces():
    """按音符密度分位选稀疏/中等/密集三首 test 曲目。"""
    df = pd.read_csv(os.path.join(str(MAESTRO_ROOT), "maestro-v3.0.0.csv"))
    test = df[df["split"] == "test"].reset_index(drop=True)
    rows = []
    for _, row in test.iterrows():
        midi_path = os.path.join(str(MAESTRO_ROOT), str(row["midi_filename"]))
        pm = pretty_midi.PrettyMIDI(midi_path)
        n_notes = sum(len(inst.notes) for inst in pm.instruments)
        dur = float(pm.get_end_time())
        rows.append({"row": row, "density": n_notes / max(dur, 1.0),
                     "n_notes": n_notes, "duration": dur})
    rows.sort(key=lambda r: r["density"])
    n = len(rows)
    picks = [rows[int(n * 0.15)], rows[int(n * 0.50)], rows[int(n * 0.85)]]
    return picks


def make_piece(pick, index):
    """生成一首曲目的全部校核产物。"""
    row = pick["row"]
    midi_path = os.path.join(str(MAESTRO_ROOT), str(row["midi_filename"]))
    wav_path = os.path.join(str(MAESTRO_ROOT), str(row["audio_filename"]))
    duration = float(row["duration"])
    start = max(0.0, (duration - EXCERPT_S - 2.0) * 0.30)

    title = f"{row['canonical_composer']} - {row['canonical_title']}"
    slug = f"{index:02d}_{str(row['canonical_title'])[:24].replace(' ', '_').replace('/', '_')}"
    out_dir = os.path.join(OUT_ROOT, slug)
    os.makedirs(out_dir, exist_ok=True)

    # 原始截段
    audio, sr = librosa.load(wav_path, sr=16000, offset=start, duration=EXCERPT_S)
    sf.write(os.path.join(out_dir, "original.wav"), audio, 16000, subtype="PCM_16")

    # 模型转录
    holder = ModelHolder(MODEL)
    result = transcribe(os.path.join(out_dir, "original.wav"), out_dir, holder,
                        title=title, use_offset_end=True)

    # GT 回放（与截段对齐、时间归零）
    gt = events_from_midi(midi_path)
    gt_events = []
    for pitch, s, e, vel in gt:
        if s < start + EXCERPT_S and e > start:
            gt_events.append(NoteEvent(pitch=pitch, start=max(0.0, s - start),
                                       end=max(0.02, min(e, start + EXCERPT_S) - start),
                                       velocity=vel))
    synthesize_preview(gt_events, os.path.join(out_dir, "gt_playback.wav"))
    try:
        write_midi(gt_events, os.path.join(out_dir, "gt.mid"))
    except Exception:
        pass

    os.rename(os.path.join(out_dir, "preview.wav"), os.path.join(out_dir, "model_playback.wav"))
    os.rename(os.path.join(out_dir, "score.mid"), os.path.join(out_dir, "model.mid"))

    info = {
        "title": title,
        "composer": row["canonical_composer"],
        "midi_filename": str(row["midi_filename"]),
        "excerpt_start_s": round(start, 2),
        "excerpt_duration_s": EXCERPT_S,
        "density_notes_per_s": round(float(pick["density"]), 2),
        "n_notes_total": int(pick["n_notes"]),
        "n_pred_events": result["n_events"],
        "n_pages": result["n_pages"],
        "difficulty": ["稀疏", "中等", "密集"][index - 1],
    }
    with open(os.path.join(out_dir, "info.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, ensure_ascii=False, indent=2)
    print(f"完成 {slug}: {title}  ({info['difficulty']}, {info['density_notes_per_s']} 音/秒)")
    return info


README = """# 人工校核包（模型 → 乐谱）

这是当前最佳分类模型（combo_loss2，v3b + onset1 + offset 损失 2.0）在 **3 首不同难度**的真实曲目上的产物。
每首曲目取 20 秒片段，包含：

| 文件 | 内容 |
|---|---|
| `original.wav` | 原始录音截段（16 kHz） |
| `score.musicxml` / `score.pdf` / `page_*.png` | 模型生成的乐谱 |
| `model_playback.wav` | 模型谱的合成回放 |
| `gt_playback.wav` | GT 的合成回放（参考，不是原曲） |
| `score.mid` | 模型谱导出的 MIDI |

## 怎么校核（建议按顺序）

1. **看谱**：先打开 `page_1.png`（或 PDF），不要听音频，判断整体是否「像一首曲子」：
   - 音符数量是否明显过多/过少？
   - 有没有大量连音线、奇怪的时值、空小节？
   - 谱面是否可读（声部、和弦、节奏分组）？
2. **听回放**：依次听 `original.wav`（原曲）→ `gt_playback.wav`（正确答案的合成）→ `model_playback.wav`（模型的合成）：
   - 音高：有没有明显错音、漏音、多音？
   - 时值：音是不是拖太长/太短？节奏是否散掉？
   - 音之间的关系：和弦是否同时响？旋律线是否连贯？
3. **打分（1–5 分）**：
   - 音准（音高对不对）
   - 时值（长短对不对）
   - 关系（和弦/声部/连贯性）
   - 可读性（这份谱给人看，能不能读）
4. 把明显错误的**时间点**记下来（例如「第 3 小节第二拍漏了一个音」），这比整体分数更有用。

## 曲目

（见下方与各目录的 `info.json`）

## 注意

- 这是 20 秒片段，谱面可能跨多页；以 `page_*.png` 为准。
- 合成回放用的是简单加法合成音色，不是真实钢琴；听「时间/音高」即可，不要以音色判断。
"""


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    picks = choose_pieces()
    infos = []
    for i, pick in enumerate(picks, 1):
        infos.append(make_piece(pick, i))
    with open(os.path.join(OUT_ROOT, "index.json"), "w", encoding="utf-8") as fh:
        json.dump(infos, fh, ensure_ascii=False, indent=2)
    readme = README + "\n" + "\n".join(
        f"- `{i:02d}_*`：{info['composer']} - {info['title']}（{info['difficulty']}，"
        f"{info['density_notes_per_s']} 音/秒）"
        for i, info in enumerate(infos, 1)) + "\n"
    with open(os.path.join(OUT_ROOT, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(readme)
    print("校核包完成:", OUT_ROOT)


if __name__ == "__main__":
    main()
