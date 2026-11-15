"""为“难段落 + 音画同步”诊断挑选固定短片段。

输入是 MAESTRO 的 MIDI 和 CSV，不需要先跑模型。脚本按几类难点给 8 秒窗口打分：

    fast_repeats  同音快速重复
    fast_runs     快速跑动
    arpeggios     低音到高音的快速琶音
    grace_notes   很短、贴近主音的装饰音
    triplets      近似三等分的连续音
    pedal_dense   踏板持续密集

输出 experiments/hard_passages/manifest.json；路径使用 $DATA_ROOT 占位符，
不写入本机绝对路径。

用法（容器内）：
    python experiments/hard_passages/select_clips.py
"""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi

from piano2.constants import MAESTRO_ROOT

OUT = Path(__file__).resolve().parent
DURATION_S = 8.0
STEP_S = 2.0
MIN_NOTES = 12


def note_window(notes, start, end):
    """返回落在窗口内的音符（按起音过滤）。"""
    return [n for n in notes if start <= n.start < end]


def cc_window(pm, start, end):
    """返回窗口内的 CC64 事件。"""
    values = []
    for inst in pm.instruments:
        for cc in inst.control_changes:
            if cc.number == 64 and start <= cc.time < end:
                values.append((cc.time, cc.value))
    return sorted(values)


def pedal_fraction(pm, start, end):
    """估算窗口内踏板踩下的时间比例。"""
    events = cc_window(pm, start, end)
    if not events:
        return 0.0
    state = False
    on_since = start
    on_time = 0.0
    for t, v in events:
        if v >= 64 and not state:
            state = True
            on_since = t
        elif v < 64 and state:
            on_time += max(0.0, t - on_since)
            state = False
    if state:
        on_time += max(0.0, end - on_since)
    return on_time / max(1e-6, end - start)


def score_window(notes, pm, start, end):
    """给窗口计算各类难点特征。"""
    ns = sorted(note_window(notes, start, end), key=lambda n: n.start)
    if len(ns) < MIN_NOTES:
        return None
    onsets = np.array([n.start for n in ns], dtype=float)
    durations = np.array([n.end - n.start for n in ns], dtype=float)
    iois = np.diff(onsets)
    iois = iois[iois > 0]
    if len(iois) == 0:
        return None
    median_ioi = float(np.median(iois))
    fast_pairs = sum(
        1 for i in range(1, len(ns))
        if ns[i].pitch == ns[i - 1].pitch and 0 < ns[i].start - ns[i - 1].start < 0.12
    )
    grace = sum(1 for d in durations if d < 0.08)
    # 琶音：连续 4 个不同音高、间隔 0.02–0.12s
    arpeggio = 0
    for i in range(len(ns) - 3):
        seq = ns[i:i + 4]
        gaps = [seq[j + 1].start - seq[j].start for j in range(3)]
        if len({n.pitch for n in seq}) == 4 and all(0.02 <= g <= 0.12 for g in gaps):
            arpeggio += 1
    # 近似三连音：三连音间隔接近，且总跨度 < 0.5s
    triplet = 0
    if len(onsets) >= 3:
        for i in range(len(onsets) - 2):
            gaps = [onsets[i + 1] - onsets[i], onsets[i + 2] - onsets[i + 1]]
            if 0.03 <= gaps[0] <= 0.20 and abs(gaps[0] - gaps[1]) <= 0.02:
                triplet += 1
    return {
        "notes": len(ns),
        "notes_per_second": round(len(ns) / (end - start), 2),
        "median_ioi_ms": round(median_ioi * 1000, 1),
        "fast_repeats": fast_pairs,
        "arpeggio_pairs": arpeggio,
        "grace_notes": grace,
        "triplet_pairs": triplet,
        "pedal_fraction": round(pedal_fraction(pm, start, end), 2),
    }


def window_candidates(pm, start, end, rows):
    """对候选窗口打分并返回特征。"""
    notes = []
    for inst in pm.instruments:
        notes.extend(inst.notes)
    notes.sort(key=lambda n: n.start)
    feats = []
    t = start
    while t + DURATION_S <= end + 1e-6:
        s = score_window(notes, pm, t, t + DURATION_S)
        if s:
            s["start"] = round(t, 2)
            s["end"] = round(t + DURATION_S, 2)
            feats.append(s)
        t += STEP_S
    return feats


def main():
    """扫描 MAESTRO 非训练 split，选出每类最佳窗口。"""
    df = pd.read_csv(MAESTRO_ROOT / "maestro-v3.0.0.csv")
    df = df[df["split"].isin(["validation", "test"])].reset_index(drop=True)
    categories = ["fast_repeats", "fast_runs", "arpeggios", "grace_notes",
                  "triplets", "pedal_dense", "long_span"]
    best = {c: None for c in categories}
    used_tracks = set()

    for idx, row in df.iterrows():
        midi_path = MAESTRO_ROOT / str(row["midi_filename"])
        try:
            pm = pretty_midi.PrettyMIDI(str(midi_path))
        except Exception:
            continue
        end = pm.get_end_time()
        if end < 12.0:
            continue
        # 每首最多看 6 个起点，控制扫描时间；覆盖开头、中间、结尾。
        starts = sorted(set(int(x) for x in np.linspace(0, max(0, end - DURATION_S), 6)))
        for s in starts:
            feats = window_candidates(pm, float(s), min(end, float(s) + DURATION_S), row)
            if not feats:
                continue
            for f in feats:
                for c in categories:
                    if c == "fast_repeats":
                        score = f["fast_repeats"]
                    elif c == "fast_runs":
                        score = f["notes_per_second"] if f["median_ioi_ms"] < 140 else 0
                    elif c == "arpeggios":
                        score = f["arpeggio_pairs"]
                    elif c == "grace_notes":
                        score = f["grace_notes"]
                    elif c == "triplets":
                        score = f["triplet_pairs"]
                    elif c == "pedal_dense":
                        score = f["pedal_fraction"] * 10 + f["notes"] / 100
                    else:  # long_span: 段落本身较长时优先
                        score = f["notes"] / 10
                    if score <= 0:
                        continue
                    if best[c] is None or score > best[c]["_score"]:
                        best[c] = {
                            "_score": round(float(score), 3),
                            "category": c,
                            "track": str(row["midi_filename"]).replace("\\", "/"),
                            "audio": str(row["audio_filename"]).replace("\\", "/"),
                            "split": str(row["split"]),
                            **{k: v for k, v in f.items() if k not in ("_score",)},
                        }
        if idx % 20 == 0:
            print(f"  scanned {idx + 1}/{len(df)}", flush=True)

    clips = []
    # 保证每个类别的片段来自不同曲目（若不够，则允许重复）。
    for c in categories:
        item = best[c]
        if item is None:
            continue
        item = dict(item)
        track = item["track"]
        if track in used_tracks:
            # 同一个曲目只保留一个类别，避免 manifest 过于集中。
            continue
        used_tracks.add(track)
        item["id"] = c
        item["reason"] = {
            "fast_repeats": "同音快速重复",
            "fast_runs": "快速跑动",
            "arpeggios": "快速琶音",
            "grace_notes": "短倚音/装饰音",
            "triplets": "近似三连音",
            "pedal_dense": "踏板密集",
            "long_span": "较长连续片段",
        }[c]
        clips.append(item)

    manifest = {
        "version": "hard_passages_v1",
        "description": "MAESTRO validation/test 中的 8 秒难片段，用于失败归因；路径使用 $DATA_ROOT 占位符。",
        "duration_s": DURATION_S,
        "categories": categories,
        "clips": clips,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2)[:4000])
    print("wrote", OUT / "manifest.json")


if __name__ == "__main__":
    main()
