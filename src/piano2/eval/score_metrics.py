"""M0：乐谱（MusicXML）可读性指标。

为什么单独写一套：music21 读谱会做语义解释，测出来的东西和「渲染器实际画出来的音符头」
不完全一致。这里用 lxml 直接读 XML 结构，和「人眼在谱面上数出来的东西」一一对应。

用法（在容器内）：
    python -c "import sys; sys.path.insert(0,'/Docker/git/piano_practice/m0'); import score_metrics; print(score_metrics.score_metrics('xxx.musicxml', audio_duration_s=65.9))"
"""

from __future__ import annotations

import numpy as np

from piano2.eval import mxlib

# 二分音符以下（含）在谱面上是「黑音符头」
BLACK_TYPES = {"eighth", "16th", "32nd", "64th", "128th", "256th"}
# 16 分音符以下（含）
SHORT16_TYPES = {"16th", "32nd", "64th", "128th", "256th"}


def _voice_key(event):
    """把 MusicXML 里缺失的 voice 统一成 '1'，避免 None 把声部数算错。"""
    return event.get("voice") or "1"


def score_metrics(xml_path, audio_duration_s=None):
    """算一份 MusicXML 的可读性指标。

    参数:
        xml_path (str): MusicXML 文件路径。
        audio_duration_s (float | None): 对应音频时长（秒）。给了才计算
            onsets_per_sec 与时长一致性指标。

    返回:
        dict: 指标字典。关键字段：
            n_noteheads: 音符头总数（含和弦内各音）
            n_onsets: 起音事件数（和弦算 1 个）
            black_ratio: 黑音符头占比（八分及更短）
            short16_ratio: 16 分及更短占比
            tie_ratio: 连音线数 / 起音事件数
            rest_duration_ratio: 休止符时长 / 谱面总时长
            voices_per_measure_mean / voices_per_measure_max: 每小节声部数
            polyphony_max / polyphony_mean: 同时发声数
            measures_per_part: 各声部小节数列表
            measures_consistent: 各声部小节数是否一致
            score_duration_s: 谱面时长（按谱面标注 BPM 折算）
            duration_err_ratio: 谱面时长与音频时长的相对误差
            onsets_per_sec: 每秒起音数
            retrigger_50ms / retrigger_100ms: 同音高极短间隔重复次数
            type_hist: 音符类型直方图
    """
    mx = mxlib.parse_musicxml(xml_path)
    tempo = mx.get("tempo") or 120.0
    parts = mx.get("parts", [])

    noteheads = []  # (part_id, event)
    rests = []
    for part in parts:
        for ev in part["events"]:
            if ev.get("grace"):
                continue
            if ev.get("is_rest"):
                rests.append(ev)
            elif ev.get("pitch") is not None:
                noteheads.append((part["id"], ev))

    n_noteheads = len(noteheads)
    n_onsets = sum(1 for _, e in noteheads if not e.get("chord"))
    type_hist = dict(sorted({k: v for k, v in _counter(e.get("type") for _, e in noteheads).items()}.items()))
    n_black = sum(v for k, v in type_hist.items() if k in BLACK_TYPES)
    n_short16 = sum(v for k, v in type_hist.items() if k in SHORT16_TYPES)
    n_untyped = sum(v for k, v in type_hist.items() if k in (None, "None", ""))

    tie_starts = sum(1 for _, e in noteheads if e.get("tie_start"))

    # 谱面总时长 = 所有事件 (onset + dur) 的最大值
    all_events = [e for part in parts for e in part["events"]]
    total_q = max((e["onset"] + e["dur"] for e in all_events), default=0.0)
    rest_q = sum(e["dur"] for e in rests if not e.get("grace"))

    # 小节与声部
    measures_per_part = [len(part["measures"]) for part in parts]
    voices_per_measure = []
    for part in parts:
        by_measure = {}
        for ev in part["events"]:
            if ev.get("is_rest") or ev.get("grace"):
                continue
            by_measure.setdefault(ev["measure"], set()).add(_voice_key(ev))
        voices_per_measure.extend(len(v) for v in by_measure.values())

    # 同时发声数：用区间计数，O(N log N)
    polyphony_max = 0
    polyphony_mean = 0.0
    if noteheads:
        starts = np.array([e["onset"] for _, e in noteheads], dtype=float)
        ends = np.array([e["onset"] + max(e["dur"], 1e-9) for _, e in noteheads], dtype=float)
        starts.sort()
        ends.sort()
        probe = np.unique(np.round(np.linspace(0, total_q, 400), 6))
        active = np.searchsorted(starts, probe, side="right") - np.searchsorted(ends, probe, side="left")
        polyphony_max = int(active.max()) if len(active) else 0
        polyphony_mean = float(active.mean()) if len(active) else 0.0

    # 同音高极短间隔重复（碎音）
    seconds_per_quarter = 60.0 / float(tempo)
    by_pitch = {}
    for _, e in noteheads:
        if not e.get("chord"):
            by_pitch.setdefault(e["pitch"], []).append(e["onset"])
    retrigger_50 = 0
    retrigger_100 = 0
    for onsets in by_pitch.values():
        onsets.sort()
        for a, b in zip(onsets, onsets[1:]):
            dt = (b - a) * seconds_per_quarter
            if dt < 0.05:
                retrigger_50 += 1
            if dt < 0.10:
                retrigger_100 += 1

    # 音域
    pitch_range = {}
    for part in parts:
        ps = [e["pitch"] for e in part["events"] if e.get("pitch") is not None and not e.get("grace")]
        if ps:
            pitch_range[part["id"]] = (int(min(ps)), int(max(ps)))

    score_duration_s = total_q * seconds_per_quarter
    # 休止占比：两个声部各自都会写休止，直接相加会超过 1；按声部数归一
    rest_duration_ratio = None
    if total_q > 0 and parts:
        rest_duration_ratio = rest_q / (total_q * len(parts))
    duration_err = None
    onsets_per_sec = None
    if audio_duration_s:
        duration_err = (score_duration_s - audio_duration_s) / audio_duration_s
        onsets_per_sec = n_onsets / float(audio_duration_s)

    return {
        "path": xml_path,
        "tempo": float(tempo),
        "n_parts": len(parts),
        "n_noteheads": n_noteheads,
        "n_onsets": n_onsets,
        "notehead_inflation": (n_noteheads / n_onsets) if n_onsets else None,
        "black_ratio": (n_black / n_noteheads) if n_noteheads else None,
        "short16_ratio": (n_short16 / n_noteheads) if n_noteheads else None,
        "tie_count": tie_starts,
        "tie_ratio": (tie_starts / n_onsets) if n_onsets else None,
        "rest_count": len(rests),
        "rest_duration_ratio": rest_duration_ratio,
        "voices_per_measure_mean": (sum(voices_per_measure) / len(voices_per_measure)) if voices_per_measure else 0.0,
        "voices_per_measure_max": max(voices_per_measure) if voices_per_measure else 0,
        "polyphony_max": polyphony_max,
        "polyphony_mean": polyphony_mean,
        "measures_per_part": measures_per_part,
        "measures_consistent": len(set(measures_per_part)) <= 1,
        "score_duration_s": score_duration_s,
        "duration_err_ratio": duration_err,
        "onsets_per_sec": onsets_per_sec,
        "retrigger_50ms": retrigger_50,
        "retrigger_100ms": retrigger_100,
        "type_hist": type_hist,
        "n_untyped": n_untyped,
        "pitch_range": pitch_range,
    }


def _counter(iterable):
    """轻量 Counter，避免额外 import；返回 dict。"""
    out = {}
    for item in iterable:
        out[item] = out.get(item, 0) + 1
    return out


def summarize_metrics(rows):
    """把多份乐谱的指标汇总成中位数表。

    参数:
        rows (list[dict]): score_metrics() 的返回列表。

    返回:
        dict: 每个指标的中位数（None 跳过），以及样本数。
    """
    keys = [
        "n_noteheads", "n_onsets", "black_ratio", "short16_ratio", "tie_ratio",
        "rest_duration_ratio", "voices_per_measure_mean", "voices_per_measure_max",
        "polyphony_max", "polyphony_mean", "retrigger_100ms", "onsets_per_sec",
        "duration_err_ratio",
    ]
    out = {"n_scores": len(rows)}
    for key in keys:
        vals = [r[key] for r in rows if r.get(key) is not None]
        out[key] = float(np.median(vals)) if vals else None
    out["frac_measures_consistent"] = (
        sum(1 for r in rows if r.get("measures_consistent")) / len(rows) if rows else None
    )
    return out
