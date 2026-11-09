"""解码后的规则后处理：八度重复抑制 + 同音高重叠截断。

两个问题都来自 decoder 的结构弱点，不需要重训：

1. 八度重复：模型把基频的八度泛音也检成音符（同起止、同力度、pitch 差 12）。
   若低八度存在且两音几乎完全重合，则删掉高八度。
2. 同音高重叠：peak_frame 解码中，同一键的重复音可能都被赋予「下一个 release」
   作为结束，导致前一个音覆盖到后一个音。同一键不可能同时响，故前音必须在后音
   开始前结束。
"""

from __future__ import annotations

import numpy as np

from piano2.core import NoteEvent


def cap_same_pitch_overlaps(events, min_dur=0.03, gap=0.02):
    """同一音高的相邻音符不能重叠；前一个音在后一个音开始前结束。

    参数:
        events (list[NoteEvent]): 已解码事件。
        min_dur (float): 截断后允许的最短时长（秒）。
        gap (float): 两个同音高音符之间保留的最小间隔（秒）。

    返回:
        list[NoteEvent]: 修正后的新事件列表。
    """
    by_pitch = {}
    for ev in events:
        by_pitch.setdefault(ev.pitch, []).append(ev)
    fixed = []
    for pitch, group in by_pitch.items():
        group = sorted(group, key=lambda e: e.start)
        for i, ev in enumerate(group):
            end = ev.end
            if i + 1 < len(group):
                nxt = group[i + 1]
                if end > nxt.start - gap:
                    end = max(ev.start + min_dur, nxt.start - gap)
            fixed.append(NoteEvent(pitch, ev.start, float(end), ev.velocity))
    return sorted(fixed, key=lambda e: (e.start, e.pitch))


def suppress_octave_duplicates(events, onset=None, frame=None,
                               start_tol=0.05, end_tol=0.10,
                               velocity_tol=30, onset_ratio=0.85):
    """删除与低八度音几乎完全重合的高八度误检。

    参数:
        events (list[NoteEvent]): 已解码事件。
        onset (numpy.ndarray | None): (T,88) onset 概率，用于判断证据强弱。
        frame (numpy.ndarray | None): (T,88) frame 概率，用于判断独立发声证据。
        start_tol (float): 起始时间容差（秒）。
        end_tol (float): 结束时间容差（秒）。
        velocity_tol (int): 力度容差。
        onset_ratio (float): 高八度 onset 概率至少达到低八度的该比例才保留。

    返回:
        tuple[list[NoteEvent], int]: (剩余事件, 删除数量)。
    """
    by_pitch = {}
    for ev in events:
        by_pitch.setdefault(ev.pitch, []).append(ev)
    fps = 31.25
    removed = 0
    keep = []
    for pitch in sorted(by_pitch):
        lower = by_pitch.get(pitch - 12, [])
        for ev in by_pitch[pitch]:
            duplicate = False
            for lo in lower:
                if abs(lo.start - ev.start) > start_tol:
                    continue
                if abs(lo.end - ev.end) > end_tol:
                    continue
                if abs(lo.velocity - ev.velocity) > velocity_tol:
                    continue
                if onset is not None:
                    k = pitch - 21
                    kl = pitch - 12 - 21
                    hi_frame = int(round(ev.start * fps))
                    lo_frame = int(round(lo.start * fps))
                    if 0 <= hi_frame < onset.shape[0] and 0 <= kl < 88:
                        hi_p = float(onset[max(0, hi_frame - 1):hi_frame + 2, k].max())
                        lo_p = float(onset[max(0, lo_frame - 1):lo_frame + 2, kl].max())
                        if hi_p >= onset_ratio * lo_p:
                            continue  # 高八度自身证据不弱，可能是真实八度齐奏
                duplicate = True
                break
            if duplicate:
                removed += 1
            else:
                keep.append(ev)
    return sorted(keep, key=lambda e: (e.start, e.pitch)), removed


def postprocess_events(events, onset=None, frame=None):
    """组合两个后处理，返回 (events, stats)。"""
    capped = cap_same_pitch_overlaps(events)
    clean, removed = suppress_octave_duplicates(capped, onset=onset, frame=frame)
    return clean, {"octave_removed": removed}
