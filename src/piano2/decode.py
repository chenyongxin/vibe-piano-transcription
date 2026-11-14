"""M0.5：可配置的解码器（概率图 → 音符事件）。

为什么重写：原 `transcribe_readable.probabilities_to_events` 把参数写死（最短音长 6 帧、
合并阈值 0.2 s 等），没法做参数搜索。这里把同一套逻辑显式参数化，行为与原实现保持一致，
方便做对照实验。

谐波抑制三模式：
    off         完全不抑制
    global      复刻原实现：整首曲子的时间列做乘法，锚点之间连乘
    conditional 目标实现：逐帧、局部时间窗、取最强衰减（M0.5 步骤 3 实现）

用法（容器内）：
    python -c "import decode_v2; print(decode_v2.DecodeConfig())"
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, replace

import numpy as np
from scipy.signal import find_peaks

from piano2.core import (  # noqa: E402
    HARMONIC_OFFSETS,
    HOP_LENGTH,
    SR,
    NoteEvent,
    suppress_harmonic_onsets,
)


@dataclass(frozen=True)
class DecodeConfig:
    """一组解码参数。

    属性:
        onset_threshold (float): onset 峰值高度阈值。
        onset_prominence (float): 峰值 prominence，用来区分"尖峰"和"慢坡"。
        frame_off_threshold (float): frame 概率低于此值开始累计 release 计数。
        min_note_frames (int): 最短音长（帧），1 帧约 32 ms。
        release_frames (int): 连续多少帧低于阈值才判定音符结束（迟滞）。
        min_reattack_gap (float): 同音高两次击打的最小间隔（秒）。
        merge_gap (float): 同音高相邻音符的合并阈值（秒）；越大越容易把重复音并成一个。
        harmonic_mode (str): "off" / "global" / "conditional"。
        harmonic_strength (float): 谐波抑制强度 0~1。
        peak_distance (int): 同一琴键两个 onset 峰的最小帧距。
        method (str): "state_machine"（原迟滞状态机）或 "peak_frame"（峰值定起点、
            frame 定终点，不做重击/最短时值/合并等结构门限）。
        harmonic_window_frames (int): 条件式抑制的时间窗半宽（帧）。
        harmonic_frame_max (float): 条件式抑制的 frame 门限：目标键在该窗内
            frame 概率均值高于此值时不抑制（说明它可能真在响）。
        frame_gate (float): 帧一致性门控：起音时刻该键 frame 概率低于此值的峰直接丢弃；
            0 表示关闭。
    """

    onset_threshold: float = 0.40
    onset_prominence: float = 0.06
    frame_off_threshold: float = 0.18
    min_note_frames: int = 6
    release_frames: int = 10
    min_reattack_gap: float = 0.15
    merge_gap: float = 0.20
    harmonic_mode: str = "off"
    harmonic_strength: float = 0.5
    peak_distance: int = 5
    method: str = "state_machine"
    harmonic_window_frames: int = 5
    harmonic_frame_max: float = 1.0
    frame_gate: float = 0.0
    use_offset_end: bool = False
    offset_threshold: float = 0.5
    offset_min_frames: int = 2
    offset_max_frames: int = 160

    def with_(self, **kwargs):
        """返回替换了若干字段的新配置（dataclass 不可变，避免调参时相互污染）。"""
        return replace(self, **kwargs)


def _round1_anchors(onset, frame, vel, cfg):
    """第一轮：用更严格的阈值找高置信度音符，返回它们用到的音高集合（锚点）。"""
    strict = cfg.with_(
        onset_threshold=cfg.onset_threshold + 0.10,
        onset_prominence=cfg.onset_prominence + 0.04,
    )
    events = _decode_without_harmonic(onset, frame, vel, strict)
    return sorted({e.pitch for e in events})


def _decode_for_key(onset_curve, frame_curve, vel_curve, midi_pitch, cfg):
    """对单个琴键跑「峰值 + 迟滞状态机」，返回该键的音符事件。

    参数:
        onset_curve (numpy.ndarray): (T,) 该键的 onset 概率。
        frame_curve (numpy.ndarray): (T,) 该键的 frame 概率。
        vel_curve (numpy.ndarray): (T,) 该键的力度预测。
        midi_pitch (int): 该键的 MIDI 音高（21–108）。
        cfg (DecodeConfig): 解码参数。

    返回:
        list[NoteEvent]: 该键的音符列表。
    """
    n_frames = len(onset_curve)
    peaks, _ = find_peaks(
        onset_curve,
        height=cfg.onset_threshold,
        prominence=cfg.onset_prominence,
        distance=cfg.peak_distance,
    )
    if len(peaks) == 0:
        return []

    peak_set = set(int(p) for p in peaks)
    min_duration = cfg.min_note_frames * HOP_LENGTH / SR
    min_gap_frames = int(cfg.min_reattack_gap * SR / HOP_LENGTH)
    frame_to_time = lambda idx: idx * HOP_LENGTH / SR  # noqa: E731

    events = []
    active = False
    note_start = 0.0
    note_vel = 64
    release_count = 0
    last_attack_frame = -999

    for frame_idx in range(n_frames):
        is_peak = frame_idx in peak_set
        now = frame_to_time(frame_idx)
        can_reattack = (frame_idx - last_attack_frame) >= min_gap_frames

        if is_peak and can_reattack:
            if active:
                # 正在响时又出现可重击的峰：先把上一个音收尾（不短于最小时值才保留）
                if now - note_start >= min_duration:
                    events.append(NoteEvent(midi_pitch, note_start, now, note_vel))
            active = True
            note_start = now
            note_vel = int(np.clip(vel_curve[frame_idx] * 127, 24, 118))
            last_attack_frame = frame_idx
            release_count = 0
            continue

        if active:
            if frame_curve[frame_idx] < cfg.frame_off_threshold:
                release_count += 1
            else:
                release_count = 0

            if release_count >= cfg.release_frames:
                note_end = frame_to_time(max(frame_idx - cfg.release_frames + 1, 0))
                if note_end - note_start >= min_duration:
                    events.append(NoteEvent(midi_pitch, note_start, note_end, note_vel))
                active = False
                release_count = 0

    if active:
        note_end = frame_to_time(n_frames - 1)
        if note_end - note_start >= min_duration:
            events.append(NoteEvent(midi_pitch, note_start, note_end, note_vel))
    return events


def _merge_same_pitch(events, gap_threshold):
    """合并同音高、间隙很小的碎片音符（重复触发会被并成一个）。

    参数:
        events (list[NoteEvent]): 待合并事件。
        gap_threshold (float): 允许合并的最大间隙（秒）。

    返回:
        list[NoteEvent]: 合并后按时间排序的事件。
    """
    merged = []
    for pitch in range(128):
        pitch_events = sorted((e for e in events if e.pitch == pitch), key=lambda e: e.start)
        for event in pitch_events:
            if merged and merged[-1].pitch == pitch and event.start - merged[-1].end <= gap_threshold:
                merged[-1].end = max(merged[-1].end, event.end)
                merged[-1].velocity = max(merged[-1].velocity, event.velocity)
            else:
                merged.append(event)
    return sorted(merged, key=lambda e: (e.start, e.pitch))


def decode_peak_frame(onset, frame, vel, cfg):
    """峰值定起点、frame 定终点；不做重击间隔、最短时值、同音合并等结构门限。

    为什么这样设计：M0.5 步骤 2 的消融表明，状态机的那些门限会无条件删掉 30–40% 的
    真实起音（参数怎么调都救不回来）；而 onset-only 指标只关心起点，不关心时值。
    所以把「起点」和「终点」解耦：起点完全交给 onset 峰值，终点交给 frame 概率。

    参数:
        onset (numpy.ndarray): (T,88) onset 概率。
        frame (numpy.ndarray): (T,88) frame 概率。
        vel (numpy.ndarray): (T,88) 力度预测。
        cfg (DecodeConfig): 解码参数（用 onset_threshold/onset_prominence/
            frame_off_threshold/release_frames/peak_distance）。

    返回:
        list[NoteEvent]: 音符事件，按开始时间排序。
    """
    from numpy.lib.stride_tricks import sliding_window_view

    T = len(onset)
    r = max(int(cfg.release_frames), 1)
    max_len_frames = int(8.0 * SR / HOP_LENGTH)  # 单音最长 8 秒，防止 frame 头部异常导致超长音
    events = []
    below = frame < cfg.frame_off_threshold
    for key_idx in range(88):
        peaks, _ = find_peaks(
            onset[:, key_idx], height=cfg.onset_threshold,
            prominence=cfg.onset_prominence, distance=cfg.peak_distance,
        )
        if len(peaks) == 0:
            continue
        if T >= r:
            window = sliding_window_view(below[:, key_idx], r)
            off_pos = np.where(window.all(axis=1))[0] + (r - 1)  # 每个释放窗口的最后一帧
        else:
            off_pos = np.array([T - 1])
        for peak in peaks:
            start_frame = int(peak)
            if cfg.frame_gate > 0.0 and frame[start_frame, key_idx] < cfg.frame_gate:
                # 帧一致性门控：起音时刻 frame 几乎为 0，说明更像泛音泄漏，丢弃
                continue
            j = int(np.searchsorted(off_pos, start_frame, side="left"))
            if j < len(off_pos):
                end_frame = int(off_pos[j]) - r + 1
            else:
                end_frame = T - 1
            end_frame = min(end_frame, start_frame + max_len_frames)
            start = start_frame * HOP_LENGTH / SR
            end = end_frame * HOP_LENGTH / SR
            if end <= start:
                end = start + 0.02
            velocity = int(np.clip(vel[start_frame, key_idx] * 127, 24, 118))
            events.append(NoteEvent(key_idx + 21, start, end, velocity))
    return sorted(events, key=lambda e: (e.start, e.pitch))


def _decode_without_harmonic(onset, frame, vel, cfg):
    """不做谐波抑制的解码主流程。"""
    all_events = []
    for key_idx in range(88):
        all_events.extend(
            _decode_for_key(onset[:, key_idx], frame[:, key_idx], vel[:, key_idx], key_idx + 21, cfg)
        )
    return _merge_same_pitch(all_events, cfg.merge_gap)


def apply_offset_ends(events, offset_probs, cfg):
    """用 offset 头（键释放事件）修正每个音符的结束时间。

    规则：从起点后 offset_min_frames 帧开始，在 offset_max_frames 帧内找第一个
    概率超过 offset_threshold 的 offset 峰；找到就把结束时间设为该峰，否则保留
    原来由 frame 推断的结束时间。

    参数:
        events (list[NoteEvent]): 已解码的音符事件。
        offset_probs (numpy.ndarray | None): (T, 88) offset 概率；None 或未启用时原样返回。
        cfg (DecodeConfig): 需要 use_offset_end / offset_threshold /
            offset_min_frames / offset_max_frames。

    返回:
        list[NoteEvent]: 修正结束时间后的事件。
    """
    if offset_probs is None or not cfg.use_offset_end:
        return events
    from scipy.signal import find_peaks

    fps = SR / HOP_LENGTH
    total = offset_probs.shape[0]
    fixed = []
    for event in events:
        key = event.pitch - 21
        start = int(round(event.start * fps))
        lo = start + cfg.offset_min_frames
        hi = min(total, start + cfg.offset_max_frames)
        end = event.end
        if 0 <= key < 88 and hi - lo >= 2:
            peaks, _ = find_peaks(offset_probs[lo:hi, key], height=cfg.offset_threshold)
            if len(peaks):
                end = (lo + int(peaks[0])) / fps
        end = max(end, event.start + cfg.offset_min_frames / fps)
        fixed.append(NoteEvent(pitch=event.pitch, start=event.start,
                               end=float(end), velocity=event.velocity))
    return fixed


def decode(onset, frame, vel, cfg, offset_probs=None):
    """把概率矩阵解码成音符事件。

    参数:
        onset (numpy.ndarray): (T, 88) onset 概率。
        frame (numpy.ndarray): (T, 88) frame 概率。
        vel (numpy.ndarray): (T, 88) 力度预测（0~1）。
        cfg (DecodeConfig): 解码参数。
        offset_probs (numpy.ndarray | None): (T, 88) offset 概率；提供且
            cfg.use_offset_end=True 时，用它来定每个音的结束时间。

    返回:
        list[NoteEvent]: 按开始时间排序的音符事件。

    异常:
        ValueError: harmonic_mode 取值非法。
    """
    working_onset = onset
    if cfg.harmonic_mode == "global":
        anchors = _round1_anchors(onset, frame, vel, cfg)
        if anchors:
            working_onset = suppress_harmonic_onsets(onset, anchors, strength=cfg.harmonic_strength)
    elif cfg.harmonic_mode == "conditional":
        working_onset = conditional_suppress(onset, frame, cfg)
    elif cfg.harmonic_mode != "off":
        raise ValueError(f"未知谐波抑制模式: {cfg.harmonic_mode}")
    if cfg.method == "peak_frame":
        events = decode_peak_frame(working_onset, frame, vel, cfg)
    elif cfg.method == "state_machine":
        events = _decode_without_harmonic(working_onset, frame, vel, cfg)
    else:
        raise ValueError(f"未知解码方法: {cfg.method}")
    return apply_offset_ends(events, offset_probs, cfg)


def conditional_suppress(onset, frame, cfg):
    """条件式谐波抑制：逐帧、局部时间窗、取最强衰减（不连乘）。

    和原实现的区别（原实现在 `transcribe_readable.suppress_harmonic_onsets`）：
        1. 原实现乘整条时间列；这里只作用在锚点前后 ±harmonic_window_frames 帧；
        2. 原实现多个锚点连乘；这里同一格取最强衰减（最小值）；
        3. 增加 frame 证据门限：目标键在该窗内 frame 均值高于 harmonic_frame_max
           时不抑制（说明它可能真的在响，是八度齐奏而不是泛音泄漏）。

    参数:
        onset (numpy.ndarray): (T,88) onset 概率。
        frame (numpy.ndarray): (T,88) frame 概率。
        cfg (DecodeConfig): 需要 harmonic_strength / harmonic_window_frames /
            harmonic_frame_max / onset_threshold / onset_prominence / peak_distance。

    返回:
        numpy.ndarray: 抑制后的 (T,88) onset 概率。
    """
    T = len(onset)
    factor = np.ones_like(onset, dtype=np.float32)
    # 锚点：用更严格的阈值找高置信度起音，只取"键 + 帧位置"
    strict_thr = float(min(cfg.onset_threshold + 0.15, 0.98))
    half = max(int(cfg.harmonic_window_frames), 1)
    for key_idx in range(88):
        peaks, _ = find_peaks(
            onset[:, key_idx], height=strict_thr,
            prominence=cfg.onset_prominence, distance=cfg.peak_distance,
        )
        for peak in peaks:
            peak = int(peak)
            t0 = max(0, peak - half)
            t1 = min(T, peak + half + 1)
            for offset, decay in HARMONIC_OFFSETS:
                target = key_idx + offset
                if not (0 <= target < 88):
                    continue
                if float(frame[t0:t1, target].mean()) > cfg.harmonic_frame_max:
                    continue  # frame 支持它，可能是真音（八度齐奏等）
                value = 1.0 - decay * cfg.harmonic_strength
                factor[t0:t1, target] = np.minimum(factor[t0:t1, target], value)
    return onset * factor


def _refine_peak(curve, peak):
    """用左右邻帧做三角形插值，返回亚帧的峰值位置（单位：帧）。

    参数:
        curve (numpy.ndarray): (T,) 单键的回归输出（0–1）。
        peak (int): 局部最大帧号。

    返回:
        float: 亚帧位置（可能带小数），单位帧。
    """
    if peak <= 0 or peak >= len(curve) - 1:
        return float(peak)
    y_a = float(curve[peak - 1])
    y_b = float(curve[peak])
    y_c = float(curve[peak + 1])
    if y_b <= max(y_a, y_c) or y_b <= 1e-6:
        return float(peak)
    if y_c > y_a:
        delta = 0.5 * (y_c - y_a) / max(y_b - y_a, 1e-6)
    else:
        delta = -0.5 * (y_a - y_c) / max(y_b - y_c, 1e-6)
    return float(peak) + float(np.clip(delta, -0.5, 0.5))


def refine_onsets(events, onset_reg, radius_frames=1, min_value=0.10):
    """用 onset_reg 头把分类起点从整数帧修到帧内。

    对每个已解码的 NoteEvent，只在它起点附近 ±radius_frames 帧内找 onset_reg
    的局部峰，再用三角插值得到亚帧位置；音符是否存在、结束时间和力度都不变。

    参数:
        events (list[NoteEvent]): 分类解码得到的音符事件。
        onset_reg (numpy.ndarray): (T, 88) 的 onset_reg 回归输出，值域 0–1。
        radius_frames (int): 在候选起点两侧搜索的帧数，默认 1。
        min_value (float): 局部峰的最低值，低于它不修正，默认 0.10。

    返回:
        list[NoteEvent]: 起点被修正后的新事件列表。

    异常:
        ValueError: onset_reg 不是二维数组。
    """
    if onset_reg.ndim != 2:
        raise ValueError(f"onset_reg 应为 (T, 88)，实际 {onset_reg.shape}")
    fps = SR / HOP_LENGTH
    total_frames = onset_reg.shape[0]
    if total_frames < 3 or not events:
        return list(events)
    refined = []
    for event in events:
        key = int(event.pitch) - 21
        if not 0 <= key < onset_reg.shape[1]:
            refined.append(event)
            continue
        curve = onset_reg[:, key]
        frame = min(max(1, int(round(event.start * fps))), total_frames - 2)
        lo = max(1, frame - radius_frames)
        hi = min(total_frames - 2, frame + radius_frames)
        best = None
        for f in range(lo, hi + 1):
            if curve[f] >= curve[f - 1] and curve[f] >= curve[f + 1] and curve[f] >= min_value:
                if best is None or curve[f] > curve[best]:
                    best = f
        start = _refine_peak(curve, best) / fps if best is not None else event.start
        refined.append(NoteEvent(event.pitch, start, event.end, event.velocity))
    return refined


def smooth_pedal_events(events, max_gap_s=0.90, release_tail_s=0.12):
    """平滑踏板区间，减少相邻小节交界处的“断—再踩”割裂感。

    规则：
    1. 每个区间的松开时刻向后延长 release_tail_s，给钢琴自然释放留一点尾巴；
    2. 相邻区间如果 gap ≤ max_gap_s，直接合并成一个连续踏板区间，不再先松开再踩下；
    3. 大间隔保留，但前一个区间仍会延长尾巴。

    参数:
        events (list[PedalEvent]): decode_pedal 得到的踏板区间。
        max_gap_s (float): 可合并的最大间隔（秒），默认 0.90。
        release_tail_s (float): 松开尾巴延长量（秒），默认 0.12。

    返回:
        list[PedalEvent]: 平滑后的区间，按开始时间排序。
    """
    ordered = sorted(events, key=lambda e: e.start)
    if not ordered:
        return []
    merged = []
    for event in ordered:
        start = max(0.0, float(event.start))
        end = max(start, float(event.end))
        if not merged:
            merged.append(PedalEvent(start, end + release_tail_s))
            continue
        prev = merged[-1]
        if start <= prev.end + max_gap_s:
            merged[-1] = PedalEvent(prev.start, max(prev.end, end) + release_tail_s)
        else:
            merged[-1] = PedalEvent(prev.start, prev.end + release_tail_s)
            merged.append(PedalEvent(start, end + release_tail_s))
    return merged


def decode_subframe(onset_reg, offset_reg, vel, onset_threshold=0.30,
                    offset_threshold=0.30, min_gap_s=0.05, fallback_dur=0.08):
    """用回归输出 + 亚帧插值解码音符事件。

    对每个琴键分别找 onset_reg / offset_reg 的局部极大值，插值得到亚帧时间；
    再把每个 onset 与它之后的第一个 offset 配成音符。

    参数:
        onset_reg (numpy.ndarray): (T, 88) onset 回归输出（0–1）。
        offset_reg (numpy.ndarray): (T, 88) offset 回归输出（0–1）。
        vel (numpy.ndarray): (T, 88) 力度概率（0–1）。
        onset_threshold (float): onset 峰高阈值。
        offset_threshold (float): offset 峰高阈值。
        min_gap_s (float): 同一个键两次 onset 的最小间隔（秒）。
        fallback_dur (float): 找不到对应 offset 时的兜底时值（秒）。

    返回:
        list[NoteEvent]: 音符事件（带亚帧时间）。
    """
    fps = SR / HOP_LENGTH
    min_distance = max(1, int(round(min_gap_s * fps)))
    events = []
    for key in range(88):
        onset_peaks, _ = find_peaks(onset_reg[:, key], height=onset_threshold,
                                    prominence=0.05, distance=min_distance)
        offset_peaks, _ = find_peaks(offset_reg[:, key], height=offset_threshold,
                                     prominence=0.05, distance=min_distance)
        if len(onset_peaks) == 0:
            continue
        onset_times = [_refine_peak(onset_reg[:, key], int(p)) / fps for p in onset_peaks]
        offset_times = sorted(_refine_peak(offset_reg[:, key], int(p)) / fps
                              for p in offset_peaks)
        used = set()
        for onset_time, onset_peak in zip(onset_times, onset_peaks):
            end_time = None
            for idx, offset_time in enumerate(offset_times):
                if idx not in used and offset_time > onset_time + 0.02:
                    end_time = offset_time
                    used.add(idx)
                    break
            if end_time is None:
                end_time = onset_time + fallback_dur
            velocity = int(np.clip(vel[int(onset_peak), key] * 127, 24, 118))
            events.append(NoteEvent(pitch=key + 21, start=float(onset_time),
                                    end=float(end_time), velocity=velocity))
    return sorted(events, key=lambda e: e.start)


@dataclass(frozen=True)
class PedalEvent:
    """一个延音踏板事件。

    属性:
        start (float): 踩下时刻（秒）。
        end (float): 松开时刻（秒）。
    """

    start: float
    end: float


def decode_pedal(pedal, pedal_on=None, on_threshold=0.5, off_threshold=0.35,
                 release_frames=3, min_gap_s=0.05):
    """把 pedal 头概率解码成踏板的踩下/松开区间。

    pedal 头预测的是「踏板状态」；这里用双阈值迟滞避免边缘抖动：
    pedal >= on_threshold 视为踩下，连续 release_frames 帧 <= off_threshold
    视为松开。pedal_on 头暂不强制参与，只作为可选输入保留。

    参数:
        pedal (numpy.ndarray): (T,) 或 (T,1) 踏板状态概率。
        pedal_on (numpy.ndarray | None): (T,) 或 (T,1) 踩下瞬间概率，暂未使用。
        on_threshold (float): 踩下判定阈值。
        off_threshold (float): 松开判定阈值。
        release_frames (int): 连续多少帧低于 off_threshold 才判定松开。
        min_gap_s (float): 相邻踏板事件之间小于该间隔时合并。

    返回:
        list[PedalEvent]: 按开始时间排序的踏板区间。
    """
    pedal = np.asarray(pedal, dtype=np.float32).reshape(-1)
    frame_s = HOP_LENGTH / SR
    events = []
    state = False
    start = 0
    below = 0
    for i, p in enumerate(pedal):
        if not state:
            if p >= on_threshold:
                state = True
                start = i
                below = 0
        else:
            if p <= off_threshold:
                below += 1
                if below >= release_frames:
                    end = i - release_frames + 1
                    if end > start:
                        events.append(PedalEvent(start * frame_s, end * frame_s))
                    state = False
                    start = 0
                    below = 0
            else:
                below = 0
    if state:
        events.append(PedalEvent(start * frame_s, len(pedal) * frame_s))

    # 合并靠得太近的踏板区间，避免 CC64 抖动
    merged = []
    for ev in events:
        if merged and ev.start - merged[-1].end <= min_gap_s:
            merged[-1] = PedalEvent(merged[-1].start, ev.end)
        else:
            merged.append(ev)
    return merged