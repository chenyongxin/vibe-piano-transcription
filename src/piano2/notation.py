"""M0：记谱 v1 —— 把音符事件变成「人能读」的五线谱。

和旧版 `transcribe_readable.write_readable_musicxml` 的区别（这就是 M0 要验证的核心）：

旧版：
    固定 120 BPM；把「按键秒数」直接乘 2 当时值；每个音单独 insert；
    不做和弦化；用 music21.makeNotation 兜底。

本版：
    1. 从音频估计 BPM 与拍点，用「拍点插值」把秒映射成四分音符位置（容忍速度起伏）；
    2. 起音量化到拍点网格（1/16 拍）；
    3. 同一只手、同一拍点的音合成和弦；
    4. 时值按「到下一个起音的间隔」重估（而不是按键时长），再拆成标准时值树 + 连音线；
    5. 左右手各自成 Part，手动建小节，补齐休止符，保证两行小节数一致；
    6. 速度和拍号来自估计值，调号用音乐理论分析推断。

为什么时值要「按到下一个起音的间隔」：演奏的 note-off 是手指离键，不是乐谱时值。
实测 MAESTRO 演奏的按键时值中位数只有 46–80 ms，直接换算会全部变成 16 分音符。

用法（容器内）：
    python notation_v1.py 输入.wav 输入.midi 输出.musicxml
"""

from __future__ import annotations

import os
import sys

import librosa
import music21
import numpy as np

MEASURE_LEN = 4.0  # 4/4 拍，一小节 4 个四分音符
GRID = 0.25  # 量化网格：1/4 拍 = 16 分音符
STANDARD_QL = (4.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.25)  # 可表示的时值（含附点）
MAX_NOTE_QL = 4.0  # 单音最长 1 小节，超出的用休止符表达


# ─────────────────────────── 速度与拍点 ───────────────────────────


def estimate_tempo_and_beats(audio_path, sr=16000, hop=512):
    """从音频估计速度与拍点位置。

    参数:
        audio_path (str): 音频路径（任意采样率，内部重采样到 sr）。
        sr (int): 分析采样率。
        hop (int): 帧移（样本）。

    返回:
        tuple[float, numpy.ndarray]:
            bpm: 折叠到 [55, 200] 区间后的速度；
            beat_times: 拍点时刻（秒），单调递增。
    """
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    bpm_arr, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=hop
    )
    bpm = float(np.atleast_1d(np.asarray(bpm_arr, dtype=float)).ravel()[0])
    if not np.isfinite(bpm) or bpm <= 0:
        bpm = 120.0
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)
    # 用实测拍点间隔确定速度，而不是用 beat_track 的报告值：
    # 报告值可能把半拍/双拍算成拍，导致「谱面时长 ≠ 音频时长」（之前 -4.4% 的来源）。
    # 拍点位置与速度都用同一套拍点定义，时间就自洽了。
    if len(beat_times) >= 3:
        intervals = np.diff(np.asarray(beat_times, dtype=float))
        median_interval = float(np.median(intervals))
        if median_interval > 1e-3:
            bpm = 60.0 / median_interval
    return bpm, np.asarray(beat_times, dtype=float)


def time_to_quarter(times, beat_times, bpm):
    """把「秒」映射成「四分音符位置」。

    用拍点序列做分段线性插值：第 k 个拍点对应位置 k。这样速度渐快渐慢（rubato）
    会被局部吸收，而不是被一个全局 BPM 抹平。

    参数:
        times (array-like): 需要转换的时刻（秒）。
        beat_times (numpy.ndarray): 拍点时刻。
        bpm (float): 兜底速度（拍点不足时使用）。

    返回:
        numpy.ndarray: 四分音符位置，0 对应第一个拍点。
    """
    times = np.asarray(times, dtype=float)
    if len(beat_times) >= 2:
        bt = np.asarray(beat_times, dtype=float)
        interval = float(np.median(np.diff(bt)))
        if not np.isfinite(interval) or interval <= 1e-3:
            interval = 60.0 / bpm
        # 向前补拍点直到覆盖 t=0，然后把「t=0 时刻」对齐到位置 0：
        # 这样谱面总时长 = 音频总时长，而不是「音频时长 - 第一个拍点」
        n_back = max(int(np.ceil(bt[0] / interval)) + 1, 1)
        lead = bt[0] - interval * np.arange(n_back, 0, -1, dtype=float)
        grid_t = np.concatenate([lead, bt])
        grid_pos = np.arange(len(grid_t), dtype=float)
        raw = np.interp(times, grid_t, grid_pos)
        return raw - float(np.interp(0.0, grid_t, grid_pos))
    return times * bpm / 60.0


# ─────────────────────────── 时值拆解 ───────────────────────────


def decompose_duration(dur_q):
    """把一个时值拆成若干「可表示的标准时值」之和（贪心，从大到小）。

    参数:
        dur_q (float): 以四分音符为单位的时值，应为 0.25 的整数倍。

    返回:
        list[float]: 标准时值列表，例如 1.25 -> [1.0, 0.25]。
    """
    pieces = []
    remaining = round(float(dur_q), 6)
    for value in STANDARD_QL:
        while remaining >= value - 1e-6:
            pieces.append(value)
            remaining = round(remaining - value, 6)
    if remaining > 1e-6:  # 理论上不会发生，属于兜底
        pieces.append(remaining)
    return pieces


def _split_by_measure(q, dur_q, measure_len=MEASURE_LEN):
    """把跨小节的音拆成每小节一段，再在段内拆成标准时值块。

    参数:
        q (float): 起始位置（四分音符）。
        dur_q (float): 时值（四分音符）。
        measure_len (float): 每小节长度（四分音符）。

    返回:
        list[tuple[int, float, float]]: [(小节序号(0 起), 小节内位置, 时值块), ...]
            列表顺序即时间顺序；长度大于 1 时调用方应加连音线。
    """
    segments = []
    cur = float(q)
    end = float(q) + float(dur_q)
    while cur < end - 1e-9:
        measure_idx = int(cur // measure_len)
        measure_start = measure_idx * measure_len
        seg_end = min(end, measure_start + measure_len)
        for piece in decompose_duration(seg_end - cur):
            segments.append((measure_idx, round(cur - measure_start, 6), piece))
            cur = round(cur + piece, 6)
    return segments


# ─────────────────────────── 核心流程 ───────────────────────────


def _clean_events(events):
    """过滤非法音高，并按 (音高, 起音) 去重（同一音高同一时刻保留力度更大的）。

    参数:
        events (list[tuple]): (pitch, start_s, end_s, velocity)。

    返回:
        list[tuple]: 清理后的事件。
    """
    best = {}
    for pitch, start, end, vel in events:
        pitch = int(round(pitch))
        if pitch < 21 or pitch > 108 or end <= start:
            continue
        key = (pitch, round(float(start), 3))
        if key not in best or vel > best[key][3]:
            best[key] = (pitch, float(start), float(end), int(vel))
    return sorted(best.values(), key=lambda e: (e[1], e[0]))


def build_score(events, audio_path, title="M0 v1", hand_split=60):
    """把音符事件构建成 music21 Score。

    参数:
        events (list[tuple]): (pitch, start_s, end_s, velocity)。
        audio_path (str): 用于估计速度与拍点的音频。
        title (str): 乐谱标题。
        hand_split (int): MIDI 音高 >= 该值的音归右手。

    返回:
        tuple[music21.stream.Score, dict]: (乐谱对象, 统计信息)。
    """
    events = _clean_events(events)
    if not events:
        raise ValueError("没有可记谱的音符事件")

    bpm, beat_times = estimate_tempo_and_beats(audio_path)
    starts = np.array([e[1] for e in events], dtype=float)
    ends = np.array([e[2] for e in events], dtype=float)
    start_q = np.round(time_to_quarter(starts, beat_times, bpm) / GRID) * GRID
    end_q = np.round(time_to_quarter(ends, beat_times, bpm) / GRID) * GRID
    start_q = np.maximum(start_q, 0.0)

    # 按手分组，再按量化位置合成和弦
    groups = {"RH": {}, "LH": {}}
    for (pitch, _, _, vel), q in zip(events, start_q):
        hand = "RH" if pitch >= hand_split else "LH"
        groups[hand].setdefault(round(float(q), 6), []).append((pitch, vel))

    # 谱面总长取「最后一个音」与「音频总时长」的较大者：
    # 如果解码漏掉了结尾的音，仅按最后一个音算会让整份谱被截短（M0.5 步骤 1 实测 -18.8%）。
    audio_duration = float(librosa.get_duration(path=audio_path))
    audio_end_q = float(time_to_quarter([audio_duration], beat_times, bpm)[0])
    total_q = float(
        np.ceil(max(end_q.max(), start_q.max() + GRID, audio_end_q) / MEASURE_LEN) * MEASURE_LEN
    )
    n_measures = max(1, int(round(total_q / MEASURE_LEN)))

    key_obj = _estimate_key(events)
    parts = {}
    for hand, clef_obj in (("RH", music21.clef.TrebleClef()), ("LH", music21.clef.BassClef())):
        onsets = sorted(groups[hand].items())
        chords = []
        for idx, (q, pitches) in enumerate(onsets):
            next_q = onsets[idx + 1][0] if idx + 1 < len(onsets) else total_q
            dur = min(max(next_q - q, GRID), MAX_NOTE_QL)
            # 同一位置可能有重复音高（不同力度），只保留一次
            uniq = {}
            for pitch, vel in pitches:
                uniq[pitch] = max(uniq.get(pitch, 0), vel)
            chords.append((q, dur, sorted(uniq.items())))
        parts[hand] = _build_part(
            hand, chords, n_measures, bpm, key_obj, clef_obj, with_tempo=(hand == "RH")
        )

    score = music21.stream.Score(id="M0v1")
    score.insert(0, music21.metadata.Metadata())
    score.metadata.title = title
    score.metadata.composer = "M0 notation v1"
    # 大谱表只需要一个乐器名（写在 StaffGroup 上）；两个 Part 各自的名字会导致
    # 版面上出现三行「Piano」标签，容易让人误以为是三行谱。
    for part in parts.values():
        part.partName = ""
        part.partAbbreviation = ""
    score.insert(0, parts["RH"])
    score.insert(0, parts["LH"])
    score.insert(
        0,
        music21.layout.StaffGroup(
            [parts["RH"], parts["LH"]], name="Piano", abbreviation="Pno.", symbol="brace"
        ),
    )
    stats = {
        "bpm": bpm,
        "n_beats": int(len(beat_times)),
        "n_measures": n_measures,
        "n_events": len(events),
        "n_chords": sum(len(v) for v in groups.values()),
        "total_quarter": total_q,
    }
    return score, stats


def _build_part(hand, chords, n_measures, bpm, key_obj, clef_obj, with_tempo=False):
    """构建一个 Part（一只手），内部只有一条声部。

    参数:
        hand (str): "RH" 或 "LH"。
        chords (list[tuple]): [(起音位置, 时值, [(音高, 力度), ...]), ...]。
        n_measures (int): 总小节数（两手必须一致）。
        bpm (float): 速度，用于写速度记号。
        key_obj (music21.key.Key): 调号。
        clef_obj (music21.clef.Clef): 谱号。
        with_tempo (bool): 是否在该声部写速度记号（只在右手写一次，避免重复）。

    返回:
        music21.stream.Part: 构建好的声部。
    """
    part = music21.stream.Part(id=hand)
    part.insert(0, music21.instrument.Piano())

    # 先按小节收集「片段」：跨小节、跨标准时值的音已经在 _split_by_measure 里切开
    per_measure = {mi: [] for mi in range(n_measures)}
    for q, dur, pitch_vels in chords:
        segments = _split_by_measure(q, dur)
        tie_flags = [(i > 0, i < len(segments) - 1) for i in range(len(segments))]
        for (mi, rel, piece), (tie_stop, tie_start) in zip(segments, tie_flags):
            per_measure[mi].append((rel, piece, pitch_vels, tie_start, tie_stop))

    for mi in range(n_measures):
        measure = music21.stream.Measure(number=mi + 1)
        if mi == 0:
            measure.insert(0, music21.meter.TimeSignature("4/4"))
            measure.insert(0, clef_obj)
            measure.insert(0, key_obj)
            if with_tempo:
                measure.insert(0, music21.tempo.MetronomeMark(number=round(bpm, 2)))
        voice = music21.stream.Voice()
        pos = 0.0
        for rel, piece, pitch_vels, tie_start, tie_stop in sorted(per_measure[mi], key=lambda x: x[0]):
            if rel > pos + 1e-6:
                _insert_rest(voice, pos, rel - pos)
            _insert_chord(voice, rel, piece, pitch_vels, tie_start, tie_stop)
            pos = max(pos, rel + piece)
        if pos < MEASURE_LEN - 1e-6:
            _insert_rest(voice, pos, MEASURE_LEN - pos)
        measure.insert(0, voice)
        try:
            measure.makeBeams(inPlace=True)
        except Exception:  # 某些情况下 music21 会拒绝（例如单声部跨小节），排版可读即可
            pass
        part.append(measure)
    return part


def _insert_rest(voice, offset, dur_q):
    """在 voice 的 offset 处插入一个能覆盖 dur_q 的休止符（必要时拆成多个）。"""
    for piece in decompose_duration(dur_q):
        rest = music21.note.Rest(quarterLength=piece)
        voice.insert(round(offset, 6), rest)
        offset = round(offset + piece, 6)


def _insert_chord(voice, offset, dur_q, pitch_vels, tie_start, tie_stop):
    """插入一个音（或和弦），并按需要设置连音线。"""
    notes = []
    for pitch, vel in pitch_vels:
        n = music21.note.Note()
        n.pitch.midi = int(pitch)
        n.duration = music21.duration.Duration(quarterLength=dur_q)
        n.volume.velocity = int(min(max(vel, 1), 127))
        n.tie = music21.tie.Tie("continue") if (tie_start and tie_stop) else (
            music21.tie.Tie("start") if tie_start else (music21.tie.Tie("stop") if tie_stop else None)
        )
        notes.append(n)
    if len(notes) == 1:
        voice.insert(round(offset, 6), notes[0])
    else:
        chord_obj = music21.chord.Chord(notes)
        voice.insert(round(offset, 6), chord_obj)


def _estimate_key(events):
    """用音高分布做调性分析（Krumhansl 轮廓），失败时退回 C 大调。"""
    try:
        s = music21.stream.Stream()
        for pitch, start, _end, _vel in events[:4000]:
            n = music21.note.Note()
            n.pitch.midi = int(pitch)
            s.insert(float(start), n)
        return s.analyze("key")
    except Exception:
        return music21.key.Key("C")


def events_from_midi(midi_path):
    """从 MIDI 读音符事件（作为 GT 输入）。

    参数:
        midi_path (str): MIDI 路径。

    返回:
        list[tuple]: (pitch, start, end, velocity)。
    """
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(midi_path)
    out = []
    for inst in pm.instruments:
        for note in inst.notes:
            out.append((int(note.pitch), float(note.start), float(note.end), int(note.velocity)))
    return sorted(out, key=lambda e: (e[1], e[0]))


def write_musicxml(events, audio_path, out_path, title="M0 v1 transcription"):
    """一步到位：事件 → MusicXML。

    参数:
        events (list[tuple]): (pitch, start_s, end_s, velocity)。
        audio_path (str): 对应音频（用于速度/拍点估计）。
        out_path (str): 输出 MusicXML 路径。
        title (str): 标题。

    返回:
        dict: 统计信息。
    """
    score, stats = build_score(events, audio_path, title=title)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    score.write("musicxml", fp=out_path)
    return stats


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("用法: python notation_v1.py 输入.wav 输入.midi 输出.musicxml")
        sys.exit(1)
    ev = events_from_midi(sys.argv[2])
    st = write_musicxml(ev, sys.argv[1], sys.argv[3])
    print(st)
    print("已写出:", sys.argv[3])
