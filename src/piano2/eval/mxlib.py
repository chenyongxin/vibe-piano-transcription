# -*- coding: utf-8 -*-
"""钢琴成谱产物只读审计：MusicXML / MIDI / WAV 最小解析库。

只依赖 lxml + numpy（没有 music21 / pretty_midi）。
所有函数都只读文件，不修改用户仓库。

用法：
    from mxlib import parse_musicxml, parse_midi, parse_wav_info, musicxml_metrics
"""

import struct

import numpy as np
from lxml import etree

STEP_SEMITONES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# 标准时值（单位：四分音符 quarterLength）
STANDARD_QL = [4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625]
STANDARD_NAMES = ["whole", "half", "quarter", "eighth", "16th", "32nd", "64th"]


def pitch_to_midi(step, alter, octave):
    """把 MusicXML 的 step/alter/octave 转成 MIDI 音高号。

    参数：
        step (str): 音名，C~B。
        alter (float|None): 变音记号，-1 降号 / 0 还原 / 1 升号。
        octave (int): 八度（MusicXML 中中央 C 为 C4）。
    返回：
        int: MIDI pitch（中央 C = 60）；无法识别时抛 ValueError。
    异常：
        KeyError: step 不在 C~B 中。
    """
    return 12 * (int(octave) + 1) + STEP_SEMITONES[step] + int(round(float(alter or 0)))


# ───────────────────────────── MusicXML ─────────────────────────────


def parse_musicxml(path):
    """用 lxml 解析一个 MusicXML（score-partwise）文件。

    处理 <attributes>/<divisions>、<backup>/<forward>（多声部时间轴）、
    <chord/>（和弦共享起点）、<rest/>、<grace/>、<tie>、
    <time-modification>（连音）、<type>/<dot>。

    参数：
        path (str): MusicXML 文件路径。
    返回：
        dict: {
            'path': str,
            'tempo': float|None,          # 谱面标注 BPM（per-minute 或 sound@tempo）
            'parts': [ { 'id': str, 'events': [ {...} ], 'measures': [...] } ],
            'n_parts': int,
        }
    事件 dict 字段：
        measure (str), onset (float, 四分音符单位), dur (float, 四分音符单位),
        pitch (int|None), midi_octave, is_rest (bool), chord (bool), grace (bool),
        tie_start (bool), tie_stop (bool), type (str|None), dots (int),
        voice (str|None), staff (str|None), print_object (bool), backup_net (float)
    异常：
        etree.XMLSyntaxError: XML 损坏且 recover 失败。
    """
    parser = etree.XMLParser(
        load_dtd=False,
        no_network=True,
        resolve_entities=False,
        recover=True,
        huge_tree=True,
    )
    root = etree.parse(path, parser).getroot()

    tempo = None
    for el in root.iter("per-minute"):
        try:
            tempo = float((el.text or "").strip())
            break
        except ValueError:
            continue
    if tempo is None:
        for el in root.iter("sound"):
            if "tempo" in el.attrib:
                try:
                    tempo = float(el.attrib["tempo"])
                    break
                except ValueError:
                    continue

    parts = []
    for p in root.findall("part"):
        pid = p.get("id", "?")
        divisions = 10080  # 未声明时的兜底（music21 默认值）
        t = 0.0  # 当前小节起点（四分音符单位）
        events = []
        measures = []
        for m in p.findall("measure"):
            mnum = m.get("number", "?")
            implicit = m.get("implicit", "no")
            mstart = t
            pos = 0.0
            last_onset = None
            last_onset_pitch = None
            for el in m:
                tag = el.tag
                if tag == "attributes":
                    d = el.find("divisions")
                    if d is not None and d.text:
                        try:
                            divisions = int(d.text.strip())
                        except ValueError:
                            pass
                elif tag == "backup":
                    try:
                        pos -= int((el.findtext("duration") or "0").strip()) / divisions
                    except ValueError:
                        pass
                elif tag == "forward":
                    try:
                        pos += int((el.findtext("duration") or "0").strip()) / divisions
                    except ValueError:
                        pass
                elif tag == "note":
                    dur_txt = el.findtext("duration")
                    try:
                        dur = int(dur_txt.strip()) / divisions if dur_txt else 0.0
                    except ValueError:
                        dur = 0.0
                    is_chord = el.find("chord") is not None
                    is_grace = el.find("grace") is not None
                    is_rest = el.find("rest") is not None
                    pitch = None
                    octave = None
                    if not is_rest:
                        st = el.find("pitch/step")
                        al = el.find("pitch/alter")
                        oc = el.find("pitch/octave")
                        if st is not None and oc is not None:
                            try:
                                pitch = pitch_to_midi(
                                    (st.text or "").strip(),
                                    (al.text or "0").strip() if al is not None else 0,
                                    (oc.text or "").strip(),
                                )
                                octave = int((oc.text or "").strip())
                            except (KeyError, ValueError):
                                pitch = None
                    pitch_unpitched = el.find("unpitched") is not None
                    ties = set()
                    for tie in el.findall("tie"):
                        ties.add(tie.get("type"))
                    for tied in el.findall("notations/tied"):
                        ties.add(tied.get("type"))
                    dots = len(el.findall("dot"))
                    tm = el.find("time-modification")
                    tmod = None
                    if tm is not None:
                        tmod = (
                            tm.findtext("actual-notes"),
                            tm.findtext("normal-notes"),
                        )
                    if is_chord and last_onset is not None:
                        onset = last_onset
                    else:
                        onset = mstart + pos
                        last_onset = onset
                    if not is_grace and not is_chord:
                        pos += dur
                    events.append(
                        {
                            "measure": mnum,
                            "onset": round(onset, 9),
                            "dur": round(dur, 9),
                            "pitch": pitch,
                            "octave": octave,
                            "is_rest": is_rest or pitch_unpitched,
                            "chord": is_chord,
                            "grace": is_grace,
                            "tie_start": "start" in ties,
                            "tie_stop": "stop" in ties,
                            "type": (el.findtext("type") or "").strip() or None,
                            "dots": dots,
                            "tmod": tmod,
                            "voice": (el.findtext("voice") or "").strip() or None,
                            "staff": (el.findtext("staff") or "").strip() or None,
                            "print_object": el.get("print-object", "yes") != "no",
                        }
                    )
            mend = mstart + pos
            measures.append({"num": mnum, "start": round(mstart, 9), "end": round(mend, 9), "implicit": implicit})
            t = mend
        parts.append({"id": pid, "events": events, "measures": measures, "duration_q": round(t, 9)})

    return {"path": path, "tempo": tempo, "parts": parts, "n_parts": len(parts)}


# ───────────────────────────── MIDI ─────────────────────────────


def _read_varlen(data, i):
    """读 MIDI 变长整数（7 位一组，最高位为延续标志）。返回 (value, next_index)。"""
    val = 0
    while True:
        b = data[i]
        i += 1
        val = (val << 7) | (b & 0x7F)
        if not (b & 0x80):
            return val, i


def parse_midi(path):
    """最小标准 MIDI 文件（SMF）解析器：只抽取 tempo 与 note on/off。

    支持 format 0/1/2、running status、meta 事件（0x51 tempo、0x2F end）。
    忽略 sysex、控制器（含踏板）、程序切换等其余事件。

    参数：
        path (str): .mid/.midi 文件路径。
    返回：
        dict: {
            'path': str, 'format': int, 'ntrks': int, 'division': int,
            'tempi': [(tick, us_per_quarter), ...],
            'notes': [ {'track':int,'channel':int,'pitch':int,'velocity':int,
                        'start_tick':int,'end_tick':int,
                        'start_s':float,'end_s':float,'dur_s':float} ],
            'duration_s': float, 'n_tracks_used': int,
        }
    异常：
        ValueError: 文件头不是 MThd。
    """
    data = open(path, "rb").read()
    if data[:4] != b"MThd":
        raise ValueError(f"not a MIDI file: {path}")
    hdr_len = struct.unpack(">I", data[4:8])[0]
    fmt, ntrks, division = struct.unpack(">HHH", data[8:14])
    tempo_events = []
    raw_notes = []  # (track, channel, pitch, velocity, start_tick, end_tick)
    i = 8 + hdr_len
    for trk in range(ntrks):
        if i + 8 > len(data) or data[i : i + 4] != b"MTrk":
            break
        trk_len = struct.unpack(">I", data[i + 4 : i + 8])[0]
        j = i + 8
        end = min(j + trk_len, len(data))
        tick = 0
        running = 0
        active = {}  # (channel,pitch) -> [start_tick, velocity]
        while j < end:
            tick += 0
            delta, j = _read_varlen(data, j)
            tick += delta
            if j >= end:
                break
            status = data[j]
            if status & 0x80:
                j += 1
                running = status
            else:
                if running == 0:
                    j += 1
                    continue
                status = running
            if status == 0xFF:
                if j >= end:
                    break
                meta_type = data[j]
                j += 1
                ln, j = _read_varlen(data, j)
                payload = data[j : j + ln]
                j += ln
                if meta_type == 0x51 and ln == 3:
                    us = (payload[0] << 16) | (payload[1] << 8) | payload[2]
                    tempo_events.append((tick, us))
                elif meta_type == 0x2F:
                    break
            elif status in (0xF0, 0xF7):
                ln, j = _read_varlen(data, j)
                j += ln
            else:
                hi = status & 0xF0
                ch = status & 0x0F
                if hi in (0x80, 0x90, 0xA0, 0xB0, 0xE0):
                    if j + 1 >= end:
                        break
                    d1, d2 = data[j], data[j + 1]
                    j += 2
                    if hi == 0x90 and d2 > 0:
                        key = (ch, d1)
                        if key in active:
                            # 同音重触发：先把旧音符关闭
                            st, vel = active[key]
                            raw_notes.append((trk, ch, d1, vel, st, tick))
                        active[key] = [tick, d2]
                    elif hi == 0x80 or (hi == 0x90 and d2 == 0):
                        key = (ch, d1)
                        if key in active:
                            st, vel = active.pop(key)
                            raw_notes.append((trk, ch, d1, vel, st, tick))
                elif hi in (0xC0, 0xD0):
                    j += 1
                else:
                    j += 0  # unknown, avoid infinite loop
                    if j < end and not (data[j] & 0x80):
                        j += 1
        for (ch, d1), (st, vel) in active.items():
            raw_notes.append((trk, ch, d1, vel, st, tick))
        i = i + 8 + trk_len

    tempi = sorted(tempo_events) if tempo_events else [(0, 500000)]
    if not tempi or tempi[0][0] != 0:
        tempi = [(0, 500000)] + tempi

    def tick_to_s(tk):
        """绝对 tick → 秒，按 tempo map 分段累加。"""
        secs = 0.0
        prev_tick, prev_us = tempi[0][0], tempi[0][1]
        if tk <= prev_tick:
            return (tk - prev_tick) * prev_us / 1e6 / division
        for cur_tick, cur_us in tempi[1:]:
            if tk <= cur_tick:
                break
            secs += (cur_tick - prev_tick) * prev_us / 1e6 / division
            prev_tick, prev_us = cur_tick, cur_us
        secs += (tk - prev_tick) * prev_us / 1e6 / division
        return secs

    notes = []
    for (trk, ch, pitch, vel, st, en) in raw_notes:
        if en <= st:
            en = st + 1
        notes.append(
            {
                "track": trk,
                "channel": ch,
                "pitch": pitch,
                "velocity": vel,
                "start_tick": st,
                "end_tick": en,
                "start_s": tick_to_s(st),
                "end_s": tick_to_s(en),
            }
        )
    for n in notes:
        n["dur_s"] = n["end_s"] - n["start_s"]
    notes.sort(key=lambda n: (n["start_s"], n["pitch"]))
    duration_s = max((n["end_s"] for n in notes), default=0.0)
    return {
        "path": path,
        "format": fmt,
        "ntrks": ntrks,
        "division": division,
        "tempi": tempi,
        "notes": notes,
        "duration_s": duration_s,
    }


# ───────────────────────────── WAV ─────────────────────────────


def parse_wav_info(path):
    """只读 WAV 头，获取采样率/声道/位深/时长（秒）。

    参数：
        path (str): .wav 路径。
    返回：
        dict: {'sample_rate': int, 'channels': int, 'bits': int, 'duration_s': float}
    异常：
        ValueError: 不是 RIFF/WAVE 文件。
    """
    with open(path, "rb") as f:
        hdr = f.read(12)
        if hdr[:4] != b"RIFF" or hdr[8:12] != b"WAVE":
            raise ValueError(f"not a WAV file: {path}")
        sr = channels = bits = None
        data_size = None
        while True:
            chunk = f.read(8)
            if len(chunk) < 8:
                break
            cid, size = chunk[:4], struct.unpack("<I", chunk[4:])[0]
            if cid == b"fmt ":
                body = f.read(size)
                fmt_tag, channels, sr = struct.unpack("<HHI", body[:8])
                bits = struct.unpack("<H", body[14:16])[0]
            elif cid == b"data":
                data_size = size
                break
            else:
                f.seek(size + (size & 1), 1)
    if sr is None or data_size is None:
        raise ValueError(f"WAV missing fmt/data chunk: {path}")
    duration = data_size / (sr * channels * max(bits, 8) / 8.0)
    return {"sample_rate": sr, "channels": channels, "bits": bits, "duration_s": round(duration, 3)}


# ───────────────────────────── 指标计算 ─────────────────────────────


def classify_ql(q):
    """把四分音符时值归类到标准音符名（whole/half/quarter/eighth/16th/32nd/64th/other）。

    参数：
        q (float): 时值（四分音符单位）。
    返回：
        str: 类别名；非标准时值返回 'other'。
    """
    for ref, name in zip(STANDARD_QL, STANDARD_NAMES):
        if abs(q - ref) <= max(1e-6, 0.01 * ref):
            return name
    return "other"


def ql_bucket(q):
    """时值粗分桶（用于黑音符占比），桶按「不短于」下界划分。

    返回：
        str: '<64th' | '64th' | '32nd' | '16th' | 'eighth' | 'quarter' | 'half' | '>=whole'
    """
    if q < 0.0625:
        return "<64th"
    if q < 0.125:
        return "64th"
    if q < 0.25:
        return "32nd"
    if q < 0.5:
        return "16th"
    if q < 1.0:
        return "eighth"
    if q < 2.0:
        return "quarter"
    if q < 4.0:
        return "half"
    return ">=whole"


def _pct(x, n):
    """安全百分比（n=0 时返回 0.0）。"""
    return 100.0 * x / n if n else 0.0


def _stats(arr):
    """返回 (n, mean, median, p05, p95, max) 的 numpy 摘要。"""
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "mean": round(float(a.mean()), 4),
        "median": round(float(np.median(a)), 4),
        "p05": round(float(np.percentile(a, 5)), 4),
        "p95": round(float(np.percentile(a, 95)), 4),
        "max": round(float(a.max()), 4),
    }


def _sounding_stats(probe_points, events):
    """给定采样时刻与音符区间，计算每个采样时刻的同时发声数（向量化扫描线）。

    参数：
        probe_points (np.ndarray): 采样时刻（与 events 相同的时值单位，升序）。
        events (list[dict]): 含 'onset'/'dur' 的音符事件。
    返回：
        dict: {'n','mean','median','p95','max','pct_gt10','pct_gt6'}；无音符时返回 {}。
    """
    if not events or len(probe_points) == 0:
        return {}
    import heapq

    ivs = sorted((e["onset"], e["onset"] + e["dur"]) for e in events)
    heap = []
    ptr = 0
    counts = np.empty(len(probe_points), dtype=int)
    for k, t in enumerate(probe_points):
        while ptr < len(ivs) and ivs[ptr][0] <= t + 1e-9:
            heapq.heappush(heap, ivs[ptr][1])
            ptr += 1
        while heap and heap[0] <= t + 1e-9:
            heapq.heappop(heap)
        counts[k] = len(heap)
    return {
        "n": int(len(counts)),
        "mean": round(float(counts.mean()), 2),
        "median": int(np.median(counts)),
        "p95": round(float(np.percentile(counts, 95)), 2),
        "max": int(counts.max()),
        "pct_gt10": round(_pct(int((counts > 10).sum()), len(counts)), 2),
        "pct_gt6": round(_pct(int((counts > 6).sum()), len(counts)), 2),
    }


def _tie_chains(parts):
    """按 part 合并连音线（tie）链，得到每个「真实音符」的总时值。

    参数：
        parts (list[dict]): parse_musicxml 的 parts。
    返回：
        list[dict]: [{'part':int,'pitch':int,'onset':float,'chain_dur':float,
                      'n_segments':int,'segment_durs':[...]}]（仅非休止音符）
    """
    out = []
    for pi, part in enumerate(parts):
        ev = [e for e in part["events"] if not e["is_rest"] and e["pitch"] is not None]
        index = {}
        for e in ev:
            index.setdefault((round(e["onset"], 6), e["pitch"]), []).append(e)
        for e in ev:
            if e["tie_stop"] or e["chord"] or e["grace"]:
                continue
            total = e["dur"]
            segs = [e["dur"]]
            cur = e
            seen = 0
            while cur["tie_start"] and seen < 200:
                key = (round(cur["onset"] + cur["dur"], 6), cur["pitch"])
                nxt = [x for x in index.get(key, []) if x["tie_stop"]]
                if not nxt:
                    break
                nx = nxt[0]
                total += nx["dur"]
                segs.append(nx["dur"])
                cur = nx
                seen += 1
            out.append(
                {
                    "part": pi,
                    "pitch": e["pitch"],
                    "onset": e["onset"],
                    "chain_dur": round(total, 9),
                    "n_segments": len(segs),
                    "segment_durs": segs,
                    "first_seg_dur": e["dur"],
                }
            )
    return out


def musicxml_metrics(mx):
    """对一个已解析的 MusicXML 结构计算全部审计指标。

    参数：
        mx (dict): parse_musicxml 的返回值。
    返回：
        dict: 指标字典（JSON 可序列化），字段见函数内注释。
    """
    tempo = mx["tempo"] or 120.0
    tempo_assumed = mx["tempo"] is None
    spq = 60.0 / tempo  # 每个四分音符多少秒
    parts = mx["parts"]

    all_noteheads = []  # 所有非休止 <note>（含 chord 成员与 tie 续接）
    all_attacks = []  # 真正的击键（非 chord 成员、非 tie-stop、非 grace）
    all_rests = []
    for pi, part in enumerate(parts):
        for e in part["events"]:
            e2 = dict(e)
            e2["part"] = pi
            if e2["is_rest"]:
                all_rests.append(e2)
            else:
                all_noteheads.append(e2)
                if not e2["chord"] and not e2["grace"] and not e2["tie_stop"]:
                    all_attacks.append(e2)

    # 总时长：各 part 最后一小节末端取最大
    dur_q = max([p["duration_q"] for p in parts], default=0.0)
    dur_s = dur_q * spq
    measures_per_part = [len(p["measures"]) for p in parts]

    # ── 时长分布（对 notehead 与 attack 分别统计）
    def dur_dist(events):
        if not events:
            return {}
        qs = np.array([e["dur"] for e in events], dtype=float)
        buckets = {}
        for b in ["<64th", "64th", "32nd", "16th", "eighth", "quarter", "half", ">=whole"]:
            buckets[b] = int(np.sum([ql_bucket(q) == b for q in qs]))
        types = {}
        for e in events:
            t = e["type"] or "none"
            if e["dots"]:
                t += "+dot"
            if e["tmod"]:
                t += "+tuplet"
            types[t] = types.get(t, 0) + 1
        return {
            "n": len(events),
            "ql": _stats(qs),
            "ql_median": float(np.median(qs)),
            "sec": _stats(qs * spq),
            "buckets": buckets,
            "flagged_ratio_pct": round(_pct(
                int(np.sum(qs < 1.0)), len(qs)), 2),  # eighth 及更短（带符尾）
            "filled_ratio_pct": round(_pct(int(np.sum(qs < 2.0)), len(qs)), 2),  # 实心符头
            "quarter_or_shorter_pct": round(_pct(int(np.sum(qs <= 1.0 + 1e-9)), len(qs)), 2),
            "exact_16th_pct": round(_pct(int(np.sum(np.abs(qs - 0.25) <= 1e-6)), len(qs)), 2),
            "exact_eighth_pct": round(_pct(int(np.sum(np.abs(qs - 0.5) <= 1e-6)), len(qs)), 2),
            "off_standard_pct": round(_pct(sum(classify_ql(q) == "other" for q in qs), len(qs)), 2),
            "types": dict(sorted(types.items(), key=lambda kv: -kv[1])),
        }

    dist_noteheads = dur_dist(all_noteheads)
    dist_attacks = dur_dist(all_attacks)

    # ── 连音线（tie）链：把跨小节的碎片合并成「真实时值」，拆解黑音符来源
    chains = _tie_chains(parts)
    chain_metrics = {}
    if chains:
        cd = np.array([c["chain_dur"] for c in chains], dtype=float)
        segs_per_chain = np.array([c["n_segments"] for c in chains], dtype=float)
        # 黑音符（printed notehead）来源拆解
        black_noteheads = [e for e in all_noteheads if e["dur"] < 1.0]
        black_tie_cont = [e for e in black_noteheads if e["tie_stop"]]
        black_frag_16 = [e for e in all_noteheads if abs(e["dur"] - 0.25) <= 1e-6]
        black_frag_16_tie = [e for e in black_frag_16 if e["tie_stop"]]
        chain_metrics = {
            "n_chains": len(chains),
            "chain_dur_q": _stats(cd),
            "chain_dur_s": _stats(cd * spq),
            "chain_flagged_ratio_pct": round(_pct(int((cd < 1.0).sum()), len(cd)), 2),
            "chain_16th_ratio_pct": round(_pct(int((np.abs(cd - 0.25) <= 1e-6).sum()), len(cd)), 2),
            "segments_per_chain": _stats(segs_per_chain),
            "pct_chains_split": round(_pct(int((segs_per_chain > 1).sum()), len(cd)), 2),
            "notehead_inflation_x": round(len(all_noteheads) / max(len(chains), 1), 3),
            "black_noteheads": len(black_noteheads),
            "black_noteheads_pct_of_all": round(_pct(len(black_noteheads), len(all_noteheads)), 2),
            "black_noteheads_tie_continuation": len(black_tie_cont),
            "black_noteheads_tie_cont_pct": round(_pct(len(black_tie_cont), len(black_noteheads)), 2) if black_noteheads else 0.0,
            "exact_16th_noteheads": len(black_frag_16),
            "exact_16th_noteheads_tie_cont": len(black_frag_16_tie),
            "exact_16th_pct_from_tie": round(_pct(len(black_frag_16_tie), len(black_frag_16)), 2) if black_frag_16 else 0.0,
        }

    # ── 密度
    density = {
        "duration_s": round(dur_s, 2),
        "duration_q": round(dur_q, 2),
        "measures_per_part": measures_per_part,
        "n_noteheads": len(all_noteheads),
        "n_attacks": len(all_attacks),
        "n_rests": len(all_rests),
        "attack_notes_per_s": round(len(all_attacks) / dur_s, 2) if dur_s > 0 else None,
        "notehead_per_s": round(len(all_noteheads) / dur_s, 2) if dur_s > 0 else None,
        "tempo": tempo,
        "tempo_assumed": tempo_assumed,
    }

    # ── 休止符（按「全部 part、全部事件的总时值」计算占比，避免多声部重复计数偏差）
    total_event_dur_q = float(sum(e["dur"] for p_ in parts for e in p_["events"]))
    rest_dur_q_total = float(sum(e["dur"] for e in all_rests))
    rests = {
        "n_rests": len(all_rests),
        "rest_duration_q_total": round(rest_dur_q_total, 2),
        "total_event_duration_q": round(total_event_dur_q, 2),
        "rest_ratio_pct_of_all_events": round(_pct(rest_dur_q_total, total_event_dur_q), 2) if total_event_dur_q > 0 else None,
        "rests_per_measure_per_part": round(len(all_rests) / max(sum(measures_per_part), 1), 2),
        "rest_types": {},
    }
    for e in all_rests:
        t = (e["type"] or "none")
        if e["dots"]:
            t += "+dot"
        rests["rest_types"][t] = rests["rest_types"].get(t, 0) + 1

    # ── tie / 跨小节
    tie_starts = [e for e in all_noteheads if e["tie_start"]]
    tie_stops = [e for e in all_noteheads if e["tie_stop"]]
    # 跨小节：tie_start 的终点处存在同 part 同 pitch 的 tie_stop，且小节号不同
    cross_measure = 0
    for pi, part in enumerate(parts):
        stop_index = {}
        for f in part["events"]:
            if not f["is_rest"] and f["tie_stop"] and f["pitch"] is not None:
                stop_index.setdefault((round(f["onset"], 6), f["pitch"]), []).append(f["measure"])
        for e in part["events"]:
            if e["is_rest"] or not e["tie_start"] or e["pitch"] is None:
                continue
            key = (round(e["onset"] + e["dur"], 6), e["pitch"])
            for mm in stop_index.get(key, []):
                if mm != e["measure"]:
                    cross_measure += 1
                    break
    ties = {
        "n_tie_start": len(tie_starts),
        "n_tie_stop": len(tie_stops),
        "cross_measure_ties": cross_measure,
        "tie_start_per_1000_noteheads": round(1000.0 * len(tie_starts) / len(all_noteheads), 2) if all_noteheads else None,
    }

    # ── polyphony（同一音列）
    onsets = np.array([e["onset"] for e in all_attacks], dtype=float)
    uniq, counts = np.unique(np.round(onsets, 6), return_counts=True)
    onset_groups = counts.astype(int)
    poly_onset = {
        "n_unique_onsets": int(len(uniq)),
        "max_notes_per_onset": int(onset_groups.max()) if len(onset_groups) else 0,
        "mean_notes_per_onset": round(float(onset_groups.mean()), 2) if len(onset_groups) else 0,
        "p95_notes_per_onset": round(float(np.percentile(onset_groups, 95)), 2) if len(onset_groups) else 0,
        "pct_onsets_gt10": round(_pct(int((onset_groups > 10).sum()), len(onset_groups)), 2),
        "pct_onsets_gt6": round(_pct(int((onset_groups > 6).sum()), len(onset_groups)), 2),
        "hist": {str(k): int(v) for k, v in zip(*np.unique(onset_groups, return_counts=True))},
    }

    # 实际同时发声数：在全局每个 onset 时刻采样，统计 [start, end) 覆盖该时刻的 notehead 数
    sounding_global = _sounding_stats(uniq, all_noteheads)
    sounding_per_part = {
        parts[i]["id"]: _sounding_stats(uniq, [e for e in all_noteheads if e["part"] == i]) for i in range(len(parts))
    }

    # ── 同音高重复触发（碎片化指标）
    def repeat_metrics(events, tag):
        if not events:
            return {}
        by_pitch = {}
        for e in events:
            by_pitch.setdefault(e["pitch"], []).append(e["onset"])
        gaps_all = []
        for p, on in by_pitch.items():
            on = sorted(on)
            gaps_all.extend(np.diff(on).tolist())
        gaps = np.array(gaps_all, dtype=float) if gaps_all else np.array([])
        n = len(events)
        return {
            "n_attacks": n,
            "n_distinct_pitches": len(by_pitch),
            "same_pitch_repeat_50ms": int((gaps <= 0.1).sum()) if gaps.size else 0,  # 0.1q @120bpm=50ms
            "same_pitch_repeat_100ms": int((gaps <= 0.2).sum()) if gaps.size else 0,
            "same_pitch_repeat_50ms_pct": round(_pct(int((gaps <= 0.1).sum()) if gaps.size else 0, n), 2),
            "same_pitch_repeat_100ms_pct": round(_pct(int((gaps <= 0.2).sum()) if gaps.size else 0, n), 2),
            "inter_onset_gap_q": _stats(gaps) if gaps.size else {"n": 0},
            "inter_onset_gap_s": _stats(gaps * spq) if gaps.size else {"n": 0},
        }

    repeat_global = repeat_metrics(all_attacks, "global")

    # ── 相邻半音同时出现（chromatic cluster）
    onset_to_pitches = {}
    for e in all_noteheads:
        onset_to_pitches.setdefault(round(e["onset"], 6), set()).add(e["pitch"])
    n_group = len(onset_to_pitches)
    n_multi = sum(1 for s in onset_to_pitches.values() if len(s) >= 2)
    n_chromatic = 0
    for s in onset_to_pitches.values():
        ps = sorted(s)
        if any(b - a == 1 for a, b in zip(ps, ps[1:])):
            n_chromatic += 1
    chroma = {
        "n_onset_groups": n_group,
        "n_groups_ge2": n_multi,
        "n_groups_with_semitone_pair": n_chromatic,
        "pct_of_all_groups": round(_pct(n_chromatic, n_group), 2),
        "pct_of_multi_groups": round(_pct(n_chromatic, n_multi), 2),
    }

    # ── 网格一致性（量化证据）
    all_onsets = np.array([e["onset"] for e in all_attacks], dtype=float)
    all_durs = np.array([e["dur"] for e in all_attacks], dtype=float)
    grid_dev_onset = np.abs(all_onsets / 0.25 - np.round(all_onsets / 0.25))
    grid_dev_dur = np.abs(all_durs / 0.25 - np.round(all_durs / 0.25))
    grid = {
        "onset_on_16th_grid_pct": round(_pct(int((grid_dev_onset < 1e-6).sum()), len(all_onsets)), 2) if len(all_onsets) else None,
        "dur_on_16th_grid_pct": round(_pct(int((grid_dev_dur < 1e-6).sum()), len(all_durs)), 2) if len(all_durs) else None,
        "min_dur_q": round(float(all_durs.min()), 6) if len(all_durs) else None,
        "max_dur_q": round(float(all_durs.max()), 6) if len(all_durs) else None,
    }

    # ── 左右手/分轨
    per_part = []
    for i, part in enumerate(parts):
        pe = [e for e in all_noteheads if e["part"] == i]
        pa = [e for e in all_attacks if e["part"] == i]
        pitches = np.array([e["pitch"] for e in pe if e["pitch"] is not None], dtype=int)
        pp_meas = len(part["measures"])
        per_part.append(
            {
                "id": part["id"],
                "n_noteheads": len(pe),
                "n_attacks": len(pa),
                "n_measures": pp_meas,
                "duration_q": part["duration_q"],
                "pitch_min": int(pitches.min()) if pitches.size else None,
                "pitch_max": int(pitches.max()) if pitches.size else None,
                "pitch_median": int(np.median(pitches)) if pitches.size else None,
                "n_distinct_pitches": int(len(np.unique(pitches))),
                "flagged_ratio_pct": round(_pct(sum(1 for e in pe if e["dur"] < 1.0), len(pe)), 2) if pe else None,
            }
        )
    hands = {"parts": per_part}
    if len(parts) >= 2:
        # 同音高同时出现在两个 part（同 onset 或时间重叠）
        p0 = [e for e in all_noteheads if e["part"] == 0]
        p1 = [e for e in all_noteheads if e["part"] == 1]
        overlap_pairs = 0
        onset_pairs = 0
        p1_by_onset = {}
        for e in p1:
            p1_by_onset.setdefault(round(e["onset"], 6), set()).add(e["pitch"])
        for e in p0:
            same = p1_by_onset.get(round(e["onset"], 6))
            if same and e["pitch"] in same:
                onset_pairs += 1
        hands["same_pitch_same_onset_across_parts"] = onset_pairs
        hands["same_pitch_same_onset_per_1000_p0_notes"] = round(1000.0 * onset_pairs / len(p0), 2) if p0 else None

    # ── 击键密度时间序列（1 秒窗口）
    if dur_s > 0 and len(all_attacks) > 1:
        at = np.array([e["onset"] * spq for e in all_attacks], dtype=float)
        order = np.argsort(at)
        at = at[order]
        win = 1.0
        idx = np.searchsorted(at, at + win, side="left")
        counts_win = (idx - np.arange(len(at))).astype(int)
        win_stats = {
            "max_notes_in_1s": int(counts_win.max()),
            "p95_notes_in_1s": round(float(np.percentile(counts_win, 95)), 2),
            "mean_notes_in_1s": round(float(counts_win.mean()), 2),
        }
    else:
        win_stats = {}

    return {
        "path": mx["path"],
        "tempo": tempo,
        "tempo_assumed": tempo_assumed,
        "n_parts": len(parts),
        "density": density,
        "durations_noteheads": dist_noteheads,
        "durations_attacks": dist_attacks,
        "tie_chains": chain_metrics,
        "rests": rests,
        "ties": ties,
        "polyphony_onset_groups": poly_onset,
        "sounding_polyphony_global": sounding_global,
        "sounding_polyphony_per_part": sounding_per_part,
        "same_pitch_repeats": repeat_global,
        "chromatic": chroma,
        "grid": grid,
        "hands": hands,
        "window_density": win_stats,
    }


def midi_metrics(mi, tempo_assumed=120.0):
    """对 parse_midi 结果计算与 MusicXML 对齐的指标（用于 GT / 预测 MIDI 对比）。

    参数：
        mi (dict): parse_midi 返回值。
        tempo_assumed (float): 仅用于展示，不影响秒级指标。
    返回：
        dict: 指标字典。
    """
    notes = mi["notes"]
    if not notes:
        return {"path": mi["path"], "n_notes": 0, "duration_s": 0.0, "notes_per_s": None}
    starts = np.array([n["start_s"] for n in notes], dtype=float)
    durs = np.array([n["dur_s"] for n in notes], dtype=float)
    pitches = np.array([n["pitch"] for n in notes], dtype=int)
    dur_s = max(mi["duration_s"], 1e-9)

    uniq, cnt = np.unique(np.round(starts, 4), return_counts=True)
    # 同音高重复触发
    gaps = []
    for p in np.unique(pitches):
        s = np.sort(starts[pitches == p])
        gaps.extend(np.diff(s).tolist())
    gaps = np.array(gaps, dtype=float) if gaps else np.array([])
    # 半音同时出现
    by_onset = {}
    for s, p in zip(np.round(starts, 4), pitches):
        by_onset.setdefault(s, set()).add(int(p))
    n_chrom = sum(1 for s in by_onset.values() if any(b - a == 1 for a, b in zip(sorted(s), sorted(s)[1:])))
    return {
        "path": mi["path"],
        "tempo_assumed": tempo_assumed,
        "division": mi["division"],
        "format": mi["format"],
        "n_notes": len(notes),
        "duration_s": round(dur_s, 2),
        "notes_per_s": round(len(notes) / dur_s, 2),
        "pitch_min": int(pitches.min()),
        "pitch_max": int(pitches.max()),
        "dur_s_stats": _stats(durs),
        "pct_dur_lt_100ms": round(_pct(int((durs < 0.1).sum()), len(durs)), 2),
        "pct_dur_lt_250ms": round(_pct(int((durs < 0.25).sum()), len(durs)), 2),
        "median_dur_s": round(float(np.median(durs)), 4),
        "max_notes_per_onset": int(cnt.max()),
        "mean_notes_per_onset": round(float(cnt.mean()), 2),
        "pct_onsets_gt10": round(_pct(int((cnt > 10).sum()), len(cnt)), 2),
        "same_pitch_repeat_50ms": int((gaps <= 0.05).sum()) if gaps.size else 0,
        "same_pitch_repeat_100ms": int((gaps <= 0.10).sum()) if gaps.size else 0,
        "same_pitch_repeat_50ms_pct": round(_pct(int((gaps <= 0.05).sum()) if gaps.size else 0, len(notes)), 2),
        "same_pitch_repeat_100ms_pct": round(_pct(int((gaps <= 0.10).sum()) if gaps.size else 0, len(notes)), 2),
        "pct_groups_with_semitone_pair": round(_pct(n_chrom, len(by_onset)), 2),
        "n_tempi": len(mi["tempi"]),
        "tempo_bpm_range": [
            round(60e6 / mi["tempi"][0][1], 1),
            round(60e6 / min(t[1] for t in mi["tempi"]), 1),
            round(60e6 / max(t[1] for t in mi["tempi"]), 1),
        ],
    }
