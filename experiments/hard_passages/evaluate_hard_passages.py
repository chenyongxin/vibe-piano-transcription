"""在固定难片段上评测当前 hybrid 模型，用于失败归因。

读取 manifest.json，对每个片段：
1. 用 sliding-window 推理 + offset/onset_reg 解码；
2. 与 GT MIDI 比较 note / onset F1，统计漏检、多检、短音召回；
3. 把解码事件写成临时 MusicXML，比较 raw 事件和谱面音符数；
4. 估算量化到 1/4 拍网格的误差，作为音画/记谱对齐的代理指标。

用法（容器内）：
    python experiments/hard_passages/evaluate_hard_passages.py \
        --model models/best.pt --out experiments/hard_passages/results.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
import soundfile as sf
import mir_eval.transcription as mtran
from lxml import etree
from scipy.signal import find_peaks

from piano2.constants import (
    BINS_PER_OCTAVE, DECODE_DEFAULTS, FMIN, HOP, MAESTRO_ROOT, N_BINS, SR,
)
from piano2.core import NoteEvent
from piano2.decode import (
    DecodeConfig, _refine_peak, apply_offset_ends, decode, refine_onsets,
)
from piano2.inference import load_model
from piano2.notation import (
    GRID, _clean_events, estimate_tempo_and_beats, time_to_quarter, write_musicxml,
)
from piano2.notation import events_from_midi

OUT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = OUT / "manifest.json"


def load_clip_audio(audio_path, start, duration):
    """读取片段音频并返回 (y, cqt)。"""
    y, _ = librosa.load(audio_path, sr=SR, offset=start, duration=duration, mono=True)
    cqt_complex = librosa.cqt(
        y, sr=SR, hop_length=HOP, fmin=FMIN,
        n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE,
    )
    cqt = librosa.amplitude_to_db(np.abs(cqt_complex), ref=np.max).T.astype(np.float32)
    return y, cqt


def infer_segment(model, cqt, device, offset_mode="first"):
    """对片段推理并按指定 offset 策略解码。

    参数:
        model (torch.nn.Module): hybrid 模型。
        cqt (np.ndarray): (T, 264) 输入特征。
        device (str): 推理设备。
        offset_mode (str): first / frame / near_frame / strongest / near_reg。

    返回:
        list[NoteEvent]: 解码后的音符事件。
    """
    from piano2.inference import infer_sliding_window

    probs = infer_sliding_window(
        model, cqt, device, with_offset=True, with_regression=True,
        with_pedal=True,
    )
    onset, frame, vel, offset, onset_reg, offset_reg, pedal, pedal_on = probs
    base_cfg = DecodeConfig(**DECODE_DEFAULTS, use_offset_end=False)
    base = decode(onset, frame, vel, base_cfg)
    base = refine_onsets(base, onset_reg)
    cfg = DecodeConfig(**DECODE_DEFAULTS, use_offset_end=True)
    if offset_mode == "frame":
        return base
    if offset_mode == "first":
        return apply_offset_ends(base, offset, cfg)
    return apply_offset_variant(base, offset, cfg, offset_mode, offset_reg)


def apply_offset_variant(events, offset_probs, cfg, mode, offset_reg=None):
    """用不同 offset 峰值选择策略修正结束时间。

    参数:
        events (list[NoteEvent]): 分类解码出的音符。
        offset_probs (np.ndarray): (T, 88) offset 概率。
        cfg (DecodeConfig): 解码配置。
        mode (str): near_frame / strongest / near_reg。
        offset_reg (np.ndarray | None): (T, 88) offset_reg 输出；near_reg 使用。

    返回:
        list[NoteEvent]: 修正后的音符。
    """
    fps = SR / HOP
    total = offset_probs.shape[0]
    fixed = []
    for event in events:
        key = event.pitch - 21
        start = int(round(event.start * fps))
        lo = start + cfg.offset_min_frames
        hi = min(total, start + cfg.offset_max_frames)
        end = event.end
        if 0 <= key < 88 and hi - lo >= 2:
            peaks, props = find_peaks(
                offset_probs[lo:hi, key], height=cfg.offset_threshold
            )
            if len(peaks):
                if mode == "strongest":
                    peak = int(peaks[np.argmax(props["peak_heights"])])
                else:
                    candidates = (peaks + lo) / fps
                    peak = int(peaks[np.argmin(np.abs(candidates - end))])
                end = (lo + peak) / fps
                end = max(end, event.start + cfg.offset_min_frames / fps)
        if offset_reg is not None and mode == "near_reg":
            frame = int(round(end * fps))
            frame = min(max(1, frame), total - 2)
            curve = offset_reg[:, key]
            best = None
            for f in range(max(1, frame - 1), min(total - 2, frame + 1) + 1):
                if curve[f] >= curve[f - 1] and curve[f] >= curve[f + 1]:
                    if best is None or curve[f] > curve[best]:
                        best = f
            if best is not None:
                end = _refine_peak(curve, best) / fps
        fixed.append(NoteEvent(event.pitch, event.start,
                               float(max(end, event.start + 0.02)),
                               event.velocity))
    return fixed


def gt_events(midi_path, start, duration):
    """读取片段 GT 音符，时间平移到片段起点。"""
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    out = []
    for inst in pm.instruments:
        for n in inst.notes:
            if start <= n.start < start + duration:
                s = n.start - start
                e = min(n.end - start, duration)
                if e > s:
                    out.append((int(n.pitch), s, float(e), int(n.velocity or 80)))
    return sorted(out, key=lambda x: (x[1], x[0]))


def match_counts(ref_iv, ref_p, est_iv, est_p, tol_s, offset_ratio=None):
    """返回 (tp, n_ref, n_est)。"""
    if len(ref_iv) == 0 or len(est_iv) == 0:
        return 0, len(ref_iv), len(est_iv)
    kwargs = dict(onset_tolerance=tol_s, pitch_tolerance=50.0,
                  offset_ratio=offset_ratio)
    if offset_ratio is not None:
        kwargs["offset_min_tolerance"] = 0.05
    matches = mtran.match_notes(ref_iv, ref_p, est_iv, est_p, **kwargs)
    return len(matches), len(ref_iv), len(est_iv)


def prf(tp, n_ref, n_est):
    """算 P/R/F1。"""
    p = tp / n_est if n_est else 0.0
    r = tp / n_ref if n_ref else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0}


def matched_errors(ref_iv, ref_p, est_iv, est_p, tol_s=0.05):
    """返回匹配音符的起点/结束误差（毫秒）。"""
    if len(ref_iv) == 0 or len(est_iv) == 0:
        return [], []
    matches = mtran.match_notes(ref_iv, ref_p, est_iv, est_p,
                                onset_tolerance=tol_s, pitch_tolerance=50.0,
                                offset_ratio=None)
    onset_ms, offset_ms = [], []
    for i, j in matches:
        onset_ms.append((est_iv[j, 0] - ref_iv[i, 0]) * 1000.0)
        offset_ms.append((est_iv[j, 1] - ref_iv[i, 1]) * 1000.0)
    return onset_ms, offset_ms


def score_note_count(events, y, sr, tmpdir):
    """将事件写成 MusicXML，数出 pitched note 元素数量。"""
    if not events:
        return 0
    wav = os.path.join(tmpdir, "clip.wav")
    sf.write(wav, y, sr, subtype="PCM_16")
    xml = os.path.join(tmpdir, "clip.musicxml")
    ev = [(e.pitch, e.start, e.end, e.velocity) for e in events]
    try:
        write_musicxml(ev, wav, xml, title="hard-passage")
        root = etree.parse(xml).getroot()
        notes = root.findall(".//{*}note")
        return sum(1 for n in notes if n.find("{*}rest") is None)
    except Exception:
        return -1


def quant_error_ms(events, y, sr):
    """估算事件映射到 1/4 拍网格的残差（毫秒），作为谱面时间偏差代理。"""
    if not events:
        return None
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        sf.write(tmp.name, y, sr, subtype="PCM_16")
        bpm, beat_times = estimate_tempo_and_beats(tmp.name)
        starts = np.array([e.start for e in events], dtype=float)
        q = time_to_quarter(starts, beat_times, bpm)
        err = np.abs(q - np.round(q / GRID) * GRID) * (60.0 / max(bpm, 1e-6)) * 1000.0
        return float(np.median(err)), float(np.percentile(err, 90))
    except Exception:
        return None
    finally:
        os.unlink(tmp.name)


def main():
    ap = argparse.ArgumentParser(description="难片段失败归因评测")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--model", default="models/best.pt")
    ap.add_argument("--out", default=str(OUT / "results.json"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--offset-mode", default="first",
                    choices=["first", "frame", "near_frame", "strongest", "near_reg"])
    args = ap.parse_args()

    with open(args.manifest, encoding="utf-8") as fh:
        manifest = json.load(fh)
    model = load_model(args.model, args.device)
    results = []

    for clip in manifest["clips"]:
        duration = float(clip.get("duration", manifest["duration_s"]))
        start = float(clip["start"])
        audio_path = str(MAESTRO_ROOT / clip["audio"])
        midi_path = str(MAESTRO_ROOT / clip["track"])
        gt = gt_events(midi_path, start, duration)
        y, cqt = load_clip_audio(audio_path, start, duration)
        events = infer_segment(model, cqt, args.device, args.offset_mode)
        clip_events = [e for e in events if 0 <= e.start <= duration]

        ref_iv = np.array([[s, e] for _p, s, e, _v in gt], dtype=float)
        ref_p = np.array([p for p, _s, _e, _v in gt], dtype=int)
        est_iv = np.array([[e.start, e.end] for e in clip_events], dtype=float)
        est_p = np.array([e.pitch for e in clip_events], dtype=int)

        counts = {}
        for name, tol, ratio in (
            ("onset50", 0.05, None), ("onset20", 0.02, None),
            ("onset10", 0.01, None),
            ("note", 0.05, 0.2),
        ):
            tp, nr, ne = match_counts(ref_iv, ref_p, est_iv, est_p, tol, ratio)
            counts[name] = {"tp": tp, "n_ref": nr, "n_est": ne, **prf(tp, nr, ne)}

        onset_err, offset_err = matched_errors(ref_iv, ref_p, est_iv, est_p)
        short_gt = [i for i, (_p, s, e, _v) in enumerate(gt) if (e - s) < 0.08]
        short_hit = 0
        if short_gt:
            tp50 = mtran.match_notes(ref_iv, ref_p, est_iv, est_p,
                                     onset_tolerance=0.05, pitch_tolerance=50.0,
                                     offset_ratio=None)
            matched_ref = {i for i, _j in tp50}
            short_hit = sum(1 for i in short_gt if i in matched_ref)

        # 记谱层：去重后的数量，以及写谱后的 pitched note 数。
        cleaned = _clean_events([(e.pitch, e.start, e.end, e.velocity) for e in clip_events])
        with tempfile.TemporaryDirectory() as td:
            score_notes = score_note_count(clip_events, y, SR, td)
        q_err = quant_error_ms(clip_events, y, SR)

        results.append({
            "id": clip["id"],
            "category": clip["category"],
            "reason": clip["reason"],
            "split": clip["split"],
            "track": clip["track"],
            "start": start,
            "duration": duration,
            "features": {k: clip[k] for k in (
                "notes", "notes_per_second", "median_ioi_ms", "fast_repeats",
                "arpeggio_pairs", "grace_notes", "triplet_pairs", "pedal_fraction",
            ) if k in clip},
            "gt_notes": len(gt),
            "est_notes": len(clip_events),
            "cleaned_notes": len(cleaned),
            "score_notes": score_notes,
            "short_gt": len(short_gt),
            "short_hit": short_hit,
            "short_recall": round(short_hit / len(short_gt), 3) if short_gt else None,
            "onset_error_ms": {
                "median": round(float(np.median(onset_err)), 1) if onset_err else None,
                "p90": round(float(np.percentile(np.abs(onset_err), 90)), 1) if onset_err else None,
                "n": len(onset_err),
            },
            "offset_error_ms": {
                "median": round(float(np.median(offset_err)), 1) if offset_err else None,
                "p90": round(float(np.percentile(np.abs(offset_err), 90)), 1) if offset_err else None,
                "n": len(offset_err),
            },
            "quant_error_ms": {"median": round(q_err[0], 1), "p90": round(q_err[1], 1)}
            if q_err else None,
            "counts": counts,
        })
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)

    aggregate = {}
    for r in results:
        c = r["category"]
        aggregate.setdefault(c, {"n_clips": 0, "gt_notes": 0, "est_notes": 0,
                                 "score_notes": 0, "short_gt": 0, "short_hit": 0})
        a = aggregate[c]
        a["n_clips"] += 1
        a["gt_notes"] += r["gt_notes"]
        a["est_notes"] += r["est_notes"]
        a["score_notes"] += max(0, r["score_notes"])
        a["short_gt"] += r["short_gt"]
        a["short_hit"] += r["short_hit"]
    out = {"version": "hard_passages_v1", "model": args.model,
           "manifest": os.path.basename(args.manifest),
           "results": results, "aggregate": aggregate}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
