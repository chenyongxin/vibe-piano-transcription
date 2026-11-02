"""M0 主脚本：评测自检 + GT/模型两条输入 + 旧/新记谱对比 + 渲染 + 报告。

分四步：
  A. 评测链路自检（oracle）：把 GT 当预测喂给 mir_eval，分数必须是 1.0；
     再做抖动/丢音/时值缩放，确认指标对各类错误敏感、方向正确。
  B. GT → 旧记谱 v0 / 新记谱 v1，比较谱面指标（疏/中/密三首）。
  C. 当前模型 → 与 GT 比 note-level F1（onset-only 与含时值两种口径），
     并做系统延迟对齐，回答「模型到底什么水平」。
  D. 当前模型 → v1 出谱，渲染，和旧产物对比。

用法（容器内）：
    cd /Docker/git/piano_practice/m0 && \
    PYTHONPATH=/Docker/git/piano_practice/m0:/Docker/git/piano_practice \
    python run_m0.py
"""

from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np
import pandas as pd
import pretty_midi
import torch
import librosa

ROOT = "/Docker/git/piano_practice"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "m0"))

import mir_eval.transcription as mtran  # noqa: E402

import notation_v1  # noqa: E402
import render_preview  # noqa: E402
import score_metrics  # noqa: E402
from transcribe_readable import (  # noqa: E402
    HOP_LENGTH,
    NoteEvent,
    SR,
    compute_cqt,
    infer_sliding_window,
    load_model,
    probabilities_to_events,
    suppress_harmonic_onsets,
    write_readable_musicxml,
)

MAESTRO = "/Docker/maestro-v3.0.0"
PROCESSED_TEST = "/Docker/maestro_processed/test"
OUT = os.path.join(ROOT, "outputs", "M0")
CKPT = os.path.join(ROOT, "trained_data", "checkpoints", "model_best.pth")
FPS = 16000 / 512  # 31.25 帧/秒
SOLO_WAV = os.path.join(ROOT, "data", "nezha2.wav")


# ─────────────────────────── 评测工具 ───────────────────────────


def load_gt(midi_path):
    """从 MIDI 读 GT 音符，返回 (intervals (N,2), pitches (N,))。"""
    pm = pretty_midi.PrettyMIDI(midi_path)
    intervals, pitches = [], []
    for inst in pm.instruments:
        for note in inst.notes:
            intervals.append([note.start, note.end])
            pitches.append(note.pitch)
    order = np.argsort([iv[0] for iv in intervals])
    return np.array(intervals, dtype=float)[order], np.array(pitches, dtype=int)[order]


def note_metrics(ref_iv, ref_p, est_iv, est_p):
    """同时算「只看起点」与「起点+时值」两种口径的 note F1。

    参数:
        ref_iv, ref_p: GT 区间与音高。
        est_iv, est_p: 预测区间与音高。

    返回:
        dict: onset_only_P/R/F1 与 note_with_offset_P/R/F1。
    """
    if len(est_iv) == 0:
        return {
            "onset_P": 0.0, "onset_R": 0.0, "onset_F1": 0.0,
            "note_P": 0.0, "note_R": 0.0, "note_F1": 0.0,
            "n_ref": int(len(ref_p)), "n_est": 0,
        }
    p1, r1, f1, _ = mtran.precision_recall_f1_overlap(
        ref_iv, ref_p, est_iv, est_p,
        onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None,
    )
    p2, r2, f2, _ = mtran.precision_recall_f1_overlap(
        ref_iv, ref_p, est_iv, est_p,
        onset_tolerance=0.05, pitch_tolerance=50.0,
        offset_ratio=0.2, offset_min_tolerance=0.05,
    )
    return {
        "onset_P": float(p1), "onset_R": float(r1), "onset_F1": float(f1),
        "note_P": float(p2), "note_R": float(r2), "note_F1": float(f2),
        "n_ref": int(len(ref_p)), "n_est": int(len(est_p)),
    }


def frame_roll_from_midi(midi_path):
    """用 pretty_midi 生成含踏板的逐帧标签（31.25 fps），与训练标签口径一致。"""
    pm = pretty_midi.PrettyMIDI(midi_path)
    roll = pm.get_piano_roll(fs=FPS)[21:109, :].T
    return roll > 0


def frame_roll_from_events(events, n_frames):
    """由预测音符事件生成逐帧矩阵（不加踏板，因为模型没有踏板头）。"""
    roll = np.zeros((n_frames, 88), dtype=bool)
    for ev in events:
        a = max(0, int(round(ev.start * FPS)))
        b = min(n_frames, int(round(ev.end * FPS)))
        if b > a and 0 <= ev.pitch - 21 < 88:
            roll[a:b, ev.pitch - 21] = True
    return roll


def frame_f1(ref_roll, est_roll):
    """逐帧二分类的 P/R/F1（所有键、所有帧一起算）。"""
    tp = int(np.logical_and(ref_roll, est_roll).sum())
    fp = int(np.logical_and(~ref_roll, est_roll).sum())
    fn = int(np.logical_and(ref_roll, ~est_roll).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"frame_P": p, "frame_R": r, "frame_F1": f, "tp": tp, "fp": fp, "fn": fn}


def best_latency(ref_iv, ref_p, est_iv, est_p, lo=-0.20, hi=0.20, step=0.01):
    """估计预测相对 GT 的系统延迟（毫秒），返回 (offset_s, 对齐后的 onset_F1)。"""
    best = (-9.9, None)
    if len(est_iv) == 0:
        return 0.0, 0.0
    for off in np.arange(lo, hi + 1e-9, step):
        shifted = est_iv + off
        m = note_metrics(ref_iv, ref_p, shifted, est_p)
        if m["onset_F1"] > best[0]:
            best = (m["onset_F1"], float(off))
    return best[1], best[0]


def oracle_self_check(midi_path):
    """评测链路自检：GT 当预测必须满分，各类扰动要朝正确方向掉分。"""
    ref_iv, ref_p = load_gt(midi_path)
    rng = np.random.default_rng(0)
    cases = {"exact_GT": (ref_iv.copy(), ref_p.copy())}
    cases["onset_jitter_32ms"] = (ref_iv + 0.032, ref_p)
    cases["onset_jitter_64ms"] = (ref_iv + 0.064, ref_p)
    dur = ref_iv[:, 1] - ref_iv[:, 0]
    cases["duration_x1.1"] = (np.stack([ref_iv[:, 0], ref_iv[:, 0] + dur * 1.1], axis=1), ref_p)
    cases["duration_x0.5"] = (np.stack([ref_iv[:, 0], ref_iv[:, 0] + dur * 0.5], axis=1), ref_p)
    keep = rng.random(len(ref_p)) > 0.1
    cases["drop_10pct"] = (ref_iv[keep], ref_p[keep])
    shuffled = ref_p.copy()
    rng.shuffle(shuffled)
    cases["shuffle_pitch"] = (ref_iv.copy(), shuffled)

    rows = []
    for name, (iv, p) in cases.items():
        m = note_metrics(ref_iv, ref_p, iv, p)
        rows.append({"case": name, **m})
    return rows


# ─────────────────────────── 模型推理 ───────────────────────────


def infer_probs(wav_path, model, device):
    """跑当前模型，返回 (onset, frame, velocity) 三个概率矩阵。"""
    cqt = compute_cqt(wav_path)
    x = torch.tensor(cqt, dtype=torch.float32).unsqueeze(0).to(device)
    return infer_sliding_window(model, x)


def decode_peaks(onset, threshold=0.6, note_len=0.15):
    """最简单的解码：每个键找 onset 峰，给一个固定时值。用来测「检测上限」。"""
    from scipy.signal import find_peaks

    iv, pp = [], []
    for k in range(88):
        peaks, _ = find_peaks(onset[:, k], height=threshold, prominence=0.02, distance=5)
        for f in peaks:
            t = f * HOP_LENGTH / SR
            iv.append([t, t + note_len])
            pp.append(k + 21)
    return iv, pp


def decode_state(onset, frame, vel, threshold=0.6, harmonic_suppress=0.0):
    """迟滞状态机解码；harmonic_suppress=0 表示不做谐波抑制。

    参数:
        onset, frame, vel: 模型概率矩阵。
        threshold (float): onset 峰阈值。
        harmonic_suppress (float): 谐波抑制强度，0 关掉。

    返回:
        list[NoteEvent]: 音符事件。
    """
    working = onset
    if harmonic_suppress > 0:
        coarse = probabilities_to_events(
            onset, frame, vel, onset_threshold=threshold + 0.1,
            onset_prominence=0.06, frame_off_threshold=0.18,
            min_note_frames=2, release_frames=5, min_reattack_gap=0.08,
        )
        anchors = list({e.pitch for e in coarse})
        working = suppress_harmonic_onsets(onset, anchors, strength=harmonic_suppress)
    return probabilities_to_events(
        working, frame, vel,
        onset_threshold=threshold, onset_prominence=0.02, frame_off_threshold=0.18,
        min_note_frames=2, release_frames=5, min_reattack_gap=0.08,
    )


def events_to_arrays(events):
    """NoteEvent 列表 → (intervals, pitches)。"""
    if not events:
        return np.zeros((0, 2)), np.zeros((0,), dtype=int)
    iv = np.array([[e.start, e.end] for e in events], dtype=float)
    p = np.array([e.pitch for e in events], dtype=int)
    return iv, p


# ─────────────────────────── 主流程 ───────────────────────────


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(os.path.join(MAESTRO, "maestro-v3.0.0.csv"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    # 选 3 首 GT：按密度 10/50/90 分位（用 npz 的 onset 正帧数估密度）
    rows = []
    for idx, row in df[df.split == "test"].iterrows():
        stem = os.path.basename(str(row["audio_filename"]))[:-4]
        npz = os.path.join(PROCESSED_TEST, f"{idx:04d}_{stem}.npz")
        if not os.path.exists(npz):
            continue
        data = np.load(npz)
        dens = float(np.count_nonzero(data["onset"])) / 3.0 / float(row["duration"])
        rows.append({"idx": idx, "row": row, "density": dens})
    tracks = sorted(rows, key=lambda r: r["density"])
    picks = {
        "sparse": tracks[int(0.10 * (len(tracks) - 1))],
        "medium": tracks[int(0.50 * (len(tracks) - 1))],
        "dense": tracks[int(0.90 * (len(tracks) - 1))],
    }

    # ---------- A. oracle 自检 ----------
    print("\n" + "=" * 70)
    print("A. 评测链路 oracle 自检（GT 当预测，应得 1.0）")
    print("=" * 70)
    oracle_rows = oracle_self_check(
        os.path.join(MAESTRO, picks["sparse"]["row"]["midi_filename"])
    )
    for r in oracle_rows:
        print(f"  {r['case']:20s} onset_F1={r['onset_F1']:.3f}  note_F1={r['note_F1']:.3f}  "
              f"R={r['onset_R']:.3f}  n_est={r['n_est']}")
    with open(os.path.join(OUT, "oracle_check.json"), "w", encoding="utf-8") as fh:
        json.dump(oracle_rows, fh, ensure_ascii=False, indent=2)

    # ---------- B. GT → v0 / v1 谱面 ----------
    print("\n" + "=" * 70)
    print("B. GT 直通记谱：v0（旧）vs v1（新）")
    print("=" * 70)
    score_rows = []
    for tag, item in picks.items():
        row = item["row"]
        wav = os.path.join(MAESTRO, row["audio_filename"])
        mid = os.path.join(MAESTRO, row["midi_filename"])
        duration = float(row["duration"])
        events = notation_v1.events_from_midi(mid)
        ev0 = [NoteEvent(p, s, e, v) for (p, s, e, v) in events]
        out_v0 = os.path.join(OUT, f"cmp_{tag}_v0.musicxml")
        write_readable_musicxml(ev0, out_v0, bpm=120, grid=0.25)
        out_v1 = os.path.join(OUT, f"cmp_{tag}_v1.musicxml")
        notation_v1.write_musicxml(events, wav, out_v1, title=f"{tag} GT v1")
        for name, path in (("v0", out_v0), ("v1", out_v1)):
            m = score_metrics.score_metrics(path, audio_duration_s=duration)
            m.update({"tier": tag, "version": name, "density": item["density"]})
            score_rows.append(m)
            png = os.path.join(OUT, f"cmp_{tag}_{name}_p1.png")
            try:
                render_preview.render_png(path, png, page=1, zoom=1.4)
            except Exception as exc:
                print(f"  render {tag}/{name} failed: {exc}")
        print(f"  {tag}: density={item['density']:.1f}/s  GT notes={len(events)}")

    # ---------- C+D. 模型推理 ----------
    print("\n" + "=" * 70)
    print("C. 当前模型 vs GT（note-level 指标 + 系统延迟对齐）")
    print("=" * 70)
    model = load_model(CKPT, device)
    model_rows = []
    for tag, item in picks.items():
        row = item["row"]
        wav = os.path.join(MAESTRO, row["audio_filename"])
        mid = os.path.join(MAESTRO, row["midi_filename"])
        print(f"\n  [{tag}] {os.path.basename(row['audio_filename'])}")
        onset_p, frame_p, vel_p = infer_probs(wav, model, device)
        ref_iv, ref_p = load_gt(mid)
        # 三种解码对照：纯峰值 / 状态机（无谐波抑制）/ 正式管线（含谐波抑制）
        variants = {}
        iv, pp = decode_peaks(onset_p, threshold=0.6)
        variants["peaks@0.6"] = (iv, pp)
        ev_nosup = decode_state(onset_p, frame_p, vel_p, threshold=0.6, harmonic_suppress=0.0)
        variants["state@0.6"] = ([[e.start, e.end] for e in ev_nosup], [e.pitch for e in ev_nosup])
        ev_off = decode_state(onset_p, frame_p, vel_p, threshold=0.4, harmonic_suppress=0.5)
        variants["official@0.4"] = ([[e.start, e.end] for e in ev_off], [e.pitch for e in ev_off])

        for vname, (vii, vpp) in variants.items():
            vii = np.asarray(vii, dtype=float) if len(vii) else np.zeros((0, 2))
            vpp = np.asarray(vpp, dtype=int) if len(vpp) else np.zeros(0, dtype=int)
            m = note_metrics(ref_iv, ref_p, vii, vpp)
            model_rows.append(
                {
                    "tier": tag, "density": item["density"], "decoder": vname,
                    "n_pred": len(vpp), "n_ref": int(len(ref_p)),
                    "onset_P": m["onset_P"], "onset_R": m["onset_R"], "onset_F1": m["onset_F1"],
                    "note_F1": m["note_F1"],
                }
            )
            print(f"    {vname:14s} pred={len(vpp):5d}/{len(ref_p):5d}  "
                  f"onset P={m['onset_P']:.3f} R={m['onset_R']:.3f} F1={m['onset_F1']:.3f}  "
                  f"note F1={m['note_F1']:.3f}")

        chosen_events = ev_nosup
        est_iv, est_p = events_to_arrays(chosen_events)
        off, f1_shift = best_latency(ref_iv, ref_p, est_iv, est_p)
        ref_roll = frame_roll_from_midi(mid)
        n_frames = min(len(onset_p), len(ref_roll))
        est_roll = frame_roll_from_events(chosen_events, n_frames)
        fmet = frame_f1(ref_roll[:n_frames], est_roll)
        model_rows.append(
            {
                "tier": tag, "density": item["density"], "decoder": "chosen_flagship",
                "n_pred": len(chosen_events), "n_ref": int(len(ref_p)),
                "onset_P": note_metrics(ref_iv, ref_p, est_iv + off, est_p)["onset_P"],
                "onset_R": note_metrics(ref_iv, ref_p, est_iv + off, est_p)["onset_R"],
                "onset_F1": note_metrics(ref_iv, ref_p, est_iv + off, est_p)["onset_F1"],
                "note_F1": note_metrics(ref_iv, ref_p, est_iv + off, est_p)["note_F1"],
                "latency_ms": off * 1000.0, **fmet,
            }
        )
        print(f"    chosen(state@0.6, 对齐后): onset_F1={f1_shift:.3f}  "
              f"latency={off*1000:.0f}ms  frame_F1={fmet['frame_F1']:.3f}")
        # 模型事件 → v1 出谱
        out_xml = os.path.join(OUT, f"model_{tag}_v1.musicxml")
        duration = float(row["duration"])
        try:
            notation_v1.write_musicxml(
                [(e.pitch, e.start, e.end, e.velocity) for e in chosen_events], wav, out_xml,
                title=f"model {tag} v1",
            )
            render_preview.render_png(out_xml, os.path.join(OUT, f"model_{tag}_v1_p1.png"), zoom=1.4)
            m = score_metrics.score_metrics(out_xml, audio_duration_s=duration)
            m.update({"tier": tag, "version": "v1"})
            score_rows.append(m)
            print(f"    v1 谱面: 黑音符={m['black_ratio']:.3f} 休止占比={m['rest_duration_ratio']:.3f} "
                  f"声部/小节={m['voices_per_measure_mean']:.2f}")
        except Exception as exc:
            print(f"    v1 出谱失败: {exc}")

    # ---------- D. nezha2（用户的实际用例） ----------
    print("\n" + "=" * 70)
    print("D. nezha2.wav：当前模型 → v0/v1 出谱")
    print("=" * 70)
    if os.path.exists(SOLO_WAV):
        onset_p, frame_p, vel_p = infer_probs(SOLO_WAV, model, device)
        events = decode_state(onset_p, frame_p, vel_p, threshold=0.6, harmonic_suppress=0.0)
        print(f"  模型事件数: {len(events)}")
        duration = float(librosa.get_duration(path=SOLO_WAV))
        ev0 = [NoteEvent(e.pitch, e.start, e.end, e.velocity) for e in events]
        out_v0 = os.path.join(OUT, "solo_nezha2_v0.musicxml")
        write_readable_musicxml(ev0, out_v0, bpm=120, grid=0.25)
        out_v1 = os.path.join(OUT, "solo_nezha2_v1.musicxml")
        notation_v1.write_musicxml(
            [(e.pitch, e.start, e.end, e.velocity) for e in events], SOLO_WAV, out_v1,
            title="nezha2 model v1",
        )
        for name, path in (("v0", out_v0), ("v1", out_v1)):
            m = score_metrics.score_metrics(path, audio_duration_s=duration)
            m.update({"tier": "nezha2", "version": name})
            score_rows.append(m)
            render_preview.render_png(path, os.path.join(OUT, f"solo_nezha2_{name}_p1.png"), zoom=1.4)
            print(f"  {name}: notes={m['n_onsets']} 黑音符={m['black_ratio']:.3f} "
                  f"连音={m['tie_count']} 休止占比={m['rest_duration_ratio']:.3f} "
                  f"声部/小节={m['voices_per_measure_mean']:.2f} 小节数={m['measures_per_part']}")
    else:
        print("  未找到 nezha2.wav，跳过")

    # ---------- 落盘 ----------
    def dump_csv(rows, name, keys):
        if not rows:
            return
        path = os.path.join(OUT, name)
        all_keys = []
        for r in rows:
            for k in r:
                if k not in all_keys:
                    all_keys.append(k)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            for r in rows:
                writer.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
        print("写入:", path)

    dump_csv(oracle_rows, "oracle_check.csv", [])
    dump_csv(score_rows, "score_metrics.csv", [])
    dump_csv(model_rows, "model_vs_gt.csv", [])
    with open(os.path.join(OUT, "model_vs_gt.json"), "w", encoding="utf-8") as fh:
        json.dump(model_rows, fh, ensure_ascii=False, indent=2)
    print("\nM0 完成。产物目录:", OUT)


if __name__ == "__main__":
    main()
