"""步骤 2 追加诊断 2：验证「峰值定起点 + frame 定长度」的解码是否更接近天花板。

同时修复上一版探针里的时间单位 bug（把帧号乘了帧率，应为除）。
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.signal import find_peaks

ROOT = "/Docker/git/piano_practice"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "m0"))

import mir_eval.transcription as mtran  # noqa: E402

import notation_v1  # noqa: E402
import prob_cache  # noqa: E402

TUNE = ("haydn", "mozart")
HOLDOUT = "beethoven"
SR, HOP = 16000, 512
FPS = SR / HOP


def frame_to_time(idx):
    return idx * HOP / SR


def load_gt(midi_path):
    ev = notation_v1.events_from_midi(midi_path)
    return (np.array([[s, e] for (_p, s, e, _v) in ev], float),
            np.array([p for (p, _s, _e, _v) in ev], int))


def metrics(ref_iv, ref_p, est_iv, est_p):
    if len(est_iv) == 0:
        return 0.0, 0.0, 0.0
    p, r, f, _ = mtran.precision_recall_f1_overlap(
        ref_iv, ref_p, est_iv, est_p, onset_tolerance=0.05,
        pitch_tolerance=50.0, offset_ratio=None)
    return float(p), float(r), float(f)


def peaks_only(onset, thr=0.6, prom=0.005, dist=5, note_len=0.15):
    iv, pp = [], []
    for k in range(88):
        peaks, _ = find_peaks(onset[:, k], height=thr, prominence=prom, distance=dist)
        for f in peaks:
            t = frame_to_time(f)
            iv.append([t, t + note_len])
            pp.append(k + 21)
    return np.asarray(iv, float) if iv else np.zeros((0, 2)), np.asarray(pp, int)


def peaks_with_frame_offsets(onset, frame, thr=0.6, prom=0.005, dist=5,
                             frame_off=0.18, release=3, max_len=6.0):
    """峰值决定起点，frame 概率决定终点；不做重击间隔/最短时值/gonged 合并。"""
    iv, pp = [], []
    T = len(onset)
    for k in range(88):
        peaks, _ = find_peaks(onset[:, k], height=thr, prominence=prom, distance=dist)
        f_curve = frame[:, k]
        for f in peaks:
            start = frame_to_time(f)
            j = f
            rel = 0
            end = frame_to_time(T - 1)
            while j < T:
                if f_curve[j] < frame_off:
                    rel += 1
                    if rel >= release:
                        end = frame_to_time(j - release + 1)
                        break
                else:
                    rel = 0
                j += 1
            if end <= start:
                end = start + 0.05
            end = min(end, start + max_len)
            iv.append([start, end])
            pp.append(k + 21)
    return np.asarray(iv, float) if iv else np.zeros((0, 2)), np.asarray(pp, int)


def main():
    caches = {n: prob_cache.load_cache(n) for n in TUNE + (HOLDOUT,)}
    gts = {n: load_gt(prob_cache.GT_MIDI[n]) for n in TUNE + (HOLDOUT,)}

    print("=== 纯峰值（修正时间单位后）===")
    for name in TUNE + (HOLDOUT,):
        iv, pp = peaks_only(caches[name]["onset"])
        p, r, f = metrics(*gts[name], iv, pp)
        print(f"  {name:10s} n={len(pp):6d} P={p:.3f} R={r:.3f} F1={f:.3f}")

    print("\n=== 峰值定起点 + frame 定终点（无状态机门限）===")
    for name in TUNE + (HOLDOUT,):
        iv, pp = peaks_with_frame_offsets(caches[name]["onset"], caches[name]["frame"])
        p, r, f = metrics(*gts[name], iv, pp)
        print(f"  {name:10s} n={len(pp):6d} P={p:.3f} R={r:.3f} F1={f:.3f}")


if __name__ == "__main__":
    main()
