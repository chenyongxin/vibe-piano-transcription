"""步骤 2 追加消融：找出状态机到底在哪一步丢掉召回。"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.signal import find_peaks

ROOT = "/Docker/git/piano_practice"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "m0"))

import mir_eval.transcription as mtran  # noqa: E402

import decode_v2  # noqa: E402
import notation_v1  # noqa: E402
import prob_cache  # noqa: E402

TUNE = ("haydn", "mozart")
HID = 16000 / 512


def load_gt(midi_path):
    ev = notation_v1.events_from_midi(midi_path)
    return (np.array([[s, e] for (_p, s, e, _v) in ev], float),
            np.array([p for (p, _s, _e, _v) in ev], int))


def arrays(events):
    if not events:
        return np.zeros((0, 2)), np.zeros(0, int)
    return (np.array([[e.start, e.end] for e in events], float),
            np.array([e.pitch for e in events], int))


def metrics(ref_iv, ref_p, est_iv, est_p):
    if len(est_iv) == 0:
        return 0.0, 0.0, 0.0
    p, r, f, _ = mtran.precision_recall_f1_overlap(
        ref_iv, ref_p, est_iv, est_p, onset_tolerance=0.05,
        pitch_tolerance=50.0, offset_ratio=None)
    return float(p), float(r), float(f)


def main():
    caches = {n: prob_cache.load_cache(n) for n in TUNE}
    gts = {"haydn": load_gt(prob_cache.GT_MIDI["haydn"]),
           "mozart": load_gt(prob_cache.GT_MIDI["mozart"])}
    base = dict(onset_threshold=0.60, onset_prominence=0.005, peak_distance=5,
                frame_off_threshold=0.18, min_note_frames=2, release_frames=3,
                min_reattack_gap=0.08, merge_gap=0.05, harmonic_mode="off")
    cases = {
        "best": base,
        "minF=1": {**base, "min_note_frames": 1},
        "release=1": {**base, "release_frames": 1},
        "merge=0": {**base, "merge_gap": 0.0},
        "gap=0": {**base, "min_reattack_gap": 0.0},
        "dist=1": {**base, "peak_distance": 1},
        "frame_off=0.05": {**base, "frame_off_threshold": 0.05},
        "all_loose": {**base, "min_note_frames": 1, "release_frames": 1, "merge_gap": 0.0,
                      "min_reattack_gap": 0.0, "peak_distance": 1},
    }
    print(f"{'case':16s} {'haydn n':>8s} {'haydn R':>8s} {'haydn F1':>9s} {'mozart n':>9s} {'mozart R':>9s} {'mozart F1':>10s}")
    for name, kw in cases.items():
        cfg = decode_v2.DecodeConfig(**kw)
        line = f"{name:16s}"
        for track in TUNE:
            c = caches[track]
            ev = decode_v2.decode(c["onset"], c["frame"], c["velocity"], cfg)
            p, r, f = metrics(*gts[track], *arrays(ev))
            line += f" {len(ev):>8d} {r:>8.3f} {f:>9.3f}"
        print(line)

    # 参照：纯峰值检测（不经过状态机）在同一阈值下的数量与召回
    print("\n[参照] 纯峰值检测（thr=0.6, distance=5）:")
    for track in TUNE:
        c = caches[track]
        onset = c["onset"]
        iv, pp = [], []
        for k in range(88):
            peaks, _ = find_peaks(onset[:, k], height=0.6, prominence=0.005, distance=5)
            for f in peaks:
                t = f * HID
                iv.append([t, t + 0.15])
                pp.append(k + 21)
        p, r, f = metrics(*gts[track], np.asarray(iv, float), np.asarray(pp, int))
        print(f"  {track:10s} n={len(pp):6d} P={p:.3f} R={r:.3f} F1={f:.3f}")


if __name__ == "__main__":
    main()
