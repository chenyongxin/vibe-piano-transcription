"""在同一首曲目上对比三种解码：纯峰值 / 无谐波抑制的迟滞状态机 / 正式管线（含谐波抑制）。"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pretty_midi
import torch

ROOT = "/Docker/git/piano_practice"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "m0"))

import mir_eval.transcription as mtran  # noqa: E402
from scipy.signal import find_peaks  # noqa: E402

import notation_v1  # noqa: E402
from transcribe_readable import (  # noqa: E402
    HOP_LENGTH, SR, compute_cqt, infer_sliding_window, load_model,
    probabilities_to_events, suppress_harmonic_onsets,
)

MAESTRO = "/Docker/piano2_data/raw/maestro-v3.0.0"
CKPT = os.path.join(ROOT, "trained_data", "checkpoints", "model_best.pth")
TRACK_SUFFIX = "MIDI-Unprocessed_08_R1_2008_01-05_ORIG_MID--AUDIO_08_R1_2008_wav--2"


def f1(ref_iv, ref_p, iv, pp):
    if len(iv) == 0:
        return 0.0, 0.0, 0.0, 0
    p, r, f, _ = mtran.precision_recall_f1_overlap(
        ref_iv, ref_p, np.asarray(iv, float), np.asarray(pp), onset_tolerance=0.05,
        pitch_tolerance=50.0, offset_ratio=None,
    )
    return float(p), float(r), float(f), len(pp)


def main():
    df = pd.read_csv(os.path.join(MAESTRO, "maestro-v3.0.0.csv"))
    row = df[df.audio_filename.str.contains(TRACK_SUFFIX)].iloc[0]
    wav = os.path.join(MAESTRO, row["audio_filename"])
    mid = os.path.join(MAESTRO, row["midi_filename"])
    ev = notation_v1.events_from_midi(mid)
    ref_iv = np.array([[s, e] for (_p, s, e, _v) in ev])
    ref_p = np.array([p for (p, _s, _e, _v) in ev])
    print("track:", os.path.basename(wav), "GT notes:", len(ref_p))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(CKPT, device)
    cqt = compute_cqt(wav)
    x = torch.tensor(cqt, dtype=torch.float32).unsqueeze(0).to(device)
    onset, frame, vel = infer_sliding_window(model, x)
    print("onset P99:", float(np.percentile(onset, 99)), "frame P99:", float(np.percentile(frame, 99)))

    # 1) 纯峰值检测
    print("\n[1] 纯峰值检测")
    for thr in (0.2, 0.4, 0.6, 0.8):
        iv, pp = [], []
        for k in range(88):
            peaks, _ = find_peaks(onset[:, k], height=thr, prominence=0.02, distance=5)
            for f in peaks:
                t = f * HOP_LENGTH / SR
                iv.append([t, t + 0.15])
                pp.append(k + 21)
        p, r, f, n = f1(ref_iv, ref_p, iv, pp)
        print(f"  thr={thr:.1f}: n={n:5d} P={p:.3f} R={r:.3f} F1={f:.3f}")

    # 2) 正式状态机，但关掉谐波抑制
    print("\n[2] 迟滞状态机（无谐波抑制）")
    for thr in (0.2, 0.4, 0.6):
        events = probabilities_to_events(
            onset, frame, vel, onset_threshold=thr, onset_prominence=0.02,
            frame_off_threshold=0.18,
        )
        iv = [[e.start, e.end] for e in events]
        pp = [e.pitch for e in events]
        p, r, f, n = f1(ref_iv, ref_p, iv, pp)
        print(f"  thr={thr:.1f}: n={n:5d} P={p:.3f} R={r:.3f} F1={f:.3f}")

    # 3) 正式管线：两轮谐波抑制
    print("\n[3] 正式管线（两轮谐波抑制）")
    for thr in (0.2, 0.4, 0.6):
        coarse = probabilities_to_events(
            onset, frame, vel, onset_threshold=thr + 0.1, onset_prominence=0.06 + 0.04,
            frame_off_threshold=0.18,
        )
        anchors = list({e.pitch for e in coarse})
        onset2 = suppress_harmonic_onsets(onset, anchors, strength=0.5)
        events = probabilities_to_events(
            onset2, frame, vel, onset_threshold=thr, onset_prominence=0.06,
            frame_off_threshold=0.18,
        )
        iv = [[e.start, e.end] for e in events]
        pp = [e.pitch for e in events]
        p, r, f, n = f1(ref_iv, ref_p, iv, pp)
        print(f"  thr={thr:.1f}: anchors={len(anchors)} n={n:5d} P={p:.3f} R={r:.3f} F1={f:.3f}")


if __name__ == "__main__":
    main()
