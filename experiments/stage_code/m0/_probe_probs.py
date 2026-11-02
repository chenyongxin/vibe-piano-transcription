"""探测：当前模型的概率分布与阈值敏感性。

回答两个问题：
  1. onset/frame 概率是否整体压缩（例如最大值只有 0.3）？
  2. 把 onset 阈值从 0.05 扫到 0.50，onset-only F1 能到多少？
如果低阈值下 F1 明显回升，说明是阈值/校准问题；否则是模型本身欠训练。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = "/Docker/git/piano_practice"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "m0"))

import mir_eval.transcription as mtran  # noqa: E402
from scipy.signal import find_peaks  # noqa: E402

import notation_v1  # noqa: E402
from transcribe_readable import (  # noqa: E402
    HOP_LENGTH, SR, compute_cqt, infer_sliding_window, load_model,
)

MAESTRO = "/Docker/maestro-v3.0.0"
CKPT = os.path.join(ROOT, "trained_data", "checkpoints", "model_best.pth")


def pct(a):
    return {f"P{int(q*100)}": float(np.percentile(a, q * 100)) for q in (0.5, 0.9, 0.99, 0.999, 1.0)}


def main():
    df = pd.read_csv(os.path.join(MAESTRO, "maestro-v3.0.0.csv"))
    row = df[df.split == "test"].sort_values("duration").iloc[1]
    wav = os.path.join(MAESTRO, row["audio_filename"])
    mid = os.path.join(MAESTRO, row["midi_filename"])
    print("track:", os.path.basename(row["audio_filename"]), "dur", round(row["duration"], 1))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(CKPT, device)
    cqt = compute_cqt(wav)
    x = torch.tensor(cqt, dtype=torch.float32).unsqueeze(0).to(device)
    onset, frame, vel = infer_sliding_window(model, x)

    print("\nonset 概率分布:", pct(onset))
    print("frame 概率分布:", pct(frame))
    print("velocity 概率分布:", pct(vel))
    print("onset>0.4 占比:", float((onset > 0.4).mean()))
    print("frame>0.4 占比:", float((frame > 0.4).mean()))

    # GT
    pm_events = notation_v1.events_from_midi(mid)
    ref_iv = np.array([[s, e] for (_p, s, e, _v) in pm_events])
    ref_p = np.array([p for (p, _s, _e, _v) in pm_events])

    print("\n阈值扫描（onset-only F1，容差 50ms，pitch 精确）:")
    rows = []
    for thr in (0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50):
        iv, pp = [], []
        for k in range(88):
            peaks, _ = find_peaks(onset[:, k], height=thr, prominence=0.02, distance=5)
            for f in peaks:
                t = f * HOP_LENGTH / SR
                iv.append([t, t + 0.15])
                pp.append(k + 21)
        if not iv:
            print(f"  thr={thr:.2f}: 0 notes")
            continue
        iv = np.asarray(iv)
        pp = np.asarray(pp)
        p_, r_, f_, _ = mtran.precision_recall_f1_overlap(
            ref_iv, ref_p, iv, pp,
            onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None,
        )
        rows.append((thr, len(iv), p_, r_, f_))
        print(f"  thr={thr:.2f}: n_pred={len(iv):5d}  P={p_:.3f}  R={r_:.3f}  F1={f_:.3f}")

    best = max(rows, key=lambda r: r[4]) if rows else None
    print("\n最佳:", best)


if __name__ == "__main__":
    main()
