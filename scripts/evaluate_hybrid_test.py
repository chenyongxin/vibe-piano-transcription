"""在 MAESTRO test split 上评测混合模型：分类音符 + onset_reg 亚帧修正起点。

用法（容器内）：
    python scripts/evaluate_hybrid_test.py \
        --model outputs/M1/hybrid_reg2/checkpoints/best.pt \
        --out outputs/eval/hybrid_reg2_test.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import mir_eval.transcription as mtran

from piano2.constants import DECODE_DEFAULTS, MAESTRO_ROOT
from piano2.core import NoteEvent
from piano2.decode import DecodeConfig, _refine_peak, decode
from piano2.eval.metrics import to_arrays
from piano2.inference import compute_cqt, infer_sliding_window, load_model
from piano2.notation import events_from_midi

FPS = 31.25


def refine_near(curve, time_s, radius_frames=1, min_value=0.10):
    """只在事件起点 ±1 帧内找峰并插值。"""
    frame = int(round(time_s * FPS))
    lo = max(1, frame - radius_frames)
    hi = min(len(curve) - 1, frame + radius_frames)
    best = None
    for f in range(lo, hi + 1):
        if curve[f] >= curve[f - 1] and curve[f] >= curve[f + 1] and curve[f] >= min_value:
            if best is None or curve[f] > curve[best]:
                best = f
    return _refine_peak(curve, best) / FPS if best is not None else time_s


def match_counts(ref_iv, ref_p, est_iv, est_p, onset_tol=0.05, offset_ratio=None):
    """返回 (tp, n_ref, n_est)。"""
    if len(est_iv) == 0 or len(ref_iv) == 0:
        return 0, len(ref_iv), len(est_iv)
    kwargs = dict(onset_tolerance=onset_tol, pitch_tolerance=50.0)
    if offset_ratio is None:
        matching = mtran.match_notes(ref_iv, ref_p, est_iv, est_p, offset_ratio=None, **kwargs)
    else:
        matching = mtran.match_notes(ref_iv, ref_p, est_iv, est_p,
                                     offset_ratio=offset_ratio,
                                     offset_min_tolerance=0.05, **kwargs)
    return len(matching), len(ref_iv), len(est_iv)


def prf(tp, n_ref, n_est):
    """由计数算 P/R/F1。"""
    p = tp / n_est if n_est else 0.0
    r = tp / n_ref if n_ref else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def main():
    ap = argparse.ArgumentParser(description="混合模型 MAESTRO test split 评测")
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="outputs/eval/hybrid_reg2_test.json")
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(str(MAESTRO_ROOT), "maestro-v3.0.0.csv"))
    test = df[df["split"] == "test"].reset_index(drop=True)
    if args.limit:
        test = test.head(args.limit)

    model = load_model(args.model, args.device)
    cfg = DecodeConfig(**DECODE_DEFAULTS, use_offset_end=True)
    totals = {"onset50": [0, 0, 0], "onset30": [0, 0, 0], "onset20": [0, 0, 0],
              "onset10": [0, 0, 0], "note": [0, 0, 0]}
    per_track = {}
    for i, row in test.iterrows():
        midi = os.path.join(str(MAESTRO_ROOT), str(row["midi_filename"]))
        wav = os.path.join(str(MAESTRO_ROOT), str(row["audio_filename"]))
        gt = events_from_midi(midi)
        ref_iv = np.array([[s, e] for _p, s, e, _v in gt], float)
        ref_p = np.array([p for p, _s, _e, _v in gt], int)

        probs = infer_sliding_window(model, compute_cqt(wav), args.device,
                                     with_offset=True, with_regression=True)
        base = decode(probs[0], probs[1], probs[2], cfg, offset_probs=probs[3])
        events = []
        for ev in base:
            key = ev.pitch - 21
            start = refine_near(probs[4][:, key], ev.start)
            events.append(NoteEvent(ev.pitch, start, ev.end, ev.velocity))
        est_iv, est_p = to_arrays(events)

        row_counts = {}
        for key, tol, ratio in (
            ("onset50", 0.05, None), ("onset30", 0.03, None),
            ("onset20", 0.02, None), ("onset10", 0.01, None),
            ("note", 0.05, 0.2),
        ):
            tp, nr, ne = match_counts(ref_iv, ref_p, est_iv, est_p, tol, ratio)
            totals[key][0] += tp
            totals[key][1] += nr
            totals[key][2] += ne
            row_counts[key] = {"tp": tp, "n_ref": nr, "n_est": ne}
        per_track[str(row["midi_filename"])] = row_counts
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(test)}", flush=True)

    result = {key: prf(*vals) for key, vals in totals.items()}
    result["_counts"] = {key: {"tp": v[0], "n_ref": v[1], "n_est": v[2]}
                         for key, v in totals.items()}
    result["_model"] = args.model
    result["_hybrid"] = True
    result["_n_tracks"] = len(test)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"summary": result, "per_track": per_track}, fh,
                  ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
