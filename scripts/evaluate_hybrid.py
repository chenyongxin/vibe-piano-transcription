"""混合评测：分类解码出音符，再用 onset_reg 局部修正起点。

用于验证 8 头模型是否能在分类 note F1 基本不降的前提下，把 onset 10 ms 指标提上去。
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import mir_eval.transcription as mt

from piano2.constants import DECODE_DEFAULTS, EVAL_TRACKS, MAESTRO_ROOT
from piano2.core import NoteEvent
from piano2.decode import DecodeConfig, _refine_peak, decode
from piano2.eval.metrics import note_metrics, to_arrays
from piano2.inference import compute_cqt, infer_sliding_window, load_model
from piano2.notation import events_from_midi

HOLDOUT = "2015/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_06_R1_2015_wav--6"
FPS = 31.25


def refine_near(curve, time_s, radius_frames=1, min_value=0.10):
    """只在事件起点 ±radius 帧内找局部最大并插值，防止被远处的峰带偏。"""
    if len(curve) < 3:
        return time_s
    frame = min(max(1, int(round(time_s * FPS))), len(curve) - 2)
    lo = max(1, frame - radius_frames)
    hi = min(len(curve) - 2, frame + radius_frames)
    best = None
    for f in range(lo, hi + 1):
        if curve[f] >= curve[f - 1] and curve[f] >= curve[f + 1] and curve[f] >= min_value:
            if best is None or curve[f] > curve[best]:
                best = f
    return _refine_peak(curve, best) / FPS if best is not None else time_s


def onset_f1(ref_iv, ref_p, est_iv, est_p, tol):
    if len(ref_iv) == 0 or len(est_iv) == 0:
        return 0.0
    matching = mt.match_notes(ref_iv, ref_p, est_iv, est_p,
                              onset_tolerance=tol / 1000.0, pitch_tolerance=50.0,
                              offset_ratio=None)
    tp = len(matching)
    p = tp / len(est_iv)
    r = tp / len(ref_iv)
    return 2 * p * r / (p + r) if p + r else 0.0


def eval_track(model, stem, device):
    wav = os.path.join(str(MAESTRO_ROOT), stem + ".wav")
    midi = os.path.join(str(MAESTRO_ROOT), stem + ".midi")
    gt = events_from_midi(midi)
    ref_iv = np.array([[s, e] for _p, s, e, _v in gt], float)
    ref_p = np.array([p for p, _s, _e, _v in gt], int)
    probs = infer_sliding_window(model, compute_cqt(wav), device,
                                 with_offset=True, with_regression=True)
    base = decode(probs[0], probs[1], probs[2],
                  DecodeConfig(**DECODE_DEFAULTS, use_offset_end=True),
                  offset_probs=probs[3])
    refined = []
    for ev in base:
        key = ev.pitch - 21
        start = refine_near(probs[4][:, key], ev.start)
        refined.append(NoteEvent(ev.pitch, start, ev.end, ev.velocity))

    out = {}
    for tag, events in (("base", base), ("refined", refined)):
        est_iv, est_p = to_arrays(events)
        m = note_metrics(ref_iv, ref_p, est_iv, est_p)
        out[tag] = {
            "note_F1": float(m["note_F1"]),
            "onset50_F1": float(m["onset_F1"]),
        }
        for tol in (30, 20, 10):
            out[tag][f"onset{tol}_F1"] = float(onset_f1(ref_iv, ref_p, est_iv, est_p, tol))
        out[tag]["n_est"] = len(events)
    out["n_ref"] = len(gt)
    return out


def main():
    ap = argparse.ArgumentParser(description="混合 onset 评测")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    model = load_model(args.model, args.device)
    tracks = {n: EVAL_TRACKS[n] for n in ("haydn", "mozart", "beethoven")}
    tracks["holdout"] = HOLDOUT
    result = {"model": args.model, "tracks": {}}
    for name, stem in tracks.items():
        result["tracks"][name] = eval_track(model, stem, args.device)
        r = result["tracks"][name]
        print(name, "base_note", round(r["base"]["note_F1"], 4),
              "refined_note", round(r["refined"]["note_F1"], 4),
              "onset10", round(r["base"]["onset10_F1"], 3),
              "->", round(r["refined"]["onset10_F1"], 3), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)
        print("写入:", args.out)


if __name__ == "__main__":
    main()
