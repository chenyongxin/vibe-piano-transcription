"""评测带回归头的模型（亚帧解码）：note F1 + 多容差 onset F1。"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import mir_eval.transcription as mtran

from piano2.constants import EVAL_TRACKS, MAESTRO_ROOT
from piano2.decode import decode_subframe
from piano2.eval.metrics import note_metrics, to_arrays
from piano2.inference import compute_cqt, infer_sliding_window, load_model
from piano2.notation import events_from_midi

HOLDOUT = "2015/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_06_R1_2015_wav--6"


def onset_counts(ref_iv, ref_p, est_iv, est_p, tol):
    """返回 (tp, n_ref, n_est)。"""
    if len(ref_iv) == 0 or len(est_iv) == 0:
        return 0, len(ref_iv), len(est_iv)
    matching = mtran.match_notes(ref_iv, ref_p, est_iv, est_p,
                                 onset_tolerance=tol, pitch_tolerance=50.0,
                                 offset_ratio=None)
    return len(matching), len(ref_iv), len(est_iv)


def f1(tp, nr, ne):
    """由计数算 F1。"""
    p = tp / ne if ne else 0.0
    r = tp / nr if nr else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def eval_track(model, stem, device):
    """对一个曲目做亚帧解码并返回指标。"""
    midi = os.path.join(str(MAESTRO_ROOT), stem + ".midi")
    wav = os.path.join(str(MAESTRO_ROOT), stem + ".wav")
    gt = events_from_midi(midi)
    ref_iv = np.array([[s, e] for _p, s, e, _v in gt], float)
    ref_p = np.array([p for p, _s, _e, _v in gt], int)
    o, f, v, of, onr, offr = infer_sliding_window(
        model, compute_cqt(wav), device, with_offset=True, with_regression=True)
    events = decode_subframe(onr, offr, v)
    est_iv, est_p = to_arrays(events)
    note = note_metrics(ref_iv, ref_p, est_iv, est_p)
    onset = {}
    for tol_ms in (50, 30, 20, 10):
        tp, nr, ne = onset_counts(ref_iv, ref_p, est_iv, est_p, tol_ms / 1000.0)
        onset[f"onset{tol_ms}_F1"] = f1(tp, nr, ne)
    return {"n_pred": len(events), "n_ref": len(gt),
            "note_F1": float(note["note_F1"]),
            "note_P": float(note["note_P"]),
            "note_R": float(note["note_R"]),
            **{k: float(vv) for k, vv in onset.items()}}


def main():
    ap = argparse.ArgumentParser(description="回归模型亚帧评测")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="outputs/eval/reg_eval.json")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    model = load_model(args.model, args.device)
    result = {"model": args.model, "tracks": {}}
    for name in ("haydn", "mozart", "beethoven"):
        result["tracks"][name] = eval_track(model, EVAL_TRACKS[name], args.device)
        r = result["tracks"][name]
        print(f"{name}: note {r['note_F1']:.3f} onset50 {r['onset50_F1']:.3f} "
              f"onset10 {r['onset10_F1']:.3f} n_pred {r['n_pred']}", flush=True)
    result["tracks"]["holdout"] = eval_track(model, HOLDOUT, args.device)
    r = result["tracks"]["holdout"]
    print(f"holdout: note {r['note_F1']:.3f} onset50 {r['onset50_F1']:.3f} "
          f"onset10 {r['onset10_F1']:.3f} n_pred {r['n_pred']}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print("写入:", args.out)


if __name__ == "__main__":
    main()
