"""在固定评测集上评测模型：note-level onset F1 与含时值 F1。

评测集是 `constants.EVAL_TRACKS` 里的三首 MAESTRO test 曲目（疏 / 中 / 密），
与 M0/M0.5 报告用的同一批，便于横向比较。

用法：
    python scripts/evaluate.py --model outputs/M1/pilot_onset2/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from piano2.constants import EVAL_TRACKS, MAESTRO_ROOT, OUTPUT_ROOT
from piano2.decode import DecodeConfig, decode
from piano2.eval.metrics import note_metrics, to_arrays
from piano2.inference import compute_cqt, infer_sliding_window, load_model
from piano2.notation import events_from_midi


def main():
    ap = argparse.ArgumentParser(description="固定评测集上的 note-level 评测")
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(OUTPUT_ROOT / "eval" / "eval_report.json"))
    args = ap.parse_args()

    model = load_model(args.model, args.device)
    cfg = DecodeConfig()
    rows = {}
    for name, stem in EVAL_TRACKS.items():
        wav = os.path.join(MAESTRO_ROOT, stem + ".wav")
        midi = os.path.join(MAESTRO_ROOT, stem + ".midi")
        gt = events_from_midi(midi)
        cqt = compute_cqt(wav)
        onset, frame, vel = infer_sliding_window(model, cqt, args.device)
        est = decode(onset, frame, vel, cfg)
        m = note_metrics(*to_arrays_obj(gt), *to_arrays(est))
        rows[name] = m
        print(f"{name:10s} onset F1={m['onset_F1']:.4f}  note F1={m['note_F1']:.4f}  "
              f"n_pred={m['n_est']} n_ref={m['n_ref']}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"model": args.model, "tracks": rows}, fh, ensure_ascii=False, indent=2)
    print("写入:", args.out)


def to_arrays_obj(events):
    """把 notation.events_from_midi 返回的元组列表转成 (intervals, pitches)。"""
    if not events:
        return np.zeros((0, 2)), np.zeros(0, dtype=int)
    return (np.array([[e[1], e[2]] for e in events], dtype=float),
            np.array([e[0] for e in events], dtype=int))


if __name__ == "__main__":
    main()
