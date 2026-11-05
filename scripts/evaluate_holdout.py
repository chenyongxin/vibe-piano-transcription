"""在留出集（holdout）上评测任意检查点。

留出曲目沿用 M0.5 的划分（`2015/...06_R1_2015...`），只用于最终对比，不参与调参。

用法：
    python scripts/evaluate_holdout.py --model outputs/M1/d2_frame_key/checkpoints/best.pt \
        --out outputs/eval/d2_frame_key_holdout.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from piano2.constants import DECODE_DEFAULTS, MAESTRO_ROOT
from piano2.decode import DecodeConfig, decode
from piano2.eval.metrics import note_metrics, to_arrays
from piano2.inference import compute_cqt, infer_sliding_window, load_model
from piano2.notation import events_from_midi

OLD_STEP3 = "/Docker/git/piano_practice/outputs/M0/m05/step3b_final.json"


def main():
    ap = argparse.ArgumentParser(description="留出集评测")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--offset-end", action="store_true")
    args = ap.parse_args()

    with open(OLD_STEP3, encoding="utf-8") as fh:
        step3 = json.load(fh)
    midi_rel = step3["holdout2"]["midi"]
    midi = os.path.join(str(MAESTRO_ROOT), midi_rel)
    wav = midi[:-5] + ".wav"

    model = load_model(args.model, args.device)
    cfg = DecodeConfig(**DECODE_DEFAULTS, use_offset_end=args.offset_end)
    probs = infer_sliding_window(model, compute_cqt(wav), args.device,
                                 with_offset=args.offset_end)
    events = decode(probs[0], probs[1], probs[2], cfg,
                    offset_probs=probs[3] if args.offset_end else None)
    gt = events_from_midi(midi)
    ref_iv = np.array([[s, e] for _p, s, e, _v in gt], float)
    ref_p = np.array([p for p, _s, _e, _v in gt], int)
    est_iv, est_p = to_arrays(events)
    m = note_metrics(ref_iv, ref_p, est_iv, est_p)
    result = {"model": args.model, "track": midi_rel,
              "n_pred": len(events), "n_ref": len(gt),
              **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}}
    print(json.dumps(result, ensure_ascii=False))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
