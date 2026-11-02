"""M0.5 步骤 2 总结算：三种解码器对照 + nezha2 出声谱。

解码器：
    A 原管线：状态机 + 全局谐波抑制（旧）
    B 状态机：网格搜索出的最优参数，谐波抑制关闭
    C 新解码：峰值定起点 + frame 定终点，谐波抑制关闭

指标：onset-only F1（主）与 note-with-offset F1（参考），precision 硬约束。

用法（容器内）：
    cd /Docker/git/piano_practice/m0 && \
    PYTHONPATH=/Docker/git/piano_practice/m0:/Docker/git/piano_practice \
    python m05_step2b.py
"""

from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

ROOT = "/Docker/git/piano_practice"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "m0"))

import mir_eval.transcription as mtran  # noqa: E402

import decode_v2  # noqa: E402
import notation_v1  # noqa: E402
import prob_cache  # noqa: E402
import render_preview  # noqa: E402
import score_metrics  # noqa: E402

OUT = os.path.join(ROOT, "outputs", "M0", "m05")
TRACKS = ("haydn", "mozart", "beethoven")
CEILING = {"haydn": 0.950, "mozart": 0.949, "beethoven": 0.797}


def load_gt(name):
    ev = notation_v1.events_from_midi(prob_cache.GT_MIDI[name])
    return (np.array([[s, e] for (_p, s, e, _v) in ev], float),
            np.array([p for (p, _s, _e, _v) in ev], int))


def arrays(events):
    if not events:
        return np.zeros((0, 2)), np.zeros(0, int)
    return (np.array([[e.start, e.end] for e in events], float),
            np.array([e.pitch for e in events], int))


def metrics(ref_iv, ref_p, est_iv, est_p):
    if len(est_iv) == 0:
        return dict(onset_P=0.0, onset_R=0.0, onset_F1=0.0, note_F1=0.0, n=0)
    p1, r1, f1, _ = mtran.precision_recall_f1_overlap(
        ref_iv, ref_p, est_iv, est_p, onset_tolerance=0.05,
        pitch_tolerance=50.0, offset_ratio=None)
    p2, r2, f2, _ = mtran.precision_recall_f1_overlap(
        ref_iv, ref_p, est_iv, est_p, onset_tolerance=0.05,
        pitch_tolerance=50.0, offset_ratio=0.2, offset_min_tolerance=0.05)
    return dict(onset_P=float(p1), onset_R=float(r1), onset_F1=float(f1),
                note_F1=float(f2), n=int(len(est_p)))


def main():
    os.makedirs(OUT, exist_ok=True)
    caches = {n: prob_cache.load_cache(n) for n in TRACKS}
    gts = {n: load_gt(n) for n in TRACKS}

    best = json.load(open(os.path.join(OUT, "step2_best_config.json"), encoding="utf-8"))
    b = best["best_config"]
    cfg_a = decode_v2.DecodeConfig(harmonic_mode="global", harmonic_strength=0.5,
                                   method="state_machine")
    cfg_b = decode_v2.DecodeConfig(
        harmonic_mode="off", method="state_machine",
        onset_threshold=b["onset_threshold"], onset_prominence=b["onset_prominence"],
        min_note_frames=b["min_note_frames"], release_frames=b["release_frames"],
        min_reattack_gap=b["min_reattack_gap"], merge_gap=b["merge_gap"])
    cfg_c = decode_v2.DecodeConfig(harmonic_mode="off", method="peak_frame",
                                   onset_threshold=0.60, onset_prominence=0.005,
                                   peak_distance=5, frame_off_threshold=0.18,
                                   release_frames=3)

    rows = []
    print(f"{'track':10s} {'decoder':12s} {'n':>6s} {'P':>6s} {'R':>6s} {'onsetF1':>8s} {'noteF1':>7s}")
    for name in TRACKS:
        c = caches[name]
        for label, cfg in (("A_old", cfg_a), ("B_state", cfg_b), ("C_peakframe", cfg_c)):
            ev = decode_v2.decode(c["onset"], c["frame"], c["velocity"], cfg)
            m = metrics(*gts[name], *arrays(ev))
            rows.append({"track": name, "decoder": label, **m})
            print(f"{name:10s} {label:12s} {m['n']:>6d} {m['onset_P']:>6.3f} {m['onset_R']:>6.3f} "
                  f"{m['onset_F1']:>8.3f} {m['note_F1']:>7.3f}")
        print()

    # 验收判定（新解码 C）：90% 天花板 + precision >= 0.85
    print("=== 新解码 C 的验收检查 ===")
    ok = True
    for name in TRACKS:
        r = next(x for x in rows if x["track"] == name and x["decoder"] == "C_peakframe")
        target = 0.9 * CEILING[name]
        pass_f = r["onset_F1"] >= target
        pass_p = r["onset_P"] >= 0.85
        ok = ok and pass_f and pass_p
        print(f"  {name:10s} F1={r['onset_F1']:.3f} (目标≥{target:.3f}) {'✓' if pass_f else '✗'}  "
              f"P={r['onset_P']:.3f} (≥0.85) {'✓' if pass_p else '✗'}")
    print("  总判定:", "通过" if ok else "未通过")
    print("  备注：密集曲目 precision 0.775 低于 0.85 底线，留给步骤 3 的条件式谐波抑制。")

    with open(os.path.join(OUT, "step2_decoder_comparison.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})
    print("\n写入:", os.path.join(OUT, "step2_decoder_comparison.csv"))

    # nezha2：新解码出谱
    c = prob_cache.load_cache("nezha2")
    ev = decode_v2.decode(c["onset"], c["frame"], c["velocity"], cfg_c)
    wav = os.path.join(ROOT, "data", "nezha2.wav")
    xml = os.path.join(OUT, "nezha2_step2c.musicxml")
    notation_v1.write_musicxml([(e.pitch, e.start, e.end, e.velocity) for e in ev], wav, xml,
                               title="nezha2 step2c peak-frame")
    m = score_metrics.score_metrics(xml, audio_duration_s=c["duration_s"])
    print(f"\nnezha2（新解码 C）: events={len(ev)} 音符头={m['n_noteheads']} 连音={m['tie_count']} "
          f"休止={m['rest_count']} 黑音符={m['black_ratio']:.3f} 小节={m['measures_per_part']} "
          f"时长误差={m['duration_err_ratio']:.3f}")
    render_preview.render_png(xml, os.path.join(OUT, "nezha2_step2c_p1.png"), zoom=1.4)


if __name__ == "__main__":
    main()
