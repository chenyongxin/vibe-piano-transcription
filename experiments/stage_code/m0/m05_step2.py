"""M0.5 步骤 2：状态机参数网格搜索（谐波抑制固定为 off）。

实验设计：
    调参集：Haydn（稀疏）、Mozart（中等）
    holdout：Beethoven（密集），只在最后评一次
    主指标：onset-only F1（容差 50 ms，音高精确）
    硬约束：precision >= 0.85（三首都要）

用法（容器内）：
    cd /Docker/git/piano_practice/m0 && \
    PYTHONPATH=/Docker/git/piano_practice/m0:/Docker/git/piano_practice \
    python m05_step2.py
"""

from __future__ import annotations

import csv
import itertools
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
TUNE = ("haydn", "mozart")
HOLDOUT = "beethoven"
CEILING = {"haydn": 0.950, "mozart": 0.949, "beethoven": 0.797}  # M0 纯峰值天花板
OLD_PRECISION = {"haydn": 0.9366, "mozart": 0.9780, "beethoven": 0.9276}

GRID = {
    "onset_threshold": [0.40, 0.60, 0.70],
    "min_note_frames": [2, 3, 6],
    "release_frames": [3, 5, 10],
    "min_reattack_gap": [0.08, 0.15],
    "merge_gap": [0.05, 0.20],
}


def load_gt(name, midi_path):
    """读 GT，返回 (intervals, pitches)。"""
    events = notation_v1.events_from_midi(midi_path)
    iv = np.array([[s, e] for (_p, s, e, _v) in events], dtype=float)
    pp = np.array([p for (p, _s, _e, _v) in events], dtype=int)
    return iv, pp


def events_to_arrays(events):
    """NoteEvent 列表 → (intervals, pitches)。"""
    if not events:
        return np.zeros((0, 2)), np.zeros(0, dtype=int)
    return (
        np.array([[e.start, e.end] for e in events], dtype=float),
        np.array([e.pitch for e in events], dtype=int),
    )


def onset_metrics(ref_iv, ref_p, est_iv, est_p):
    """算 onset-only 的 P/R/F1。"""
    if len(est_iv) == 0:
        return 0.0, 0.0, 0.0
    p, r, f, _ = mtran.precision_recall_f1_overlap(
        ref_iv, ref_p, est_iv, est_p,
        onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None,
    )
    return float(p), float(r), float(f)


def main():
    os.makedirs(OUT, exist_ok=True)
    caches = {name: prob_cache.load_cache(name) for name in TUNE + (HOLDOUT,)}
    gts = {
        "haydn": load_gt("haydn", prob_cache.GT_MIDI["haydn"]),
        "mozart": load_gt("mozart", prob_cache.GT_MIDI["mozart"]),
        "beethoven": load_gt("beethoven", prob_cache.GT_MIDI["beethoven"]),
    }
    for name in TUNE + (HOLDOUT,):
        print(f"{name}: T={caches[name]['onset'].shape[0]}  帧, GT={len(gts[name][1])} 音")

    keys = list(GRID)
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    print(f"网格: {len(combos)} 组配置；调参曲目 {TUNE}；holdout {HOLDOUT}")

    rows = []
    for ci, values in enumerate(combos, 1):
        cfg = decode_v2.DecodeConfig(harmonic_mode="off", **dict(zip(keys, values)))
        scores = {}
        for name in TUNE:
            c = caches[name]
            ev = decode_v2.decode(c["onset"], c["frame"], c["velocity"], cfg)
            iv, pp = events_to_arrays(ev)
            p, r, f = onset_metrics(*gts[name], iv, pp)
            scores[name] = {"P": p, "R": r, "F1": f}
        mean_f1 = float(np.mean([scores[n]["F1"] for n in TUNE]))
        mean_p = float(np.mean([scores[n]["P"] for n in TUNE]))
        row = {**dict(zip(keys, values)), "tune_F1": mean_f1, "tune_P": mean_p,
               "haydn_F1": scores["haydn"]["F1"], "mozart_F1": scores["mozart"]["F1"],
               "haydn_P": scores["haydn"]["P"], "mozart_P": scores["mozart"]["P"],
               "haydn_R": scores["haydn"]["R"], "mozart_R": scores["mozart"]["R"]}
        rows.append(row)
        if ci % 20 == 0 or ci == len(combos):
            print(f"  {ci}/{len(combos)} 当前最好 tune_F1={max(r2['tune_F1'] for r2 in rows):.4f}")

    rows.sort(key=lambda r: -r["tune_F1"])
    print("\n=== 调参集 top 10 ===")
    print(f"{'thr':>5s} {'minF':>5s} {'rel':>4s} {'gap':>5s} {'merge':>6s}  {'tune_F1':>8s} {'haydn':>7s} {'mozart':>7s} {'tuneP':>6s}")
    for r in rows[:10]:
        print(f"{r['onset_threshold']:>5.2f} {r['min_note_frames']:>5d} {r['release_frames']:>4d} "
              f"{r['min_reattack_gap']:>5.2f} {r['merge_gap']:>6.2f}  {r['tune_F1']:>8.4f} "
              f"{r['haydn_F1']:>7.4f} {r['mozart_F1']:>7.4f} {r['tune_P']:>6.4f}")

    # holdout：取调参集前 5 名，在 Beethoven 上各评一次
    print(f"\n=== holdout（{HOLDOUT}，天花板 {CEILING[HOLDOUT]:.3f}）前 5 名 ===")
    holdout_rows = []
    for r in rows[:5]:
        cfg = decode_v2.DecodeConfig(
            harmonic_mode="off",
            onset_threshold=r["onset_threshold"], min_note_frames=r["min_note_frames"],
            release_frames=r["release_frames"], min_reattack_gap=r["min_reattack_gap"],
            merge_gap=r["merge_gap"],
        )
        c = caches[HOLDOUT]
        ev = decode_v2.decode(c["onset"], c["frame"], c["velocity"], cfg)
        iv, pp = events_to_arrays(ev)
        p, rec, f = onset_metrics(*gts[HOLDOUT], iv, pp)
        holdout_rows.append({"cfg": cfg, "P": p, "R": rec, "F1": f, "n": len(ev),
                             **{k: r[k] for k in keys}})
        print(f"  thr={r['onset_threshold']:.2f} minF={r['min_note_frames']} rel={r['release_frames']} "
              f"gap={r['min_reattack_gap']:.2f} merge={r['merge_gap']:.2f} -> "
              f"P={p:.3f} R={rec:.3f} F1={f:.3f} (n={len(ev)})")

    best = holdout_rows[0]
    cfg = best["cfg"]
    print("\n=== 验收检查（90% 天花板 / precision ≥0.85）===")
    ok = True
    per_track = {}
    for name in TUNE + (HOLDOUT,):
        c = caches[name]
        ev = decode_v2.decode(c["onset"], c["frame"], c["velocity"], cfg)
        iv, pp = events_to_arrays(ev)
        p, rec, f = onset_metrics(*gts[name], iv, pp)
        target = 0.9 * CEILING[name]
        pass_f = f >= target
        pass_p = p >= 0.85
        ok = ok and pass_f and pass_p
        per_track[name] = {"P": p, "R": rec, "F1": f, "target": target, "n": len(ev)}
        print(f"  {name:10s} P={p:.3f} R={rec:.3f} F1={f:.3f}  目标 F1≥{target:.3f} {'✓' if pass_f else '✗'}  "
              f"P≥0.85 {'✓' if pass_p else '✗'}")
    print("总判定:", "通过" if ok else "未通过")

    # 落盘
    with open(os.path.join(OUT, "step2_grid.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})
    best_cfg_dict = {
        "onset_threshold": cfg.onset_threshold, "onset_prominence": cfg.onset_prominence,
        "frame_off_threshold": cfg.frame_off_threshold,
        "min_note_frames": cfg.min_note_frames, "release_frames": cfg.release_frames,
        "min_reattack_gap": cfg.min_reattack_gap, "merge_gap": cfg.merge_gap,
        "harmonic_mode": cfg.harmonic_mode,
    }
    with open(os.path.join(OUT, "step2_best_config.json"), "w", encoding="utf-8") as fh:
        json.dump({"best_config": best_cfg_dict, "per_track": per_track,
                   "passed": ok}, fh, ensure_ascii=False, indent=2)
    print("\n写入:", os.path.join(OUT, "step2_grid.csv"))
    print("写入:", os.path.join(OUT, "step2_best_config.json"))

    # 用最佳参数给 nezha2 出谱（看效果）
    c = prob_cache.load_cache("nezha2")
    ev = decode_v2.decode(c["onset"], c["frame"], c["velocity"], cfg)
    wav = os.path.join(ROOT, "data", "nezha2.wav")
    xml = os.path.join(OUT, "nezha2_step2.musicxml")
    notation_v1.write_musicxml([(e.pitch, e.start, e.end, e.velocity) for e in ev], wav, xml,
                               title="nezha2 step2 best state machine")
    m = score_metrics.score_metrics(xml, audio_duration_s=c["duration_s"])
    print(f"\nnezha2 步骤 2 谱面: events={len(ev)} 音符头={m['n_noteheads']} 连音={m['tie_count']} "
          f"休止={m['rest_count']} 黑音符={m['black_ratio']:.3f} 小节={m['measures_per_part']}")
    render_preview.render_png(xml, os.path.join(OUT, "nezha2_step2_p1.png"), zoom=1.4)


if __name__ == "__main__":
    main()
