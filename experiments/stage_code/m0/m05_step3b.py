"""M0.5 步骤 3b：扩展网格 + 全新留出曲目（避免在 holdout 上调参）。

方法学修正：
    步骤 3 把 Beethoven 当 holdout，但网格是按稀疏/中等曲目调出来的，
    调出来的抑制强度对密集曲目不够。这里改成：
        开发集：Haydn、Mozart、Beethoven（含密集，专门用来调 precision）
        最终留出：另选一首全新密集曲目（不参与任何调参），只评一次。

用法（容器内）：
    cd /Docker/git/piano_practice/m0 && \
    PYTHONPATH=/Docker/git/piano_practice/m0:/Docker/git/piano_practice \
    python m05_step3b.py
"""

from __future__ import annotations

import csv
import itertools
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

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
FIGS = os.path.join(OUT, "figs")
MAESTRO = "/Docker/maestro-v3.0.0"
PROCESSED_TEST = "/Docker/maestro_processed/test"
FPS = 16000 / 512
DEV = ("haydn", "mozart", "beethoven")
CEILING = {"haydn": 0.950, "mozart": 0.949, "beethoven": 0.797}


def load_gt(name, midi_path):
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
        return dict(P=0.0, R=0.0, F1=0.0, n=0)
    p, r, f, _ = mtran.precision_recall_f1_overlap(
        ref_iv, ref_p, est_iv, est_p, onset_tolerance=0.05,
        pitch_tolerance=50.0, offset_ratio=None)
    return dict(P=float(p), R=float(r), F1=float(f), n=int(len(est_p)))


def pick_holdout2():
    """从 test 集里挑一首未用过的、密集的曲目做最终留出。"""
    df = pd.read_csv(os.path.join(MAESTRO, "maestro-v3.0.0.csv"))
    used = {
        "MIDI-Unprocessed_XP_19_R1_2004_01-02_ORIG_MID--AUDIO_19_R1_2004_02_Track02_wav",
        "MIDI-Unprocessed_08_R1_2008_01-05_ORIG_MID--AUDIO_08_R1_2008_wav--2",
        "MIDI-UNPROCESSED_14-15_R1_2014_MID--AUDIO_15_R1_2014_wav--2",
    }
    cands = []
    for idx, row in df[df.split == "test"].iterrows():
        stem = os.path.basename(str(row["audio_filename"]))[:-4]
        if stem in used:
            continue
        npz = os.path.join(PROCESSED_TEST, f"{idx:04d}_{stem}.npz")
        if not os.path.exists(npz):
            continue
        dur = float(row["duration"])
        if not (110 <= dur <= 260):  # 控制推理时间
            continue
        data = np.load(npz)
        dens = float(np.count_nonzero(data["onset"])) / 3.0 / dur
        cands.append((dens, idx, row))
    cands.sort(key=lambda x: -x[0])
    # 取密度最接近 16 音/秒 的一首
    best = min(cands, key=lambda x: abs(x[0] - 16.0))
    print(f"holdout2 候选: {os.path.basename(best[2]['audio_filename'])} 密度 {best[0]:.1f}/s")
    return best[2]


def ensure_cache_holdout2(row):
    name = "holdout2"
    path = prob_cache.cache_path(name)
    if os.path.exists(path):
        print("holdout2 已有缓存")
        return
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wav = os.path.join(MAESTRO, row["audio_filename"])
    print(f"为 holdout2 计算概率: {wav}")
    prob_cache.compute_and_cache(name, wav, device)
    print("holdout2 缓存完成")


def main():
    os.makedirs(FIGS, exist_ok=True)
    row2 = pick_holdout2()
    ensure_cache_holdout2(row2)

    caches = {n: prob_cache.load_cache(n) for n in DEV}
    caches["holdout2"] = prob_cache.load_cache("holdout2")
    gts = {n: load_gt(n, prob_cache.GT_MIDI[n]) for n in DEV}
    gts["holdout2"] = load_gt("holdout2", os.path.join(MAESTRO, row2["midi_filename"]))

    # 基线：步骤 2 的峰值+frame 解码
    cfg_c = decode_v2.DecodeConfig(harmonic_mode="off", method="peak_frame",
                                   onset_threshold=0.60, onset_prominence=0.005,
                                   peak_distance=5, frame_off_threshold=0.18,
                                   release_frames=3)
    base = {n: metrics(*gts[n], *arrays(decode_v2.decode(
        caches[n]["onset"], caches[n]["frame"], caches[n]["velocity"], cfg_c)))
        for n in DEV + ("holdout2",)}
    print("\n=== 基线 C（无抑制） ===")
    for n, m in base.items():
        print(f"  {n:10s} P={m['P']:.4f} R={m['R']:.4f} F1={m['F1']:.4f} n={m['n']}")

    # 扩展网格：在开发集（含密集曲目 Beethoven）上调参
    print("\n=== 扩展网格（开发集 = Haydn + Mozart + Beethoven） ===")
    rows = []
    for strength, window, frame_max, gate in itertools.product(
            (0.5, 0.7, 0.9, 1.0), (5, 8, 12), (0.6, 1.0), (0.0, 0.15)):
        cfg = cfg_c.with_(harmonic_mode="conditional", harmonic_strength=strength,
                          harmonic_window_frames=window, harmonic_frame_max=frame_max,
                          frame_gate=gate)
        res = {n: metrics(*gts[n], *arrays(decode_v2.decode(
            caches[n]["onset"], caches[n]["frame"], caches[n]["velocity"], cfg)))
            for n in DEV}
        min_p = min(res[n]["P"] for n in DEV)
        mean_f1 = float(np.mean([res[n]["F1"] for n in DEV]))
        rows.append(dict(strength=strength, window=window, frame_max=frame_max, gate=gate,
                         min_P=min_p, mean_F1=mean_f1,
                         **{f"{n}_P": res[n]["P"] for n in DEV},
                         **{f"{n}_F1": res[n]["F1"] for n in DEV}))
    rows.sort(key=lambda r: (-(r["min_P"] >= 0.85), -r["mean_F1"]))
    print("top 8（优先 precision≥0.85，再比 F1）:")
    for r in rows[:8]:
        print(f"  s={r['strength']:.1f} w={r['window']:2d} fmax={r['frame_max']:.1f} gate={r['gate']:.2f}"
              f" -> minP={r['min_P']:.4f} meanF1={r['mean_F1']:.4f} "
              f"(bee P={r['beethoven_P']:.3f} F1={r['beethoven_F1']:.3f})")
    best = rows[0]
    cfg_final = cfg_c.with_(harmonic_mode="conditional",
                            harmonic_strength=best["strength"],
                            harmonic_window_frames=best["window"],
                            harmonic_frame_max=best["frame_max"],
                            frame_gate=best["gate"])

    # 最终留出：holdout2 只评一次
    print("\n=== 最终留出（不参与调参） ===")
    final_rows = []
    for n in DEV + ("holdout2",):
        ev = decode_v2.decode(caches[n]["onset"], caches[n]["frame"], caches[n]["velocity"], cfg_final)
        m = metrics(*gts[n], *arrays(ev))
        final_rows.append({"track": n, **m})
        print(f"  {n:10s} P={m['P']:.4f} R={m['R']:.4f} F1={m['F1']:.4f} n={m['n']}")

    # 验收：F1 ≥ 90% 天花板（holdout2 的天花板未知，用同一解码器的峰值上限近似）
    ev_peak = decode_v2.decode(caches["holdout2"]["onset"], caches["holdout2"]["frame"],
                               caches["holdout2"]["velocity"], cfg_c)
    peak_f1 = metrics(*gts["holdout2"], *arrays(ev_peak))["F1"]
    print(f"\nholdout2 的天花板（峰值+frame，无抑制）F1={peak_f1:.4f}")
    for r in final_rows:
        target = 0.9 * (CEILING.get(r["track"], peak_f1))
        ok_f = r["F1"] >= target
        ok_p = r["P"] >= 0.85
        print(f"  {r['track']:10s} F1={r['F1']:.3f}≥{target:.3f} {'✓' if ok_f else '✗'}  "
              f"P={r['P']:.3f}≥0.85 {'✓' if ok_p else '✗'}")

    with open(os.path.join(OUT, "step3b_grid.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})
    with open(os.path.join(OUT, "step3b_final.json"), "w", encoding="utf-8") as fh:
        json.dump({"holdout2": {"audio": row2["audio_filename"], "midi": row2["midi_filename"],
                               "peak_ceiling_F1": peak_f1},
                   "final_config": {k: getattr(cfg_final, k) for k in (
                       "method", "harmonic_mode", "harmonic_strength",
                       "harmonic_window_frames", "harmonic_frame_max", "frame_gate",
                       "onset_threshold", "onset_prominence", "peak_distance",
                       "frame_off_threshold", "release_frames")},
                   "results": final_rows}, fh, ensure_ascii=False, indent=2)

    # ---------- 图 1：四种解码器对比（含新数据） ----------
    cfg_a = decode_v2.DecodeConfig(harmonic_mode="global", harmonic_strength=0.5,
                                   method="state_machine")
    cfg_b = decode_v2.DecodeConfig(
        harmonic_mode="off", method="state_machine", onset_threshold=0.70,
        onset_prominence=0.06, min_note_frames=2, release_frames=3,
        min_reattack_gap=0.08, merge_gap=0.05)
    decoders = {"A_old": cfg_a, "B_state": cfg_b, "C_peakframe": cfg_c, "D_conditional": cfg_final}
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.0), sharey=True)
    labels = list(decoders)
    colors = ["#90caf9", "#a5d6a7", "#ef9a9a"]
    for ax, name in zip(axes, DEV + ("holdout2",)):
        res = {l: metrics(*gts[name], *arrays(decode_v2.decode(
            caches[name]["onset"], caches[name]["frame"], caches[name]["velocity"], cfg)))
            for l, cfg in decoders.items()}
        xs = np.arange(len(labels))
        for i, metric in enumerate(("P", "R", "F1")):
            ax.bar(xs + (i - 1) * 0.26, [res[l][metric] for l in labels], 0.26,
                   label=metric, color=colors[i], edgecolor="black", linewidth=0.4)
        ax.set_xticks(xs)
        ax.set_xticklabels([l.replace("_", "\n") for l in labels], fontsize=7)
        ax.set_title(name)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("score")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Decoders: precision / recall / F1 (A old, B state machine, C peak+frame, D conditional suppression)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig1_decoders.png"), dpi=130)
    plt.close(fig)

    # ---------- 图 2：误检案例（用 Beethoven 开发集） ----------
    c = caches["beethoven"]
    sup_full = decode_v2.conditional_suppress(c["onset"], c["frame"], cfg_final)
    ev_c = decode_v2.decode(c["onset"], c["frame"], c["velocity"], cfg_c)
    ev_d = decode_v2.decode(c["onset"], c["frame"], c["velocity"], cfg_final)
    iv_c, p_c = arrays(ev_c)
    iv_d, p_d = arrays(ev_d)
    ref_iv, ref_p = gts["beethoven"]
    d_set = {(int(p), round(t, 2)) for t, p in zip(iv_d[:, 0], p_d)}
    case = None
    for t, p in zip(iv_c[:, 0], p_c):
        if (int(p), round(t, 2)) in d_set:
            continue
        close = (np.abs(ref_iv[:, 0] - t) < 0.06) & (ref_p == p)
        if close.any():
            continue
        for offset, _decay in decode_v2.HARMONIC_OFFSETS:
            anchor_pitch = int(p) - offset
            if np.any((p_c == anchor_pitch) & (np.abs(iv_c[:, 0] - t) < 0.12)):
                case = (t, int(p), anchor_pitch, offset)
                break
        if case:
            break
    if case:
        t0, k2, k1, offset = case
        half = 0.6
        i0 = max(0, int((t0 - half) * FPS))
        i1 = min(len(c["onset"]), int((t0 + half) * FPS))
        tt = np.arange(i0, i1) / FPS
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(tt, c["onset"][i0:i1, k1 - 21], color="#1976d2",
                     label=f"onset: anchor MIDI {k1}")
        axes[0].plot(tt, c["onset"][i0:i1, k2 - 21], color="#d81b60",
                     label=f"onset: harmonic MIDI {k2} (before)")
        axes[0].plot(tt, sup_full[i0:i1, k2 - 21], color="#d81b60", linestyle="--",
                     label=f"onset: harmonic MIDI {k2} (after conditional suppression)")
        axes[0].axhline(cfg_c.onset_threshold, color="gray", linestyle=":", label="detection threshold")
        axes[0].axvline(t0, color="black", linestyle="--", linewidth=0.8)
        axes[0].set_ylabel("onset probability")
        axes[0].legend(fontsize=7, loc="upper right")
        axes[0].set_title(f"False positive at MIDI {k2} ({t0:.2f}s) removed: anchor MIDI {k1} +{offset} semitones")
        axes[1].plot(tt, c["frame"][i0:i1, k1 - 21], color="#1976d2", label=f"frame: MIDI {k1}")
        axes[1].plot(tt, c["frame"][i0:i1, k2 - 21], color="#d81b60", label=f"frame: MIDI {k2}")
        axes[1].axhline(cfg_c.frame_off_threshold, color="gray", linestyle=":", label="frame gate")
        axes[1].set_ylabel("frame probability")
        axes[1].set_xlabel("time (s)")
        axes[1].legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS, "fig2_case_study.png"), dpi=130)
        plt.close(fig)

    # ---------- 图 3：抑制范围热图 ----------
    ratio = np.where(c["onset"] > 1e-6, sup_full / np.maximum(c["onset"], 1e-6), 1.0)
    win = int(8 * FPS)
    best_start, best_hits = 0, -1
    for s in range(0, len(ratio) - win, win // 2):
        hits = int((ratio[s:s + win] < 0.999).sum())
        if hits > best_hits:
            best_hits, best_start = hits, s
    seg = ratio[best_start:best_start + win, :].T
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(seg, aspect="auto", origin="lower", cmap="viridis", vmin=0.5, vmax=1.0,
                   extent=[best_start / FPS, (best_start + win) / FPS, 21, 108])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("MIDI pitch")
    ax.set_title("Conditional suppression: ratio after/before (dark = suppressed); local in time")
    fig.colorbar(im, ax=ax, label="onset ratio")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig3_suppression_map.png"), dpi=130)
    plt.close(fig)

    # ---------- nezha2 出谱 ----------
    c = prob_cache.load_cache("nezha2")
    ev = decode_v2.decode(c["onset"], c["frame"], c["velocity"], cfg_final)
    wav = os.path.join(ROOT, "data", "nezha2.wav")
    xml = os.path.join(OUT, "nezha2_step3.musicxml")
    notation_v1.write_musicxml([(e.pitch, e.start, e.end, e.velocity) for e in ev], wav, xml,
                               title="nezha2 step3")
    m = score_metrics.score_metrics(xml, audio_duration_s=c["duration_s"])
    print(f"\nnezha2（D）: events={len(ev)} 音符头={m['n_noteheads']} 连音={m['tie_count']} "
          f"休止={m['rest_count']} 黑音符={m['black_ratio']:.3f} 小节={m['measures_per_part']}")
    render_preview.render_png(xml, os.path.join(OUT, "nezha2_step3_p1.png"), zoom=1.4)
    print("写入:", os.path.join(OUT, "step3b_grid.csv"))


if __name__ == "__main__":
    main()
