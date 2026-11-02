"""M0.5 步骤 3：条件式谐波抑制 + 帧一致性门控，把密集曲目的 precision 拉回 0.85。

实验设计：
    调参集：Haydn、Mozart；holdout：Beethoven
    解码器：
        A  原管线（状态机 + 全局谐波抑制）
        B  步骤 2 最优状态机（无抑制）
        C  峰值+frame（无抑制，步骤 2 结论）
        D  C + 条件式谐波抑制
        E  C + 帧一致性门控
        F  C + 条件式抑制 + 门控

产出：
    - 指标 CSV / JSON
    - 三张可视化图：解码器对比、误检案例、抑制作用范围
    - nezha2 谱面

用法（容器内）：
    cd /Docker/git/piano_practice/m0 && \
    PYTHONPATH=/Docker/git/piano_practice/m0:/Docker/git/piano_practice \
    python m05_step3.py
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
TRACKS = ("haydn", "mozart", "beethoven")
TUNE = ("haydn", "mozart")
HOLDOUT = "beethoven"
CEILING = {"haydn": 0.950, "mozart": 0.949, "beethoven": 0.797}
FPS = 16000 / 512


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
        return dict(P=0.0, R=0.0, F1=0.0, n=0)
    p, r, f, _ = mtran.precision_recall_f1_overlap(
        ref_iv, ref_p, est_iv, est_p, onset_tolerance=0.05,
        pitch_tolerance=50.0, offset_ratio=None)
    return dict(P=float(p), R=float(r), F1=float(f), n=int(len(est_p)))


def eval_cfg(caches, gts, names, cfg):
    """在一组曲目上评测一个解码配置，返回逐曲指标。"""
    out = {}
    for name in names:
        c = caches[name]
        ev = decode_v2.decode(c["onset"], c["frame"], c["velocity"], cfg)
        out[name] = metrics(*gts[name], *arrays(ev))
    return out


def main():
    os.makedirs(FIGS, exist_ok=True)
    caches = {n: prob_cache.load_cache(n) for n in TRACKS}
    gts = {n: load_gt(n) for n in TRACKS}
    step2 = json.load(open(os.path.join(OUT, "step2_best_config.json"), encoding="utf-8"))
    b = step2["best_config"]

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

    # ---------- D：条件式谐波抑制网格 ----------
    print("=== D: 条件式谐波抑制（调参集） ===")
    grid_d = []
    for strength, window, frame_max in itertools.product(
            (0.3, 0.5, 0.7), (3, 5, 8), (0.25, 0.4, 0.6)):
        cfg = cfg_c.with_(harmonic_mode="conditional", harmonic_strength=strength,
                          harmonic_window_frames=window, harmonic_frame_max=frame_max)
        res = eval_cfg(caches, gts, TUNE, cfg)
        mean_f1 = float(np.mean([res[n]["F1"] for n in TUNE]))
        min_p = float(min(res[n]["P"] for n in TUNE))
        grid_d.append(dict(strength=strength, window=window, frame_max=frame_max,
                           mean_F1=mean_f1, min_P=min_p,
                           **{f"{n}_F1": res[n]["F1"] for n in TUNE}))
    grid_d.sort(key=lambda r: (-(r["min_P"] >= 0.85), -r["mean_F1"]))
    print("  top 5（先满足 precision≥0.85，再比 F1）:")
    for r in grid_d[:5]:
        print(f"    strength={r['strength']:.1f} window={r['window']} frame_max={r['frame_max']:.2f}"
              f" -> mean_F1={r['mean_F1']:.4f} min_P={r['min_P']:.4f}")
    d = grid_d[0]
    cfg_d = cfg_c.with_(harmonic_mode="conditional", harmonic_strength=d["strength"],
                        harmonic_window_frames=d["window"], harmonic_frame_max=d["frame_max"])

    # ---------- E：帧一致性门控网格 ----------
    print("\n=== E: 帧一致性门控（调参集） ===")
    grid_e = []
    for gate in (0.05, 0.1, 0.15, 0.2, 0.3):
        cfg = cfg_c.with_(frame_gate=gate)
        res = eval_cfg(caches, gts, TUNE, cfg)
        mean_f1 = float(np.mean([res[n]["F1"] for n in TUNE]))
        min_p = float(min(res[n]["P"] for n in TUNE))
        grid_e.append(dict(gate=gate, mean_F1=mean_f1, min_P=min_p))
        print(f"    gate={gate:.2f} -> mean_F1={mean_f1:.4f} min_P={min_p:.4f}")
    gate_best = max(grid_e, key=lambda r: r["mean_F1"])
    cfg_e = cfg_c.with_(frame_gate=gate_best["gate"])

    # ---------- F：抑制 + 门控 ----------
    cfg_f = cfg_d.with_(frame_gate=gate_best["gate"])

    # ---------- 全量评测 ----------
    decoders = {"A_old": cfg_a, "B_state_best": cfg_b, "C_peak_frame": cfg_c,
                "D_conditional": cfg_d, "E_frame_gate": cfg_e, "F_cond_gate": cfg_f}
    all_rows = []
    print("\n=== 六种解码器在三首曲目上的结果 ===")
    print(f"{'decoder':14s} " + " ".join(
        f"{n[:8]:>8s}({CEILING[n]:.3f})" for n in TRACKS))
    for label, cfg in decoders.items():
        line = f"{label:14s} "
        for name in TRACKS:
            r = eval_cfg(caches, gts, (name,), cfg)[name]
            all_rows.append({"decoder": label, "track": name, **r})
            line += f"{r['F1']:>8.3f}({r['P']:.2f})"
        print(line)

    # 选最终方案：满足所有曲目 precision≥0.85 且 F1 ≥ 90% 天花板
    def passes(label):
        rows = [r for r in all_rows if r["decoder"] == label]
        return all(r["P"] >= 0.85 and r["F1"] >= 0.9 * CEILING[r["track"]] for r in rows)

    final_label = None
    for label in ("F_cond_gate", "D_conditional", "E_frame_gate"):
        if passes(label):
            final_label = label
            break
    print("\n最终方案:", final_label or "无（未通过验收）")
    cfg_final = decoders[final_label] if final_label else cfg_d

    with open(os.path.join(OUT, "step3_decoders.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        for r in all_rows:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})

    # ---------- 图 1：解码器对比 ----------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    labels = ["A_old", "B_state_best", "C_peak_frame", "D_conditional", "E_frame_gate", "F_cond_gate"]
    colors = ["#9e9e9e", "#607d8b", "#2196f3", "#4caf50", "#ff9800", "#e91e63"]
    for ax, name in zip(axes, TRACKS):
        sub = {r["decoder"]: r for r in all_rows if r["track"] == name}
        xs = np.arange(len(labels))
        width = 0.26
        for i, metric in enumerate(("P", "R", "F1")):
            vals = [sub[l][metric] for l in labels]
            ax.bar(xs + (i - 1) * width, vals, width, label=metric,
                   color=["#90caf9", "#a5d6a7", "#ef9a9a"][i], edgecolor="black", linewidth=0.4)
        ax.set_xticks(xs)
        ax.set_xticklabels([l.replace("_", "\n") for l in labels], fontsize=7)
        ax.set_title(f"{name} (ceiling F1={CEILING[name]:.3f})")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("score")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Decoders: precision / recall / F1 (higher is better)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig1_decoders.png"), dpi=130)
    plt.close(fig)

    # ---------- 图 2：一个被条件式抑制纠正的误检案例 ----------
    c = caches[HOLDOUT]
    ev_c = decode_v2.decode(c["onset"], c["frame"], c["velocity"], cfg_c)
    ev_d = decode_v2.decode(c["onset"], c["frame"], c["velocity"], cfg_d)
    sup_full = decode_v2.conditional_suppress(c["onset"], c["frame"], cfg_d)
    iv_c, p_c = arrays(ev_c)
    iv_d, p_d = arrays(ev_d)
    ref_iv, ref_p = gts[HOLDOUT]
    # 找：C 有、D 没有、GT 也没有的音（真误检被去掉）
    case = None
    d_set = {(int(p), round(t, 2)) for t, p in zip(iv_d[:, 0], p_d)}
    for t, p in zip(iv_c[:, 0], p_c):
        if (int(p), round(t, 2)) in d_set:
            continue
        # GT 在 ±60ms 内没有同音高
        close = (np.abs(ref_iv[:, 0] - t) < 0.06) & (ref_p == p)
        if close.any():
            continue
        # 找一个附近的锚点（低 12/19/24… 半音）
        for offset, _decay in decode_v2.HARMONIC_OFFSETS:
            anchor_pitch = int(p) - offset
            cand = np.where((p_c == anchor_pitch) & (np.abs(iv_c[:, 0] - t) < 0.12))[0]
            if len(cand):
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
        axes[0].plot(tt, c["onset"][i0:i1, k1 - 21], label=f"onset: anchor key MIDI {k1}",
                     color="#2196f3")
        axes[0].plot(tt, c["onset"][i0:i1, k2 - 21], label=f"onset: harmonic key MIDI {k2}",
                     color="#e91e63")
        axes[0].plot(tt, sup_full[i0:i1, k2 - 21],
                     label="onset: harmonic key after conditional suppression", color="#e91e63",
                     linestyle="--")
        axes[0].axhline(cfg_c.onset_threshold, color="gray", linestyle=":", label="threshold")
        axes[0].axvline(t0, color="black", linestyle="--", linewidth=0.8)
        axes[0].set_ylabel("onset probability")
        axes[0].legend(fontsize=7, loc="upper right")
        axes[0].set_title(f"Case study: false positive at MIDI {k2} removed by harmonic suppression "
                          f"(anchor {k1}, +{offset} semitones)")
        axes[1].plot(tt, c["frame"][i0:i1, k1 - 21], color="#2196f3", label=f"frame: MIDI {k1}")
        axes[1].plot(tt, c["frame"][i0:i1, k2 - 21], color="#e91e63", label=f"frame: MIDI {k2}")
        axes[1].axhline(cfg_c.frame_off_threshold, color="gray", linestyle=":", label="frame off threshold")
        axes[1].set_ylabel("frame probability")
        axes[1].set_xlabel("time (s)")
        axes[1].legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS, "fig2_case_study.png"), dpi=130)
        plt.close(fig)
        print(f"\n案例图: 误检 MIDI {k2} @ {t0:.2f}s，锚点 MIDI {k1}（+{offset} 半音）")

    # ---------- 图 3：抑制的作用范围（局部热图） ----------
    sup = decode_v2.conditional_suppress(c["onset"], c["frame"], cfg_d)
    ratio = np.where(c["onset"] > 1e-6, sup / np.maximum(c["onset"], 1e-6), 1.0)
    # 选抑制最多的一段 8 秒
    win = int(8 * FPS)
    best_start, best_hits = 0, -1
    for s in range(0, len(ratio) - win, win // 2):
        hits = int((ratio[s:s + win] < 0.999).sum())
        if hits > best_hits:
            best_hits, best_start = hits, s
    seg = ratio[best_start:best_start + win, :].T  # (88, t)
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(seg, aspect="auto", origin="lower", cmap="viridis", vmin=0.5, vmax=1.0,
                   extent=[best_start / FPS, (best_start + win) / FPS, 21, 108])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("MIDI pitch")
    ax.set_title("Conditional suppression: ratio = after / before (lower = more suppressed); "
                 "suppression is local in time, not global")
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
    print(f"\nnezha2（{final_label}）: events={len(ev)} 音符头={m['n_noteheads']} "
          f"连音={m['tie_count']} 休止={m['rest_count']} 黑音符={m['black_ratio']:.3f} "
          f"小节={m['measures_per_part']}")
    render_preview.render_png(xml, os.path.join(OUT, "nezha2_step3_p1.png"), zoom=1.4)

    # 存配置
    with open(os.path.join(OUT, "step3_best_config.json"), "w", encoding="utf-8") as fh:
        json.dump({"final_decoder": final_label, "config": {
            "method": cfg_final.method, "harmonic_mode": cfg_final.harmonic_mode,
            "harmonic_strength": cfg_final.harmonic_strength,
            "harmonic_window_frames": cfg_final.harmonic_window_frames,
            "harmonic_frame_max": cfg_final.harmonic_frame_max,
            "frame_gate": cfg_final.frame_gate,
            "onset_threshold": cfg_final.onset_threshold,
            "onset_prominence": cfg_final.onset_prominence,
            "peak_distance": cfg_final.peak_distance,
            "frame_off_threshold": cfg_final.frame_off_threshold,
            "release_frames": cfg_final.release_frames,
        }}, fh, ensure_ascii=False, indent=2)
    print("写入:", os.path.join(OUT, "step3_decoders.csv"))


if __name__ == "__main__":
    main()
