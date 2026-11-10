"""M0.5 步骤 3 收尾：归因实验（抑制 vs 门控）、修正案例图、重出 nezha2 谱。

为什么做归因：步骤 3b 的最终配置里同时有「条件式抑制」和「帧一致性门控」，
需要量化各自的贡献，避免把功劳记错。

用法（容器内）：
    cd /Docker/git/piano_practice/m0 && \
    PYTHONPATH=/Docker/git/piano_practice/m0:/Docker/git/piano_practice \
    python m05_step3_finalize.py
"""

from __future__ import annotations

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
FPS = 16000 / 512
DEV = ("haydn", "mozart", "beethoven", "holdout2")


def arrays(events):
    if not events:
        return np.zeros((0, 2)), np.zeros(0, int)
    return (np.array([[e.start, e.end] for e in events], float),
            np.array([e.pitch for e in events], int))


def metrics(ref_iv, ref_p, est_iv, est_p):
    if len(est_iv) == 0:
        return dict(P=0.0, R=0.0, F1=0.0)
    p, r, f, _ = mtran.precision_recall_f1_overlap(
        ref_iv, ref_p, est_iv, est_p, onset_tolerance=0.05,
        pitch_tolerance=50.0, offset_ratio=None)
    return dict(P=float(p), R=float(r), F1=float(f))


def main():
    final = json.load(open(os.path.join(OUT, "step3b_final.json"), encoding="utf-8"))
    cfg_final = decode_v2.DecodeConfig(**{k: v for k, v in final["final_config"].items()})
    cfg_c = decode_v2.DecodeConfig(harmonic_mode="off", method="peak_frame",
                                   onset_threshold=0.60, onset_prominence=0.005,
                                   peak_distance=5, frame_off_threshold=0.18,
                                   release_frames=3)
    holdout2_midi = os.path.join("/Docker/piano2_data/raw/maestro-v3.0.0", final["holdout2"]["midi"])
    gts = {}
    for name in DEV:
        ev = notation_v1.events_from_midi(
            prob_cache.GT_MIDI.get(name, holdout2_midi))
        gts[name] = (np.array([[s, e] for (_p, s, e, _v) in ev], float),
                     np.array([p for (p, _s, _e, _v) in ev], int))
    caches = {n: prob_cache.load_cache(n) for n in DEV}

    # 消融：无 / 只门控 / 只抑制 / 两者都有
    variants = {
        "none": cfg_c,
        "gate_only": cfg_c.with_(frame_gate=cfg_final.frame_gate),
        "suppress_only": cfg_c.with_(harmonic_mode="conditional",
                                     harmonic_strength=cfg_final.harmonic_strength,
                                     harmonic_window_frames=cfg_final.harmonic_window_frames,
                                     harmonic_frame_max=cfg_final.harmonic_frame_max),
        "both": cfg_final,
    }
    print("=== 归因（precision / F1） ===")
    print(f"{'variant':14s} " + "  ".join(f"{n[:9]:>16s}" for n in DEV))
    for label, cfg in variants.items():
        line = f"{label:14s} "
        for name in DEV:
            c = caches[name]
            ev = decode_v2.decode(c["onset"], c["frame"], c["velocity"], cfg)
            m = metrics(*gts[name], *arrays(ev))
            line += f"  P={m['P']:.3f} F1={m['F1']:.3f}"
        print(line)

    # 修正案例图：明确标注是哪一种机制去掉了误检
    c = caches["beethoven"]
    sup = decode_v2.conditional_suppress(c["onset"], c["frame"], cfg_final)
    ev_c = decode_v2.decode(c["onset"], c["frame"], c["velocity"], cfg_c)
    ev_f = decode_v2.decode(c["onset"], c["frame"], c["velocity"], cfg_final)
    iv_c, p_c = arrays(ev_c)
    iv_f, p_f = arrays(ev_f)
    f_set = {(int(p), round(t, 2)) for t, p in zip(iv_f[:, 0], p_f)}
    ref_iv, ref_p = gts["beethoven"]
    case = None
    for t, p in zip(iv_c[:, 0], p_c):
        if (int(p), round(t, 2)) in f_set:
            continue
        if np.any((np.abs(ref_iv[:, 0] - t) < 0.06) & (ref_p == p)):
            continue
        for offset, _d in decode_v2.HARMONIC_OFFSETS:
            anchor = int(p) - offset
            if np.any((p_c == anchor) & (np.abs(iv_c[:, 0] - t) < 0.12)):
                case = (t, int(p), anchor, offset)
                break
        if case:
            break
    if case:
        t0, k2, k1, off = case
        half = 0.6
        i0 = max(0, int((t0 - half) * FPS))
        i1 = min(len(c["onset"]), int((t0 + half) * FPS))
        tt = np.arange(i0, i1) / FPS
        fig, axes = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True)
        ax = axes[0]
        ax.plot(tt, c["onset"][i0:i1, k1 - 21], color="#1976d2", label=f"onset: anchor MIDI {k1}")
        ax.plot(tt, c["onset"][i0:i1, k2 - 21], color="#d81b60", label=f"onset: MIDI {k2} (before)")
        ax.plot(tt, sup[i0:i1, k2 - 21], color="#d81b60", linestyle="--",
                label=f"onset: MIDI {k2} (after suppression)")
        ax.axhline(cfg_c.onset_threshold, color="gray", linestyle=":", label="onset threshold")
        ax.axvline(t0, color="black", linestyle="--", linewidth=0.8)
        ax.set_ylabel("onset probability")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_title(f"False positive at MIDI {k2} ({t0:.2f}s): anchor MIDI {k1} (+{off} semitones)")
        ax = axes[1]
        ax.plot(tt, c["frame"][i0:i1, k1 - 21], color="#1976d2", label=f"frame: anchor MIDI {k1}")
        ax.plot(tt, c["frame"][i0:i1, k2 - 21], color="#d81b60", label=f"frame: MIDI {k2}")
        ax.axhline(cfg_final.frame_gate, color="gray", linestyle=":", label="frame gate (0.15)")
        ax.set_ylabel("frame probability")
        ax.set_xlabel("time (s)")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_title(f"Frame evidence at MIDI {k2} is ~{float(c['frame'][i0:i1, k2-21].mean()):.3f}, "
                     f"below the gate -> note dropped", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS, "fig2_case_study.png"), dpi=130)
        plt.close(fig)
        print(f"\n案例图已更新: MIDI {k2} @ {t0:.2f}s，锚点 MIDI {k1}（+{off} 半音）")

    # nezha2：用修好的记谱层（总时长按音频兜底）重出谱
    cfg_use = cfg_final
    c = prob_cache.load_cache("nezha2")
    ev = decode_v2.decode(c["onset"], c["frame"], c["velocity"], cfg_use)
    wav = os.path.join(ROOT, "data", "nezha2.wav")
    xml = os.path.join(OUT, "nezha2_step3.musicxml")
    notation_v1.write_musicxml([(e.pitch, e.start, e.end, e.velocity) for e in ev], wav, xml,
                               title="nezha2 step3 final")
    m = score_metrics.score_metrics(xml, audio_duration_s=c["duration_s"])
    print(f"\nnezha2 final: events={len(ev)} 音符头={m['n_noteheads']} 连音={m['tie_count']} "
          f"休止={m['rest_count']} 黑音符={m['black_ratio']:.3f} 小节={m['measures_per_part']} "
          f"时长误差={m['duration_err_ratio']:.4f}")
    render_preview.render_png(xml, os.path.join(OUT, "nezha2_step3_p1.png"), zoom=1.4)


if __name__ == "__main__":
    main()
