"""M0.5 步骤 1：只改一件事——关掉谐波抑制，其余参数不动。

对照组（三行）：
    1. 旧产物：inference/readable_prediction.musicxml（旧解码 + 旧记谱）
    2. 旧解码 + 新记谱：harmonic=global（复刻原实现）+ notation_v1
    3. 关抑制 + 新记谱：harmonic=off（步骤 1）+ notation_v1

用法（容器内）：
    cd /Docker/git/piano_practice/m0 && \
    PYTHONPATH=/Docker/git/piano_practice/m0:/Docker/git/piano_practice \
    python m05_step1.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = "/Docker/git/piano_practice"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "m0"))

import decode_v2  # noqa: E402
import notation_v1  # noqa: E402
import prob_cache  # noqa: E402
import render_preview  # noqa: E402
import score_metrics  # noqa: E402

OUT = os.path.join(ROOT, "outputs", "M0", "m05")
WAV = os.path.join(ROOT, "data", "nezha2.wav")
OLD_PRODUCT = os.path.join(ROOT, "inference", "readable_prediction.musicxml")

KEYS = [
    "n_noteheads", "n_onsets", "tie_count", "voices_per_measure_mean",
    "rest_count", "black_ratio", "measures_per_part", "measures_consistent",
    "score_duration_s", "duration_err_ratio",
]


def show(name, events_len, metrics):
    """打印一行对比结果。"""
    parts = []
    for k in KEYS:
        v = metrics.get(k)
        if v is None:
            s = "None"
        elif isinstance(v, float):
            s = f"{v:.3f}"
        else:
            s = str(v)
        parts.append(f"{k}={s}")
    print(f"\n[{name}] 模型事件数={events_len}")
    print("   " + "  ".join(parts))


def main():
    os.makedirs(OUT, exist_ok=True)
    cached = prob_cache.load_cache("nezha2")
    onset, frame, vel = cached["onset"], cached["frame"], cached["velocity"]
    duration = cached["duration_s"]
    print(f"nezha2 缓存: T={onset.shape[0]} 帧, 时长 {duration:.1f}s")
    print(f"onset 概率: 均值={onset.mean():.4f}, P99={float(np.percentile(onset, 99)):.4f}")

    rows = []

    # 1) 旧产物（旧解码 + 旧记谱）
    if os.path.exists(OLD_PRODUCT):
        m = score_metrics.score_metrics(OLD_PRODUCT, audio_duration_s=duration)
        show("旧产物（旧解码+旧记谱）", -1, m)
        rows.append({"case": "old_product", "n_events": -1, **m})

    # 2) 旧解码 + 新记谱：harmonic=global，参数与原管线一致
    cfg_global = decode_v2.DecodeConfig(harmonic_mode="global", harmonic_strength=0.5)
    ev_global = decode_v2.decode(onset, frame, vel, cfg_global)
    print(f"\n旧解码事件数: {len(ev_global)}")
    xml_global = os.path.join(OUT, "nezha2_global_v1.musicxml")
    notation_v1.write_musicxml(
        [(e.pitch, e.start, e.end, e.velocity) for e in ev_global], WAV, xml_global,
        title="nezha2 global-harmonic + notation v1",
    )
    m_global = score_metrics.score_metrics(xml_global, audio_duration_s=duration)
    show("旧解码（谐波抑制开）+ 新记谱", len(ev_global), m_global)
    rows.append({"case": "global_decode_v1_notation", "n_events": len(ev_global), **m_global})
    render_preview.render_png(xml_global, os.path.join(OUT, "nezha2_global_v1_p1.png"), zoom=1.4)

    # 3) 关抑制 + 新记谱（步骤 1）
    cfg_off = cfg_global.with_(harmonic_mode="off")
    ev_off = decode_v2.decode(onset, frame, vel, cfg_off)
    print(f"\n关抑制事件数: {len(ev_off)}")
    xml_off = os.path.join(OUT, "nezha2_noharm_v1.musicxml")
    notation_v1.write_musicxml(
        [(e.pitch, e.start, e.end, e.velocity) for e in ev_off], WAV, xml_off,
        title="nezha2 no-harmonic + notation v1",
    )
    m_off = score_metrics.score_metrics(xml_off, audio_duration_s=duration)
    show("关抑制（步骤 1）+ 新记谱", len(ev_off), m_off)
    rows.append({"case": "no_harmonic_v1_notation", "n_events": len(ev_off), **m_off})
    render_preview.render_png(xml_off, os.path.join(OUT, "nezha2_noharm_v1_p1.png"), zoom=1.4)

    # 汇总表
    print("\n=== 汇总 ===")
    print(f"{'case':30s} {'events':>7s} {'noteheads':>10s} {'ties':>6s} {'voices':>7s} {'rests':>6s} {'black':>7s} {'measures':>12s}")
    for r in rows:
        print(f"{r['case']:30s} {r['n_events']:>7d} {r['n_noteheads']:>10d} {r['tie_count']:>6d} "
              f"{r['voices_per_measure_mean']:>7.2f} {r['rest_count']:>6d} {r['black_ratio']:>7.3f} "
              f"{str(r['measures_per_part']):>12s}")

    # 写 CSV
    import csv
    path = os.path.join(OUT, "step1_comparison.csv")
    all_keys = []
    for r in rows:
        for k in r:
            if k not in all_keys:
                all_keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print("写入:", path)


if __name__ == "__main__":
    main()
