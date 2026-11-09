"""为人工校核包生成「不受音色/音量干扰」的对比材料。

每首曲目：
    *_norm.wav              原始 / GT / 模型回放统一响度后的音频
    comparison_pianoroll.png  GT 与模型的钢琴卷帘对比（同时间轴）
    compare.mid              两个轨道（GT / 模型），可放进钢琴音源里分别试听
    score.pdf               重新生成的 PDF（白底）

用法（容器内）：
    python scripts/make_comparison_assets.py
"""

from __future__ import annotations

import json
import os

import librosa
import numpy as np
import pretty_midi
import soundfile as sf

from piano2 import plot_style

plot_style.use_cjk_font()
import matplotlib.pyplot as plt  # noqa: E402

C = plot_style
ROOT = "/Docker/piano_validation"
TARGET_RMS = 10 ** (-20 / 20.0)   # -20 dBFS


def normalize(src, dst):
    """读取音频并统一响度，写回 16-bit WAV。"""
    y, sr = librosa.load(src, sr=22050)
    rms = float(np.sqrt(np.mean(y ** 2))) or 1e-6
    y = y * (TARGET_RMS / rms)
    peak = float(np.max(np.abs(y))) or 1.0
    if peak > 0.95:
        y = y / peak * 0.95
    sf.write(dst, y.astype(np.float32), sr, subtype="PCM_16")


def combine_midi(gt_path, model_path, out_path):
    """把 GT 与模型写成两个轨道，方便在钢琴音源里 mute/solo 对比。"""
    out = pretty_midi.PrettyMIDI()
    for name, path in (("GT", gt_path), ("model", model_path)):
        pm = pretty_midi.PrettyMIDI(path)
        inst = pretty_midi.Instrument(program=0, name=name)
        for src_inst in pm.instruments:
            for n in src_inst.notes:
                inst.notes.append(pretty_midi.Note(velocity=n.velocity, pitch=n.pitch,
                                                   start=n.start, end=n.end))
        inst.notes.sort(key=lambda n: n.start)
        out.instruments.append(inst)
    out.write(out_path)


def _draw_roll(ax, midi_path, color, duration, alpha=1.0, linewidth=2.2):
    """在给定坐标轴上画一个 MIDI 的钢琴卷帘。"""
    pm = pretty_midi.PrettyMIDI(midi_path)
    for inst in pm.instruments:
        for n in inst.notes:
            if n.start > duration:
                continue
            ax.plot([n.start, min(n.end, duration)], [n.pitch, n.pitch],
                    color=color, linewidth=linewidth, alpha=alpha,
                    solid_capstyle="butt")
    ax.set_xlim(0, duration)
    ax.set_ylabel("MIDI 音高", fontsize=9)
    ax.grid(alpha=0.2, linestyle=":")


def pianoroll_figure(gt_mid, model_mid, title, out_png, duration=20.0):
    """画三格对比：GT、模型、以及叠在一起（透明）。"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.2), sharex=True, sharey=True)
    _draw_roll(axes[0], gt_mid, C.COLOR_DARK, duration)
    axes[0].set_title("GT", fontsize=9.5, loc="left")
    _draw_roll(axes[1], model_mid, C.COLOR_ORANGE, duration)
    axes[1].set_title("模型输出", fontsize=9.5, loc="left")
    _draw_roll(axes[2], gt_mid, C.COLOR_DARK, duration, alpha=0.55, linewidth=2.6)
    _draw_roll(axes[2], model_mid, C.COLOR_ORANGE, duration, alpha=0.55, linewidth=2.6)
    axes[2].set_title("叠加（蓝=GT，橙=模型；重合处呈深色）", fontsize=9.5, loc="left")
    axes[2].set_xlabel("时间（秒）", fontsize=9)
    fig.suptitle(title + "：GT / 模型 / 叠加", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, facecolor="white")
    plt.close(fig)


def main():
    index = json.load(open(os.path.join(ROOT, "index.json"), encoding="utf-8"))
    for i, info in enumerate(index, 1):
        d = [x for x in os.listdir(ROOT) if x.startswith(f"{i:02d}_")][0]
        p = os.path.join(ROOT, d)
        for src, dst in (("original.wav", "original_norm.wav"),
                         ("gt_playback.wav", "gt_playback_norm.wav"),
                         ("model_playback.wav", "model_playback_norm.wav")):
            if os.path.exists(os.path.join(p, src)):
                normalize(os.path.join(p, src), os.path.join(p, dst))
        if os.path.exists(os.path.join(p, "gt.mid")) and os.path.exists(os.path.join(p, "model.mid")):
            try:
                combine_midi(os.path.join(p, "gt.mid"), os.path.join(p, "model.mid"),
                             os.path.join(p, "compare.mid"))
            except Exception as exc:
                print("combine midi failed:", exc)
        pianoroll_figure(os.path.join(p, "gt.mid"), os.path.join(p, "model.mid"),
                         f"{i:02d} {info['title']}",
                         os.path.join(p, "comparison_pianoroll.png"))
        print("处理完成:", d)


if __name__ == "__main__":
    main()
