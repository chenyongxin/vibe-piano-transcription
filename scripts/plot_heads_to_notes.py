"""把 8 个输出头和解码流程画成一张图。

输出：docs/figs/47_heads_to_notes.png
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, "/Docker/git/piano2")
sys.path.insert(0, "/Docker/git/piano2/src")

from piano2 import plot_style  # noqa: E402
from piano2.constants import DECODE_DEFAULTS  # noqa: E402
from piano2.decode import DecodeConfig, decode  # noqa: E402
from piano2.inference import compute_cqt, infer_sliding_window, load_model  # noqa: E402
from piano2.notation import events_from_midi  # noqa: E402

plot_style.use_cjk_font()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

C = plot_style
FPS = 31.25
ROOT = "/Docker/piano2_data/validation/piano_validation/01_Sonata_in_C_major,_D._84"


def main():
    model = load_model("/Docker/git/piano2/outputs/M1/hybrid_reg2/checkpoints/best.pt", "auto")
    wav = os.path.join(ROOT, "original.wav")
    gt = events_from_midi(os.path.join(ROOT, "gt.mid"))
    probs = infer_sliding_window(model, compute_cqt(wav), "auto",
                                 with_offset=True, with_pedal=True)
    onset, frame, vel, offset, pedal, pedal_on = probs
    events = decode(onset, frame, vel,
                    DecodeConfig(**DECODE_DEFAULTS, use_offset_end=True),
                    offset_probs=offset)

    # 选一个孤立、时值适中的音，便于看清各头的形状
    def isolated(n):
        pp, ss, ee, vv = n
        return (0.10 <= ee - ss <= 0.45 and
                all(not (p2 == pp and abs(s2 - ss) < 0.45 and s2 != ss)
                    for p2, s2, e2, v2 in gt))

    candidates = [n for n in gt if isolated(n) and n[1] > 0.5]
    p, s, e, v = candidates[0] if candidates else gt[0]
    key = p - 21
    t0 = max(0.0, s - 0.25)
    t1 = min(len(onset) / FPS, e + 0.25)
    frames = np.arange(int(t0 * FPS), int(t1 * FPS) + 1)
    times = frames / FPS

    fig = plt.figure(figsize=(13, 9.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.1, 1.1], hspace=0.35)
    ax = fig.add_subplot(gs[0])

    curves = [
        (onset[frames, key], "onset（起音概率）", C.COLOR_BLUE),
        (frame[frames, key], "frame（按住状态）", C.COLOR_GREEN),
        (offset[frames, key], "offset（释放概率）", C.COLOR_RED),
        (vel[frames, key], "velocity（力度 0–1）", C.COLOR_ORANGE),
    ]
    for y, label, color in curves:
        ax.plot(times, y, linewidth=2, color=color, label=label)
    ax.plot(times, pedal[frames, 0], linewidth=2, color="#8e44ad",
            linestyle="--", label="pedal（踏板状态，T×1）")
    ax.axvline(s, color=C.COLOR_DARK, linestyle=":", linewidth=1.2)
    ax.axvline(e, color=C.COLOR_DARK, linestyle=":", linewidth=1.2)
    ax.set_xlim(t0, t1)
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlabel("时间（秒）", fontsize=9)
    ax.set_ylabel("头输出（sigmoid 后）", fontsize=9)
    ax.set_title(
        f"同一个音（MIDI {p}，{s:.2f}–{e:.2f}s）：各头输出一条帧级曲线"
        f"（onset/frame/velocity/offset 为 T×88，pedal 为 T×1；这里取第 {key} 键）",
        fontsize=10)
    ax.legend(fontsize=8, ncol=3, frameon=True, loc="upper right")
    ax.grid(alpha=0.2, linestyle=":")

    # 解码流程图
    ax = fig.add_subplot(gs[1])
    ax.axis("off")
    steps = [
        ("onset 概率\n(T,88)", "按每个键找峰\n得到候选起点", C.COLOR_BLUE),
        ("velocity\n(T,88)", "在起点帧取值\n×127", C.COLOR_ORANGE),
        ("frame/offset\n(T,88)", "释放阈值/offset 峰\n得到结束帧", C.COLOR_GREEN),
        ("音符事件\npitch/start/end/vel", "→ MusicXML/PDF/MIDI\n→ 回放", C.COLOR_DARK),
    ]
    xs = [0.01, 0.26, 0.51, 0.76]
    for x, (left, right, color) in zip(xs, steps):
        ax.add_patch(FancyBboxPatch((x, 0.42), 0.19, 0.42, transform=ax.transAxes,
                                    boxstyle="round,pad=0.02", facecolor="white",
                                    edgecolor=color, linewidth=1.8))
        ax.text(x + 0.095, 0.72, left, transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color=color)
        ax.text(x + 0.095, 0.53, right, transform=ax.transAxes, ha="center", va="center",
                fontsize=8, color=C.COLOR_GRAY)
    for x in (0.21, 0.46, 0.71):
        ax.annotate("", xy=(x + 0.045, 0.63), xytext=(x, 0.63),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops={"arrowstyle": "->", "color": C.COLOR_DARK, "linewidth": 1.4})
    ax.text(0.5, 0.93, "解码：把多个头翻译成 NoteEvent，再交给记谱/渲染",
            transform=ax.transAxes, ha="center", fontsize=10)
    ax.text(0.5, 0.13, "pedal（T×1）单独解码成 CC64，用于回放和谱面踏板线",
            transform=ax.transAxes, ha="center", fontsize=9, color="#8e44ad")
    ax.text(0.5, 0.03, "注意：模型不是只输出一个 (T,88)，而是同时输出 8 个头。",
            transform=ax.transAxes, ha="center", fontsize=9, color=C.COLOR_DARK)

    fig.suptitle("模型输出头 → 音符事件 → 乐谱/回放", fontsize=12)
    fig.savefig("/Docker/git/piano2/docs/figs/47_heads_to_notes.png", dpi=150,
                facecolor="white")
    plt.close(fig)
    print("note", p, s, e, "n_events", len(events))


if __name__ == "__main__":
    main()
