"""解释 MIDI→帧对齐与回归/有限元插值的关系。

输出：
    docs/figs/45_midi_frame_alignment.png
    docs/figs/46_fem_vs_regression.png
"""

from __future__ import annotations

import numpy as np

from piano2 import plot_style

plot_style.use_cjk_font()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

C = plot_style
FPS = 31.25
FRAME_MS = 32.0
EVENT_MS = 11.0
J = 5


def fig_alignment():
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.25, 0.9, 0.55], hspace=0.45)

    # A. timeline
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(-8, 110)
    ax.set_ylim(-0.2, 3.4)
    ax.axis("off")

    # CQT frames
    for i in range(4):
        x = i * FRAME_MS
        ax.axvline(x, ymin=0.35, ymax=0.95, color=C.COLOR_BLUE, alpha=0.5, linewidth=1.5)
        ax.add_patch(FancyBboxPatch((x - 11, 2.55), 22, 0.5, boxstyle="round,pad=0.05",
                                    facecolor="white", edgecolor=C.COLOR_BLUE))
        ax.text(x, 2.80, f"T={i}", ha="center", va="center", fontsize=10, color=C.COLOR_BLUE)
    ax.text(42, 3.25, "CQT / 模型：32 ms 一帧", fontsize=10, color=C.COLOR_BLUE)

    # MIDI ticks
    for t in np.arange(0, 101, 4):
        ax.plot([t, t], [0.55, 0.80], color="#b0b0b0", linewidth=0.6)
    ax.text(42, 0.95, "MIDI：约 1 ms 一个 tick（只用于训练监督）", fontsize=9,
            color=C.COLOR_GRAY)
    ax.plot([EVENT_MS, EVENT_MS], [0.45, 2.5], color=C.COLOR_RED, linestyle="--", linewidth=1.5)
    ax.text(EVENT_MS + 2, 1.25, "真实 onset = 11 ms\nT = 11/32 = 0.344 帧",
            fontsize=9, color=C.COLOR_RED)

    ax.text(0, 1.85,
            "分类标签：最近帧 T=0 标 1\n→ 只能预测 0 或 32 ms",
            fontsize=9, color=C.COLOR_DARK)

    # B. labels
    ax = fig.add_subplot(gs[1])
    frames = np.arange(-4, 5)
    centers = frames * FRAME_MS
    exact = np.array([max(0.0, 1.0 - abs(c - EVENT_MS) / (J * FRAME_MS)) for c in centers])
    ax.axvline(EVENT_MS, color=C.COLOR_RED, linestyle="--", linewidth=1.2, label="真实事件")
    ax.plot(centers, exact, color=C.COLOR_ORANGE, marker="o", linewidth=2,
            label="onset_reg 精确三角目标")
    ax.stem([0], [1.0], basefmt=" ", linefmt=":", markerfmt="o")
    ax.scatter([0], [1.0], color=C.COLOR_BLUE, s=80, zorder=5, label="onset1 分类标签 = 1")
    ax.set_xlim(-4.5 * FRAME_MS, 4.5 * FRAME_MS)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel("帧中心时刻（ms）", fontsize=9)
    ax.set_ylabel("目标值", fontsize=9)
    ax.set_title("同样一个 11 ms 事件：分类只标一帧；回归把亚帧位置编码进数值", fontsize=10)
    ax.grid(alpha=0.2, linestyle=":")
    ax.legend(fontsize=8, frameon=False, loc="upper right")

    # C. data flow
    ax = fig.add_subplot(gs[2])
    ax.axis("off")
    boxes = [
        (0.02, "CQT\n(T, 264)\n32 ms/帧", C.COLOR_BLUE),
        (0.27, "对齐到帧\nT = t × FPS", C.COLOR_GRAY),
        (0.50, "标签\n(T, 9, 88)\n7 分类 + 2 回归", C.COLOR_ORANGE),
        (0.75, "模型输出\n(T, 88)\n每帧每键", C.COLOR_GREEN),
    ]
    for x, text, color in boxes:
        ax.add_patch(FancyBboxPatch((x, 0.18), 0.20, 0.55, transform=ax.transAxes,
                                    boxstyle="round,pad=0.02", facecolor="white",
                                    edgecolor=color, linewidth=1.8))
        ax.text(x + 0.10, 0.46, text, transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color=color)
    for x in (0.23, 0.48, 0.73):
        ax.annotate("", xy=(x + 0.03, 0.46), xytext=(x, 0.46),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops={"arrowstyle": "->", "color": C.COLOR_DARK})
    ax.text(0.5, 0.95, "MIDI 的 1 ms 精度没有变成更多输出帧，而是变成标签数值里的相位",
            transform=ax.transAxes, ha="center", fontsize=10, color=C.COLOR_DARK)
    ax.text(0.5, 0.05, "推理时没有 MIDI；模型只能从 CQT 预测这些数值",
            transform=ax.transAxes, ha="center", fontsize=9, color=C.COLOR_GRAY)
    fig.suptitle("MIDI（约 1 ms）如何进入 32 ms 帧对齐的数据", fontsize=12)
    fig.savefig("/Docker/git/piano2/docs/figs/45_midi_frame_alignment.png", dpi=150,
                facecolor="white")
    plt.close(fig)


def fig_fem():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=False)

    # Left: linear shape functions
    ax = axes[0]
    x = np.linspace(0, 1, 100)
    q1 = 1 - x
    q2 = x
    ax.plot(x, q1, color=C.COLOR_BLUE, linewidth=2, label="q1 = 1 - x")
    ax.plot(x, q2, color=C.COLOR_ORANGE, linewidth=2, label="q2 = x")
    ax.scatter([0, 1], [1, 0], color=C.COLOR_BLUE, s=60)
    ax.scatter([0, 1], [0, 1], color=C.COLOR_ORANGE, s=60)
    ax.axvline(0.344, color=C.COLOR_RED, linestyle="--", linewidth=1.2)
    ax.text(0.36, 0.52, "x=0.344\nq1=0.656, q2=0.344", fontsize=9, color=C.COLOR_RED)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel("区间坐标 x", fontsize=9)
    ax.set_ylabel("形函数值", fontsize=9)
    ax.set_title("有限元：在节点间插值场值", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.2, linestyle=":")

    # Right: triangle target / vertex
    ax = axes[1]
    frames = np.arange(-4, 5)
    centers = frames * 32.0
    exact = np.array([max(0.0, 1.0 - abs(c - EVENT_MS) / (5 * 32)) for c in centers])
    ax.plot(centers, exact, color=C.COLOR_ORANGE, marker="o", linewidth=2,
            label="onset_reg 目标 f(T)")
    ax.axvline(EVENT_MS, color=C.COLOR_RED, linestyle="--", linewidth=1.2, label="要恢复的顶点")
    ax.annotate("插值找顶点", xy=(EVENT_MS, 1.0), xytext=(35, 0.75),
                fontsize=9, color=C.COLOR_RED,
                arrowprops={"arrowstyle": "->", "color": C.COLOR_RED})
    ax.set_xlim(-4.2 * 32, 4.2 * 32)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel("帧中心时刻（ms）", fontsize=9)
    ax.set_ylabel("模型输出 / 目标值", fontsize=9)
    ax.set_title("我们的回归：用三角目标的相邻取值反推顶点", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.2, linestyle=":")

    fig.suptitle("有限元插值 vs 回归亚帧定位：数学形似，语义不同", fontsize=12)
    fig.tight_layout()
    fig.savefig("/Docker/git/piano2/docs/figs/46_fem_vs_regression.png", dpi=150,
                facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    fig_alignment()
    fig_fem()
    print("done")
