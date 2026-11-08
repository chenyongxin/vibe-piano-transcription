"""可视化 offset 标签的 1/2/3 帧窗口，以及对应的实测结果。

左：同一段 note-off 在三种标签窗口下的位置（帧网格、窗口中心、偏差）
右：正样本率（数据）与留出集 note F1（训练结果）对比

输出：docs/figs/31_offset_label_windows.png
"""

from __future__ import annotations

import numpy as np

from piano2 import plot_style

plot_style.use_cjk_font()
import matplotlib.pyplot as plt  # noqa: E402

C = plot_style
FPS = 31.25
HOP_MS = 32.0


def panel_windows(ax):
    """画三种 offset 窗口：1 帧、2 帧、3 帧居中。"""
    end_ms = 96.0          # note-off 正好在帧中心
    ax.set_xlim(-10, 210)
    ax.set_ylim(0, 3.1)
    # 帧网格
    for k in range(0, 7):
        center = k * HOP_MS
        ax.axvline(center, color="#bbbbbb", linewidth=0.8, linestyle="--", zorder=0)
        ax.text(center, 2.9, f"{center:.0f}", ha="center", fontsize=7.5,
                color=C.COLOR_GRAY)
    # 音符条与 note-off
    ax.add_patch(plt.Rectangle((0, 2.62), end_ms, 0.09,
                               facecolor=C.COLOR_DARK, alpha=0.85))
    ax.plot([end_ms, end_ms], [2.55, 2.75], color=C.COLOR_RED, linewidth=1.5)
    ax.text(end_ms + 2, 2.78, "note-off = 96 ms", fontsize=8, color=C.COLOR_RED)

    rows = [
        ("offset 1 帧", 2.0, [3], "中心 0 ms", C.COLOR_GREEN),
        ("offset 2 帧", 1.1, [3, 4], "中心 +16 ms", C.COLOR_ORANGE),
        ("offset 3 帧居中", 0.2, [2, 3, 4], "中心 0 ms", C.COLOR_BLUE),
    ]
    for label, y, frames, center_note, color in rows:
        ax.text(-14, y + 0.15, label, ha="right", va="center", fontsize=9, color=color)
        for f in frames:
            ax.add_patch(plt.Rectangle((f * HOP_MS - HOP_MS / 2, y - 0.13),
                                       HOP_MS, 0.26, facecolor=color, alpha=0.35,
                                       edgecolor=color, linewidth=1.0))
        center = np.mean(frames) * HOP_MS
        ax.plot(center, y, marker="v", markersize=8, color=color)
        ax.text(148, y, center_note, ha="left", va="center", fontsize=8, color=color)

    ax.set_yticks([])
    ax.set_xlabel("时间（ms），帧中心间隔 32 ms", fontsize=9)
    ax.set_title("① 同一个 note-off（96 ms）在三种 offset 标签窗口下的位置", fontsize=10)


def panel_rates(ax):
    """正样本率。"""
    labels = ["1 帧", "2 帧", "3 帧"]
    rates = [0.0034, 0.0069, 0.0103]
    colors = [C.COLOR_GREEN, C.COLOR_ORANGE, C.COLOR_BLUE]
    ax.bar(labels, rates, color=colors, edgecolor=C.COLOR_DARK, linewidth=0.5)
    for i, v in enumerate(rates):
        ax.text(i, v + 0.0002, f"{v:.4f}", ha="center", fontsize=8.5)
    ax.set_ylim(0, 0.0125)
    ax.set_ylabel("正样本率", fontsize=9)
    ax.set_title("② offset 通道的正样本率（数据 v3c / v3b / v3d 实测）", fontsize=10)
    ax.grid(axis="y", alpha=0.25, linestyle=":")


def panel_results(ax):
    """留出集 note F1。"""
    labels = ["1 帧（v3c）", "2 帧（v3b）", "3 帧（v3d）"]
    scores = [0.251, 0.372, 0.285]
    colors = [C.COLOR_GREEN, C.COLOR_ORANGE, C.COLOR_BLUE]
    ax.bar(labels, scores, color=colors, edgecolor=C.COLOR_DARK, linewidth=0.5)
    for i, v in enumerate(scores):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8.5)
    ax.set_ylim(0, 0.45)
    ax.set_ylabel("留出集 note F1", fontsize=9)
    ax.set_title("③ 对应的训练结果（20 轮，offset 损失 1.0）", fontsize=10)
    ax.grid(axis="y", alpha=0.25, linestyle=":")


def main():
    fig = plt.figure(figsize=(13, 5.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.65, 1])
    panel_windows(fig.add_subplot(gs[:, 0]))
    panel_rates(fig.add_subplot(gs[0, 1]))
    panel_results(fig.add_subplot(gs[1, 1]))
    fig.suptitle("offset 标签窗口：1 帧太稀、2 帧偏晚但够稳、3 帧居中但太糊", fontsize=12)
    fig.tight_layout()

    from fig_utils import check_clipping

    problems = check_clipping(fig)
    if problems:
        raise SystemExit("图形越界：\n" + "\n".join(problems))

    out = "/Docker/git/piano2/docs/figs/31_offset_label_windows.png"
    plot_style.save(fig, out)
    plt.close(fig)
    print("写入:", out)


if __name__ == "__main__":
    main()
