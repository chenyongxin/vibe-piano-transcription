"""画一张图解释 onset 标签「3 帧 vs 2 帧」的差别。

要讲清三件事：
    1. 时间轴按 32 ms 一帧切开；MIDI note-on 落到某帧上（第 1 帧的起点）；
    2. 旧实现标记 [start-1, start+2)，共 3 帧，其中一帧在起音之前；
       新实现标记 [start, start+2)，共 2 帧；
    3. 实际影响：两个音相隔 64 ms 时，旧标签会在某一帧重叠、粘成连续块；
       新标签首尾相接但不重叠。旧实现能分开的最小间隔是 96 ms，新实现是 64 ms。

输出：doc/figs/01_onset_label.png

用法（容器内）：
    cd /Docker/git/piano_practice/m0 && python make_onset_figure.py
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = "/Docker/git/piano_practice"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "m0"))

import plot_style  # noqa: E402

plot_style.use_cjk_font()

OUT = os.path.join(ROOT, "doc", "figs")
C = plot_style
FRAME_MS = 512 / 16000 * 1000.0  # 32 ms


def cell(ax, frame, y, fill, edge, label="1", hatch=None, height=0.62, width=0.86):
    """在 (frame, y) 处画一个标签格子。"""
    ax.add_patch(Rectangle((frame + 0.07, y), width, height, facecolor=fill,
                           edgecolor=edge, linewidth=1.4, hatch=hatch, zorder=3))
    ax.text(frame + 0.5, y + height / 2, label, ha="center", va="center",
            fontsize=9, color=C.COLOR_DARK, zorder=4)


def frame_axis(ax, x0, x1, y0):
    """画帧网格、底部帧号与时间单位说明。"""
    for f in range(x0, x1 + 1):
        ax.plot([f, f], [y0, 4.35], color="#d5d8dc", linewidth=0.9,
                linestyle="--", zorder=1)
    for f in range(x0, x1):
        ax.text(f + 0.5, y0 - 0.28, f"第 {f} 帧", ha="center", va="top",
                fontsize=8.5, color=C.COLOR_GRAY)
    ax.text((x0 + x1) / 2, y0 - 0.72, "时间轴（1 格 = 1 帧 = 32 ms）",
            ha="center", va="top", fontsize=9.5, color=C.COLOR_GRAY)


def panel_labels(ax):
    """上图：三种 onset 标签在帧轴上的位置。"""
    ax.set_xlim(-0.2, 7.6)
    ax.set_ylim(-1.5, 5.4)
    ax.axis("off")
    frame_axis(ax, 0, 6, 0.0)

    onset_x = 1.0
    ax.annotate("", xy=(onset_x, 4.3), xytext=(onset_x, 3.45),
                arrowprops={"arrowstyle": "-|>", "color": C.COLOR_RED, "linewidth": 2.2})
    ax.text(onset_x + 0.1, 4.36, "真实起音 t0（MIDI note-on，落到第 1 帧起点）",
            fontsize=9.5, color=C.COLOR_RED, ha="left", va="bottom")

    rows = [
        ("旧标签：[start-1, start+2)", [0, 1, 2], C.COLOR_BG_RED, C.COLOR_RED,
         "3 帧 = 96 ms", [0]),
        ("新标签：[start, start+2)", [1, 2], C.COLOR_BG_GREEN, C.COLOR_GREEN,
         "2 帧 = 64 ms", []),
        ("更激进的 1 帧变体：[start, start+1)", [1], C.COLOR_BG_BLUE, C.COLOR_BLUE,
         "1 帧 = 32 ms", []),
    ]
    for row_idx, (name, frames, fill, edge, width_text, pre_roll) in enumerate(rows):
        y = 2.55 - row_idx * 1.0
        ax.text(-0.15, y + 0.31, name, ha="right", va="center", fontsize=9.5,
                color=C.COLOR_DARK)
        for f in frames:
            hatch = "///" if f in pre_roll else None
            cell(ax, f, y, fill, edge, hatch=hatch)
        ax.text(frames[-1] + 1.15, y + 0.31, width_text, ha="left", va="center",
                fontsize=9, color=edge)

    ax.text(-0.15, -1.12,
            "斜线格 = 落在真实起音之前的那一帧（旧实现特有）：模型会学成「提前 32 ms 起音」；"
            "2 帧标签没有这一问题。",
            fontsize=9.5, color=C.COLOR_RED, ha="left", va="top")
    ax.text(3.35, 5.15, "① 三种 onset 标签写法（同一帧轴上对比）",
            fontsize=12, weight="bold", color=C.COLOR_DARK, ha="left")


def panel_merge(ax):
    """下图：两个相隔 64 ms 的重复音，在两种标签下是否重叠。"""
    ax.set_xlim(-0.2, 7.6)
    ax.set_ylim(-1.5, 5.4)
    ax.axis("off")
    frame_axis(ax, 0, 6, 0.0)

    for onset_frame in (1, 3):
        ax.annotate("", xy=(onset_frame, 4.3), xytext=(onset_frame, 3.45),
                    arrowprops={"arrowstyle": "-|>", "color": C.COLOR_RED, "linewidth": 1.8})
    ax.text(1.1, 4.36, "两个重复音，相隔 2 帧 = 64 ms", fontsize=9.5,
            color=C.COLOR_RED, ha="left", va="bottom")

    # 旧标签：0,1,2 与 2,3,4 在帧 2 重叠 → 连成 0–4 共 5 帧
    y_old = 1.55
    ax.text(-0.15, y_old + 0.31, "旧标签：在帧 2 重叠", ha="right", va="center",
            fontsize=9.5, color=C.COLOR_DARK)
    for f in range(0, 5):
        cell(ax, f, y_old, C.COLOR_BG_RED, C.COLOR_RED)
    ax.text(5.2, y_old + 0.31, "连成 0–4 共 5 帧\n→ 两个音分不开",
            ha="left", va="center", fontsize=9, color=C.COLOR_RED)

    # 新标签：1,2 与 3,4 相接但不重叠
    y_new = 0.35
    ax.text(-0.15, y_new + 0.31, "新标签：相接但不重叠", ha="right", va="center",
            fontsize=9.5, color=C.COLOR_DARK)
    for f in (1, 2, 3, 4):
        cell(ax, f, y_new, C.COLOR_BG_GREEN, C.COLOR_GREEN)
    ax.text(5.2, y_new + 0.31, "各自 2 帧、互不重叠\n→ 能分开",
            ha="left", va="center", fontsize=9, color=C.COLOR_GREEN)

    ax.text(-0.15, -1.12,
            "结论：旧实现能分开的最小间隔是 3 帧（96 ms）；新实现是 2 帧（64 ms）。",
            fontsize=9.5, color=C.COLOR_DARK, ha="left", va="top")
    ax.text(3.35, 5.15, "② 实际影响：间隔 64 ms 时旧标签开始粘连",
            fontsize=12, weight="bold", color=C.COLOR_DARK, ha="left")


def main():
    os.makedirs(OUT, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 9.6),
                             gridspec_kw={"height_ratios": [1.25, 1]})
    fig.subplots_adjust(hspace=0.40, left=0.21, right=0.98, top=0.97, bottom=0.05)
    panel_labels(axes[0])
    panel_merge(axes[1])

    fig.text(0.21, 0.995,
             "onset 标签：3 帧 vs 2 帧（每帧 32 ms；评测容差 50 ms ≈ 1.5 帧）",
             fontsize=13.5, weight="bold", color=C.COLOR_DARK, ha="left", va="top")

    path = os.path.join(OUT, "01_onset_label.png")
    plot_style.save(fig, path)
    plt.close(fig)
    print("写入:", path)
    print(f"1 帧 = {FRAME_MS:.0f} ms")


if __name__ == "__main__":
    main()
