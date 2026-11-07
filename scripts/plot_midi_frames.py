"""图解 MIDI 事件、时间分辨率与 CQT 帧量化的关系。

四格：
    ① MIDI 是事件表（不是每 1ms 采样）
    ② 三种时间刻度对比：wav 采样 / MIDI tick / CQT 帧
    ③ 量化到 32ms 帧网格：±16ms 从哪来
    ④ 实测偏差：2 帧标签 +16ms 系统偏差，1 帧 ≈ 0

输出：docs/figs/24_midi_vs_frames.png
"""

from __future__ import annotations

import numpy as np

from piano2 import plot_style

plot_style.use_cjk_font()
import matplotlib.pyplot as plt  # noqa: E402

C = plot_style


def panel_events(ax):
    """① MIDI 事件表：事件发生在任意时刻，不是等间隔采样。"""
    rows = [("note-on", 2.2, [(0.12, "pitch=60"), (0.44, "pitch=64"), (0.61, "pitch=67")],
             C.COLOR_DARK),
            ("note-off", 1.2, [(0.30, ""), (0.52, ""), (0.80, "")], C.COLOR_BLUE),
            ("CC64（踏板）", 0.2, [(0.05, "值=0"), (0.35, "值=80"), (0.70, "值=0")],
             C.COLOR_GREEN)]
    for label, y, events, color in rows:
        ax.axhline(y, color="#eeeeee", linewidth=0.8, zorder=0)
        for t, note in events:
            ax.plot([t, t], [y - 0.22, y + 0.22], color=color, linewidth=1.6)
            ax.plot(t, y + 0.22, marker="o", markersize=3, color=color)
            if note:
                ax.text(t, y + 0.34, note, fontsize=7, ha="center", color=color)
        ax.text(-0.02, y, label, ha="right", va="center", fontsize=9, color=C.COLOR_DARK)
    ax.annotate("", xy=(0.30, 2.62), xytext=(0.12, 2.62),
                arrowprops={"arrowstyle": "<->", "color": "#999999", "linewidth": 0.8})
    ax.text(0.21, 2.66, "时间差（delta-time）", fontsize=7.5, ha="center",
            color="#666666")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.4, 2.95)
    ax.set_yticks([])
    ax.set_xlabel("时间（秒）", fontsize=9)
    ax.set_title("① MIDI = 事件表：事件发生在任意时刻，不是等间隔采样", fontsize=10)


def panel_resolution(ax):
    """② 三种时间刻度（对数轴）。"""
    items = [("wav 采样（16 kHz）", 0.0625, C.COLOR_BLUE),
             ("MIDI tick（480/拍 @120 BPM）", 1.0417, C.COLOR_ORANGE),
             ("我们的 CQT 帧（hop=512）", 32.0, C.COLOR_GREEN)]
    for i, (label, value, color) in enumerate(items):
        y = 2 - i
        ax.plot([0.04, value], [y, y], color=color, alpha=0.75, linewidth=9, solid_capstyle="butt")
        ax.text(value * 1.15, y, f"{value:.4g} ms", va="center", fontsize=9, color=color)
        ax.text(0.05, y - 0.38, label, fontsize=8, color=C.COLOR_DARK)
    ax.set_xscale("log")
    ax.set_xlim(0.04, 200)
    ax.set_xticks([0.0625, 0.25, 1, 4, 32, 100])
    ax.set_xticklabels(["0.06", "0.25", "1", "4", "32", "100"], fontsize=8)
    ax.set_ylim(-0.6, 2.6)
    ax.set_yticks([])
    ax.set_xlabel("时间间隔（毫秒，对数轴）", fontsize=9)
    ax.set_title("② 时间分辨率：MIDI 比帧网格细约 30 倍", fontsize=10)
    ax.text(0.05, 2.42, "MIDI 没有「采样率」，只有时间单位 tick", fontsize=8,
            color=C.COLOR_GRAY)


def panel_quant(ax):
    """③ 32ms 帧网格量化：最坏误差 ±16ms。"""
    for k in range(0, 6):
        center = k * 32.0
        ax.axvline(center, color="#bbbbbb", linewidth=0.8, linestyle="--", zorder=0)
        ax.text(center, 1.75, f"{center:.0f}", ha="center", fontsize=7.5,
                color=C.COLOR_GRAY)
    ax.set_xlim(-16, 176)
    ax.set_ylim(0, 2.1)
    onset = 46.0
    ax.plot([onset], [1.15], marker="v", markersize=9, color=C.COLOR_RED)
    ax.text(onset, 1.3, "真实起点 46 ms", ha="center", fontsize=8, color=C.COLOR_RED)
    ax.annotate("", xy=(32, 0.95), xytext=(onset, 0.95),
                arrowprops={"arrowstyle": "<->", "color": C.COLOR_RED, "linewidth": 1.0})
    ax.text((onset + 32) / 2, 0.78, "取整误差 14 ms（≤16）", ha="center",
            fontsize=8, color=C.COLOR_RED)
    ax.plot([32], [0.5], marker="o", markersize=6, color=C.COLOR_DARK)
    ax.text(32, 0.42, "最近帧中心 32 ms", fontsize=8, color=C.COLOR_DARK, ha="center")
    # 2 帧标签窗口：帧 1 与帧 2，中心在 48ms
    ax.add_patch(plt.Rectangle((32 - 16, 0.02), 64, 0.18, facecolor=C.COLOR_BG_ORANGE,
                               edgecolor=C.COLOR_ORANGE, linewidth=0.8))
    ax.plot([48], [0.11], marker="^", markersize=7, color=C.COLOR_ORANGE)
    ax.text(48, 0.27, "2 帧标签中心 = +16 ms", ha="center", fontsize=7.5,
            color=C.COLOR_ORANGE)
    ax.set_yticks([])
    ax.set_xlabel("时间（毫秒）", fontsize=9)
    ax.set_title("③ 量化到 32 ms 帧：随机误差 ±16 ms；2 帧标签再偏 +16 ms", fontsize=10)


def panel_measured(ax):
    """④ 实测带符号偏差（上一轮模型）。"""
    labels = ["2 帧 onset\nHaydn", "2 帧 onset\nMozart", "1 帧 onset\nHaydn", "1 帧 onset\nMozart"]
    vals = [14.7, 12.9, -0.5, -1.1]
    colors = [C.COLOR_ORANGE] * 2 + [C.COLOR_GREEN] * 2
    ax.bar(labels, vals, color=colors, edgecolor=C.COLOR_DARK, linewidth=0.5)
    ax.axhline(0, color=C.COLOR_DARK, linewidth=0.8)
    ax.axhspan(-16, 16, color=C.COLOR_GRAY, alpha=0.12)
    ax.text(3.45, 16.2, "灰色带 = ±16 ms", ha="right", fontsize=8,
            color=C.COLOR_GRAY)
    for i, v in enumerate(vals):
        ax.text(i, v + (1.2 if v >= 0 else -1.8), f"{v:+.1f}", ha="center", fontsize=8.5)
    ax.set_ylim(-17, 19)
    ax.set_ylabel("估计起点 - GT（ms）", fontsize=9)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.set_title("④ 实测：2 帧系统性偏晚约 13–15 ms，1 帧归零", fontsize=10)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.8))
    panel_events(axes[0, 0])
    panel_resolution(axes[0, 1])
    panel_quant(axes[1, 0])
    panel_measured(axes[1, 1])
    fig.suptitle("MIDI 事件 vs CQT 帧：±16 ms 量化误差从哪来", fontsize=13)
    fig.tight_layout()

    from fig_utils import check_clipping

    problems = check_clipping(fig)
    if problems:
        raise SystemExit("图形越界（会被裁掉）：\n" + "\n".join(problems))

    out = "/Docker/git/piano2/docs/figs/24_midi_vs_frames.png"
    plot_style.save(fig, out)
    plt.close(fig)
    print("写入:", out)


if __name__ == "__main__":
    main()
