"""把 offset 标签窗口画成「格子表」，直接看每帧是 0 还是 1。

三行分别是 1 帧 / 2 帧 / 3 帧窗口；列是帧号（帧中心时刻）。
输出：docs/figs/33_offset_label_grid.png
"""

from __future__ import annotations

import numpy as np

from piano2 import plot_style

plot_style.use_cjk_font()
import matplotlib.pyplot as plt  # noqa: E402

C = plot_style
CENTERS = np.arange(0, 7) * 32.0     # 帧 0..6 的中心时刻（ms）
NOTE_OFF_FRAME = 3                   # note-off 正好落在第 3 帧中心（96 ms）


def main():
    fig, ax = plt.subplots(figsize=(11, 4.6))
    rows = [
        ("1 帧", [3], "1 格；中心 = 96 ms；正样本率 0.34%；note F1 0.251", C.COLOR_GREEN),
        ("2 帧", [3, 4], "2 格；中心 = 112 ms（+16 ms）；正样本率 0.69%；note F1 0.372", C.COLOR_ORANGE),
        ("3 帧", [2, 3, 4], "3 格；中心 = 96 ms；正样本率 1.03%；note F1 0.285", C.COLOR_BLUE),
    ]
    for r, (label, positive, note, color) in enumerate(rows):
        y = len(rows) - r
        ax.text(-0.75, y, label, ha="right", va="center", fontsize=10, color=color)
        for f in range(len(CENTERS)):
            filled = f in positive
            ax.add_patch(plt.Rectangle((f - 0.42, y - 0.30), 0.84, 0.60,
                                       facecolor=color if filled else "white",
                                       alpha=0.45 if filled else 1.0,
                                       edgecolor=color if filled else "#cccccc",
                                       linewidth=1.2 if filled else 0.8))
            ax.text(f, y, "1" if filled else "0", ha="center", va="center",
                    fontsize=10, color=C.COLOR_DARK)
        ax.text(6.75, y, note, ha="left", va="center", fontsize=8, color=color)

    ax.plot([NOTE_OFF_FRAME, NOTE_OFF_FRAME], [0.55, 3.7], color=C.COLOR_RED,
            linestyle="--", linewidth=1.2, zorder=0)
    ax.text(NOTE_OFF_FRAME + 0.05, 3.55, "note-off = 96 ms（第 3 帧中心）",
            ha="left", fontsize=9, color=C.COLOR_RED)
    ax.set_xlim(-1.3, 12.2)
    ax.set_ylim(0.4, 3.8)
    ax.set_xticks(np.arange(len(CENTERS)))
    ax.set_xticklabels([f"帧 {f}\n{c:.0f} ms" for f, c in enumerate(CENTERS)], fontsize=8)
    ax.set_yticks([])
    ax.set_title("offset 标签窗口：note-off 附近有几帧被标成正样本（1）", fontsize=11)
    ax.text(-1.3, 0.55, "注意：窗口不是音符时值、也不是音频窗长，只是训练目标里 note-off 附近的正样本范围。",
            fontsize=8.5, color=C.COLOR_GRAY)
    fig.tight_layout()

    from fig_utils import check_clipping

    problems = check_clipping(fig)
    if problems:
        raise SystemExit("图形越界：\n" + "\n".join(problems))

    out = "/Docker/git/piano2/docs/figs/33_offset_label_grid.png"
    plot_style.save(fig, out)
    plt.close(fig)
    print("写入:", out)


if __name__ == "__main__":
    main()
