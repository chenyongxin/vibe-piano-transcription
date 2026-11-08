"""图解：分类目标 vs 回归目标，以及亚帧插值。

四格：
    ① 分类标签：事件落在两帧之间，但目标只能标一帧
    ② 回归标签：目标值编码「帧中心到事件的距离」
    ③ 推理：用相邻帧的值插值出亚帧时间
    ④ 实测：onset 高分辨率提升，但 offset 回归失败导致 note F1 下降

输出：docs/figs/32_classification_vs_regression.png
"""

from __future__ import annotations

import numpy as np

from piano2 import plot_style

plot_style.use_cjk_font()
import matplotlib.pyplot as plt  # noqa: E402

C = plot_style
FPS = 31.25
FRAMES = np.arange(0, 7)          # 帧号 0..6
CENTERS = FRAMES * 32.0           # 帧中心时刻（ms）
EVENT = 109.0                     # 真实事件时刻（ms），落在 96 与 128 之间
J = 5                             # 三角目标半宽（帧）


def panel_class(ax):
    """① 分类标签：只有最近一帧为 1。"""
    nearest = int(np.argmin(np.abs(CENTERS - EVENT)))
    labels = np.zeros(len(FRAMES))
    labels[nearest] = 1.0
    ax.bar(CENTERS, labels, width=22, color=C.COLOR_BLUE, edgecolor=C.COLOR_DARK)
    ax.axvline(EVENT, color=C.COLOR_RED, linestyle="--", linewidth=1.2)
    ax.text(EVENT + 3, 0.78, f"真实事件 {EVENT:.0f} ms", fontsize=8, color=C.COLOR_RED)
    ax.set_xlim(-20, 210)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("时间（ms），帧中心间隔 32 ms", fontsize=9)
    ax.set_ylabel("标签值", fontsize=9)
    ax.set_title("① 分类标签：只有最近一帧为 1（时间被量化）", fontsize=10)
    ax.text(0.5, 0.55, "预测只能落在 96 或 128 ms", fontsize=8.5,
            transform=ax.transAxes, ha="center", color=C.COLOR_GRAY)


def panel_reg(ax):
    """② 回归标签：三角目标编码距离。"""
    targets = np.clip(1.0 - np.abs(CENTERS - EVENT) / (J * 32.0), 0.0, 1.0)
    ax.plot(CENTERS, targets, color=C.COLOR_ORANGE, linewidth=1.8, marker="o",
            markersize=5, label="回归目标 g(Δ)")
    ax.axvline(EVENT, color=C.COLOR_RED, linestyle="--", linewidth=1.2)
    for x, y in zip(CENTERS, targets):
        if y > 0:
            ax.text(x, y + 0.03, f"{y:.2f}", ha="center", fontsize=7, color=C.COLOR_ORANGE)
    ax.text(EVENT + 3, 0.15, f"真实事件 {EVENT:.0f} ms\n在 96 与 128 之间", fontsize=8,
            color=C.COLOR_RED)
    ax.set_xlim(-20, 210)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("时间（ms）", fontsize=9)
    ax.set_ylabel("目标值（0–1）", fontsize=9)
    ax.set_title("② 回归标签：每帧的值编码「到事件的距离」", fontsize=10)
    ax.legend(loc="upper right", fontsize=8, frameon=False)


def panel_interp(ax):
    """③ 亚帧插值：由相邻三帧恢复事件时间。"""
    targets = np.clip(1.0 - np.abs(CENTERS - EVENT) / (J * 32.0), 0.0, 1.0)
    a, b, c = 2, 3, 4       # 帧 64 / 96 / 128
    ya, yb, yc = targets[a], targets[b], targets[c]
    delta = 0.5 * (yc - ya) / (yb - ya)
    recovered = CENTERS[b] + delta * 32.0
    ax.plot(CENTERS, targets, color=C.COLOR_ORANGE, linewidth=1.6, marker="o", markersize=5)
    ax.axvline(EVENT, color=C.COLOR_RED, linestyle="--", linewidth=1.2)
    ax.plot([CENTERS[a], CENTERS[b], CENTERS[c]], [ya, yb, yc], "o",
            color=C.COLOR_BLUE, markersize=6)
    ax.plot([recovered], [np.interp(recovered, CENTERS, targets)], "v",
            color=C.COLOR_GREEN, markersize=10)
    ax.annotate(f"插值恢复 = {recovered:.1f} ms", xy=(recovered, 0.55),
                xytext=(recovered + 4, 0.30), fontsize=8.5, color=C.COLOR_GREEN,
                arrowprops={"arrowstyle": "->", "color": C.COLOR_GREEN})
    ax.text(0.02, 0.95, "用 A/B/C 三帧的值解三角形\nδ = 0.5·32·(yC-yA)/(yB-yA)",
            transform=ax.transAxes, fontsize=8, va="top", color=C.COLOR_DARK)
    ax.set_xlim(-20, 210)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("时间（ms）", fontsize=9)
    ax.set_ylabel("输出值", fontsize=9)
    ax.set_title("③ 推理：插值到亚帧，不再被 32 ms 限制", fontsize=10)


def panel_results(ax):
    """④ 实测：onset10 提升，note 因 offset 崩。"""
    labels = ["Haydn", "Mozart", "Beethoven", "留出集"]
    cls = [0.555, 0.585, 0.469, 0.516]
    reg = [0.778, 0.818, 0.548, 0.688]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, cls, width=0.4, color=C.COLOR_GRAY, label="分类（32 ms）")
    ax.bar(x + 0.2, reg, width=0.4, color=C.COLOR_ORANGE, label="回归 + 亚帧")
    for i, (c, r) in enumerate(zip(cls, reg)):
        ax.text(i - 0.2, c + 0.02, f"{c:.2f}", ha="center", fontsize=7)
        ax.text(i + 0.2, r + 0.02, f"{r:.2f}", ha="center", fontsize=7, color=C.COLOR_ORANGE)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("onset F1 @10 ms", fontsize=9)
    ax.set_title("④ 实测：onset10 明显提升；但 note F1 从 0.410 掉到 0.299", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.2, linestyle=":")


def main():
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6))
    panel_class(axes[0, 0])
    panel_reg(axes[0, 1])
    panel_interp(axes[1, 0])
    panel_results(axes[1, 1])
    fig.suptitle("分类目标 vs 回归目标：标签从「0/1」变成「到事件的距离」", fontsize=13)
    fig.tight_layout()

    from fig_utils import check_clipping

    problems = check_clipping(fig)
    if problems:
        raise SystemExit("图形越界：\n" + "\n".join(problems))

    out = "/Docker/git/piano2/docs/figs/32_classification_vs_regression.png"
    plot_style.save(fig, out)
    plt.close(fig)
    print("写入:", out)


if __name__ == "__main__":
    main()
