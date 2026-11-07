"""图解 onset 概率图（鼓包）、2 帧标签与找峰。

四格：
    ① 二维概率图：时间帧 × 琴键，一个起音是一个「鼓包」
    ② 单个琴键的时间曲线：1 帧 / 2 帧标签与模型输出鼓包、找峰位置
    ③ 三维视角：每个琴键一条时间曲线，峰在每条曲线上找
    ④ 为什么最初用 2 帧：量化与对齐抖动下的容错窗口

输出：docs/figs/25_onset_bulge_peak.png
"""

from __future__ import annotations

import numpy as np

from piano2 import plot_style

plot_style.use_cjk_font()
import matplotlib.pyplot as plt  # noqa: E402

C = plot_style
T, K = 48, 88


def blob(time, key, t0, k0, wt=1.3, wk=1.2):
    """二维高斯鼓包：在 (t0, k0) 处生成起音概率。"""
    return np.exp(-0.5 * (((time - t0) / wt) ** 2 + ((key - k0) / wk) ** 2))


def panel_heatmap(ax):
    """① 时间 × 琴键的 onset 概率图。"""
    time = np.arange(T)[:, None]
    key = np.arange(K)[None, :]
    prob = blob(time, key, 20, 60) + 0.6 * blob(time, key, 30, 64, wt=1.0)
    im = ax.imshow(prob.T, origin="lower", aspect="auto", cmap="magma",
                   extent=[0, T, 0, K], vmin=0, vmax=1)
    # 2 帧标签窗口（第 20、21 帧，键 60）
    ax.add_patch(plt.Rectangle((19.5, 59.5), 2, 1, fill=False, edgecolor="#00e5ff", linewidth=1.5))
    ax.annotate("2 帧标签窗口\n（第 20–21 帧）", xy=(21, 60.8), xytext=(25, 72),
                fontsize=8, color="#00e5ff",
                arrowprops={"arrowstyle": "->", "color": "#00e5ff"})
    ax.set_xlabel("帧（1 帧 = 32 ms）", fontsize=9)
    ax.set_ylabel("琴键（MIDI 音高）", fontsize=9)
    ax.set_title("① onset 概率图：每个起音是时间×琴键上的一个鼓包", fontsize=10)


def panel_curve(ax):
    """② 单键曲线：标签窗口 vs 模型输出鼓包 vs 找峰。"""
    t = np.arange(12, 30)
    # 模型输出：中心在 20.5 帧的鼓包（被 2 帧标签拉宽）
    curve = np.exp(-0.5 * ((t - 20.5) / 1.25) ** 2)
    ax.plot(t, curve, color=C.COLOR_BLUE, linewidth=1.8, label="模型输出的 onset 概率")
    ax.vlines([20, 21], 0, 1.15, color=C.COLOR_ORANGE, linewidth=2,
              label="2 帧标签（第 20、21 帧为正）")
    ax.plot([20, 21], [1.12, 1.12], "o", color=C.COLOR_ORANGE, markersize=5)
    peak_idx = int(np.argmax(curve))
    peak_x = int(t[peak_idx])
    ax.plot(peak_x, curve[peak_idx], "v", color=C.COLOR_RED, markersize=9,
            label=f"找到的峰（第 {peak_x} 帧）")
    ax.axvline(20.5, color=C.COLOR_GRAY, linewidth=1.0, linestyle=":")
    ax.text(22.3, 0.55, "2 帧窗口中心在 20.5 帧\n比峰位晚半帧 = 16 ms",
            fontsize=8, color=C.COLOR_GRAY, ha="left", va="center")
    ax.set_xlim(12, 30)
    ax.set_ylim(0, 1.25)
    ax.set_xlabel("帧", fontsize=9)
    ax.set_ylabel("概率", fontsize=9)
    ax.legend(loc="upper right", fontsize=7.5, frameon=False)
    ax.set_title("② 2 帧标签把鼓包拉宽，峰位有半帧的不确定", fontsize=10)


def panel_3d(ax):
    """③ 三维视角：时间 × 键 × 概率。"""
    tt = np.linspace(0, T, 80)
    kk = np.linspace(40, 80, 60)
    TIME, KEY = np.meshgrid(tt, kk)
    Z = blob(TIME, KEY, 20, 60) + 0.6 * blob(TIME, KEY, 30, 64, wt=1.0)
    ax.plot_surface(TIME, KEY, Z, cmap="magma", linewidth=0, antialiased=True, alpha=0.9)
    ax.set_xlabel("帧", fontsize=8)
    ax.set_ylabel("琴键", fontsize=8)
    ax.set_zlabel("概率", fontsize=8)
    ax.set_title("③ 三维看：每个琴键一条时间曲线，找峰在每条曲线上进行", fontsize=10)
    ax.tick_params(labelsize=7)


def panel_tolerance(ax):
    """④ 为什么最初用 2 帧：量化与对齐的容错。"""
    ax.set_xlim(-0.2, 3.4)
    ax.set_ylim(0, 2.4)
    centers = [0, 1, 2, 3]
    for c in centers:
        ax.axvline(c, color="#bbbbbb", linewidth=0.8, linestyle="--", zorder=0)
        ax.text(c, 2.15, f"帧 {c}", ha="center", fontsize=8, color=C.COLOR_GRAY)
    ax.add_patch(plt.Rectangle((0, 0.55), 2, 0.35, facecolor=C.COLOR_BG_ORANGE,
                               edgecolor=C.COLOR_ORANGE, linewidth=1))
    ax.text(1, 0.72, "2 帧标签窗口", ha="center", fontsize=8.5, color=C.COLOR_ORANGE)
    ax.add_patch(plt.Rectangle((1, 1.2), 1, 0.35, facecolor=C.COLOR_BG_GREEN,
                               edgecolor=C.COLOR_GREEN, linewidth=1))
    ax.text(2, 1.37, "1 帧标签", ha="center", fontsize=8.5, color=C.COLOR_GREEN)
    for x in (0.3, 1.15, 1.7):
        ax.plot(x, 0.25, marker="v", color=C.COLOR_RED, markersize=8)
    ax.text(1.0, 0.05, "真实起音可能落在帧内任意位置（±16 ms）", ha="center",
            fontsize=8, color=C.COLOR_DARK)
    ax.set_yticks([])
    ax.set_xlabel("帧编号", fontsize=9)
    ax.set_title("④ 最初用 2 帧：覆盖取整误差与对齐抖动，代价是半帧偏晚", fontsize=10)


def main():
    fig = plt.figure(figsize=(13.5, 9.2))
    gs = fig.add_gridspec(2, 2)
    panel_heatmap(fig.add_subplot(gs[0, 0]))
    panel_curve(fig.add_subplot(gs[0, 1]))
    panel_3d(fig.add_subplot(gs[1, 0], projection="3d"))
    panel_tolerance(fig.add_subplot(gs[1, 1]))
    fig.suptitle("onset 鼓包、2 帧标签与找峰：半帧误差从哪里来", fontsize=13)
    fig.tight_layout()

    from fig_utils import check_clipping

    problems = check_clipping(fig)
    if problems:
        raise SystemExit("图形越界（会被裁掉）：\n" + "\n".join(problems))

    out = "/Docker/git/piano2/docs/figs/25_onset_bulge_peak.png"
    plot_style.save(fig, out)
    plt.close(fig)
    print("写入:", out)


if __name__ == "__main__":
    main()
