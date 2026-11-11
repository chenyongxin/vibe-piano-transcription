"""画 11 ms 事件在整数帧 T 上的分类/回归目标，解释亚帧插值。

输出：docs/figs/44_v4_subframe_intuition.png
"""

from __future__ import annotations

import numpy as np

from piano2 import plot_style

plot_style.use_cjk_font()
import matplotlib.pyplot as plt  # noqa: E402

C = plot_style
FPS = 31.25
J = 5


def exact_value(frame, event_time):
    """精确事件时间下的三角目标。"""
    center = frame / FPS
    return max(0.0, 1.0 - abs(center - event_time) / (J / FPS))


def rounded_value(frame, event_time):
    """当前 v4 实现：把事件 round 到最近整数帧后，再按整数帧距离生成目标。"""
    event_frame = int(round(event_time * FPS))
    return max(0.0, 1.0 - abs(frame - event_frame) / J)


def main():
    event_time = 0.011  # 11 ms
    frames = np.arange(-4, 5)
    exact = np.array([exact_value(f, event_time) for f in frames])
    rounded = np.array([rounded_value(f, event_time) for f in frames])

    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.plot(frames, exact, color=C.COLOR_ORANGE, marker="o", linewidth=2,
            label="精确事件时间目标（v5 的目标）")
    ax.plot(frames, rounded, color=C.COLOR_GRAY, marker="s", linestyle="--",
            linewidth=1.8, label="当前 v4：round 到最近帧后的目标")
    ax.axvline(event_time * FPS, color=C.COLOR_RED, linestyle="--", linewidth=1.4,
               label="真实事件 T=0.344（11 ms）")
    ax.axhline(0, color="#dddddd", linewidth=0.8)
    ax.scatter([0], [1.0], color=C.COLOR_BLUE, s=60, zorder=5,
               label="分类 onset1：只标 T=0 这一帧")
    ax.annotate("精确目标在相邻帧不对称\nT0=0.931, T1=0.869",
                xy=(0.5, 0.90), xytext=(-3.6, 0.82), fontsize=9,
                arrowprops={"arrowstyle": "->", "color": C.COLOR_ORANGE})
    ax.set_xticks(frames)
    ax.set_xlabel("整数帧 T（帧中心 = T × 32 ms）", fontsize=10)
    ax.set_ylabel("目标值", fontsize=10)
    ax.set_ylim(-0.05, 1.12)
    ax.set_xlim(-4.3, 4.3)
    ax.grid(alpha=0.2, linestyle=":")
    ax.legend(fontsize=8.5, loc="upper left", frameon=True)
    ax.set_title("11 ms 事件在整数帧上的目标：精确距离 vs 当前 v4 的取整", fontsize=12)
    fig.tight_layout()
    fig.savefig("/Docker/git/piano2/docs/figs/44_v4_subframe_intuition.png",
                dpi=150, facecolor="white")
    plt.close(fig)
    print("frames", frames.tolist())
    print("exact", np.round(exact, 4).tolist())
    print("rounded", np.round(rounded, 4).tolist())


if __name__ == "__main__":
    main()
