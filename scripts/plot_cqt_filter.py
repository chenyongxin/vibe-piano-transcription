"""画一张「CQT 的滤波器 h 在做什么」的图解。

四格：
    ① 滤波器 = 窗函数 × 复正弦（小波）
    ② 内积：信号段 × 小波 → 一个复数（取模 = 能量）
    ③ 频率响应：Hann 窗 vs 矩形窗，说明「两侧为 0」的作用
    ④ 短音频的边界问题：1 秒音频放不下 1.87 秒的低频窗

输出：docs/figs/23_cqt_filter.png
"""

from __future__ import annotations

import numpy as np

from piano2 import plot_style

plot_style.use_cjk_font()
import matplotlib.pyplot as plt  # noqa: E402

SR, BPO, FMIN = 16000, 36, 27.5
Q = 1.0 / (2 ** (1 / BPO) - 1)
C = plot_style


def wavelet(freq, n=None):
    """返回 (复小波, 窗长采样数, 时间轴秒)。"""
    n = n or int(round(Q * SR / freq))
    t = np.arange(n) / SR
    window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(n) / n))
    return window * np.exp(2j * np.pi * freq * t), n, t


def panel_wavelet(ax):
    """① 滤波器 = 窗函数 × 复正弦。"""
    freq = 440.0
    psi, n, t = wavelet(freq)
    ax.plot(t * 1000, psi.real, color=C.COLOR_BLUE, linewidth=0.8, label="复小波实部")
    ax.plot(t * 1000, np.abs(psi), color=C.COLOR_ORANGE, linewidth=1.2,
            linestyle="--", label="窗函数包络")
    ax.plot(t * 1000, -np.abs(psi), color=C.COLOR_ORANGE, linewidth=1.2, linestyle="--")
    ax.set_xlim(0, t[-1] * 1000)
    ax.set_xlabel("窗内时间（ms）", fontsize=9)
    ax.set_ylabel("幅度", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(f"① 滤波器 h = 窗函数 × 复正弦（{freq:.0f} Hz，窗长 {n/SR*1000:.0f} ms）",
                 fontsize=10.5)
    ax.text(t[-1]*500, -1.12, f"窗内约 {Q:.0f} 个周期，两侧被窗函数压到 0", fontsize=8.5,
            color=C.COLOR_GRAY, ha="center")


def panel_inner(ax):
    """② 内积：信号段 × 小波 → 复数。"""
    freq = 440.0
    psi, n, t = wavelet(freq)
    rng = np.random.default_rng(0)
    signal = 0.6 * np.sin(2 * np.pi * freq * t) + 0.05 * rng.standard_normal(n)
    product = signal * np.conj(psi)
    coeff = np.sum(product) / n
    ax.plot(t * 1000, signal[:n], color="#bbbbbb", linewidth=0.7, label="信号 x[n]")
    ax.plot(t * 1000, np.abs(psi), color=C.COLOR_ORANGE, linewidth=0.9,
            linestyle="--", label="|h[n]|")
    ax.plot(t * 1000, product.real / max(1e-9, np.max(np.abs(product.real))) * 0.6,
            color=C.COLOR_GREEN, linewidth=0.9, label="乘积的实部")
    ax.set_xlim(0, t[-1] * 1000)
    ax.set_xlabel("窗内时间（ms）", fontsize=9)
    ax.set_ylabel("幅度", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("② 内积：Σ x[n]·h[n] → 一个复数；|系数| 就是该 bin 的能量", fontsize=10.5)
    ax.text(t[-1]*500, -0.95, f"这一帧这一 bin 的能量 = |X| ≈ {abs(coeff):.2f}",
            fontsize=9, color=C.COLOR_DARK, ha="center")


def panel_response(ax):
    """③ 频率响应：Hann 窗 vs 矩形窗。"""
    n = int(round(Q * SR / 440.0))
    t = np.arange(n) / SR
    hann = 0.5 * (1 - np.cos(2 * np.pi * np.arange(n) / n))
    rect = np.ones(n)
    spec_hann = np.abs(np.fft.rfft(hann * np.cos(2 * np.pi * 440 * t), 8192))
    spec_rect = np.abs(np.fft.rfft(rect * np.cos(2 * np.pi * 440 * t), 8192))
    freqs = np.fft.rfftfreq(8192, 1 / SR)
    spec_hann = spec_hann / spec_hann.max()
    spec_rect = spec_rect / spec_rect.max()
    mask = freqs <= 1200
    ax.plot(freqs[mask], 20 * np.log10(spec_hann[mask] + 1e-9),
            color=C.COLOR_BLUE, linewidth=1.3, label="Hann 窗（我们的窗）")
    ax.plot(freqs[mask], 20 * np.log10(spec_rect[mask] + 1e-9),
            color=C.COLOR_RED, linewidth=1.0, alpha=0.8, label="矩形窗（不衰减两侧）")
    ax.set_ylim(-80, 5)
    ax.set_xlabel("频率（Hz）", fontsize=9)
    ax.set_ylabel("幅度（dB，归一化）", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("③ 窗函数的作用：压住两侧 → 频谱泄漏小很多", fontsize=10.5)
    ax.annotate("矩形窗旁瓣高\n（泄漏到隔壁 bin）", xy=(520, -13), xytext=(650, -45),
                fontsize=8, color=C.COLOR_RED,
                arrowprops={"arrowstyle": "->", "color": C.COLOR_RED})
    ax.grid(alpha=0.2, linestyle=":")


def panel_short(ax):
    """④ 短音频 vs 低频窗：窗放不下。"""
    ax.set_xlim(0, 2.0)
    ax.set_ylim(0, 2.4)
    ax.add_patch(plt.Rectangle((0, 0.9), 1.0, 0.5, facecolor=C.COLOR_BG_GREEN,
                               edgecolor=C.COLOR_GREEN, linewidth=1.2))
    ax.text(0.5, 1.15, "1 秒音频", ha="center", va="center", fontsize=10)
    ax.add_patch(plt.Rectangle((0, 1.55), 1.87, 0.5, facecolor=C.COLOR_BG_ORANGE,
                               edgecolor=C.COLOR_ORANGE, linewidth=1.2, alpha=0.9))
    ax.text(0.95, 1.8, "低频 bin 的窗（1.87 s）", ha="center", va="center", fontsize=10)
    ax.annotate("", xy=(1.87, 2.15), xytext=(1.0, 2.15),
                arrowprops={"arrowstyle": "<->", "color": C.COLOR_RED, "linewidth": 1.2})
    ax.text(1.43, 2.24, "只能补零", ha="center", fontsize=9, color=C.COLOR_RED)
    ax.text(0.02, 0.55, "信号在窗内只占一半多 → 低频 bin 的边缘效应变强、能量被低估；\n"
                        "歌曲有几分钟长，这个问题只在开头/结尾的少数帧出现。",
            fontsize=9, color=C.COLOR_DARK)
    ax.set_yticks([])
    ax.set_xlabel("时间（秒）", fontsize=9)
    ax.set_title("④ 1 秒音频放不下低频窗：两侧要补零", fontsize=10.5)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
    panel_wavelet(axes[0, 0])
    panel_inner(axes[0, 1])
    panel_response(axes[1, 0])
    panel_short(axes[1, 1])
    fig.suptitle("CQT 的滤波器 h：小波 → 内积 → 一个 bin 的能量", fontsize=13)
    fig.tight_layout()
    out = "/Docker/git/piano2/docs/figs/23_cqt_filter.png"
    plot_style.save(fig, out)
    plt.close(fig)
    print("写入:", out)


if __name__ == "__main__":
    main()
