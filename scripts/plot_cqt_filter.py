"""画一张「CQT 的滤波器 h 在做什么」的图解（清爽版）。

四格：
    ① 滤波器 = 窗函数 × 复正弦（小波）
    ② 内积：信号段 × 小波 → 一个复数（|X| = 能量）
    ③ 频率响应：Hann 窗 vs 矩形窗（旁瓣/泄漏）
    ④ 短音频的边界问题：1 秒音频放不下 1.87 秒的低频窗

排版约定：图例一律放在坐标轴下方，数据区里不放任何文字标注，
避免文字压曲线、颜色与曲线混淆。

输出：docs/figs/23_cqt_filter.png
"""

from __future__ import annotations

import numpy as np

from piano2 import plot_style

plot_style.use_cjk_font()
import matplotlib.pyplot as plt  # noqa: E402

SR, BPO = 16000, 36
Q = 1.0 / (2 ** (1 / BPO) - 1)
C = plot_style


def wavelet(freq):
    """返回 (复小波, 窗长采样数, 时间轴秒)。"""
    n = int(round(Q * SR / freq))
    t = np.arange(n) / SR
    window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(n) / n))
    return window * np.exp(2j * np.pi * freq * t), n, t


def legend_below(ax, ncol=2):
    """把图例放到坐标轴下方（不占数据区）。"""
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=ncol,
              fontsize=8, frameon=False)


def panel_wavelet(ax):
    """① h = 窗函数 × 复正弦。"""
    freq = 440.0
    psi, n, t = wavelet(freq)
    ax.plot(t * 1000, psi.real, color=C.COLOR_BLUE, linewidth=0.8, label="复小波实部")
    ax.plot(t * 1000, np.abs(psi), color=C.COLOR_ORANGE, linewidth=1.2,
            linestyle="--", label="窗函数包络")
    ax.plot(t * 1000, -np.abs(psi), color=C.COLOR_ORANGE, linewidth=1.2,
            linestyle="--", label="_nolegend_")
    ax.set_xlim(0, t[-1] * 1000)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlabel("窗内时间（ms）", fontsize=9)
    ax.set_ylabel("幅度", fontsize=9)
    ax.set_title(f"① h = 窗函数 × 复正弦（{freq:.0f} Hz，窗长 {n/SR*1000:.0f} ms，"
                 f"约 {Q:.0f} 个周期）", fontsize=10)
    legend_below(ax)
    ax.grid(alpha=0.18, linestyle=":")


def panel_inner(ax):
    """② 内积：信号 × 小波 → 复数。"""
    freq = 440.0
    psi, n, t = wavelet(freq)
    rng = np.random.default_rng(0)
    signal = 0.6 * np.sin(2 * np.pi * freq * t) + 0.05 * rng.standard_normal(n)
    product = signal * np.conj(psi)
    coeff = np.sum(product) / n
    product_norm = product.real / max(1e-9, np.max(np.abs(product.real))) * 0.6

    ax.plot(t * 1000, signal, color="#8e8e8e", linewidth=0.7, alpha=0.8, label="信号 x[n]")
    ax.plot(t * 1000, np.abs(psi) * 0.95, color=C.COLOR_ORANGE, linewidth=1.0,
            linestyle="--", label="|h[n]|（窗包络）")
    ax.plot(t * 1000, product_norm, color=C.COLOR_GREEN, linewidth=0.9,
            label="x[n]·h[n] 的实部（归一化）")
    ax.set_xlim(0, t[-1] * 1000)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlabel("窗内时间（ms）", fontsize=9)
    ax.set_ylabel("幅度（归一化）", fontsize=9)
    ax.set_title(f"② 内积 Σ x[n]·h[n] → 一个复数；本帧 |X| ≈ {abs(coeff):.2f}",
                 fontsize=10)
    legend_below(ax, ncol=2)
    ax.grid(alpha=0.18, linestyle=":")


def panel_response(ax):
    """③ 窗函数对频谱泄漏的影响。"""
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
    ax.semilogy(freqs[mask], spec_hann[mask], color=C.COLOR_BLUE, linewidth=1.3,
                label="Hann 窗（本项目使用）")
    ax.semilogy(freqs[mask], spec_rect[mask], color=C.COLOR_RED, linewidth=1.0,
                alpha=0.75, label="矩形窗（不压两侧）")
    ax.set_ylim(1e-8, 1.5)
    ax.set_xlabel("频率（Hz）", fontsize=9)
    ax.set_ylabel("幅度（归一化，对数轴）", fontsize=9)
    ax.set_title("③ 压住两侧 → 旁瓣低很多 → 漏到隔壁 bin 的能量更少", fontsize=10)
    legend_below(ax)
    ax.grid(alpha=0.18, which="both", linestyle=":")


def panel_short(ax):
    """④ 1 秒音频 vs 1.87 秒低频窗。"""
    ax.set_xlim(0, 2.0)
    ax.set_ylim(0.2, 2.5)
    # 信号
    ax.add_patch(plt.Rectangle((0, 0.75), 1.0, 0.5, facecolor=C.COLOR_BG_GREEN,
                               edgecolor=C.COLOR_GREEN, linewidth=1.2))
    ax.text(0.5, 1.0, "1 秒音频", ha="center", va="center", fontsize=10)
    # 窗：左段有信号，右段补零
    ax.add_patch(plt.Rectangle((0, 1.45), 1.0, 0.5, facecolor=C.COLOR_BG_ORANGE,
                               edgecolor=C.COLOR_ORANGE, linewidth=1.2))
    ax.add_patch(plt.Rectangle((1.0, 1.45), 0.87, 0.5, facecolor="white",
                               edgecolor=C.COLOR_ORANGE, linewidth=1.2, hatch="//"))
    ax.text(0.5, 1.7, "低频 bin 的窗（1.87 秒）", ha="center", va="center", fontsize=10)
    ax.text(1.435, 1.7, "补零", ha="center", va="center", fontsize=9,
            color=C.COLOR_DARK)
    ax.text(1.0, 0.45, "信号只占窗的一半多 → 低频 bin 有边缘效应；\n"
                       "歌曲有几分钟长，只影响开头/结尾的少数帧。",
            ha="center", fontsize=9, color=C.COLOR_DARK)
    ax.set_yticks([])
    ax.set_xlabel("时间（秒）", fontsize=9)
    ax.set_title("④ 1 秒音频放不下 1.87 秒的低频窗：两侧补零", fontsize=10)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.4))
    panel_wavelet(axes[0, 0])
    panel_inner(axes[0, 1])
    panel_response(axes[1, 0])
    panel_short(axes[1, 1])
    fig.suptitle("CQT 的滤波器 h：小波 → 内积 → 一个 bin 的能量", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    from fig_utils import check_clipping

    problems = check_clipping(fig)
    if problems:
        raise SystemExit("图形越界（会被裁掉）：\n" + "\n".join(problems))

    out = "/Docker/git/piano2/docs/figs/23_cqt_filter.png"
    plot_style.save(fig, out)
    plt.close(fig)
    print("写入:", out)


if __name__ == "__main__":
    main()
