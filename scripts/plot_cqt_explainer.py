"""画一张 CQT 解剖图（教学用）：bin 分布、滤波器窗长、帧移重叠、单音响应。

输出：docs/figs/22_cqt_explained.png

用法（容器内）：
    cd /Docker/git/piano2 && python scripts/plot_cqt_explainer.py
"""

from __future__ import annotations

import librosa
import numpy as np

from piano2 import plot_style

plot_style.use_cjk_font()
import matplotlib.pyplot as plt  # noqa: E402

SR, HOP, N_BINS, BPO, FMIN = 16000, 512, 264, 36, 27.5
Q = 1.0 / (2 ** (1 / BPO) - 1)
C = plot_style


def bin_freq(k):
    """第 k 个 CQT bin 的中心频率（Hz）。"""
    return FMIN * 2 ** (k / BPO)


def panel_bins(ax):
    """① DFT 的等间隔 bin vs CQT 的等比间隔 bin。"""
    ax.set_xscale("log")
    ax.set_xlim(24, 5200)
    ax.set_ylim(-0.2, 2.6)
    ax.grid(axis="x", alpha=0.15, linestyle=":")
    ax.set_axisbelow(True)

    dft = np.linspace(120, 4000, 28)
    ax.vlines(dft, 1.65, 2.1, color=C.COLOR_BLUE, linewidth=3)
    ax.text(26, 2.18, "DFT / STFT：间隔恒定（Δf = fs/N）", fontsize=10, color=C.COLOR_BLUE)

    cqt = np.array([bin_freq(k) for k in range(0, 264, 3)])
    cqt = cqt[(cqt > 24) & (cqt < 5200)]
    ax.vlines(cqt, 0.45, 0.95, color=C.COLOR_ORANGE, linewidth=1.3)
    ax.text(26, 1.02, "CQT：间隔 ∝ 频率（每半音 3 个 bin）", fontsize=10, color=C.COLOR_ORANGE)

    for f in (27.5, 110, 440, 1760, 4186):
        ax.text(f, 0.12, f"{f:.0f} Hz", ha="center", fontsize=8, color=C.COLOR_GRAY)
    ax.set_xlabel("频率（对数轴）", fontsize=9)
    ax.set_yticks([])
    ax.set_title("① 频率 bin 的分布：等间隔 vs 等比间隔", fontsize=11)


def panel_wavelets(ax):
    """② 三个频率 bin 的滤波器窗长：同一 Q、不同宽度。"""
    t = np.linspace(0, 1.0, 3000)
    for i, (freq, color) in enumerate(((55.0, C.COLOR_BLUE),
                                       (440.0, C.COLOR_ORANGE),
                                       (1760.0, C.COLOR_GREEN))):
        win = Q / freq                            # 窗长（秒）
        u = (t - 0.5) / win
        env = np.where(np.abs(u) <= 0.5, 0.5 * (1 + np.cos(2 * np.pi * u)), 0.0)
        y = 2.2 - i * 0.95
        ax.plot(t, env * 0.7 + y, color=color, linewidth=1.6,
                label=f"{freq:.0f} Hz：窗长 {win * 1000:.0f} ms（约 51 个周期）")
        ax.axhline(y, color="#dddddd", linewidth=0.6, zorder=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1,
              fontsize=8, frameon=False)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.2, 2.85)
    ax.set_yticks([])
    ax.set_xlabel("时间（秒）", fontsize=9)
    ax.set_title("② 每个 bin 的滤波器窗长：频率越低，窗越长", fontsize=11)


def panel_frames(ax):
    """③ 帧移 hop：不同窗长下的相邻帧重叠。"""
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 3.1)
    rows = [(2.35, Q / 55.0, "低频：窗 1.87 s（重叠严重）", C.COLOR_BLUE),
            (1.35, Q / 440.0, "中频：窗 0.12 s（部分重叠）", C.COLOR_ORANGE),
            (0.35, Q / 1760.0, "高频：窗 0.03 s（几乎不重叠）", C.COLOR_GREEN)]
    for y, width, label, color in rows:
        centers = np.arange(0.16, 1.0, HOP / SR)
        for c0 in centers[:8]:
            ax.add_patch(plt.Rectangle((c0 - width / 2, y - 0.16), width, 0.32,
                                       facecolor=color, alpha=0.30,
                                       edgecolor=color, linewidth=0.8))
        ax.text(0.015, y + 0.25, label, fontsize=8.5, color=C.COLOR_DARK)
    ax.text(0.5, 2.95, "每帧中心间隔 = hop = 512 采样 = 32 ms", ha="center",
            fontsize=9, color=C.COLOR_GRAY)
    ax.set_yticks([])
    ax.set_xlabel("时间（秒）", fontsize=9)
    ax.set_title("③ 帧移 hop：低频窗大面积重叠，高频窗几乎不重叠", fontsize=11)


def panel_tone(ax):
    """④ 单音 C4 在 CQT 上的响应。"""
    f0 = 261.63
    t = np.arange(int(1.5 * SR)) / SR
    y = (0.5 * np.sin(2 * np.pi * f0 * t)).astype(np.float32)
    cqt = np.abs(librosa.cqt(y, sr=SR, hop_length=HOP, fmin=FMIN,
                             n_bins=N_BINS, bins_per_octave=BPO))
    energy = cqt.mean(axis=1)
    ks = np.arange(110, 126)
    vals = energy[ks] / energy.max()
    colors = [C.COLOR_ORANGE if k == 117 else C.COLOR_LIGHT_ORANGE for k in ks]
    ax.bar(ks, vals, color=colors, edgecolor=C.COLOR_DARK, linewidth=0.5)
    ax.bar([117], [vals[ks == 117][0]], color=C.COLOR_ORANGE,
           edgecolor=C.COLOR_DARK, linewidth=0.5, label="bin 117 = C4（每半音 3 个 bin）")
    ax.set_xlabel("CQT bin 编号", fontsize=9)
    ax.set_ylabel("相对能量", fontsize=9)
    ax.set_title("④ 纯音 C4（261.63 Hz）→ 能量集中在 bin 117 附近 3 格", fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=1,
              fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.25, linestyle=":")


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.2))
    panel_bins(axes[0, 0])
    panel_wavelets(axes[0, 1])
    panel_frames(axes[1, 0])
    panel_tone(axes[1, 1])
    fig.suptitle("CQT 解剖：bin 是等比间隔的频率；每个 bin 有自己的窗长（常数 Q）",
                 fontsize=13)
    fig.tight_layout()
    out = "/Docker/git/piano2/docs/figs/22_cqt_explained.png"
    plot_style.save(fig, out)
    plt.close(fig)
    print("写入:", out)


if __name__ == "__main__":
    main()
