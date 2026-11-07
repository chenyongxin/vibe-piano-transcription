"""为教程定制五张图（不再跨文档复用同一张总览图）。

产物：
    docs/figs/26_tutorial_cqt_example.png    教程 02：真实 CQT 片段（T×264）
    docs/figs/27_tutorial_label_channels.png 教程 03：真实标签通道
    docs/figs/28_tutorial_slicing_batch.png  教程 04：切片与 batch 示意
    docs/figs/29_tutorial_four_forms.png     教程 01：四种数据形态
    docs/figs/30_tutorial_pipeline_stages.png 教程 05：四段流水线接口
"""

from __future__ import annotations

import json
import os

import numpy as np

from piano2 import plot_style

plot_style.use_cjk_font()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

C = plot_style
DATA = "/Docker/maestro_processed_v2"
OUT = "/Docker/git/piano2/docs/figs"
FPS = 31.25


def load_example():
    """取验证集第一段，返回 (cqt, lab, pedal, seconds)。"""
    with open(os.path.join(DATA, "validation", "segments.json"), encoding="utf-8") as fh:
        seg = json.load(fh)[0]
    base = seg["file"][: -len(".cqt.npy")]
    start, n = seg["start"], seg["n"]
    cqt = np.load(os.path.join(DATA, "validation", seg["file"]), mmap_mode="r")
    lab = np.load(os.path.join(DATA, "validation", base + ".lab.npy"), mmap_mode="r")
    pedal = np.load(os.path.join(DATA, "validation", base + ".pedal.npy"), mmap_mode="r")
    return (np.asarray(cqt[start:start + n]), np.asarray(lab[start:start + n]),
            np.asarray(pedal[start:start + n]), n / FPS)


def save(fig, name):
    from fig_utils import check_clipping

    problems = check_clipping(fig)
    if problems:
        raise SystemExit(f"{name} 越界：\n" + "\n".join(problems))
    path = os.path.join(OUT, name)
    plot_style.save(fig, path)
    plt.close(fig)
    print("写入:", path)


def fig_cqt_example():
    cqt, _lab, _pedal, seconds = load_example()
    fig, ax = plt.subplots(figsize=(11, 4.8))
    im = ax.imshow(cqt.T, origin="lower", aspect="auto", cmap="magma",
                   extent=[0, seconds, 0, 264])
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("能量（dB）", fontsize=9)
    # 标注 3 个八度参考线
    for key in (21, 45, 69, 93):
        bin_idx = (key - 21) * 3
        if 0 <= bin_idx < 264:
            ax.axhline(bin_idx, color="white", linewidth=0.5, alpha=0.35, linestyle=":")
            ax.text(seconds + 0.05, bin_idx, f"键 {key}", fontsize=7, color="white",
                    va="center")
    ax.set_xlabel("时间（秒）", fontsize=9)
    ax.set_ylabel("264 个频率 bin", fontsize=9)
    ax.set_title("真实的 CQT 片段：横轴 313 帧 = 10 秒，纵轴 264 bin（88 键 × 3/半音）",
                 fontsize=10)
    ax.text(0.02, 0.95, "亮线 = 基频与谐波", transform=ax.transAxes, fontsize=9,
            color="white", va="top")
    fig.tight_layout()
    save(fig, "26_tutorial_cqt_example.png")


def fig_label_channels():
    _cqt, lab, pedal, seconds = load_example()
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2))

    ax = axes[0, 0]
    ax.imshow(lab[:, 0, :].T, origin="lower", aspect="auto", cmap="Greys",
              extent=[0, seconds, 0, 88], interpolation="nearest")
    ax.set_title("① frame（含踏板延音）", fontsize=10)
    ax.set_xlabel("时间（秒）", fontsize=9)
    ax.set_ylabel("88 键", fontsize=9)

    ax = axes[0, 1]
    notes = []
    for f in range(lab.shape[0]):
        for key in range(88):
            if lab[f, 1, key] and (not notes or notes[-1][0] != key or f - notes[-1][1] > 6):
                notes.append((key, f))
    notes = notes[:4]
    for idx, (key, f) in enumerate(notes):
        y = len(notes) - idx
        ax.add_patch(plt.Rectangle((f / FPS, y - 0.3), 1 / FPS, 0.6,
                                   color=C.COLOR_ORANGE, label="onset1（1 帧）" if idx == 0 else None))
        ax.add_patch(plt.Rectangle((f / FPS, y - 0.3), 2 / FPS, 0.6, fill=False,
                                   edgecolor=C.COLOR_BLUE, linewidth=1.2,
                                   label="onset2（2 帧）" if idx == 0 else None))
        ax.text(f / FPS + 3.2 / FPS, y, f"键 {key+21}：帧 {f} / {f+1}", fontsize=7.5,
                va="center")
    ax.set_xlim(0, min(seconds, 12))
    ax.set_ylim(0.2, len(notes) + 0.8)
    ax.set_xlabel("时间（秒）", fontsize=9)
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=7.5, frameon=False)
    ax.set_title("② onset1 vs onset2 的标签宽度", fontsize=10)

    ax = axes[1, 0]
    times, vels = [], []
    for f in range(lab.shape[0]):
        for key in range(88):
            if lab[f, 4, key]:
                times.append(f / FPS)
                vels.append(int(lab[f, 4, key]))
    ax.scatter(times, vels, s=6, color=C.COLOR_BLUE, alpha=0.6)
    ax.set_xlim(0, seconds)
    ax.set_ylim(0, 130)
    ax.set_xlabel("时间（秒）", fontsize=9)
    ax.set_ylabel("velocity（0–127）", fontsize=9)
    ax.set_title("③ velocity：每个起音一个力度值", fontsize=10)

    ax = axes[1, 1]
    ax.step(np.arange(lab.shape[0]) / FPS, pedal[:, 0], where="post",
            color=C.COLOR_GREEN, linewidth=1.2)
    ax.set_xlim(0, seconds)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("时间（秒）", fontsize=9)
    ax.set_ylabel("踏板状态（0/1）", fontsize=9)
    ax.set_title("④ pedal：由 CC64 转成逐帧状态", fontsize=10)

    fig.suptitle("同一段 10 秒音频的四个标签通道（从 MIDI 计算出来的监督信号）", fontsize=12)
    fig.tight_layout()
    save(fig, "27_tutorial_label_channels.png")


def _box(ax, xy, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.012",
                                facecolor="white", edgecolor=color, linewidth=1.4))
    ax.text(xy[0] + w / 2, xy[1] + h - 0.06, title, ha="center", va="top",
            fontsize=9.5, color=color)
    ax.text(xy[0] + w / 2, xy[1] + h / 2 - 0.01, body, ha="center", va="center",
            fontsize=8)


def _arrow(ax, x1, x2, y):
    ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>",
                                 mutation_scale=12, color=C.COLOR_GRAY))


def fig_slicing_batch():
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0.05, 0.78), 0.9, 0.10, facecolor=C.COLOR_BG_BLUE,
                               edgecolor=C.COLOR_BLUE))
    ax.text(0.5, 0.83, "一首曲目：T 帧 × 264（T 随曲目长度变化）", ha="center",
            va="center", fontsize=9)
    for i in range(3):
        x = 0.08 + i * 0.30
        ax.add_patch(plt.Rectangle((x, 0.55), 0.26, 0.10, facecolor=C.COLOR_BG_ORANGE,
                                   edgecolor=C.COLOR_ORANGE))
        ax.text(x + 0.13, 0.60, f"段 {i+1}：313 帧", ha="center", va="center", fontsize=8.5)
    ax.text(0.5, 0.47, "segments.json 记录每段的（文件、起始帧、长度）", ha="center",
            fontsize=8.5, color=C.COLOR_GRAY)
    _arrow(ax, 0.5, 0.5, 0.44)
    ax.add_patch(plt.Rectangle((0.18, 0.22), 0.28, 0.14, facecolor="white",
                               edgecolor=C.COLOR_BLUE, linewidth=1.2))
    ax.text(0.32, 0.29, "一个样本\nx (313, 264)\n+ 4 个标签通道", ha="center",
            va="center", fontsize=8.5)
    ax.add_patch(plt.Rectangle((0.56, 0.22), 0.28, 0.14, facecolor="white",
                               edgecolor=C.COLOR_GREEN, linewidth=1.2))
    ax.text(0.70, 0.29, "一个 batch\n(16, 313, 264)\n16 个样本堆叠", ha="center",
            va="center", fontsize=8.5)
    _arrow(ax, 0.46, 0.56, 0.29)
    ax.text(0.5, 0.10, "读取时用内存映射（memmap）：只加载需要的 10 秒，不整首解压",
            ha="center", fontsize=9, color=C.COLOR_DARK)
    ax.set_title("从整首歌到训练 batch：切片、样本与堆叠", fontsize=11)
    fig.tight_layout()
    save(fig, "28_tutorial_slicing_batch.png")


def fig_four_forms():
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = [
        ("波形 wav", "16 kHz 采样\n连续声音", "输入", C.COLOR_BLUE),
        ("CQT 矩阵", "T × 264\n模型看见的特征", "给模型看", C.COLOR_ORANGE),
        ("标签/概率矩阵", "T × 88\n训练监督信号", "监督/输出", C.COLOR_GREEN),
        ("符号谱 MIDI / MusicXML", "音符事件、时值\n给人读的乐谱", "标签来源 / 最终输出", C.COLOR_DARK),
    ]
    for i, (title, body, role, color) in enumerate(boxes):
        x = 0.03 + i * 0.245
        _box(ax, (x, 0.32), 0.21, 0.42, title, body, color)
        ax.text(x + 0.105, 0.24, role, ha="center", fontsize=8, color=color)
        if i < 3:
            _arrow(ax, x + 0.215, x + 0.24, 0.53)
    ax.set_title("同一段钢琴音乐的四种数据形态：谁给谁、各自扮演什么角色", fontsize=11)
    fig.tight_layout()
    save(fig, "29_tutorial_four_forms.png")


def fig_pipeline_stages():
    fig, ax = plt.subplots(figsize=(12, 5.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    stages = [
        ("① 音频 → CQT", "输入：wav（任意采样率）\n输出：(T, 264)\n工具：librosa", "音频读取/重采样失败"),
        ("② 概率图", "输入：(T, 264)\n输出：5–6 个 (T, 88) 概率\n工具：CNN + Transformer", "显存不足/推理超时"),
        ("③ 音符事件", "输入：概率矩阵\n输出：音符列表\n工具：规则解码器", "阈值不当→漏检/误检"),
        ("④ 乐谱", "输入：音符列表\n输出：MusicXML/PDF\n工具：记谱 + verovio", "排版失败/字体缺失"),
    ]
    for i, (title, body, failure) in enumerate(stages):
        x = 0.02 + i * 0.245
        _box(ax, (x, 0.38), 0.21, 0.40, title, body, C.COLOR_BLUE if i % 2 == 0 else C.COLOR_ORANGE)
        ax.text(x + 0.105, 0.28, "常见失败：" + failure, ha="center", fontsize=7.5,
                color=C.COLOR_RED)
        if i < 3:
            _arrow(ax, x + 0.215, x + 0.24, 0.58)
    ax.set_title("四段流水线：每段的输入、输出、工具与失败模式", fontsize=11)
    fig.tight_layout()
    save(fig, "30_tutorial_pipeline_stages.png")


def main():
    fig_cqt_example()
    fig_label_channels()
    fig_slicing_batch()
    fig_four_forms()
    fig_pipeline_stages()


if __name__ == "__main__":
    main()
