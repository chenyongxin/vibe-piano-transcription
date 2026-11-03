"""画一张「我们的数据长什么样」的图：一段 10 秒音频的特征与全部 6 类标签。

上半：波形、CQT、frame、onset2、velocity、pedal（共用同一条时间轴）；
下半：一首曲目怎么切成 313 帧的段，一个样本与一个 batch 的张量形状。

输出：docs/figs/16_data_types.png

用法（容器内）：
    cd /Docker/git/piano2 && python scripts/plot_data_example.py
"""

from __future__ import annotations

import json
import os

import librosa
import numpy as np
import pandas as pd

from piano2 import plot_style

plot_style.use_cjk_font()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

from piano2.constants import DATA_ROOT, FPS, MAESTRO_ROOT  # noqa: E402

ROOT = "/Docker/git/piano2"
SEG = 313


def choose_segment():
    """挑一个音符密度较高的 10 秒段（避免选到静音开头）。"""
    with open(DATA_ROOT / "train" / "segments.json", encoding="utf-8") as fh:
        segments = json.load(fh)
    best = None
    for seg in segments[:400]:
        base = seg["file"][: -len(".cqt.npy")]
        lab = np.load(DATA_ROOT / "train" / (base + ".lab.npy"), mmap_mode="r")
        density = float(np.asarray(lab[seg["start"]:seg["start"] + seg["n"], 2, :]).sum())
        if best is None or density > best[0]:
            best = (density, seg)
    return best[1]


def find_wav(seg):
    """由数据文件名找到原始 wav 路径。"""
    base = seg["file"][: -len(".cqt.npy")]
    stem = base.split("_", 1)[1]
    df = pd.read_csv(MAESTRO_ROOT / "maestro-v3.0.0.csv")
    row = df[df["audio_filename"].astype(str).str.endswith(stem + ".wav")].iloc[0]
    return MAESTRO_ROOT / str(row["audio_filename"])


def main():
    seg = choose_segment()
    wav_path = find_wav(seg)
    start_frame = seg["start"]
    n = seg["n"]
    t0 = start_frame / FPS
    dur = n / FPS

    y, _ = librosa.load(str(wav_path), sr=16000, offset=t0, duration=dur)
    base = seg["file"][: -len(".cqt.npy")]
    cqt_full = np.load(DATA_ROOT / "train" / (base + ".cqt.npy"), mmap_mode="r")
    full_t = cqt_full.shape[0]
    cqt = np.asarray(cqt_full[start_frame:start_frame + n])
    lab = np.asarray(np.load(DATA_ROOT / "train" / (base + ".lab.npy"), mmap_mode="r")
                     [start_frame:start_frame + n])
    pedal_raw = np.asarray(np.load(DATA_ROOT / "train" / (base + ".pedal.npy"), mmap_mode="r")
                           [start_frame:start_frame + n])
    frame = lab[:, 0, :]
    onset2 = lab[:, 2, :]
    velocity = lab[:, 4, :] / 127.0
    pedal_state = pedal_raw[:, 0].astype(float)
    pedal_on = pedal_raw[:, 1].astype(bool)
    t = np.arange(len(y)) / 16000.0
    tf = np.arange(n) / FPS

    fig = plt.figure(figsize=(14.5, 13.2))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.0, 1.2, 1.05, 1.45],
                          hspace=0.55, wspace=0.22)

    ax_wav = fig.add_subplot(gs[0, 0])
    ax_wav.plot(t, y, color=plot_style.COLOR_BLUE, linewidth=0.4)
    ax_wav.set_title("① 波形（wav，16 kHz）", fontsize=11)
    ax_wav.set_xlabel("时间（秒）", fontsize=9)
    ax_wav.set_ylabel("振幅", fontsize=9)
    ax_wav.grid(alpha=0.25, linestyle=":")

    ax_cqt = fig.add_subplot(gs[0, 1])
    im = ax_cqt.imshow(cqt.T, aspect="auto", origin="lower", cmap="magma",
                       extent=[0, dur, 0, 264], vmin=-80, vmax=0)
    ax_cqt.set_title("② CQT：313 帧 × 264 个频率 bin", fontsize=11)
    ax_cqt.set_xlabel("时间（秒）", fontsize=9)
    ax_cqt.set_ylabel("频率 bin（88 键 × 3）", fontsize=9)
    fig.colorbar(im, ax=ax_cqt, fraction=0.025, pad=0.01, label="dB")

    ax_frame = fig.add_subplot(gs[1, 0])
    ax_frame.imshow(frame.T, aspect="auto", origin="lower", cmap="Greys",
                    extent=[0, dur, 21, 109], vmin=0, vmax=1)
    ax_frame.set_title("③ frame 标签：正在发声（含踏板延音）", fontsize=11)
    ax_frame.set_xlabel("时间（秒）", fontsize=9)
    ax_frame.set_ylabel("MIDI 音高", fontsize=9)

    ax_onset = fig.add_subplot(gs[1, 1])
    ax_onset.imshow(onset2.T, aspect="auto", origin="lower", cmap="Blues",
                    extent=[0, dur, 21, 109], vmin=0, vmax=1)
    ax_onset.set_title("④ onset2 标签：起音（每音 2 帧）", fontsize=11)
    ax_onset.set_xlabel("时间（秒）", fontsize=9)
    ax_onset.set_ylabel("MIDI 音高", fontsize=9)
    ax_onset.text(0.02, 0.94, f"正样本率 {onset2.mean():.2%}（很稀疏）",
                  transform=ax_onset.transAxes, fontsize=8.5, va="top",
                  bbox={"facecolor": "white", "edgecolor": "#bbb",
                        "boxstyle": "round,pad=0.3"})

    ax_vel = fig.add_subplot(gs[2, 0])
    vel_ma = np.ma.masked_where(onset2 == 0, velocity)
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad("white")
    im2 = ax_vel.imshow(vel_ma.T, aspect="auto", origin="lower", cmap=cmap,
                        extent=[0, dur, 21, 109], vmin=0, vmax=1)
    ax_vel.set_title("⑤ velocity 标签：只在起音帧有值", fontsize=11)
    ax_vel.set_xlabel("时间（秒）", fontsize=9)
    ax_vel.set_ylabel("MIDI 音高", fontsize=9)
    fig.colorbar(im2, ax=ax_vel, fraction=0.025, pad=0.01, label="力度（0–1）")

    ax_ped = fig.add_subplot(gs[2, 1])
    down = pedal_state > 0.5
    ax_ped.fill_between(tf, 0, pedal_state, step="post",
                        color=plot_style.COLOR_LIGHT_ORANGE,
                        edgecolor=plot_style.COLOR_ORANGE, linewidth=1.0)
    on_times = tf[pedal_on]
    ax_ped.vlines(on_times, 0, 1.05, color=plot_style.COLOR_RED,
                  linestyle="--", linewidth=1.0)
    ax_ped.set_ylim(-0.05, 1.15)
    ax_ped.set_title(f"⑥ pedal 标签：踏板状态（踩下 {down.mean():.0%}；虚线 = 踩下瞬间）",
                     fontsize=11)
    ax_ped.set_xlabel("时间（秒）", fontsize=9)
    ax_ped.set_ylabel("踏板（0/1）", fontsize=9)
    ax_ped.set_yticks([0, 1])

    # ── 下半：切片与张量形状示意 ──
    ax = fig.add_subplot(gs[3, :])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.93, "⑦ 一首曲目怎么切成样本、又怎么拼成 batch", fontsize=11)
    ax.add_patch(Rectangle((0.03, 0.58), 0.30, 0.14, facecolor=plot_style.COLOR_BG_BLUE,
                           edgecolor=plot_style.COLOR_BLUE, linewidth=1.2))
    ax.text(0.18, 0.65, f"一首曲目：T = {full_t:,} 帧（≈{full_t / FPS:.0f} 秒）",
            ha="center", va="center", fontsize=9)
    for i in range(7):
        x = 0.03 + 0.30 * i / 7
        ax.plot([x, x], [0.58, 0.72], color=plot_style.COLOR_BLUE, linewidth=0.6, alpha=0.7)
    ax.add_patch(Rectangle((0.03 + 0.30 * 2 / 7, 0.58), 0.30 / 7, 0.14,
                           facecolor=plot_style.COLOR_LIGHT_ORANGE,
                           edgecolor=plot_style.COLOR_ORANGE, linewidth=1.4))
    ax.text(0.09, 0.52, "切出 313 帧（10 秒）的段", ha="left", va="center",
            fontsize=9, color=plot_style.COLOR_ORANGE)
    ax.add_patch(FancyArrowPatch((0.34, 0.65), (0.43, 0.65), arrowstyle="-|>",
                                 mutation_scale=14, color=plot_style.COLOR_DARK))

    boxes = [
        ("CQT 样本\n(313, 264) float32\n≈ 326 KB", 0.44, 0.52, plot_style.COLOR_BG_BLUE,
         plot_style.COLOR_BLUE),
        ("标签样本\n(313, 6, 88) uint8\n≈ 165 KB", 0.44, 0.30, plot_style.COLOR_BG_GREEN,
         plot_style.COLOR_GREEN),
        ("踏板样本\n(313, 2) uint8", 0.44, 0.10, plot_style.COLOR_BG_ORANGE,
         plot_style.COLOR_ORANGE),
    ]
    for text, x_, y_, fill, edge in boxes:
        ax.add_patch(Rectangle((x_, y_), 0.16, 0.18, facecolor=fill, edgecolor=edge,
                               linewidth=1.2))
        ax.text(x_ + 0.08, y_ + 0.09, text, ha="center", va="center", fontsize=8.5)

    ax.text(0.64, 0.75, "batch（B = 16）后送进模型：", fontsize=9.5)
    ax.text(0.64, 0.62, "x       (16, 313, 264) float32   → 模型输入\n"
                        "onset   (16, 313, 88)  float32   → 起音标签\n"
                        "frame   (16, 313, 88)  float32   → 发声标签\n"
                        "velocity(16, 313, 88)  float32   → 力度标签\n"
                        "pedal   (16, 313, 1)   float32   → 踏板标签",
            fontsize=8.8, va="top", linespacing=1.6)
    ax.text(0.64, 0.01, "单个样本读取 1.5 ms（memmap 切片），\n整批 16 个约 24 ms；"
                        "旧格式整首解压要 155 ms/样本。", fontsize=8.5,
            color=plot_style.COLOR_GRAY)

    fig.suptitle(f"数据 v2 长什么样：{os.path.basename(str(wav_path))[:46]}… 第 {start_frame} 帧起",
                 fontsize=12, y=0.995)
    out = os.path.join(ROOT, "docs/figs/16_data_types.png")
    plot_style.save(fig, out)
    plt.close(fig)
    print("写入:", out)


if __name__ == "__main__":
    main()
