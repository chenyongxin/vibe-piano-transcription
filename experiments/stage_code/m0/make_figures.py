"""M0.5 文档配图：流水线示意图 + 数据划分（开发集/留出集）示意图。

为什么单独画：这两张图是为了解释概念，不依赖实验数据，所以用 matplotlib 直接画。

用法（容器内）：
    cd /Docker/git/piano_practice/m0 && python make_figures.py
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

OUT = "/Docker/git/piano_practice/doc/figs"


def draw_pipeline(path):
    """画四段流水线，并标出「解码」一词的两种含义。"""
    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    boxes = [
        (0.3, 4.5, 2.6, 1.3, "1. audio\n16 kHz mono", "#e3f2fd"),
        (3.6, 4.5, 2.9, 1.3, "2. probability maps\nT frames x 88 keys x 3 heads", "#fff3e0"),
        (7.2, 4.5, 2.6, 1.3, "3. note events\npitch / start / end / velocity", "#e8f5e9"),
        (10.4, 4.5, 2.4, 1.3, "4. score\nMusicXML", "#f3e5f5"),
    ]
    for x, y, w, h, text, color in boxes:
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                             facecolor=color, edgecolor="black", linewidth=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)

    arrows = [
        (2.9, 5.15, 0.7, "model\nCNN + Transformer"),
        (6.5, 5.15, 0.7, "decoding\nrules: peaks + frame"),
        (9.8, 5.15, 0.6, "notation\nbeats / chords / voices"),
    ]
    for x, y, w, text in arrows:
        ax.annotate("", xy=(x + w, y), xytext=(x, y),
                    arrowprops=dict(arrowstyle="-|>", color="black", linewidth=1.6))
        ax.text(x + w / 2, y + 0.55, text, ha="center", va="bottom", fontsize=9)

    ax.text(6.5, 2.7, "Two meanings of the word 'decoder'", ha="center", fontsize=11, weight="bold")
    ax.add_patch(mpatches.FancyBboxPatch((0.3, 0.9), 5.8, 1.6, boxstyle="round,pad=0.05",
                                         facecolor="#ffebee", edgecolor="#c62828", linewidth=1.2))
    ax.text(3.2, 1.7, "(A) neural decoder (seq2seq)\ngenerates output token by token\n-- NOT used in this project",
            ha="center", va="center", fontsize=9)
    ax.add_patch(mpatches.FancyBboxPatch((6.9, 0.9), 5.8, 1.6, boxstyle="round,pad=0.05",
                                         facecolor="#e8f5e9", edgecolor="#2e7d32", linewidth=1.2))
    ax.text(9.8, 1.7, "(B) rule-based decoding\nprobability map -> note events\n-- THIS is what we call decoding",
            ha="center", va="center", fontsize=9)
    ax.text(6.5, 0.45, "In this project 'decoding' never means a neural network decoder.",
            ha="center", fontsize=9, style="italic")

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def draw_holdout(path):
    """画数据划分与「开发集 / 留出集」的使用方式。"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.6),
                             gridspec_kw={"height_ratios": [1, 1.4]})

    ax = axes[0]
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 1.6)
    ax.axis("off")
    ax.set_title("MAESTRO data split (number of recordings)", fontsize=11)
    parts = [(0.2, 5.4, "train\n962", "#c8e6c9"), (5.8, 1.2, "validation\n137", "#fff9c4"),
             (7.2, 4.4, "test\n177", "#bbdefb")]
    for x, w, text, color in parts:
        ax.add_patch(mpatches.FancyBboxPatch((x, 0.5), w, 0.7, boxstyle="round,pad=0.03",
                                             facecolor=color, edgecolor="black"))
        ax.text(x + w / 2, 0.85, text, ha="center", va="center", fontsize=10)

    ax = axes[1]
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    ax.set_title("How we use data in this round: dev set vs holdout", fontsize=11)

    ax.add_patch(mpatches.FancyBboxPatch((0.2, 1.7), 5.0, 1.1, boxstyle="round,pad=0.05",
                                         facecolor="#fff3e0", edgecolor="#ef6c00", linewidth=1.3))
    ax.text(2.7, 2.25, "dev set (tuning)\nHaydn / Mozart / Beethoven",
            ha="center", va="center", fontsize=9.5)
    ax.add_patch(mpatches.FancyBboxPatch((6.6, 1.7), 4.6, 1.1, boxstyle="round,pad=0.05",
                                         facecolor="#e8f5e9", edgecolor="#2e7d32", linewidth=1.3))
    ax.text(8.9, 2.25, "holdout (evaluated once)\nnew dense piece",
            ha="center", va="center", fontsize=9.5)

    ax.annotate("", xy=(6.5, 2.25), xytext=(5.3, 2.25),
                arrowprops=dict(arrowstyle="-|>", color="#ef6c00", linewidth=1.6,
                                connectionstyle="arc3,rad=0.35"))
    ax.annotate("", xy=(5.3, 1.95), xytext=(6.5, 1.95),
                arrowprops=dict(arrowstyle="-|>", color="#ef6c00", linewidth=1.6,
                                connectionstyle="arc3,rad=0.35"))
    ax.text(5.9, 2.45, "tune\nmany rounds", ha="center", fontsize=8, color="#ef6c00")
    ax.annotate("", xy=(8.9, 1.55), xytext=(8.9, 1.7),
                arrowprops=dict(arrowstyle="-|>", color="#2e7d32", linewidth=1.6))
    ax.text(9.0, 1.28, "one shot", ha="left", fontsize=8, color="#2e7d32")

    ax.text(0.2, 0.75, "tuning on the same data that evaluates you = cheating (overfitting to that data).\n"
                       "So: dev set for choices, holdout for the final number.",
            ha="left", va="center", fontsize=9, style="italic")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    draw_pipeline(os.path.join(OUT, "00_pipeline.png"))
    draw_holdout(os.path.join(OUT, "00_holdout.png"))
    print("生成:", OUT)
