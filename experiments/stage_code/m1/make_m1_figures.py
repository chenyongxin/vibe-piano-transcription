"""M1 配图：pilot 学习曲线 + onset 标签宽度 A/B + 旧模型基线。

数值来源（可复现）：
    outputs/M1/pilot_onset{1,2,3}/summary.json（最终验证指标）
    以及各 run 训练日志里的 [val] 行（中间步骤指标）
    outputs/M1/baseline_old_model.json（旧模型在同样 120 个验证段上的指标）

输出：doc/figs/13_m1_pilot.png

用法（容器内）：
    cd /Docker/git/piano_practice/m1 && \
    PYTHONPATH=/Docker/git/piano_practice/m0 python make_m1_figures.py
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt

ROOT = "/Docker/git/piano_practice"
sys.path.insert(0, os.path.join(ROOT, "m0"))

import plot_style  # noqa: E402

plot_style.use_cjk_font()
C = plot_style
OUT = os.path.join(ROOT, "doc", "figs")
M1 = os.path.join(ROOT, "outputs", "M1")

# 中间验证点来自 pilot_onset2 的训练日志
PILOT_STEPS = [250, 500, 750, 1000]
PILOT_P = [0.283, 0.345, 0.406, 0.410]
PILOT_R = [0.470, 0.739, 0.764, 0.784]
PILOT_F1 = [0.353, 0.471, 0.530, 0.539]


def load_json(name):
    with open(os.path.join(M1, name), encoding="utf-8") as fh:
        return json.load(fh)


def main():
    old = load_json("baseline_old_model.json")
    ab = {}
    for ch in ("onset1", "onset2", "onset3"):
        ab[ch] = load_json(os.path.join(f"pilot_{ch}", "summary.json"))["final_val"]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

    ax = axes[0]
    ax.plot(PILOT_STEPS, PILOT_F1, "o-", color=C.COLOR_GREEN, linewidth=2, label="F1")
    ax.plot(PILOT_STEPS, PILOT_P, "s--", color=C.COLOR_BLUE, linewidth=1.3, label="精确率")
    ax.plot(PILOT_STEPS, PILOT_R, "^--", color=C.COLOR_ORANGE, linewidth=1.3, label="召回率")
    ax.axhline(old["F1"], color=C.COLOR_GRAY, linestyle=":", linewidth=1.5,
               label=f"旧模型基线 F1={old['F1']:.3f}")
    ax.set_xlabel("训练步数", fontsize=10.5)
    ax.set_ylabel("验证集指标（120 个固定段）", fontsize=10.5)
    ax.set_title("① pilot：1000 步内 F1 持续上升", fontsize=11.5)
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(fontsize=9, loc="lower right")

    ax = axes[1]
    labels = ["1 帧", "2 帧（新默认）", "3 帧（旧）"]
    keys = ["onset1", "onset2", "onset3"]
    f1s = [ab[k]["F1"] for k in keys]
    ps = [ab[k]["P"] for k in keys]
    colors = [C.COLOR_LIGHT_GREEN, C.COLOR_LIGHT_GREEN, C.COLOR_LIGHT_RED]
    xs = range(3)
    width = 0.35
    ax.bar([x - width / 2 for x in xs], f1s, width, color=colors,
           edgecolor=C.COLOR_DARK, linewidth=0.8, label="F1")
    ax.bar([x + width / 2 for x in xs], ps, width, color="white", edgecolor=C.COLOR_BLUE,
           linewidth=0.8, hatch="//", label="精确率")
    for x, (f, p) in enumerate(zip(f1s, ps)):
        ax.text(x - width / 2, f + 0.02, f"{f:.3f}", ha="center", fontsize=9)
        ax.text(x + width / 2, p + 0.02, f"{p:.3f}", ha="center", fontsize=9)
    ax.axhline(old["F1"], color=C.COLOR_GRAY, linestyle=":", linewidth=1.5,
               label=f"旧模型基线 {old['F1']:.3f}")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("1000 步后的验证指标", fontsize=10.5)
    ax.set_title("② onset 标签宽度 A/B（各跑 1000 步）", fontsize=11.5)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.legend(fontsize=9, loc="upper right")

    fig.tight_layout()
    path = os.path.join(OUT, "13_m1_pilot.png")
    plot_style.save(fig, path)
    plt.close(fig)
    print("写入:", path)


if __name__ == "__main__":
    main()
