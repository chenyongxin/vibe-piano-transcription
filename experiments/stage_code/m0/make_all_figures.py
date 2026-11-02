"""重新生成文档配图（中文、干净布局）。

和第一版相比的改动：
    1. 所有标签改中文（容器里装了 fonts-noto-cjk 与 fonts-wqy-zenhei）；
    2. 文字全部放进框内或框间留白，不允许压边框（第一版箭头标签压了框线）；
    3. 布局参考 daylight 项目：低饱和配色、留白充足、每张图只讲一件事。

输出：
    doc/figs/00_pipeline.png        流水线四段 + 「解码」两种含义
    doc/figs/00_holdout.png         数据划分与开发集/留出集
    doc/figs/10_decoders.png        四种解码器的 P/R/F1
    doc/figs/11_case_study.png      误检案例
    doc/figs/12_suppression_map.png 抑制的局部作用范围
    同时把后三张复制到 outputs/M0/m05/figs/ 供实验报告引用。

用法（容器内）：
    cd /Docker/git/piano_practice/m0 && \
    PYTHONPATH=/Docker/git/piano_practice/m0:/Docker/git/piano_practice \
    python make_all_figures.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = "/Docker/git/piano_practice"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "m0"))

import plot_style  # noqa: E402

plot_style.use_cjk_font()

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

import mir_eval.transcription as mtran  # noqa: E402

import decode_v2  # noqa: E402
import notation_v1  # noqa: E402
import prob_cache  # noqa: E402

DOC_FIGS = os.path.join(ROOT, "doc", "figs")
M05_FIGS = os.path.join(ROOT, "outputs", "M0", "m05", "figs")
M05_OUT = os.path.join(ROOT, "outputs", "M0", "m05")
FPS = 16000 / 512
DEV = ("haydn", "mozart", "beethoven", "holdout2")
DEV_CN = {
    "haydn": "Haydn（稀疏 5.9 音/秒）",
    "mozart": "Mozart（中等 9.5 音/秒）",
    "beethoven": "Beethoven（密集 16.7 音/秒）",
    "holdout2": "新留出曲目（密集 15.6 音/秒）",
}
COLOR = plot_style


def box(ax, x, y, w, h, text, fill, edge, fontsize=11, text_color=None):
    """在归一化坐标里画一个圆角框并居中写字。"""
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.012",
            facecolor=fill, edgecolor=edge, linewidth=1.4,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=text_color or COLOR.COLOR_DARK, linespacing=1.6)


def arrow(ax, x0, y0, x1, y1, color=COLOR.COLOR_DARK):
    """画一条箭头。"""
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16,
            color=color, linewidth=1.6, shrinkA=0, shrinkB=0,
        )
    )


# ─────────────────────────── 图 1：流水线 ───────────────────────────


def fig_pipeline():
    fig = plt.figure(figsize=(14, 7.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    stages = [
        ("① 音频 wav\n16 kHz 单声道", COLOR.COLOR_BG_BLUE, COLOR.COLOR_BLUE),
        ("② 概率图\nT 帧 × 88 键 × 3 组输出头", COLOR.COLOR_BG_ORANGE, COLOR.COLOR_ORANGE),
        ("③ 音符事件\n音高 / 起点 / 终点 / 力度", COLOR.COLOR_BG_GREEN, COLOR.COLOR_GREEN),
        ("④ 乐谱\nMusicXML（或 MIDI）", "#f4ecf7", "#8e44ad"),
    ]
    labels = [
        "模型：CNN + Transformer",
        "解码：规则层（峰值定起点、帧定终点）",
        "记谱：拍点 / 量化 / 和弦 / 声部",
    ]
    x0, w, h = 0.03, 0.40, 0.115
    ys = [0.80, 0.58, 0.36, 0.14]
    for (text, fill, edge), y in zip(stages, ys):
        box(ax, x0, y, w, h, text, fill, edge, fontsize=11.5)
    for i, label in enumerate(labels):
        y_top, y_bot = ys[i], ys[i + 1] + h
        arrow(ax, x0 + 0.07, y_top, x0 + 0.07, y_bot)
        ax.text(x0 + 0.10, (y_top + y_bot) / 2, label, ha="left", va="center",
                fontsize=9.5, color=COLOR.COLOR_DARK)

    # 右侧说明：「解码」一词的两种含义
    ax.text(0.745, 0.93, "「解码」在这里是哪个意思？", ha="center", fontsize=12.5,
            weight="bold", color=COLOR.COLOR_DARK)
    box(ax, 0.50, 0.60, 0.47, 0.24,
        "（A）神经网络解码器（seq2seq）\n把输出逐词生成出来\n——本项目没有用到",
        COLOR.COLOR_BG_RED, COLOR.COLOR_RED, fontsize=10.5)
    box(ax, 0.50, 0.30, 0.47, 0.24,
        "（B）规则解码（本项目）\n概率图 → 音符事件\n峰值定起点 · 帧定终点 · 谐波抑制 · 帧门控",
        COLOR.COLOR_BG_GREEN, COLOR.COLOR_GREEN, fontsize=10.5)
    ax.text(0.745, 0.20, "下文说「解码」时，指的都是（B）。", ha="center",
            fontsize=10, style="italic", color=COLOR.COLOR_DARK)

    plot_style.save(fig, os.path.join(DOC_FIGS, "00_pipeline.png"))


# ─────────────────────────── 图 2：数据划分 ───────────────────────────


def fig_holdout():
    fig = plt.figure(figsize=(13.5, 6.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.03, 0.93, "MAESTRO 数据划分（按曲目数量）", fontsize=12.5,
            weight="bold", color=COLOR.COLOR_DARK)
    train = (0.03, 0.64, "训练集\n962 首", COLOR.COLOR_BG_GREEN, COLOR.COLOR_GREEN)
    valid = (0.68, 0.10, "验证集\n137 首", COLOR.COLOR_BG_ORANGE, COLOR.COLOR_ORANGE)
    test = (0.79, 0.18, "测试集\n177 首", COLOR.COLOR_BG_BLUE, COLOR.COLOR_BLUE)
    for x, w, text, fill, edge in (train, valid, test):
        box(ax, x, 0.73, w, 0.13, text, fill, edge, fontsize=10.5)

    ax.text(0.03, 0.63, "本轮实验怎么用这些数据", fontsize=12.5,
            weight="bold", color=COLOR.COLOR_DARK)
    box(ax, 0.03, 0.36, 0.40, 0.20,
        "开发集（模拟卷）\nHaydn / Mozart / Beethoven\n用来调参，可以反复用",
        COLOR.COLOR_BG_ORANGE, COLOR.COLOR_ORANGE, fontsize=10.5)
    box(ax, 0.60, 0.36, 0.37, 0.20,
        "留出集（正式考卷）\n一首没碰过的密集曲目\n只评一次",
        COLOR.COLOR_BG_GREEN, COLOR.COLOR_GREEN, fontsize=10.5)
    arrow(ax, 0.445, 0.46, 0.585, 0.46)
    ax.text(0.515, 0.505, "最后评一次", ha="center", va="bottom", fontsize=9.5,
            color=COLOR.COLOR_GREEN)

    ax.text(0.03, 0.18, "为什么必须分开：拿调参用的数据去报成绩，等于提前看了考卷（过拟合），分数会虚高。",
            fontsize=10, style="italic", color=COLOR.COLOR_DARK)

    plot_style.save(fig, os.path.join(DOC_FIGS, "00_holdout.png"))


# ─────────────────────────── 实验图（需要缓存概率） ───────────────────────────


def load_data():
    step3 = json.load(open(os.path.join(M05_OUT, "step3b_final.json"), encoding="utf-8"))
    caches = {n: prob_cache.load_cache(n) for n in DEV}
    holdout2_midi = os.path.join("/Docker/maestro-v3.0.0", step3["holdout2"]["midi"])
    gts = {}
    for name in DEV:
        midi = prob_cache.GT_MIDI.get(name, holdout2_midi)
        ev = notation_v1.events_from_midi(midi)
        gts[name] = (np.array([[s, e] for (_p, s, e, _v) in ev], float),
                     np.array([p for (p, _s, _e, _v) in ev], int))
    return caches, gts, step3


def arrays(events):
    if not events:
        return np.zeros((0, 2)), np.zeros(0, int)
    return (np.array([[e.start, e.end] for e in events], float),
            np.array([e.pitch for e in events], int))


def metrics(ref_iv, ref_p, est_iv, est_p):
    if len(est_iv) == 0:
        return dict(P=0.0, R=0.0, F1=0.0)
    p, r, f, _ = mtran.precision_recall_f1_overlap(
        ref_iv, ref_p, est_iv, est_p, onset_tolerance=0.05,
        pitch_tolerance=50.0, offset_ratio=None)
    return dict(P=float(p), R=float(r), F1=float(f))


def fig_decoders(caches, gts, step3):
    cfg_final = decode_v2.DecodeConfig(**step3["final_config"])
    cfg_c = decode_v2.DecodeConfig(harmonic_mode="off", method="peak_frame",
                                   onset_threshold=0.60, onset_prominence=0.005,
                                   peak_distance=5, frame_off_threshold=0.18,
                                   release_frames=3)
    cfg_a = decode_v2.DecodeConfig(harmonic_mode="global", harmonic_strength=0.5,
                                   method="state_machine")
    cfg_b = decode_v2.DecodeConfig(harmonic_mode="off", method="state_machine",
                                   onset_threshold=0.70, onset_prominence=0.06,
                                   min_note_frames=2, release_frames=3,
                                   min_reattack_gap=0.08, merge_gap=0.05)
    decoders = [("原管线", cfg_a), ("状态机", cfg_b), ("峰值+帧", cfg_c), ("最终", cfg_final)]

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6), sharey=True)
    metric_names = ["精确率", "召回率", "F1"]
    metric_keys = ["P", "R", "F1"]
    metric_colors = ["#7fb3d5", "#7dcea0", "#f1948a"]
    for ax, name in zip(axes, DEV):
        res = []
        for _label, cfg in decoders:
            c = caches[name]
            ev = decode_v2.decode(c["onset"], c["frame"], c["velocity"], cfg)
            res.append(metrics(*gts[name], *arrays(ev)))
        xs = np.arange(len(decoders))
        for i, (mname, mk, mcolor) in enumerate(zip(metric_names, metric_keys, metric_colors)):
            vals = [r[mk] for r in res]
            ax.bar(xs + (i - 1) * 0.26, vals, 0.26, label=mname, color=mcolor,
                   edgecolor="#555555", linewidth=0.4)
        ax.set_xticks(xs)
        ax.set_xticklabels([d[0] for d in decoders], fontsize=9)
        ax.set_title(DEV_CN[name], fontsize=11)
        ax.set_ylim(0, 1.06)
        ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.6)
    axes[0].set_ylabel("分数", fontsize=10.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    plot_style.save(fig, os.path.join(DOC_FIGS, "10_decoders.png"))
    plot_style.save(fig, os.path.join(M05_FIGS, "fig1_decoders.png"))
    plt.close(fig)


def fig_case_study(caches, gts, step3):
    cfg_final = decode_v2.DecodeConfig(**step3["final_config"])
    cfg_c = decode_v2.DecodeConfig(harmonic_mode="off", method="peak_frame",
                                   onset_threshold=0.60, onset_prominence=0.005,
                                   peak_distance=5, frame_off_threshold=0.18,
                                   release_frames=3)
    c = caches["beethoven"]
    sup = decode_v2.conditional_suppress(c["onset"], c["frame"], cfg_final)
    ev_c = decode_v2.decode(c["onset"], c["frame"], c["velocity"], cfg_c)
    ev_f = decode_v2.decode(c["onset"], c["frame"], c["velocity"], cfg_final)
    iv_c, p_c = arrays(ev_c)
    iv_f, p_f = arrays(ev_f)
    f_set = {(int(p), round(t, 2)) for t, p in zip(iv_f[:, 0], p_f)}
    ref_iv, ref_p = gts["beethoven"]
    case = None
    for t, p in zip(iv_c[:, 0], p_c):
        if (int(p), round(t, 2)) in f_set:
            continue
        if np.any((np.abs(ref_iv[:, 0] - t) < 0.06) & (ref_p == p)):
            continue
        for offset, _decay in decode_v2.HARMONIC_OFFSETS:
            anchor = int(p) - offset
            if np.any((p_c == anchor) & (np.abs(iv_c[:, 0] - t) < 0.12)):
                case = (float(t), int(p), anchor, offset)
                break
        if case:
            break
    if not case:
        raise RuntimeError("没有找到可展示的误检案例")
    t0, k2, k1, offset = case
    half = 0.6
    i0 = max(0, int((t0 - half) * FPS))
    i1 = min(len(c["onset"]), int((t0 + half) * FPS))
    tt = np.arange(i0, i1) / FPS

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.6), sharex=True)
    ax = axes[0]
    ax.plot(tt, c["onset"][i0:i1, k1 - 21], color=COLOR.COLOR_BLUE, linewidth=1.6,
            label=f"锚点 MIDI {k1} 的 onset")
    ax.plot(tt, c["onset"][i0:i1, k2 - 21], color=COLOR.COLOR_RED, linewidth=1.6,
            label=f"泛音 MIDI {k2} 抑制前")
    ax.plot(tt, sup[i0:i1, k2 - 21], color=COLOR.COLOR_RED, linestyle="--", linewidth=1.4,
            label=f"泛音 MIDI {k2} 抑制后")
    ax.axhline(cfg_c.onset_threshold, color=COLOR.COLOR_GRAY, linestyle=":", linewidth=1.2,
               label="检测阈值 0.6")
    ax.axvline(t0, color="#333333", linestyle="--", linewidth=0.9)
    ax.set_ylabel("onset 概率", fontsize=10.5)
    ax.set_ylim(-0.03, 1.08)
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.92)
    ax.set_title(f"一条被纠正的误检：MIDI {k2} 在 {t0:.2f} 秒（锚点 MIDI {k1}，+{offset} 半音）",
                 fontsize=11.5)
    ax.grid(alpha=0.25, linestyle=":", linewidth=0.6)

    ax = axes[1]
    ax.plot(tt, c["frame"][i0:i1, k1 - 21], color=COLOR.COLOR_BLUE, linewidth=1.6,
            label=f"锚点 MIDI {k1} 的 frame")
    ax.plot(tt, c["frame"][i0:i1, k2 - 21], color=COLOR.COLOR_RED, linewidth=1.6,
            label=f"泛音 MIDI {k2} 的 frame")
    ax.axhline(cfg_final.frame_gate, color=COLOR.COLOR_GRAY, linestyle=":", linewidth=1.2,
               label=f"帧门限 {cfg_final.frame_gate:.2f}")
    ax.set_ylabel("frame 概率", fontsize=10.5)
    ax.set_xlabel("时间（秒）", fontsize=10.5)
    ax.set_ylim(-0.03, 1.0)
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.92)
    ax.grid(alpha=0.25, linestyle=":", linewidth=0.6)
    mean_frame = float(c["frame"][i0:i1, k2 - 21].mean())
    ax.text(0.02, 0.62, f"泛音键的 frame 概率平均只有 {mean_frame:.3f}，\n"
                        f"远低于门限 {cfg_final.frame_gate:.2f} → 判定为假音并丢弃",
            transform=ax.transAxes, fontsize=9.5, va="center", ha="left",
            bbox={"facecolor": "#fdedec", "edgecolor": "#c0392b",
                  "boxstyle": "round,pad=0.4"})
    fig.tight_layout()
    plot_style.save(fig, os.path.join(DOC_FIGS, "11_case_study.png"))
    plot_style.save(fig, os.path.join(M05_FIGS, "fig2_case_study.png"))
    plt.close(fig)


def fig_suppression_map(caches, step3):
    cfg_final = decode_v2.DecodeConfig(**step3["final_config"])
    c = caches["beethoven"]
    sup = decode_v2.conditional_suppress(c["onset"], c["frame"], cfg_final)
    ratio = np.where(c["onset"] > 1e-6, sup / np.maximum(c["onset"], 1e-6), 1.0)
    win = int(8 * FPS)
    best_start, best_hits = 0, -1
    for s in range(0, len(ratio) - win, win // 2):
        hits = int((ratio[s:s + win] < 0.999).sum())
        if hits > best_hits:
            best_hits, best_start = hits, s
    seg = ratio[best_start:best_start + win, :].T

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(seg, aspect="auto", origin="lower", cmap="viridis",
                   vmin=0.5, vmax=1.0,
                   extent=[best_start / FPS, (best_start + win) / FPS, 21, 108])
    ax.set_xlabel("时间（秒）", fontsize=10.5)
    ax.set_ylabel("MIDI 音高", fontsize=10.5)
    ax.set_title("条件式抑制的作用范围：抑制后 / 抑制前（越暗压得越多）", fontsize=11.5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("onset 比例", fontsize=10)
    fig.tight_layout()
    plot_style.save(fig, os.path.join(DOC_FIGS, "12_suppression_map.png"))
    plot_style.save(fig, os.path.join(M05_FIGS, "fig3_suppression_map.png"))
    plt.close(fig)


def main():
    os.makedirs(DOC_FIGS, exist_ok=True)
    os.makedirs(M05_FIGS, exist_ok=True)
    print("字体:", plot_style.use_cjk_font())
    fig_pipeline()
    fig_holdout()
    caches, gts, step3 = load_data()
    fig_decoders(caches, gts, step3)
    fig_case_study(caches, gts, step3)
    fig_suppression_map(caches, step3)
    print("完成:", DOC_FIGS)


if __name__ == "__main__":
    main()
