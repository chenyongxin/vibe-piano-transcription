# M0：评测闭环 + 记谱 v1

这是 G0 阶段的实验代码，跑在用户的 `piano` Docker 容器里。目的不是做产品，而是回答三个问题：

1. 评测代码可信吗？（oracle 自检）
2. 模型到底什么水平？（onset-only / note-with-offset / 逐帧 F1，三种解码对照）
3. 记谱层换成「拍点估计 + 和弦化 + 时值重估」后，谱面能好多少？（GT 直通 + 模型输出两条路）

结论见 `../outputs/M0/report.md` 与 `../doc/04_m0_results.md`。

## 环境

容器基于 `pytorch/pytorch:2.9.0-cuda12.6-cudnn9-runtime`，已装 torch / librosa / music21 / pretty_midi / mir_eval。

额外依赖（渲染预览用，M0 期间已装进容器）：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple lxml verovio pymupdf
apt-get install -y librsvg2-bin   # 需要能装 apt 包（rsvg-convert）
apt-get install -y fonts-noto-cjk fonts-wqy-zenhei   # 画中文图用；没有字体时 matplotlib 会把中文画成方块
```

为什么需要 rsvg-convert：verovio 输出 SVG，但它的字体内嵌在 SVG 里；PyMuPDF 的 SVG 解析不支持 CSS class，会渲染成白图。librsvg 能正确解析。

## 一键重跑

```bash
docker exec -w /Docker/git/piano_practice/m0 \
  -e PYTHONPATH=/Docker/git/piano_practice/m0:/Docker/git/piano_practice \
  piano python run_m0.py
```

产物写到 `../outputs/M0/`。

## 文件说明

| 文件 | 作用 |
|---|---|
| `run_m0.py` | 主流程：oracle 自检 → GT v0/v1 → 模型 vs GT（三种解码）→ nezha2 出谱 |
| `notation_v1.py` | 记谱 v1：拍点估计、量化、和弦化、时值重估、手动建小节、MusicXML 输出 |
| `score_metrics.py` | 谱面可读性指标（黑音符占比、休止符、声部数、连音、时长一致性） |
| `render_preview.py` | verovio → SVG → rsvg-convert → PNG；带白图检测 |
| `mxlib.py` | 零依赖 MusicXML / MIDI 解析器（来自审计阶段，已修和弦时值 bug） |
| `decode_v2.py` | 可配置解码器：状态机 / 峰值+帧、谐波抑制三模式、帧一致性门控 |
| `prob_cache.py` | 把模型概率缓存到磁盘，调参时只做解码、不重复推理 |
| `m05_step1.py` | 步骤 1：只关谐波抑制，看 nezha2 前后对比 |
| `m05_step2.py` | 步骤 2：状态机参数网格搜索 |
| `m05_step2b.py` | 步骤 2 总结：三种解码器对照 |
| `m05_step3.py` | 步骤 3：条件式谐波抑制初版 |
| `m05_step3b.py` | 步骤 3b：开发集/留出集修正 + 扩展网格 |
| `m05_step3_finalize.py` | 步骤 3 收尾：归因实验、案例图、nezha2 最终谱 |
| `preflight.py` | 训练前门禁 G1–G8：数据/泄漏/特征/评测/解码/资源/可复现性 |
| `plot_style.py` | 绘图风格：中文字体注册、统一配色、保存工具 |
| `make_all_figures.py` | 重新生成 `doc/figs/` 下全部文档配图（中文） |
| `_probe_decode.py` | 复现「谐波抑制吃掉 80% 音符」的对照实验 |
| `_probe_probs.py` | 概率分布与阈值扫描（证明模型概率是双峰、检测本身很好） |

## 复现要点

- 评测集：MAESTRO test 按音符密度取 10/50/90 分位三首（`run_m0.py` 里自动选）。
- 指标口径：`mir_eval.transcription.precision_recall_f1_overlap`，onset 容差 50 ms、pitch 容差 50 cents；`offset_ratio=None` 为 onset-only，`offset_ratio=0.2` 为含时值。
- 系统延迟：滑窗推理有 10–20 ms 系统偏差，比较前用 `best_latency()` 对齐。
- 逐帧标签：`pretty_midi.get_piano_roll` 默认应用 sustain pedal，与训练标签口径一致。
