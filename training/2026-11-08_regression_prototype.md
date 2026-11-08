# 2026-11-08 回归 + 亚帧原型：首轮结果

## 摘要

本文记录 2026-11-08 的第一次「回归目标 + 亚帧推理」实验：先在当前最佳分类模型上拿到 MAESTRO test split 的可比基准，再把模型改成 8 头（分类 6 + 回归 2），用三角目标训练 `reg_base`、`reg_pure`、`reg_pure2` 三组。读者不需要读过之前的报告；读完后能知道我们的 test split 与论文的差距、onset 回归与 offset 回归各自的表现、以及为什么 note-with-offset 仍然没修好。

## 1. 背景与目的

之前的结论：分类配方（v3b + onset1 + offset 损失 2.0）在留出集上有 0.403 的 note F1，但没有 MAESTRO test split 的可比数字；同时我们决定把目标改成追 SOTA 的高分辨率指标，路线是 **CQT + CNN/Transformer + 回归目标 + 亚帧推理**（区别于论文的 log-mel + biGRU）。

这一轮要回答：

1. 当前分类模型在 MAESTRO test split（177 首）上与论文差多少？
2. 在同一 CQT 骨干上加 onset_reg / offset_reg 两个回归头，能不能提升时间分辨率与 note F1？
3. 如果不行，瓶颈在 onset 还是 offset？

## 2. 实验设计

| 项目 | 内容 |
|---|---|
| 硬件 | RTX 4060 8GB |
| 数据 | data_v4：v3b 7 通道 + onset_reg / offset_reg 三角回归目标（J=5，uint8 0–255） |
| 模型 | 8 头：分类 6（onset/frame/velocity/pedal/pedal_on/offset 分类）+ 回归 2（onset_reg/offset_reg） |
| 训练 | 20 轮 71,160 步，batch 16，lr 3e-4 cosine，标签 onset1 |
| 解码 | 分类路径：原 `decode`（offset-end）；回归路径：`decode_subframe`（局部最大帧 + 邻帧三角插值） |

| 组 | 配置 | 目的 |
|---|---|---|
| `reg_base` | offset 分类损失 1.0 + 回归损失 0.5 | 回归与分类共存 |
| `reg_pure` | offset 分类损失 0 + 回归损失 1.0 | 去掉 offset 分类头的竞争 |
| `reg_pure2` | offset 分类损失 0 + 回归损失 2.0 | 加强回归监督 |

## 3. 结果

### 3.1 MAESTRO test split 基准（当前分类最佳模型）

| 指标 | 数值 |
|---|---|
| onset F1 @50 ms | 0.8709（P 0.8909 / R 0.8517） |
| onset F1 @30 ms | 0.8606 |
| onset F1 @20 ms | 0.8039 |
| onset F1 @10 ms | 0.5067 |
| note-with-offset F1 | 0.3272（P 0.3347 / R 0.3200） |

参考论文（ByteDance，[arXiv:2010.01815](https://arxiv.org/abs/2010.01815)）在 MAESTRO 上的 onset F1 为 **96.72%**，pedal onset F1 91.86%。我们与它的 onset 差距约 **9.6 个点**。

### 3.2 回归模型的开发集 / 留出集

| 模型 | note F1（H/M/B） | 留出 note | onset50 F1（H/M/B） | 留出 onset50 | onset10 F1（H/M/B） | 留出 onset10 |
|---|---|---|---|---|---|---|
| 分类最佳（combo_loss2） | 0.328 / 0.504 / 0.215 | 0.410 | 0.963 / 0.966 / 0.824 | 0.868 | 0.555 / 0.585 / 0.469 | 0.516 |
| reg_base | 0.182 / 0.452 / 0.176 | 0.283 | 0.925 / 0.925 / 0.809 | 0.864 | 未测 | 未测 |
| reg_pure | 0.231 / 0.428 / 0.159 | 0.244 | 0.931 / 0.917 / 0.809 | 0.863 | 0.745 / 0.775 / 0.507 | 0.667 |
| **reg_pure2** | 0.204 / 0.437 / 0.165 | 0.299 | **0.946 / 0.941 / 0.839** | **0.880** | **0.778 / 0.818 / 0.548** | **0.688** |

**核心发现**：

- **onset 回归成功**：在 10 ms 容差下，reg_pure2 比分类模型高 **8–23 个点**（Haydn 0.555→0.778、Mozart 0.585→0.818、Beethoven 0.469→0.548、留出集 0.516→0.688）；onset_reg 峰位置相对 GT 的中位误差只有 **+0.3 ms**，77% 在 10 ms 内。
- **offset 回归失败**：offset_reg 峰相对 GT note-off 的中位偏差 **+10.7 ~ +13.7 ms**，只有 22–24% 在 10 ms 内；用亚帧 offset 配对后预测时长中位 **1.6–2.3 秒**（GT 150 ms），note F1 崩到 0.16–0.44。
- 去掉 offset 分类头（reg_pure / reg_pure2）没有修好 offset 回归；加强回归权重只略微改善 onset50/onset10，note F1 仍远低于分类路径。

## 4. 观察与分析

- **为什么 onset 回归好、offset 回归差**：onset 是锤击瞬态，音频证据强；offset 是琴键释放，没有明显瞬态，且常被踏板/混响掩盖，回归目标更难学。论文也强调 offset 的歧义与鲁棒性问题。
- **note F1 的崩溃不是解码配对单一原因**：offset_reg 本身偏差 14 ms、误差分布宽，配出的时值系统性错误；即使配对规则再改，输入信号也不够准。
- **高分辨率 onset 已经有价值**：10 ms 容差的提升是实质性的，说明「回归目标 + 插值」在 onset 上可行；下一步应把 offset 当成独立难题，而不是顺手一起回归。
- **当前路线的定位**：分类配方仍是「可用谱面」的最佳选择（note F1 0.410、test split note 0.327）；回归原型是「高分辨率 onset」的原型，还没到能替代分类配方。

## 5. 下一步猜想

| 假设 | 实验设计 | 预期 | 成本 |
|---|---|---|---|
| offset 需要独立子模块/更强监督 | 单独 offset 回归子模块 + 加权回归损失（起止附近权重高） | offset 误差 ≤10 ms 占比明显提升 | 1 天 + 1.6 小时 |
| offset 回归目标用「后半段」信息 | 在 onset 后一段时间窗内回归，减少与 onset 目标的重叠 | 配对的时值更准 | 1 天 |
| 高分辨率 onset 可先落地 | 把 reg_pure2 的 onset 回归接到分类模型的 duration 上（onset 用亚帧、时值用 offset 分类） | 10 ms onset F1 提升且 note 不降 | 约半天 |
| offset 分类 + frame 继续优化 | 现有分类配方继续调 offset 解码/损失 | note F1 小幅提升 | 1 天 |

## 6. 结论

这一轮验证了：**「CQT + CNN/Transformer + 回归目标 + 亚帧推理」在 onset 上是通的**（10 ms F1 从 0.52 提到 0.69），但在 offset 上失败（中位误差约 14 ms，note F1 崩掉）。因此当前阶段应保留分类配方作为可用模型，把 offset 回归作为独立研究课题；同时，回归 onset 的高分辨率成果可以作为下一步「onset 亚帧 + 分类时值」混合方案的部件。

## 7. 相关链接

- 基准结果：`outputs/eval/test_split_baseline.json`
- 回归评测：`outputs/eval/reg_base.json`、`reg_pure.json`、`reg_pure2.json`
- 误差分析：`outputs/eval/reg_diagnostics.json`、`reg_local_refine.json`、`classifier_tight_onset.json`
- 数据与代码：`scripts/build_labels_v4.py`、`src/piano2/decode.py` 的 `decode_subframe`
