# 2026-11-12 v5 精确时间回归目标：首轮负结果

## 摘要

本文记录数据 v5 的首轮实验：把回归标签从「事件取整到最近帧」改成「按 MIDI 精确事件时间计算三角距离」，再训练 `hybrid_reg3` 并与 v4 的 `hybrid_reg2` 对比。读者不需要读过之前报告；读完后能知道 v5 标签改了什么、为什么没有带来提升、以及当前应该保留哪个模型。

## 1. 背景

v4 的回归目标用 `int(round(note.start × FPS))` 作为中心，三角峰落在最近整数帧，没有严格编码亚帧相位。v5 保留 v4 的 7 个分类通道，重算 `onset_reg` / `offset_reg`：

```text
g(frame) = 1 - abs(frame/FPS - event_time) / (J × 32 ms)
target = round(g × 255)
```

事件时间用 MIDI 精确秒数，不再取整。

## 2. 实验

| 项目 | 内容 |
|---|---|
| 数据 | `maestro_processed_v5`（CQT/pedal 硬链接 v4） |
| 模型 | 8 头模型，分类 6 + onset_reg/offset_reg |
| 训练 | 20 轮 / 71,160 步，offset-loss 2.0 + reg-loss 0.5 |
| 对比 | `hybrid_reg2`（v4，相同配置） |

## 3. 结果

### 3.1 留出集

| 指标 | hybrid_reg2（v4） | hybrid_reg3（v5） |
|---|---:|---:|
| note F1（base） | 0.4036 | 0.3922 |
| note F1（refined） | **0.4046** | 0.3917 |
| onset10（base） | 0.5240 | 0.5096 |
| onset10（refined） | **0.6600** | 0.6334 |
| onset20（refined） | **0.8311** | 0.8148 |

训练 val F1：v5 **0.8212**，v4 0.8187（v5 略高），但留出集反而更低。

### 3.2 MAESTRO test split（177 首）

| 指标 | hybrid_reg2 | hybrid_reg3 |
|---|---:|---:|
| onset F1 (50 ms) | 0.8701 | **0.8722** |
| onset F1 (30 ms) | 0.8530 | 0.8530 |
| onset F1 (20 ms) | **0.8119** | 0.8080 |
| onset F1 (10 ms) | **0.6264** | 0.6139 |
| note-with-offset F1 | **0.3314** | 0.3220 |

v5 只在 onset50 上略高 0.0021，其余关键指标（onset10、note）都更低。

## 4. 解释

- v5 的相位确实修对了：事件 1.008 s（帧坐标 31.51）在 v4 是 204/255，在 v5 变成 229/230；
- 但当事件接近两帧中点时，精确目标把峰拆成两个约 0.9 的值，而不是单个 1.0 峰；
- 回归头的局部峰更平，插值对噪声更敏感，**监督信号变弱**；
- 单次训练 + 略高的 val F1 + 更低的 test，说明 v5 的差异也可能包含随机波动/轻微过拟合，不能仅凭这一轮断定“相位标签一定无用”。

## 5. 下一步

| 方案 | 内容 | 目的 |
|---|---|---|
| A | **v5 + J=2/3** | 让三角更尖，保留相位同时提高峰的可检测性 |
| B | **v4/v5 混合目标** | 保留 v4 的 1.0 峰，同时往精确相位方向偏移 |
| C | **多种子复现** | 判断 0.392 vs 0.404 是真实退化还是训练波动 |
| D | 保留 v4 reg2，转向 offset/时值设计 | 当前最佳路线 |

当前结论：**hybrid_reg2 仍是高分辨率 best，v5 暂不替换。**

## 6. 相关链接

- 模型：`outputs/M1/hybrid_reg2/checkpoints/best.pt`、`hybrid_reg3/checkpoints/best.pt`
- 评测：`outputs/eval/hybrid_reg2_test.json`、`hybrid_reg3_test.json`
- 数据构建：`scripts/build_labels_v5.py`
- 队列：`scripts/v5_queue.sh`
