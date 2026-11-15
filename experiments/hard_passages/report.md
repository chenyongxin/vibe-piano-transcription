# 难段落 + 音画同步诊断报告（v1）

## 摘要

本报告用 6 个固定的 MAESTRO 难片段，把当前 `hybrid_reg2` 的失败拆成三层：

- **检测层**：快速重复、快速跑动、踏板密集时 onset 检测下降；
- **时值层**：`note-with-offset F1` 远低于 onset F1，主要问题是 offset/时值而不是音符完全没检到；
- **记谱/同步层**：写成 MusicXML 后的音符数没有变少（连音/拆分反而增加），但量化误差和音画时间轴仍有残差。

结论是：**当前最值得先修的是时值/offset 和 timing 映射，而不是继续盲目加 note 检测容量**；快速装饰音和 tuplets 需要单独的记谱策略。

## 1. 目的与范围

本报告回答三个问题：

1. 当前模型在快速/装饰/踏板密集段落里，音符是没检到，还是时值不对？
2. 从 raw 事件到 MusicXML，音符数量是减少还是增加？
3. 音画同步的误差量级有多大，下一轮应该先修哪一层？

不包含：

- 主观听感评分；
- 浏览器端真实高亮延迟测量（本轮只使用量化残差作为代理，见第 5 节）；
- 训练新模型。

## 2. 数据

`manifest.json` 固定了 6 个 8 秒片段：

| 类别 | 来源 | 选择依据 |
|---|---|---|
| fast_repeats | MAESTRO test | 同音快速重复 |
| fast_runs | MAESTRO validation | 极密集快速跑动 |
| arpeggios | MAESTRO validation | 快速琶音音型 |
| grace_notes | MAESTRO validation | 大量短音/装饰音 |
| triplets | MAESTRO validation | 近似三等分连续音 |
| pedal_dense | MAESTRO validation | 踏板覆盖 96% 的密集织体 |

路径全部使用 `$DATA_ROOT` 占位符；重新生成：

```bash
cd "$REPO_ROOT"
python experiments/hard_passages/select_clips.py
```

## 3. 方法

每个片段：

1. 读取对应 MAESTRO WAV 的 8 秒片段，计算 32 ms CQT；
2. 用 `hybrid_reg2` 做 sliding-window 推理；
3. 分类头解码音符，`onset_reg` 修正起点，分类 offset 定结束；
4. 与 GT MIDI 用 `mir_eval` 比较：
   - onset F1 @ 50 / 20 / 10 ms；
   - note-with-offset F1（offset ratio 0.2，min tolerance 50 ms）；
5. 把解码事件写成临时 MusicXML，数 pitched note 数量；
6. 估算事件映射到 1/4 拍网格的残差。

运行：

```bash
python experiments/hard_passages/evaluate_hard_passages.py \
  --model models/best.pt \
  --out experiments/hard_passages/results.json
```

## 4. 结果

| 类别 | GT 数 | 预测数 | raw→XML 音符 | onset50 F1 | onset10 F1 | note F1 | 短音召回 | offset 误差中位数 | 量化残差中位数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fast_repeats | 98 | 68 | 68→74 | 0.759 | 0.542 | 0.241 | 0.641 | 115.9 ms | 26.3 ms |
| fast_runs | 316 | 249 | 249→254 | 0.694 | 0.414 | 0.368 | 0.665 | 45.0 ms | 35.0 ms |
| arpeggios | 102 | 97 | 97→105 | 0.764 | 0.442 | 0.221 | 0.724 | 93.9 ms | 25.1 ms |
| grace_notes | 263 | 255 | 255→268 | 0.884 | 0.571 | 0.120 | 0.869 | 147.7 ms | 31.4 ms |
| triplets | 110 | 106 | 106→119 | 0.944 | 0.676 | 0.574 | 0.926 | 41.5 ms | 27.4 ms |
| pedal_dense | 221 | 195 | 195→209 | 0.692 | 0.423 | 0.317 | 0.479 | 30.1 ms | 24.8 ms |

聚合：

- onset50：**0.779**（TP 810 / GT 1110 / 预测 970）
- onset10：**0.495**
- note-with-offset：**0.293**（offset 窗口修复后；修复前 0.207）
- 短音召回：**0.746**

> 这些片段是刻意挑的难点，不是随机 MAESTRO 分布；不要把它们和 test split 的 0.870/0.626/0.331 直接比较。

## 5. 失败归因

### 5.1 检测层：漏检和“过检”同时存在

- `fast_repeats`、`fast_runs`、`pedal_dense` 的 onset50 只有 0.69–0.76；
- 预测数普遍少于 GT，说明快速同音/密集织体中有真实漏检；
- 同时 precision 也不是 1，说明某些谐波/重复被误检。

### 5.2 时值层：note F1 低的主因不是“没检到”

- onset50 明显高于 note F1（例如 grace_notes 0.884 vs 0.077）；
- 说明大量音符起点检到了，但 **结束时间/offset 不匹配**；
- offset 误差中位数约 30–148 ms，p90 仍可达数百毫秒；旧版 160 帧窗口时中位数 50–222 ms；
- 这直接解释了为什么回放听起来“有音”，但谱面时值和量化后的节奏不对。

**优先级判断：时值/offset 是第一优先级，不是继续增加 onset 检测。**

已在 0.2.5 把 offset 搜索窗口从 160 帧缩到 20 帧、阈值提高到 0.7；
holdout note F1 从 0.4046 提升到 0.4533，难段落聚合从 0.212 提升到 0.293。
完整 test split 验证进行中。

### 5.3 记谱层：不是“谱面把音吃掉了”

- 6 个片段的 MusicXML pitched note 数都**大于或等于 raw 事件数**；
- 增加来自跨小节/超长音的拆分、连音和记谱时值拆分；
- 同音高、同起点的重复事件会被去重；同时起音会合并成和弦；
- 这些会让“谱面看起来少一些”，但本轮没有发现整体数量丢失。

如果用户感觉谱面少音，更可能是：

- 同音重复/和弦合并；
- 量化把相邻音吸到一起；
- 长曲目超过 20 页；
- 或某类装饰音在模型层已经漏检，而不是写谱层。

### 5.4 音画同步层

本轮量化残差中位数约 **25–35 ms**，p90 约 **40–62 ms**；这是谱面时间的下界误差。

需要注意：本轮**没有直接测量浏览器高亮延迟**。当前 app 的同步机制是：

- 播放使用原始音频时间轴；
- 谱面使用量化时间轴；
- 通过 verovio timemap 做全局比例映射；
- 每 0.04 s 采样一次当前元素。

剩余误差主要来自：

- 全局比例映射无法吸收局部 rubato；
- 0.04 s 采样粒度；
- 浏览器音频和渲染延迟；
- 快速段落的小节边界跨越；
- 长曲目累计漂移。

下一轮应加一个 **UI timing instrumentation**：记录每个 measure 边界对应的音频时间与高亮时间，输出误差分布。

## 6. 结论与下一步

### 结论

1. **检测不是唯一瓶颈**：onset50 在 0.69–0.94；
2. **时值/offset 是当前最大瓶颈**：note F1 只有 0.077–0.482；
3. **写谱层没有整体丢音**：raw→XML 音符数反而增加；
4. **量化残差约 25–35 ms**，是音画不同步的重要下界；
5. 快速重复、踏板密集还叠加检测漏音。

### 建议下一步

| 优先级 | 任务 | 原因 |
|---|---|---|
| 1 | 改进 offset/时值：offset 头、offset_reg 或时值后处理 | note F1 低的主因 |
| 2 | 加入 UI 音画误差埋点，测量真实高亮延迟 | 把“感觉不同步”变成可测数据 |
| 3 | 局部拍点/小节映射，替代全局线性缩放 | 解决长曲目/fast passage 漂移 |
| 4 | 三连音/装饰音/琶音的专门记谱规则 | 检测到但记谱层表达不出 |
| 5 | 针对快速同音/踏板密集的检测增强 | onset50 仍低 |

## 7. 复现与文件

```text
experiments/hard_passages/
  select_clips.py
  manifest.json
  evaluate_hard_passages.py
  results.json
  report.md
```

所有路径都用 `$REPO_ROOT` / `$DATA_ROOT` 占位符；不包含本机绝对路径。
