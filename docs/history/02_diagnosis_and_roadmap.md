# 钢琴转录项目：诊断报告与改进路线

审阅日期：2026-10-31
审阅对象：`E:\Docker\git\piano_practice`（对照其重构版 `E:\Docker\git\piano_transcriber`）
结论性质：讨论稿。按约定，本文不含实现代码，只给诊断、实验设计、路线与取舍。

## 摘要

本文是对 2026-10-31 版钢琴转录项目的诊断记录：解释「黑音符太多」的主因、模型侧的欠训练与 frame 标签被踏板污染两个独立问题，并给出改进路线与实验设计。读者不需要看过旧项目代码；读完后能理解当时每个判断的证据来源，以及为什么问题要拆开治。

---

## 2. 证据基础与可信度

本次没有跑推理：本机 Docker daemon 未启动，宿主 Python 只有 numpy/pandas/lxml/Pillow，没有 torch/librosa/music21。所有数字来自四类证据：静态代码审计、训练日志解析、已有产物解析（MusicXML/MIDI/npz）、MIDI 与音频头的独立复现。下表说明结论强度。

| 结论 | 强度 | 关键数字 |
|---|---|---|
| GT 直通记谱配方仍有 86.3% 黑音符 | 强：直接实验 | 24 首 GT，中位 86.3%，Chamber2 92.25% |
| 当前产物 75.4% 黑音符由排版制造 | 强：解析产物 | 1049 音符头 / 484 击键，tie 269 + 小节切割 72 |
| frame 标签含踏板延音 | 强：独立复现 | pedal 版与 npz 一致率 100.00%，no-pedal 版 97.12% |
| 训练只有 6200 步、LR 崩到底 | 强：日志 + 代码 | 31 步/epoch × 200，末 LR 2.384e-8 |
| 参数搜索 F1 不可引用 | 强：CSV + mir_eval 源码 + 算术 | 上限 0.264，实测 0.0444 |
| 每 epoch 只采样 11% 数据 | 强：代码 + 数据集统计 | 17.513 h / 159.24 h |
| max-seconds 用了 60 | 强推断 | 三首前 60 s GT 795/539/762，均值 698.67 吻合 n_ref=698 |
| 模型真实 onset-only F1 | **未验证** | 区间 [0.044, 0.308]，需实验 E1 测 |

详细审计报告在 `E:\Docker\dsh_scratch\piano_review\`（六份，含可复现脚本）。这些脚本只读，未改动仓库任何文件。

---

## 3. 现状全景

### 3.1 三个代码库

| 目录 | 时间 | 内容 | 判断 |
|---|---|---|---|
| `piano_practice` | 2026-01 至 05 | 扁平脚本：step0 数据工厂、train_full、inference_industrial、transcribe_readable | 你的学习主线，算法都在这里 |
| `piano_transcriber` | 2026-08-01 | 重构版：src/ 模块化、Dockerfile、doc/01–06、首次自动评测、参数搜索 | 工程骨架更好，但**算法逐字相同** |
| `piano_transcription` | 2025-11 下载 | ByteDance 高分辨率钢琴转录论文官方实现（含论文 PDF、models.py、losses.py、data_generator.py） | 现成的对照基线，很有用 |

一个必须知道的事实：两个仓库的 `model_best.pth` 的 SHA256 完全相同（78,626,869 字节，71 个张量）。对比 `src/transcribe.py` 与 `transcribe_readable.py`，35 个差异块全部是注释、常量搬迁、相对路径和格式化；状态机参数（0.40 / 0.06 / 0.18 / 6 / 10 / 0.15）逐字相同，仍然固定 120 BPM、0.25 网格、C4 硬分左右手、无踏板、music21.makeNotation。**重构版带来的是工程骨架和文档，不是算法进步。**

### 3.2 数据与权重资产

| 资产 | 规模 | 说明 |
|---|---|---|
| `E:\Docker\maestro_processed` | 1276 个 npz，17.34 GiB | train 962 / val 137 / test 177，文件级无泄漏 |
| 训练集总时长 | 159.67 h | 平均每首 18672 帧（约 10 分钟） |
| 检查点 | 41 个 × 74.98 MiB = 3.00 GiB | 磁盘上有 3 份副本，合计约 9 GiB |
| 原始数据 | `E:\Docker\maestro-v3.0.0` | 还在，可以重新预处理 |
| 官方参考实现 | `E:\Docker\git\piano_transcription` | 可作对照与预训练权重来源 |

### 3.3 当前数据流

```text
音频 (任意采样率)
  → librosa.load(sr=16000)
  → CQT 264 bins (fmin=A0, 3 bins/半音, hop 512 ≈ 32 ms/帧)
  → amplitude_to_db(ref=np.max)   # 每首按自身最大值归一，dB 范围恒为 [-80, 0]
  → CNN(2×conv + MaxPool(1,2)) → Linear(8448→512) → PE → 4 层 TransformerEncoder
  → 3 个头：onset / frame / velocity，各 (T, 88)
  → 滑窗平均 (window 2048, stride 1024)
  → find_peaks + 双阈值迟滞状态机 + 两轮谐波抑制 + 同音高合并
  → 音符事件 (pitch, start, end, velocity)
  → 固定 120 BPM + 0.25 四分音符网格 + C4 硬分左右手 + music21.makeNotation
  → MIDI + MusicXML
```

### 3.4 能力现状

- 检测偏稀疏：`readable_prediction_v2.mid` 只有 36 个音 / 30 s（1.2 音/s），`readable_prediction.musicxml` 484 击键 / 142 s（3.41 音/s）。参考 GT 密度中位 8.64 音/s（本次评测的三首是 7.6 音/s）。
- 但流水线本身是通的：过拟合小样本时 `prediction_level5_overfit.mid` 有 4996 个音，GT 是 4991——说明「CQT → 模型 → 状态机 → MIDI」这条链没有致命的工程 bug，瓶颈在训练与后段。
- Level 5 全量模型曾经「哑火」过一次（`prediction_level5.mid` 只有 56 字节、0 个音），最可疑原因是 `train_v5.py` 用 1000 帧训练、`export_midi_v5.py` 用 2048 帧滑窗推理，窗长不一致。这是训练/推理配置漂移的典型事故，后面做实验时建议一律用同一份常量模块（重构版的 `src/constants.py` 已经做对了）。

---

## 4. 「黑音符」到底怎么来的

### 4.1 归因表

| 机制 | 证据 | 对黑音符的贡献 |
|---|---|---|
| (a) 假阳性（多检测了音） | 当前产物同音高 50/100 ms 重触发为 0；击键密度低于 GT | 当前产物约 0；旧产物 `result_industrial` 才是主因（32.29 音/s，GT 的 3.8 倍） |
| (b) 碎音（同一个音被切成多段） | 模型原始 MIDI 无碎音；模型输出无重触发 | 当前产物约 0 |
| (c) 记谱配方 | GT 直通后中位 86.3% 黑音符 | **主因，约 75.4%** |
| (d) 量化/BPM 假设 | 100% 的 onset 和时值都钉在 0.25 拍；无 beat tracking；16 分音符里 69.4% 是 tie 尾巴 | 与 (c) 是同一件事的两个侧面 |

**结论：当前产物里黑音符主要是 (c)+(d)，检测精度不是主要矛盾。** 这个结论很重要，因为它把「要不要先去重训」这个问题顺序颠倒了：先把谱面修到「即使给完美输入也可读」，再去提升检测，才不会被记谱噪声掩盖掉模型的改进。

### 4.2 三条机理

**机理一：固定 120 BPM。** 120 BPM 下一拍 0.5 s，1/16 音符只有 125 ms。真实钢琴曲速度从 40 到 200 BPM 都有。曲子如果是 60 BPM，你的谱面相当于把节奏整体压缩了一倍。

**机理二：把按键秒数当时值。** 演奏的 note-off 是「手指离键」，不是乐谱上的时值。实测这三首 GT 的按键时值中位数只有 46 / 80 / 62 ms，而 1/16 网格是 125 ms——量化之后，**每一个按键都变成一个 16 分音符**。这不是模型的问题，是「用 note-off 当谱面时值」这个建模假设的问题。

**机理三：没有和弦化，没有声部规划。** 代码把每个音符直接 `insert(offset, note)` 进同一个 Part；同一时刻的多个音没有被合成和弦，`music21.makeNotation` 会把它们摊成多个声部。产物里实测每小节平均 8.65 个声部（最多 10 个）、1260 个休止符（占事件时长 67.65%）、9.77 个休止符每小节每声部。一个音头加一根 tie 是「视觉上的一根黑签」，一个声部一个声部地填休止符，就是满页黑点。

**实例**（20 秒测试文件里真实存在）：MIDI 中一个 2.62 s 的 D4，被排成 half + quarter + 16th + half 共 4 个头、3 条 tie、跨 2 小节。人写谱只需要一个附点二分音符。

### 4.3 为什么不能靠提高阈值压黑音符

历史上试过：`ms4_very_safe` 的策略把 4 拍压到只剩 1 个音（0.5 音/s）；`final_readable_score` 对 GT 的 recall 只有 0.116（漏掉 88% 的真音）。用提高 onset 阈值来减少黑音符，代价是把谱子掏空——因为你砍掉的是真音，而黑音符来自记谱。**阈值要按 F1 调，不能按「看起来黑不黑」调。**

另外发现一个具体的历史 bug：`final_readable_score.musicxml`（3 月那版）把「秒 = 四分音符」来写（f=1，60 BPM 语义），而当前 `seconds_to_quarters` 用的是 f=2。同一仓库存在两套时值口径。这一处口径修正就让黑音符头从 83.1% 降到 55.2%。建议以后统一走一个常量模块并加断言：**谱面总时长必须等于音频总时长**（允许 1 个量化网格内的误差）。

### 4.4 记谱层应该怎么做

按收益从高到低：

1. **同 onset 和弦化**：把落在同一量化位置（或 ±30 ms 内）的音符合成一个 `music21.chord.Chord`。这一步能直接消掉大部分「多声部 + 休止符汤」。
2. **先估计速度**：`librosa.beat.beat_track` 或对 onset 包络做自相关，得到 BPM 与拍点；网格改为「拍点相对位置 × 1/12 或 1/24」，而不是固定 120 BPM。
3. **时值重估**：谱面时值优先按「到下一个 onset 的间隔」和声部归属推断，按键时长只作为上限参考；踏板延音不进入谱面时值。
4. **声部规划**：先做 RH/LH 的音域与和弦连续性划分，再在每只手内部按音区/时值分配 2 个声部（不用 C4 硬切）。
5. **交给成熟引擎（可选但强烈建议先做）**：把干净 MIDI 丢给 MuseScore Studio 的命令行导入（`mscore -o out.musicxml in.mid -M midi_import_options.xml`），它的 MIDI 导入是自适应网格且有 `Is human performance` 选项，能处理人类演奏的不对齐。这可以作为「自研量化」的对照基线，也能让你先看到「排版对了以后谱子长什么样」。
6. **小节数与总时长一致性断言**：现在 RH 71 小节、LH 58 小节，长度不一致，渲染会错行。

---

## 5. 模型侧的真实问题（与黑音符是两个独立的病）

### 5.1 训练量

| 指标 | 数值 | 说明 |
|---|---|---|
| 总梯度步数 | 6200 | 31 步/epoch × 200 epoch |
| 每 epoch 采样量 | 962 个窗口 = 17.5 h | 训练集的 11.00% |
| LR 峰值区间 | epoch 5–6 | 之后按 0.95 每 epoch 衰减 |
| LR < 1e-6 的步数占比 | 36% | 等效冻结 |
| 末 LR | 2.384e-8 | 峰值的 0.0048% |
| 等效峰值更新步数 | 约 713 步 | ΣLR 只有恒定峰值的 11.5% |
| 总耗时 | 2.06 h | 4090 上（本机 4060 未参与训练） |

对照：Onsets and Frames 的训练量在 5 万步量级，本项目是它的 12%。**这不是「再跑 100 个 epoch」能补的**，因为 LR 已经衰减到零，继续跑等于原地不动。要么重设 LR schedule 从头训，要么把数据管线修好后重训。

### 5.2 数据与标签

| 问题 | 数字 | 影响 |
|---|---|---|
| frame 含踏板延音 | 正样本率 1.96% → 4.85%（8 首抽样） | 模型学「发声」不是「按键」，时值系统性偏长 |
| onset 目标展宽 | `[start-1, start+2)` = 96 ms，标准是 32 ms | onset 头学成平台，推理阈值连续触发 |
| 29.5% 的 onset 帧 frame=0 | 标签自相矛盾 | onset 与 frame 两个头互相打架 |
| 无任何数据增强 | 正式管线 0 处 | 对 nezha2 这类域外音频泛化差 |
| 短文件 np.pad 补 0 | dB 域 0 是最响 | 4 个文件受影响，静音被伪装成最响 |
| 验证集随机窗口 | 每 epoch 只看 12.79% | val loss 跨 epoch 不可比，「best」选在噪声上 |
| 输入未标准化 | dB 值直接进 Conv | 换录音电平即分布漂移 |
| MaxPool(1,2) 错位 | 264 → 132 = 1.5 bin/键，33.3% 的 pooled bin 跨半音 | 频率池化跨越相邻琴键，弱化音高判别 |

### 5.3 架构

- 只有 3 个头（onset / frame / velocity），缺 **offset 头**（时值）和 **pedal 头**（踏板）。ByteDance 那条线是 5 头。
- velocity 头几乎没学到东西：输出的力度曲线全曲几乎平在 0.2–0.3，只在 onset 处有小凸起；而且标签是绝对 MIDI 力度（除以 128），输入却被逐文件 `ref=np.max` 归一化，跨文件的绝对力度信息已经被抹掉。力度不作为第一优先级。
- PositionalEncoding 的 buffer 是 5000 行，训练/推理窗口 2048，只用 41%，却让每份检查点多存 9.77 MiB。
- 无 gradient clipping、无 EMA、用 Adam 而非 AdamW、无 early stopping；checkpoint 只存 state_dict，不可恢复训练；41 个中间档没有价值。

### 5.4 评测（最需要先修的）

| 问题 | 证据 |
|---|---|
| 口径与论文不可比 | `mir_eval.transcription.evaluate` 默认 offset_ratio=0.2，算的是 note-with-offset；论文报的 95.32 是 onset-only |
| 指标被时值硬约束压死 | 后处理强制最短 192 ms；三首 GT 时值中位 46/80/62 ms；<92 ms 占 84.7%/54.1%/71.5%；此口径理论上限 0.264 |
| 检测数上限 | 最优配置 n_pred=127 vs n_ref=698，全对也只有 F1 ≤ 0.308 |
| 参数区间没标定 | 三个参数最优值全在网格边界 |
| 默认参数不在网格 | 脚本里「相对默认」的分支永不执行 |
| 在 test split 上搜参 | 官方 test 被污染；validation 137 首有 npz 却没用 |
| 只用了三首最短最密的曲子 | 11.6 音/s，偏向极端 |
| 无 oracle 自检 | 没有任何「GT 当预测应得 1.0」的验证 |
| 丢弃了 11/14 项指标 | mir_eval 返回里本来就有 onset-only 与 offset 分解 |

注意：**不是数据错位**。用 WAV 32 ms RMS 包络与 MIDI onset 密度做互相关，最佳时移只有 +32/+64/+32 ms；npz 的 onset 事件数（877/855/1392）与 MIDI 音符数（882/860/1400）差不到 0.6%。评测链路的时间基是对的，错的是指标口径。

### 5.5 现在模型的真实水平

未知，但区间可估：note-with-offset F1 实测 0.0444 ≈ 结构上限 0.264 的 1/6，也 ≈ 检测上限 0.308 的 1/7；每行隐含的 TP 恒在 7.4–20.8 之间，n_pred 从 127 涨到 562 时 TP 不涨——说明模型「不是完全瞎」，但兑现率只有约 1/5。**必须先做 E1 实验把这个数测出来，否则后面所有改进都无法归因。**

---

## 6. 先做五个诊断实验（不做实验不要动手改代码）

每个实验都要求：固定输入、固定测试集、留 before/after 数字。

### E0：GT 直通记谱（30 分钟，本地 CPU）

- 目的：确认「黑音符是记谱问题」，并建立记谱层的 oracle。
- 做法：取 10–24 首 MAESTRO GT MIDI，分别走三条路：(i) 现有 `write_readable_musicxml`；(ii) MuseScore CLI 导入导出；(iii) 手工量化（拍点 + 和弦化）。
- 判据：黑音符头占比、每小节声部数、休止符占比、总时长一致性、音符头膨胀率。
- 教学点：把「模型好坏」与「记谱好坏」解耦，这是一切工程调试的基本功。

### E1：评测闭环自检（半天，本地 CPU/GPU 少量）

- 目的：让评测数字第一次变得可信。
- 做法：
  1. oracle 自检：GT 当预测 → 必须 1.0；GT 加 ±32/64 ms onset 抖动、±10% 时值扰动 → 得到敏感度曲线。
  2. 双口径：onset-only F1（`offset_ratio=None`，和论文可比）与 note-with-offset F1 分开报；frame F1 单独报。
  3. 固定测试集：MAESTRO test 取 5–10 首整曲（或固定 60 s 起点的同一段），缓存模型概率（每首只推理一次）。
  4. 时间对齐：模型输出相对 GT 有 +15–45 ms 系统延迟，比较前要估偏移；不校正时过拟合模型 precision 会从 0.999 掉到 0.565。
  5. 只用 validation 调参，test 只在最后报一次。
- 判据：oracle 自检为 1.0；同一份概率重复跑两次数字一致。
- 教学点：指标定义、TP 匹配、offset 容差；为什么「loss 下降」不等于「F1 上升」。

### E2：后处理上界（1 小时，本地 CPU）

- 目的：分离模型责任与后处理责任。
- 做法：把 GT 标签矩阵（onset/frame/velocity）直接喂进 `probabilities_to_events`，得到「后处理上界」；再与模型概率的结果对比。
- 判据：上界与实测的差距就是模型的贡献；若上界 ≤ 0.4，说明后处理本身有硬限制（比如 192 ms 下限），先改后处理。
- 教学点：误差分解——这是深度学习项目里最值钱的一项技能。

### E3：天花板参照（1–2 小时，本地 GPU）

- 目的：知道「同一段音频、同一套记谱代码」在近 SOTA 检测下能到什么水平。
- 做法：用 `piano-transcription-inference` 0.0.6（ByteDance 预训练，MIT 代码 + CC-BY-4.0 权重，16 kHz 输入，权重约 172 MB）推理同样的测试音频，输出 MIDI 后走你的记谱层。
- 判据：如果它的谱面仍然黑，问题 100% 在记谱；如果它的谱面干净，你的模型就是短板。
- 教学点：外部基线 + 迁移学习思路；也顺便验证「预训练权重 + 自己的后段」这条工程路线。

### E4：训练量对照（可选，autoDL 几小时）

- 目的：确认欠训练的程度。
- 做法：修好数据管线后，分别跑 6200 步（复刻现状）与 5 万步（正确 schedule），在同一测试集上报 onset F1。
- 判据：F1 差值；如果差不到 0.05，说明问题在标签而不在步数。
- 教学点：学习率调度与总步数的耦合；「训练不动了」到底是欠拟合、过拟合还是 LR 死了。

---

## 7. 改进路线

### P0：评测闭环 + 记谱层（半天到 1 天，本地，不用 GPU）

目标：让「谱面可读性」第一次不依赖模型。

- 修 `write_readable_musicxml`（当前 `transcribe_readable.py:286-327`，重构版 `src/transcribe.py:340-390`）：
  - 同 onset 和弦化；
  - 用 `librosa.beat.beat_track` 估计 BPM，网格改成拍点相对位置；
  - 时值按「下一个 onset 间隔」重估，不再直接用 `end-start`；
  - 每小节声部数上限（比如 2/手），左右手按音域连续性而不是 C4 硬切；
  - 补齐 RH/LH 小节数一致；加「谱面总时长 = 音频总时长」断言。
- 加一个 `midi_to_score.py` 的小工具（GT 或模型 MIDI 都能进），配 E0 的指标输出。
- 验收：GT 直通黑音符占比从 86.3% 降到 25% 以下；模型输出（当前权重）的谱面黑音符占比明显下降；两个声部小节数一致。
- 教学点：结构化数据建模；量化误差；「算法输出 → 人类可读格式」的工程转换。

### P1：后处理（1 天，本地 CPU）

目标：把「概率 → 音符」这一层从手调改成可验证。

- 捡回历史里更好的设计（`prototype/` 里有，正式管线没保留）：
  - 同帧复音上限 Top-K（v2 用的 7，GT 最大 6）；
  - frame > 0.85 时允许「救回」漏 onset 的持续音；
  - 最小击弦间隔回到 80 ms（当前 160 ms 会吞掉同音反复和颤音）；
  - 最短音长从 192 ms 降到 2 帧（64 ms），时值交给记谱层处理。
- 加硬约束：note-on 必须 `onset 峰` 与 `frame 支持` 同时成立（现在只要 onset 过阈值就开音）。
- 谐波抑制降级为条件式：只在「frame 不支持该键」时才压，不再对整首曲子全局按音高衰减（现在 C4 一旦被确认，全曲 C5 的 onset 都被乘 0.85）。
- 用 E1 的固定测试集做网格搜索，tune/holdout 分开。
- 验收：onset-only F1 提升；同音高 50 ms 重触发率、碎片率下降；GT 直通的谱面指标不退化。
- 教学点：阈值搜索方法学、过拟合到测试集、混淆矩阵式错误分析。

### P2：数据 + 训练（autoDL 4090，6–10 小时）

目标：把模型从「欠训练的玩具」推到「可用基线」。

数据（0.5 天 + 机器时间）：

- 把 1276 个 npz 重切成 10–20 s 段（或建段索引），解决每步解压整首歌的问题；
- 从 MIDI 重新生成标签：onset 改成 `[start, start+2)` 两帧；frame 二选一（见 §8.1）；额外生成 `pedal_roll`；
- 短文件填充改成 -80 dB；固定验证窗口；
- 可选：把 CQT 换成 log-mel 229 做 A/B（见 §8.4）。

模型（改动很小）：

- 加 offset 头 + pedal 头（4–5 头）；
- `MaxPool2d((1,3))` 让 264 → 88，与输出头对齐（现在是 (1,2)，33.3% 的 bin 跨半音）；
- 输入按数据集统计量标准化。

训练：

- AdamW + cosine + warmup，grad clip 1.0；pos_weight 按真实先验（onset 约 88、frame 约 14.5，可先取 30/10 起步）；
- 按 onset-only F1 选模型，不按 val BCE；
- 目标量级：5 万步以上，batch 32，10 s 段。

验收：onset-only F1 ≥ 0.80；note-with-offset F1 ≥ 0.45；pedal onset F1 ≥ 0.80。达不到就先做错误分析再改，不要盲目加层数。

教学点：数据管线的效率决定有效算力；标签不平衡；多任务学习；模型选择指标；可复现性。

### P3：浏览器应用（2–3 天）

目标：把「音频进、曲谱出」做成能给别人用的服务。

见 §9。验收：上传 ≤3 分钟音频 → 30 秒内返回 MusicXML/MIDI/PDF，页面能渲染、能播放、能下载；有中间可视化（CQT、onset/frame 热图）辅助教学。

---

## 8. 训练修复手册

### 8.1 标签口径：先做一个决定

当前 frame 标签 = 发声（含踏板）。两条路：

- **路线 A（省事）**：保留含踏板的 frame 标签，但加一个 pedal 头，让后处理知道「现在是踏板在延音」。谱面时值由 P0/P1 的记谱层重估，不依赖 frame 的 end。
- **路线 B（干净）**：frame 标签改成不含踏板的按键时长，踏板单独一个头；谱面时值 = 按键时长 + 记谱层修正。

建议先 A（改动小、能直接复用现有 npz 的 frame），把 B 作为对照实验。

### 8.2 数据管线

| 改动 | 为什么 | 预期收益 |
|---|---|---|
| 段索引（10–20 s） | 现在每步解压整首歌，单 epoch 只消费 11% | 同时间步数 ×5–10 |
| 固定验证窗口 | val loss 才可跨 epoch 比较 | best model 选择才有意义 |
| onset 改 2 帧 | 标签与 32 ms 帧率匹配 | onset 时刻更准 |
| 填充 -80 dB | 静音不能被伪装成最响 | 修一个语义 bug |
| 可选数据增强 | 增益抖动、±2 半音移调（CQT 移 3×n 个 bin）、加噪 | 域外泛化 |
| 队列/worker 调整 | 压缩比高，解压是 CPU 瓶颈 | 提高 GPU 利用率 |

注意：时间拉伸对 CQT 不能直接做（要用重采样重算 CQT），代价高，第一轮可以不做；增益抖动、移调、加噪声最便宜。

### 8.3 训练循环

- 优化器 AdamW（weight decay 1e-4），LR 3e-4，warmup 1000 步，cosine 到 1e-5；
- grad clip 1.0，AMP（现在已有 AMP，保留）；
- 分头记录 loss（现在只记录三项之和，看不出哪一头坏了）；
- 每 N 步在固定验证集上算 onset F1，按 F1 保存；
- checkpoint 只留 best + latest，存 optimizer/epoch/config，能续训；
- 固定随机种子，并确认 DataLoader worker 的随机性受控（Windows 上 np.random 不受 torch 种子控制）。

### 8.4 特征：CQT 还是 log-mel

你文档里「Mel → CQT」的理由（CQT 每个半音 3 bins、与钢琴物理对齐）是对的，但有一个事实需要纠正：**Onsets and Frames 和 ByteDance 高分辨率论文用的都是 log-mel（229 bins），不是 CQT**；O&F 的消融实验里把输入换成 CQT，Frame/Note/Note+Offset F1 从 78.30/82.29/50.22 掉到 73.07/76.38/41.14。另外一个物理事实：你的 CQT 最高只到 4350 Hz，瞬态与高次泛音信息被截掉了，而 onset 检测恰恰依赖这些。

建议：P2 里加一组 log-mel 229 的对照（30 分钟算特征 + 一次训练），用 E1 的指标比较。这是很好的教学实验：**不要只凭物理直觉选特征，要用对照实验说话**。如果时间紧，先不动 CQT，把 P0/P1 做完再说。

### 8.5 预算

| 阶段 | 机器 | 时间 | 说明 |
|---|---|---|---|
| 数据重切 + 标签重生成 | 本机 CPU | 0.5–2 h | 不用重算 CQT 的话很快；重算 CQT 约 2 h |
| 训练 5 万步 | autoDL 4090 | 4–8 h | 10 s 段、batch 32 |
| 评测（固定测试集） | 4090 | 20 min | 缓存概率后纯 CPU |
| 4060 本地小规模复现 | 本机 8 GB | 1–2 h | seq 1024、batch 4、grad accum 8 |

---

## 9. 浏览器应用方案

### 9.1 架构

```text
浏览器 (Vite + React + TypeScript)
  ├─ 上传音频 / 选择示例
  ├─ OSMD 或 alphaTab 渲染 MusicXML（含播放光标）
  ├─ 下载 .musicxml / .mid / .pdf
  └─ 教学面板：CQT 图、onset/frame 热图、钢琴卷帘 vs GT
        │ HTTP
        ▼
FastAPI (Docker)
  ├─ POST /api/transcribe  → 入队，返回 job_id
  ├─ GET  /api/jobs/{id}   → 状态 + 进度
  ├─ GET  /api/result/{id}/score.musicxml|.mid|.pdf
  └─ 单 GPU worker（asyncio / RQ），一次只跑一个推理
        │
        ├─ torch 模型推理（FP16，滑窗）
        ├─ 后处理 + 记谱（P0/P1 的代码）
        └─ MuseScore CLI + xvfb 生成 PDF
```

### 9.2 技术选型

| 环节 | 选择 | 理由 |
|---|---|---|
| 后端 | FastAPI + uvicorn | 轻、异步、Python 生态一致 |
| 任务队列 | 先用进程内单 worker，再上 RQ/Redis | 4060 单卡并发本来就只有 1–2 |
| 渲染 | OSMD 2.2.1（BSD-3） | MusicXML 渲染成熟；**本身不带播放** |
| 播放 + 光标 | alphaTab 1.8.4（MPL-2.0）或 Tone.js | alphaTab 一个库覆盖显示 + 播放 + 光标 |
| 排版 | MuseScore Studio 4.7.6 CLI | `-o` 是 converter 模式，可导 musicxml/mid/pdf；Docker 里需要 xvfb；量化用 `-M midi_import_options.xml` |
| 无头兜底 | Verovio 6.4.0（LGPL-3.0） | MusicXML → SVG，轻量 |
| 容器 | CUDA 基础镜像 + MuseScore + xvfb | 一个镜像包住模型与排版 |

### 9.3 接口设计（最小版）

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/transcribe` | multipart 音频，参数：是否要 PDF、是否保留中间图 |
| GET | `/api/jobs/{id}` | 状态、阶段（CQT / 推理 / 后处理 / 排版）、进度 |
| GET | `/api/result/{id}.musicxml` | 乐谱 |
| GET | `/api/result/{id}.mid` | MIDI |
| GET | `/api/result/{id}.pdf` | 排版后 PDF |
| GET | `/api/result/{id}/debug` | CQT PNG、onset/frame 热图、钢琴卷帘 |

限制：音频 ≤ 3 分钟、16 kHz 单声道（服务端转码）、单次请求排队、结果保留 1 小时。

### 9.4 风险与合规

- **许可**：MAESTRO v3.0.0 是 CC BY-NC-SA 4.0（非商业）。如果模型权重是从它训练出来的，公开提供服务有授权风险。建议：只做本地/学习/小范围分享；要公网开放就换成许可更宽松的数据集，或明确标注非商业用途。
- **算力**：4060 8 GB 一次只能跑一个推理；并发要排队。3 分钟音频的推理耗时目前未实测（E3 顺手测）。
- **质量预期**：P0/P1 完成后谱面「可读但需人工修」，不要承诺「直接出版级」。产品文案建议定位成「AI 初稿 + 人工编辑」。
- **隐私**：上传音频要声明保留策略。

### 9.5 最小可行版本（MVP）

先不做 PDF、不做播放、不做用户系统：上传 → 转码 → 推理 → 后处理 → 返回 MusicXML + MIDI + 一张 debug 图 → 前端用 OSMD 渲染。这个版本一天能做完，而且足够验证整条链路。

---

## 10. 负清单（不建议做）

1. **不要用提高阈值来压黑音符。** 会把谱子掏空（4 拍只剩 1 个音、recall 0.116）。
2. **不要 fine-tune 现有 checkpoint 就期待翻盘。** LR 已经衰到 1e-8，权重在退化解上；600 步 fine-tune 不会动。
3. **不要用 val BCE 选模型、也不要跨 epoch 比较随机窗口的 val loss。**
4. **不要在 test split 上搜参。** 用 validation 调，test 只报一次。
5. **没有 oracle 自检的 F1 一律不可信。**
6. **不要自研端到端 audio→MusicXML。** 没有公开 SOTA，且排版是另一个学科。
7. **不要在浏览器端跑模型。**
8. **不要用 MAESTRO 训练结果做商业服务。**
9. **不要继续用固定 120 BPM + 0.25 网格 + C4 硬切左右手。**
10. **不要只凭物理直觉选特征**（CQT vs mel 要做对照实验）。

---

## 11. 需要你拍板的三件事

1. **主仓选哪个？** 建议把 `piano_transcriber` 作为主仓（有 git、有模块化、有 eval 骨架），`piano_practice` 归档为学习历程，只保留 `doc/` 与 `prototype/`。否则后续改进会分裂成两份代码。顺带把 3 份检查点副本（9 GiB）清成一份，并补 `.gitignore`。

2. **记谱路线先走哪条？** 建议先用 MuseScore CLI 做基线（半天，能看到「排版正确时长什么样」），再自研量化对照。自研的好处是教学价值高、可控；坏处是一周内很难超过成熟引擎。

3. **重训什么时候做？** 建议顺序是 P0（评测）→ P1（后处理）→ P2（重训）。如果只想去 autoDL 体验一把，也可以先做 P2 里「数据重切 + 5 万步」的最小版本，但要先知道：**不修评测，你无法判断它到底变好没有。**

另外两个小决定：特征是否加 log-mel 对照（建议加）；浏览器应用是否现在做（建议 P1 之后做 MVP，不等 P2 训练完）。

---

## 12. 附录

### A. 审计报告索引

六份只读审计报告（含可复现脚本，未改动仓库）：

| 文件 | 内容 |
|---|---|
| `E:\Docker\dsh_scratch\piano_review\01_data_labels.md` | 数据与标签：npz 统计、踏板复现、CQT 口径、采样效率 |
| `E:\Docker\dsh_scratch\piano_review\02_training.md` | 训练日志 200 epoch 全曲线、检查点、架构审计 |
| `E:\Docker\dsh_scratch\piano_review\03_output_artifacts.md` | 14 个 MusicXML/MIDI 的黑音符拆解、GT 直通实验 |
| `E:\Docker\dsh_scratch\piano_review\04_prototype_history.md` | Level 1–5 演进、历代后处理算法、可捡回的设计 |
| `E:\Docker\dsh_scratch\piano_review\05_research.md` | 指标口径、开源基线、渲染方案、许可 |
| `E:\Docker\dsh_scratch\piano_review\06_current_state.md` | 重构版代码、eval.py 审计、参数搜索复核 |

### B. 指标口径速查

- `mir_eval.transcription.precision_recall_f1_overlap`：onset_tolerance 0.05 s、pitch_tolerance 50 cents、offset_ratio 0.2、offset_min_tolerance 0.05 s；`offset_ratio=None` 即 onset-only（论文口径）。
- `mir_eval.transcription.evaluate` 默认 offset_ratio=0.2，且返回 14 项指标（含 no-offset 版本），不要只取 3 项。
- 可引用的 MAESTRO test 数字：Onsets and Frames onset F1 95.32（无增广）/ 94.80（带增广）；ByteDance 高分辨率 onset F1 96.72、pedal onset F1 91.86。
- 谱面可读性指标建议固定为：黑音符头占比、每小节声部数、休止符占比、同音高 50 ms 重触发率、音符头膨胀率、RH/LH 小节数一致性、谱面总时长与音频时长的相对误差。

### C. 关键行号速查（原版 → 重构版）

| 逻辑 | `piano_practice` | `piano_transcriber` |
|---|---|---|
| CQT + dB 归一化 | `transcribe_readable.py:54-65` | `src/transcribe.py:58-66` |
| 滑窗推理 | `transcribe_readable.py:77-108` | `src/transcribe.py:79-110` |
| 状态机 | `transcribe_readable.py:143-238` | `src/transcribe.py:188-260` |
| 谐波抑制 | `transcribe_readable.py:124-140` | `src/transcribe.py:150-186` |
| 记谱 | `transcribe_readable.py:286-327` | `src/transcribe.py:340-390` |
| 模型 | `model_industrial.py:35-99` | `src/model.py:43-120` |
| 训练循环 | `train_full.py:31-133` | `src/train.py:40-140` |
| 数据加载 | `dataset_industrial.py:24-57` | `src/dataset.py` |
| 标签生成 | `step0_data_factory.py:36-66` | `src/preprocess.py:45-120` |
| 评测 | 无 | `src/eval.py`（口径需修） |

### D. 参考资料

- MAESTRO v3.0.0：https://magenta.tensorflow.org/datasets/maestro （CC BY-NC-SA 4.0）
- Onsets and Frames：https://arxiv.org/abs/1710.11153 （MAESTRO test onset F1 95.32）
- High-resolution piano transcription with pedals：https://arxiv.org/abs/2010.01815 （onset F1 96.72，pedal onset 91.86）
- 官方参考实现（本地已克隆）：`E:\Docker\git\piano_transcription`
- piano-transcription-inference（预训练基线）：https://pypi.org/project/piano-transcription-inference/
- MuseScore CLI 文档：https://man.archlinux.org/man/extra/musescore/mscore.1.en.txt
- OSMD：https://github.com/opensheetmusicdisplay/opensheetmusicdisplay
- alphaTab：https://www.alphatab.net/docs/introduction
- mir_eval.transcription：https://mir-eval.readthedocs.io/latest/api/transcription.html
- music21 Stream.quantize：https://www.music21.org/music21docs/moduleReference/moduleStream.html

---

## 附录 E. 关键可视化索引（后续补充）

本报告写于实验之前；后续 M0/M0.5 的实验图表已整理到 `doc/figs/`，建议对照阅读：

| 图 | 说明 |
|---|---|
| `figs/00_pipeline.png` | 四段流水线，以及「解码」一词两种含义的区别 |
| `figs/00_holdout.png` | 数据划分与开发集 / 留出集（dev / holdout）的用法 |
| `figs/10_decoders.png` | 四种解码器的精确率 / 召回率 / F1 对比 |
| `figs/11_case_study.png` | 一条泛音误检如何被条件式抑制与帧门控去掉 |
| `figs/12_suppression_map.png` | 抑制只作用于局部时间与音高，而非全局 |
| `figs/20_nezha2_old.png` | 旧管线的 nezha2 谱面（结构错乱） |
| `figs/21_nezha2_step1.png` | 只关谐波抑制后的谱面 |
| `figs/22_nezha2_step3.png` | 最终解码器的谱面 |
| `figs/30_gt_direct.png` | 标准答案直通新记谱（记谱层上限） |

---

本报告未修改任何代码。所有审计脚本与中间产物在 `E:\Docker\dsh_scratch\piano_review\`，可自行复跑核对。

## 结论（存档，2026-10-31）

## 0. 一句话

「黑音符太多」主要不是你训练简化造成的，而是**记谱层（概率 → 离散音符 → 五线谱）的配方问题**：把完全正确的 ground truth 喂进你现在的记谱代码，黑音符占比中位数仍有 86.3%。模型侧另有两个独立的病：**训练量只有应有规模的 1% 左右**，以及 **frame 标签被延音踏板污染**。这两件事要分开治。

---

## 1. 结论先行

1. **黑音符主因在记谱层，不在检测精度。** 用完全正确的 GT MIDI 走同一套 `write_readable_musicxml`，24 首 GT 的黑音符头占比中位数 86.3%（区间 40.9%–97.4%），Chamber2 是 92.25%。也就是说，检测做到 100% 正确，谱面照样满页黑音符。当前 2026-05 版产物里 43.09% 的黑音符头中，75.4% 是排版制造的碎片（tie 续接 269 个、小节线切断 72 个），模型真实短音只占 24.6%。

2. **机制是三条叠加**：固定 120 BPM 假设；1/16 网格（120 BPM 下只有 125 ms）；把「按键秒数」直接当谱面时值写。三者合起来，演奏者一次 50–80 ms 的按键必然被量化成一个 16 分音符，长音则被小节线和标准时值树切碎成多个音头加 tie。

3. **模型侧的真问题是欠训练，不是过拟合。** 200 epoch 只有 31 步/epoch，合计 6200 个梯度步；学习率从 5e-4 按 0.95 每 epoch 衰减，最后 36% 的步数里 LR 低于 1e-6（等于冻结），等效只有约 713 步峰值更新。末段 train 0.1179 / val 0.1249，gap 5.9% 且不扩大——是平台，不是过拟合。所谓 best model（epoch 160，val 0.1206）其优势只有 2.4 个标准差，跟验证噪声的最小值不可区分。

4. **标签口径有系统性偏差。** frame 标签来自 `pretty_midi.get_piano_roll`，默认会应用 sustain pedal：踏板按下时声音仍在响，所以 frame 标签实际是「发声」而非「按键」，8 首抽样上正样本率从 1.96% 涨到 4.85%（2.47 倍），踏板占时约 52.5%。发声连音中位 466 ms，而 MIDI 按键时长均值只有 218 ms。当前模型没有 pedal 头，只能隐式拟合这件事——这是 frame 概率抖动、后处理被迫用 10 帧 release 加 0.2 s 合并来兜底的根源。

5. **你目前唯一的量化评测（参数搜索 F1）不可引用。** 最高的 0.0444 有三个结构原因：（a）`mir_eval.transcription.evaluate` 默认按 note-with-offset 口径（offset_ratio=0.2）评分，而后处理强制所有音符时值不小于 192 ms，这三首 GT 的时值中位数只有 46/80/62 ms——即使 onset 全对，这个口径下的理论上限只有 0.264；（b）三组参数的最优值全在搜索空间边界（onset 0.20、frame_off 0.28、harmonic 0.6），说明搜索区间没有标定；（c）默认参数 (0.40, 0.18, 0.5) 根本不在网格里，脚本里「相对默认的提升」分支从未执行。此外没有 oracle 自检、在官方 test split 上搜参、只用了三首最短最密的曲子。

6. **正确的下一步顺序是：先修评测闭环，再修记谱层，再改后处理，最后才重训。** 前两件事在本地 CPU 上就能做完，半天到一天，收益立竿见影，且不需要 GPU、不需要重训。

7. **重训值得做，但必须连数据管线一起改。** 现在每个 epoch 只随机采样 962 个窗口（占训练集 159.67 h 的 11.00%），而每个样本都要解压整首歌的 npz（平均 19.7 MB）——算力大量耗在 I/O 上。改成 10–20 秒的段索引之后，同样的 6–10 小时在 4090 上能跑到 5 万步以上（现在的 8 倍），这才是「去 autoDL 跑几个 epoch」能看见进步的姿势。

8. **「一个音频进、人可读曲谱出」的最省力路径是两段式**：audio → MIDI（你自己的模型）→ 成熟排版引擎（MuseScore CLI 或自研量化）→ MusicXML → 浏览器渲染。端到端 audio→score 目前没有公开 SOTA 基准（最接近的 MIDI2ScoreTransformer 输入仍是演奏 MIDI），不建议自己造这条路。

---
