# 06 阶段程序说明（M0 / M0.5 / M0.6 / M1）

## 1. 为什么把它们放进仓库

这些程序是**实验过程**，不是产品代码。放进仓库有三个理由：

1. **可追溯**：每个结论（「解码器是主因」「标签宽 3 帧」「pilot 0.539」）都能顺着脚本与报告查到原始数字；
2. **可复跑**：答辩或换机器时，能按同样步骤重跑关键实验；
3. **不污染产品代码**：它们放在 `experiments/stage_code/`，不进 `src/`，
   也不要求 CI 通过——产品代码只有 `src/piano2/` 与 `scripts/`。

## 2. 目录约定

| 目录 | 内容 |
|---|---|
| `experiments/stage_code/m0/` | M0（诊断、评测链路、探针）与 M0.5（解码器实验）、M0.6（门禁）的程序 |
| `experiments/stage_code/m1/` | M1（数据 v2、模型 v2、训练、pilot）的程序 |
| `experiments/reports/` | 各阶段报告（Markdown） |
| `experiments/artifacts/` | 关键图表、训练日志、summary、manifest、label_check |
| `experiments/registry/` | 实验登记表（EXP-000 基线） |

## 3. M0：诊断与评测链路

| 程序（stage_code/m0） | 作用 | 关键产物 |
|---|---|---|
| `run_m0.py` | 主流程：oracle 自检 → GT 直通 v0/v1 → 模型 vs GT（三种解码）→ nezha2 出谱 | `outputs/M0/report.md`、对比图 |
| `prob_cache.py` | 把模型概率缓存到磁盘，调参时只做解码、不重复推理 | `cache/*.npz` |
| `_probe_decode.py` | 复现「全局谐波抑制吃掉 80% 音符」的对照实验 | 探针日志 |
| `_probe_probs.py` | 概率分布与阈值扫描（证明检测本身没问题） | 直方图 |

关键结论（详见 `history/04_m0_results.md`）：

- 「黑音符太多」主要不是模型欠训练，而是记谱层配方问题（GT 直通仍 86.3%）；
- 训练量只有应有规模的约 1%，frame 标签被踏板污染；
- 旧解码器的全局谐波抑制是召回率崩掉的主因。

## 4. M0.5：解码器重构（三个步骤 + 归因）

| 程序（stage_code/m0） | 步骤 | 作用 | 产物 |
|---|---|---|---|
| `m05_step1.py` | 步骤 1 | 只关谐波抑制，看 nezha2 谱面前后对比 | `step1_report.md` |
| `m05_step2.py` | 步骤 2 | 状态机参数网格搜索 | `step2_report.md`、`step2_best_config.json` |
| `m05_step2b.py` | 步骤 2b | 三种解码器对照（旧 / 调参状态机 / 峰值+帧） | 对照表 |
| `m05_step3.py` | 步骤 3 | 条件式谐波抑制初版 | `step3_grid.csv` 等 |
| `m05_step3b.py` | 步骤 3b | 开发集/留出集修正 + 扩展网格 | `step3b_grid.csv`、`step3b_final.json` |
| `m05_step3_finalize.py` | 收尾 | 归因实验（门控/抑制单独与组合）、案例图、nezha2 最终谱 | 最终配置与图 |
| `make_figures.py` | 配图 | 早期图表 | 图 |
| `make_all_figures.py` / `make_onset_figure.py` / `plot_style.py` | 配图 | 文档用的全部配图（中文、可复现） | `docs/figs/*.png` |

关键结论（详见 `history/06_m05_discussion.md`）：

- 解码器从 F1 0.196 → 0.952（稀疏曲目）；密集曲目 0.166 → 0.798；
- 密集曲目精确率 0.858（目标 0.85 达标）；留出集 0.835（差 0.015，留到 M3 解决）；
- 调试过程记录了 15 个「坑」（`history/07_debug_log.md`），包括我判断错的地方。

## 5. M0.6：训练前门禁

| 程序（stage_code/m0） | 作用 | 产物 |
|---|---|---|
| `preflight.py` | G1–G8 自动检查：数据完整性、划分泄漏、特征一致性、评测 oracle、解码天花板、基线登记、资源预算、可复现性 | `preflight_report.json`、`preflight_summary.md`、`registry/EXP-000_baseline.json` |

关键结论（详见 `history/08_training_readiness.md`）：

- 6 PASS / 1 WARN / 1 FAIL；
- 唯一 FAIL 是 onset 标签宽 3 帧（M1 已修复）；
- G7 实测：0.082 秒/步、1.63 GB 显存 → 7.1 万步在 4060 上约 1.65 小时；
- **训练比预想便宜**，真正的风险是数据正确性。

## 6. M1：数据 v2、模型 v2、pilot

| 程序（stage_code/m1） | 对应新仓位置 | 作用 |
|---|---|---|
| `build_data_v2.py` | `scripts/build_data.py` | 构建数据 v2（memmap CQT + 标签通道 + 段索引 + manifest） |
| `dataset_v2.py` | `src/piano2/dataset.py` | 段数据集（memmap 随机切片，固定验证段） |
| `model_v2.py` | `src/piano2/model.py` | 五头模型（MaxPool 1×3、输入归一化、踏板头） |
| `train_v2.py` | `src/piano2/train.py` | 训练循环（按 F1 选模型、可续训、`--dry-run`） |
| `check_labels.py` | `scripts/check_labels.py` | 标签宽度自检（onset2 = 2 帧） |
| `eval_old_model.py` | —（一次性对照） | 旧模型在新验证段上的基线（F1 0.738） |
| `make_m1_figures.py` | —（一次性配图） | `docs/figs/13_m1_pilot.png` |

关键结论（详见 `history/09_m1_data_v2_and_pilot.md`）：

- 数据 v2 33.12 GB，标签宽度修复到位（onset2 均值 2.007 帧）；
- pilot：1000 步 85 秒，验证 F1 0.353 → 0.539；
- A/B：1 帧 0.582 / 2 帧 0.539 / 3 帧 0.400（单种子，趋势）；
- 旧模型基线 0.738，pilot 尚未超过（只用了 1.4% 的训练量）。

## 7. 怎么跑这些阶段程序

它们依赖**旧仓库**的路径与缓存（`E:\Docker\git\piano_practice\outputs\`、`trained_data\`、
旧数据 `/Docker/piano2_data/processed/maestro_processed`），所以：

- 想复跑某个阶段实验：切到旧仓库 `m0/`、`m1/` 下按当时的命令跑（README 里有）；
- 只想看结论：读 `experiments/reports/` 与 `history/`；
- 想用最终版代码做同样的事：用新仓的 `scripts/`（`build_data.py`、`transcribe.py`、`evaluate.py`）。

新仓**不**保证这些阶段脚本能直接运行（路径依赖旧环境），这是有意为之：
产品代码与实验代码分离，避免「实验脚本混进产品」。
