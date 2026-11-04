# piano2：钢琴音频 → 可读乐谱

输入一段钢琴独奏录音，输出 MusicXML 乐谱（可渲染 PNG/PDF）。

本项目是「钢琴自动转谱」项目的最终版仓库：M0（诊断）→ M0.5（解码器重构）→ M0.6（训练前门禁）
→ M1（数据 v2 与训练脚本）→ **M2 建仓（当前）** → M3 全量训练 → M4 浏览器应用。

## 现状

| 部分 | 状态 |
|---|---|
| 解码器（概率图 → 音符事件） | 定型：onset F1 0.952 / 0.943 / 0.798（稀疏/中等/密集），留出集 0.821 |
| 记谱（事件 → MusicXML） | v1.1：黑音符占比 0.724、左右手小节一致、时长误差 -2.83% |
| 模型与训练 | 五头模型 + 训练脚本 + 本地 pilot 通过（1000 步 F1 0.539，仍在上升） |
| 评测 | oracle 自检 + 固定评测集 + 实验登记表 |
| 全量训练（4090） | **待做（M3）** |
| 浏览器应用 | 待做（M4） |

## 快速开始

```bash
# 安装（容器内或本机虚拟环境）
pip install -e .

# 单元测试（10 个测试；无数据/无 GPU 时自动跳过依赖项）
python -m pytest -q

# 音频 → 乐谱
python scripts/transcribe.py --wav /path/to/audio.wav --out-dir outputs/transcribe

# 固定评测集评测
python scripts/evaluate.py --model /path/to/checkpoint.pt
```

数据与模型权重不随仓库分发，路径用环境变量指定（见 `src/piano2/constants.py`）。

## 目录

| 目录 | 内容 |
|---|---|
| `src/piano2/` | 产品代码：特征、模型、训练、解码、记谱、渲染、评测 |
| `scripts/` | 命令行入口：`build_data.py`、`transcribe.py`、`evaluate.py`、`check_labels.py` |
| `tests/` | pytest 测试 |
| `docs/` | 产品与技术参考：总览、数据、模型训练、解码、记谱评测、阶段程序、路线图、M4 应用设计；`docs/history/` 是历史文档 |
| `tutorial/` | **教学与入门**：从声音到乐谱的每一步、图解、常见问题、应用使用教程 |
| `experiments/` | 各里程碑的阶段程序、报告、产物与实验登记表 |

## 文档

- [`docs/01_overview.md`](docs/01_overview.md)：项目总览与流水线
- [`docs/02_data.md`](docs/02_data.md)：数据 v2 设计与重建
- [`docs/03_model_training.md`](docs/03_model_training.md)：模型结构与训练策略
- [`docs/04_decoding.md`](docs/04_decoding.md)：解码器原理、参数与效果
- [`docs/05_notation_eval.md`](docs/05_notation_eval.md)：记谱、渲染与评测口径
- [`docs/06_stage_programs.md`](docs/06_stage_programs.md)：M0/M0.5/M0.6/M1 程序说明
- [`docs/07_roadmap.md`](docs/07_roadmap.md)：M3 训练与 M4 应用计划
- [`docs/08_m4_app.md`](docs/08_m4_app.md)：M4 浏览器应用（Docker 本地交付）设计
- [`docs/00_glossary.md`](docs/00_glossary.md)：术语表

## 教程

- [`tutorial/README.md`](tutorial/README.md)：教程目录与阅读路径
- [`tutorial/02_features_cqt_hop_frames.md`](tutorial/02_features_cqt_hop_frames.md)：CQT、hop、帧、T
- [`tutorial/03_labels_from_midi.md`](tutorial/03_labels_from_midi.md)：6 个标签通道与踏板
- [`tutorial/06_faq.md`](tutorial/06_faq.md)：常见问题（黑音符、holdout、PUA、训练成本）
- [`tutorial/07_running_app.md`](tutorial/07_running_app.md)：浏览器应用使用教程（M4 完成后补全）

## 重要说明

- 训练数据 MAESTRO v3.0.0 采用 CC BY-NC-SA 4.0 许可，**本项目仅用于非商业用途**；
- 解码器与记谱层在 M0/M0.5 已证明不是主要瓶颈，改动顺序见 `docs/01_overview.md` 第 6 节。
