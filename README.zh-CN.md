# 钢琴转录

**[English](README.md) | [简体中文](README.zh-CN.md)**

把独奏钢琴录音变成可编辑的乐谱，全程本地运行。

![钢琴转录 Web 应用](docs/readme/home-zh.png)

## 功能

- 上传 WAV / MP3 / M4A / FLAC，最多 5 首、单首 ≤ 500 MB；
- 混合模型把音频解码成音符事件，再生成可读乐谱；
- 结果页逐页查看，播放时自动跟随并高亮当前小节；
- 真实钢琴音色回放，支持踏板（CC64）；
- 下载 **MusicXML**、**PDF**、**MIDI**。

![结果页与谱面查看器](docs/readme/result-zh.png)

## 工作流程

```text
音频
  → log-CQT（32 ms 一帧，264 bins）
  → CNN + Transformer
  → 8 个头：onset / frame / velocity / offset / pedal / pedal_on /
            onset_reg / offset_reg
  → 解码成 NoteEvent(pitch, start, end, velocity)
  → MusicXML / PDF / MIDI / 钢琴回放
```

模型用**分类头**判断音符是否存在、何时结束，用 **onset 回归头**在帧内修正起音位置。
原始音符事件用于事件级指标和回放；乐谱会量化到可读的节奏网格（16 分音符）。

## 结果

MAESTRO v3.0.0 test split（177 首，原始音符事件）：

| 指标 | 混合模型 |
|---|---:|
| note-with-offset F1（key-off 口径） | **0.379** |
| onset F1 @ 50 ms | 0.870 |
| onset F1 @ 20 ms | 0.812 |
| **onset F1 @ 10 ms** | **0.626** |

10 ms onset F1 为 **0.626**，分类基线为 **0.507**。offset 结束窗口修复（阈值 0.7、搜索 20 帧）
把 test split 的 note-with-offset F1 从 **0.331** 提升到 **0.379**，onset 指标不变；
holdout 单曲达到 10 ms **0.660**、note F1 **0.453**。

作为参考，ByteDance 的钢琴转录模型在三段 validation 上达到 onset F1 @10 ms **0.901**、
note F1 **0.444**。它的 note-off 更接近声学延音/踏板释放，因此 note F1 与我们的
key-off 口径不能直接对比。

## 快速开始

前置条件、系统依赖、本地安装、CPU/GPU Docker、模型权重放置和排错见
**[INSTALL.zh-CN.md](INSTALL.zh-CN.md)**。

### 本地 Python

```bash
python -m pip install -e .

# 把训练好的权重放到 models/best.pt，或设置 PIANO2_CKPT：
#   export PIANO2_CKPT=/path/to/checkpoint.pt

python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

浏览器打开 <http://localhost:8000>。

真实钢琴回放需要 `fluidsynth` 和 SoundFont（如 `FluidR3_GM.sf2`）；
缺少时自动回退到轻量合成器。

### Docker：CPU 与 GPU 两个版本

仓库提供两个 Dockerfile：

| 文件 | 适用 | 启动 |
|---|---|---|
| `Dockerfile.cpu` | 无 NVIDIA GPU 的笔记本/服务器 | `docker build -f Dockerfile.cpu -t vibe-piano-transcription:cpu .` |
| `Dockerfile.gpu` | NVIDIA CUDA（RTX 4060 等） | `docker build -f Dockerfile.gpu -t vibe-piano-transcription:gpu .` |

```bash
# CPU
docker run --rm -p 8000:8000 \
  -v "$PWD/models:/workspace/models" \
  vibe-piano-transcription:cpu

# GPU
docker run --rm --gpus all -p 8000:8000 \
  -v "$PWD/models:/workspace/models" \
  vibe-piano-transcription:gpu
```

> **平台说明**：GPU 镜像需要 NVIDIA container runtime 和 NVIDIA GPU。
> macOS 的 Docker 不能把 Apple GPU（Metal）或 NVIDIA GPU 透传给 Linux 容器；
> Intel Mac 和 Apple Silicon Mac 都应使用 `Dockerfile.cpu`。
> 在 macOS 上执行 `docker run --gpus all` 会直接失败。如果 Apple Silicon 上某个依赖
> 没有 arm64 wheel，可用 `--platform linux/amd64` 构建（会走模拟、较慢），不要改用 GPU 镜像。

也可以用 compose 的 profile：

```bash
docker compose --profile cpu up --build
# 或：docker compose --profile gpu up --build
```

源码构建的镜像需要把 checkpoint 挂载到 `models/best.pt`，或设置 `PIANO2_CKPT`。
GitHub Release 发布的镜像已经包含权重，不需要挂载。

如果要在源码构建时把 Release 权重一起打进去：

```bash
docker build -f Dockerfile.gpu \
  --build-arg MODEL_URL=https://github.com/chenyongxin/vibe-piano-transcription/releases/download/v0.3.0/best.pt \
  -t vibe-piano-transcription:gpu .
```

Release 镜像由 [`.github/workflows/release.yml`](.github/workflows/release.yml)
发布到 GitHub Container Registry（GHCR，GitHub 容器镜像库）：

```bash
# CPU，无需挂载模型
docker run --rm -p 8000:8000 \
  ghcr.io/chenyongxin/vibe-piano-transcription:v0.3.0-cpu

# GPU，无需挂载模型
docker run --rm --gpus all -p 8000:8000 \
  ghcr.io/chenyongxin/vibe-piano-transcription:v0.3.0-gpu
```

`:cpu` 和 `:gpu` 是最近一次发布的可移动别名。

## 复现模型参数

当前 app 使用 `hybrid_reg2`（数据 v4 + offset-loss 2.0 + reg-loss 0.5）。

```bash
# 1. 从 https://magenta.tensorflow.org/datasets/maestro 下载 MAESTRO v3.0.0，解压到：
#    data/raw/maestro-v3.0.0/maestro-v3.0.0.csv, *.midi, *.wav
export PIANO2_MAESTRO_ROOT="$PWD/data/raw/maestro-v3.0.0"

# 2. 构建 CQT + 标签（脚本默认使用容器内的数据路径；
#    换机器时用环境变量覆盖）
export PIANO2_DATA_ROOT="$PWD/data/processed/maestro_processed_v2"
export PIANO2_V2_ROOT="$PWD/data/processed/maestro_processed_v2"
python scripts/build_data.py --workers 8

export PIANO2_V3A_ROOT="$PWD/data/processed/maestro_processed_v3a"
python scripts/build_labels_v3a.py --workers 8
export PIANO2_V3B_ROOT="$PWD/data/processed/maestro_processed_v3b"
python scripts/build_labels_v3b.py --workers 8
export PIANO2_V4_ROOT="$PWD/data/processed/maestro_processed_v4"
python scripts/build_labels_v4.py --workers 8

# 3. 训练 hybrid_reg2（20 轮 / 71,160 步；RTX 4060 约 1.6 小时）
export PIANO2_DATA_ROOT="$PWD/data/processed/maestro_processed_v4"
python -m piano2.train --max-steps 71160 --batch 16 --val-every 500 \
  --save-every 200 --val-segments 120 --num-workers 4 \
  --run-name hybrid_reg2 --label-channel onset1 \
  --offset-loss 2.0 --reg-loss 0.5 --seed 0

# 4. 在 MAESTRO test split 上评测
python scripts/evaluate_hybrid_test.py \
  --model outputs/M1/hybrid_reg2/checkpoints/best.pt \
  --out outputs/eval/hybrid_reg2_test.json
```

训练产物在 `outputs/M1/hybrid_reg2/checkpoints/best.pt`；复制到 `models/best.pt`
后 app 就会加载它。完整实验和消融见
[训练报告](training/2026-11-11_duration_offset_hybrid.md)。

## 项目结构

| 路径 | 内容 |
|---|---|
| `app/` | FastAPI 后端、任务队列、谱面渲染、静态前端 |
| `src/piano2/` | 特征、模型、训练、解码、记谱、评测 |
| `tutorial/` | 从声音到乐谱的逐步教程 |
| `docs/` | 架构、数据、解码、记谱、应用设计 |
| `training/` | 实验报告与结果 |
| `scripts/` | 命令行入口与评测脚本 |
| `tests/` | pytest 测试 |

## 文档

- [安装指南](INSTALL.zh-CN.md)
- [App changelog](docs/09_app_changelog.md)
- [模型与转录局限](docs/10_limitations.md)
- [难段落失败归因报告](experiments/hard_passages/report.md)
- [教程目录](tutorial/README.md)
- [项目总览](docs/01_overview.md)
- [数据与标签](docs/02_data.md)
- [模型与训练](docs/03_model_training.md)
- [解码](docs/04_decoding.md)
- [应用设计](docs/08_m4_app.md)
- [实验报告](training/README.md)

## 局限

详细的模型、记谱与音画对齐限制见 [docs/10_limitations.md](docs/10_limitations.md)。

- 只面向**独奏钢琴**；混音、合奏不在范围内。
- 乐谱是可读的节奏表示，不是毫秒级演奏记录。亚帧时间保留在原始事件和回放中，
  但 MusicXML/PDF 会被量化。
- 密集踏板织体、快速重复音仍然困难。
- 目前主要在 MAESTRO 上训练，跨钢琴/跨房间泛化仍在评测中。

## 数据与许可说明

训练数据来自 MAESTRO v3.0.0，采用 CC BY-NC-SA 4.0 许可。模型权重按
**非商业/研究用途**分发，详见 [MODEL_LICENSE.md](MODEL_LICENSE.md)；源码使用
MIT License，详见 [LICENSE](LICENSE) 和第三方说明 [NOTICE.md](NOTICE.md)。

本项目仅用于研究与非商业用途；第三方模型名称和结果仅用于对比。

> 文档使用 `$REPO_ROOT`、`$DATA_ROOT`、`$PROJECT_HOME` 占位符，
> 不使用某台机器的绝对路径；约定见 [INSTALL.zh-CN.md](INSTALL.zh-CN.md)。
