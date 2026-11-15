# 安装指南

**[English](INSTALL.md) | [简体中文](INSTALL.zh-CN.md)**

本文介绍本地 Python 安装、CPU/GPU Docker、模型权重和快速验证。文档不使用本机绝对路径；
请把下面的占位符替换成你自己的位置。

## 0. 路径约定

文档统一使用这些占位符，而不是某台机器的路径：

| 占位符 | 含义 |
|---|---|
| `$REPO_ROOT` | 本仓库根目录（包含 `app/`、`src/`、`tutorial/`） |
| `$DATA_ROOT` | 数据根目录，包含 `raw/` 和 `processed/` |
| `$PROJECT_HOME` | 同时包含仓库和数据目录的父目录 |
| `$OUTPUT_ROOT` | checkpoint、日志、评测结果的输出根目录 |
| `$SOTA_REPO_ROOT` | 可选的外部参考仓库（用于对比） |

例如你的机器上 `$REPO_ROOT` 可能是 `./piano2`，`$DATA_ROOT` 可能是
`./piano2_data`；不要把另一台机器的路径复制进配置。

## 1. 前置条件

| 项目 | 要求 |
|---|---|
| Python | 3.10+ |
| 操作系统 | 推荐 Linux；macOS / Windows 可用 Docker 或原生 Python |
| GPU | 可选；自动检测 NVIDIA CUDA，否则走 CPU |
| 磁盘 | app + 模型约 1 GB；完整 MAESTRO 训练数据数十 GB |
| 模型权重 | 转录必需；不随 git 仓库分发 |

真实钢琴回放和谱面渲染还需要系统依赖：

- Linux：`ffmpeg`、`libsndfile1`、`librsvg2-bin`、`fluidsynth`、`fluid-soundfont-gm`、CJK 字体；
- macOS：Homebrew 安装 `ffmpeg`、`libsndfile`、`librsvg`、`fluid-synth`；
  没有 FluidSynth/SoundFont 时自动回退到简易合成器；
- Windows：推荐 Docker Desktop；原生安装需要 `ffmpeg`、`fluidsynth`、
  `rsvg-convert` 和 SoundFont。

## 2. 本地 Python 安装

```bash
cd "$REPO_ROOT"
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

把 checkpoint 放到 `"$REPO_ROOT/models/best.pt"`，或显式指定：

```bash
export PIANO2_CKPT="$REPO_ROOT/models/best.pt"
# Windows PowerShell: $env:PIANO2_CKPT = "$REPO_ROOT\models\best.pt"
```

模型权重不进入 git。下面“模型权重”和 README 的复现章节会说明来源。

## 3. 模型权重

app 需要训练好的 checkpoint（当前推荐 `hybrid_reg2`），它和源码仓库分开分发，
例如作为 GitHub Release 附件：

```bash
mkdir -p "$REPO_ROOT/models"
# 把你从项目 release 下载的 checkpoint 复制过来：
cp "$CHECKPOINT_PATH" "$REPO_ROOT/models/best.pt"
```

也可以设置 `PIANO2_CKPT` 指向任意可读的 checkpoint。

checkpoint 是训练脚本保存的 PyTorch 文件；加载器用 `strict=False` 兼容旧版少头模型，
缺失的头会退回可用的分类/offset 推理路径。

## 4. 启动 app

```bash
cd "$REPO_ROOT"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

浏览器打开 <http://localhost:8000>，检查后端：

```bash
curl http://localhost:8000/api/health
```

应返回 `status: ok`、设备信息和 `checkpoint_exists: true`。

## 5. Docker：CPU 与 GPU

仓库提供了两个 Dockerfile，镜像启动后会自动运行 Web 应用：

| 文件 | 用途 |
|---|---|
| `Dockerfile.cpu` | 无 GPU 的笔记本/服务器 |
| `Dockerfile.gpu` | NVIDIA CUDA 机器 |
| `Dockerfile` | GPU 便捷入口 |

CPU：

```bash
docker build -f Dockerfile.cpu -t piano-transcription:cpu .
docker run --rm -p 8000:8000 \
  -v "$REPO_ROOT/models:/workspace/models" \
  piano-transcription:cpu
```

GPU：

```bash
docker build -f Dockerfile.gpu -t piano-transcription:gpu .
docker run --rm --gpus all -p 8000:8000 \
  -v "$REPO_ROOT/models:/workspace/models" \
  piano-transcription:gpu
```

Compose：

```bash
docker compose --profile cpu up --build
# 或：docker compose --profile gpu up --build
```

如果希望零挂载启动，把 checkpoint 放到 GitHub Release，构建时传 `MODEL_URL`：

```bash
docker build -f Dockerfile.gpu \
  --build-arg MODEL_URL="https://github.com/<owner>/<repo>/releases/download/v0.2.6/best.pt" \
  -t piano-transcription:gpu .
```

**macOS**：Docker 不能把 Apple Metal 或 NVIDIA GPU 透传给 Linux 容器；
Intel 和 Apple Silicon Mac 都请使用 CPU 镜像，`--gpus all` 在 macOS 会失败。

GitHub Actions 工作流 [`.github/workflows/docker.yml`](.github/workflows/docker.yml)
可构建并推送 CPU/GPU 镜像到 GHCR。

## 6. 验证安装

```bash
cd "$REPO_ROOT"
python -m pytest -q
```

预期全部测试通过。然后上传一段独奏钢琴 WAV/MP3，确认：

- 队列能完成转录；
- 结果页显示乐谱；
- 播放器可用；
- MusicXML / PDF / MIDI 下载按钮产生文件。

当前 app 版本为 0.2.6；见 [app changelog](docs/09_app_changelog.md) 和 [模型与转录局限](docs/10_limitations.md)。

## 7. 复现 checkpoint

完整命令和指标见 [README.md § 复现模型参数](README.zh-CN.md#复现模型参数)。简版：

```bash
export PIANO2_MAESTRO_ROOT="$DATA_ROOT/raw/maestro-v3.0.0"
export PIANO2_DATA_ROOT="$DATA_ROOT/processed/maestro_processed_v2"
export PIANO2_V2_ROOT="$PIANO2_DATA_ROOT"
python scripts/build_data.py --workers 8

export PIANO2_V3A_ROOT="$DATA_ROOT/processed/maestro_processed_v3a"
python scripts/build_labels_v3a.py --workers 8
export PIANO2_V3B_ROOT="$DATA_ROOT/processed/maestro_processed_v3b"
python scripts/build_labels_v3b.py --workers 8
export PIANO2_V4_ROOT="$DATA_ROOT/processed/maestro_processed_v4"
python scripts/build_labels_v4.py --workers 8

export PIANO2_DATA_ROOT="$DATA_ROOT/processed/maestro_processed_v4"
python -m piano2.train --max-steps 71160 --batch 16 --val-every 500 \
  --save-every 200 --val-segments 120 --num-workers 4 \
  --run-name hybrid_reg2 --label-channel onset1 \
  --offset-loss 2.0 --reg-loss 0.5 --seed 0
```

checkpoint 输出在 `$REPO_ROOT/outputs/M1/hybrid_reg2/checkpoints/best.pt`。
把它复制到 `"$REPO_ROOT/models/best.pt"` 即可给 app 使用。

## 8. 常见问题

| 现象 | 处理 |
|---|---|
| `FileNotFoundError: 找不到模型检查点` | 把 checkpoint 放到 `models/best.pt`，或设置 `PIANO2_CKPT` |
| `/api/health` 显示 `checkpoint_exists: false` | 同上 |
| 回放像电子音 | 安装 FluidSynth 和 SoundFont；缺失时会回退 |
| `torch.cuda.is_available()` 为 false | 用 CPU 镜像，或安装 NVIDIA 驱动/容器运行时 |
| Docker 构建无法访问 registry | 配置镜像/代理；这是环境相关问题 |
| 8000 端口被占用 | 用 `--port 8001`，浏览器打开 `http://localhost:8001` |
| 浏览器仍显示旧 UI | 强制刷新（`Ctrl+F5`）；app 的静态资源已设 `no-store` |
| macOS 上 GPU 镜像启动失败 | 改用 `Dockerfile.cpu`；macOS 不支持 GPU 容器透传 |
| 长曲目第 20 页以后没有谱 | app 有意最多渲染 20 页 |

## 9. 说明

- 服务默认绑定 `127.0.0.1`；不要在没有认证/TLS 的情况下直接暴露到公网；
- 转录音频只在本地处理，不会上传到第三方服务；
- MAESTRO 训练数据采用 CC BY-NC-SA 4.0，仅用于研究与非商业用途。
