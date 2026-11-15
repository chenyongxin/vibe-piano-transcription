# Installation

**[English](INSTALL.md) | [简体中文](INSTALL.zh-CN.md)**

This guide covers a local Python install, the CPU/GPU Docker images, model weights,
and a quick verification. It is intentionally path-free: substitute the placeholders
below with your own locations.

## 0. Path convention

All documentation uses these placeholders instead of machine-specific paths:

| Placeholder | Meaning |
|---|---|
| `$REPO_ROOT` | root of this repository (the directory containing `app/`, `src/`, `tutorial/`) |
| `$DATA_ROOT` | data root containing `raw/` and `processed/` |
| `$PROJECT_HOME` | parent directory that contains the repo and data directories |
| `$OUTPUT_ROOT` | output root for checkpoints, logs, and evaluation files |
| `$SOTA_REPO_ROOT` | optional external reference repository used for comparison |

For example, on your machine `$REPO_ROOT` might be `./piano2` and `$DATA_ROOT`
might be `./piano2_data`; do not copy a path from another machine.

## 1. Prerequisites

| Component | Requirement |
|---|---|
| Python | 3.10+ |
| OS | Linux recommended; macOS and Windows work with Docker or native Python |
| GPU | optional; NVIDIA CUDA is auto-detected, otherwise CPU is used |
| Disk | app + model: about 1 GB; full MAESTRO training data: tens of GB |
| Model checkpoint | required for transcription; not included in git |

System packages for realistic playback and score rendering:

- Linux: `ffmpeg`, `libsndfile1`, `librsvg2-bin`, `fluidsynth`, `fluid-soundfont-gm`, CJK fonts.
- macOS: `ffmpeg`, `libsndfile`, `librsvg`, `fluid-synth` (Homebrew); the app falls back
  to a simple synthesizer if FluidSynth/SoundFont is missing.
- Windows: Docker Desktop is recommended; native setup needs `ffmpeg`,
  `fluidsynth`, `rsvg-convert`, and a SoundFont.

## 2. Local Python install

```bash
cd "$REPO_ROOT"
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Put the checkpoint at `"$REPO_ROOT/models/best.pt"`, or point to it explicitly:

```bash
export PIANO2_CKPT="$REPO_ROOT/models/best.pt"
# Windows PowerShell: $env:PIANO2_CKPT = "$REPO_ROOT\models\best.pt"
```

Model weights are not committed to git. See
[Model weights](#3-model-weights) and the
[reproduction guide](README.md#reproduce-the-model).

## 3. Model weights

The app needs a trained checkpoint (the current recommended model is `hybrid_reg2`).
It is distributed separately from the source repository, for example as a GitHub
Release asset:

```bash
mkdir -p "$REPO_ROOT/models"
# Copy the checkpoint you downloaded from the project release:
cp "$CHECKPOINT_PATH" "$REPO_ROOT/models/best.pt"
```

Alternatively, set `PIANO2_CKPT` to any readable checkpoint path.

The checkpoint is a PyTorch file saved by the training script. The loader supports
older checkpoints with fewer heads via `strict=False`; missing heads fall back to
available classification/offset behavior.

## 4. Run the app

```bash
cd "$REPO_ROOT"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open <http://localhost:8000>. Check the backend:

```bash
curl http://localhost:8000/api/health
```

It should return `status: ok`, a device, and `checkpoint_exists: true`.

## 5. Docker: CPU and GPU

The repository contains two Dockerfiles, both starting the web app automatically:

| File | Target |
|---|---|
| `Dockerfile.cpu` | CPU-only laptops and servers |
| `Dockerfile.gpu` | NVIDIA CUDA machines |
| `Dockerfile` | convenience GPU entry point |

CPU:

```bash
docker build -f Dockerfile.cpu -t piano-transcription:cpu .
docker run --rm -p 8000:8000 \
  -v "$REPO_ROOT/models:/workspace/models" \
  piano-transcription:cpu
```

GPU:

```bash
docker build -f Dockerfile.gpu -t piano-transcription:gpu .
docker run --rm --gpus all -p 8000:8000 \
  -v "$REPO_ROOT/models:/workspace/models" \
  piano-transcription:gpu
```

Compose:

```bash
docker compose --profile cpu up --build
# or: docker compose --profile gpu up --build
```

For a zero-mount build, attach the checkpoint to a GitHub Release and pass `MODEL_URL`:

```bash
docker build -f Dockerfile.gpu \
  --build-arg MODEL_URL="https://github.com/<owner>/<repo>/releases/download/v0.2.6/best.pt" \
  -t piano-transcription:gpu .
```

**macOS:** Docker cannot expose Apple Metal or NVIDIA GPUs to Linux containers.
Use the CPU image on both Intel and Apple Silicon Macs; `--gpus all` fails on macOS.

GitHub Actions [`.github/workflows/docker.yml`](.github/workflows/docker.yml) can build
and push the CPU/GPU images to GHCR.

## 6. Verify the installation

```bash
cd "$REPO_ROOT"
python -m pytest -q
```

Expected result: all tests pass. Then upload a short solo-piano WAV/MP3 and check:

- the queue completes;
- the result page shows a score;
- the audio player works;
- MusicXML/PDF/MIDI download buttons produce files.

The app is currently at version 0.2.6; see the
[app changelog](docs/09_app_changelog.md) and
[model/transcription limitations](docs/10_limitations.md).

## 7. Reproduce a checkpoint

Full commands and expected metrics are in
[README.md § Reproduce the model](README.md#reproduce-the-model). The short version:

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

The checkpoint appears at `$REPO_ROOT/outputs/M1/hybrid_reg2/checkpoints/best.pt`.
Copy it to `"$REPO_ROOT/models/best.pt"` for the app.

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `FileNotFoundError: 找不到模型检查点` | place a checkpoint at `models/best.pt` or set `PIANO2_CKPT` |
| `checkpoint_exists: false` in `/api/health` | same as above |
| Playback sounds dry/electronic | install FluidSynth and a SoundFont; app falls back automatically |
| `torch.cuda.is_available()` is false | use CPU image or install an NVIDIA driver + container runtime |
| Docker build cannot reach a registry | configure a registry mirror/proxy; this is environment-specific |
| Port 8000 occupied | use `--port 8001` and open `http://localhost:8001` |
| Browser still shows an old UI | hard refresh (`Ctrl+F5`); the app serves static files with `no-store` |
| GPU image on macOS fails | use `Dockerfile.cpu`; macOS cannot pass GPUs into Linux containers |
| Long piece has no score after page 20 | the app intentionally caps rendered pages at 20 |

## 9. Notes

- The server binds to `127.0.0.1` by default; do not expose it publicly without adding
  authentication and TLS.
- Transcription runs locally; the app does not upload audio to a third-party service.
- MAESTRO training data is CC BY-NC-SA 4.0; this project is for research and
  non-commercial use.
