# Piano Transcription

**[English](README.md) | [简体中文](README.zh-CN.md)**

Turn solo piano recordings into editable sheet music — locally.

![Piano transcription web app](docs/readme/home.png)

## What it does

- Upload WAV / MP3 / M4A / FLAC recordings (up to 5 tracks, 500 MB each).
- A hybrid model turns audio into note events, then into readable music notation.
- Review the result page by page, with the current measure highlighted during playback.
- Listen back with a realistic piano sound and pedal (CC64) support.
- Download **MusicXML**, **PDF**, and **MIDI**.

![Result page with score viewer](docs/readme/result.png)

## How it works

```text
audio
  → log-CQT (32 ms frames, 264 bins)
  → CNN + Transformer
  → 8 heads: onset / frame / velocity / offset / pedal / pedal_on /
             onset_reg / offset_reg
  → decode to NoteEvent(pitch, start, end, velocity)
  → MusicXML / PDF / MIDI / piano playback
```

The model keeps a **classification backbone** for note presence and duration,
and uses a **sub-frame onset regression** head to refine note starts inside a frame.
Raw note events are used for event-level metrics and playback; the score is quantized
to a readable rhythmic grid (16th notes).

## Results

Results on the MAESTRO v3.0.0 test split (177 tracks, raw note events):

| Metric | Hybrid model |
|---|---:|
| note-with-offset F1 (key-off ground truth) | **0.379** |
| onset F1 @ 50 ms | 0.870 |
| onset F1 @ 20 ms | 0.812 |
| **onset F1 @ 10 ms** | **0.626** |

The 10 ms onset F1 is **0.626**, compared with **0.507** for the classification-only
baseline. The offset-end window fix (0.7 threshold, 20-frame search) raised test-split
note-with-offset F1 from **0.331** to **0.379**; onset metrics were unchanged.
A holdout track reaches **0.660** at 10 ms and **0.453** note F1.

For reference, a ByteDance piano-transcription model measured on three validation
excerpts reached **0.901** onset F1 @10 ms and **0.444** note F1. Its note-off targets
are closer to acoustic sustain/pedal release, so the note-F1 comparison is not
apples-to-apples with key-off ground truth.

## Quick start

For prerequisites, system packages, local install, Docker CPU/GPU, model-weight
placement, and troubleshooting, see **[INSTALL.md](INSTALL.md)**.

### Local Python

```bash
python -m pip install -e .

# Put a trained checkpoint at models/best.pt, or set PIANO2_CKPT:
#   export PIANO2_CKPT=/path/to/checkpoint.pt

python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open <http://localhost:8000>.

Realistic piano playback needs `fluidsynth` and a SoundFont such as `FluidR3_GM.sf2`;
without them the app falls back to a lightweight synthesizer.

### Docker: CPU and GPU

Two Dockerfiles are included:

| File | Target | Start |
|---|---|---|
| `Dockerfile.cpu` | CPU-only laptops / servers | `docker build -f Dockerfile.cpu -t vibe-piano-transcription:cpu .` |
| `Dockerfile.gpu` | NVIDIA CUDA (RTX 4060+) | `docker build -f Dockerfile.gpu -t vibe-piano-transcription:gpu .` |

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

> **Platform note:** the GPU image requires the NVIDIA container runtime and an
> NVIDIA GPU. macOS cannot expose Apple GPU (Metal) or NVIDIA GPUs to Linux
> containers; use `Dockerfile.cpu` on both Intel and Apple Silicon Macs.
> `docker run --gpus all` will fail on macOS. If an Apple Silicon build cannot
> find an arm64 wheel for a dependency, build with `--platform linux/amd64`
> (emulated and slower) instead of switching to the GPU image.

Or use the compose profiles:

```bash
docker compose --profile cpu up --build
# or: docker compose --profile gpu up --build
```

The source-built images need a checkpoint mounted at `models/best.pt` (or set
`PIANO2_CKPT`). The published GitHub Release images are built from the release
tag and already contain the checkpoint, so no mount is needed.

For a source build with the release checkpoint baked in:

```bash
docker build -f Dockerfile.gpu \
  --build-arg MODEL_URL=https://github.com/chenyongxin/vibe-piano-transcription/releases/download/v0.3.0/best.pt \
  -t vibe-piano-transcription:gpu .
```

Release images are published to GitHub Container Registry (GHCR) by
[`.github/workflows/release.yml`](.github/workflows/release.yml):

```bash
# CPU, no model mount
docker run --rm -p 8000:8000 \
  ghcr.io/chenyongxin/vibe-piano-transcription:v0.3.0-cpu

# GPU, no model mount
docker run --rm --gpus all -p 8000:8000 \
  ghcr.io/chenyongxin/vibe-piano-transcription:v0.3.0-gpu
```

The moving aliases `:cpu` and `:gpu` point to the latest published release.

## Reproduce the model

The current app uses `hybrid_reg2` (data v4 + offset-loss 2.0 + reg-loss 0.5).

```bash
# 1. Download MAESTRO v3.0.0 from https://magenta.tensorflow.org/datasets/maestro and put the extracted folder here:
#    data/raw/maestro-v3.0.0/maestro-v3.0.0.csv, *.midi, *.wav
export PIANO2_MAESTRO_ROOT="$PWD/data/raw/maestro-v3.0.0"

# 2. Build CQT + labels (the scripts default to a container data path;
#    override the roots when running elsewhere).
export PIANO2_DATA_ROOT="$PWD/data/processed/maestro_processed_v2"
export PIANO2_V2_ROOT="$PWD/data/processed/maestro_processed_v2"
python scripts/build_data.py --workers 8

export PIANO2_V3A_ROOT="$PWD/data/processed/maestro_processed_v3a"
python scripts/build_labels_v3a.py --workers 8
export PIANO2_V3B_ROOT="$PWD/data/processed/maestro_processed_v3b"
python scripts/build_labels_v3b.py --workers 8
export PIANO2_V4_ROOT="$PWD/data/processed/maestro_processed_v4"
python scripts/build_labels_v4.py --workers 8

# 3. Train hybrid_reg2 (20 epochs, 71,160 steps; about 1.6 h on RTX 4060)
export PIANO2_DATA_ROOT="$PWD/data/processed/maestro_processed_v4"
python -m piano2.train --max-steps 71160 --batch 16 --val-every 500 \
  --save-every 200 --val-segments 120 --num-workers 4 \
  --run-name hybrid_reg2 --label-channel onset1 \
  --offset-loss 2.0 --reg-loss 0.5 --seed 0

# 4. Evaluate on MAESTRO test split
python scripts/evaluate_hybrid_test.py \
  --model outputs/M1/hybrid_reg2/checkpoints/best.pt \
  --out outputs/eval/hybrid_reg2_test.json
```

The checkpoint is written to
`outputs/M1/hybrid_reg2/checkpoints/best.pt`; copy it to `models/best.pt`
for the app. Full experiment details and ablations are in
[the training report](training/2026-11-11_duration_offset_hybrid.md).

## Project layout

| Path | Contents |
|---|---|
| `app/` | FastAPI backend, job queue, score rendering, static UI |
| `src/piano2/` | features, model, training, decoding, notation, evaluation |
| `tutorial/` | step-by-step guides from audio to score |
| `docs/` | architecture, data, decoding, notation, app design |
| `training/` | experiment reports and results |
| `scripts/` | command-line entry points and evaluation scripts |
| `tests/` | pytest suite |

## Documentation

- [Installation guide](INSTALL.md)
- [App changelog](docs/09_app_changelog.md)
- [Model and transcription limitations](docs/10_limitations.md)
- [Hard-passage failure report](experiments/hard_passages/report.md)
- [Tutorial index](tutorial/README.md)
- [Project overview](docs/01_overview.md)
- [Data and labels](docs/02_data.md)
- [Model and training](docs/03_model_training.md)
- [Decoding](docs/04_decoding.md)
- [App design](docs/08_m4_app.md)
- [Experiment reports](training/README.md)

## Limitations

See [docs/10_limitations.md](docs/10_limitations.md) for the detailed model, notation,
and audio-visual alignment limitations.

- Designed for **solo piano**; mixes and ensemble recordings are out of scope.
- The score is a readable rhythmic representation, not a millisecond-accurate
  performance transcription. Sub-frame timing is preserved in raw events and playback,
  but is quantized in MusicXML/PDF.
- Dense, pedaled textures and fast repeated notes remain difficult.
- Trained on MAESTRO; generalization to unseen pianos/rooms is still being evaluated.

## Data and license notes

Training data comes from MAESTRO v3.0.0, which is distributed under
CC BY-NC-SA 4.0. The checkpoint is distributed for non-commercial research and
evaluation; see [MODEL_LICENSE.md](MODEL_LICENSE.md). The source code is MIT
licensed; see [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md) for third-party
notices.

The project is intended for research and non-commercial use. Third-party model
names and results are referenced for comparison only.

> Documentation uses `$REPO_ROOT`, `$DATA_ROOT`, and `$PROJECT_HOME` placeholders
> instead of machine-specific paths; see [INSTALL.md](INSTALL.md) for the convention.
