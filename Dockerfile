# GPU 运行镜像：NVIDIA CUDA，适合本地/服务器 GPU 部署。
#
# 用法：
#   docker build -f Dockerfile.gpu -t piano-transcription:gpu .
#   docker run --rm --gpus all -p 8000:8000 \
#     -v "$PWD/models:/workspace/models" piano-transcription:gpu
#
# 模型权重不进入 git；可挂载，也可在构建时用 MODEL_URL 下载 release 权重。
# macOS 不支持 NVIDIA GPU 透传，Mac 用户请改用 Dockerfile.cpu。

ARG BASE_IMAGE=pytorch/pytorch:2.9.0-cuda12.6-cudnn9-runtime
FROM ${BASE_IMAGE}

ARG APT_MIRROR=mirrors.aliyun.com
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG MODEL_URL=""

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Shanghai \
    MPLBACKEND=Agg \
    PIANO2_SOUNDFONT=/usr/share/sounds/sf2/FluidR3_GM.sf2

RUN set -eux; \
    sed -i "s|archive.ubuntu.com|${APT_MIRROR}|g; s|security.ubuntu.com|${APT_MIRROR}|g" \
        /etc/apt/sources.list /etc/apt/sources.list.d/*.list 2>/dev/null || true; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates curl ffmpeg libsndfile1 librsvg2-bin \
        fonts-noto-cjk fonts-wqy-zenhei \
        fluidsynth fluid-soundfont-gm; \
    rm -rf /var/lib/apt/lists/*; \
    printf '[global]\nindex-url = %s\n' "${PIP_INDEX_URL}" > /etc/pip.conf

WORKDIR /workspace
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /workspace
RUN pip install --no-cache-dir -e .

# 可选：在构建阶段下载 release 权重，实现真正的零挂载启动。
RUN if [ -n "$MODEL_URL" ]; then \
        mkdir -p /workspace/models && \
        curl -L --fail --retry 3 "$MODEL_URL" -o /workspace/models/best.pt; \
    fi

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
