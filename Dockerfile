# piano2 运行镜像
#
# 设计原则：镜像只装「算 + 画」需要的东西。数据与权重不放进镜像，
# 用挂载或环境变量指定（见 src/piano2/constants.py）。
#
# 国内网络下用 build args 换镜像站，例如：
#   docker build --build-arg APT_MIRROR=mirrors.tuna.tsinghua.edu.cn \
#                --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple .
ARG BASE_IMAGE=pytorch/pytorch:2.9.0-cuda12.6-cudnn9-runtime
FROM ${BASE_IMAGE}

ARG APT_MIRROR=archive.ubuntu.com
ARG PIP_INDEX_URL=https://pypi.org/simple

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Shanghai \
    MPLBACKEND=Agg

# fonts-noto-cjk / fonts-wqy-zenhei：画中文图；librsvg2-bin：把 MusicXML 渲染成 PNG
RUN set -eux; \
    sed -i "s|archive.ubuntu.com|${APT_MIRROR}|g; s|security.ubuntu.com|${APT_MIRROR}|g" \
        /etc/apt/sources.list /etc/apt/sources.list.d/*.list 2>/dev/null || true; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        fonts-noto-cjk fonts-wqy-zenhei librsvg2-bin; \
    rm -rf /var/lib/apt/lists/*; \
    printf '[global]\nindex-url = %s\n' "${PIP_INDEX_URL}" > /etc/pip.conf

WORKDIR /workspace
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /workspace
RUN pip install --no-cache-dir -e .

CMD ["bash"]
