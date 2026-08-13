# A training-only image derived from AI Toolkit's supported CUDA container.
# Keep AI_TOOLKIT_REVISION immutable: job configs reference the resulting image
# tag, rather than installing a moving dependency set during paid GPU time.
FROM nvidia/cuda:13.0.3-runtime-ubuntu24.04

ARG AI_TOOLKIT_REVISION

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0 10.0 12.0"

RUN apt-get update && apt-get install --no-install-recommends -y \
    git \
    curl \
    build-essential \
    cmake \
    wget \
    python3.12 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-wheel \
    python3-venv \
    ffmpeg \
    python3-opencv \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/pip3 /usr/local/bin/pip

# These are AI Toolkit's current Linux/cu130 manager pins. Install the torch
# trio first, exactly as its Dockerfile and manual installation prescribe.
RUN python -m pip install --break-system-packages \
    torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu130

WORKDIR /opt/ai-toolkit
RUN test -n "${AI_TOOLKIT_REVISION}" \
    && git clone https://github.com/ostris/ai-toolkit.git . \
    && git checkout "${AI_TOOLKIT_REVISION}"

# Match AI Toolkit's Dockerfile dependency order. Do not replace individual
# packages at job runtime: rebuild and retag this image for dependency changes.
RUN python -m pip install --break-system-packages -r requirements.txt \
    && python -m pip install --break-system-packages setuptools==69.5.1 \
    && python -m pip install --break-system-packages -U \
        torchcodec==0.15.0 \
        natten==0.21.7+torch2130cu130 --find-links https://whl.natten.org \
        https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.47/flash_attn-2.8.3+cu130torch2.13-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl \
    && python -c "import flash_attn, natten, torchcodec; print('accelerators OK:', flash_attn.__version__, natten.__version__, torchcodec.__version__)"

COPY image/run_ai_toolkit_container.py /opt/job/run_ai_toolkit_container.py

LABEL org.opencontainers.image.source="https://github.com/ostris/ai-toolkit" \
      org.opencontainers.image.description="Pinned AI Toolkit training image for Hugging Face Jobs"

WORKDIR /opt/ai-toolkit
