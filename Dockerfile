# =============================================================================
# ReconViaGen API — Docker Image
#
# Target hardware : NVIDIA RTX Pro 6000 (Blackwell, sm_120)
# CUDA toolkit    : 12.8.1
# cuDNN           : 9
# Python          : 3.10
# PyTorch         : 2.7.1 + cu128
#
# Build:
#   docker build -t reconviagen:latest .
#
# Run (quick test):
#   docker run --gpus all -p 52069:52069 reconviagen:latest
#
# NOTE: First request triggers lazy model download from Hugging Face.
#       Mount /pegaai:/notebooks:rw and set HF_HOME (see docker-compose.yml).
# =============================================================================

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

# ── Build-time arguments ──────────────────────────────────────────────────────
# sm_120 = RTX Pro 6000 (Blackwell). Matches local install environment.
ARG TORCH_CUDA_ARCH_LIST="12.0"

# Parallel compile jobs for ninja. flash-attn is capped separately at 4.
ARG MAX_JOBS=4

# ── Proxy (build-time + runtime) ──────────────────────────────────────────────
ARG http_proxy="http://proxy.intra:80"
ARG https_proxy="http://proxy.intra:80"
ARG no_proxy="localhost,127.0.0.1"

ENV http_proxy=${http_proxy} \
    https_proxy=${https_proxy} \
    HTTP_PROXY=${http_proxy} \
    HTTPS_PROXY=${https_proxy} \
    no_proxy=${no_proxy} \
    NO_PROXY=${no_proxy}

# ── Environment variables ─────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Match local install: export CUDA_HOME=/usr/local/cuda-12.8
    CUDA_HOME=/usr/local/cuda-12.8 \
    # Match local install: export TORCH_CUDA_ARCH_LIST="12.0"
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    MAX_JOBS=${MAX_JOBS} \
    # Required by TRELLIS sparse-conv backend
    SPCONV_ALGO=native \
    # Hugging Face model cache → mounted from host /pegaai/model_team/huggingface_cache
    HF_HOME=/notebooks/model_team/huggingface_cache \
    TRANSFORMERS_CACHE=/notebooks/model_team/huggingface_cache \
    HUGGINGFACE_HUB_CACHE=/notebooks/model_team/huggingface_cache

# ── apt proxy config ──────────────────────────────────────────────────────────
RUN printf 'Acquire::http::Proxy "%s";\nAcquire::https::Proxy "%s";\n' \
      "${http_proxy}" "${https_proxy}" \
      > /etc/apt/apt.conf.d/99proxy

# ── System packages ───────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    ninja-build \
    cmake \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ── Python 3.10 as default interpreter ───────────────────────────────────────
RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
 && python -m pip install --upgrade --no-cache-dir pip setuptools wheel

# ── Ensure /usr/local/cuda-12.8 exists (CUDA_HOME points here) ───────────────
RUN test -d /usr/local/cuda-12.8 \
 || ln -sf /usr/local/cuda /usr/local/cuda-12.8

WORKDIR /app

# =============================================================================
# STEP 1 — PyTorch 2.7.1 + CUDA 12.8
#
# Exact versions confirmed working on RTX Pro 6000 (sm_120) locally.
# =============================================================================
RUN pip install --no-cache-dir \
    torch==2.7.1 \
    torchvision==0.22.1 \
    torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

# =============================================================================
# STEP 2 — Pure-Python / non-CUDA packages
# =============================================================================
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# =============================================================================
# STEP 3 — xformers 0.0.31
# =============================================================================
RUN pip install --no-cache-dir xformers==0.0.31

# =============================================================================
# STEP 4 — spconv-cu120
# =============================================================================
RUN pip install --no-cache-dir spconv-cu120

# =============================================================================
# STEP 5 — utils3d (pure Python, no CUDA compile)
# =============================================================================
RUN pip install --no-cache-dir \
    "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"

# =============================================================================
# STEP 6 — Lock torch back to 2.7.1 cu128 (ABI finalise point)
#
# requirements-api.txt deps (rembg, lpips, dreamsim…) may downgrade torch to
# the CPU PyPI version. Force-reinstall here so every CUDA extension below
# compiles and runs against the exact same torch ABI.
# =============================================================================
RUN pip install --no-cache-dir --force-reinstall \
    torch==2.7.1 \
    torchvision==0.22.1 \
    torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

# =============================================================================
# STEP 7 — nvdiffrast (from NVlabs GitHub, built against finalised torch)
# =============================================================================
RUN pip install --no-cache-dir --no-build-isolation \
    git+https://github.com/NVlabs/nvdiffrast.git

# =============================================================================
# STEP 8 — diff-gaussian-rasterization (from mip-splatting submodule)
#
# Uses the mip-splatting fork which is confirmed compatible with torch 2.7.1.
# =============================================================================
RUN git clone --depth 1 --recurse-submodules \
    https://github.com/autonomousvision/mip-splatting.git \
    /tmp/mip-splatting \
 && cd /tmp/mip-splatting/submodules/diff-gaussian-rasterization \
 && pip install --no-cache-dir --no-build-isolation . \
 && rm -rf /tmp/mip-splatting

# =============================================================================
# STEP 9 — flash-attn 2.8.0.post2 (built against finalised torch)
#
# Version pinned to match local install. MAX_JOBS=4 to prevent OOM.
# =============================================================================
RUN MAX_JOBS=4 pip install --no-cache-dir --no-build-isolation \
    flash-attn==2.8.0.post2

# =============================================================================
# Application source
# =============================================================================
COPY trellis/           /app/trellis/
COPY reconviagen_api.py /app/reconviagen_api.py
# wheels/ contains local Python packages (vggt, dust3r) that trellis imports
# via relative sys.path ("wheels/vggt"). Must live at /app/wheels/ so that
# the relative path resolves correctly when uvicorn runs from WORKDIR /app.
COPY wheels/            /app/wheels/

RUN mkdir -p /notebooks /app/outputs

# ── Port ──────────────────────────────────────────────────────────────────────
EXPOSE 52069

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK \
    --interval=30s \
    --timeout=15s \
    --start-period=300s \
    --retries=5 \
    CMD curl -f http://localhost:52069/health || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
CMD ["python", "-m", "uvicorn", "reconviagen_api:app", \
     "--host", "0.0.0.0", \
     "--port", "52069", \
     "--workers", "1", \
     "--log-level", "info"]
