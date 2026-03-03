# =============================================================================
# ReconViaGen API — Docker Image
#
# Target hardware : NVIDIA RTX Pro 6000 (Blackwell, sm_120)
# CUDA toolkit    : 12.8
# cuDNN           : 9
# Python          : 3.10
# PyTorch         : 2.6.0 + cu126  (first release with native sm_120 support)
#
# Build:
#   docker build -t reconviagen:latest .
#
# Run (quick test):
#   docker run --gpus all -p 52069:52069 reconviagen:latest
#
# NOTE: First request triggers lazy model download from Hugging Face.
#       Mount a volume at /cache/huggingface to persist the model across
#       container restarts (see docker-compose.yml).
# =============================================================================

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

# ── Build-time arguments ──────────────────────────────────────────────────────
# CUDA arch list — sm_120 = RTX Pro 6000 (Blackwell).
# Broader list kept for portability; trim to just "12.0" to speed up builds.
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0;12.0"

# Parallel compile jobs for ninja. Raise on build machines with many cores.
# Warning: high values may cause OOM during flash-attn compilation.
ARG MAX_JOBS=4

# ── Proxy (build-time + runtime) ──────────────────────────────────────────────
# Required for apt-get, pip, and git clone to reach the internet through the
# corporate proxy. Both lowercase and uppercase variants are set because
# different tools honour different conventions.
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
    # CUDA
    CUDA_HOME=/usr/local/cuda \
    # Exported so every pip-compiled CUDA extension picks up the right targets
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    MAX_JOBS=${MAX_JOBS} \
    # Required by TRELLIS sparse-conv backend
    SPCONV_ALGO=native \
    # Hugging Face model cache.
    # Host path /pegaai is mounted as /notebooks inside the container,
    # so the effective host location is /pegaai/model_team/huggingface_cache.
    HF_HOME=/notebooks/model_team/huggingface_cache \
    TRANSFORMERS_CACHE=/notebooks/model_team/huggingface_cache \
    HUGGINGFACE_HUB_CACHE=/notebooks/model_team/huggingface_cache

# ── apt proxy config (explicit, in case apt ignores ENV vars) ────────────────
RUN printf 'Acquire::http::Proxy "%s";\nAcquire::https::Proxy "%s";\n' \
      "${http_proxy}" "${https_proxy}" \
      > /etc/apt/apt.conf.d/99proxy

# ── System packages ───────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python runtime + headers (needed to compile C extensions)
    python3.10 \
    python3.10-dev \
    python3-pip \
    # CUDA extension build chain
    build-essential \
    ninja-build \
    cmake \
    git \
    wget \
    curl \
    # OpenGL / EGL (required by nvdiffrast for differentiable rasterization)
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    # OpenMP (used by scipy, trimesh, etc.)
    libgomp1 \
    # Video encoding backend for imageio-ffmpeg
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ── Python 3.10 as default interpreter ───────────────────────────────────────
RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
 && python -m pip install --upgrade --no-cache-dir pip setuptools wheel

WORKDIR /app

# =============================================================================
# STEP 1 — PyTorch 2.6 with CUDA 12.6 wheels
#
# cu126 wheels are forward-compatible with the CUDA 12.8 runtime.
# PyTorch 2.6 is the first stable release that includes sm_120 (Blackwell)
# kernels compiled into the distributed binaries.
# =============================================================================
RUN pip install --no-cache-dir \
    torch==2.6.0 \
    torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu126

# =============================================================================
# STEP 2 — Pure-Python / non-CUDA packages
#
# torch / torchvision / xformers / spconv / flash-attn / nvdiffrast /
# diff-gaussian-rasterization / utils3d are installed in later steps because
# they either need a version matched to torch or require CUDA compilation.
# =============================================================================
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# =============================================================================
# STEP 3 — xformers (must be version-matched to torch)
# =============================================================================
RUN pip install --no-cache-dir xformers \
    --index-url https://download.pytorch.org/whl/cu126

# =============================================================================
# STEP 4 — spconv
#
# spconv-cu120 (CUDA 12.0 ABI) is binary-compatible with the CUDA 12.8
# runtime. If you hit "illegal instruction" or sparse-conv errors at runtime,
# rebuild from source:
#
#   pip install "git+https://github.com/traveller59/spconv.git"
#
# A cu128 wheel may also be available on PyPI for newer spconv releases.
# =============================================================================
RUN pip install --no-cache-dir spconv-cu120==2.3.6

# =============================================================================
# STEP 5 — nvdiffrast (build from local source)
#
# The repository ships the nvdiffrast 0.3.3 source under extensions/nvdiffrast.
# Building from source here ensures the CUDA kernels are compiled for sm_120
# using the TORCH_CUDA_ARCH_LIST defined above.
# =============================================================================
COPY extensions/nvdiffrast /tmp/nvdiffrast
RUN pip install --no-cache-dir --no-build-isolation /tmp/nvdiffrast \
 && rm -rf /tmp/nvdiffrast

# =============================================================================
# STEP 6 — diff-gaussian-rasterization (build from source)
#
# The pre-built HuggingFace wheel targets older CUDA architectures.
# Cloning and building from source generates proper sm_120 kernels.
# =============================================================================
RUN git clone --depth 1 --recurse-submodules \
    https://github.com/graphdeco-inria/diff-gaussian-rasterization.git \
    /tmp/diff-gaussian-rasterization \
 && pip install --no-cache-dir /tmp/diff-gaussian-rasterization \
 && rm -rf /tmp/diff-gaussian-rasterization

# =============================================================================
# STEP 7 — flash-attention (build from source — expect ~20-40 min)
#
# Pre-built wheels from PyPI / HuggingFace are compiled for older CUDA
# architectures and may not include sm_120 kernels.
# Building from source with MAX_JOBS controls CPU/RAM usage.
#
# To skip (if your model variant doesn't require flash-attn):
#   comment out this RUN block and add `--no-deps` to the transformers install.
# =============================================================================
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# =============================================================================
# STEP 8 — utils3d (pinned git commit, same as original requirements.txt)
# =============================================================================
RUN pip install --no-cache-dir \
    "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"

# =============================================================================
# Application source
# =============================================================================
COPY trellis/        /app/trellis/
COPY reconviagen_api.py /app/reconviagen_api.py

# Create mount point and output directory.
# /notebooks is the container-side mount of the host's /pegaai directory.
# The actual huggingface_cache path is created at runtime when the volume
# is mounted; we only need the mount-point directory to exist here.
RUN mkdir -p /notebooks /app/outputs

# ── Port ──────────────────────────────────────────────────────────────────────
EXPOSE 52069

# ── Health check ──────────────────────────────────────────────────────────────
# start-period is long because the first request triggers a lazy model load
# (~minutes depending on download speed and VRAM).
HEALTHCHECK \
    --interval=30s \
    --timeout=15s \
    --start-period=300s \
    --retries=5 \
    CMD curl -f http://localhost:52069/health || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
# Single worker: the GPU pipeline is protected by asyncio.Lock() internally,
# so multiple uvicorn workers would fight for the GPU without benefit.
CMD ["python", "-m", "uvicorn", "reconviagen_api:app", \
     "--host", "0.0.0.0", \
     "--port", "52069", \
     "--workers", "1", \
     "--log-level", "info"]
