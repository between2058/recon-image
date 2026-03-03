#!/usr/bin/env bash
# =============================================================================
# docker-build.sh — Helper script for building and running ReconViaGen
# =============================================================================
set -euo pipefail

IMAGE="reconviagen:latest"
CONTAINER="reconviagen-api"
PORT=52069

usage() {
  cat <<EOF
Usage: $0 <command>

Commands:
  build          Build the Docker image (full build, ~30-60 min first time)
  build-fast     Build for ONLY sm_120 — shorter compile time, less portable
  up             Start container with docker compose
  down           Stop and remove container
  logs           Tail container logs
  shell          Open a bash shell inside the running container
  health         Check the /health endpoint
  test           Send a test single-image request (requires test.png)
  clean          Remove image and named volumes

Environment variables:
  HF_TOKEN       Hugging Face token (if model repo requires authentication)
EOF
}

PROXY_ARGS=(
  --build-arg http_proxy="http://proxy.intra:80"
  --build-arg https_proxy="http://proxy.intra:80"
  --build-arg no_proxy="localhost,127.0.0.1"
)

build() {
  echo "🔨 Building $IMAGE (all architectures: 8.0;8.6;8.9;9.0;10.0;12.0)..."
  docker build \
    "${PROXY_ARGS[@]}" \
    --build-arg TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0;12.0" \
    --build-arg MAX_JOBS="$(nproc)" \
    -t "$IMAGE" .
  echo "✅ Build complete: $IMAGE"
}

build_fast() {
  echo "⚡ Building $IMAGE (sm_120 only — RTX Pro 6000 target)..."
  docker build \
    "${PROXY_ARGS[@]}" \
    --build-arg TORCH_CUDA_ARCH_LIST="12.0" \
    --build-arg MAX_JOBS="$(nproc)" \
    -t "$IMAGE" .
  echo "✅ Fast build complete: $IMAGE"
}

up() {
  docker compose up -d
  echo "🚀 Container started. API: http://localhost:$PORT"
  echo "   Health: http://localhost:$PORT/health"
}

down() {
  docker compose down
}

logs() {
  docker compose logs -f reconviagen
}

shell() {
  docker exec -it "$CONTAINER" bash
}

health() {
  curl -s http://localhost:$PORT/health | python3 -m json.tool
}

test_single() {
  if [[ ! -f test.png ]]; then
    echo "❌ test.png not found. Place an image named test.png in the project root."
    exit 1
  fi
  echo "📤 Sending test.png to /generate-single ..."
  curl -X POST http://localhost:$PORT/generate-single \
    -F "file=@test.png" \
    -F "seed=42" \
    -F "simplify=0.95" \
    -F "texture_size=1024" \
    | python3 -m json.tool
}

clean() {
  read -p "⚠️  This removes the image and HF model cache volume. Continue? [y/N] " yn
  case $yn in
    [Yy]*)
      docker compose down -v
      docker rmi "$IMAGE" 2>/dev/null || true
      echo "🧹 Cleaned."
      ;;
    *) echo "Aborted." ;;
  esac
}

case "${1:-}" in
  build)      build ;;
  build-fast) build_fast ;;
  up)         up ;;
  down)       down ;;
  logs)       logs ;;
  shell)      shell ;;
  health)     health ;;
  test)       test_single ;;
  clean)      clean ;;
  *)          usage ;;
esac
