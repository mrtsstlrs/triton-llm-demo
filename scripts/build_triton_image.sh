#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRITON_CONTAINER_VERSION="${TRITON_CONTAINER_VERSION:-25.12}"
BASE_IMAGE="${BASE_IMAGE:-triton-base:${TRITON_CONTAINER_VERSION}}"

echo "Building base image: ${BASE_IMAGE}"
docker build -t "${BASE_IMAGE}" -f "${ROOT_DIR}/Dockerfile.base" "${ROOT_DIR}"

echo "Building Triton image with metrics + vLLM backend"
(
  cd "${ROOT_DIR}/server"
  ./build.py \
    --no-container-pull \
    --image "base,${BASE_IMAGE}" \
    --target-platform linux \
    --target-machine x86_64 \
    --backend python \
    --backend vllm \
    --backend ensemble \
    --endpoint http \
    --endpoint grpc \
    --enable-logging \
    --enable-stats \
    --enable-metrics \
    --enable-gpu-metrics \
    --enable-cpu-metrics \
    --enable-gpu
)
