# Экспериментальная сборка через единый Dockerfile

`Dockerfile.unified` - отдельная попытка собрать Triton Server `24.04` без текущего wrapper-flow `scripts/build_triton_image.sh`.

Текущая рабочая сборка остается основной. Этот Dockerfile нужен для проверки альтернативного подхода и пока должен рассматриваться как experimental.

## Идея

Вместо upstream container-based build, который генерирует `server/build/docker_build` и запускает вложенные Docker builds, используется:

```bash
build.py --no-container-build
```

Это позволяет собрать Triton прямо внутри Docker build stage без Docker-in-Docker и без доступа к `/var/run/docker.sock`.

Dockerfile состоит из трех stages:

- `runtime-base`: Astra base image, публичный NVIDIA CUDA apt repo, runtime CUDA/cuBLAS/cuDNN/TensorRT/DCGM, Python venv и vLLM dependencies.
- `builder`: build dependencies, CUDA/TensorRT dev packages, clone Triton Server `r24.04`, patch `third_party`, запуск `build.py --no-container-build`.
- `runtime`: финальный runtime image, копирует `/opt/tritonserver` из builder stage.

## Сборка

```bash
docker build \
  --progress=plain \
  -f Dockerfile.unified \
  -t tritonserver:24.04-unified \
  .
```

Переопределяемые параметры:

```bash
docker build \
  -f Dockerfile.unified \
  -t tritonserver:24.04-unified \
  --build-arg ASTRA_BASE_IMAGE=registry.astralinux.ru/library/astra/ubi18-python311:1.8.5 \
  --build-arg TRITON_VERSION=2.45.0 \
  --build-arg TRITON_CONTAINER_VERSION=24.04 \
  --build-arg TRITON_SERVER_REPO=https://github.com/triton-inference-server/server.git \
  --build-arg TRITON_SERVER_REF=r24.04 \
  --build-arg VLLM_VERSION=0.4.0.post1 \
  --build-arg BUILD_PARALLEL=1 \
  .
```

`BUILD_PARALLEL=1` используется по умолчанию для первого debug прохода. Это медленнее, но не прячет реальную ошибку сборки за параллельным `gmake: *** [Makefile:136: all] Error 2`. После успешной сборки можно попробовать увеличить, например `--build-arg BUILD_PARALLEL=4`.

## Что отличается от текущего flow

- Не используется `Dockerfile.base`.
- Не используется `scripts/build_triton_image.sh`.
- Не используется generated `server/build/docker_build`.
- Не нужен Docker daemon внутри Docker build.
- Source tree Triton всегда клонируется внутри build stage.
- Сборка может отличаться от upstream container-based flow, потому что `build.py --no-container-build` выполняет CMake build напрямую в текущей ОС stage.
- Для Astra/OpenSSL 3 в core build добавлены `-Wno-deprecated-declarations`, иначе старый gRPC из Triton `r24.04` шумит deprecated warning'ами OpenSSL 3.
- В builder stage установлен `libre2-dev`: direct core build компилирует `triton-core/src/filesystem/api.cc`, который включает `re2/re2.h`.

## Риски

- Этот путь пока не прогнан end-to-end.
- Финальный image может отличаться от образа, собранного upstream Triton container-based flow.
- Список runtime dependencies может потребовать уточнения после первой полной сборки.
- vLLM limitation остается прежним: `vllm==0.4.0.post1` / `torch 2.1.2+cu121` не поддерживает RTX 5070 `sm_120`.
- Workaround `Acquire::https::download.astralinux.ru::Verify-Peer "false";` остается техническим долгом и требует замены на корректную CA/OCSP настройку для production hardening.

## Smoke test после сборки

```bash
docker run --rm --gpus all --name triton-unified-smoke \
  -v "$PWD/model_repository:/models:ro" \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  tritonserver:24.04-unified \
  --model-repository=/models
```

```bash
curl -i http://127.0.0.1:8000/v2/health/ready
curl -i http://127.0.0.1:8000/v2/models/python_model/ready
```
