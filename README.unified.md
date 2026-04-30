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
  --build-arg ENABLE_PYTORCH_BACKEND=1 \
  --build-arg PYTORCH_CXX11_ABI=0 \
  .
```

`BUILD_PARALLEL=1` используется по умолчанию для первого debug прохода. Это медленнее, но не прячет реальную ошибку сборки за параллельным `gmake: *** [Makefile:136: all] Error 2`. После успешной сборки можно попробовать увеличить, например `--build-arg BUILD_PARALLEL=4`.

PyTorch backend включён по умолчанию, но без `nvcr.io`: CMake получает `TRITON_PYTORCH_DOCKER_IMAGE=` и пути к torch wheel внутри `/opt/venv`. Для изоляции проблем сборки его можно временно отключить через `--build-arg ENABLE_PYTORCH_BACKEND=0`.

## Что отличается от текущего flow

- Не используется `Dockerfile.base`.
- Не используется `scripts/build_triton_image.sh`.
- Не используется generated `server/build/docker_build`.
- Не нужен Docker daemon внутри Docker build.
- Source tree Triton всегда клонируется внутри build stage.
- Сборка может отличаться от upstream container-based flow, потому что `build.py --no-container-build` выполняет CMake build напрямую в текущей ОС stage.
- Для Astra/OpenSSL 3 в core build добавлены `-Wno-deprecated-declarations`, иначе старый gRPC из Triton `r24.04` шумит deprecated warning'ами OpenSSL 3.
- В builder stage установлен `libre2-dev`: direct core build компилирует `triton-core/src/filesystem/api.cc`, который включает `re2/re2.h`.
- PyTorch backend включён по умолчанию и собирается без `nvcr.io`: `TRITON_PYTORCH_DOCKER_IMAGE` принудительно пустой, а `TRITON_PYTORCH_INCLUDE_PATHS` / `TRITON_PYTORCH_LIB_PATHS` указывают на pip-installed torch в `/opt/venv`.
- Для PyTorch backend отключены TorchTRT и TorchVision, чтобы не тянуть зависимости из NVIDIA PyTorch container.
- Для PyTorch backend добавлен `_GLIBCXX_USE_CXX11_ABI=0`, что соответствует ABI большинства pip wheels PyTorch 2.1.x. Если используемый torch wheel собран с новым ABI, переопределите `--build-arg PYTORCH_CXX11_ABI=1`.
- Для PyTorch backend release flags снижены до `-O1 -DNDEBUG -g0`, потому что компиляция `src/libtorch.cc` против pip LibTorch headers может падать от нехватки памяти без явного compiler diagnostic.
- В `LD_LIBRARY_PATH` добавлены `torch/lib` и `site-packages/nvidia/*/lib`, потому что CUDA/cuDNN/NCCL зависимости PyTorch wheel лежат не в `/usr/local/cuda`, а внутри Python environment.
- ONNX Runtime backend не включён в default unified build: без заранее подготовленных `TRITON_ONNXRUNTIME_INCLUDE_PATHS` / `TRITON_ONNXRUNTIME_LIB_PATHS` upstream `onnxruntime_backend` генерирует Docker build на базе `nvcr.io/nvidia/tritonserver:*`.

## Риски

- Этот путь пока не прогнан end-to-end.
- Финальный image может отличаться от образа, собранного upstream Triton container-based flow.
- Список runtime dependencies может потребовать уточнения после первой полной сборки.
- Triton upstream рекомендует PyTorch backend собирать от NGC PyTorch image или от совместимой custom LibTorch раскладки. В этом Dockerfile используется pip wheel `torch`, чтобы исключить `nvcr.io`; это требует отдельной проверки TorchScript model smoke-тестом.
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
