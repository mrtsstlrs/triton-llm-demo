# Сборка Triton Server 24.04 из Astra Linux base image

Этот репозиторий содержит wrapper-сборку Triton Inference Server `2.45.0` / `r24.04` из базового образа Astra:

```text
registry.astralinux.ru/library/astra/ubi18-python311:1.8.5
```

Цель сборки: получить локальный образ `tritonserver:latest` без использования `nvcr.io` / NGC container images и без auth в инфраструктуру NVIDIA. Публичные apt-репозитории NVIDIA CUDA используются для CUDA/cuBLAS/cuDNN/TensorRT пакетов.

## Почему не один Dockerfile

Упаковать весь процесс в один Dockerfile технически возможно только не-best-practice способом: запускать Docker-in-Docker внутри build stage или полностью переписать upstream Triton `server/build.py` flow. Это плохо воспроизводится, требует privileged-доступа к Docker daemon и усложняет аудит.

Текущий процесс сознательно разделен на две фазы:

1. `Dockerfile.base` собирает pinованный base image `triton-base:24.04` на базе Astra Linux с CUDA/cuBLAS/cuDNN/TensorRT и Python build deps.
2. `scripts/build_triton_image.sh` клонирует Triton Server source tree при необходимости, запускает upstream `server/build.py --dryrun`, получает generated Dockerfile'ы Triton, применяет Astra-совместимые патчи и запускает generated `server/build/docker_build`.

Практически это эквивалентно двум Dockerfile-фазам, но без vendoring generated Dockerfile'ов из Triton. Это снижает drift относительно upstream `r24.04`: при смене Triton tag generated Dockerfile'ы снова создаются upstream-скриптом, а wrapper явно валидирует версию.

## Требования

- Docker с поддержкой BuildKit.
- NVIDIA Container Toolkit на host, если нужно запускать GPU inference.
- Доступ к интернету на время сборки:
  - `registry.astralinux.ru` для Astra base image.
  - `developer.download.nvidia.com` для публичных CUDA/cuDNN/TensorRT apt packages.
  - `github.com` для Triton component repositories.
  - `download.docker.com` не используется wrapper'ом после патча generated Dockerfile.

## Подготовка Triton source tree

Ручное клонирование не требуется. Если директория `server/` отсутствует, `scripts/build_triton_image.sh` сам выполнит:

```bash
git clone --branch r24.04 --single-branch \
  https://github.com/triton-inference-server/server.git \
  server
```

Если `server/` уже существует, wrapper использует существующий checkout и не делает auto-checkout, чтобы не затирать локальные изменения. В этом случае он только проверяет `server/TRITON_VERSION` и ожидает `2.45.0`.

Source location и ref можно переопределить:

- `TRITON_SERVER_REPO`, по умолчанию `https://github.com/triton-inference-server/server.git`.
- `TRITON_SERVER_DIR`, по умолчанию `<repo-root>/server`.
- `TRITON_SERVER_REF`, по умолчанию `r24.04`.

## Основная команда сборки

```bash
./scripts/build_triton_image.sh
```

Результат:

```text
triton-base:24.04
tritonserver:latest
```

Основные параметры можно переопределить через environment variables:

```bash
TRITON_CONTAINER_VERSION=24.04 \
TRITON_VERSION=2.45.0 \
UPSTREAM_CONTAINER_VERSION=24.04 \
DCGM_VERSION=3.2.6 \
VLLM_VERSION=0.4.0.post1 \
TRITON_SERVER_REPO=https://github.com/triton-inference-server/server.git \
TRITON_SERVER_DIR="$PWD/server" \
TRITON_SERVER_REF=r24.04 \
BASE_IMAGE=triton-base:24.04 \
BUILD_DIR="$PWD/server/build" \
./scripts/build_triton_image.sh
```

`ALLOW_TRITON_VERSION_MISMATCH=1` существует только для осознанной диагностики mixed source/dependency build. Для production-сборки его использовать не нужно.

## Что делает `Dockerfile.base`

`Dockerfile.base`:

- Берет базу `registry.astralinux.ru/library/astra/ubi18-python311:1.8.5`.
- Устанавливает build deps: `build-essential`, `git`, `git-lfs`, `cmake`, `ninja-build`, `pkg-config`, `patchelf`, Python venv deps.
- Подключает публичный NVIDIA CUDA apt repo для Ubuntu 22.04.
- Устанавливает CUDA stack для Triton 24.04:
  - CUDA `12.4`.
  - cuBLAS `12.4.5.8-1`.
  - cuDNN `9.1.0.70-1`.
  - TensorRT `8.6.1.6-1+cuda12.0`.
- Создает `/opt/venv` и устанавливает Python build helpers: `distro`, `packaging`, `requests`, `pyyaml`.

TensorRT версия отличается от NGC runtime для Triton 24.04: в NGC используется TensorRT `8.6.3`, но без NGC auth из публичного CUDA apt repo доступен `8.6.1.6`.

## Что делает `scripts/build_triton_image.sh`

Скрипт выполняет сборку end-to-end:

1. Определяет корень репозитория, версии Triton/CUDA-related компонентов и build directory.
2. Проверяет наличие `TRITON_SERVER_DIR`.
3. Если `TRITON_SERVER_DIR` отсутствует, клонирует `TRITON_SERVER_REPO` на ref `TRITON_SERVER_REF`.
4. Если `TRITON_SERVER_DIR` уже существует, использует его как есть и не меняет branch/ref.
5. Проверяет `server/TRITON_VERSION`. По умолчанию сборка останавливается, если source tree не соответствует `TRITON_VERSION=2.45.0`.
6. Собирает base image:

```bash
docker build -t triton-base:24.04 -f Dockerfile.base .
```

7. Переходит в `server/` и запускает upstream Triton build generator в dry-run режиме:

```bash
python ./build.py \
  --dryrun \
  --no-container-pull \
  --no-container-interactive \
  --version 2.45.0 \
  --container-version 24.04 \
  --upstream-container-version 24.04 \
  --image base,triton-base:24.04 \
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
```

`--dryrun` важен: upstream `build.py` только генерирует `server/build/*`, но не начинает сборку сразу. Это дает wrapper'у возможность пропатчить generated Dockerfile'ы до запуска Docker build.

8. Применяет патчи к generated файлам в `server/build/`.
9. Запускает generated script:

```bash
server/build/docker_build
```

## Патчи generated Dockerfile'ов

Wrapper не меняет upstream source files в `server/`. Все изменения применяются только к generated build artifacts в `server/build/`.

Применяемые изменения:

- `Dockerfile.buildbase`: установка Docker CLI берется из Astra repo через `docker.io`. Это заменяет upstream block с `download.docker.com/linux/ubuntu`, который ломается на Astra codename `1.8_x86-64`.
- `Dockerfile.buildbase`: Kitware apt repo заменен на `pip3 install cmake==3.27.7`, потому что upstream Ubuntu codename logic неприменим к Astra.
- `Dockerfile.buildbase` и runtime `Dockerfile`: DCGM ставится как публичный apt package `datacenter-gpu-manager=1:3.2.6`.
- `Dockerfile.buildbase`: клонируется `triton-inference-server/third_party` tag `r24.04` в `/opt/triton-third-party`.
- `third_party`: для libevent отключаются samples/benchmarks/tests/regress, а проблемный `arc4random_addrandom` участок патчится для совместимости с Astra/glibc.
- `cmake_build`: добавляется `-DFETCHCONTENT_SOURCE_DIR_REPO-THIRD-PARTY=/opt/triton-third-party`, чтобы использовать пропатченный локальный `third_party`.
- `cmake_build`: для Python backend добавляется `-DCMAKE_CXX_FLAGS=-Wno-error=deprecated-declarations`. Это нужно из-за Boost `1.79` и GCC 12, где `std::unary_function` deprecated warning иначе превращается в error.
- Runtime `Dockerfile`: добавляется apt config для Astra repo TLS issue внутри intermediate containers:

```text
Acquire::https::download.astralinux.ru::Verify-Peer "false";
```

Это workaround для build environment. Для production hardening лучше заменить его на корректную установку доверенного CA/OCSP цепочки, если это возможно в целевой инфраструктуре.

- Runtime `Dockerfile`: удаляется upstream CUDA Compat symlink block, который некорректен для текущего Astra/CUDA layout.
- Runtime `Dockerfile`: Python runtime фиксируется на `PYVER=3.11`, чтобы соответствовать base image.
- Runtime `Dockerfile`: pin'ятся Python dependencies для vLLM backend:
  - `numpy<2`
  - `protobuf<5`
  - `huggingface-hub<1`
  - `tokenizers<0.20`
  - `transformers==4.39.3`
  - `vllm==0.4.0.post1`

Эти pins нужны для совместимости Triton 24.04 vLLM backend с Python 3.11 и текущим dependency graph.

## Проверка результата

Проверить, что image собран:

```bash
docker image ls triton-base:24.04 tritonserver:latest
```

Минимальная проверка Python/vLLM packages:

```bash
docker run --rm --entrypoint /bin/bash tritonserver:latest -c \
  '/opt/venv/bin/python -c "import numpy, transformers, torch, vllm; print(numpy.__version__); print(transformers.__version__); print(torch.__version__); print(vllm.__version__)"'
```

Ожидаемые версии для текущей сборки:

```text
numpy 1.26.4
transformers 4.39.3
torch 2.1.2+cu121
vllm 0.4.0.post1
```

Запуск Triton с model repository:

```bash
docker run --rm --gpus all --name triton-smoke \
  -v "$PWD/model_repository:/models:ro" \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  --entrypoint /opt/tritonserver/bin/tritonserver \
  tritonserver:latest \
  --model-repository=/models
```

Readiness:

```bash
curl -i http://127.0.0.1:8000/v2/health/ready
curl -i http://127.0.0.1:8000/v2/models/python_model/ready
```

Если host-среда изолирует опубликованные Docker ports, проверку можно выполнить внутри контейнера:

```bash
docker exec triton-smoke curl -i http://127.0.0.1:8000/v2/health/ready
```

## Известные ограничения

- Текущий vLLM stack `vllm==0.4.0.post1` подтягивает `torch 2.1.2+cu121`. Он не поддерживает RTX 5070 / CUDA capability `sm_120`.
- На RTX 5070 `vllm_model` падает с:

```text
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

- Для RTX 5070 нужен другой PyTorch/vLLM/CUDA stack с поддержкой `sm_120` или сборка соответствующих CUDA extensions под эту архитектуру.
- `python_model` из `model_repository` сейчас является smoke-test моделью загрузки backend: она создает custom metric, но не определяет input/output контракт и не предназначена для полноценного inference request.

## Безопасность и воспроизводимость

- Сборка не использует `nvcr.io` images и не требует NGC auth.
- Upstream source tree автоматически клонируется на `r24.04`, если `server/` отсутствует.
- Если `server/` уже существует, wrapper не выполняет destructive checkout и только валидирует `TRITON_VERSION`.
- Base dependency versions заданы явно в `Dockerfile.base`.
- Generated Dockerfile patches находятся в одном месте: `scripts/build_triton_image.sh`.
- Wrapper не модифицирует upstream `server/` source tree.
- Workaround `Verify-Peer "false"` для Astra apt repo нужно рассматривать как технический долг и заменить на корректную CA/OCSP настройку перед production hardening.
