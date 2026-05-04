## Triton Server на Astra Linux: ONNX Runtime, PyTorch, Python

`Dockerfile.astra-onnx-pytorch-python` собирает slim runtime-образ Triton Server на базе Astra Linux.
Образ рассчитан на serving моделей через backend'ы `onnxruntime`, `pytorch` и `python`.

Итоговый образ:

```text
tritonserver:24.04-py3-astra-slim
```

Проверенный размер локального image: `13.7GB`.

Dockerfile использует:

```text
FROM nvcr.io/nvidia/tritonserver:24.04-py3 AS triton
FROM registry.astralinux.ru/library/astra/ubi18:1.8.5
```

В Astra runtime переносятся:

- `/opt/tritonserver`
- Python 3.10 runtime, нужный Python backend'у Triton 24.04
- CUDA 12.4 runtime libraries
- cuDNN / NCCL / минимальный TensorRT runtime, который нужен PyTorch backend'у
- HPCX UCX/UCC/OpenMPI libraries

Образ запускает Triton Server `2.45.0` / release `24.04` на Astra runtime. Проверенные backend'ы: `onnxruntime`, `pytorch`, `python`.

Для уменьшения размера в Dockerfile удаляются:

- backend'ы `tensorflow`, `dali`, `openvino`, `fil`, `tensorrt`;
- TensorRT builder/parser libraries, не нужные для serving текущих ONNX/PyTorch моделей;
- `pip`, `wheel`, Python test/cache payload;
- DCGM validation suite;
- дублирующая установка CUDA runtime через apt.

Сборка:

```bash
docker build \
  -t tritonserver:24.04-py3-astra-slim \
  -f Dockerfile.astra-onnx-pytorch-python \
  .
```

Проверить backend libraries:

```bash
docker run --rm --entrypoint /bin/bash tritonserver:24.04-py3-astra-slim -lc \
  'find /opt/tritonserver/backends -maxdepth 2 -type f -name "libtriton_*.so" -printf "%h/%f\n" | sort'
```

Для `onnxruntime` и `pytorch` также полезно проверить, что нет незакрытых dynamic dependencies:

```bash
docker run --rm --entrypoint /bin/bash tritonserver:24.04-py3-astra-slim -lc \
  'ldd /opt/tritonserver/backends/onnxruntime/libtriton_onnxruntime.so | grep "not found" || true'

docker run --rm --entrypoint /bin/bash tritonserver:24.04-py3-astra-slim -lc \
  'ldd /opt/tritonserver/backends/pytorch/libtriton_pytorch.so | grep "not found" || true'
```

## vLLM flavor на Astra

`Dockerfile.astra-vllm` собирает отдельный Astra runtime только для Triton vLLM backend.

Итоговый образ:

```text
tritonserver:24.04-vllm-python-astra-slim
```

Проверенный размер локального image: `9.97GB`.

Сборка:

```bash
docker build \
  -t tritonserver:24.04-vllm-python-astra-slim \
  -f Dockerfile.astra-vllm \
  .
```

В этом flavor переносится:

- Triton Server `2.45.0`;
- Python backend и `backends/vllm/model.py`;
- Python 3.10 runtime;
- vLLM Python stack из `nvcr.io/nvidia/tritonserver:24.04-vllm-python-py3`;
- CUDA/cuDNN/NCCL libraries из Python wheel-пакетов `site-packages/nvidia`.

Для уменьшения размера не переносится полный CUDA toolkit и не устанавливается полный DCGM пакет. Копируется только `libdcgm.so*`, потому что `tritonserver` бинарно связан с этой библиотекой. Запускать этот flavor лучше с `--allow-gpu-metrics=false`.

Проверка runtime:

```bash
docker run --rm --entrypoint /bin/bash tritonserver:24.04-vllm-python-astra-slim -lc \
  'python3 - <<PY
import torch, vllm
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("vllm", vllm.__version__)
PY'
```

Проверенные версии:

```text
torch 2.1.2+cu121
cuda 12.1
vllm 0.4.0.post1
```

Старт Triton без загрузки модели:

```bash
docker run --rm --gpus all --name triton-vllm \
  -v "$PWD/model_repository:/models:ro" \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  tritonserver:24.04-vllm-python-astra-slim \
  tritonserver --model-repository=/models \
  --model-control-mode=explicit \
  --allow-gpu-metrics=false
```

Ограничение: `vllm_model` из текущего `model_repository` ссылается на `Qwen/Qwen2.5-1.5B-Instruct`, поэтому при загрузке модели Triton будет скачивать веса из Hugging Face, если они не закэшированы. Кроме того, vLLM stack Triton 24.04 использует `torch 2.1.2+cu121`; на RTX 5070 / `sm_120` этот stack ожидаемо не подходит без более нового PyTorch/vLLM/CUDA набора или пересборки CUDA extensions.

## TensorRT flavor на Astra

`Dockerfile.astra-trt` собирает отдельный Astra runtime только для Triton TensorRT backend.

Итоговый образ:

```text
tritonserver:24.04-trt-astra-slim
```

Проверенный размер локального image: `3.35GB`.

Сборка:

```bash
docker build \
  -t tritonserver:24.04-trt-astra-slim \
  -f Dockerfile.astra-trt \
  .
```

В этом flavor переносится:

- Triton Server `2.45.0`;
- TensorRT backend;
- TensorRT runtime/parser/builder resource libraries из `nvcr.io/nvidia/tritonserver:24.04-py3`;
- минимальный набор CUDA libraries, который нужен TensorRT backend'у.

Для уменьшения размера не переносятся Python/ONNX/PyTorch/TensorFlow/DALI/OpenVINO/FIL backend'ы и полный CUDA toolkit. Полный DCGM пакет не устанавливается; копируется только `libdcgm.so*`, потому что `tritonserver` бинарно связан с этой библиотекой. Запускать этот flavor лучше с `--allow-gpu-metrics=false`.

Старт Triton без загрузки модели:

```bash
docker run --rm --gpus all --name triton-trt \
  -v "$PWD/model_repository:/models:ro" \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  tritonserver:24.04-trt-astra-slim \
  tritonserver --model-repository=/models \
  --model-control-mode=explicit \
  --allow-gpu-metrics=false
```

Ограничение: `tensorrt_add` из текущего `model_repository` не содержит готовый `.plan`. Для реального теста нужен заранее собранный TensorRT engine, совместимый с GPU и TensorRT `8.6.3` из Triton 24.04. На RTX 5070 / `sm_120` этот старый TensorRT stack может не подойти.

## Тестовые модели

`model_repository/` содержит тестовые модели для проверки backend'ов и Triton API.

### `onnx_add`

Backend: `onnxruntime`.

Файлы:

```text
model_repository/onnx_add/config.pbtxt
model_repository/onnx_add/1/model.onnx
```

Контракт:

- `INPUT0`: `FP32`, shape `[1, 4]`
- `INPUT1`: `FP32`, shape `[1, 4]`
- `OUTPUT0`: `FP32`, shape `[1, 4]`
- операция: `OUTPUT0 = INPUT0 + INPUT1`

### `pytorch_addsub`

Backend: `pytorch`.

Файлы:

```text
model_repository/pytorch_addsub/config.pbtxt
model_repository/pytorch_addsub/1/model.pt
```

Контракт:

- `INPUT0`: `FP32`, shape `[4]`
- `INPUT1`: `FP32`, shape `[4]`
- `OUTPUT0`: `FP32`, shape `[4]`
- `OUTPUT1`: `FP32`, shape `[4]`
- операции:
  - `OUTPUT0 = INPUT0 + INPUT1`
  - `OUTPUT1 = INPUT0 - INPUT1`

### `pytorch_addsub_batch`

Backend: `pytorch`.

Это batched-вариант `pytorch_addsub` для проверки dynamic batching.

Файлы:

```text
model_repository/pytorch_addsub_batch/config.pbtxt
model_repository/pytorch_addsub_batch/1/model.pt
```

Настройки:

- `max_batch_size: 8`
- `dynamic_batching`
- `preferred_batch_size: [2, 4, 8]`

Контракт одного элемента batch такой же, как у `pytorch_addsub`: два входа `FP32 [4]`, два выхода `FP32 [4]`. В HTTP request shape включает batch dimension, например `[2, 4]`.

### `python_model`

Backend: `python`.

Echo-модель для проверки Python backend и custom metrics.

Контракт:

- `INPUT`: `BYTES`, shape `[1]`
- `OUTPUT`: `BYTES`, shape `[1]`
- операция: возвращает строку `echo: <INPUT>`

### Нерабочие/отложенные модели

- `vllm_model`: в `24.04-py3` нет backend `vllm`, поэтому эта модель не должна загружаться в текущем образе.
- `tensorrt_add`: текущий slim flavor не содержит TensorRT backend, поэтому эта модель не должна загружаться в этом образе.

## Запуск Triton Server

Рекомендуемый режим запуска для текущего `model_repository` - `MODE_EXPLICIT`, чтобы загружать только нужные модели:

```bash
docker run --rm --gpus all --name triton \
  -v "$PWD/model_repository:/models:ro" \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  tritonserver:24.04-py3-astra-slim \
  tritonserver --model-repository=/models \
  --model-control-mode=explicit \
  --load-model=onnx_add \
  --load-model=pytorch_addsub
```

Для batch-теста модель можно загрузить сразу:

```bash
docker run --rm --gpus all --name triton \
  -v "$PWD/model_repository:/models:ro" \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  tritonserver:24.04-py3-astra-slim \
  tritonserver --model-repository=/models \
  --model-control-mode=explicit \
  --load-model=onnx_add \
  --load-model=pytorch_addsub \
  --load-model=pytorch_addsub_batch
```

Если опубликованные Docker порты недоступны из host-среды, выполняйте HTTP-проверки внутри контейнера:

```bash
docker exec triton curl -s http://127.0.0.1:8000/v2/health/ready
```

## Проверка Triton API

Health:

```bash
curl -i http://127.0.0.1:8000/v2/health/live
curl -i http://127.0.0.1:8000/v2/health/ready
```

Model readiness:

```bash
curl -i http://127.0.0.1:8000/v2/models/onnx_add/ready
curl -i http://127.0.0.1:8000/v2/models/pytorch_addsub/ready
```

ONNX HTTP inference:

```bash
curl -s http://127.0.0.1:8000/v2/models/onnx_add/infer \
  -H 'Content-Type: application/json' \
  -d '{
    "inputs": [
      {
        "name": "INPUT0",
        "shape": [1, 4],
        "datatype": "FP32",
        "data": [[1, 2, 3, 4]]
      },
      {
        "name": "INPUT1",
        "shape": [1, 4],
        "datatype": "FP32",
        "data": [[10, 20, 30, 40]]
      }
    ],
    "outputs": [
      { "name": "OUTPUT0" }
    ]
  }' | jq
```

Ожидаемый `OUTPUT0`:

```text
[11.0, 22.0, 33.0, 44.0]
```

PyTorch HTTP inference:

```bash
curl -s http://127.0.0.1:8000/v2/models/pytorch_addsub/infer \
  -H 'Content-Type: application/json' \
  -d '{
    "inputs": [
      {
        "name": "INPUT0",
        "shape": [4],
        "datatype": "FP32",
        "data": [1, 2, 3, 4]
      },
      {
        "name": "INPUT1",
        "shape": [4],
        "datatype": "FP32",
        "data": [10, 20, 30, 40]
      }
    ],
    "outputs": [
      { "name": "OUTPUT0" },
      { "name": "OUTPUT1" }
    ]
  }' | jq
```

Ожидаемые значения:

```text
OUTPUT0 = [11.0, 22.0, 33.0, 44.0]
OUTPUT1 = [-9.0, -18.0, -27.0, -36.0]
```

Python echo inference:

```bash
curl -s http://127.0.0.1:8000/v2/repository/models/python_model/load -X POST

curl -s http://127.0.0.1:8000/v2/models/python_model/infer \
  -H 'Content-Type: application/json' \
  -d '{
    "inputs": [
      {
        "name": "INPUT",
        "shape": [1],
        "datatype": "BYTES",
        "data": ["hello from triton"]
      }
    ],
    "outputs": [
      { "name": "OUTPUT" }
    ]
  }' | jq
```

Ожидаемый `OUTPUT`:

```text
echo: hello from triton
```

Batch inference:

```bash
curl -s -X POST http://127.0.0.1:8000/v2/repository/models/pytorch_addsub_batch/load

curl -s http://127.0.0.1:8000/v2/models/pytorch_addsub_batch/infer \
  -H 'Content-Type: application/json' \
  -d '{
    "inputs": [
      {
        "name": "INPUT0",
        "shape": [2, 4],
        "datatype": "FP32",
        "data": [[1, 2, 3, 4], [5, 6, 7, 8]]
      },
      {
        "name": "INPUT1",
        "shape": [2, 4],
        "datatype": "FP32",
        "data": [[10, 20, 30, 40], [1, 1, 1, 1]]
      }
    ],
    "outputs": [
      { "name": "OUTPUT0" },
      { "name": "OUTPUT1" }
    ]
  }' | jq
```

Ожидаемые значения:

```text
OUTPUT0 = [[11.0, 22.0, 33.0, 44.0], [6.0, 7.0, 8.0, 9.0]]
OUTPUT1 = [[-9.0, -18.0, -27.0, -36.0], [4.0, 5.0, 6.0, 7.0]]
```

Batch stats:

```bash
curl -s http://127.0.0.1:8000/v2/models/pytorch_addsub_batch/stats | jq
```

В ответе должен появиться `batch_stats` с `batch_size: 2`.

Metrics:

```bash
curl -s http://127.0.0.1:8002/metrics | grep -E \
  'nv_inference_count|nv_inference_request_success|nv_gpu_utilization|nv_gpu_memory_used_bytes'
```

Repository load/unload:

```bash
curl -s -X POST http://127.0.0.1:8000/v2/repository/models/onnx_add/unload
curl -i http://127.0.0.1:8000/v2/models/onnx_add/ready

curl -s -X POST http://127.0.0.1:8000/v2/repository/models/onnx_add/load
curl -i http://127.0.0.1:8000/v2/models/onnx_add/ready
```

Repository index:

```bash
curl -s -X POST http://127.0.0.1:8000/v2/repository/index | jq
```

gRPC-тест через SDK image:

```bash
docker run -i --rm --network container:triton \
  --entrypoint python3 \
  nvcr.io/nvidia/tritonserver:24.04-py3-sdk - <<'PY'
import numpy as np
import tritonclient.grpc as grpcclient

client = grpcclient.InferenceServerClient(url="127.0.0.1:8001")
print("live", client.is_server_live())
print("ready", client.is_server_ready())
print("onnx_ready", client.is_model_ready("onnx_add"))

x = np.array([[1, 2, 3, 4]], dtype=np.float32)
y = np.array([[10, 20, 30, 40]], dtype=np.float32)

inputs = [
    grpcclient.InferInput("INPUT0", x.shape, "FP32"),
    grpcclient.InferInput("INPUT1", y.shape, "FP32"),
]
inputs[0].set_data_from_numpy(x)
inputs[1].set_data_from_numpy(y)

result = client.infer(
    "onnx_add",
    inputs=inputs,
    outputs=[grpcclient.InferRequestedOutput("OUTPUT0")],
)
print(result.as_numpy("OUTPUT0").tolist())
PY
```

Ожидаемый вывод:

```text
live True
ready True
onnx_ready True
[[11.0, 22.0, 33.0, 44.0]]
```

## Известные ограничения

- В образе оставлены только backend'ы `onnxruntime`, `pytorch`, `python` и маленькие utility backend'ы `identity`, `repeat`, `square`.
- Backend'ы `tensorflow`, `dali`, `openvino`, `fil`, `tensorrt`, `vllm` отсутствуют.
- TensorRT engine build внутри этого образа не поддерживается: TensorRT backend и builder/parser libraries удалены для уменьшения размера.
- В runtime нет `pip`, `wheel` и Python CLI tooling; Python runtime-библиотеки оставлены для Triton Python backend.
- PyTorch backend использует runtime-библиотеки из Triton/NVIDIA 24.04 stack. Версия PyTorch stack: `2.3.0a0+6ddf5cf85e`.
- `python_model` из `model_repository` является тестовой моделью Python backend: она создает custom metric и отвечает echo-строкой на `BYTES` input.
- `Dockerfile.astra-onnx-pytorch-python` использует `nvcr.io/nvidia/tritonserver:24.04-py3` как source stage для переноса Triton Server и согласованных runtime-библиотек в Astra image.
- Старый vLLM stack `vllm==0.4.0.post1`, который рассматривался ранее, подтягивал `torch 2.1.2+cu121` и не поддерживал RTX 5070 / CUDA capability `sm_120`.
- На GPU с `sm_120` такой `vllm_model` падал с:

```text
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

- Для GPU с `sm_120` нужен другой PyTorch/vLLM/CUDA stack с поддержкой `sm_120` или сборка соответствующих CUDA extensions под эту архитектуру.

## Безопасность и воспроизводимость

- Образ собирается из явно заданных base images: `nvcr.io/nvidia/tritonserver:24.04-py3` и `registry.astralinux.ru/library/astra/ubi18:1.8.5`.
- Runtime-зависимости Triton берутся из того же NGC release image, что снижает риск несовместимости версий backend'ов и системных библиотек.
- `model_repository` содержит как рабочие тестовые модели, так и отложенные заготовки. Для штатного запуска используйте `--model-control-mode=explicit` и перечисляйте модели через `--load-model`.
- Для production hardening отдельно проверьте политику обновления base images, сканирование уязвимостей, доступ к NGC/Astra registry, лимиты контейнера и внешний слой auth/TLS.
