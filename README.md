# Triton Server на Astra Linux

`Dockerfile.astra-py310-shim` собирает целевой runtime-образ Triton Server на базе Astra Linux.

Итоговый образ:

```text
tritonserver:24.04-astra-shim
```

Dockerfile использует:

```text
FROM nvcr.io/nvidia/tritonserver:24.04-py3 AS triton
FROM registry.astralinux.ru/library/astra/ubi18:1.8.5
```

В Astra runtime переносятся:

- `/opt/tritonserver`
- Python 3.10 runtime, нужный Python backend'у Triton 24.04
- CUDA 12.4 runtime libraries
- cuDNN / NCCL / TensorRT runtime libraries
- HPCX UCX/UCC/OpenMPI libraries

Образ запускает Triton Server `2.45.0` / release `24.04` на Astra runtime. Проверенные backend'ы: `onnxruntime`, `pytorch`, `python`.

Сборка:

```bash
docker build \
  -t tritonserver:24.04-astra-shim \
  -f Dockerfile.astra-py310-shim \
  .
```

Проверить backend libraries:

```bash
docker run --rm --entrypoint /bin/bash tritonserver:24.04-astra-shim -lc \
  'find /opt/tritonserver/backends -maxdepth 2 -type f -name "libtriton_*.so" -printf "%h/%f\n" | sort'
```

Для `onnxruntime` и `pytorch` также полезно проверить, что нет незакрытых dynamic dependencies:

```bash
docker run --rm --entrypoint /bin/bash tritonserver:24.04-astra-shim -lc \
  'ldd /opt/tritonserver/backends/onnxruntime/libtriton_onnxruntime.so | grep "not found" || true'

docker run --rm --entrypoint /bin/bash tritonserver:24.04-astra-shim -lc \
  'ldd /opt/tritonserver/backends/pytorch/libtriton_pytorch.so | grep "not found" || true'
```

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
- `tensorrt_add`: содержит ONNX-заготовку и config для TensorRT, но `.plan` не создан. TensorRT `8.6.3` из Triton 24.04 не смог собрать engine на GPU с `sm_120`.

## Запуск Triton Server

Рекомендуемый режим запуска для текущего `model_repository` - `MODE_EXPLICIT`, чтобы загружать только нужные модели:

```bash
docker run --rm --gpus all --name triton \
  -v "$PWD/model_repository:/models:ro" \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  tritonserver:24.04-astra-shim \
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
  tritonserver:24.04-astra-shim \
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

- Текущий vLLM stack `vllm==0.4.0.post1` подтягивает `torch 2.1.2+cu121`. Он не поддерживает RTX 5070 / CUDA capability `sm_120`.
- На GPU с `sm_120` `vllm_model` падает с:

```text
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

- Для GPU с `sm_120` нужен другой PyTorch/vLLM/CUDA stack с поддержкой `sm_120` или сборка соответствующих CUDA extensions под эту архитектуру.
- `python_model` из `model_repository` является тестовой моделью Python backend: она создает custom metric и отвечает echo-строкой на `BYTES` input.
- `Dockerfile.astra-py310-shim` использует `nvcr.io/nvidia/tritonserver:24.04-py3` как source stage для переноса Triton Server и согласованных runtime-библиотек в Astra image.
- TensorRT backend в образе присутствует, но на `sm_120` TensorRT `8.6.3` из Triton 24.04 не смог собрать тестовый engine через `trtexec`.

## Безопасность и воспроизводимость

- Образ собирается из явно заданных base images: `nvcr.io/nvidia/tritonserver:24.04-py3` и `registry.astralinux.ru/library/astra/ubi18:1.8.5`.
- Runtime-зависимости Triton берутся из того же NGC release image, что снижает риск несовместимости версий backend'ов и системных библиотек.
- `model_repository` содержит как рабочие тестовые модели, так и отложенные заготовки. Для штатного запуска используйте `--model-control-mode=explicit` и перечисляйте модели через `--load-model`.
- Для production hardening отдельно проверьте политику обновления base images, сканирование уязвимостей, доступ к NGC/Astra registry, лимиты контейнера и внешний слой auth/TLS.
