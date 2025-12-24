## Сборка Triton from source

Для GPU-поддержки Triton обязательно нужны CUDA/cuBLAS/cuDNN, установленные в системные include/lib пути.
Точные версии CUDA/cuDNN/TensorRT под конкретный релиз Triton нужно смотреть в Framework Containers Support Matrix — Triton https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html

```
git clone https://github.com/triton-inference-server/server.git
cd server
# git checkout r25.11
# или другая версия
```

