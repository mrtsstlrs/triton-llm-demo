FROM nvcr.io/nvidia/tritonserver:24.02-py3 AS triton

FROM registry.astralinux.ru/library/astra/ubi18-python310:1.8.5

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libb64-0d \
    libnuma1 \
    libgomp1 \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb 

RUN apt-get update && apt-get install -y cuda-cudart-12-4 datacenter-gpu-manager

RUN mkdir -p /models

COPY --from=triton /opt/tritonserver /opt/tritonserver

ENV PATH="/opt/tritonserver/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/tritonserver/lib"

EXPOSE 8000 8001 8002

CMD ["tritonserver", "--model-repository=/models"]
