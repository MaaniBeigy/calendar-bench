FROM python:3.14.4

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Build-time toggle for CUDA-enabled retrieval / embeddings.
#   docker build --build-arg INSTALL_CUDA=1 --build-arg CUDA_INDEX_URL=...
# When INSTALL_CUDA=1 we install a CUDA build of PyTorch + sentence-
# transformers so the local embedding path picks up the GPU at runtime
# (the env var COMPUTE_DEVICE selects which device to bind).  When 0
# (the default) the image stays CPU-only and small.  GPU passthrough
# from the host is the user's responsibility (e.g. `--gpus all` /
# docker-compose `deploy.resources.reservations.devices`).
ARG INSTALL_CUDA=0
ARG CUDA_INDEX_URL=https://download.pytorch.org/whl/cu126
# Pin the torch wheel to a build that explicitly bundles CUDA 12.x
# (look for the `+cuXYZ` local-version suffix).  Without this pin pip
# silently grabs the latest torch from the index — currently 2.11.x —
# which is CUDA 13 only and fails with "NVIDIA driver too old" on any
# driver that does not yet support CUDA 13.
ARG TORCH_PIN=torch==2.10.0+cu126
ENV INSTALL_CUDA=${INSTALL_CUDA} \
    CUDA_INDEX_URL=${CUDA_INDEX_URL} \
    TORCH_PIN=${TORCH_PIN}

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Optional CUDA path: install a GPU build of torch + sentence-transformers
# + the GPU-only Python deps (`stable-baselines3` etc., listed in
# `requirements-gpu.txt`).  torch is installed FIRST from PyTorch's
# own wheel index (pinned via `TORCH_PIN` to a CUDA-12.x build) so
# the subsequent `pip install -r requirements-gpu.txt` sees the GPU
# torch already present and does not re-pull the PyPI `torch` wheel
# (which since 2.12 transitively pulls `cuda-toolkit` +
# `nvidia-*` wheels even on CPU-only hosts — the regression that
# motivated splitting this file).  sentence-transformers comes from
# PyPI on top.
COPY requirements-gpu.txt ./
RUN if [ "${INSTALL_CUDA}" = "1" ]; then \
        pip install --index-url "${CUDA_INDEX_URL}" \
            "${TORCH_PIN}" \
        && pip install sentence-transformers \
        && pip install -r requirements-gpu.txt ; \
    fi

COPY src ./src
COPY tests ./tests
COPY pytest.ini ./pytest.ini
COPY .coveragerc ./.coveragerc
COPY __init__.py ./__init__.py

ENV PYTHONPATH=/app

CMD ["python", "-m", "src.scripts.import_ontologies"]
