FROM nvcr.io/nvidia/pytorch:26.02-py3

ENV PIP_CONSTRAINT=

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    build-essential \
    ninja-build \
    python3-dev \
    nano && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools packaging ninja

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

RUN python - <<'PY'
import torch
import flash_attn
import transformers
import vllm
from vllm import LLM, SamplingParams

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("flash_attn:", flash_attn.__version__)
print("transformers:", transformers.__version__)
print("vllm:", vllm.__version__)
print("imports OK")
PY

WORKDIR /workdir
CMD ["/bin/bash"]
