#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
api_image=${LLMTF_API_IMAGE:-llmtf:api}
hf_image=${LLMTF_HF_IMAGE:-llmtf:hf-cu129}
vllm_image=${LLMTF_VLLM_IMAGE:-llmtf:vllm-cu129}

docker build -f "$repo_root/docker/Dockerfile.api" -t "$api_image" "$repo_root"
docker run --rm -v "$repo_root:/workdir:ro" "$api_image" \
  python -c "import importlib.util; import llmtf.llm; from llmtf.backends import APIBackend; from llmtf.evaluator import Evaluator; assert importlib.util.find_spec('torch') is None; assert importlib.util.find_spec('vllm') is None"
docker run --rm -v "$repo_root:/workdir:ro" "$api_image" \
  python evaluate_model_api.py --help
docker run --rm -v "$repo_root:/workdir:ro" "$api_image" \
  python -m unittest discover -s tests -p 'test_*.py' -v

docker build -f "$repo_root/docker/Dockerfile.hf" -t "$hf_image" "$repo_root"
docker run --rm --gpus all --ipc=host -v "$repo_root:/workdir:ro" "$hf_image" \
  python -c "import torch, transformers, triton; from fla.utils import get_available_device; from llmtf.backends import HFBackend; assert torch.cuda.is_available(); assert triton.runtime.driver.active.get_current_target().backend == 'cuda'; assert get_available_device() == 'cuda'; print(torch.__version__, torch.version.cuda, transformers.__version__)"

docker build -f "$repo_root/docker/Dockerfile.vllm" \
  --build-arg "HF_BASE_IMAGE=$hf_image" -t "$vllm_image" "$repo_root"
docker run --rm --gpus all --ipc=host -v "$repo_root:/workdir:ro" "$vllm_image" \
  python -c "import shutil, torch, triton, vllm; from fla.utils import get_available_device; from llmtf.backends import HFBackend, VLLMBackend; assert shutil.which('nvcc') is not None; assert torch.cuda.is_available(); assert triton.runtime.driver.active.get_current_target().backend == 'cuda'; assert get_available_device() == 'cuda'; print(torch.__version__, torch.version.cuda, vllm.__version__)"
