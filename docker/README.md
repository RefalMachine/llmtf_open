# Docker profiles

The images are deliberately separate. `api` has no model runtime, `hf` adds a
CUDA 12.9 / PyTorch 2.11 stack, and `vllm` is built on the resulting HF image.
None of them use NGC.

## Build

Run all commands from the repository root:

```bash
docker build -f docker/Dockerfile.api -t llmtf:api .

docker build -f docker/Dockerfile.hf -t llmtf:hf-cu129 .

docker build \
  -f docker/Dockerfile.vllm \
  --build-arg HF_BASE_IMAGE=llmtf:hf-cu129 \
  -t llmtf:vllm-cu129 .
```

If outbound traffic must use a proxy listening on the Docker host at port
`8118`, Linux builds can reuse it without putting the address into an image:

```bash
docker build \
  --network=host \
  --build-arg HTTP_PROXY=http://127.0.0.1:8118 \
  --build-arg HTTPS_PROXY=http://127.0.0.1:8118 \
  -f docker/Dockerfile.hf \
  -t llmtf:hf-cu129 .
```

Apply the same three options to the API and vLLM builds. They are Docker's
predefined proxy build arguments and are intentionally not declared with
`ARG` or persisted with `ENV` in these Dockerfiles.

The default vLLM wheel is Linux `x86_64`. Override `VLLM_WHEEL_URL` for another
architecture. To reduce kernel build time for a homogeneous fleet, override the
HF build argument, for example:

```bash
docker build \
  -f docker/Dockerfile.hf \
  --build-arg 'FLASH_ATTN_CUDA_ARCHS=80' \
  -t llmtf:hf-cu129-a100 .
```

For this FlashAttention release use `80` for Ampere/Ada (`SM8x`), `90` for
Hopper, `100` for B200/GB200, and `120` for RTX 50. Keep the broad default when
the same image must run across several families. This argument narrows the FA2
build only; `causal-conv1d==1.6.2.post1` uses its own fixed CUDA architecture
list.

## Run

Mount the checkout and caches at runtime; do not copy credentials into an
image:

```bash
docker run --rm -it \
  -v "$PWD:/workdir" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  llmtf:api
```

```bash
docker run --rm -it \
  --gpus all \
  --ipc=host \
  -v "$PWD:/workdir" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  llmtf:hf-cu129
```

```bash
docker run --rm -it \
  --gpus all \
  --ipc=host \
  -v "$PWD:/workdir" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  llmtf:vllm-cu129
```

Pass API keys with `--env`/`--env-file` or an orchestrator secret at runtime.
Never use `ARG`, `ENV`, `COPY`, or image labels for secret values.

## Scope and current limitation

The HF image installs the three optimized paths used by Qwen3.5:

- `flash-attn==2.7.4.post1` for full attention;
- `flash-linear-attention==0.4.0` (Python import `fla`) for Gated DeltaNet;
- `causal-conv1d==1.6.2.post1` for the convolutional state.

All profiles intentionally use `numpy>=2,<2.3`. vLLM 0.21 requires
`opencv-python-headless>=4.13`, and its Python 3.9+ wheels require NumPy 2.
Profile files under `requirements/profiles/` are the only dependency source;
do not add a conflicting monolithic requirements file.

The API dependency profile and its source import path are torch-free.
`APIBackend` uses the framework-owned lightweight sampling config, backend
exports are lazy, and evaluator seeding imports torch only when it is present.
The default API profile sends a conservative OpenAI-compatible payload and does
not require `/tokenize`. Select `--api_profile vllm` to enable vLLM extensions
such as assistant continuation, stop token ids and template kwargs. See
`docs/api_backend.md` for the capability contract.

Run the complete build/import gate with:

```bash
bash docker/validate_profiles.sh
```

The script builds all three images, proves that the API image has neither torch
nor vLLM, and runs CUDA/import checks in the HF and vLLM images. Model-level
generation is a separate GPU validation step and must be repeated after a
runtime or image change.

## GPU smoke check

Run this before model-level tests:

```bash
docker run --rm --gpus all --ipc=host llmtf:vllm-cu129 \
  python -c 'import torch, vllm; print(torch.__version__, torch.version.cuda, vllm.__version__, torch.cuda.get_device_name(0))'
```

This container check is authoritative. A sandboxed host shell may fail to run
`nvidia-smi` even when Docker GPU passthrough works; do not use that failure to
skip the container check. In the current workspace the command above sees an
RTX 4090.

Imports alone do not validate compiled kernels. The validated profiles load
Qwen3.5 instruct/Base checkpoints through HF, local vLLM and vLLM API. Repeat
that model-level smoke whenever images, CUDA, torch, Transformers, vLLM or
compiled kernels change.
