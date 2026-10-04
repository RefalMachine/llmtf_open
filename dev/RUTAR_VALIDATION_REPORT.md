# RuTaR integration validation — 2026-10-04

## Protocol

New explicit task: `rutar/closed`, `calculate_tokens_proba`, choices `0`/`1`,
accuracy. The two shared legal configs now select it (Foundational k=5,
Instruct k=0), each without a sample limit. No benchmark schema, backend,
model, dependencies, Dockerfiles or runtime pins were changed for this integration.

Pinned data: RuTaR commit `76c6ef0cafe89fe1b400b673478066108b0493ac`,
`rutar.xlsx` SHA-256
`5ce0824ea3d806c2475ac2d1880cfd1795a34ff1ac31a65b94a14d75cd136dc7`.
209 nonempty rows; exclude ID 119 (missing question), ID 117 (duplicate of
114), and five fixed demonstrations (0–4). Evaluation: 202 questions,
118 negative / 84 positive. Demonstration questions and source letters
are excluded for every k. This is the versioned LLMTF closed-book protocol,
not the upstream RAG experiment. See [protocol](../docs/rutar.md).

## Logic and data checks

- `python3 tests/test_refactor_logic.py`: 40 checks, 0 failures, both host and API image.
- `python -m unittest discover -s tests -p test_rutar.py`: 9 tests passed in API image.
- `python -m unittest discover -s tests -p test_legalbench_ru_integration.py`:
  7 tests passed in API image, including YAML commands for HF/vLLM/API and
  the existing-API runner's third task group.
- `python3 -m compileall -q llmtf evaluate_model.py evaluate_model_api.py benchmark show_results.py dev/tools examples`: passed.
- `git diff --check`: passed.
- Pytest is absent on host and in the API image; unittest and the standalone
  pure-logic suite were used.
- Actual pinned XLSX was parsed and checked against its hash, exclusions,
  split size and label distribution.
- The final default source is the published HF Parquet mirror. Anonymous
  `load_dataset` from a clean cache returned all 209 raw rows; row-by-row
  content and Parquet SHA match the local export.
- Final default task loading matches the original XLSX task view exactly:
  207 unique nonempty questions, 202 evaluation questions. A second container
  with `--network none` and `HF_HUB_OFFLINE=1` read the same pinned snapshot
  from cache successfully.
- API profile has no torch or vLLM installed. No dependency or image rebuild
  was needed. Authentication used the existing HF login only for publication;
  default loading and public validation were anonymous.

Tests cover sparse XLSX coordinates, empty rows, shared/inline strings,
Parquet/XLSX equivalence, HF revision pinning, hash rejection, missing questions, duplicate/conflicting labels, fixed k,
no letter-answer leakage, overflow, unknown token count, probability ties
and invalid values, registry opt-in, provenance before loading, fingerprint
changes, evaluator probability dispatch, cache reuse and PPL skip.

## Environment and commands

Existing images, mounted source; no rebuild:

| Profile | Image | Image ID |
|---|---|---|
| API | `llmtf:api` | `d10c2abd2015` |
| HF | `llmtf:hf-cu129` | `2146b9f44aa2` |
| vLLM | `llmtf:vllm-cu129` | `d32605e9216b` |

Docker GPU check passed: NVIDIA GeForce RTX 4090,
`torch.cuda.is_available() == True`, torch `2.11.0+cu129`, CUDA `12.9`,
Transformers `5.9.0`; vLLM `0.21.0+cu129`.

Model snapshots mounted from `/home/mtikhomi/.cache/huggingface`:

- Instruct: `Qwen/Qwen3.5-2B`, `15852e8c16360a2fea060d615a32b45270f8a8fc`.
- Base: `Qwen/Qwen3.5-2B-Base`, `b1485b2fa6dfa1287294f269f5fb618e03d52d7c`.

The following shell variables denote exact paths used in the commands:

```bash
repo=/var/mtikhomi/work_folder/projects/devel/llmtf_open
cache=/home/mtikhomi/.cache/huggingface
instruct=/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc
base=/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c
```

For each backend `hf`/`vllm`, with the matching image above:

```bash
docker run --rm --gpus all --ipc=host \
  -e CUDA_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=1 -e PYTHONPATH=/workdir \
  -v "$repo":/workdir -v "$cache":/root/.cache/huggingface \
  -v /tmp:/audit -w /workdir "$image" \
  python -m dev.tools.validate_rutar --backend "$backend" \
  --model "$instruct" --data /audit/rutar.xlsx \
  --output "/audit/rutar-validation/$backend/instruct"

docker run --rm --gpus all --ipc=host \
  -e CUDA_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=1 -e PYTHONPATH=/workdir \
  -v "$repo":/workdir -v "$cache":/root/.cache/huggingface \
  -v /tmp:/audit -w /workdir "$image" \
  python -m dev.tools.validate_rutar --backend "$backend" \
  --model "$base" --base --data /audit/rutar.xlsx \
  --output "/audit/rutar-validation/$backend/base"
```

The helper selects thinking off, `assistant_prefill_policy=portable`,
context 16000, k=0/5, 8 questions per cell; local vLLM memory utilization .8.
Logs and model artifacts remain under `/tmp/rutar-*.log` and
`/tmp/rutar-validation/`. No credentials are included in commands or artifacts.

## HF publication

Public mirror: [RefalMachine/RuTaR](https://huggingface.co/datasets/RefalMachine/RuTaR),
revision `fdcc020429120cd81945b1b8060f5aa6a9840a18`.
Parquet SHA-256:
`da6cf232b3b513ef37284b6c5f7f807e6e77e643235a4006d62cb3ee25b4de8e`.
The full 209-row table, original XLSX and 480-entry source corpus are preserved.
The card links upstream repository, pinned commit, both original files and
per-record source documents; it describes the conversion and upstream's
unspecified license. No dataset files were committed to LLMTF.

Export/publish command in the API image, with the account's existing HF login
and writable temporary Hub/Xet caches:

```bash
python -m dev.tools.publish_rutar --workbook /audit/rutar.xlsx \
  --sources /audit/rutar-sources.json --output /audit/rutar-hf --upload
```

The upload was explicitly requested by the user. Export validated exact
Parquet roundtrip, and publication validated anonymous `load_dataset` and
raw pinned Parquet hash from fresh caches. The tool refuses to overwrite
an existing different mirror. Its first attempt hit a read-only Xet cache;
repeating with temporary caches completed the publication.

## Runtime results

Every cell below ran 8 questions, k=0/5, thinking off. Initial HF/local-vLLM/API
cells used the original pinned XLSX; the final API repeat used the default
HF loader. API clients ran inside `llmtf:api` against a temporary local
vLLM server for Instruct, on port 18765, context 16000, memory .8,
`--language-model-only --enforce-eager`. The temporary server was stopped.

| Backend/model | k=0 accuracy | k=5 accuracy |
|---|---:|---:|
| HF Instruct, original XLSX | 0.500 | 0.625 |
| HF Base, original XLSX | 0.625 | 0.625 |
| local vLLM Instruct, original XLSX | 0.500 | 0.625 |
| local vLLM Base, original XLSX | 0.625 | 0.625 |
| API Instruct, original XLSX | 0.500 | 0.625 |
| API Instruct, default HF mirror | 0.500 | 0.750 |

Total: 12 totals / 96 processed samples. API repeats had identical samples
and prompts, but differing server probabilities; one k=5 question changed
from an exact probability tie (scored incorrect) to the correct argmax.
This is runtime variation, not a changed mirror input. These small cells
establish task integration in the stated runtime only; they do not establish
local/API numerical parity or a full legal benchmark/model ranking. Base API
and thinking-on were not tested for this task.

Final HF-source API invocation:

```bash
python -m dev.tools.validate_rutar --backend api --model rutar-instruct \
  --output /audit/rutar-hf-validation/api/instruct
```

Inspected matching `_params`/`_total` fingerprints, sample counts, both
candidate probabilities, effective shot traces and dataset provenance.
Final HF-source artifacts: `/tmp/rutar-hf-validation/`; initial matrix:
`/tmp/rutar-validation/`. All underlying data, exclusions and prompt content
are identical between the two sources; fingerprints correctly change to
include the newly pinned HF snapshot.
