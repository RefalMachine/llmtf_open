# Reasoning boundary for ordinary chat continuation

Date: 2026-10-10.

## Change

Before the answer pass, the shared reasoning strategy removes trailing whitespace
from the reasoning body and inserts exactly one newline before
`THINK_CLOSE_MARKER`. This covers natural closure, budget exhaustion and confirmed
EOS. Internal paragraphs, the answer separator and the task's assistant prefill
are preserved. The budget phrase is still added only for budget exhaustion.

The Qwen3.5 template performs the same tail normalization. Previously, an extra
newline before the close marker made Transformers' `continue_final_message`
containment check fail. The original failing example was explicitly constructed
in the reproduction script; it does not establish the frequency or origin of
this formatting in historical model outputs.

`LLM.get_params()` records
`normalization_policy=single_newline_before_close_v1` for reasoning-capable models.
Use a new output identity or explicit recalculation when rerunning old results.
The default API transport remains `chat`; the optional `rendered_prompt` code and
the article launcher's explicit selections were not changed.

## Checks

- Dependency-free suite: 40 passed.
- API backend, backend contract, benchmark config and reasoning termination:
  58 unittest cases passed in separate processes.
- Native Qwen tokenizer: 3 tests passed, including 24 nonempty normalized
  combinations and empty reasoning. The original extra-newline error reproduced;
  normalized inputs passed actual `apply_chat_template` with
  `continue_final_message=True`. No GPU or network was needed for these tests.
- Real HF, local vLLM and CPU-only API: 66 generate/probability samples per
  backend, covering hybrid off/on, forced empty/nonempty EOS and strict reasoning.
  HF also passed 8 PPL samples; API/vLLM explicitly reported unsupported PPL.
- API-specific regression: the unnormalized input returned HTTP 400 from
  `/tokenize`; normalized natural/budget/EOS inputs passed token counting,
  generation and candidate probabilities with matching prompt lengths.
  Two additional full two-pass checks generated a close marker in the real first
  pass and continued through chat (generate and probability). The first-pass
  allowed-token control was test-only.
- API checks explicitly selected `chat` and rejected any request to
  `/v1/completions`. The API client contained neither torch nor vLLM.
- Required compileall and `git diff --check` passed. Host pytest was unavailable.

Model: `Qwen/Qwen3.5-2B`, revision
`15852e8c16360a2fea060d615a32b45270f8a8fc`. GPU: RTX 4090.
Runtime: torch 2.11.0+cu129, CUDA 12.9, Transformers 5.9.0, vLLM 0.21.0.
Context 4096; reasoning budget/floor 32; answer budget 8. vLLM used test-only
eager execution. Existing images were used without rebuilding:

| Profile | Image ID |
| --- | --- |
| API | `d10c2abd2015f960ff251f0eb984ddbab08abb86292adc874cef96d2eb8d7f8f` |
| HF | `2146b9f44aa2964ebda2be52463da49936ec1d6b338baa826125016191611915` |
| vLLM | `d32605e9216b680faa60a510628cbc385d4ae58507125e9ea499697f03ec4722` |

The runtime checks reused the existing EOS validation harness under
`papers/LLMTF_legal/experiments/validation/eos-2026-10-06/`, with separate outputs.
The API copy selected `chat` and asserted that raw completions were never used.
Local evidence is in `/tmp/llmtf-chat-boundary/`; temporary test drivers are
`/tmp/llmtf-chat-boundary-api.py` and `/tmp/llmtf-chat-boundary-cases.py`.
These temporary files and the ignored historical harness are not Git artifacts.

The native-template regression is retained in Git and can be run offline in the
existing HF image with source and HF cache mounted. From the repository root:

```bash
python3 tests/test_refactor_logic.py
python3 -m unittest discover -s tests -p test_reasoning_termination.py
# Run separately from suites which stub Transformers:
LLMTF_REASONING_TEMPLATE_MODEL=/path/to/pinned/Qwen3.5-2B/snapshot \
  python3 -m unittest discover -s tests -p test_reasoning_chat_template.py
```

This is a targeted continuation regression check, not a new full v4 matrix,
Base-model validation, benchmark quality estimate or local/API parity claim.
The three native-tokenizer tests skip unless the local snapshot is explicitly
selected; a skipped host run is not native-template validation.
