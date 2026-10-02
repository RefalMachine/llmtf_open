"""Run the frozen LegalBench-RU matrix inside a selected Docker profile.

Credentials come only from the environment. Use distinct output directories for
models/backends. Re-running identical cells reuses completed framework totals.
"""
import argparse
import copy
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['hf', 'vllm', 'api'], required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base', action='store_true')
    parser.add_argument('--end-thinking-token-id', type=int, default=248069,
                        help='Pinned Qwen3.5-2B tokenizer close-marker id')
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--thinking-modes', nargs='+', choices=['off', 'on'], default=['off', 'on'])
    parser.add_argument('--assistant-prefill-policy', choices=['auto', 'exact', 'portable', 'best_effort'], default='auto')
    parser.add_argument('--probe-api-prefill', action='store_true')
    parser.add_argument('--api-base', default='http://127.0.0.1:18765')
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--phases', nargs='+',
                        choices=['smoke', 'matrix', 'conditions', 'strict', 'full'],
                        default=['smoke', 'matrix', 'conditions', 'strict'])
    args = parser.parse_args()
    from llmtf.llm import LLM
    from llmtf.evaluator import Evaluator
    from llmtf.reasoning import ModelKind
    from llmtf.tasks import TASK_REGISTRY

    if args.backend == 'hf':
        from llmtf.backends import HFBackend
        backend = HFBackend(model_context_len=32768)
    elif args.backend == 'vllm':
        from llmtf.backends import VLLMBackend
        backend = VLLMBackend(model_context_len=32768, gpu_memory_utilization=.8)
    else:
        from llmtf.backends import APIBackend
        backend = APIBackend(args.api_base, model_context_len=32768,
                             api_profile='vllm', request_timeout=300)
    for name, spec in TASK_REGISTRY.items():
        if name.startswith('legalbench_ru/'):
            spec['params']['data_path'] = args.data
    model = LLM(backend=backend, assistant_prefill_policy=args.assistant_prefill_policy,
                probe_api_prefill=args.probe_api_prefill)
    model.from_pretrained(
        args.model,
        conversation_template_path=(
            'conversation_configs/default_foundational.json' if args.base else 'auto'),
        is_foundational=args.base,
        model_kind='plain' if args.base else 'hybrid',
        end_thinking_token_id=None if args.base else args.end_thinking_token_id,
        max_new_tokens_reasoning=256, min_new_tokens_reasoning=64,
    )
    generation = copy.deepcopy(model.generation_config)
    generation.max_new_tokens = 512
    generation.temperature = 0.0
    generation.do_sample = False
    generation.repetition_penalty = 1.0
    generation.presence_penalty = 0.0
    generation.num_return_sequences = 1
    evaluator = Evaluator()
    summaries = []

    def run(label, mode, shots, thinking=False, limit=8, selection='smoke'):
        name = 'legalbench_ru/' + mode + ('_smoke' if selection == 'smoke' else '')
        summary = evaluator.evaluate(
            model, str(args.output / label / mode), datasets_names=[name],
            few_shot_count=shots, generation_config=generation,
            batch_size=1 if args.backend == 'hf' else 8,
            max_sample_per_dataset=limit, enable_thinking=thinking,
            name_suffix=f'{shots}shot',
        )
        summaries.append({'label': label, 'mode': mode,
                          'exit_code': summary.exit_code, 'failed': summary.failed})
        (args.output / 'validation_summary.json').write_text(json.dumps(summaries, indent=2))
        if summary.exit_code:
            raise RuntimeError(f'Failed cell {label}/{mode}: {summary.failed}')

    phases = {'smoke'} if args.smoke_only else set(args.phases)
    if args.full:
        phases.add('full')
    try:
        if 'smoke' in phases:
            run('initial_smoke', 'closed', 5 if args.base else 0, limit=1)
        if 'matrix' in phases:
            for shots in ((0, 1, 5) if args.base else (0, 5)):
                for thinking in ((False,) if args.base else tuple(m == 'on' for m in args.thinking_modes)):
                    run(f'{shots}shot_thinking_{thinking}', 'closed', shots, thinking)
        if 'conditions' in phases:
            for mode in ('grounded', 'distractor', 'temporal'):
                run('conditions', mode, 5 if args.base else 0)
        if 'strict' in phases and not args.base:
            model.reasoning_config.model_kind = ModelKind.reasoning
            run('strict_reasoning', 'closed', 0, True, limit=1)
        if 'full' in phases:
            model.reasoning_config.model_kind = ModelKind.plain if args.base else ModelKind.hybrid
            for mode in ('closed', 'grounded', 'distractor', 'temporal'):
                run('full', mode, 5 if args.base else 0, limit=100000, selection='full')
    finally:
        close = getattr(model, 'close', None)
        if close:
            close()


if __name__ == '__main__':
    main()
