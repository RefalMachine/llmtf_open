"""Targeted RuTaR runtime check using the pinned HF mirror or a local data file."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('hf', 'vllm', 'api'), required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', help='Optional pinned Parquet or original XLSX for offline checks')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base', action='store_true')
    parser.add_argument('--api-base', default='http://127.0.0.1:18765')
    parser.add_argument('--shots', type=int, nargs='+', default=[0, 5])
    parser.add_argument('--limit', type=int, default=8)
    args = parser.parse_args()
    from llmtf.llm import LLM
    from llmtf.evaluator import Evaluator
    from llmtf.tasks.rutar import RuTaR
    if args.backend == 'hf':
        from llmtf.backends import HFBackend
        backend = HFBackend(model_context_len=16000)
    elif args.backend == 'vllm':
        from llmtf.backends import VLLMBackend
        backend = VLLMBackend(model_context_len=16000, gpu_memory_utilization=.8)
    else:
        from llmtf.backends import APIBackend
        backend = APIBackend(args.api_base, api_profile='vllm', model_context_len=16000)
    model = LLM(backend=backend, assistant_prefill_policy='portable')
    model.from_pretrained(args.model, is_foundational=args.base,
        conversation_template_path='conversation_configs/default_foundational.json' if args.base else 'auto',
        model_kind='plain' if args.base else 'hybrid')
    evaluator = Evaluator()
    evaluator.add_new_task('rutar/closed', RuTaR, {'data_path': args.data}, allow_override=True)
    try:
        for shots in args.shots:
            summary = evaluator.evaluate(model, str(args.output / f'{shots}shot'),
                datasets_names=['rutar/closed'], few_shot_count=shots,
                max_sample_per_dataset=args.limit, batch_size=1 if args.backend == 'hf' else 8,
                enable_thinking=False, name_suffix=f'{shots}shot')
            if summary.exit_code:
                raise RuntimeError(f'RuTaR failed: {summary.failed}')
    finally:
        close = getattr(model.backend, 'close', None)
        if close:
            close()


if __name__ == '__main__':
    main()
