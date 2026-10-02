"""Validate 0/1/5-shot Shlepa artifacts with a real HF, vLLM or API model.

Run from the repository root inside an existing compatible Docker profile.
API credentials are supplied through the environment. Use separate output
directories for models/backends; this smoke does not estimate a full benchmark.
"""
import argparse
import copy
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['hf', 'vllm', 'api'], required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base', action='store_true')
    parser.add_argument('--api-base', default='http://127.0.0.1:18795')
    parser.add_argument('--samples', type=int, default=8)
    parser.add_argument('--ppl', action='store_true')
    args = parser.parse_args()
    if args.ppl and args.backend != 'hf':
        parser.error('--ppl requires the HF backend')

    from llmtf.evaluator import Evaluator
    from llmtf.llm import LLM

    if args.backend == 'hf':
        from llmtf.backends import HFBackend
        backend = HFBackend(model_context_len=16000)
    elif args.backend == 'vllm':
        from llmtf.backends import VLLMBackend
        backend = VLLMBackend(model_context_len=16000)
    else:
        from llmtf.backends import APIBackend
        backend = APIBackend(args.api_base, model_context_len=16000,
                             api_profile='vllm', request_timeout=300, num_procs=4)
    model = LLM(backend=backend)
    model.from_pretrained(
        args.model,
        conversation_template_path=('conversation_configs/default_foundational.json'
                                    if args.base else 'auto'),
        is_foundational=args.base, model_kind='plain' if args.base else 'hybrid',
        end_thinking_token_id=None if args.base else 248069,
    )
    model.generation_config.temperature = 0.0
    model.generation_config.do_sample = False
    names = ['shlepa/moviesmc', 'shlepa/musicmc', 'shlepa/lawmc', 'shlepa/booksmc']
    try:
        evaluator = Evaluator()
        for shots in (0, 1, 5):
            result = evaluator.evaluate(
                model, str(args.output), datasets_names=names, few_shot_count=shots,
                batch_size=8 if args.backend == 'hf' else 10000000,
                max_sample_per_dataset=args.samples, enable_thinking=False,
                name_suffix=f'{shots}shot',
            )
            if result.exit_code:
                raise RuntimeError(result.failed)
        if args.ppl:
            result = evaluator.evaluate_ppl(
                model, str(args.output), datasets_names=['shlepa/lawmc'],
                few_shot_count=5, batch_size=8,
                max_sample_per_dataset=args.samples, name_suffix='5shot_ppl',
            )
            if result.exit_code:
                raise RuntimeError(result.failed)
        if args.backend != 'hf':
            unsupported = evaluator.evaluate_ppl(
                model, str(args.output / 'unsupported_ppl'),
                datasets_names=['shlepa/lawmc'], few_shot_count=5,
                max_sample_per_dataset=1,
            )
            if unsupported.exit_code != 1 or not all(
                    'calculate_logsoftmax' in reason for reason in unsupported.failed.values()):
                raise ValueError('Expected explicit unsupported PPL capability')
        if args.backend == 'api':
            from llmtf.tasks.shlepa import ShlepaSmallMMLU
            task = ShlepaSmallMMLU('Vikhrmodels/law_mc')
            payloads, _ = task.load_dataset(model, 15999, 1, 5)
            generation = copy.deepcopy(model.generation_config)
            generation.max_new_tokens = 1
            _, predictions, _ = model.generate_batch(
                messages=[payloads[0]['messages']],
                generation_config=generation, enable_thinking=False,
            )
            if len(predictions) != 1:
                raise ValueError('API generation batch alignment mismatch')
            (args.output / 'generation_smoke.json').write_text(json.dumps({
                'few_shot_count': 5, 'batch_count': len(predictions),
                'output_length': len(predictions[0]),
            }, indent=2) + '\n')
        entries = []
        for total_path in sorted(args.output.glob('*_total.jsonl')):
            total = json.loads(total_path.read_text())
            samples_path = Path(str(total_path).removesuffix('_total.jsonl') + '.jsonl')
            records = json.loads(samples_path.read_text())
            requested = total['run_config']['task']['few_shot_count']
            for record in records:
                trace = record['sample']['_shlepa']
                if not (trace['requested_shots'] == trace['effective_shots'] == requested
                        and len(trace['demonstration_indices']) == requested
                        and trace['dataset_index'] not in trace['demonstration_indices']):
                    raise ValueError('Artifact demonstration trace mismatch')
            entries.append({
                'task': total['task_name'], 'count': len(records),
                'shots': requested, 'score': total['leaderboard_result'],
                'max_prompt_tokens': max((r['sample']['_shlepa']['prompt_token_count']
                                          or 0) for r in records),
                'run_fingerprint': total['run_fingerprint'],
            })
        if len(entries) != 12 + int(args.ppl):
            raise ValueError('Incomplete Shlepa matrix')
        (args.output / 'validation_summary.json').write_text(json.dumps({
            'backend': args.backend, 'model_kind': 'base' if args.base else 'instruct',
            'sample_limit': args.samples, 'entries': entries,
        }, indent=2) + '\n')
    finally:
        close = getattr(model, 'close', None)
        if close:
            close()


if __name__ == '__main__':
    main()
