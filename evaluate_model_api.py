import argparse
import sys

from llmtf.cli import (
    add_continuation_flags,
    add_thinking_flags,
    merge_backend_kwargs,
    validate_execution_args,
)
from llmtf.evaluator import Evaluator
from llmtf.llm import LLM


def build_parser():
    parser = argparse.ArgumentParser(
        description='Evaluate through an OpenAI-compatible API'
    )
    parser.add_argument('--base_url', required=True)
    parser.add_argument('--model_name_or_path', required=True)
    parser.add_argument(
        '--api_profile', choices=['auto', 'openai', 'vllm'],
        default=argparse.SUPPRESS,
        help=(
            "auto probes optional token counting but sends a conservative "
            "OpenAI payload; vllm enables continuation and vLLM extensions"
        ),
    )
    parser.add_argument('--api_key', default=None,
                        help='Compatibility option; prefer OPENAI_API_KEY')
    parser.add_argument('--output_dir', required=True)
    add_thinking_flags(parser)
    add_continuation_flags(parser, api=True)
    parser.add_argument('--dataset_names', nargs='+', default='all')
    parser.add_argument('--few_shot_count', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_sample_per_dataset', type=int, default=10000000000010)
    parser.add_argument('--model_context_len', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--max_new_tokens_reasoning', type=int, default=4096)
    parser.add_argument('--min_new_tokens_reasoning', type=int, default=1024)
    parser.add_argument('--force_recalc', action='store_true')
    parser.add_argument('--name_suffix', type=str, default=None)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)
    parser.add_argument('--num_return_sequences', type=int, default=1)
    parser.add_argument('--end_thinking_token_id', type=int, default=None)
    parser.add_argument('--model_kind', choices=['plain', 'reasoning', 'hybrid'], default='plain')
    parser.add_argument('--is_foundational', action='store_true')
    parser.add_argument('--ppl_scoring', action='store_true',
                        help='Returns an explicit unsupported-capability failure')
    parser.add_argument('--request_timeout', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--max_retries', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--retry_backoff', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--num_procs', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--redact_endpoint', action=argparse.BooleanOptionalAction,
                        default=argparse.SUPPRESS)
    parser.add_argument('--backend_kwargs', default=None,
                        help='JSON object of APIBackend options; explicit CLI options win')
    return parser


def main(argv=None):
    from llmtf.backends import APIBackend

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_execution_args(
            args.model_kind, args.enable_thinking, ppl=args.ppl_scoring
        )
        if args.enable_thinking and args.model_kind in {'hybrid', 'reasoning'} \
                and args.end_thinking_token_id is None:
            raise ValueError(
                '--enable_thinking requires --end_thinking_token_id for reliable two-pass execution'
            )
        if args.enable_thinking and args.num_return_sequences != 1:
            raise ValueError('Two-pass reasoning requires --num_return_sequences 1')
        values = vars(args)
        explicit = {'api_base': args.base_url}
        if 'api_profile' in values:
            explicit['api_profile'] = values['api_profile']
        if args.api_key is not None:
            explicit['api_key'] = args.api_key
        for name in ('model_context_len', 'request_timeout', 'max_retries',
                     'retry_backoff', 'num_procs', 'redact_endpoint'):
            if name in values:
                explicit[name] = values[name]
        backend_kwargs = merge_backend_kwargs(
            APIBackend, args.backend_kwargs, explicit
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))

    model = LLM(
        backend=APIBackend(**backend_kwargs),
        assistant_prefill_policy=args.assistant_prefill_policy,
        probe_api_prefill=args.probe_api_prefill,
    )
    model.from_pretrained(
        args.model_name_or_path,
        is_foundational=args.is_foundational,
        model_kind=args.model_kind,
        max_new_tokens_reasoning=args.max_new_tokens_reasoning,
        min_new_tokens_reasoning=args.min_new_tokens_reasoning,
        end_thinking_token_id=args.end_thinking_token_id,
    )
    model.generation_config.temperature = args.temperature
    model.generation_config.repetition_penalty = args.repetition_penalty
    model.generation_config.presence_penalty = args.presence_penalty
    model.generation_config.num_return_sequences = args.num_return_sequences
    model.generation_config.do_sample = args.temperature != 0.0

    evaluator = Evaluator()
    common = dict(
        model=model, output_dir=args.output_dir, datasets_names=args.dataset_names,
        few_shot_count=args.few_shot_count, batch_size=args.batch_size,
        max_sample_per_dataset=args.max_sample_per_dataset,
        force_recalc=args.force_recalc, name_suffix=args.name_suffix,
    )
    if args.ppl_scoring:
        summary = evaluator.evaluate_ppl(**common)
    else:
        summary = evaluator.evaluate(
            **common, enable_thinking=args.enable_thinking
        )
    return summary.exit_code


if __name__ == '__main__':
    sys.exit(main())
