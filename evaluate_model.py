import argparse
import json
import sys

from llmtf.cli import (
    add_continuation_flags,
    add_thinking_flags,
    merge_backend_kwargs,
    validate_execution_args,
)
from llmtf.evaluator import Evaluator
from llmtf.llm import LLM


def _bool_value(value):
    return str(value).lower() not in ('0', 'false', 'no')


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', required=True)
    parser.add_argument('--conv_path', default='auto')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--dataset_names', nargs='+', default='all')
    parser.add_argument('--few_shot_count', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_sample_per_dataset', type=int, default=10000000000000)
    parser.add_argument('--model_context_len', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--max_new_tokens_reasoning', type=int, default=4096)
    parser.add_argument('--min_new_tokens_reasoning', type=int, default=1024)
    parser.add_argument('--end_thinking_token_id', type=int, default=None)
    parser.add_argument('--vllm', action='store_true')
    add_thinking_flags(parser)
    add_continuation_flags(parser)
    parser.add_argument('--disable_sliding_window', action=argparse.BooleanOptionalAction,
                        default=argparse.SUPPRESS)
    parser.add_argument('--disable_prefix_caching', action=argparse.BooleanOptionalAction,
                        default=argparse.SUPPRESS)
    parser.add_argument('--force_recalc', action='store_true')
    parser.add_argument('--ppl_scoring', action='store_true')
    parser.add_argument('--name_suffix', type=str, default=None)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)
    parser.add_argument('--num_return_sequences', type=int, default=1)
    parser.add_argument('--is_foundational', action='store_true')
    parser.add_argument('--model_kind', choices=['plain', 'reasoning', 'hybrid'], default='plain')

    # Backend constructor overrides use SUPPRESS so JSON values survive when a
    # same-named CLI option was not actually supplied.
    parser.add_argument('--device_map', type=str, default=argparse.SUPPRESS)
    parser.add_argument('--tensor_parallel_size', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--alpha_scale', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--not_scale_lm_head', action=argparse.BooleanOptionalAction,
                        default=argparse.SUPPRESS)
    parser.add_argument('--torch_dtype', type=str, default=argparse.SUPPRESS)
    parser.add_argument('--attn_implementation', type=str, default=argparse.SUPPRESS)
    parser.add_argument('--load_in_8bit', action=argparse.BooleanOptionalAction,
                        default=argparse.SUPPRESS)
    parser.add_argument('--trust_remote_code', action=argparse.BooleanOptionalAction,
                        default=argparse.SUPPRESS)
    parser.add_argument('--use_fast_tokenizer', type=_bool_value, default=argparse.SUPPRESS)
    parser.add_argument('--gpu_memory_utilization', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--calculate_tokens_proba_logprobs_count', type=int,
                        default=argparse.SUPPRESS)
    parser.add_argument('--limit_mm_per_prompt', type=json.loads, default=argparse.SUPPRESS)
    parser.add_argument('--backend_kwargs', default=None,
                        help='JSON object of backend constructor options; explicit CLI options win')
    return parser


def _backend_from_args(args):
    if args.vllm:
        from llmtf.backends import VLLMBackend
        backend_cls = VLLMBackend
        names = {
            'device_map', 'tensor_parallel_size', 'model_context_len',
            'gpu_memory_utilization', 'calculate_tokens_proba_logprobs_count',
            'limit_mm_per_prompt', 'trust_remote_code', 'use_fast_tokenizer',
            'disable_sliding_window',
        }
    else:
        from llmtf.backends import HFBackend
        backend_cls = HFBackend
        names = {
            'device_map', 'model_context_len', 'alpha_scale', 'not_scale_lm_head',
            'torch_dtype', 'attn_implementation', 'load_in_8bit',
            'trust_remote_code', 'use_fast_tokenizer',
        }
    values = vars(args)
    explicit = {name: values[name] for name in names if name in values}
    if args.vllm and 'disable_prefix_caching' in values:
        explicit['enable_prefix_caching'] = not values['disable_prefix_caching']
    return backend_cls(**merge_backend_kwargs(
        backend_cls, args.backend_kwargs, explicit
    ))


def main(argv=None):
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
            raise ValueError(
                'Two-pass reasoning requires --num_return_sequences 1'
            )
        backend = _backend_from_args(args)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))

    model = LLM(
        backend=backend,
        assistant_prefill_policy=args.assistant_prefill_policy,
    )
    model.from_pretrained(
        args.model_name_or_path,
        conversation_template_path=args.conv_path,
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
