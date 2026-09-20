"""Run the example task through API, Hugging Face, or local vLLM."""

import argparse
import sys


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate the bundled toy task with a selected backend"
    )
    parser.add_argument(
        "--backend", choices=["api", "hf", "vllm"], default="api"
    )
    parser.add_argument("--base-url")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", default="llmtf_example_results")
    parser.add_argument(
        "--api-profile",
        choices=["auto", "openai", "vllm"],
        default="auto",
    )
    parser.add_argument("--model-context-len", type=int, default=None)
    parser.add_argument("--conversation-template", default="auto")
    parser.add_argument("--is-foundational", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--few-shot-count", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=4)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.backend == "api" and not args.base_url:
        parser.error("--base-url is required with --backend api")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        parser.error("--gpu-memory-utilization must be in the interval (0, 1]")

    from examples.custom_task import TASK_NAME, ToySentimentTask
    from llmtf.evaluator import Evaluator
    from llmtf.llm import LLM

    if args.backend == "api":
        from llmtf.backends import APIBackend

        backend = APIBackend(
            api_base=args.base_url,
            model_context_len=args.model_context_len,
            api_profile=args.api_profile,
        )
    elif args.backend == "hf":
        from llmtf.backends import HFBackend

        backend = HFBackend(model_context_len=args.model_context_len)
    else:
        from llmtf.backends import VLLMBackend

        backend = VLLMBackend(
            model_context_len=args.model_context_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
        )
    model = LLM(backend=backend)
    model.from_pretrained(
        args.model_name,
        conversation_template_path=args.conversation_template,
        is_foundational=args.is_foundational,
        model_kind="plain",
    )
    model.generation_config.temperature = 0.0
    model.generation_config.do_sample = False

    evaluator = Evaluator()
    evaluator.add_new_task(TASK_NAME, ToySentimentTask, {})
    summary = evaluator.evaluate(
        model=model,
        output_dir=args.output_dir,
        datasets_names=[TASK_NAME],
        few_shot_count=args.few_shot_count,
        batch_size=args.batch_size,
        max_sample_per_dataset=args.max_samples,
        enable_thinking=False,
    )
    return summary.exit_code


if __name__ == "__main__":
    sys.exit(main())
