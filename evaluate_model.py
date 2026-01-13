import argparse
from llmtf.model import HFModelReasoning, VLLMModelReasoning
from llmtf.evaluator import Evaluator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path')
    parser.add_argument('--conv_path')
    parser.add_argument('--output_dir')
    parser.add_argument('--dataset_names', nargs='+', default='all')
    parser.add_argument('--device_map', type=str, default='auto')
    parser.add_argument('--few_shot_count', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_sample_per_dataset', type=int, default=10000000000000)
    parser.add_argument('--max_len', type=int, default=4000)
    parser.add_argument('--max_new_tokens_reasoning', type=int, default=None)
    parser.add_argument('--vllm', action='store_true')
    parser.add_argument('--disable_thinking', action='store_true')
    parser.add_argument('--disable_sliding_window', action='store_true')
    parser.add_argument('--disable_prefix_caching', action='store_true')
    parser.add_argument('--force_recalc', action='store_true')
    parser.add_argument('--alpha_scale', type=float, default=1.0)
    parser.add_argument('--not_scale_lm_head', action='store_true')
    parser.add_argument('--ppl_scoring', action='store_true')
    parser.add_argument('--name_suffix', type=str, default=None)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)
    parser.add_argument('--num_return_sequences', type=int, default=1)
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    parser.add_argument('--is_foundational', action='store_true')
    args = parser.parse_args()
    
    evaluator = Evaluator()
    
    MODEL_CLASS = VLLMModelReasoning if args.vllm else HFModelReasoning
    model = MODEL_CLASS(
        conversation_template_path=args.conv_path, device_map=args.device_map, tensor_parallel_size=args.tensor_parallel_size, 
        alpha_scale=args.alpha_scale, not_scale_lm_head=args.not_scale_lm_head, max_seq_len_to_capture=args.max_len)
    model.from_pretrained(args.model_name_or_path, conversation_template_path=args.conv_path, is_foundational=args.is_foundational, max_new_tokens_reasoning=args.max_new_tokens_reasoning)
    
    model.generation_config.temperature = args.temperature
    model.generation_config.repetition_penalty = args.repetition_penalty
    model.generation_config.presence_penalty = args.presence_penalty
    model.generation_config.num_return_sequences = args.num_return_sequences
    model.generation_config.do_sample = True
    if args.temperature == 0.0:
        model.generation_config.do_sample = False
    
    if args.ppl_scoring:
        evaluator.evaluate_ppl(model, args.output_dir, args.dataset_names, args.max_len, args.few_shot_count, batch_size=args.batch_size, max_sample_per_dataset=args.max_sample_per_dataset, force_recalc=args.force_recalc, name_suffix=args.name_suffix)
    else:
        evaluator.evaluate(model, args.output_dir, args.dataset_names, args.max_len, args.few_shot_count, batch_size=args.batch_size, max_sample_per_dataset=args.max_sample_per_dataset, force_recalc=args.force_recalc, name_suffix=args.name_suffix, enable_thinking=not args.disable_thinking)