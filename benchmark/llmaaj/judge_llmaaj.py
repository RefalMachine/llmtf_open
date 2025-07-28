import argparse
from llmtf.model import ApiVLLMModel
from llmtf.evaluator import Evaluator
from llmtf.tasks.llm_as_a_judge_style_control import LLMAsJudgeStyleControl, confident_score_mean, get_results_from_file
import os
import json
import codecs


def augd(d, i, dlen):
    if i == 0:
        return d + '}'
    if i == dlen - 1:
        return '{' + d
    return '{' + d + '}'

def load_llmtf_results(llmtf_results_path):
    with codecs.open(llmtf_results_path, 'r', 'utf-8') as file:
        data = file.read()
    data = data.split('}\n{')
    data = [json.loads(augd(d, i, len(data)))['metric']['score'] for i, d in enumerate(data)]
    return data

def convert_battles(llmtf_results_path, battles_path):
    results = load_llmtf_results(llmtf_results_path)
    with codecs.open(battles_path, 'w', 'utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--judge_base_url')
    parser.add_argument('--judge_model_name_or_path')
    parser.add_argument('--judge_api_key')
    parser.add_argument('--judge_model_name')
    parser.add_argument('--benchmark_name')
    parser.add_argument('--model_name')
    parser.add_argument('--model_results_path')
    parser.add_argument('--load_prev_results', action='store_true')
    parser.add_argument('--disable_thinking', action='store_true')
    parser.add_argument('--max_len', type=int, default=int(4096*2.5))
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)
    parser.add_argument('--force_recalc', action='store_true')

    args = parser.parse_args()
    os.environ['OPENAI_API_KEY'] = args.judge_api_key
    evaluator = Evaluator()
    
    model = ApiVLLMModel(args.judge_base_url, enable_thinking=not args.disable_thinking)
    model.from_pretrained(args.judge_model_name_or_path)
    
    model.generation_config.temperature = args.temperature
    model.generation_config.repetition_penalty = args.repetition_penalty
    model.generation_config.presence_penalty = args.presence_penalty
    model.generation_config.do_sample = True
    if args.temperature == 0.0:
        model.generation_config.do_sample = False

    idx = 0
    references = []
    baselines_dir = f'benchmark/llmaaj/{args.benchmark_name}/baselines'
    for baseline in os.listdir(baselines_dir):
        if baseline.endswith('.json'):
            references.append(
                {
                    'model_name': baseline[:baseline.rfind('.')],
                    'path': os.path.join(baselines_dir, baseline)
                }
            )
    previous_battles_path = []
    battles_dir = f'benchmark/llmaaj/{args.benchmark_name}/judges/{args.judge_model_name}/battles'
    previous_battles_path = [f'{battles_dir}/{fname}' for fname in os.listdir(battles_dir)]
    curr_battles_path = f'{battles_dir}/{args.model_name}.json'

    if curr_battles_path in previous_battles_path and not args.force_recalc:
        print(f'Results for {args.model_name} already calculated in {battles_dir}. Skip judge.')
    else:
        task = LLMAsJudgeStyleControl(
            model_outputs={'model_name': args.model_name, 'path': args.model_results_path},
            references_outputs=references,
            previous_battles_path=previous_battles_path if args.load_prev_results else []
        )

        output_dir = f'benchmark/llmaaj/{args.benchmark_name}/judges/{args.judge_model_name}/outputs/{args.model_name}'
        evaluator = Evaluator()
        evaluator.evaluate_dataset(
            task=task,
            model=model,
            output_dir=output_dir,
            max_len=args.max_len,
            few_shot_count=0,
            generation_config=None,
            batch_size=256000,
            max_sample_per_dataset=10000000000000000
        )

        convert_battles(f'{output_dir}/llm_as_judge.jsonl', curr_battles_path)
        if curr_battles_path not in previous_battles_path:
            previous_battles_path.append(curr_battles_path)

    if not args.load_prev_results:
        previous_battles_path = [curr_battles_path]
    battles = get_results_from_file(previous_battles_path)
    rating = confident_score_mean(battles, args.model_name)
    print(f'Model: {args.model_name}\nRating: {rating}')
