import argparse
from llmtf.model import ApiVLLMModel
from llmtf.evaluator import Evaluator
import os
import json
import codecs
import re


def load_questions(input_path):
    with codecs.open(input_path, 'r', 'utf-8') as file:
        return json.load(file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_url')
    parser.add_argument('--model_name_or_path')
    parser.add_argument('--api_key')
    parser.add_argument('--model_name')
    parser.add_argument('--benchmark_name')
    parser.add_argument('--disable_thinking', action='store_true')
    parser.add_argument('--disable_filtering_think_block', action='store_true')
    parser.add_argument('--max_new_tokens', type=int, default=4096)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--repetition_penalty', type=float, default=1.05)
    parser.add_argument('--presence_penalty', type=float, default=0.0)

    args = parser.parse_args()
    os.environ['OPENAI_API_KEY'] = args.api_key
    evaluator = Evaluator()
    
    model = ApiVLLMModel(args.base_url, enable_thinking=not args.disable_thinking)
    model.from_pretrained(args.model_name_or_path)
    
    model.generation_config.temperature = args.temperature
    model.generation_config.repetition_penalty = args.repetition_penalty
    model.generation_config.presence_penalty = args.presence_penalty
    model.generation_config.do_sample = True
    model.generation_config.max_new_tokens = args.max_new_tokens
    if args.temperature == 0.0:
        model.generation_config.do_sample = False

    questions = load_questions(f'benchmark/llmaaj/{args.benchmark_name}/questions.json')
    messages = [[{'role': 'user', 'content': r['instruction']}] for r in questions]
    results = model.generate_batch(messages)

    outputs = []

    for i in range(len(questions)):
        generated = results[1][i]
        if not args.disable_filtering_think_block and generated.startswith('<think>'):
            generated = re.sub(r'<think>[\s\S]*?<\/think>', '', generated).strip()

        outputs.append({
            'instruction': questions[i]['instruction'],
            'output': generated,
            'generator': args.model_name,
            'dataset': args.benchmark_name,
            'generated_tokens': results[2][i]['generated_len'][0],
            'datasplit': "eval"
        })

    with codecs.open(f'benchmark/llmaaj/{args.benchmark_name}/model_results/{args.model_name}.json', 'w', 'utf-8') as file:
        json.dump(outputs, file, ensure_ascii=False, indent=4)