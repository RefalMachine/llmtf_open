from llmtf.model import ApiVLLMModel
from llmtf.tasks.llm_as_a_judge import LLMAsJudge
from llmtf.evaluator import Evaluator
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_answers_path')
    parser.add_argument('--reference_answers_path')
    parser.add_argument('--judge_url')
    parser.add_argument('--output_dir')
    args = parser.parse_args()

    model = ApiVLLMModel(args.judge_url)
    model.from_pretrained()

    model.generation_config.max_new_tokens = 1024
    model.generation_config.repetition_penalty = 1.0
    model.generation_config.do_sample = False
    model.generation_config.temperature = 0.0
    model.generation_config
    model.num_procs = 50
    print(model.model_name)

    data_path = args.model_answers_path
    task = LLMAsJudge(model_outputs={'model_name': os.path.basename(data_path), 'path': data_path}, references_outputs=[{'model_name': 'reference', 'path': args.reference_answers_path}], len_control=True)
    evaluator = Evaluator()
    evaluator.evaluate_dataset(
        task=task,
        model=model,
        output_dir=args.output_dir,
        max_len=5000,
        few_shot_count=0,
        generation_config=None,
        batch_size=256,
        max_sample_per_dataset=10000000000
    )
    evaluator.create_report(args.output_dir)