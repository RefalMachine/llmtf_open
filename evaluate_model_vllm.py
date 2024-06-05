import argparse
from llmtf.model import VLLMModel
from llmtf.evaluator import Evaluator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path')
    parser.add_argument('--conv_path')
    parser.add_argument('--output_dir')
    parser.add_argument('--dataset_names', nargs='+', default='all')
    parser.add_argument('--device_map', type=str, default='auto')
    parser.add_argument('--max_len', type=int, default=2048)
    parser.add_argument('--few_shot_count', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_sample_per_dataset', type=int, default=10000000000000)
    args = parser.parse_args()

    model = VLLMModel(args.conv_path, device_map=args.device_map)
    model.from_pretrained(args.model_name_or_path)

    evaluator = Evaluator()
    evaluator.evaluate(model, args.output_dir, args.dataset_names, args.max_len, args.few_shot_count, batch_size=args.batch_size, max_sample_per_dataset=args.max_sample_per_dataset)