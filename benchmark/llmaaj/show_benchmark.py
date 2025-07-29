import argparse
from llmtf.tasks.llm_as_a_judge import confident_score_mean_with_ties_and_ci, get_results_from_file
import os
import numpy as np


def calculate_avg_length(battles_dir, model_name):
    """
    Calculate the average length of model answers from the model's own battles file,
    excluding sequences that entered a loop (identified by <think>\\n at the beginning of the text).
    
    Args:
        battles_dir: Directory containing the battles JSON files
        model_name: Name of the model to calculate average length for
        
    Returns:
        Average length in characters
    """
    # Construct the filename for the model's own battles file
    model_battles_file = os.path.join(battles_dir, f"{model_name}.json")
    
    # Check if the model's battles file exists
    if not os.path.exists(model_battles_file):
        print(f"Warning: Battles file for {model_name} not found at {model_battles_file}")
        return 0
    
    # Read the model's battles file
    battles = get_results_from_file(model_battles_file)
    
    # Extract answers where the model is the main model (not reference)
    lens = []
    for battle in battles:
        if battle.get('outcome') == 'invalid':
            continue
            
        # Only count answers where the model is the main model
        if battle['model_name'] == model_name:
            answer = battle['answers']['model']
            if not answer.startswith('<think>\n'):
                lens.append(len(answer))
    return int(np.median(lens))

def show_benchmark(benchmark_name, judge_model_name):
    """
    Show benchmark results including ELO ratings and average generation length in markdown format.
    
    Args:
        benchmark_name: Name of the benchmark (e.g., 'ru_arena-hard-v0.1')
        judge_model_name: Name of the judge model (e.g., 'deepseek')
    """
    battles_dir = f'benchmark/llmaaj/{benchmark_name}/judges/{judge_model_name}/battles'
    
    # Get all model names from the battles files
    model_names = set()
    all_battles = []
    
    for filename in os.listdir(battles_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(battles_dir, filename)
            battles = get_results_from_file(filepath)
            all_battles.extend(battles)
            
            # Extract model names
            for battle in battles:
                if battle.get('outcome') != 'invalid':
                    model_names.add(battle['model_name'])
                    model_names.add(battle['reference_model_name'])
    
    # Calculate ELO ratings
    ratings = confident_score_mean_with_ties_and_ci(all_battles, n_bootstrap=100, confidence_level=0.95)
    
    # Calculate average generation length for each model
    length_stats = {}
    for model_name in sorted(model_names):
        avg_length = calculate_avg_length(battles_dir, model_name)
        length_stats[model_name] = avg_length
    
    # Create markdown output with combined table
    markdown_lines = []
    markdown_lines.append(f"# Benchmark Results: {benchmark_name}")
    markdown_lines.append(f"**Judge Model:** {judge_model_name}")
    markdown_lines.append("")
    markdown_lines.append("## Model Performance Summary")
    markdown_lines.append("")
    markdown_lines.append("| Model | ELO Rating | STD | 95% CI | Median Length (chars) |")
    markdown_lines.append("|-------|------------|-----|--------|-------------------|")
    
    # Sort models by ELO rating (descending)
    sorted_models = sorted(ratings.keys(), key=lambda x: ratings[x]['rating'], reverse=True)
    
    for model_name in sorted_models:
        rating = ratings[model_name]['rating']
        std = ratings[model_name]['std']
        ci_lower = ratings[model_name]['ci_lower']
        ci_upper = ratings[model_name]['ci_upper']
        avg_length = length_stats[model_name]
        markdown_lines.append(f"| {model_name} | {rating} | {std} | [{ci_lower}, {ci_upper}] | {avg_length} |")
    
    
    # Print markdown
    print("\n" + "\n".join(markdown_lines))
    
    # Save markdown to results.md file in the battles directory
    results_path = os.path.join(battles_dir, '..', 'results.md')
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(markdown_lines))
    
    return {
        'ratings': ratings,
        'avg_lengths': length_stats,
        'markdown': "\n".join(markdown_lines),
        'results_path': results_path
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show benchmark results with ELO ratings and average generation length')
    parser.add_argument('--benchmark_name', type=str, required=True, help='Name of the benchmark (e.g., ru_arena-hard-v0.1)')
    parser.add_argument('--judge_model_name', type=str, required=True, help='Name of the judge model (e.g., deepseek)')
    
    args = parser.parse_args()
    
    show_benchmark(args.benchmark_name, args.judge_model_name)
