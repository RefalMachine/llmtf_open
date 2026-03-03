import os
import sys
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import re

def extract_info_from_filename(filename):
    base = os.path.basename(filename)
    name, ext = os.path.splitext(base)
    if ext != '.jsonl':
        return None, None, False

    if name.endswith('_test'):
        model = name[:-5]
        return model, 'test', True

    match = re.search(r'^(.*)_iter_(\d+)$', name)
    if match:
        model = match.group(1)
        iter_num = int(match.group(2))
        return model, iter_num, False

    return None, None, False

def read_last_json_score(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = json.load(f)
        score = content[-1].get('leaderboard_score')
        if score is not None:
            return float(score)
        return None
        # # jsons = []
        # # start = -1
        # # brace_count = 0
        # # in_string = False
        # # escape_next = False
        
        # # for i, char in enumerate(content):
        # #     if char == '\\' and not escape_next:
        # #         escape_next = True
        # #         continue
        # #     elif escape_next:
        # #         escape_next = False
        # #         continue
                
        # #     if char == '"' and not escape_next:
        # #         in_string = not in_string
        # #         continue
                
        # #     if not in_string:
        # #         if char == '{':
        # #             if brace_count == 0:
        # #                 start = i
        # #             brace_count += 1
        # #         elif char == '}':
        # #             brace_count -= 1
        # #             if brace_count == 0 and start != -1:
        # #                 json_str = content[start:i+1]
        # #                 try:
        # #                     data = json.loads(json_str)
        # #                     jsons.append(data)
        # #                 except json.JSONDecodeError:
        # #                     pass
        # #                 start = -1
        
        # # if not jsons:
        # #     return None
        
        # last_json = jsons[-1]
        # score = last_json.get('leaderboard_score')
        # if score is not None:
        #     return float(score)
        # return None
    except (IOError, KeyError) as e:
        print(f"Ошибка при обработке файла {filepath}: {e}")
        return None

def collect_scores(folder_path):
    scores = defaultdict(dict)
    test_scores = {}

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if not file.endswith('.jsonl'):
                continue
            full_path = os.path.join(root, file)
            model, key, is_test = extract_info_from_filename(file)
            if model is None:
                print(f"Пропускаем файл с нераспознанным именем: {file}")
                continue

            score = read_last_json_score(full_path)
            if score is None:
                print(f"В файле {file} нет leaderboard_score или ошибка чтения")
                continue

            if is_test:
                test_scores[model] = score
            else:
                scores[model][key] = score

    return scores, test_scores

def build_dataframe(scores, test_scores):
    all_models = set(scores.keys()) | set(test_scores.keys())
    all_iters = sorted(set(it for model_dict in scores.values() for it in model_dict.keys()))

    df = pd.DataFrame(index=sorted(all_models), columns=all_iters + ['test'])

    for model in all_models:
        for it in all_iters:
            df.loc[model, it] = scores.get(model, {}).get(it, None)
        df.loc[model, 'test'] = test_scores.get(model, None)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    avg_row = df.mean(axis=0, numeric_only=True)
    df.loc['Average'] = avg_row
    
    return df

def save_markdown_table(df, output_file='leaderboard.md'):
    df_display = df.fillna('')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(df_display.to_markdown())
    print(f"Markdown таблица сохранена в {output_file}")

def plot_scores(df, task_name, output_plot='leaderboard.png'):
    plt.figure(figsize=(12, 7))

    models = [idx for idx in df.index if idx != 'Average']
    iters = [col for col in df.columns if col != 'test']
    if not iters:
        print("Нет данных по итерациям")
        return

    max_iter = max(iters)

    iter_values = {it: [] for it in iters}

    for model in models:
        x_vals = []
        y_vals = []
        for it in iters:
            val = df.loc[model, it]
            if pd.notna(val):
                x_vals.append(it)
                y_vals.append(val)
                iter_values[it].append(val)
        if x_vals:
            plt.plot(x_vals, y_vals, marker='o', linewidth=1.5, alpha=0.7, label=model)

    avg_x = [it for it in iters if iter_values[it]]
    avg_y = [sum(iter_values[it]) / len(iter_values[it]) for it in avg_x]
    if avg_x:
        plt.plot(avg_x, avg_y, marker='D', linewidth=2, color='black',
                 linestyle='--', label='Average (mean)', zorder=10)

    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.title(f'Evolution on {task_name}') # ...
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_plot, dpi=150, bbox_inches='tight')
    print(f"График сохранён в {output_plot}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Сбор leaderboard_score из JSONL-файлов и построение таблицы/графика')
    parser.add_argument('folder', help='Путь к папке с JSONL-файлами')
    parser.add_argument('task_name', help='Название задачи')
    parser.add_argument('--output_dir', '-o', default='./',
                       help='Директория, в которую будут сохранены файлы')
    parser.add_argument('--transpose', '-tr', action='store_true',
                       help='Транспонировать таблицу (строки = итерации/test, столбцы = модели)')
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    
    print(f"Обработка папки: {args.folder}")
    scores, test_scores = collect_scores(args.folder)
    
    if not scores and not test_scores:
        print("Не найдено ни одного файла с данными.")
    
    df = build_dataframe(scores, test_scores)

    path_friendly_task_name = re.sub(r'[^\w\-_\. ]', '_', args.task_name)
    
    if args.transpose:
        df_table = df.T
    else:
        df_table = df
    save_markdown_table(df_table, os.path.join(args.output_dir, f"{path_friendly_task_name}_table.md"))
    
    plot_scores(df, args.task_name, os.path.join(args.output_dir, f"{path_friendly_task_name}_plot.png"))


if __name__ == "__main__":
    main()