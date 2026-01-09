import argparse
import logging
from pathlib import Path
import os
import json
import numpy as np
from sklearn.utils import resample
from llmtf.tasks import TASK_REGISTRY
from benchmark.task_groups import task_groups

logger = logging.getLogger(__name__)

def extract_task_datas():
    name_to_aggrigation = {}
    for tasks in task_groups:
        params_tg = tasks["params"]
        task_names = params_tg["dataset_names"].split() # as it called in TASK_REGISTRY
        name_suffix = params_tg.get("name_suffix", None)
        for name in task_names:
            task_tr = TASK_REGISTRY.get(name, None)
            if task_tr is None:
                logger.warning(f"task \"{name}\" from task_groups.py is not present in TASK_REGISTRY")
                continue

            task_class = task_tr["class"]
            params_tr = task_tr.get("params", {}).copy()
            if "name_suffix" not in params_tr.keys(): # name_suffix in TASK_REGISTRY params has higher priority
                params_tr["name_suffix"] = name_suffix
            task = task_class(**params_tr)

            task_name = task.run_name()
            task_name_augmented = task_name.replace('/', '_')

            name_to_aggrigation[task_name_augmented] = (
                task_name,
                task.ALLOW_BOOTSTRAPPING,
                task.aggregation(),
                task.leaderboard_aggregation
            )

    for task_info in TASK_REGISTRY.values():
        task_class = task_info["class"]
        task_params = task_info.get("params", {})
        task = task_class(**task_params)
        task_name = task.run_name()
        task_name_augmented = task_name.replace('/', '_')

        if task_name_augmented not in name_to_aggrigation.keys():
            name_to_aggrigation[task_name_augmented] = (
                task_name,
                task.ALLOW_BOOTSTRAPPING,
                task.aggregation(),
                task.leaderboard_aggregation
            )
    
    return name_to_aggrigation

def models_iterator(log_dir):
    # looking through all model directories
    for name_dir in log_dir.iterdir():
        if not name_dir.is_dir():
            continue

        # considering two-level directories
        subdirs = [item for item in name_dir.iterdir() if item.is_dir()]
        
        if not subdirs:
            if name_dir.name[0] != '.':
                yield name_dir, name_dir.name
        else:
            for subdir in subdirs:
                if subdir.name[0] != '.':
                    model_name = f"{name_dir.name}/{subdir.name}"
                    yield subdir, model_name

def read_jsonl(total_path):
    data = []
    with open(total_path, 'r', encoding='utf-8') as f:
        content = f.read()
        parts = content.split('}\n{')        
        if len(parts) > 1:
            data.append(json.loads(parts[0] + '}'))            
            for part in parts[1:-1]:
                data.append(json.loads('{' + part + '}'))
            if parts[-1].strip():
                data.append(json.loads('{' + parts[-1]))
        else:
            if content.strip():
                data.append(json.loads(content))
    return data

def bootstrap(aggregation, metrics, n_bags, n_samples):
    aggrigations = []
    for _ in range(n_bags):
        metrics_bag = resample(metrics, replace=True, n_samples=n_samples)
        metrics_res = {metric: aggregation[metric]([m[metric] for m in metrics_bag]) for metric in metrics[0].keys()}
        leaderboard_res = leaderboard_aggregation(metrics_res)
        aggrigations.append(leaderboard_res)
    mean = np.mean(aggrigations)
    std = np.std(aggrigations)
    return (mean, std)

def save_table(
    model_bench_to_score,
    output_dir,
    filename="results.md",
    show_time=False
    ):
    models = sorted(model_bench_to_score.keys())
    all_tasks = set()
    
    for model_scores in model_bench_to_score.values():
        all_tasks.update(model_scores.keys())

    tasks = sorted(all_tasks)
    
    lines = []
    header = ["Model"]
    if show_time:
        for task in tasks:
            header += [task, "time"]
        header.append("total time")
    else:
        header += tasks
    lines.append("| " + " | ".join(header) + " |")
    
    separator = ["---"] * (len(tasks) + 1)
    lines.append("| " + " | ".join(separator) + " |")

    for model in models:
        total_time = 0.0
        model_scores = model_bench_to_score[model]
        row = [model]
        
        for task in tasks:
            info = model_scores.get(task, {})
            score = info.get("score", "—")
            row.append(score)
            if show_time:
                time = info.get("time", "—")
                if time != "—":
                    total_time += time
                    time = f"{time:.1f}"
                row.append(time)
        if total_time == 0.0:
            row.append("—")
        else:
            row.append(f"{total_time:.0f}")
        
        lines.append("| " + " | ".join(row) + " |")
    
    content = "\n".join(lines)
    
    output_path = Path(output_dir) / filename
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--n_bags', type=int, default=0)
    parser.add_argument('--n_samples', type=int, default=0)
    parser.add_argument('--show_time', action='store_true')

    args = parser.parse_args()
    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir)
    n_bags = args.n_bags
    n_samples = args.n_samples
    show_time = args.show_time
    do_bootstrap = n_bags > 0 and n_samples > 0

    for directory in [log_dir, output_dir]:
        if not directory.exists() or not directory.is_dir():
            logger.error(f"no such directory \"{directory}\"")
            raise ValueError(f"no such directory \"{directory}\"")
    
    name_to_aggrigation = extract_task_datas()

    model_bench_to_score = {}
    for (model_dir, model_name) in models_iterator(log_dir):
        model_bench_to_score[model_name] = {}
        for total_path in model_dir.glob("*_total.jsonl"):
            log_path = str(model_dir / total_path.name.replace("_total.jsonl", ".jsonl"))
            log_name = total_path.name.replace("_total.jsonl", "")
    
            if log_name in name_to_aggrigation.keys():
                task_name, allow_bootstrap, aggregation, leaderboard_aggregation = name_to_aggrigation[log_name]
            else:
                logger.warning(f"task \"{log_name}\" is not found in default task groups or TASK_REGISTRY")
                continue

            model_bench_to_score[model_name][task_name] = {}
            if allow_bootstrap and do_bootstrap:
                try:
                    log = read_jsonl(log_path)
                except Exception as e:
                    logger.error(f"failed to parse \"{task_name}\" task log: {e}")
                    continue

                metrics = []
                for sample in log:
                    metrics.append(sample["metric"])
                try:
                    mean, std = bootstrap(aggregation, metrics, args.n_bags, args.n_samples)
                except Exception as e:
                    logger.error(f"failed to bootstrap results of task \"{task_name}\":\n{e}")
                    continue
                model_bench_to_score[model_name][task_name]["score"] = f"{mean:.3f} ± {std:.3f}"

                if show_time:
                    with open(str(model_dir / total_path.name), 'r', encoding='utf-8') as f:
                        log = json.load(f)
                    time = log["time"]
            else:
                with open(str(model_dir / total_path.name), 'r', encoding='utf-8') as f:
                    log = json.load(f)
                leaderboard_res = log["leaderboard_result"]
                model_bench_to_score[model_name][task_name]["score"] = f"{leaderboard_res:.3f}"
                time = log["time"]
            model_bench_to_score[model_name][task_name]["time"] = time

            if not allow_bootstrap and do_bootstrap:
                logger.warning(f"task \"{task_name}\" does not support bootstrapping")

    save_table(model_bench_to_score, output_dir, show_time=show_time)
        