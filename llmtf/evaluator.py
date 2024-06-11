#from llmtf.task import *
from llmtf.tasks import TASK_REGISTRY
from llmtf.base import Task
from llmtf.utils import CustomTimer, SimpleTaskLogger, MaxLenContext
import os
import json
from tqdm import tqdm
import codecs
import inspect
import time
import numpy as np
import logging
from logging import FileHandler

 
class Evaluator():
    def __init__(self):
        pass

    def add_new_task(self, task_name, task_cls):
        assert issubclass(task_cls, Task)
        TASK_REGISTRY[task_name] = task_cls

    def evaluate(self, model, output_dir, datasets_names='all', max_len=4096, few_shot_count=5, generation_config=None, batch_size=1, max_sample_per_dataset=100000000):
        self.init_logger(output_dir)

        if generation_config is not None:
            model.logger.warning('Custom generation_config receives full priority over internal generation config. Compose it carefully. Not tested yet!')

        if datasets_names == 'all':
            datasets_names = list(TASK_REGISTRY.keys())
        
        for dataset_name in datasets_names:
            task = TASK_REGISTRY[dataset_name]()
            task.init_logger()
            with MaxLenContext(task, model, max_len, generation_config) as prompt_max_len:
                self.evaluate_dataset(task, model, output_dir, prompt_max_len, few_shot_count, generation_config, batch_size, max_sample_per_dataset)
        
        self.create_report(output_dir)


    def evaluate_dataset(self, task, model, output_dir, max_len, few_shot_count, generation_config, batch_size, max_sample_per_dataset):
        model.add_stop_strings(task.additional_stop_strings)
        with CustomTimer(task.logger, 'Loading Dataset'):
            messages, samples = task.load_dataset(model, max_len, max_sample_per_dataset, few_shot_count)

        metrics = []
        with SimpleTaskLogger(output_dir, task.name) as logger, CustomTimer(task.logger, 'Processing Dataset'):
            for i in tqdm(range(0, len(messages), batch_size)):
                messages_batch = messages[i:i+batch_size]
                messages_batch = {k: [subdict[k] for subdict in messages_batch] for k in messages_batch[0]}
                if generation_config is not None:
                    messages_batch['generation_config'] = generation_config
    
                prompts, y_preds, infos = getattr(model, task.method + '_batch')(**messages_batch)
                for j in range(len(y_preds)):
                    metrics.append(task.evaluate(samples[i+j]['sample'], y_preds[j]))
                    logger.log_sample(samples[i+j]['sample'], y_preds[j], prompts[j], metrics[-1], infos[j])
        
        
        task.logger.info(f'Results for {task.name}:')
        metrics_res = {metric: task.aggregation()[metric]([m[metric] for m in metrics]) for metric in metrics[0].keys()}
        with SimpleTaskLogger(output_dir, task.name + '_total') as logger:
            logger.log_json({'task_name': task.name, 'results': metrics_res, 'leaderboard_result': task.leaderboard_aggregation(metrics_res)})

        with SimpleTaskLogger(output_dir, task.name + '_params') as logger:
            params = {}
            params['custom_generation_config'] = generation_config
            params['model_params'] = model.get_params()
            params['task_params'] = {'max_len': max_len, 'few_shot_count': few_shot_count, 'batch_size': batch_size, 'max_sample_per_dataset': max_sample_per_dataset, 'method': task.method}
            logger.log_json(params)

        model.reset_stop_strings()
        task.logger.info(str(metrics_res))

    def create_report(self, output_dir):
        reports = {}
        for file_name in os.listdir(output_dir):
            if file_name.endswith('_total.jsonl'):
                with codecs.open(os.path.join(output_dir, file_name), 'r', 'utf-8') as file:
                    task_report = json.load(file)
                reports[task_report['task_name']] = task_report['leaderboard_result']
        task_names = sorted(list(reports.keys()))
        task_metrics = [reports[t] for t in task_names]
        task_names = ['mean'] + task_names
        task_metrics = [np.mean(task_metrics)] + task_metrics
        with codecs.open(os.path.join(output_dir, 'evaluation_results.txt'), 'w', 'utf-8') as file:
            file.write('\t'.join(task_names) + '\n')
            file.write('\t'.join([f'{m:.3f}' for m in task_metrics]))

    def init_logger(self, output_dir):
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        default_log_name = 'evaluation_log.txt'
        logger = logging.getLogger('llmtf')

        for handler in logger.handlers:
            if handler.__class__ == logging.FileHandler and handler.baseFilename.endswith(default_log_name):
                logger.removeHandler(handler)

        fh = FileHandler(os.path.join(output_dir, default_log_name))
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(levelname)s: %(asctime)s: %(name)s: %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)


                

