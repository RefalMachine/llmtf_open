from llmtf.task import *
import os
import json
from tqdm import tqdm
import codecs
import inspect

class SimpleTaskLogger():
    def __init__(self, output_dir, task_name, append=False):
        self.output_dir = output_dir
        self.task_name = task_name.replace('/', '_')
        self.append = append

    def __enter__(self):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        self.file = codecs.open(os.path.join(self.output_dir, self.task_name + '.jsonl'), 'a' if self.append else 'w', 'utf-8')
        return self
 
    def __exit__(self, *args):
        self.file.close()

    def log_sample(self, sample, pred, prompt, metric):
        self.log_json({'metric': metric, 'predict': pred, 'sample': sample, 'prompt': prompt})
    
    def log_json(self, json_data, indent=4):
        self.file.write(json.dumps(json_data, ensure_ascii=False, indent=indent) + '\n')
 
class Evaluator():
    def __init__(self, max_sample_per_dataset):
        self.dataset_types = {
            'MultiQ': MultiQ,
            'PARus': PARus,
            'RCB': RCB,
            'ruMMLU': ruMMLU,
            'ruOpenBookQA': ruOpenBookQA,
            'ruTiE': ruTiE,
            'ruWorldTree': ruWorldTree,
            'RWSD': RWSD,
            'USE': USE
        }
        self.max_sample_per_dataset = max_sample_per_dataset

    def evaluate(self, model, output_dir, datasets_names='all', max_len=4096, few_shot_count=5, generation_config=None, batch_size=1):
        #TODO: max_len setter
        model.generation_config.max_length = max_len

        if datasets_names == 'all':
            datasets_names = list(self.dataset_types.keys())

        summary_logger = SimpleTaskLogger(output_dir, 'summary', append=True)
        with summary_logger:
            for dataset_name in datasets_names:
                task = self.dataset_types[dataset_name]()
                self.evaluate_dataset(task, model, output_dir, summary_logger, max_len, few_shot_count, generation_config, batch_size)

    def evaluate_dataset(self, task, model, output_dir, summary_logger_open, max_len, few_shot_count, generation_config, batch_size):
        logger = SimpleTaskLogger(output_dir, task.name)
        messages, samples = task.load_dataset(model, max_len, few_shot_count)

        metrics = []
        with logger:
            for i in tqdm(range(0, min(self.max_sample_per_dataset, len(messages)), batch_size)):
                messages_batch = messages[i:i+batch_size]
                messages_batch = {k: [subdict[k] for subdict in messages_batch] for k in messages_batch[0]}
                if generation_config is not None:
                    messages_batch['generation_config'] = generation_config

                if batch_size == 1:
                    messages_batch = {k: messages_batch[k][0] if type(messages_batch[k]) == list else messages_batch[k] for k in messages_batch}
                    prompt, y_pred = getattr(model, task.method)(**messages[i])
                    prompts = [prompt]
                    y_preds = [y_pred]
                else:    
                    prompts, y_preds = getattr(model, task.method + '_batch')(**messages_batch)

                for j in range(len(y_preds)):
                    metrics.append(task.evaluate(samples[i+j]['sample'], y_preds[j]))
                    logger.log_sample(samples[i], y_preds[j], prompts[j], metrics[-1])
        
        metrics_res = {metric: task.aggregation()[metric]([m[metric] for m in metrics]) for metric in metrics[0].keys()}
        print(task.name)
        print(metrics_res)
        summary_logger_open.log_json({'task_name': task.name, 'results': metrics_res})