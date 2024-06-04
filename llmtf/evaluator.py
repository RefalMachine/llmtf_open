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
    def __init__(self):
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

    def evaluate(self, model, output_dir, datasets_names='all', max_len=4096, few_shot_count=5, generation_config=None, batch_size=1, max_sample_per_dataset=1000000):
        #TODO: max_len setter
        model.generation_config.max_length = max_len
        if generation_config is not None:
            generation_config.max_length = max_len

        if datasets_names == 'all':
            datasets_names = list(self.dataset_types.keys())
        
        for dataset_name in datasets_names:
            task = self.dataset_types[dataset_name]()
            self.evaluate_dataset(task, model, output_dir, max_len, few_shot_count, generation_config, batch_size, max_sample_per_dataset)
        
        self.create_report(output_dir)


    def evaluate_dataset(self, task, model, output_dir, max_len, few_shot_count, generation_config, batch_size, max_sample_per_dataset):
        messages, samples = task.load_dataset(model, max_len, few_shot_count)
        metrics = []
        with SimpleTaskLogger(output_dir, task.name) as logger:
            for i in tqdm(range(0, min(max_sample_per_dataset, len(messages)), batch_size)):
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
        with SimpleTaskLogger(output_dir, task.name + '_total') as logger:
            logger.log_json({'task_name': task.name, 'results': metrics_res})

        with SimpleTaskLogger(output_dir, task.name + '_params') as logger:
            params = {}
            params['custom_generation_config'] = generation_config
            params['model_params'] = model.get_params()
            params['task_params'] = {'max_len': max_len, 'few_shot_count': few_shot_count, 'batch_size': batch_size, 'max_sample_per_dataset': max_sample_per_dataset, 'method': task.method}
            logger.log_json(params)

        print(task.name)
        print(metrics_res)

    def create_report(self, output_dir):
        reports = {}
        for file_name in os.listdir(output_dir):
            if file_name.endswith('_total.jsonl'):
                with codecs.open(os.path.join(output_dir, file_name), 'r', 'utf-8') as file:
                    task_report = json.load(file)
                reports[task_report['task_name']] = task_report['results']
        task_names = sorted(list(reports.keys()))
        task_metrics = [np.mean([reports[t][m] for m in reports[t]]) for t in task_names]
        task_names = ['mean'] + task_names
        task_metrics = [np.mean(task_metrics)] + task_metrics
        with codecs.open(os.path.join(output_dir, 'evaluation_results.txt'), 'w', 'utf-8') as file:
            file.write('\t'.join(task_names) + '\n')
            file.write('\t'.join([f'{m:.3f}' for m in task_metrics]))


                

