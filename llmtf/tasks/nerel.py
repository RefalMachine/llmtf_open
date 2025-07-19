from tqdm.auto import tqdm
import os
from typing import Dict, List, Tuple
import json
import numpy as np
import ast
import re
from datasets import load_dataset
from llmtf.base import Task
import llmtf

class NestedNER(Task):
    def __init__(self, instruction, **kwargs):
        super().__init__(**kwargs)
        self.method = "generate"
        self._max_new_tokens = 256
        self.instruction = instruction
        
    def task_name(self) -> str:
        return 'NEREL-SIMPLE'

    def aggregation(self) -> Dict: 
        return {
            "tp" : np.sum,
            "fp" : np.sum,
            "fn" : np.sum,
            "micro-f1" : lambda data : (2 * sum([d[0] for d in data])) / \
                (2 * sum([d[0] for d in data]) + sum([d[1] for d in data]) + sum([d[2] for d in data]) + 1e-10)
        }
    
    def leaderboard_aggregation(self, metrics: Dict) -> float:
        return metrics['micro-f1']

    def evaluate(self, sample, gen_pred) -> Dict:
        golds = []
        preds = []
            
        sample_entity_types = sample['span_entity_types']
        sample_start_chars = sample['span_entity_start_chars']
        sample_end_chars = sample['span_entity_end_chars']
        for tag, s, e in zip(sample_entity_types, sample_start_chars, sample_end_chars):
            golds.append((tag, sample['context'][s : e], s, e))
        try:
            filtered_pred = gen_pred.replace('Ответ:', '').strip()
            lines = filtered_pred.split('\n')
            for line in lines:
                splits = line.split(':')
                try:
                    tag = splits[0]
                except KeyError:
                    continue
                preds_str = ':'.join(splits[1:])
                try:
                    ps = ast.literal_eval(preds_str.strip())
                    for p in ps:
                        try:
                            res = [(tag, p, int(m.start()), int(m.end())) \
                                    for m in re.finditer(p, sample['context'])]
                        except:
                            print(tag)
                            print(preds_str.strip())
                            print("PS", ps)
                            print(p)
                            print(sample['context'])
                            res = [(tag, p, -1, -1)]
                        preds.extend(res)
                except SyntaxError:
                    continue
                except TypeError:
                    continue
        except ValueError as err:
            pass

        tp = float(len([p for p in preds if p in golds]))
        fp = float(len([p for p in preds if p not in golds]))
        fn = float(len([t for t in golds if t not in preds]))
        
        return {
            "tp" : tp,
            "fp" : fp,
            "fn" : fn,
            "micro-f1" : (tp, fp, fn)
        }

    def create_messages(self, sample, with_answer = False) -> List[Dict]:
        messages = []
        bot_message = 'Ответ:\n'
        if with_answer:
            tagged_lines = []
            for tag in self.tags:
                entities = ['"' + sample['context'][s : e] + '"' 
                            for s, e, t in zip(sample['span_entity_start_chars'], sample['span_entity_end_chars'], sample['span_entity_types']) \
                            if t == tag]
                tagged_lines.append(f'{tag}: [{", ".join(entities)}]')
            bot_message = 'Ответ:\n' + '\n'.join(tagged_lines)
        
        messages.append({'role' : 'user', 'content' : self.instruction.replace('{text}', sample['context'])})
        messages.append({'role' : 'bot', 'content' : bot_message})

        return messages

    def filter_sample(self, sample, tags):
        tag_idx = [i for i in range(len(sample['span_entity_types'])) if sample['span_entity_types'][i] in tags]
        sample['span_entity_types'] = [sample['span_entity_types'][i] for i in tag_idx]
        sample['span_entity_start_chars'] = [sample['span_entity_start_chars'][i] for i in tag_idx]
        sample['span_entity_end_chars'] = [sample['span_entity_end_chars'][i] for i in tag_idx]
        return sample

    def load_dataset(self, model: llmtf.base.LLM, max_len: int, max_sample_per_dataset: int, few_shot_count: int) -> Tuple[List[Dict], List[Dict]]:
        print("Preparing datasets.")
        self.tags = ['DISTRICT', 'CITY', 'STATE_OR_PROVINCE', 'COUNTRY', 'PERSON', 'PROFESSION', 'DATE']

        ds = load_dataset('RefalMachine/nerel_simple')
        prompt_dataset = ds['train']
        test_dataset = ds['test']
        test_dataset = test_dataset.select(range(min(max_sample_per_dataset, len(test_dataset))))

        prompt_dataset = [self.filter_sample(s, self.tags) for s in prompt_dataset]
        test_dataset = [self.filter_sample(s, self.tags) for s in test_dataset]
        
        k = min(len(prompt_dataset), few_shot_count)
        
        messages = []
        for sample in tqdm(test_dataset):
            sample_messages = self.create_messages(sample)
            for i in range(k):
                few_shot_messages = self.create_messages(prompt_dataset[i], with_answer = True)
                few_shot_messages_len = model.count_tokens_for_prompt(model.apply_model_prompt(few_shot_messages + sample_messages))
                if few_shot_messages_len >= max_len:
                    break
                sample_messages = few_shot_messages + sample_messages
            messages.append(sample_messages)

        print("Datasets prepared.")

        messages = [{'messages': m} for m in messages]
        samples = [{'sample': s} for s in test_dataset]
        
        return messages, samples
    
    def get_answer(self, sample):
        tagged_lines = []
        for tag in self.tags:
            entities = ['"' + sample['context'][s : e] + '"' 
                        for s, e, t in zip(sample['span_entity_start_chars'], sample['span_entity_end_chars'], sample['span_entity_types']) \
                        if t == tag]
            tagged_lines.append(f'{tag}: [{", ".join(entities)}]')
        return '\n'.join(tagged_lines)
    
    
class NEREL_BIO(Task):
    def __init__(self, instruction, **kwargs):
        super().__init__(**kwargs)
        self.method = "generate"
        self._max_new_tokens = 512
        self.instruction = instruction
        self.entity_types = ['FINDING', 'DISO', 'INJURY_POISONING', 'PHYS', 'DEVICE', 'LABPROC', 'ANATOMY', 'CHEM']
        
    def task_name(self) -> str:
        return 'NEREL-BIO'

    def aggregation(self) -> Dict: 
        return {
            "micro-f1" : lambda data : (2 * sum([d[0] for d in data])) / \
                (2 * sum([d[0] for d in data]) + sum([d[1] for d in data]) + sum([d[2] for d in data]) + 1e-10)
        }
    
    def leaderboard_aggregation(self, metrics: Dict) -> float:
        return metrics['micro-f1']

    def evaluate(self, sample, gen_pred) -> Dict:
        try:
            gen_pred = gen_pred.replace('```json', '').strip()
            gen_pred = gen_pred.replace('```', '').strip()
            predict = json.loads(gen_pred)
            predict = [p for p in predict if len(p) == 2 and type(p[0]) == str and type(p[1]) == str]
        except:
            predict = []
        
        golds = []
        text = sample['text']
        sample_entity_types = sample['span_entity_types']
        sample_start_chars = sample['span_entity_start_chars']
        sample_end_chars = sample['span_entity_end_chars']
        for tag, s, e in zip(sample_entity_types, sample_start_chars, sample_end_chars):
            golds.append((tag, sample['text'][s:e], s, e))
        predict_verified = []
        entity2lastpos = {p[1]: 0 for p in predict}
        for tag, entity in predict:
            if tag not in self.entity_types:
                continue
            s = text.find(entity, entity2lastpos[entity])
            if s >= 0:
                e = s + len(entity)
                predict_verified.append((tag, sample['text'][s:e], s, e))
                entity2lastpos[entity] = e

        tp = float(len([p for p in predict_verified if p in golds]))
        fp = float(len([p for p in predict_verified if p not in golds]))
        fn = float(len([t for t in golds if t not in predict_verified]))
        return {'micro-f1': (tp, fp, fn)}

    def create_messages(self, sample, with_answer = False) -> List[Dict]:
        messages = []
        bot_message = '```json'
        if with_answer:
            sample_pseudo_predict = list(zip(*[sample['span_entity_types'], sample['span_entity_surface'], sample['span_entity_start_chars']]))
            sample_pseudo_predict = sorted(sample_pseudo_predict, key=lambda x: x[-1])
            sample_pseudo_predict = [p[:2] for p in sample_pseudo_predict]
            sample_pseudo_predict = [p for p in sample_pseudo_predict if p[0] in self.entity_types]
            bot_message = '```json\n' + json.dumps(sample_pseudo_predict, ensure_ascii=False, indent=4).strip() + '\n```'
        
        messages.append({'role' : 'user', 'content' : self.instruction.replace('{text}', sample['text'])})
        messages.append({'role' : 'bot', 'content' : bot_message})

        return messages

    def filter_sample(self, sample, tags):
        tag_idx = [i for i in range(len(sample['span_entity_types'])) if sample['span_entity_types'][i] in tags]
        sample['span_entity_types'] = [sample['span_entity_types'][i] for i in tag_idx]
        sample['span_entity_start_chars'] = [sample['span_entity_start_chars'][i] for i in tag_idx]
        sample['span_entity_end_chars'] = [sample['span_entity_end_chars'][i] for i in tag_idx]
        return sample

    def load_dataset(self, model: llmtf.base.LLM, max_len: int, max_sample_per_dataset: int, few_shot_count: int) -> Tuple[List[Dict], List[Dict]]:
        print("Preparing datasets.")
        ds = load_dataset('RefalMachine/nerel-bio-simple')
        prompt_dataset = ds['validation']
        test_dataset = ds['test']
        test_dataset = test_dataset.select(range(min(max_sample_per_dataset, len(test_dataset))))
        
        k = min(len(prompt_dataset), few_shot_count)
        
        messages = []
        for sample in tqdm(test_dataset):
            sample_messages = self.create_messages(sample)
            for i in range(k):
                few_shot_messages = self.create_messages(prompt_dataset[i], with_answer = True)
                few_shot_messages_len = model.count_tokens_for_prompt(model.apply_model_prompt(few_shot_messages + sample_messages))
                if few_shot_messages_len >= max_len:
                    break
                sample_messages = few_shot_messages + sample_messages
            messages.append(sample_messages)

        print("Datasets prepared.")

        messages = [{'messages': m} for m in messages]
        samples = [{'sample': s} for s in test_dataset]
        
        return messages, samples
    
    def get_answer(self, sample):
        sample_pseudo_predict = list(zip(*[sample['span_entity_types'], sample['span_entity_surface'], sample['span_entity_start_chars']]))
        sample_pseudo_predict = sorted(sample_pseudo_predict, key=lambda x: x[-1])
        sample_pseudo_predict = [p[:2] for p in sample_pseudo_predict]
        sample_pseudo_predict = [p for p in sample_pseudo_predict if p[0] in self.entity_types]
        return '\n' + json.dumps(sample_pseudo_predict, ensure_ascii=False, indent=4).strip() + '\n```'