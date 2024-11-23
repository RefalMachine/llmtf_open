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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.method = "generate"
        self._max_new_token = 128
        
    @classmethod
    def name(cls) -> str:
        return 'NEREL'

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
        
        messages.append({'role' : 'user', 'content' : f'''Извлеки из заданного ниже текста все вложенные именованные сущности всех представленных ниже классов.
Сущности могут быть представлены только целым словом, окружённым пробелами или знаками препинания, либо непрерывной последовательностью целых слов, разделённых пробелами.
Оставь сущности в том виде, в каком они даны в тексте, не изменяй и не склоняй их, иначе тебе будет выставлен штраф 100$.

**Классы**
{self.tags}

DISTRICT - район города.
CITY - город.
STATE_OR_PROVINCE - штат или конкретная область / субьект / округ.
COUNTRY - страна.
PERSON - конкретный человек с ФИО.
PROFESSION - профессия.
DATE - дата.

Требуемый формат для каждого класса: "Класс: ["сущность", ..., "сущность"]". Вместо "Класс" используй соответствующие классы, представленные выше. Сущности каждого класса выведи на отдельной строке.
Если сущностей соответствующего класса в тексте нет, выведи на соответствующей строке "Класс: []".

**Пример**
Будущий ученый тайно покинул дом 15 декабря 1730 года и вскоре он догнал торговый обоз, шедший в Москву.
->
DISTRICT: []
CITY: ["Москву"]
STATE_OR_PROVINCE: []
COUNTRY: []
PERSON: []
PROFESSION: ["ученый"]
DATE: ["15 декабря 1730 года"]

Теперь извлеки вложенные именованные сущности для следующего текста.
**Текст**
{sample['context']}'''})
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