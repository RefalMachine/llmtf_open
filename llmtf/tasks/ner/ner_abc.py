from abc import abstractmethod
from functools import partial
from typing import List, Dict, Tuple
import re
import json
from llmtf.base import SimpleFewShotHFTask
from llmtf.utils import Multiset
from llmtf.metrics import mean

def list_to_dict_multiset(l, tags):
    d = {tag: [] for tag in tags}
    tags = set(tags)
    for (tag, entity) in l:
        if tag in tags:
            d[tag].append(entity)
    for tag in tags:
        d[tag] = Multiset(d[tag])
    return d

def f1_macro(tp_fn_fp, tags):
    tp_total = {k: 0 for k in tags}
    fn_total = {k: 0 for k in tags}
    fp_total = {k: 0 for k in tags}

    for s in tp_fn_fp:
        tt, fn, fp = s
        for tag in tags:
            if tag not in tt.keys():
                continue
            tp_total[tag] += tt[tag]
            fn_total[tag] += fn[tag]
            fp_total[tag] += fp[tag]

    f1 = [2 * tp_total[tag] / (2 * tp_total[tag] + fp_total[tag] + fn_total[tag]) \
          if tp_total[tag] > 0 else 0 for tag in tags]
    return mean(f1)

class NerAbc(SimpleFewShotHFTask):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        assert self.instruction, "no instruction provided for llm"
        self.method = "generate"

    @abstractmethod
    def get_answer(self, sample) -> Dict[str, List[str]]:
        pass
    
    @abstractmethod
    def get_answer_str(self, sample) -> str:
        pass
    
    def create_messages(self, sample, with_answer: bool) -> List[Dict[str, str]]:
        instruction = self.instruction.format(text=sample["query"])
        messages = [
            {"role": "user", "content": instruction},
        ]
        if with_answer:
            messages.append({"role": "assistant", "content": self.get_answer_str(sample)})
        return messages

    @abstractmethod
    def extract_answer(self, gen_pred: str):
        pass

    @abstractmethod
    def evaluate(self, sample, gen_pred: str) -> Dict:
        pass
        
    def aggregation(self) -> Dict:
        return {"f1-macro": partial(f1_macro, tags=self.TAGS)}


class NerDictAbc(NerAbc):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
    
    def extract_answer(self, gen_pred: str):
        tag_lines = gen_pred.split('\n')
        answer = {}
        for tag_line in tag_lines:
            match = re.match(r'((?:\w-)?\w+):\s*\[([^\]]+)\]', tag_line)
            if match and len(match.groups()) == 2:
                tag = match.group(1)
                tokens = [token.strip() for token in match.group(2).split(',')]
                answer[tag] = tokens
        return answer

    def evaluate(self, sample, gen_pred: str) -> Dict:
        y_pred = self.extract_answer(gen_pred)
        y_gold = self.get_answer(sample)
        
        y_pred = {k: Multiset(v) for k, v in y_pred.items()}
        y_gold = {k: Multiset(v) for k, v in y_gold.items()}

        true_positives = {k: v.intersect(y_gold[k]).count() for k, v in y_pred.items() if k in y_gold.keys()}
        false_negatives = {k: v.subtract(y_gold[k]).count() for k, v in y_pred.items() if k in y_gold.keys()}
        false_positives = {k: v.subtract(y_pred[k]).count() for k, v in y_gold.items() if k in y_pred.keys()}
        return {"f1-macro": (true_positives, false_negatives, false_positives)}


class NerJsonAbc(NerAbc):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)

    def extract_answer(self, gen_pred: str):
        try:
            gen_pred = gen_pred.replace('```json', '').strip()
            gen_pred = gen_pred.replace('```', '').strip()
            predict = json.loads(gen_pred)
        except:
            predict = []
        return predict

    def evaluate(self, sample, gen_pred) -> Dict:
        y_pred = self.extract_answer(gen_pred)
        y_gold = self.get_answer(sample)
        
        y_pred = list_to_dict_multiset(y_pred, self.TAGS)
        y_gold = list_to_dict_multiset(y_gold, self.TAGS)

        true_positives = {k: v.intersect(y_gold[k]).count() for k, v in y_pred.items() if k in y_gold.keys()}
        false_negatives = {k: v.subtract(y_gold[k]).count() for k, v in y_pred.items() if k in y_gold.keys()}
        false_positives = {k: v.subtract(y_pred[k]).count() for k, v in y_gold.items() if k in y_pred.keys()}
        return {"f1-macro": (true_positives, false_negatives, false_positives)}


class NerInPlaceAbc(NerAbc):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)

    @abstractmethod
    def check_text(self, gen_pred) -> bool:
        pass

    def extract_answer(self, gen_pred: str):
        matches = re.findall(r'<(\w+)>(.*?)</\1>', gen_pred)
        return [list(match) for match in matches]
    
    def evaluate(self, sample, gen_pred) -> Dict:
        if not self.check_text(sample, gen_pred):
            y_pred = []
        else:
            y_pred = self.extract_answer(gen_pred)
        y_gold = self.get_answer(sample)
        
        y_pred = list_to_dict_multiset(y_pred, self.TAGS)
        y_gold = list_to_dict_multiset(y_gold, self.TAGS)

        true_positives = {k: v.intersect(y_gold[k]).count() for k, v in y_pred.items() if k in y_gold.keys()}
        false_negatives = {k: v.subtract(y_gold[k]).count() for k, v in y_pred.items() if k in y_gold.keys()}
        false_positives = {k: v.subtract(y_pred[k]).count() for k, v in y_gold.items() if k in y_pred.keys()}
        return {"f1-macro": (true_positives, false_negatives, false_positives)}
