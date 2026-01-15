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
    for item in l:
        if len(item) != 2:
            continue
        tag, entity = item
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
        for k, v in tt.items():
            if k in tags:
                tp_total[k] += tt[k]
        for k, v in fn.items():
            if k in tags:
                fn_total[k] += fn[k]
        for k, v in fp.items():
            if k in tags:
                fp_total[k] += fp[k]

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
    def get_gold_entities(self, sample) -> Dict[str, List[str]]:
        pass
    
    @abstractmethod
    def get_answer_str(self, sample) -> str:
        pass

    def get_answer(self, sample) -> str:
        return self.get_answer_str(sample)
    
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
        y_pred_raw = self.extract_answer(gen_pred)
        y_gold_raw = self.get_gold_entities(sample)

        pred_tags = y_pred_raw.keys()
        gold_tags = y_gold_raw.keys()
        combined_tags = set(list(pred_tags) + list(gold_tags))

        y_pred, y_gold = {}, {}
        empty_multiset = Multiset()
        for tag in combined_tags:
            if tag in pred_tags:
                y_pred[tag] = Multiset(y_pred_raw[tag])
            else:
                y_pred[tag] = empty_multiset

            if tag in gold_tags:
                y_gold[tag] = Multiset(y_gold_raw[tag])
            else:
                y_gold[tag] = empty_multiset

        true_positives = {tag: y_pred[tag].intersect(y_gold[tag]).count() for tag in combined_tags}
        false_negatives = {tag: y_gold[tag].subtract(y_pred[tag]).count() for tag in combined_tags}
        false_positives = {tag: y_pred[tag].subtract(y_gold[tag]).count() for tag in combined_tags}

        true_positives = {k: v for k, v in true_positives.items() if v != 0}
        false_negatives = {k: v for k, v in false_negatives.items() if v != 0}
        false_positives = {k: v for k, v in false_positives.items() if v != 0}
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
        y_gold = self.get_gold_entities(sample)

        pred_tags = set([x[0] for x in y_pred if len(x) > 0 and x[0] in self.TAGS])
        gold_tags = set([x[0] for x in y_gold if len(x) > 0 and x[0] in self.TAGS])
        combined_tags = set(list(pred_tags) + list(gold_tags))
        
        y_pred = list_to_dict_multiset(y_pred, combined_tags)
        y_gold = list_to_dict_multiset(y_gold, combined_tags)

        true_positives = {tag: y_pred[tag].intersect(y_gold[tag]).count() for tag in combined_tags}
        false_negatives = {tag: y_gold[tag].subtract(y_pred[tag]).count() for tag in combined_tags}
        false_positives = {tag: y_pred[tag].subtract(y_gold[tag]).count() for tag in combined_tags}

        true_positives = {k: v for k, v in true_positives.items() if v != 0}
        false_negatives = {k: v for k, v in false_negatives.items() if v != 0}
        false_positives = {k: v for k, v in false_positives.items() if v != 0}
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
        y_gold = self.get_gold_entities(sample)

        pred_tags = set([x[0] for x in y_pred if len(x) > 0 and x[0] in self.TAGS])
        gold_tags = set([x[0] for x in y_gold if len(x) > 0 and x[0] in self.TAGS])
        combined_tags = set(list(pred_tags) + list(gold_tags))
        
        y_pred = list_to_dict_multiset(y_pred, combined_tags)
        y_gold = list_to_dict_multiset(y_gold, combined_tags)

        true_positives = {tag: y_pred[tag].intersect(y_gold[tag]).count() for tag in combined_tags}
        false_negatives = {tag: y_gold[tag].subtract(y_pred[tag]).count() for tag in combined_tags}
        false_positives = {tag: y_pred[tag].subtract(y_gold[tag]).count() for tag in combined_tags}

        true_positives = {k: v for k, v in true_positives.items() if v != 0}
        false_negatives = {k: v for k, v in false_negatives.items() if v != 0}
        false_positives = {k: v for k, v in false_positives.items() if v != 0}
        return {"f1-macro": (true_positives, false_negatives, false_positives)}
