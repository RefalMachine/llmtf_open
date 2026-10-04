from abc import abstractmethod
from functools import partial
from typing import List, Dict, Tuple
import re
import json
import hashlib
from pathlib import Path
from llmtf.base import SimpleFewShotHFTask
from llmtf.utils import Multiset
from llmtf.metrics import mean


def parse_ner_json(output):
    """Read one entity array, allowing only an optional enclosing code fence.

    Return validity separately: a malformed answer is not a correct empty answer.
    Entity strings are never normalized or edited by the parser.
    """
    if not isinstance(output, str):
        return [], False
    text = output.strip()
    fence = re.fullmatch(r'```(?:json)?\s*\n?(.*?)\s*```', text, flags=re.DOTALL)
    if fence:
        text = fence.group(1)
    try:
        parsed = json.loads(text)
    except (ValueError, RecursionError):
        return [], False
    if not isinstance(parsed, list):
        return [], False
    for pair in parsed:
        if (not isinstance(pair, list) or len(pair) != 2
                or not all(isinstance(value, str) and value for value in pair)):
            return [], False
    return parsed, True


def list_to_dict_multiset(l, tags):
    d = {tag: [] for tag in tags}
    tags = set(tags)
    for item in l:
        if not isinstance(item, list) or len(item) != 2:
            continue
        tag, entity = item
        if isinstance(tag, str) and tag in tags:
            d[tag].append(entity)
    for tag in tags:
        d[tag] = Multiset(list(filter(lambda x: isinstance(x, str), d[tag])))
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

    def get_task_provenance(self):
        # The concrete task's hash does not cover this shared scoring module.
        return {'ner_scoring_version': 'exact_string_multiset_v2',
                'ner_shared_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


class NerDictAbc(NerAbc):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)

    @abstractmethod
    def get_gold_entities(self, sample) -> Dict[str, List[str]]:
        pass
    
    def get_answer_str(self, sample) -> str:
        answer = self.get_gold_entities(sample)
        answer_str = ""
        for tag, tokens in answer.items():
            answer_str += f"{tag}: [" + ', '.join(tokens) + "]\n"
        return answer_str
    
    def extract_answer(self, gen_pred: str):
        if not isinstance(gen_pred, str):
            return {}
        tag_lines = gen_pred.split('\n')
        answer = {}
        for tag_line in tag_lines:
            match = re.fullmatch(r'\s*([\w-]+):\s*\[(.*)\]\s*', tag_line)
            if match and len(match.groups()) == 2:
                tag = match.group(1)
                tokens = [token.strip() for token in match.group(2).split(',') if token.strip()]
                answer.setdefault(tag, []).extend(tokens)
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

    @abstractmethod
    def get_gold_entities(self, sample) -> List[str]:
        pass
    
    def get_answer_str(self, sample) -> str:
        answer = self.get_gold_entities(sample)
        
        answer_str = '```json\n' + json.dumps(answer, ensure_ascii=False, indent=4).strip() + '\n```'
        return answer_str
    
    def extract_answer(self, gen_pred: str):
        return parse_ner_json(gen_pred)[0]

    def evaluate(self, sample, gen_pred) -> Dict:
        y_pred = self.extract_answer(gen_pred)
        y_gold = self.get_gold_entities(sample)

        combined_tags = set(self.TAGS)
        
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

    def check_text(self, sample, gen_pred: str):
        if not isinstance(gen_pred, str):
            return False
        text_pred = re.sub(r"<\w+>|</\w+>", "", gen_pred)
        # Token spacing is reconstructed by the task; retain every other
        # character, including punctuation outside the old regex allowlist.
        text_gold = ''.join(sample["tokens"])
        return re.sub(r'\s+', '', text_gold) == re.sub(r'\s+', '', text_pred)

    def extract_answer(self, gen_pred: str):
        if not isinstance(gen_pred, str):
            return []
        matches = re.findall(r'<(\w+)>(.*?)</\1>', gen_pred, flags=re.DOTALL)
        return [list(match) for match in matches]
    
    def evaluate(self, sample, gen_pred) -> Dict:
        if not isinstance(gen_pred, str) or not self.check_text(sample, gen_pred):
            y_pred = []
        else:
            y_pred = self.extract_answer(gen_pred)
        y_gold = self.get_gold_entities(sample)

        combined_tags = set(self.TAGS)
        
        y_pred = list_to_dict_multiset(y_pred, combined_tags)
        y_gold = list_to_dict_multiset(y_gold, combined_tags)

        true_positives = {tag: y_pred[tag].intersect(y_gold[tag]).count() for tag in combined_tags}
        false_negatives = {tag: y_gold[tag].subtract(y_pred[tag]).count() for tag in combined_tags}
        false_positives = {tag: y_pred[tag].subtract(y_gold[tag]).count() for tag in combined_tags}

        true_positives = {k: v for k, v in true_positives.items() if v != 0}
        false_negatives = {k: v for k, v in false_negatives.items() if v != 0}
        false_positives = {k: v for k, v in false_positives.items() if v != 0}
        return {"f1-macro": (true_positives, false_negatives, false_positives)}


def get_gold_entities_bio_dict(self, sample) -> Dict[str, List[str]]:
    tagged_tokens = {tag: [] for tag in self.TAGS}
    for token, tag in zip(sample["tokens"], sample["tags"]):
        mod_idx = tag.find('-') + 1
        tag_mod, tag_base = tag[:mod_idx], tag[mod_idx:]

        if tag_base not in self.TAGS:
            continue
        if tag_mod == 'I-':
            tagged_tokens[tag_base][-1] += " " + token
        else:
            tagged_tokens[tag_base].append(token)
    if 'O' in tagged_tokens.keys():
        del tagged_tokens['O']
    return tagged_tokens
    
def get_gold_entities_bio_list(self, sample) -> List[str]:
    tagged_tokens = []
    last_tag_idx = {}
    i = 0
    for token, tag in zip(sample["tokens"], sample["tags"]):
        mod_idx = tag.find('-') + 1
        tag_mod, tag_base = tag[:mod_idx], tag[mod_idx:]

        if tag_base not in self.TAGS:
            continue
        if tag_mod == 'B-':
            tagged_tokens.append([tag_base, token])
            last_tag_idx[tag_base] = i
            i += 1
        elif tag_mod == 'I-':
            tagged_tokens[last_tag_idx[tag_base]][1] += " " + token
        else:
            tagged_tokens.append([tag_base, token])
            i += 1
    return tagged_tokens    

NO_SPACE_AFTER = set(list(".,!?:;»)") + ["..."])
NO_SPACE_BEFORE = set("(«")

def join_tokens(tokens):
    text = ""
    prev_token = ""
    for token in tokens:
        if token not in NO_SPACE_AFTER and prev_token not in NO_SPACE_BEFORE:
            text += " "
        text += token
        prev_token = token
    return text[1:]

def get_answer_str_bio_in_place(self, sample) -> str:
    new_tokens = []
    entity_tokens = []
    prev_tag_base = ""
    for token, tag in zip(sample["tokens"], sample["tags"]):
        mod_idx = tag.find('-') + 1
        tag_mod, tag_base = tag[:mod_idx], tag[mod_idx:]

        if not tag_mod == "I-" and entity_tokens:
            new_tokens.append(f"<{prev_tag_base}>{' '.join(entity_tokens)}</{prev_tag_base}>")
            entity_tokens = []
        if tag_base not in self.TAGS:
            new_tokens.append(token)
        elif tag_mod == "B-":
            entity_tokens = [token]
        elif tag_mod == "I-":
            entity_tokens.append(token)
        else:
            new_tokens.append(f"<{tag_base}>{token}</{tag_base}>")
        prev_tag_base = tag_base
    if len(entity_tokens) > 0:
        new_tokens.append(f"<{tag_base}>{' '.join(entity_tokens)}</{tag_base}>")
    return join_tokens(new_tokens)
