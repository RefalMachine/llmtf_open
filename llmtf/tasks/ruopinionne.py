from llmtf.base import SimpleFewShotHFTask
import json
import copy
from tqdm import tqdm
from datasets import Dataset, DatasetDict
from datasets import load_dataset as load_dataset_hf
from llmtf.base import Task, SimpleFewShotHFTask, LLM
import json
from collections import Counter
from typing import Dict, List, Tuple

def tk(text):
    tokens = text.split()
    token_offsets = []
    i = 0
    for token in tokens:
        pos = text[i:].find(token)
        token_offsets.append((i + pos, i + pos + len(token)))
        i += pos + len(token)
    return token_offsets


def check_opinion_exist(htep, opinions_iter, check_diff_spans_valid_func):
    """ This function assess the new htep to be registered with respect to the
        task limitations on span values of `holder`, `target`, and `polarity`
    """

    exist = False

    # Unpack teh original tuple
    h, t, e, p = htep

    for o in opinions_iter:

        # Unpack the registered opinion
        h2, t2, e2, p2 = o

        is_matched = h == h2 and t == t2 and p == p2

        # Check whether `o` and given `htep` are matched.
        if not is_matched:
            continue

        # Extra check in the case when spans differs.
        if e != e2:
            check_diff_spans_valid_func(e, e2)
            continue

        # Otherwise it means that element exist.
        exist = True

    return exist


def convert_char_offsets_to_token_idxs(char_offsets, token_offsets):
    """
    char_offsets: list of str
    token_offsets: list of tuples

    >>> text = "I think the new uni ( ) is a great idea"
    >>> char_offsets = ["8:19"]
    >>> token_offsets =
    [(0,1), (2,7), (8,11), (12,15), (16,19), (20,21), (22,23), (24,26), (27,28), (29,34), (35,39)]

    >>> convert_char_offsets_to_token_idxs(char_offsets, token_offsets)
    >>> (2,3,4)
    """
    token_idxs = []

    for char_offset in char_offsets:
        bidx, eidx = char_offset.split(":")
        bidx, eidx = int(bidx), int(eidx)
        for i, (b, e) in enumerate(token_offsets):
            if b >= eidx or e <= bidx:
                intoken = False
            else:
                intoken = True
            if intoken:
                token_idxs.append(i)
    return frozenset(token_idxs)


def convert_opinion_to_tuple(sentence):
    text = sentence["text"]
    opinions = sentence["opinions"]
    opinion_tuples = []
    token_offsets = tk(text)

    if len(opinions) > 0:
        for opinion in opinions:

            # Extract idxs parts.
            holder_char_idxs = opinion["Source"][1]
            target_char_idxs = opinion["Target"][1]
            exp_char_idxs = opinion["Polar_expression"][1]

            # Compose elements of the new opinion.
            holder = frozenset(["AUTHOR"]) \
                if holder_char_idxs[0] == "NULL" \
                else convert_char_offsets_to_token_idxs(holder_char_idxs, token_offsets)
            target = convert_char_offsets_to_token_idxs(target_char_idxs, token_offsets)
            exp = convert_char_offsets_to_token_idxs(exp_char_idxs, token_offsets)
            polarity = opinion["Polarity"]

            assert polarity in ["POS", "NEG"], "wrong polarity mark: {}".format(sentence["sent_id"])

            htep = (holder, target, exp, polarity)

            def __check_diff_spans_valid_func(e1, e2):

                # There are no intersections.
                if len(e1.intersection(e2)) == 0:
                    return True

                # Intersections exist => raise an exception.
                #print(e1)
                #print(e2)

                raise Exception("expressions for the same holder, target and polarity "
                                "must not overlap: {}".format(sentence["sent_id"]))

            exist = check_opinion_exist(
                htep=htep,
                opinions_iter=iter(opinion_tuples),
                check_diff_spans_valid_func=__check_diff_spans_valid_func)

            if not exist:
                opinion_tuples.append(htep)

    return opinion_tuples

def filter_opinions(sentence):
    text = sentence["text"]
    opinions = sentence["opinions"]
    opinion_tuples = []
    token_offsets = tk(text)

    opinions_ok = []
    if len(opinions) > 0:
        for opinion in opinions:

            # Extract idxs parts.
            holder_char_idxs = opinion["Source"][1]
            target_char_idxs = opinion["Target"][1]
            exp_char_idxs = opinion["Polar_expression"][1]

            # Compose elements of the new opinion.
            holder = frozenset(["AUTHOR"]) \
                if holder_char_idxs[0] == "NULL" \
                else convert_char_offsets_to_token_idxs(holder_char_idxs, token_offsets)
            target = convert_char_offsets_to_token_idxs(target_char_idxs, token_offsets)
            exp = convert_char_offsets_to_token_idxs(exp_char_idxs, token_offsets)
            polarity = opinion["Polarity"]

            assert polarity in ["POS", "NEG"], "wrong polarity mark: {}".format(sentence["sent_id"])

            htep = (holder, target, exp, polarity)

            def __check_diff_spans_valid_func(e1, e2):

                # There are no intersections.
                if len(e1.intersection(e2)) == 0:
                    return True

                # Intersections exist => raise an exception.
                #print('Intersection')
                #print(e1)
                #print(e2)
                #print()

                raise Exception("expressions for the same holder, target and polarity "
                                "must not overlap: {}".format(sentence["sent_id"]))

            try:
                exist = check_opinion_exist(
                    htep=htep,
                    opinions_iter=iter(opinion_tuples),
                    check_diff_spans_valid_func=__check_diff_spans_valid_func)
            except:
                exist = True

            opinions_ok.append(not exist)
            if not exist:
                opinion_tuples.append(htep)
    sentence["opinions"] = [o for i, o in enumerate(sentence["opinions"]) if opinions_ok[i]]
    return sentence

def sent_tuples_in_list(sent_tuple1, list_of_sent_tuples, keep_polarity=True):
    holder1, target1, exp1, pol1 = sent_tuple1
    if len(holder1) == 0:
        holder1 = frozenset(["_"])
    if len(target1) == 0:
        target1 = frozenset(["_"])
    for holder2, target2, exp2, pol2 in list_of_sent_tuples:
        if len(holder2) == 0:
            holder2 = frozenset(["_"])
        if len(target2) == 0:
            target2 = frozenset(["_"])
        if (
            len(holder1.intersection(holder2)) > 0
            and len(target1.intersection(target2)) > 0
            and len(exp1.intersection(exp2)) > 0
        ):
            if keep_polarity:
                if pol1 == pol2:
                    return True
            else:
                return True
    return False

def weighted_score(sent_tuple1, list_of_sent_tuples):
    best_overlap = 0
    holder1, target1, exp1, pol1 = sent_tuple1
    if len(holder1) == 0:
        holder1 = frozenset(["_"])
    if len(target1) == 0:
        target1 = frozenset(["_"])
    for holder2, target2, exp2, pol2 in list_of_sent_tuples:
        if len(holder2) == 0:
            holder2 = frozenset(["_"])
        if len(target2) == 0:
            target2 = frozenset(["_"])
        if (
            len(holder2.intersection(holder1)) > 0
            and len(target2.intersection(target1)) > 0
            and len(exp2.intersection(exp1)) > 0
        ):
            holder_overlap = len(holder2.intersection(holder1)) / len(holder1)
            target_overlap = len(target2.intersection(target1)) / len(target1)
            exp_overlap = len(exp2.intersection(exp1)) / len(exp1)
            overlap = (holder_overlap + target_overlap + exp_overlap) / 3
            if overlap > best_overlap:
                best_overlap = overlap
    return best_overlap


def tuple_precision(gold, pred, keep_polarity=True, weighted=True):
    """
    Weighted true positives / (true positives + false positives)
    """
    weighted_tp = []
    tp = []
    fp = []
    #
    for sent_idx in pred.keys():
        ptuples = pred[sent_idx]
        gtuples = gold[sent_idx]
        for stuple in ptuples:
            if sent_tuples_in_list(stuple, gtuples, keep_polarity):
                if weighted:
                    weighted_tp.append(weighted_score(stuple, gtuples))
                    tp.append(1)
                else:
                    weighted_tp.append(1)
                    tp.append(1)
            else:
                fp.append(1)
    return sum(weighted_tp) / (sum(tp) + sum(fp) + 0.0000000000000001)


def tuple_recall(gold, pred, keep_polarity=True, weighted=True):
    """
    Weighted true positives / (true positives + false negatives)
    """
    weighted_tp = []
    tp = []
    fn = []
    #
    for sent_idx in pred.keys():
        ptuples = pred[sent_idx]
        gtuples = gold[sent_idx]
        for stuple in gtuples:
            if sent_tuples_in_list(stuple, ptuples, keep_polarity):
                if weighted:
                    weighted_tp.append(weighted_score(stuple, ptuples))
                    tp.append(1)
                else:
                    weighted_tp.append(1)
                    tp.append(1)
            else:
                fn.append(1)
    return sum(weighted_tp) / (sum(tp) + sum(fn) + 0.0000000000000001)


def tuple_f1(gold, pred, keep_polarity=True, weighted=True):
    prec = tuple_precision(gold, pred, keep_polarity, weighted)
    rec = tuple_recall(gold, pred, keep_polarity, weighted)
    return 2 * (prec * rec) / (prec + rec + 0.00000000000000001)

def do_eval_core(results: Dict):
    """ Represent a core of the evaluation approach for
        the RuOpinionNE-2024 Competition.
    """
    
    gold = [r['gold'] for r in results]
    preds = [r['preds'] for r in results]
    assert(isinstance(gold, list))
    assert(isinstance(preds, list))

    # read in gold and predicted data, convert to dictionaries
    # where the sent_ids are keys
    check_gold = dict([(s["sent_id"], s['text']) for s in gold])
    check_preds = dict([(s["sent_id"], s['text']) for s in preds])

    g = set(check_gold.keys())
    p = set(check_preds.keys())

    assert g.issubset(p), "missing some sentences: {}".format(g.difference(p))
    assert p.issubset(g), "predictions contain sentences that are not in golds: {}".format(p.difference(g))
    for k in g:
        assert check_gold[k] == check_preds[k], "texts are not the same: {}".format(k)

    gold = dict([(s["sent_id"], convert_opinion_to_tuple(s)) for s in gold])
    preds = dict([(s["sent_id"], convert_opinion_to_tuple(s)) for s in preds])

    return tuple_f1(gold, preds)

def read_jsonl(path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)

def get_data_type(sample):
    if len(sample['opinions']) == 0:
        return 'empty'
    if len(sample['opinions']) > 1:
        return 'multi'
    if sample['opinions'][0]['Source'][0][0] == 'NULL':
        return 'src_null'
    if sample['opinions'][0]['Source'][0][0] == 'AUTHOR':
        return 'src_author'
    if len(sample['opinions']) == 1 and len(sample['opinions'][0]['Polar_expression'][0]) > 1:
        return 'multi_expr'
    
    return 'default'

def load_dataset(dataset_path, test=False):
    split = 'test' if test else 'train'
    data = load_dataset_hf(dataset_path, split=split).to_dict()

    dkeys = list(data.keys()) 
    deconvert = lambda x: [{k: x[k][i] for k in dkeys} for i in range(len(x[dkeys[0]]))]
    data = deconvert(data)
    #data = list(read_jsonl(dataset_path))
    keys = list(data[0].keys())
    convert = lambda x: {k: [d[k] for d in x] for k in keys}
    
    if test:
        tds = Dataset.from_dict(convert(data))
        ds = DatasetDict()

        ds['test'] = tds
        return ds
        
    for i in range(len(data)):
        data[i]['type'] = get_data_type(data[i])

    
    types_all = Counter([d['type'] for d in data])

    val_sent_ids = []
    for sample_type in types_all:
        val_sent_ids += [d['sent_id'] for d in data if d['type'] == sample_type][:5]

    val_sent_ids_set = set(val_sent_ids)

    train = [d for d in data if d['sent_id'] not in val_sent_ids_set]
    val = []
    for i in range(5):
        for j in range(len(types_all)):
            val += [d for d in data if d['sent_id'] == val_sent_ids[i + j * 5]]
    
    #print(val[:10])
    tds = Dataset.from_dict(convert(train))
    vds = Dataset.from_dict(convert(val))

    ds = DatasetDict()

    ds['test'] = tds
    ds['prompt'] = vds
    return ds
    
def pred2opinions_default(y_pred):
    y_pred = y_pred.strip()#y_pred[3:].lstrip()
    try:
        y_predict_dict = json.loads(y_pred)
    except:
        try:
            y_predict_dict = eval(y_pred)
        except:
            y_predict_dict = []
    if type(y_predict_dict) == dict:
        y_predict_dict = [y_predict_dict]
    return y_predict_dict

def pred2opinions_default(y_pred):
    if y_pred.startswith('```json') and y_pred.endswith('```'):
        y_pred = y_pred[len('```json'):-len('```')].strip()
    try:
        y_predict_dict = json.loads(y_pred)
    except:
        try:
            y_predict_dict = eval(y_pred)
        except:
            #print(y_pred)
            y_predict_dict = []
    if type(y_predict_dict) == dict:
        y_predict_dict = [y_predict_dict]
    return y_predict_dict

class RuOpinionNE(SimpleFewShotHFTask):
    def __init__(self, instruction, short_instruction=None, pred2opinions=pred2opinions_default, test=False, repeate_instruction=False, restrict_generation=False, **kwargs):
        super().__init__(**kwargs)
        self._max_new_tokens = 256
        self.instruction = instruction
        self.short_instruction = short_instruction
        if self.short_instruction is None:
            self.short_instruction = self.instruction 
        self.method = 'generate'
        self.test = test
        self.pred2opinions = pred2opinions
        self.repeate_instruction = repeate_instruction
        self.restrict_generation = restrict_generation

    def task_name(self):
        return 'RuOpinionNE'.lower()

    def dataset_args(self, test=False) -> Dict:
        return {'dataset_path': 'RefalMachine/ruopinionne', 'test': test}

    def test_split_name(self) -> str:
        return 'test'

    def prompt_split_name(self) -> str:
        return 'prompt'

    def aggregation(self) -> Dict:
        if self.test:
            return {'f1': lambda x: 0.0}
        else:
            return {"f1": do_eval_core}

    def evaluate(self, sample, y_pred) -> Dict:
        y_predict_dict = self.pred2opinions(y_pred)
        y_pred = {}
        y_pred['opinions'] = self._validate_and_add_pos_to_opinion(y_predict_dict, sample)
        y_pred['sent_id'] = sample['sent_id']
        y_pred['text'] = sample['text']
        filter_opinions(y_pred)
        return {"f1": {'gold': sample, 'preds': y_pred}}

    def _validate_dict(self, opinion, text):
        keys_ok = 'Source' in opinion and 'Target' in opinion and 'Polarity' in opinion and 'Expression' in opinion
        if not keys_ok:
            return False
        
        if type(opinion['Expression']) == str:
            opinion['Expression'] = [opinion['Expression']]

        types_ok = type(opinion['Source']) == str and type(opinion['Target']) == str and type(opinion['Polarity']) == str and type(opinion['Expression']) == list
        #return types_ok
        if not types_ok:
            return False
        
        #polarity_ok = opinion['Polarity'] in ['NEG', 'POS']
        #return polarity_ok
        exist_ok = (opinion['Source'].lower() in text.lower() or opinion['Source'] in ['AUTHOR', 'NULL']) and opinion['Polarity'] in ['NEG', 'POS'] and opinion['Target'].lower() in text.lower()
        for expr in opinion['Expression']:
            exist_ok = exist_ok and expr.lower() in text.lower()
        return exist_ok
    
    def _locate(self, entity, text):
        if entity == 'AUTHOR':
            return 'NULL'
        elif entity == 'NULL':
            return '0:0'
        else:
            if entity in text:
                pos = text.find(entity)
            else:
                pos = text.lower().find(entity.lower())
            return f'{pos}:{pos + len(entity)}'
        
    def _validate_and_add_pos_to_opinion(self, opinions, sample):
        text = sample['text']
        opinions_validates = []
        for i in range(len(opinions)):
            if not self._validate_dict(opinions[i], text):
                #print('Fail to validate')
                #print(opinions[i])
                #print(text)
                continue

            opinions_validates.append(
                {
                    'Source': [[opinions[i]['Source']], [self._locate(opinions[i]['Source'], text)]],
                    'Target': [[opinions[i]['Target']], [self._locate(opinions[i]['Target'], text)]],
                    'Polarity': opinions[i]['Polarity'],
                    'Polar_expression': [opinions[i]['Expression'], [self._locate(expr, text) for expr in opinions[i]['Expression']]],
                }
            )
        return opinions_validates
            
    def load_dataset(self, model: LLM, max_len: int, max_sample_per_dataset: int, few_shot_count: int) -> Tuple[List[Dict], List[Dict]]:
        assert model.support_method(self.method)

        samples = self._load_dataset(model, max_len, max_sample_per_dataset, few_shot_count)
        messages = [{'messages': s['messages']} for s in samples]
        samples = [{'sample': s['sample']} for s in samples]
        if self.restrict_generation:
            for i in range(len(messages)):
                #print(samples[i])
                messages[i]['allowed_token_ids'] = copy.deepcopy(samples[i]['sample']['allowed_token_ids'])
                #del samples[i]['allowed_token_ids']

        return messages, samples
    
    def _load_dataset(self, model: LLM, max_len: int, max_sample_per_dataset: int, few_shot_count: int) -> List:
        samples = []
        dataset = load_dataset(**self.dataset_args(test=False))
        test_dataset = dataset[self.test_split_name()]
        prompt_dataset = dataset[self.prompt_split_name()]

        if self.test:
            dataset = load_dataset(**self.dataset_args(test=True))
            test_dataset = dataset[self.test_split_name()]

        test_dataset = test_dataset.select(range(min(max_sample_per_dataset, len(test_dataset))))
        prompt_dataset = prompt_dataset.select(range(self.prompt_dataset_start_idx(), min(self.prompt_dataset_start_idx() + few_shot_count, len(prompt_dataset))))
                
        for sample in tqdm(test_dataset):
            s = copy.deepcopy(sample)
            samples.append({'messages': self._prepare_messages(sample, model, max_len, few_shot_count, prompt_dataset), 'sample': s})
            if self.restrict_generation:
                samples[-1]['sample']['allowed_token_ids'] = self._infer_allowed_tokens(s, model)

        return samples

    def _prepare_messages(self, sample: Dict, model: LLM, max_len: int, few_shot_count: int, prompt_dataset: Dataset) -> List:
        k = min(few_shot_count, len(prompt_dataset))

        zero_shot_messages = self.create_messages(copy.deepcopy(sample), with_answer=False, full_instruct=k==0 or self.repeate_instruction)
        zero_shot_messages_len = model.count_tokens_for_prompt(model.apply_model_prompt(zero_shot_messages))
        if zero_shot_messages_len >= max_len:
            self.logger.warning(f'WARNING: sample zero-shot len {zero_shot_messages_len} greater then {max_len}. Will be truncated.')

        message_groups = [self.create_messages(copy.deepcopy(prompt_dataset[i]), with_answer=True, full_instruct=(i==0)) for i in range(k)]
        
        for i in range(k):
            messages = []
            for group in message_groups[:k-i]:
                messages += group
            few_shot_messages_len = model.count_tokens_for_prompt(model.apply_model_prompt(messages + zero_shot_messages))
            if few_shot_messages_len < max_len:
                return messages + zero_shot_messages
        else:
            return zero_shot_messages
    
    def create_messages(self, sample, with_answer=False, full_instruct=True) -> List[Dict]:
        full_instruction = self.instruction
        short_instruction = self.short_instruction

        instruction = full_instruction if full_instruct else short_instruction
        #inputs = {'text': sample['text']}
        messages = [{'role': 'user', 'content': instruction.replace('{text}', sample['text'])}]
        if with_answer:
            messages.append(self._format_answer(sample, with_answer))
        return messages
    
    def _infer_allowed_tokens(self, sample, model):
        default_str = "[\n    {\n        \"Source\": \"{X}\",\n        \"Source\": \"AUTHOR\",\n        \"Source\": \"NULL\",\n        \"Target\": \"{X}\",\n        \"Polarity\": \"NEG\",\n        \"Polarity\": \"POS\"\n        \"Expression\": [\n            \"{X}\"\n        ]\n    },\n    {\n        \"Source\": \"{X}\",\n        \"Source\": \"AUTHOR\",\n        \"Source\": \"NULL\",\n        \"Target\": \"{X}\",\n        \"Polarity\": \"NEG\",\n        \"Polarity\": \"POS\"\n        \"Expression\": [\n            \"{X}\"\n        ]\n    }\n]"
        default_str = default_str.replace('{X}', model.tokenizer.eos_token)

        allowed_tokens_ids = model.tokenizer(default_str, add_special_tokens=False)['input_ids']
        allowed_tokens_ids += model.tokenizer("\n", add_special_tokens=False)['input_ids']
        allowed_tokens_ids += model.tokenizer("\n\n", add_special_tokens=False)['input_ids']

        allowed_tokens_ids += model.tokenizer(sample['text'], add_special_tokens=False)['input_ids']
        for w in sample['text'].split(' '):
            allowed_tokens_ids += model.tokenizer(w, add_special_tokens=False)['input_ids']

        allowed_tokens_ids = sorted(list(set(allowed_tokens_ids)))
        #model.tokenizer.vocab_size
        #allowed_tokens_ids = [t for t in allowed_tokens_ids if t < model.tokenizer.vocab_size]

        return allowed_tokens_ids

    def _format_answer(self, sample, with_answer=False):
        key_map = [['Source', 'Source'], ['Target', 'Target'], ['Polarity', 'Polarity'], ['Polar_expression', 'Expression']]
        data = []
        for opinion in sample['opinions']:
            converted_sample = {}
            for key_pair in key_map:
                if key_pair[0] == 'Polarity':
                    value = opinion[key_pair[0]]
                elif key_pair[0] == 'Polar_expression':
                    value = opinion[key_pair[0]][0]
                else:
                    value = opinion[key_pair[0]][0][0]
                converted_sample[key_pair[1]] = value
            data.append(converted_sample)
        #data = [{key_pair[1]: opinion[key_pair[0]][0][0] if key_pair[0] != 'Polarity' else opinion[key_pair[0]] for key_pair in key_map} for opinion in sample['opinions']]
        data = json.dumps(data, ensure_ascii=False, indent=4)

        #prefix = '''***json'''
        prefix = ''
        if with_answer:
            prefix += data

        return {'role': 'bot', 'content': prefix}
    
    def get_answer(self, sample):
        return self._format_answer(sample, with_answer=True)['content']

def convert_opinion_to_tuple_simple(sentence):
    text = sentence["text"]
    opinions = sentence["opinions"]
    opinion_tuples = []
    token_offsets = tk(text)

    if len(opinions) > 0:
        for opinion in opinions:
            holder_char_idxs = opinion["Source"][1]
            target_char_idxs = opinion["Target"][1]
            
            holder = frozenset(["AUTHOR"]) if holder_char_idxs[0] == "NULL" else convert_char_offsets_to_token_idxs(holder_char_idxs, token_offsets)
            target = convert_char_offsets_to_token_idxs(target_char_idxs, token_offsets)
            polarity = opinion["Polarity"]

            assert polarity in ["POS", "NEG"], "wrong polarity mark: {}".format(sentence["sent_id"])
            
            # Simplified tuple without expression
            htep = (holder, target, frozenset(), polarity)

            def __check_diff_spans_valid_func(e1, e2):
                if len(e1.intersection(e2)) == 0:
                    return True
                raise Exception("expressions for the same holder, target and polarity must not overlap: {}".format(sentence["sent_id"]))

            # Simplified check_opinion_exist call
            if not check_opinion_exist(htep, opinion_tuples, __check_diff_spans_valid_func):
                opinion_tuples.append(htep)
    return opinion_tuples

def sent_tuples_in_list_simple(sent_tuple1, list_of_sent_tuples, keep_polarity=True):
    holder1, target1, _, pol1 = sent_tuple1
    if len(holder1) == 0:
        holder1 = frozenset(["_"])
    if len(target1) == 0:
        target1 = frozenset(["_"])
    for holder2, target2, _, pol2 in list_of_sent_tuples:
        if len(holder2) == 0:
            holder2 = frozenset(["_"])
        if len(target2) == 0:
            target2 = frozenset(["_"])
        if len(holder1.intersection(holder2)) > 0 and len(target1.intersection(target2)) > 0:
            if keep_polarity:
                if pol1 == pol2:
                    return True
            else:
                return True
    return False

def weighted_score_simple(sent_tuple1, list_of_sent_tuples):
    best_overlap = 0
    holder1, target1, _, pol1 = sent_tuple1
    if len(holder1) == 0:
        holder1 = frozenset(["_"])
    if len(target1) == 0:
        target1 = frozenset(["_"])
    for holder2, target2, _, pol2 in list_of_sent_tuples:
        if len(holder2) == 0:
            holder2 = frozenset(["_"])
        if len(target2) == 0:
            target2 = frozenset(["_"])
        if len(holder2.intersection(holder1)) > 0 and len(target2.intersection(target1)) > 0:
            holder_overlap = len(holder2.intersection(holder1)) / len(holder1)
            target_overlap = len(target2.intersection(target1)) / len(target1)
            overlap = (holder_overlap + target_overlap) / 2
            if overlap > best_overlap:
                best_overlap = overlap
    return best_overlap

def tuple_precision_simple(gold, pred, keep_polarity=True, weighted=True):
    weighted_tp = []
    tp = []
    fp = []
    for sent_idx in pred.keys():
        ptuples = pred[sent_idx]
        gtuples = gold[sent_idx]
        for stuple in ptuples:
            if sent_tuples_in_list_simple(stuple, gtuples, keep_polarity):
                if weighted:
                    weighted_tp.append(weighted_score_simple(stuple, gtuples))
                    tp.append(1)
                else:
                    weighted_tp.append(1)
                    tp.append(1)
            else:
                fp.append(1)
    return sum(weighted_tp) / (sum(tp) + sum(fp) + 1e-18)

def tuple_recall_simple(gold, pred, keep_polarity=True, weighted=True):
    weighted_tp = []
    tp = []
    fn = []
    for sent_idx in pred.keys():
        ptuples = pred[sent_idx]
        gtuples = gold[sent_idx]
        for stuple in gtuples:
            if sent_tuples_in_list_simple(stuple, ptuples, keep_polarity):
                if weighted:
                    weighted_tp.append(weighted_score_simple(stuple, ptuples))
                    tp.append(1)
                else:
                    weighted_tp.append(1)
                    tp.append(1)
            else:
                fn.append(1)
    return sum(weighted_tp) / (sum(tp) + sum(fn) + 1e-18)

def tuple_f1_simple(gold, pred, keep_polarity=True, weighted=True):
    prec = tuple_precision_simple(gold, pred, keep_polarity, weighted)
    rec = tuple_recall_simple(gold, pred, keep_polarity, weighted)
    return 2 * (prec * rec) / (prec + rec + 1e-18)

def do_eval_core_simple(results: Dict):
    gold = [r['gold'] for r in results]
    preds = [r['preds'] for r in results]
    
    check_gold = dict([(s["sent_id"], s['text']) for s in gold])
    check_preds = dict([(s["sent_id"], s['text']) for s in preds])

    g = set(check_gold.keys())
    p = set(check_preds.keys())

    assert g.issubset(p), "missing some sentences: {}".format(g.difference(p))
    assert p.issubset(g), "predictions contain sentences that are not in golds: {}".format(p.difference(g))
    for k in g:
        assert check_gold[k] == check_preds[k], "texts are not the same: {}".format(k)

    gold = dict([(s["sent_id"], convert_opinion_to_tuple_simple(s)) for s in gold])
    preds = dict([(s["sent_id"], convert_opinion_to_tuple_simple(s)) for s in preds])

    return tuple_f1_simple(gold, preds)


class RuOpinionNESimple(RuOpinionNE):
    def task_name(self):
        return 'RuOpinionNESimple'.lower()

    def aggregation(self) -> Dict:
        if self.test:
            return {'f1': lambda x: 0.0}
        else:
            return {"f1": do_eval_core_simple}

    def _validate_dict(self, opinion, text):
        keys_ok = 'Source' in opinion and 'Target' in opinion and 'Polarity' in opinion
        if not keys_ok:
            return False
        
        types_ok = type(opinion['Source']) == str and type(opinion['Target']) == str and type(opinion['Polarity']) == str
        if not types_ok:
            return False
        
        exist_ok = (opinion['Source'].lower() in text.lower() or opinion['Source'] in ['AUTHOR', 'NULL']) and \
                   opinion['Polarity'] in ['NEG', 'POS'] and \
                   opinion['Target'].lower() in text.lower()
        return exist_ok

    def _validate_and_add_pos_to_opinion(self, opinions, sample):
        text = sample['text']
        opinions_validates = []
        for i in range(len(opinions)):
            if not self._validate_dict(opinions[i], text):
                #print('Fail to validate')
                #print(opinions[i])
                #print(text)
                continue

            opinions_validates.append({
                'Source': [[opinions[i]['Source']], [self._locate(opinions[i]['Source'], text)]],
                'Target': [[opinions[i]['Target']], [self._locate(opinions[i]['Target'], text)]],
                'Polarity': opinions[i]['Polarity'],
                # Add a dummy Polar_expression for compatibility with downstream processing
                'Polar_expression': [['-'], ['0:0']],
            })
        return opinions_validates

    def _format_answer(self, sample, with_answer=False):
        key_map = [['Source', 'Source'], ['Target', 'Target'], ['Polarity', 'Polarity']]
        data = []
        for opinion in sample['opinions']:
            converted_sample = {}
            for key_pair in key_map:
                if key_pair[0] == 'Polarity':
                    value = opinion[key_pair[0]]
                else:
                    value = opinion[key_pair[0]][0][0]
                converted_sample[key_pair[1]] = value
            data.append(converted_sample)
        data = json.dumps(data, ensure_ascii=False, indent=4)

        prefix = ''
        if with_answer:
            prefix += data

        return {'role': 'bot', 'content': prefix}
