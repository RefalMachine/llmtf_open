import sklearn
from rouge_score import rouge_scorer
try:
    import pymorphy3 as pymorphy2
except:
    import pymorphy2
import string
import re
import numpy as np

def mean(arr):
    return sum(arr) / max(len(arr), 1)

def f1_macro_score(items):
    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    score = sklearn.metrics.f1_score(golds, preds, average="macro")
    return score

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """Compute max metric between prediction and each ground truth."""
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def mcc(items):
    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    score = sklearn.metrics.matthews_corrcoef(golds, preds)
    return score

class CustomTokenizer():
    def __init__(self):
        self.punkt_regex = re.compile('[%s]' % re.escape(string.punctuation))
        self.morph = pymorphy2.MorphAnalyzer()

    def tokenize(self, text):
        #print([t for t in ''.join(text.split())])
        return self.preprocess_sentence(text)

    def preprocess_sentence(self, text: str):
        text = text.lower()
        text = self.punkt_regex.sub(' ', text)
        words = [self.morph.parse(token)[0].normal_form for token in text.split() if len(token.strip()) > 0]
        return words

rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], tokenizer=CustomTokenizer())
def rouge1(gold, generated):
    return rouge_scorer.score(gold, generated)['rouge1']

def rouge2(gold, generated):
    return rouge_scorer.score(gold, generated)['rouge2']

def rougel(gold, generated):
    return rouge_scorer.score(gold, generated)['rougeL']

def get_order_relevance_k(labels, scores, k):
    scores = -np.array(scores)
    labels = np.array(labels).astype(float)
    order = np.argsort(scores)[:k]
    order_relevance = labels[order]
    return order_relevance

def r_precision(labels, scores):
    total_labels = np.array(labels).astype(int).sum()
    order_relevance = get_order_relevance_k(labels, scores, total_labels)
    return order_relevance.sum() / total_labels

llm_judge_instruction = [
    {"role": "user", "content": "Look at the following two responses and judge whether they are equivalent. Respond only with yes or no."},
    {"role": "user", "content": "Response 1: Yes\nResponse 2: No\n"},
    {"role": "assistant", "content": "No"},
    {"role": "user", "content": "Response 1: Banana\nResponse 2: banana  \n"},
    {"role": "assistant", "content": "Yes"},
    {"role": "user", "content": "Response 1: 4/8\nResponse 2: x=1/2\n"},
    {"role": "assistant", "content": "Yes"},
    {"role": "user", "content": "Response 1: 1.5\nResponse 2: 2.5\n"},
    {"role": "assistant", "content": "No"},
    {"role": "user", "content": "Response 1: USA\nResponse 2: United States of America\n"},
    {"role": "assistant", "content": "Yes"},
    {"role": "user", "content": "Response 1: Russia\nResponse 2: Moscow\n"},
    {"role": "assistant", "content": "No"},
    {"role": "user", "content": "Response 1: Tokio\nResponse 2: Paris\n"},
    {"role": "assistant", "content": "No"},
    {"role": "user", "content": "Response 1: Natalie\nResponse 2: Maria\n"},
    {"role": "assistant", "content": "No"},
    {"role": "user", "content": "Response 1:  no, he's not\nResponse 2: No he is not in example\n"},
    {"role": "assistant", "content": "Yes"},
    {"role": "user", "content": "Response 1: {}\nResponse 2: {}\n"},
]

def llm_judge_accuracy(model, gold, generated, instruction=llm_judge_instruction):
    instruction = instruction[:-1] + [{"role": instruction[-1]["role"], "content": instruction[-1]["content"].format(gold, generated)}]
    _, response, _ = model.generate(instruction)
    if response[:3] == "Yes":
        return 1
    elif response[:2] == "No":
        return 0
    else:
        raise Exception(f"LLM metric generated unexpected text. Expected `Yes` or `No`, generated: {response}")
