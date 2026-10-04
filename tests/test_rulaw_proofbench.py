"""Task contracts; run in the existing API container without GPU or torch."""
import json
import copy
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from llmtf.base import PromptTooLongError
from llmtf.config import SamplingConfig
from llmtf.reasoning import ReasoningConfig
from llmtf.tasks import TASK_REGISTRY
from llmtf.task_selection import resolve_task_names
from llmtf.tasks.rulaw_proofbench import RuLawProofBench, RuLawProofBenchMCQ
from llmtf.tasks.rulaw_proofbench.judge import parse_verdict, judge_records, aggregate_judgments
from llmtf.tasks.rulaw_proofbench.scoring import parse_answer, aggregate, score, SCORER_NAME
from llmtf.evaluator import Evaluator
from llmtf.tasks.rulaw_proofbench.data import load_rows, manifest


def build():
    return [], load_rows(), []


class Model:
    logger=logging.getLogger('rulaw_test')
    reasoning_config=ReasoningConfig()
    generation_config=SamplingConfig(max_new_tokens=96)
    def get_params(self):return {'model':'synthetic-contract-test'}
    def count_tokens_for_messages(self,messages):return None
    def support_method(self,method):return method in ('generate','calculate_tokens_proba')
    def get_model_context_len(self):return 16000
    def add_stop_strings(self,stops):pass
    def reset_stop_strings(self):pass


class RuLawTests(unittest.TestCase):
    def test_pinned_public_data(self):
        spec = manifest()
        rows = load_rows()
        self.assertEqual(len(rows), spec["test_count"])
        self.assertEqual(len(spec["dataset_revision"]), 40)
        self.assertTrue(all(row["context"] for row in rows))
        for row in rows:
            self.assertNotRegex(row["context"], r"(?m)^(?:tk|sk|gk|59):\d+\s*$")

    def test_exact_five_shots_and_unchanged_test_membership(self):
        from llmtf.tasks.rulaw_proofbench import task as task_module

        test = load_rows("mcq")[:2]
        train = []
        for index in range(5):
            row = copy.deepcopy(test[0])
            row.update(id=f"demo-{index}", article_key=f"demo:{index}")
            row["question"] = f"Separate demonstration {index}"
            row["mcq"]["question"] = row["question"]
            train.append(row)
        spec = dict(manifest(), test_count=2, demonstration_count=5)

        def fixture(config, split, data_dir):
            return copy.deepcopy(train if split == "train" else test)

        with patch.object(task_module, "manifest", return_value=spec), patch.object(
            task_module, "load_rows", side_effect=fixture
        ):
            task = RuLawProofBenchMCQ()
            zero_messages, zero_samples = task.load_dataset(Model(), 16000, 2, 0)
            messages, samples = task.load_dataset(Model(), 16000, 2, 5)
            self.assertEqual(
                [sample["sample"]["id"] for sample in zero_samples],
                [sample["sample"]["id"] for sample in samples],
            )
            for message, sample in zip(messages, samples):
                self.assertEqual(len(message["messages"]), 12)
                self.assertEqual(message["messages"][-1]["content"], "Ответ:")
                self.assertEqual(sample["sample"]["_rulaw_proofbench"]["effective_shots"], 5)
                for index, demonstration in enumerate(train):
                    self.assertEqual(
                        message["messages"][index * 2 + 1]["content"],
                        "Ответ: " + demonstration["mcq"]["gold_label"],
                    )
            self.assertTrue(all(len(message["messages"]) == 2 for message in zero_messages))
            with self.assertRaises(ValueError):
                task.load_dataset(Model(), 16000, 2, 6)
            model = Model()
            model.count_tokens_for_messages = lambda _: 20000
            with self.assertRaises(PromptTooLongError):
                task.load_dataset(model, 16000, 2, 5)
            train[0]["article_key"] = test[0]["article_key"]
            with self.assertRaisesRegex(ValueError, "overlap"):
                task.load_dataset(Model(), 16000, 2, 5)

    def test_published_demonstrations(self):
        train = load_rows("mcq", "train")
        test = load_rows("mcq", "test")
        self.assertEqual(len(train), 5)
        self.assertEqual(len(test), 300)
        self.assertEqual({row["mcq"]["gold_label"] for row in train}, set("ABCD"))
        self.assertFalse({row["article_key"] for row in train} & {row["article_key"] for row in test})
        task = RuLawProofBenchMCQ()
        messages, samples = task.load_dataset(Model(), 16000, 300, 5)
        self.assertEqual(len(samples), 300)
        for message, sample in zip(messages, samples):
            self.assertEqual(len(message["messages"]), 12)
            self.assertEqual(sample["sample"]["_rulaw_proofbench"]["demonstration_ids"], [row["id"] for row in train])

    def test_normalized_exact_grammar(self):
        for text,unit,gold in [
            ('17,50 часов в неделю','hours_per_week','17.5'),
            ('35/2 часа в неделю','hours_per_week','17.5'),
            ('¼','fraction','1/4'), ('2 / 8','fraction','1/4'),
            ('0,25','fraction','1/4'), ('25 %','fraction','1/4'),
            ('«Да».','none','да'), (' ГОД. ','none','1 год'),
            ('1,00 год','none','1 год'), ('5 календарных дней','calendar_days','5'),
            ('НЕДОСТАТОЧНО   ДАННЫХ.','days','недостаточно данных'),
        ]:
            with self.subTest(text=text):self.assertEqual(parse_answer(text,unit),gold)
        for text,unit in [
            ('Да, но нет','none'), ('не да','none'), ('нет, да','none'),
            ('20 или 30','days'), ('Вероятно, 20 дней','days'),
            ('30 дней. 20 дней','days'), ('20 дней, а не 30','days'),
            ('до 35 дней','days'), ('не более 35 дней','days'),
            ('одна треть','fraction'), ('пять дней','days'),
            ('15 полных лет','hours_per_week'), ('5 часов','days'),
            ('5 дней','hours'), ('50%','days'), ('1','none'), ('1/0','fraction'),
            ('17½','hours'), ('ничтожность','none'), ('не следует из данной нормы','none'),
            ('Доверенность сохраняет силу в течение года со дня ее совершения.','none'),
            ('Год со дня совершения.','none'), ('', 'days'), (None, 'days'),
        ]:
            with self.subTest(text=text):self.assertIsNone(parse_answer(text,unit))

    def test_numeric_rules_cover_values_outside_dataset(self):
        from fractions import Fraction
        for n in range(100):
            for unit,suffix in [('days','дней'),('calendar_days','календарных дней'),
                                ('hours','часов'),('hours_per_week','часов в неделю')]:
                for text in [str(n), f'{n}.00', f'{n},0 {suffix}', f'{n*2}/2 {suffix}']:
                    self.assertEqual(parse_answer(text,unit),str(n))
            for denominator in (2,3,4,5,7,10):
                self.assertEqual(parse_answer(f'{2*n}/{2*denominator}','fraction'),str(Fraction(n,denominator)))

    def test_all_canonical_outcomes_and_no_article_special_cases(self):
        _,rows,_=build()
        for row in rows:
            self.assertTrue(score(row,row['gold']['value'])['correct'])
            alternate=dict(row,article_key='unseen:999')
            for text in [row['gold']['value'],'до 35 дней','17.50','Год со дня совершения.']:
                expected=score(row,text);actual=score(alternate,text)
                self.assertEqual(expected['prediction'],actual['prediction'])
                self.assertEqual(expected['correct'],actual['correct'])

    def test_modes_zero_shot_and_overflow(self):
        for mode in ('closed','grounded'):
            task=RuLawProofBench(mode=mode)
            messages,samples=task.load_dataset(Model(),16000,3,0)
            self.assertEqual(len(samples),3)
            self.assertTrue(all(len(x['messages'])==1 for x in messages))
            self.assertEqual('контрольного режима чтения' in messages[0]['messages'][0]['content'],mode=='grounded')
            with self.assertRaises(ValueError):task.load_dataset(Model(),16000,3,6)
            model=Model();model.count_tokens_for_messages=lambda _:20000
            with self.assertRaises(PromptTooLongError):task.load_dataset(model,16000,3,0)

    def test_macro_and_pair_denominators(self):
        def r(i,a,ok,p):return dict(id=i,article_key=a,correct=ok,question_type='branch',gold='да',prediction='да' if ok else 'нет',format_valid=True,pair_ids=p,unknown_but_determinate=False)
        value,detail=aggregate([r('a1','A',True,['p']),r('a2','A',False,['p']),r('b1','B',True,['partial'])])
        self.assertEqual(value,.75);self.assertAlmostEqual(detail['micro_accuracy'],2/3)
        self.assertEqual(detail['minimal_pairs'],dict(complete=1,partial=1,both_correct=0))
        with self.assertRaises(ValueError):aggregate([r('same','A',True,[])]*2)

    def test_registry_and_provenance(self):
        for mode in ('closed','grounded'):
            name='rulaw_proofbench/'+mode
            self.assertIn(name,TASK_REGISTRY)
            self.assertNotIn(name,resolve_task_names('all',TASK_REGISTRY))
        p=RuLawProofBench().get_task_provenance()
        self.assertIn('data/open/test.parquet', p['dataset']['files'])
        self.assertIn('scoring.py',p['implementation_sha256'])
        self.assertEqual(p['scorer'],SCORER_NAME)
        self.assertNotEqual(p,RuLawProofBench(mode='grounded').get_task_provenance())

    def test_evaluator_generate_and_ppl_skip(self):
        _,rows,_=build();gold={r['question']:r['gold']['value'] for r in rows}
        model=Model()
        def generate_batch(messages,**kwargs):
            return ['test']*len(messages),[gold[m[-1]['content']] for m in messages],[{} for _ in messages]
        model.generate_batch=generate_batch
        with tempfile.TemporaryDirectory() as directory:
            result=Evaluator().evaluate(model,directory,datasets_names=['rulaw_proofbench/closed'],few_shot_count=0,max_sample_per_dataset=8)
            self.assertEqual(result.exit_code,0)
            total=json.loads(next(Path(directory).glob('*_total.jsonl')).read_text())
            self.assertEqual(total['results']['score'],1)
        with tempfile.TemporaryDirectory() as directory:
            result=Evaluator().evaluate_ppl(model,directory,datasets_names=['rulaw_proofbench/closed'])
            self.assertIn('rulaw_proofbench/closed',result.skipped)

    def test_mcq_probability_contract_and_parent_gold(self):
        task=RuLawProofBenchMCQ()
        messages,samples=task.load_dataset(Model(),16000,300,0)
        outcomes=[]
        for message,sample in zip(messages,samples):
            row=sample['sample'];mcq=row['mcq'];labels=[o['label'] for o in mcq['options']]
            self.assertEqual(message['tokens_of_interest'],labels)
            self.assertEqual(message['messages'][-1],{'role':'assistant','content':'Ответ:'})
            self.assertEqual(next(o['value'] for o in mcq['options'] if o['label']==mcq['gold_label']),row['gold']['value'])
            values={k:float(k==mcq['gold_label']) for k in labels}
            outcomes.append(task.evaluate(row,values)['score'])
            wrong=next(k for k in labels if k!=mcq['gold_label'])
            self.assertFalse(task.evaluate(row,{k:float(k==wrong) for k in labels})['score']['correct'])
        self.assertEqual(task._aggregate(outcomes)[0],1)
        for values in ({},dict(A=1,B=0,C=0,D=0,E=0),{k:float('nan') for k in labels},{k:0 for k in labels}):
            with self.assertRaises(ValueError):task.evaluate(row,values)
        with self.assertRaises(ValueError):RuLawProofBenchMCQ(judge_model=Model())

    def test_judge_alignment_failure_and_dual_metric(self):
        for text in ('{"equivalent": "true", "reason": "x"}','{"equivalent":true}', 'not json'):
            with self.assertRaises(ValueError):parse_verdict(text)
        _,rows,_=build();row=rows[0]
        task=RuLawProofBench(judge_model=Model());task.selected_ids={row['id']};task.eligible_count=150
        result=task.evaluate(row,'Произвольная формулировка')
        self.assertEqual(set(result),{'score','llm_judge_accuracy'})
        model=Model()
        model.generate_batch=lambda ms,**kw:(ms,['{"equivalent": true, "reason": "fixture"}']*len(ms),[{}]*len(ms))
        items=[result['llm_judge_accuracy']];decisions=judge_records(model,items)
        value,details=aggregate_judgments(items,decisions)
        self.assertEqual(value,1);self.assertEqual(details['disagreements']['strict_false_judge_true'],1)
        self.assertEqual(details['samples'][0]['id'],row['id'])
        model.generate_batch=lambda ms,**kw:(ms,[],[])
        with self.assertRaises(ValueError):judge_records(model,items)
        decisions[0]['id']='wrong'
        with self.assertRaises(ValueError):aggregate_judgments(items,decisions)

    def test_evaluator_persists_two_metrics(self):
        model=Model();judge=Model()
        _,rows,_=build();gold={r['question']:r['gold']['value'] for r in rows}
        model.generate_batch=lambda messages,**kw:(messages,[gold[m[-1]['content']] for m in messages],[{}]*len(messages))
        judge.generate_batch=lambda ms,**kw:(ms,['{"equivalent":true,"reason":"fixture"}']*len(ms),[{}]*len(ms))
        name='rulaw_proofbench/closed'
        registered=dict(TASK_REGISTRY[name]);registered['params']={'mode':'closed','judge_model':judge}
        with patch.dict(TASK_REGISTRY,{name:registered}),tempfile.TemporaryDirectory() as directory:
            result=Evaluator().evaluate(model,directory,datasets_names=[name],few_shot_count=0,max_sample_per_dataset=8)
            self.assertEqual(result.exit_code,0)
            total=json.loads(next(Path(directory).glob('*_total.jsonl')).read_text())
            self.assertEqual(total['results'],{'score':1,'llm_judge_accuracy':1})


if __name__=='__main__':unittest.main()
