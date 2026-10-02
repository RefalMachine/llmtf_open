"""Synthetic protocol contracts. Optional pinned corpus audit via LEGALBENCH_DATA."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dev.tools.legalbench_ru_report import data,scoring,paired,read_run
from llmtf.provenance import fingerprint_run_config
from llmtf.task_selection import resolve_task_names
prompts=__import__(scoring.__package__+'.prompts',fromlist=['prompts'])


def row(kind='extraction',answer='10',**kwargs):
    return dict(task='synthetic',id='one',track='knowledge',domain='test',answer_type=kind,answer=answer,question='Synthetic question',split='public',reasoning_type='test',**kwargs)


class ProtocolTests(unittest.TestCase):
    def test_numeric_boundaries(self):
        for wrong in ('210','-10','10.5','10,5','010','x10'):
            self.assertEqual(scoring.score(row(),wrong)['score'],0,wrong)
        self.assertEqual(scoring.score(row(),'10')['score'],1)
        self.assertEqual(scoring.score(row(answer='0010'),'10')['score'],0)
        self.assertEqual(scoring.score(row(answer='10%'),'10%')['score'],1)
        self.assertEqual(scoring.score(row(answer='10%'),'10')['score'],0)
        self.assertEqual(scoring.score(row(),'210')['reference_score'],1)

    def test_json_parsing(self):
        for text in ('{"tool":4,"args":{}}','{"tool":[],"args":{}}','{"tool":"x","args":[]}','{"tool":null} {}','prefix {"tool":null}','{"tool":null,"tool":"x"}','{"tool":"x","args":{"v":NaN}}','{"tool":"x","args":{"v":Infinity}}','{"tool":"x","args":{"v":1e999}}'):
            self.assertNotEqual(scoring.parse_call(text)[1],'ok',text)
        for text in ('{"tool":"x","args":{"v":"} escaped \\""}}','```json\n{"tool":null}\n```'):
            self.assertEqual(scoring.parse_call(text)[1],'ok',text)

    def test_full_routing_and_refusal(self):
        r=row('tool_call',{'tool':'right.lookup','args':{'a':1}})
        x=scoring.score(r,'{"tool":"wrong.lookup","args":{"a":1}}')
        self.assertEqual((x['score'],x['reference_score']),(0,1))
        n=row('tool_call',{'tool':None,'args':{}})
        self.assertEqual(scoring.score(n,'{"tool":"x","args":{"a":null}}')['score'],0)
        self.assertEqual(scoring.score(n,'{"tool":null}')['score'],1)
        self.assertEqual(scoring.score(n,'')['score'],0)

    def test_typed_arguments_and_partial_credit(self):
        r=row('tool_call',{'tool':'x.f','args':{'a':1,'b':'AB'}})
        s=scoring.score(r,'{"tool":"x.f","args":{"a":1,"b":"ab","extra":1}}')
        self.assertAlmostEqual(s['score'],.8)
        self.assertEqual(s['gold_call_exact'],0)
        self.assertEqual(s['unannotated_args'],1)
        self.assertFalse(scoring._typed_equal(True,1))
        self.assertFalse(scoring._typed_equal([1,2],[2,1]))

    def test_labels(self):
        self.assertEqual(scoring.score(row('binary','Да'),'Да, нет')['parse_status'],'ambiguous')
        self.assertEqual(scoring.score(row('multiple_choice','A',choices=['first','second','third','fourth']),'a')['score'],1)
        self.assertEqual(scoring.score(row('multiple_choice','A'),'A B')['score'],0)

    def test_citations(self):
        expected={('ГК','1'),('ТК','2')}
        for text in ('ст. 1 ГК РФ; ст. 2 ТК РФ','ГК РФ ст. 1; ТК РФ ст. 2','ГК РФ ст. 1, ТК РФ ст. 2','ст. 1 ГК РФ, ст. 2 ТК РФ'):
            self.assertEqual(scoring.extract_norms(text)[0],expected,text)
        self.assertEqual(scoring.extract_norms('ГК РФ ст. 1, 2')[0],{('ГК','1'),('ГК','2')})
        s=scoring.score(row('norm_citation',['ГК РФ ст.1','ТК РФ ст.2']),'ГК РФ ст.1')
        self.assertAlmostEqual(s['score'],2/3)
        self.assertEqual(s['reference_score'],.667)

    def test_prompt_whitelist(self):
        r=row(answer='SECRET',accept=['SECRET2'],explanation='SECRET3',gold_norm='SECRET4')
        for secret in ('SECRET','SECRET2','SECRET3','SECRET4'):
            self.assertNotIn(secret,prompts.build_prompt(r,'extraction'))
        self.assertNotEqual(prompts.build_prompt(dict(r,norm_text='NORM'),'extraction','grounded'),prompts.build_prompt(r,'extraction'))

    def test_aggregation(self):
        a=scoring.score(row('binary','Да'),'Да');b=scoring.score(dict(row('binary','Нет'),id='two'),'Да')
        value,details=scoring.aggregate([a,b])
        self.assertEqual(value,.5);self.assertEqual(details['binary']['balanced_accuracy'],.5)
        with self.assertRaises(ValueError):scoring.aggregate([a,a])
        with self.assertRaises(ValueError):scoring.aggregate([])

    def test_registry_opt_in(self):
        registry={'old':{},'experiment':{'include_in_all':False}}
        self.assertEqual(resolve_task_names('all',registry),['old'])
        self.assertEqual(resolve_task_names('experiment',registry),['experiment'])

    def test_pair_missing_members_fail_closed(self):
        config={'task':{'provenance':{},'few_shot_count':0}}
        a={'config':config,'records':{('t','1'):({'score':0,'reference_score':0},{})}}
        b={'config':config,'records':{('t','1'):({'score':1,'reference_score':1},{})}}
        self.assertEqual(paired(a,b,[['t','1']])['delta'],1)
        self.assertNotIn('delta',paired(a,b,[['t','1'],['t','2']]))
        with self.assertRaises(ValueError):paired(a,b,[])
        b['config']={'task':{'provenance':{},'few_shot_count':5}}
        with self.assertRaises(ValueError):paired(a,b,[['t','1']])

    def test_offline_replay_integrity(self):
        split = data.read_resource('split_manifest.json')
        task, ident = split['evaluation'][0]
        sample = dict(row(), task=task, id=ident)
        sample['_legalbench'] = {'requested_shots': 0, 'effective_shots': 0,
                                'demonstration_keys': []}
        metric = scoring.score(sample, '10')
        provenance = {'split': split, 'mode': 'closed', 'selection': 'full',
                      'resource_sha256': {}, 'primary': 'corrected'}
        config = {'task': {'provenance': provenance, 'few_shot_count': 0}}
        identity = {'run_config': config,
                    'run_fingerprint': fingerprint_run_config(config)}
        with tempfile.TemporaryDirectory() as directory:
            stem = Path(directory) / 'legalbench_ru_closed'
            total_path = Path(str(stem) + '_total.jsonl')
            total_path.write_text(json.dumps(dict(identity, results={'score': 1})))
            Path(str(stem) + '_params.jsonl').write_text(json.dumps(identity))
            samples_path = Path(str(stem) + '.jsonl')
            artifact = {'sample': sample, 'predict': '10', 'metric': {'score': metric}}
            samples_path.write_text(json.dumps([artifact]))
            result = read_run(total_path)
            self.assertEqual(len(result['missing_keys']), 815)
            samples_path.write_text(json.dumps([artifact, artifact]))
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                read_run(total_path)
            samples_path.write_text(json.dumps([dict(artifact, predict='210')]))
            with self.assertRaisesRegex(ValueError, 'Replay differs'):
                read_run(total_path)
            samples_path.write_text(json.dumps([artifact]))
            total_path.write_text(json.dumps(dict(identity, results={'score': 0})))
            with self.assertRaisesRegex(ValueError, 'total mismatch'):
                read_run(total_path)

    def test_wrong_hash(self):
        with tempfile.NamedTemporaryFile() as f:
            with self.assertRaisesRegex(ValueError,'SHA-256'):data.load_rows(f.name)

    @unittest.skipUnless(os.environ.get('LEGALBENCH_DATA'),'pinned corpus not supplied')
    def test_full_pinned_corpus(self):
        rows=data.load_rows(os.environ['LEGALBENCH_DATA'])
        self.assertEqual(len(rows),846)
        for mode,n in [('closed',816),('grounded',205),('distractor',189),('temporal',8)]:
            selected,pools=data.select_rows(rows,mode)
            self.assertEqual(len(selected),n)
            self.assertEqual(sum(map(len,pools.values())),30)
        for r in rows:
            with self.subTest(key=data.key(r)):
                answer=prompts.oracle_output(r,r['answer_type'])
                s=scoring.score(r,answer)
                self.assertEqual(s['score'],1)
                self.assertEqual(s['reference_score'],1)
                self.assertEqual(scoring.score(r,'')['score'],0)
                self.assertEqual(scoring.score(r,'')['reference_score'],0)


if __name__=='__main__':unittest.main()
