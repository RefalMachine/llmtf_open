"""Offline dual-scoring replay and strict paired comparisons of completed artifacts.

python -m dev.tools.legalbench_ru_report DIR [DIR ...] --output report.json
Requires no LLM, network or third-party packages.
"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys

from llmtf.provenance import fingerprint_run_config

# Load the stdlib-only protocol without importing the global task registry.
_PROTOCOL = '_llmtf_legalbench_report_protocol'
_ROOT = Path(__file__).resolve().parents[2] / 'llmtf/tasks/legalbench_ru'
if _PROTOCOL not in sys.modules:
    spec = importlib.util.spec_from_file_location(_PROTOCOL, _ROOT/'__init__.py', submodule_search_locations=[str(_ROOT)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[_PROTOCOL] = package
    spec.loader.exec_module(package)
data = __import__(_PROTOCOL+'.data', fromlist=['data'])
scoring = __import__(_PROTOCOL+'.scoring', fromlist=['scoring'])


def compatibility(config):
    result=copy.deepcopy(config)
    task=result['task']
    # Only selection identity and context mode may differ. Sampling, model,
    # template, protocol/resources, shot count and budgets must match exactly.
    for name in ('name','registry_name'):
        task.pop(name,None)
    task.get('init_params',{}).pop('mode',None)
    task['provenance'].pop('mode',None)
    return result


def read_run(total_path):
    total_path=Path(total_path)
    total=json.loads(total_path.read_text())
    stem=str(total_path).removesuffix('_total.jsonl')
    samples=json.loads(Path(stem+'.jsonl').read_text())
    params=json.loads(Path(stem+'_params.jsonl').read_text())
    config=total['run_config']; provenance=config['task']['provenance']
    if (total.get('run_fingerprint') != fingerprint_run_config(config)
            or params['run_fingerprint'] != total['run_fingerprint']
            or params['run_config'] != config):
        raise ValueError('Params/total identity mismatch')
    for name, expected in provenance['resource_sha256'].items():
        if data.sha256(_ROOT/name)!=expected:
            raise ValueError(f'Replay resource version mismatch: {name}')
    split=provenance['split']; mode=provenance['mode']; selection=provenance['selection']
    if mode=='upstream_all_zero_shot':
        allowed={tuple(k) for k in split['evaluation']} | {tuple(k) for pool in split['demonstrations'].values() for k in pool}
    else:
        allowed={tuple(k) for k in split['evaluation']}
    records={}
    catalog=data.read_resource('tool_catalog.json')
    for artifact in samples:
        row=artifact['sample']; k=data.key(row)
        if k in records or k not in allowed:
            raise ValueError(f'Duplicate or unexpected member {k}')
        if mode in data.CONTEXT_FIELDS and not row.get(data.CONTEXT_FIELDS[mode]):
            raise ValueError('Missing context')
        trace=row['_legalbench']; shots=config['task']['few_shot_count']
        expected_demos=split['demonstrations'][data.bucket(row)][:shots]
        if trace['requested_shots']!=shots or trace['effective_shots']!=shots or trace['demonstration_keys']!=expected_demos:
            raise ValueError('Demonstration trace mismatch')
        replay=scoring.score(row,artifact['predict'],catalog)
        if replay!=artifact['metric']['score']:
            raise ValueError('Replay differs from stored sample score')
        records[k]=(replay,row)
    primary, details=scoring.aggregate([r for r,_ in records.values()],provenance['primary'])
    if abs(primary-total['results']['score'])>1e-12:
        raise ValueError('Replayed total mismatch')
    if selection=='smoke':
        expected=split['smoke'] if mode in ('closed','upstream_all_zero_shot') else split['context_smoke'][mode]
    elif mode in data.CONTEXT_FIELDS:
        expected=split['cohorts'][mode]
    else:
        expected=list(allowed)
    missing=set(map(tuple,expected))-records.keys()
    if records.keys()-set(map(tuple,expected)):
        raise ValueError('Sample outside requested cohort')
    return {'mode':mode,'config':config,'records':records,'details':details,'missing_keys':sorted(missing),'source':str(total_path)}


def paired(left,right,cohort):
    if compatibility(left['config'])!=compatibility(right['config']):
        raise ValueError('Incompatible paired run configurations')
    expected=set(map(tuple,cohort))
    if not expected:
        raise ValueError('Empty paired cohort')
    missing=expected-left['records'].keys() | expected-right['records'].keys()
    result={'expected_count':len(expected),'matched_count':len(expected-missing),'missing_keys':sorted(missing),'complete':not missing}
    if not missing:
        result['delta']=sum(right['records'][k][0]['score']-left['records'][k][0]['score'] for k in expected)/len(expected)
        result['reference_delta']=sum(right['records'][k][0]['reference_score']-left['records'][k][0]['reference_score'] for k in expected)/len(expected)
    return result


def report(paths):
    runs={}
    for path in paths:
        for total in sorted(Path(path).glob('legalbench_ru*_total.jsonl')):
            run=read_run(total)
            if run['mode'] in runs:
                raise ValueError('Supply only one configuration per mode (separate models/shots)')
            runs[run['mode']]=run
    if not runs:
        raise ValueError('No completed LegalBench-RU totals')
    output={'primary_mode':'closed','runs':{m:{'source':r['source'],'missing_keys':r['missing_keys'],**r['details']} for m,r in runs.items()},'paired':{}}
    for left,right,cohort in [('closed','grounded','grounded'),('grounded','distractor','distractor'),('grounded','temporal','temporal'),('closed','temporal','temporal')]:
        if left in runs and right in runs:
            p=runs[right]['config']['task']['provenance']
            expected=p['split']['context_smoke' if p['selection']=='smoke' else 'cohorts'][cohort]
            output['paired'][right+' - '+left]=paired(runs[left],runs[right],expected)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories',nargs='+')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=json.dumps(report(args.directories),ensure_ascii=False,indent=2)
    if args.output:args.output.write_text(result+'\n')
    else:print(result)


if __name__=='__main__':main()
