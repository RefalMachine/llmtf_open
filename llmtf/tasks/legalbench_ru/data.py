"""Pinned JSONL loading and explicit, composite-key selection (no Arrow coercion)."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parent
CONTEXT_FIELDS = {'grounded': 'norm_text', 'distractor': 'distractor_text', 'temporal': 'temporal_text'}


def read_resource(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(row):
    return (row['task'], row['id'])


def bucket(row):
    return ('extraction_' + row['track']) if row['answer_type'] == 'extraction' else row['answer_type']


def validate_rows(rows, catalog):
    seen = set()
    tools = {c['server'] + '.' + c['tool']: c for c in catalog}
    if len(tools) != len(catalog) or len(tools) != 79:
        raise ValueError('Invalid tool catalog')
    for c in catalog:
        if not all(isinstance(c.get(f), str) for f in ('server', 'tool', 'desc')) or not isinstance(c.get('params'), list):
            raise ValueError('Invalid catalog entry')
    for row in rows:
        for field in ('task', 'id', 'question', 'domain', 'reasoning_type', 'answer_type', 'track', 'split'):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f'Missing/invalid {field}')
        if key(row) in seen:
            raise ValueError(f'Duplicate composite key {key(row)}')
        seen.add(key(row))
        if row['split'] not in ('public', 'holdout') or row['track'] not in ('knowledge', 'reasoning', 'tool-use'):
            raise ValueError('Unknown split/track')
        types = {'binary': str, 'multiple_choice': str, 'extraction': str, 'norm_citation': list, 'tool_call': dict}
        kind = row['answer_type']
        if kind not in types or not isinstance(row.get('answer'), types[kind]):
            raise ValueError('Unknown answer type or missing/invalid gold')
        answer = row['answer']
        if kind == 'binary' and answer not in ('Да', 'Нет'):
            raise ValueError('Invalid binary gold')
        if kind == 'multiple_choice' and (answer not in 'ABCD' or len(answer) != 1 or not isinstance(row.get('choices'), list) or len(row['choices']) != 4):
            raise ValueError('Invalid choice gold/options')
        if kind == 'norm_citation' and (not answer or not all(isinstance(x, str) and x for x in answer)):
            raise ValueError('Invalid citation gold')
        for field in ('context', *CONTEXT_FIELDS.values()):
            if field in row and not isinstance(row[field], str):
                raise ValueError(f'Invalid {field}')
        if 'accept' in row and (not isinstance(row['accept'], list) or not all(isinstance(x, str) for x in row['accept'])):
            raise ValueError('Invalid accept')
        if kind == 'tool_call':
            if 'tool' not in answer or not isinstance(answer.get('args'), dict):
                raise ValueError('Invalid gold call')
            tool = answer['tool']
            if tool is not None and (not isinstance(tool, str) or tool not in tools or not set(answer['args']) <= set(tools[tool]['params'])):
                raise ValueError('Gold tool/args outside catalog')
    return rows


def load_rows(data_path=None):
    manifest = read_resource('manifest.json')
    if data_path is None:
        from huggingface_hub import hf_hub_download
        data_path = hf_hub_download(repo_id=manifest['dataset_repo'], repo_type='dataset', revision=manifest['dataset_revision'], filename=manifest['filename'])
    if sha256(data_path) != manifest['dataset_sha256']:
        raise ValueError('LegalBench-RU dataset SHA-256 mismatch')
    if sha256(ROOT / 'tool_catalog.json') != manifest['catalog_sha256']:
        raise ValueError('LegalBench-RU catalog SHA-256 mismatch')
    raw = [json.loads(line) for line in Path(data_path).read_text(encoding='utf-8').splitlines() if line.strip()]
    canaries = [r for r in raw if '_canary' in r]
    rows = [r for r in raw if '_canary' not in r]
    if len(canaries) != manifest['canary_count'] or len(rows) != manifest['row_count'] or set(canaries[0]) != {'_canary'}:
        raise ValueError('Unexpected canary/row count')
    return validate_rows(rows, read_resource('tool_catalog.json'))


def select_rows(rows, mode, selection='full'):
    split = read_resource('split_manifest.json')
    by_key = {key(r): r for r in rows}
    pools = {b: [by_key[tuple(k)] for k in keys] for b, keys in split['demonstrations'].items()}
    demo_keys = [key(r) for pool in pools.values() for r in pool]
    if len(set(demo_keys)) != 30 or any(len(p) != 5 for p in pools.values()):
        raise ValueError('Invalid demonstration split')
    for b, pool in pools.items():
        if any(bucket(r) != b or any(r.get(f) for f in CONTEXT_FIELDS.values()) for r in pool):
            raise ValueError('Invalid demo bucket/context')
    evaluation = [r for r in rows if key(r) not in set(demo_keys)]
    if [list(key(r)) for r in evaluation] != split['evaluation']:
        raise ValueError('Evaluation membership/order changed')
    if mode == 'upstream_all_zero_shot':
        evaluation = list(rows)
    elif mode != 'closed':
        field = CONTEXT_FIELDS[mode]
        evaluation = [r for r in evaluation if r['track'] == 'reasoning' and r.get(field)]
    if selection == 'smoke':
        selected = split['smoke'] if mode in ('closed', 'upstream_all_zero_shot') else split['context_smoke'][mode]
        selected = set(map(tuple, selected))
        evaluation = [r for r in evaluation if key(r) in selected]
        if len(evaluation) != len(selected):
            raise ValueError('Missing frozen smoke member')
    elif selection != 'full':
        raise ValueError('Unknown selection')
    return evaluation, pools
