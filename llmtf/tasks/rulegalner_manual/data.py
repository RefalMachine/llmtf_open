"""Pinned upstream manual annotations; download originals without republishing data."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).parent
ALL_TAGS = ('LAW', 'PROVISION', 'PENALTY', 'PERSON', 'ORG', 'DATE')


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def manifest():
    return json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))


def document_key(row):
    return (row['meta']['source_split'], row['meta']['doc_id'])


def text_key(row):
    return ' '.join(row['text'].split())


def _download(spec, cache_dir):
    import requests
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / (spec['sha256'] + '.jsonl')
    if target.exists():
        return target
    url = 'https://drive.usercontent.google.com/download'
    with requests.get(url, params={'id': spec['file_id'], 'export': 'download',
                                   'confirm': 't'}, timeout=(15, 60)) as response:
        response.raise_for_status()
        data = response.content
    if sha256(data) != spec['sha256']:
        raise ValueError('RuLegalNER manual download SHA-256 mismatch (upstream changed or returned HTML)')
    with tempfile.NamedTemporaryFile(dir=cache_dir, delete=False) as tmp:
        tmp.write(data)
        temporary = Path(tmp.name)
    try:
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def validate_rows(rows, split, expected_count):
    if len(rows) != expected_count:
        raise ValueError(f'Unexpected RuLegalNER manual {split} row count')
    ids = set()
    for row in rows:
        if type(row.get('id')) is not int or row['id'] in ids:
            raise ValueError(f'Invalid or duplicate {split} ID')
        ids.add(row['id'])
        if not isinstance(row.get('text'), str) or not row['text'].strip():
            raise ValueError('Missing RuLegalNER manual text')
        meta = row.get('meta', {})
        if not isinstance(meta.get('source_split'), str) or type(meta.get('doc_id')) is not int:
            raise ValueError('Missing RuLegalNER manual source document identity')
        labels = row.get('label')
        if not isinstance(labels, list):
            raise ValueError('RuLegalNER manual labels must be a list')
        spans = []
        for label in labels:
            if not isinstance(label, list) or len(label) != 3:
                raise ValueError('Invalid RuLegalNER manual span')
            start, end, tag = label
            if (type(start) is not int or type(end) is not int
                    or not 0 <= start < end <= len(row['text']) or tag not in ALL_TAGS):
                raise ValueError('Invalid RuLegalNER manual span bounds or class')
            if any(start < previous_end and previous_start < end for previous_start, previous_end in spans):
                raise ValueError('Overlapping RuLegalNER manual spans')
            spans.append((start, end))
    return rows


def load_splits(data_dir=None):
    spec = manifest()
    data_dir = data_dir or os.environ.get('LLMTF_RULEGALNER_MANUAL_DATA_DIR')
    if data_dir is None:
        from huggingface_hub.constants import HF_HOME
        cache_dir = Path(HF_HOME) / 'llmtf' / 'rulegalner_manual'
    splits = {}
    for split, source in spec['files'].items():
        path = Path(data_dir) / source['filename'] if data_dir else _download(source, cache_dir)
        raw = path.read_bytes()
        if sha256(raw) != source['sha256']:
            raise ValueError(f'RuLegalNER manual {split} SHA-256 mismatch')
        rows = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
        splits[split] = validate_rows(rows, split, source['rows'])
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if ({document_key(row) for row in splits[a]} & {document_key(row) for row in splits[b]}
                    or {text_key(row) for row in splits[a]} & {text_key(row) for row in splits[b]}):
                raise ValueError(f'RuLegalNER manual split overlap: {a}/{b}')
    return splits
