"""Load the cached pinned HF Parquet; retain an offline upstream XLSX audit reader."""
from decimal import Decimal
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import unicodedata
from xml.etree import ElementTree
from zipfile import ZipFile

ROOT = Path(__file__).parent
NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def manifest():
    return json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def question_key(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def read_workbook(data):
    """Respect cell coordinates: the upstream sheet includes sparse/empty rows."""
    with ZipFile(BytesIO(data)) as workbook:
        shared = ElementTree.fromstring(workbook.read('xl/sharedStrings.xml'))
        strings = [''.join(t.text or '' for t in cell.findall('.//s:t', NS))
                   for cell in shared.findall('s:si', NS)]
        sheet = ElementTree.fromstring(workbook.read('xl/worksheets/sheet1.xml'))
    headers, rows = {}, []
    for row in sheet.findall('s:sheetData/s:row', NS):
        values = {}
        for cell in row.findall('s:c', NS):
            column = re.sub(r'\d+$', '', cell.attrib['r'])
            value = cell.find('s:v', NS)
            if cell.get('t') == 's' and value is not None:
                text = strings[int(value.text)]
            elif cell.get('t') == 'inlineStr':
                text = ''.join(t.text or '' for t in cell.findall('.//s:t', NS))
            else:
                text = value.text if value is not None else ''
            if text:
                values[column] = text
        if row.attrib['r'] == '1':
            headers = values
        elif values:
            result = {headers[col]: text for col, text in values.items() if col in headers}
            result['id'] = str(int(Decimal(values['A'])))
            rows.append(result)
    if not {'question_for_llm', 'true_answer', 'source_url'} <= set(headers.values()):
        raise ValueError('RuTaR workbook schema changed')
    return rows


def validate_rows(rows, spec):
    if len(rows) != spec['raw_row_count'] or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Unexpected RuTaR row count or duplicate ID')
    missing, duplicates, seen, result = [], [], {}, []
    for raw in rows:
        value = Decimal(raw['true_answer'])
        if value not in (0, 1):
            raise ValueError('RuTaR gold must be 0 or 1')
        question = (raw.get('question_for_llm') or '').strip()
        if not question:
            missing.append(raw['id'])
            continue
        key = question_key(question)
        if key in seen:
            if seen[key] != str(int(value)):
                raise ValueError('Conflicting RuTaR labels for duplicate question')
            duplicates.append(raw['id'])
            continue
        seen[key] = str(int(value))
        # Keep only task inputs and audit metadata; letter answers never enter prompts.
        result.append({'id': raw['id'], 'question': question, 'answer': str(int(value)),
                       'title': raw['title'], 'date_publication': raw['date_publication'],
                       'source_url': raw.get('source_url') or '', 'letter_type': raw['letter_type']})
    if missing != spec['excluded_missing_question_ids'] or duplicates != spec['excluded_duplicate_question_ids']:
        raise ValueError('RuTaR exclusion membership changed')
    return result


def load_rows(data_path=None):
    spec = manifest()
    if data_path is None:
        from huggingface_hub import hf_hub_download
        data_path = hf_hub_download(repo_id=spec['dataset_repo'], repo_type='dataset',
                                    revision=spec['dataset_revision'], filename=spec['dataset_filename'])
    path = Path(data_path)
    is_workbook = path.suffix == '.xlsx'
    expected_hash = spec['sha256'] if is_workbook else spec['dataset_sha256']
    if sha256(path.read_bytes()) != expected_hash:
        raise ValueError('RuTaR dataset SHA-256 mismatch')
    if is_workbook:
        rows = read_workbook(path.read_bytes())
    else:
        import pyarrow.parquet as pq
        rows = pq.read_table(path).to_pylist()
    return validate_rows(rows, spec)


def split_rows(rows):
    spec = manifest()
    by_id = {row['id']: row for row in rows}
    demonstrations = [by_id[key] for key in spec['demonstration_ids']]
    excluded_questions = {question_key(row['question']) for row in demonstrations}
    excluded_sources = {row['title'] for row in demonstrations}
    evaluation = [row for row in rows if question_key(row['question']) not in excluded_questions
                  and row['title'] not in excluded_sources]
    if len(evaluation) != spec['evaluation_row_count']:
        raise ValueError('RuTaR evaluation split changed')
    return evaluation, demonstrations
