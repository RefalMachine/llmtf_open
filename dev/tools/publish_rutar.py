"""Export a lossless RuTaR mirror; optionally publish the reviewed directory to HF."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workbook', type=Path, required=True)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repo', default='RefalMachine/RuTaR')
    parser.add_argument('--upload', action='store_true')
    args = parser.parse_args()
    import pyarrow as pa
    import pyarrow.parquet as pq
    from llmtf.tasks.rutar.data import manifest, read_workbook, validate_rows
    spec = manifest()
    if digest(args.workbook) != spec['sha256']:
        raise ValueError('Upstream XLSX hash mismatch')
    raw = read_workbook(args.workbook.read_bytes())
    validate_rows(raw, spec)
    columns = ['id', 'title', 'date_publication', 'full_text', 'question_letter',
               'answer_letter', 'letter_type', 'source_url', 'question_for_llm',
               'found_sources', 'true_answer']
    records = [{key: int(float(row[key])) if key == 'true_answer' else row.get(key)
                for key in columns} for row in raw]
    schema = pa.schema([(key, pa.int64() if key == 'true_answer' else pa.string())
                        for key in columns])
    table = pa.Table.from_pylist(records, schema=schema)
    args.output.mkdir(parents=True, exist_ok=True)
    for folder in ('data', 'original', 'sources'):
        (args.output / folder).mkdir(exist_ok=True)
    target = args.output / 'data/test.parquet'
    pq.write_table(table, target, compression='zstd')
    if pq.read_table(target).to_pylist() != records:
        raise ValueError('Parquet roundtrip mismatch')
    (args.output / 'original/rutar.xlsx').write_bytes(args.workbook.read_bytes())
    corpus = json.loads(args.sources.read_text(encoding='utf-8'))
    if not isinstance(corpus, dict) or len(corpus) != 480:
        raise ValueError('Unexpected upstream source corpus')
    (args.output / 'sources/sources_dataset_for_rutar.json').write_bytes(args.sources.read_bytes())
    publication = {'repository': args.repo, 'upstream_repository': spec['repository'],
                   'upstream_revision': spec['revision'], 'upstream_filename': spec['filename'],
                   'upstream_sha256': spec['sha256'], 'row_count': len(records),
                   'filename': 'data/test.parquet', 'parquet_sha256': digest(target),
                   'sources_sha256': digest(args.sources), 'source_count': len(corpus),
                   'format_version': 'rutar_lossless_parquet_v1', 'pyarrow_version': pa.__version__}
    (args.output / 'manifest.json').write_text(json.dumps(publication, indent=2) + '\n')
    repo_url = spec['repository']; rev = spec['revision']
    card = f'''---
language:
- ru
task_categories:
- text-classification
tags:
- legal
- taxes
- reasoning
pretty_name: RuTaR (Russian Tax Reasoning)
size_categories:
- n<1K
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test.parquet
---

# RuTaR — Russian Tax Reasoning

This is a format mirror of [rutar-anonymous/RuTaR]({repo_url}), attributed to
its upstream authors. It contains Russian tax questions derived from letters
of the Russian Ministry of Finance and Federal Tax Service. Original binary
labels are preserved: **1 = yes, 0 = no**.

## Original source and attribution

- [Upstream repository and documentation]({repo_url})
- [Exact upstream commit]({repo_url}/tree/{rev})
- [Original workbook]({repo_url}/blob/{rev}/rutar.xlsx)
- [Original reference-source corpus]({repo_url}/blob/{rev}/sources_dataset_for_rutar.json)
- Individual government-letter links are retained in `source_url`, with
  document titles, dates and issuing body in each record. One source URL is missing upstream.

The upstream snapshot does not include a license file or a license statement.
This mirror does not assert or grant a new license; consult the original
repository/authors for reuse terms. No author identity, paper citation or
license has been inferred from the anonymous repository name.

## Files and preservation

- `data/test.parquet`: all **209** nonempty workbook records, in original order.
- `original/rutar.xlsx`: byte-for-byte original workbook (SHA-256 `{spec['sha256']}`).
- `sources/sources_dataset_for_rutar.json`: byte-for-byte original RAG corpus,
  **480** citation-to-text entries.
- `manifest.json`: upstream revision, file checksums and conversion version.

`test` is a packaging split for the complete upstream table; upstream supplies
no train/test split. No questions, duplicates or gold labels are removed here.
Blank question at ID `119` and duplicate questions at IDs `114`/`117` remain
available for auditing. Empty worksheet rows are omitted.

The unnamed workbook index becomes a stable string `id`. `true_answer` becomes
an integer 0/1. Other original columns remain strings with blank cells represented
as null. `found_sources` retains its original string representation of a list.
Dates remain original strings. Original XLSX is retained for exact source recovery.

## Columns

| Column | Meaning |
|---|---|
| `id` | Original workbook numeric index, represented as a string |
| `title` | Government document title |
| `date_publication` | Original document publication date |
| `full_text` | Full original letter, including its answer |
| `question_letter` / `answer_letter` | Question/answer parsed from the letter |
| `letter_type` | Issuing body (`minfin` or Federal Tax Service identifier) |
| `source_url` | Link to the original government document |
| `question_for_llm` | Upstream prepared yes/no question (one null) |
| `found_sources` | Original string list of references; texts are in the source corpus |
| `true_answer` | Gold: 1 yes, 0 no |

## Loading

```python
from datasets import load_dataset
rows = load_dataset("{args.repo}", split="test")
```

Pin `revision` to a Hub commit for reproducible evaluation. HF caches the Parquet
snapshot locally, so ordinary repeated runs reuse downloaded files.

## Evaluation considerations

The full letters and their parsed answers contain evidence used to author the
questions; feeding them into a closed-book prompt leaks gold information.
For closed-book evaluation, use only `question_for_llm` plus an explicit answer
instruction, and keep demonstration questions/documents separate from evaluation.
LLMTF reserves IDs 0–4 as a fixed demonstration pool, drops missing question 119
and duplicate 117, and evaluates 202 questions for both zero-shot and five-shot.
That is an LLMTF protocol, not an upstream split or a reproduction of upstream RAG.

Labels refer to the source documents and their historical dates; they have not
been rewritten to reflect current tax law. This mirror preserves upstream
annotations and does not add legal expert validation.
'''
    (args.output / 'README.md').write_text(card, encoding='utf-8')
    print(json.dumps(publication, indent=2))
    if not args.upload:
        return
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import RepositoryNotFoundError, EntryNotFoundError
    api = HfApi()
    account = api.whoami()['name']
    if args.repo.split('/')[0] != account:
        raise ValueError('Target repository must belong to the authenticated account')
    try:
        info = api.repo_info(args.repo, repo_type='dataset')
    except RepositoryNotFoundError:
        info = None
    upload = info is None
    if info is not None:
        files = set(api.list_repo_files(args.repo, repo_type='dataset', revision=info.sha))
        upload = files <= {'.gitattributes'}
        if not upload:
            try:
                existing = hf_hub_download(args.repo, 'manifest.json', repo_type='dataset', revision=info.sha)
            except EntryNotFoundError:
                raise ValueError('Refusing to overwrite an existing repository without a mirror manifest')
            if json.loads(Path(existing).read_text()) != publication:
                raise ValueError('Existing HF mirror differs; refusing overwrite')
            revision = info.sha
    if upload:
        api.create_repo(args.repo, repo_type='dataset', private=False, exist_ok=True)
        commit = api.upload_folder(repo_id=args.repo, repo_type='dataset', folder_path=args.output,
                                   parent_commit=info.sha if info is not None else None,
                                   commit_message='Mirror pinned upstream RuTaR with Parquet and source attribution')
        revision = commit.oid
    result = dict(publication, revision=revision)
    (args.output.parent / 'rutar-publication.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Published:', f'https://huggingface.co/datasets/{args.repo}/tree/{revision}')
    # Verify public availability from a fresh cache without the account token.
    from datasets import load_dataset
    anonymous = load_dataset(args.repo, split='test', revision=revision, token=False,
                             cache_dir=str(args.output.parent / 'rutar-anonymous-cache'))
    if list(anonymous) != records:
        raise ValueError('Anonymous Hub roundtrip mismatch')
    mirrored = hf_hub_download(args.repo, 'data/test.parquet', repo_type='dataset',
                              revision=revision, token=False,
                              cache_dir=str(args.output.parent / 'rutar-anonymous-files'))
    if digest(mirrored) != publication['parquet_sha256']:
        raise ValueError('Anonymous Hub Parquet hash mismatch')
    print('Anonymous pinned Hub roundtrip verified:', len(anonymous), 'rows')


if __name__ == '__main__':
    main()
