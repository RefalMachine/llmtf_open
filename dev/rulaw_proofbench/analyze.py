"""Baselines and paired article-cluster bootstrap for full evaluator artifacts."""
import argparse
from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path
import random

try:
    from .generate import ROOT, SEED, build
except ImportError:
    from generate import ROOT, SEED, build

# Load the stdlib scorer without importing the framework's optional task dependencies.
spec=importlib.util.spec_from_file_location('rulaw_scoring',ROOT.parents[1]/'llmtf/tasks/rulaw_proofbench/scoring.py')
scoring=importlib.util.module_from_spec(spec);spec.loader.exec_module(scoring)


def baselines(rows):
    global_choice=Counter(r['gold']['value'] for r in rows).most_common(1)[0][0]
    groups=defaultdict(list)
    for r in rows:groups[r['article_key']].append(r)
    per_article={a:Counter(r['gold']['value'] for r in rs).most_common(1)[0][0] for a,rs in groups.items()}
    result={}
    for name,predict in [('always_unknown',lambda r:'недостаточно данных'),
                         ('global_gold_majority',lambda r:global_choice),
                         ('article_gold_majority',lambda r:per_article[r['article_key']])]:
        records=[scoring.score(r,predict(r)) for r in rows]
        macro,details=scoring.aggregate(records);result[name]=dict(article_macro_accuracy=macro,details=details)
    result['interpretation']='Gold-informed majority constants are descriptive diagnostic references, not independently trained baselines.'
    return result


def load_predictions(path,rows,allow_partial=False):
    items=json.loads(path.read_text(encoding='utf-8'))
    expected={r['id']:r for r in rows};results={}
    for item in items:
        sample=item['sample'];identifier=sample['id']
        if identifier not in expected or identifier in results:raise ValueError('Unknown/duplicate result ID')
        gold=expected[identifier]
        if any(sample[k]!=gold[k] for k in ('question','gold','rule_sha256','as_of')):raise ValueError('Different dataset snapshot')
        results[identifier]=scoring.score(gold,item['predict'])
    if not allow_partial and results.keys() != expected.keys():
        raise ValueError(f'Full {len(rows)}-record artifact required; use --allow-partial for diagnostics')
    return results


def paired_bootstrap(left,right,iterations=10000):
    if left.keys()!=right.keys():raise ValueError('Paired runs must have identical sample IDs')
    by_article=defaultdict(list)
    for key,r in left.items():by_article[r['article_key']].append(float(r['correct'])-float(right[key]['correct']))
    differences=[sum(values)/len(values) for values in by_article.values()]
    rng=random.Random(SEED)
    boots=sorted(sum(rng.choices(differences,k=len(differences)))/len(differences) for _ in range(iterations))
    return dict(direction='left_minus_right',article_clusters=len(differences),sample_pairs=len(left),
                macro_difference=sum(differences)/len(differences),iterations=iterations,seed=SEED,
                percentile_95_interval=[boots[int(.025*iterations)],boots[min(iterations-1,int(.975*iterations))]],
                interpretation='Uncertainty over the selected article clusters; not representative sampling of all Russian law.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--left',type=Path);p.add_argument('--right',type=Path)
    p.add_argument('--allow-partial',action='store_true');p.add_argument('--output',type=Path)
    a = p.parse_args()
    from llmtf.tasks.rulaw_proofbench.data import load_rows

    rows = load_rows()
    report = {'scorer': scoring.SCORER_NAME, 'baselines': baselines(rows)}
    loaded = {}
    if a.right and not a.left:p.error('--right requires --left')
    for side in ('left','right'):
        path=getattr(a,side)
        if path:
            loaded[side]=load_predictions(path,rows,a.allow_partial)
            macro,details=scoring.aggregate(list(loaded[side].values()))
            report[side]=dict(path=str(path),article_macro_accuracy=macro,details=details)
    if len(loaded)==2:report['paired_bootstrap']=paired_bootstrap(loaded['left'],loaded['right'])
    text=json.dumps(report,ensure_ascii=False,indent=2)+'\n'
    if a.output:a.output.write_text(text,encoding='utf-8')
    else:print(text)


if __name__=='__main__':main()
