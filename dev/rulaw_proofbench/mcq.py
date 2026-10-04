"""Deterministic MMLU-like derivative with 3/4 mutually distinct consequences."""
import json
import random
from collections import Counter

from .generate import ROOT, SEED, build, canonical, digest, jsonl
from .rules import UNKNOWN, OUTSIDE

VERSION='rulaw_mcq_v1'
LETTERS='ABCD'


def option_text(value,unit):
    if value in (UNKNOWN,OUTSIDE):return value.capitalize()
    if unit in ('days','calendar_days','hours','hours_per_week'):
        number=float(value);n=int(number)
        form=1 if number!=n else 0 if n%10==1 and n%100!=11 else 1 if n%10 in (2,3,4) and n%100 not in (12,13,14) else 2
        words=('день','дня','дней') if unit in ('days','calendar_days') else ('час','часа','часов')
        text=value.replace('.',',')+' '+words[form]
        if unit=='hours_per_week':text+=' в неделю'
        return text
    return value.capitalize()


def build_mcq(cards=None, parents=None, gold_positions=None):
    if cards is None and parents is None:
        cards, parents, _ = build()
    elif cards is None or parents is None:
        raise ValueError("Provide both cards and parent records")
    rng=random.Random(SEED);positions={3:0,4:0};result={}
    for card in cards:
        rows=[r for r in parents if r['rule_id']==card['rule_id']];rng.shuffle(rows)
        outcomes={branch['value'] for branch in card['branches']}|{UNKNOWN}
        binary=outcomes=={'да','нет',UNKNOWN}
        # Base consideration duration excludes the extra 30 days in part 2.
        # 60 is an explicit extension-confusion distractor, not a new legal gold.
        if card['article_key']=='59:12':outcomes.add('60')
        perturbations = set()
        if not binary and len(outcomes) < 4:
            from fractions import Fraction

            numeric = sorted(outcomes - {UNKNOWN, OUTSIDE})
            if not numeric:
                raise ValueError("No numeric consequence for distractor construction")
            reference = Fraction(numeric[0])
            step = Fraction(1, reference.denominator)
            offset = 1
            while len(outcomes) < 4:
                candidate = reference + offset * step
                offset += 1
                if candidate >= 0 and str(candidate) not in outcomes:
                    outcomes.add(str(candidate))
                    perturbations.add(str(candidate))
        special=outcomes & {UNKNOWN,OUTSIDE}
        ordinary=sorted(outcomes-special)
        for row in rows:
            value=row['gold']['value'];k=3 if binary else 4
            if binary:
                selected=['да','нет',UNKNOWN]
            else:
                count=k-len(special)
                candidates=[v for v in ordinary if v!=value];rng.shuffle(candidates)
                selected=sorted(special)+([value] if value in ordinary else [])
                selected+=candidates[:count-(value in ordinary)]
            assert len(selected)==k and len(set(selected))==k and value in selected
            wrong=[v for v in selected if v!=value];rng.shuffle(wrong)
            correct=positions[k]%k;positions[k]+=1;wrong.insert(correct,value)
            if gold_positions is not None:
                correct = gold_positions[row['id']]
                if type(correct) is not int or not 0 <= correct < k:
                    raise ValueError('Invalid fixed demonstration answer position')
                wrong.remove(value)
                wrong.insert(correct, value)
            options=[]
            for label,v in zip(LETTERS,wrong):
                witness=next((r['id'] for r in rows if r['gold']['value']==v),None)
                options.append(dict(label=label,value=v,text=option_text(v,card['unit']),
                    evidence={'kind':'same_rule_outcome','witness_id':witness} if v!='60' or card['article_key']!='59:12'
                             else {'kind':'excluded_extension_confusion','source':'59:12:part2','calculation':'30 + 30'}))
                if v in perturbations:
                    options[-1]['evidence'] = {
                        'kind': 'numeric_perturbation',
                        'rule': 'positive_rational_steps_from_a_canonical_outcome',
                    }
            body='\n'.join(row['question'].splitlines()[:-1])
            for phrase in ('Ответьте «да» или «нет».','Ответьте обыкновенной дробью.','Дайте короткий ответ.'):
                body=body.replace(phrase,'')
            body=body.replace('ответьте «не следует из указанной нормы»','верен вывод «не следует из указанной нормы»').strip()
            instructions=row['question'].splitlines()[-1].replace('Дайте только короткий ответ без объяснения.','').strip()
            instructions=instructions.replace('ответьте','выберите вывод').replace('означает ответ','означает вывод')
            body+='\n'+instructions+'\n\n'+'\n'.join(f"{o['label']}. {o['text']}" for o in options)
            body+='\n\nВыберите единственный правильный вариант. Ответьте только его буквой: '+', '.join(LETTERS[:k])+'.'
            result[row['id']]=dict(id=row['id'],parent_id=row['id'],article_key=row['article_key'],
                version=VERSION,rule_sha256=row['rule_sha256'],parent_sha256=digest(canonical(row).encode()),
                question=body,options=options,gold_label=LETTERS[correct],choice_count=k,
                validation={'unique_canonical_answer':sum(o['value']==value for o in options)==1,
                            'gold_from_parent_oracle':True,'not_independent_from_open_version':True})
    return [result[r['id']] for r in parents]


def validate_mcq(rows=None):
    expected=build_mcq()
    if rows is None:rows=[json.loads(s) for s in (ROOT/'mcq_dataset.jsonl').read_text().splitlines()]
    if rows!=expected:raise ValueError('MCQ differs from deterministic construction')
    counts=Counter(r['choice_count'] for r in rows)
    positions={str(k):dict(Counter(r['gold_label'] for r in rows if r['choice_count']==k)) for k in counts}
    for card in {r['article_key'] for r in rows}:
        rs=[r for r in rows if r['article_key']==card];c=Counter(r['gold_label'] for r in rs)
        assert max(c.values())-min(c.values())<=1
    return dict(version=VERSION,records=len(rows),choice_counts=dict(counts),gold_positions=positions,
                random_choice_micro_accuracy=sum(1/r['choice_count'] for r in rows)/len(rows),
                semantic_distractor_validation=(
                    'Canonical consequences are mutually exclusive under the parent unanimity semantics; '
                    'branch outcomes, the documented extension confusion, and explicitly marked numeric '
                    'perturbations where needed. Synthetic distractors are not legal branches.'
                ))


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true')
    if parser.parse_args().check:
        print(json.dumps(validate_mcq(),ensure_ascii=False,indent=2));return
    rows=build_mcq();(ROOT/'mcq_dataset.jsonl').write_bytes(jsonl(rows))
    (ROOT/'mcq_validation.json').write_text(json.dumps(validate_mcq(rows),ensure_ascii=False,indent=2)+'\n')
    print(f'Built {len(rows)} linked MCQ records; freeze after validation.')


if __name__=='__main__':main()
