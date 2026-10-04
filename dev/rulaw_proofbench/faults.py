"""Public fault experiment against a frozen validator, including coherent rule faults."""
from collections import Counter, defaultdict
from copy import deepcopy
import json

try:
    from .generate import ROOT, build, digest, jsonl, make_record, read_json
    from .validate import validate_record
except ImportError:
    from generate import ROOT, build, digest, jsonl, make_record, read_json
    from validate import validate_record


def threshold_nodes(expr):
    if not isinstance(expr,list):return
    if expr[0]=='lt':yield expr
    for item in expr[1:]:
        yield from threshold_nodes(item)


def experiment():
    cards,rows,_=build();lookup={c['rule_id']:c for c in cards}
    frozen=digest((ROOT/'validate.py').read_bytes())
    faulty=[]
    def add(kind,row,description,card=None):
        detectors=validate_record(row,card or lookup[row['rule_id']])
        record=dict(id=f'{kind}-{sum(f["class"]==kind for f in faulty):03}',**{'class':kind},
                    original_id=row['id'],description=description,record=row,
                    detected_by=detectors,validator_sha256=frozen)
        if card:record['mutated_card']=card
        faulty.append(record)
    for original in rows[:10]:
        r=deepcopy(original);r['gold']['value']='999';add('wrong_gold',r,'Заменён только gold на 999.')
        r=deepcopy(original);r['locator']='несуществующая часть 999';r['question']=r['question'].replace(original['locator'],r['locator']);add('wrong_clause',r,'Согласованно подменены локатор и ссылка в вопросе.')
        r=deepcopy(original);r['as_of']='2030-01-01';r['question']=r['question'].replace(original['as_of'],r['as_of']);add('wrong_date',r,'Согласованно подменены дата поля и дата вопроса.')
        r=deepcopy(original);r['source_ids']=['nonexistent-official-document'];add('wrong_source',r,'Подмена ссылки на отсутствующий источник.')
        r=deepcopy(original);r['target_scope']='whole_case_outcome';r['question']=r['question'].replace('Дайте только короткий ответ без объяснения.','Каков окончательный исход реального дела с учётом всех норм права?')
        add('scope_attack',r,'Расширена область вывода до всего дела. Совместимый контрпример и первичная норма сохранены в counterexample_log.jsonl для '+original['rule_id'])
    for original in [r for r in rows if any(type(v) is bool for v in r['facts'].values())][:10]:
        r=deepcopy(original);key=next(k for k,v in r['facts'].items() if type(v) is bool);r['facts'][key]=not r['facts'][key]
        add('condition_inversion',r,'Инвертирован булев слот '+key+' без изменения текста/доказательства.')
    for original in [r for r in rows if ' не ' in r['question']][:10]:
        r=deepcopy(original);r['question']=r['question'].replace(' не ',' ',1);add('missing_negation',r,'Удалено первое отдельное отрицание из текста вопроса.')
    # Mutate actual executable exceptions, then regenerate question, gold and proof.
    # A repeat render cannot catch this: the isolated table must disagree.
    for original in rows:
        if sum(f['class']=='omitted_exception' for f in faulty)>=10:break
        c=deepcopy(lookup[original['rule_id']]);c['branches']=c['branches'][1:]
        changed=make_record(c,original['facts'])
        if changed['gold']!=original['gold']:
            add('omitted_exception',changed,'Удалена первая приоритетная ветвь; все поля пересозданы из повреждённого правила.',c)
    for original in rows:
        if sum(f['class']=='boundary_error' for f in faulty)>=10:break
        original_card=lookup[original['rule_id']]
        for index in range(sum(1 for b in original_card['branches'] for _ in threshold_nodes(b['when']))):
            c=deepcopy(original_card);nodes=[n for b in c['branches'] for n in threshold_nodes(b['when'])]
            nodes[index][2]+=1
            changed=make_record(c,original['facts'])
            if changed['gold']!=original['gold']:
                add('boundary_error',changed,'Порог < n заменён на < n+1; gold, вопрос и proof пересозданы.',c)
                break
    false_positives=[r['id'] for r in rows if validate_record(r,lookup[r['rule_id']])]
    matrix=defaultdict(Counter)
    for f in faulty:
        matrix[f['class']]['injected']+=1
        matrix[f['class']]['detected']+=bool(f['detected_by'])
        matrix[f['class']].update(f['detected_by'])
    report=dict(validator_sha256=frozen,classes=dict(matrix),false_positives=false_positives,
                clean_records=len(rows),missed=[f['id'] for f in faulty if not f['detected_by']],
                inference_limit='Known synthetic corruption detection, not an estimate of unknown legal error prevalence.')
    # Preserve earlier failures if a validator was subsequently repaired.
    old_path=ROOT/'fault_report.json'
    if old_path.exists():
        old=read_json('fault_report.json')
        if old['missed'] or old['false_positives']:
            (ROOT/('fault_report_failed_'+old['validator_sha256'][:12]+'.json')).write_text(json.dumps(old,ensure_ascii=False,indent=2)+'\n')
    (ROOT/'fault_injection.jsonl').write_bytes(jsonl(faulty))
    old_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return int(bool(report['missed'] or false_positives or len(faulty)!=90))


if __name__=='__main__':
    raise SystemExit(experiment())
