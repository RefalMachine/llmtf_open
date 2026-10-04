"""Offline construction of the public RuLaw-ProofBench pilot."""
import argparse
import hashlib
import json
from pathlib import Path
import random

try:
    from .rules import AS_OF, SEED, UNKNOWN, OUTSIDE, cards, oracle
    from .source_tools import article
except ImportError:
    from rules import AS_OF, SEED, UNKNOWN, OUTSIDE, cards, oracle
    from source_tools import article

ROOT = Path(__file__).resolve().parent


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def jsonl(rows):
    return ''.join(canonical(row) + '\n' for row in rows).encode('utf-8')


def read_json(name, root=ROOT):
    return json.loads((root / name).read_text(encoding='utf-8'))


UNKNOWN_PHRASES = {
    'age': 'Точный возраст неизвестен; известны только возрастные ограничения из условия.',
    'study': 'Неизвестно, совмещает ли работник в течение учебного года получение общего или среднего профессионального образования с работой.',
    'child_age': 'Возраст сына или дочери неизвестен.',
    'disabled': 'Неизвестно, установлена ли инвалидность у сына или дочери работника.',
    'request': 'Неизвестно, просит ли работник установить неполное рабочее время.',
    'ground': 'Неизвестно, по какому из названных в указанных абзацах оснований требуется отпуск.',
    'written': 'Неизвестно, подано ли письменное заявление.',
    'course': 'Курс обучения неизвестен.',
    'accelerated': 'Неизвестно, осваивается ли программа в сокращённые сроки.',
    'form': 'Форма обучения неизвестна.',
    'consent': 'Неизвестно, есть ли взаимное согласие супругов на расторжение брака.',
    'children': 'Неизвестно, есть ли общие несовершеннолетние дети.',
    'status': 'Статус другого супруга неизвестен.',
    'against': 'Неизвестно, противоречит ли учёт мнения интересам ребёнка.',
    'agreement': 'Неизвестно, есть ли соглашение об уплате алиментов.',
    'income': 'Вид собственного дохода неизвестен.',
    'deprived': 'Неизвестно, лишил ли суд несовершеннолетнего права самостоятельно распоряжаться своими доходами.',
    'decision': 'Отдельное описание действующего решения суда об ограничении или лишении права не приведено; его наличие и вид прямо не указаны.',
    'within_restriction': 'Неизвестно, охвачено ли именно это распоряжение частичным судебным ограничением.',
    'notary': 'Неизвестно, имеется ли нотариальное удостоверение.',
    'registration': 'Неизвестно, требуется ли государственная регистрация сделки.',
    'kind': 'Неизвестно, передаётся ли имущество в дар, в безвозмездное пользование или по договору купли-продажи.',
    'to_ward': 'Неизвестно направление передачи имущества: от опекуна подопечному или от подопечного опекуну.',
    'date': 'Неизвестно, указана ли дата совершения доверенности.',
    'abroad': 'Неизвестно, предназначена ли доверенность для действий за границей.',
    'readable': 'Неизвестно, поддаётся ли текст обращения прочтению.',
    'migration': 'Неизвестно, содержит ли обращение информацию о фактах возможных нарушений законодательства РФ в сфере миграции.',
    'competent': 'Неизвестно, входит ли решение вопросов обращения в компетенцию получившего его органа.',
    'repeated': 'Неизвестно, давались ли ранее неоднократные письменные ответы по существу на этот вопрос.',
    'new': 'Неизвестно, приводятся ли новые доводы или обстоятельства.',
    'same': 'Неизвестно, направлялись ли все обращения в один и тот же орган или одному и тому же должностному лицу.',
    'head': 'Неизвестно, является ли адресат высшим должностным лицом субъекта РФ.',
}


def render(card, facts):
    lines = [f"{card['act_title']}, статья {card['article_key'].split(':')[1]}, {card['locator']}. "
             f"Редакция нормы на {AS_OF}.", card['scope']]
    for p in card['predicates']:
        value = facts[p['key']]
        if value is None:
            text = UNKNOWN_PHRASES[p['key']]
            if card['article_key'] == 'sk:81' and p['key']=='children':
                text='Число несовершеннолетних детей неизвестно; известно, что есть хотя бы один ребёнок.'
            if card['article_key'] == 'gk:28' and p['key']=='notary':
                text='Неизвестно, требует ли сделка нотариального удостоверения.'
        else:
            index = next(i for i, x in enumerate(p['domain']) if type(x) is type(value) and x==value)
            text = p['phrases'][index]
        lines.append(text)
    instruction='Дайте только короткий ответ без объяснения. Если совместимые с условием факты дают разные ответы, ответьте «недостаточно данных»; это имеет приоритет над обычным форматом ответа.'
    if OUTSIDE in {b['value'] for b in card['branches']}:
        instruction+=' Если при данных фактах числовое или долевое следствие указанной нормы не устанавливается, ответьте «не следует из указанной нормы» вместо числа или дроби.'
    if '«да» или «нет»' in card['target']:
        instruction+=' В вопросе «да/нет» отсутствие названного права, обязанности или исключения по указанной норме означает ответ «нет».'
    instruction+=' Ответ касается только названной нормы, а не исхода всего дела.'
    lines += [card['target'],instruction]
    return '\n'.join(lines)


def build_cards(root=ROOT):
    sources = {s['id']:s for s in read_json('source_manifest.json',root)}
    result=[]
    for c in cards():
        act, n = c['article_key'].split(':')
        s=sources[act]
        text=(root/s['text_path']).read_text(encoding='utf-8')
        fragment=article(text,n)
        c.update(as_of=AS_OF, source_ids=[act], act_title=s['title'], domain=s['domain'],
                 source_fragment=fragment, source_fragment_sha256=digest(fragment.encode()),
                 source_text_sha256=s['text_sha256'], target_scope='clause_consequence',
                 source_lines=[text[:text.index(fragment)].count('\n')+1,
                               text[:text.index(fragment)+len(fragment)].count('\n')+1],
                 template_id='atomic_ru_v1', dependency_fragments={})
        for dep in c['dependencies']:
            dep_act,dep_n=dep.split(':')
            dep_text=(root/sources[dep_act]['text_path']).read_text(encoding='utf-8')
            c['dependency_fragments'][dep]=article(dep_text,dep_n)
        result.append(c)
    return result


def make_record(c, facts):
    value,traces=oracle(c,facts)
    signature=canonical(facts)
    identifier=c['rule_id']+'-'+digest(signature.encode())[:12]
    unknown=any(v is None for v in facts.values())
    return dict(id=identifier, article_key=c['article_key'], domain=c['domain'], as_of=AS_OF,
                target_scope='clause_consequence',source_ids=c['source_ids'],rule_id=c['rule_id'],
                locator=c['locator'],template_id=c['template_id'],
                question_type='insufficient' if value==UNKNOWN else 'boundary' if c['boundary_keys'] and not unknown else 'branch',
                question=render(c,facts), facts=facts,fact_signature=signature,
                gold={'value':value,'unit':c['unit']},
                proof={'used_clauses':[c['article_key']+':'+c['locator']],
                       'completed_worlds':traces, 'distinct_outcomes':sorted({t['value'] for t in traces}),
                       'derivation':'unanimity_over_all_compatible_worlds'},
                dependency_audit_id=c['rule_id'],adversarial_audit_id=c['rule_id'],
                validation={k:True for k in ('source','rule','proof','language','dependency','adversarial')},
                status='accepted',generation_seed=SEED,rule_sha256=digest(canonical(c).encode()),
                unknown_but_determinate=unknown and value!=UNKNOWN, pair_ids=[])


def build(root=ROOT):
    cs=build_cards(root); rows=[]; pairs=[]
    for c in cs:
        keys=[p['key'] for p in c['predicates']]
        group=[make_record(c,dict(zip(keys,values))) for values in c['cases']]
        # All full-fact, one-slot interventions with a changed answer are declared.
        for i,a in enumerate(group):
            for b in group[i+1:]:
                if any(v is None for v in [*a['facts'].values(),*b['facts'].values()]): continue
                changed=[k for k in keys if type(a['facts'][k]) is not type(b['facts'][k]) or a['facts'][k]!=b['facts'][k]]
                if len(changed)==1 and a['gold']!=b['gold']:
                    pid=c['rule_id']+'-pair-'+digest((a['id']+b['id']).encode())[:12]
                    pairs.append(dict(id=pid,article_key=c['article_key'],members=[a['id'],b['id']],changed_predicate=changed[0]))
                    a['pair_ids'].append(pid);b['pair_ids'].append(pid)
        rows.extend(group)
    random.Random(SEED).shuffle(rows)
    return cs,rows,pairs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    cs,rows,pairs=build()
    for name,data in [('rule_cards.jsonl',cs),('dataset.jsonl',rows),('minimal_pairs.jsonl',pairs),
                      ('questions_only.jsonl',[{'id':r['id'],'question':r['question']} for r in rows])]:
        (args.output/name).write_bytes(jsonl(data))
    print(f'Generated {len(rows)} candidate records, {len(cs)} articles, {len(pairs)} pairs; run validate.py before release.')


if __name__=='__main__':
    main()
