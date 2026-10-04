"""Fail-closed offline acceptance. Exit 0 means construction checks, not expert certification."""
import argparse
from collections import Counter
import csv
from itertools import product
import json
from pathlib import Path
import sys

try:
    from .generate import ROOT, AS_OF, SEED, build, canonical, digest, jsonl, read_json, render
    from .rules import oracle, worlds, UNKNOWN
    from .source_tools import extract, article
    from .compare import independent_value, SECOND
except ImportError:
    from generate import ROOT, AS_OF, SEED, build, canonical, digest, jsonl, read_json, render
    from rules import oracle, worlds, UNKNOWN
    from source_tools import extract, article
    from compare import independent_value, SECOND

VERSION='rulaw_validator_v1'


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def validate_record(row, card):
    """Independent detectors; also usable on public corrupted copies."""
    errors=[]
    def check(ok, detector):
        if not ok: errors.append(detector)
    try:
        check(row['article_key']==card['article_key'] and row['rule_id']==card['rule_id'] and
              row['locator']==card['locator'] and row['source_ids']==card['source_ids'], 'source_binding')
        check(row['as_of']==AS_OF and row['target_scope']=='clause_consequence', 'date_and_scope')
        check(row['status']=='accepted' and row['generation_seed']==SEED and
              row['validation']=={k:True for k in ('source','rule','proof','language','dependency','adversarial')},'acceptance_fields')
        check(row['rule_sha256']==digest(canonical(card).encode()),'rule_integrity')
        check(row['question']==render(card,row['facts']), 'language_render')
        check(row['fact_signature']==canonical(row['facts']), 'fact_binding')
        value, traces=oracle(card,row['facts'])
        check(row['gold']=={'value':value,'unit':card['unit']}, 'oracle_gold')
        check(row['proof']=={'used_clauses':[card['article_key']+':'+card['locator']],
                            'completed_worlds':traces,'distinct_outcomes':sorted({t['value'] for t in traces}),
                            'derivation':'unanimity_over_all_compatible_worlds'}, 'proof_replay')
        independent={independent_value(card['article_key'], t['facts']) for t in traces}
        gold=next(iter(independent)) if len(independent)==1 else UNKNOWN
        check(row['gold']['value']==gold,'independent_table')
        check(row['unknown_but_determinate']==(any(v is None for v in row['facts'].values()) and value!=UNKNOWN),'unknown_semantics')
    except (KeyError,TypeError,ValueError,StopIteration):
        errors.append('schema_or_domain')
    return sorted(set(errors))


def validate(root=ROOT, require_freeze=True, require_faults=True):
    failures=[]; details={}; errors={}
    def check(ok, message):
        if not ok: failures.append(message)
    try:
        cs,expected,pairs=build(root)
        rows=read_rows(root/'dataset.jsonl')
        check((root/'dataset.jsonl').read_bytes()==jsonl(expected),'dataset differs from canonical deterministic regeneration')
        check((root/'rule_cards.jsonl').read_bytes()==jsonl(cs),'rule cards differ from source construction')
        check((root/'minimal_pairs.jsonl').read_bytes()==jsonl(pairs),'minimal pairs differ from complete one-slot comparison')
        check((root/'questions_only.jsonl').read_bytes()==jsonl([{'id':r['id'],'question':r['question']} for r in expected]),'question-only export differs')
        sources={s['id']:s for s in read_json('source_manifest.json',root)}
        for sid,s in sources.items():
            raw=(root/s['raw_path']).read_bytes(); text=(root/s['text_path']).read_text(encoding='utf-8')
            check(digest(raw)==s['sha256'] and digest(text.encode())==s['text_sha256'],f'source hash: {sid}')
            check(extract(raw)==text,f'source extraction: {sid}')
            check(text.rstrip().endswith(s['full_document_end']) and s['number'] in text.splitlines()[-1],f'source completeness: {sid}')
            check(s['as_of']==AS_OF and s['edition_audit_status']=='verified_selected_fragments',f'edition audit incomplete: {sid}')
            check(s['url'].startswith('http://pravo.gov.ru/proxy/ips/'),f'non-primary source: {sid}')
        for s in read_json('supporting_sources.json',root):
            raw=(root/s['raw_path']).read_bytes();txt=(root/s['text_path']).read_text(encoding='utf-8')
            check(digest(raw)==s['sha256'] and digest(txt.encode())==s['text_sha256'] and extract(raw)==txt,f'supporting source {s["id"]}')
        edition=read_json('edition_audit.json',root)
        check(edition['as_of']==AS_OF and edition['status']=='selected_fragments_audited','edition audit status')
        check('вступает в силу с 30 марта 2025 года' in (root/'sources/amendment-2024-547.txt').read_text(encoding='utf-8'),'deferred electronic amendment date')
        for sid,s in sources.items():
            before=(root/s['text_path']).read_text(encoding='utf-8')
            after=(root/('sources/'+sid+'-comparison.txt')).read_text(encoding='utf-8')
            for n in s['unchanged_articles']:
                check(article(before,n)==article(after,n),f'edition fragment changed {sid}:{n}')
        audits=read_json('audit_resolutions.json',root)
        check(audits['status']=='resolved_with_documented_limits','unresolved language/dependency audit')
        check(audits['rules_sha256']==digest((root/'rules.py').read_bytes()),'audit refers to different rule code')
        check(audits['generator_sha256']==digest((root/'generate.py').read_bytes()),'audit refers to different renderer')
        findings=read_json('adversarial_review.json',root)
        finding_ids={f['id'] for f in findings['global_findings']}
        finding_ids.update(f['id'] for r in findings['rules'] for f in r['findings'])
        check({r['finding_id'] for r in audits['resolutions']}==finding_ids,'unaccounted audit findings')
        final_review=read_json('final_language_review.json',root)
        check(not final_review['blocking_findings'],'unresolved final language findings')
        for name,sha in final_review['reviewed_files_sha256'].items():
            check(digest((root/name).read_bytes())==sha,f'final language review obsolete: {name}')
        for review in read_json('independent_review.json',root)['rules']:
            for q in review['quotes']:
                check(q['text'] in (root/sources[q['source']]['text_path']).read_text(encoding='utf-8'),f'independent quote {review["id"]}')
        by_id={c['rule_id']:c for c in cs}
        comparison_count=0; partial_count=0
        for c in cs:
            source_text=(root/sources[c['source_ids'][0]]['text_path']).read_text(encoding='utf-8')
            fragment=article(source_text,c['article_key'].split(':')[1])
            check(fragment==c['source_fragment'] and digest(fragment.encode())==c['source_fragment_sha256'],f'fragment {c["rule_id"]}')
            for p in c['predicates']:
                check(p['anchor'] in fragment, f'predicate anchor {c["rule_id"]}/{p["key"]}')
            for dep, frag in c['dependency_fragments'].items():
                a,n=dep.split(':');check(article((root/sources[a]['text_path']).read_text(encoding='utf-8'),n)==frag,f'dependency {dep}')
            for w in worlds(c,{p['key']:None for p in c['predicates']}):
                gold,_=oracle(c,w)
                check(gold==independent_value(c['article_key'],w),f'independent complete-world mismatch {c["rule_id"]}: {w}')
                comparison_count+=1
            # Include every partial vector, not only selected evaluation questions.
            for values in product(*[p['domain']+[None] for p in c['predicates']]):
                f=dict(zip([p['key'] for p in c['predicates']],values))
                ws=worlds(c,f)
                if not ws: continue
                answers={independent_value(c['article_key'],w) for w in ws}
                gold,_=oracle(c,f)
                check(gold==(next(iter(answers)) if len(answers)==1 else UNKNOWN),f'independent partial-world mismatch {c["rule_id"]}: {f}')
                partial_count+=1
        for row in rows:
            result=validate_record(row,by_id.get(row.get('rule_id'),{}))
            if result:errors[row.get('id','<missing>')]=result
        check(not errors,'record validation errors')
        counts=Counter(r['article_key'] for r in rows)
        check(100<=len(rows)<=200,'expected 100..200 records')
        check(10<=len(counts)<=20,'expected 10..20 articles')
        check(len({k.split(':')[0] for k in counts})>=4,'expected >=4 acts')
        check(len({r['domain'] for r in rows})>=3,'expected >=3 domains')
        check(all(6<=n<=12 and n/len(rows)<=.12 for n in counts.values()),'article quota')
        check(len({r['id'] for r in rows})==len(rows),'duplicate IDs')
        check(len({' '.join(r['question'].lower().split()) for r in rows})==len(rows),'duplicate questions')
        check(max(Counter((r['article_key'],r['fact_signature']) for r in rows).values())<=2,'duplicate fact vectors')
        for a in counts:
            check(len({r['fact_signature'] for r in rows if r['article_key']==a})>=6,f'insufficient distinct vectors: {a}')
        probes={'tk:92':{'age':[15,16,17]},'tk:93':{'child_age':[13,14,15,17,18,19]},
                'tk:94':{'age':[14,15,16,17]},'sk:19':{'status':['prison_2','prison_3','prison_4']},
                'sk:57':{'age':[9,10,11]},'sk:81':{'children':[1,2,3,4]},'gk:28':{'age':[5,6,7]}}
        for a,fields in probes.items():
            for field,values in fields.items():
                observed={r['facts'][field] for r in rows if r['article_key']==a}
                check(set(values)<=observed,f'missing boundary probes {a}/{field}')
        for p in pairs:
            members=[r for r in rows if r['id'] in p['members']]
            check(len(members)==2 and members[0]['gold']!=members[1]['gold'],f'bad pair {p["id"]}')
        registry=list(csv.DictReader((root/'candidate_registry.csv').open(encoding='utf-8')))
        check({r['article_key'] for r in registry if r['status']=='accepted'}==set(counts),'candidate registry mismatch')
        if require_freeze:
            freeze=read_json('release_manifest.json',root)
            for name,sha in freeze['files'].items():check(digest((root/name).read_bytes())==sha,f'frozen file changed: {name}')
            check(freeze['as_of']==AS_OF and freeze['seed']==SEED,'freeze configuration')
        if require_faults:
            faults=read_rows(root/'fault_injection.jsonl')
            check(len(faults)>=90,'fault experiment incomplete')
            fault_counts=Counter(f['class'] for f in faults)
            required={'wrong_gold','condition_inversion','missing_negation','wrong_clause','wrong_date','omitted_exception','boundary_error','wrong_source','scope_attack'}
            check(required<=set(fault_counts) and all(fault_counts[k]>=10 for k in required),'fault classes incomplete')
            for f in faults:
                c=f.get('mutated_card') or by_id[f['record']['rule_id']]
                detected=validate_record(f['record'],c)
                check(bool(detected) and detected==f['detected_by'],f'undetected/unreproduced fault {f["id"]}')
                check(f['validator_sha256']==digest((root/'validate.py').read_bytes()),'fault experiment ran a different validator')
            details['faults']=dict(Counter(f['class'] for f in faults))
        details.update(records=len(rows),articles=len(counts),acts=len(sources),domains=len({r['domain'] for r in rows}),
                       article_counts=dict(counts),answer_counts=dict(Counter(r['gold']['value'] for r in rows)),
                       question_types=dict(Counter(r['question_type'] for r in rows)),
                       unknown_but_determinate=sum(r['unknown_but_determinate'] for r in rows),
                       minimal_pairs=len(pairs),complete_world_comparisons=comparison_count,partial_world_comparisons=partial_count,
                       candidate_articles=len(registry),article_rejections=dict(Counter(r['rejection_code'] for r in registry if r['status']=='rejected')),
                       false_positives=len(errors))
    except (OSError,ValueError,KeyError,TypeError,IndexError) as e:
        failures.append(f'{type(e).__name__}: {e}')
    return dict(validator_version=VERSION,as_of=AS_OF,seed=SEED,
                release_status='incomplete' if failures else 'construction_validated',
                measurement_status='pending_multimodel_cluster_evaluation',expert_review=False,human_audit=False,
                guarantees='Finite formalization consistency and archived-source traceability; not exhaustive legal correctness.',
                failures=failures,record_errors=errors,details=details)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--report',type=Path)
    a=p.parse_args(); report=validate(a.root)
    if a.report:a.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 1 if report['failures'] else 0


if __name__=='__main__':
    sys.exit(main())
