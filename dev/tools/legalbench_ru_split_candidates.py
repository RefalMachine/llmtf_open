"""Reproduce the initial candidate split; human review precedes any new freeze.

Writes a candidate manifest, never the packaged frozen resource.
"""
import argparse
import json,hashlib,re,random,collections,difflib,importlib.util
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data',required=True,type=Path)
parser.add_argument('--output',required=True,type=Path)
args=parser.parse_args()
r=[json.loads(line) for line in args.data.read_text().splitlines() if line.strip()]
r=[row for row in r if '_canary' not in row]
key=lambda x:(x['task'],x['id'])
bucket=lambda x:'extraction_'+x['track'] if x['answer_type']=='extraction' else x['answer_type']
norm=lambda x:re.sub(r'\W+',' ',(x['question']+' '+x.get('context','')).lower().replace('ё','е')).strip()
texts={key(x):norm(x) for x in r}
# Conservatively reject demo candidates with >= .72 token Jaccard similarity.
tokens={k:set(t.split()) for k,t in texts.items()}
neighbors={k:[] for k in texts}
for i,a in enumerate(r):
 for b in r[i+1:]:
  ka,kb=key(a),key(b);ta,tb=tokens[ka],tokens[kb];s=len(ta&tb)/len(ta|tb)
  if s>=.72:
   neighbors[ka].append(kb);neighbors[kb].append(ka)
eligible=[x for x in r if not any(x.get(f) for f in ('norm_text','distractor_text','temporal_text')) and not neighbors[key(x)]]
random.Random(555).shuffle(eligible)
pools={}
for b in sorted(set(map(bucket,r))):
 c=[x for x in eligible if bucket(x)==b]; chosen=[]
 if b=='binary':
  for label in ['Да','Нет','Да','Нет','Да']:
   chosen.append(next(x for x in c if x['answer']==label and x not in chosen))
 elif b=='multiple_choice':
  for label in 'ABCDA':chosen.append(next(x for x in c if x['answer']==label and x not in chosen))
 elif b=='tool_call':
  servers=set()
  for x in c:
   tool=x['answer']['tool']
   if tool and tool.split('.')[0] not in servers:
    chosen.append(x);servers.add(tool.split('.')[0])
   if len(chosen)==4:break
  chosen.append(next(x for x in c if x['answer']['tool'] is None))
 else:
  for x in c:
   if x['task'] not in {y['task'] for y in chosen}:chosen.append(x)
   if len(chosen)==5:break
  chosen += [x for x in c if x not in chosen][:5-len(chosen)]
  if b=='norm_citation' and not any(len(x['answer'])>1 for x in chosen):
   chosen[-1]=next(x for x in c if len(x['answer'])>1 and x not in chosen)
 assert len(chosen)==5,(b,len(chosen))
 pools[b]=chosen
used={key(x) for c in pools.values() for x in c};ev=[x for x in r if key(x) not in used]
sm=[]
for kind in ('norm_citation','extraction','multiple_choice'):
 sm.append(next(x for x in ev if x['answer_type']==kind and (kind!='norm_citation' or len(x['answer'])>1)))
for label in ('Да','Нет'):sm.append(next(x for x in ev if x['answer_type']=='binary' and x['answer']==label))
for positive in (True,False):sm.append(next(x for x in ev if x['answer_type']=='tool_call' and bool(x['answer']['tool'])==positive))
sm.append(next(x for x in ev if x.get('temporal_text') and x not in sm))
manifest={'version':'singleton_buckets_v1','seed':555,'duplicate_audit':{'policy':'question+context normalized token Jaccard >= 0.72 excludes candidate','candidate_singletons':len(eligible),'flagged_pairs':sum(map(len,neighbors.values()))//2,'legal_expert_review':False},'demonstrations':{b:[key(x) for x in c] for b,c in pools.items()},'evaluation':[key(x) for x in ev],'smoke':[key(x) for x in sm],'context_smoke':{m:[key(x) for x in ev if x.get(f)][:8] for m,f in [('grounded','norm_text'),('distractor','distractor_text'),('temporal','temporal_text')]}}
args.output.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print('Wrote candidate split:', args.output, 'evaluation count:', len(ev))
