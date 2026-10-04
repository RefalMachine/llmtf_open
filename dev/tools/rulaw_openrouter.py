"""Bounded, cached OpenRouter research controls; credentials never enter artifacts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import logging
import os
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request

from llmtf.config import SamplingConfig


class OpenRouterResearchClient:
    def __init__(self,model,credential_file,output,proxy=None,max_cost=2.0):
        self.model=model;self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.cache=self.output/'requests';self.cache.mkdir(exist_ok=True)
        self._api_key=os.environ.get('OPENROUTER_API_KEY')
        if credential_file:
            data=Path(credential_file).read_text().strip()
            if not data.startswith('OPENROUTER_API_KEY='):raise ValueError('Unexpected runtime credential format')
            self._api_key=data.split('=',1)[1]
        if not self._api_key:raise ValueError('Runtime OPENROUTER_API_KEY required')
        self.proxy=proxy;self.max_cost=max_cost;self.lock=threading.Lock()
        self.cost=sum(json.loads(p.read_text())['usage_cost_usd'] for p in self.cache.glob('*.json'))
        self.generation_config=SamplingConfig(temperature=0,do_sample=False,max_new_tokens=96)
        self.logger=logging.getLogger('rulaw_openrouter')

    def get_params(self):
        return dict(transport='OpenRouter research client',model_name_or_path=self.model,
                    endpoint='https://openrouter.ai/api/v1',native_reasoning={'enabled':False},
                    concurrency=4,temperature=0,cache='request-payload-sha256',max_cost_usd=self.max_cost)

    def _request(self,path,payload=None):
        handlers=[urllib.request.ProxyHandler({'http':self.proxy,'https':self.proxy})] if self.proxy else []
        opener=urllib.request.build_opener(*handlers)
        headers={'Authorization':'Bearer '+self._api_key,'Content-Type':'application/json'}
        data=json.dumps(payload,ensure_ascii=False).encode() if payload is not None else None
        request=urllib.request.Request('https://openrouter.ai/api/v1/'+path,data=data,headers=headers)
        for attempt in range(3):
            try:
                with opener.open(request,timeout=120) as response:return json.load(response)
            except urllib.error.HTTPError as error:
                if error.code not in (408,429,500,502,503,504) or attempt==2:
                    raise RuntimeError(f'OpenRouter HTTP {error.code}; response body omitted to protect credentials') from None
            except urllib.error.URLError:
                if attempt==2:raise RuntimeError('OpenRouter network request failed') from None
            time.sleep(2**attempt)

    def catalog(self):
        model=next((m for m in self._request('models')['data'] if m['id']==self.model),None)
        if model is None:raise ValueError('Requested model not present in OpenRouter catalog')
        result={k:model.get(k) for k in ('id','name','context_length','pricing','supported_parameters')}
        (self.output/'model_catalog.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        return result

    def _generate(self,messages,config):
        payload=dict(model=self.model,messages=messages,temperature=0,max_tokens=config.max_new_tokens,
                     reasoning={'enabled':False},usage={'include':True},provider={'require_parameters':True})
        signature=hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        path=self.cache/(signature+'.json')
        if path.exists():record=json.loads(path.read_text())
        else:
            with self.lock:
                if self.cost+.02>self.max_cost:raise RuntimeError('Local research cost cap reached')
            response=self._request('chat/completions',payload)
            usage=response.get('usage',{});cost=usage.get('cost')
            if cost is None:raise RuntimeError('OpenRouter did not report cost; stopped before further requests')
            record=dict(request=payload,response=response,usage_cost_usd=float(cost))
            path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
            with self.lock:self.cost+=float(cost)
        # Routing can violate the requested mode. Preserve and charge rejected
        # responses, but never use them as no-reasoning experimental evidence.
        for attempt in range(3):
            reported=record['response'].get('usage',{}).get('completion_tokens_details',{}).get('reasoning_tokens',0)
            if not reported:break
            record.setdefault('rejected_reasoning_responses',[]).append(record['response'])
            with self.lock:
                if self.cost+.02>self.max_cost:raise RuntimeError('Local research cost cap reached during mode retry')
            response=self._request('chat/completions',payload)
            cost=response.get('usage',{}).get('cost')
            if cost is None:raise RuntimeError('Retry cost missing')
            record['response']=response;record['usage_cost_usd']+=float(cost)
            path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
            with self.lock:self.cost+=float(cost)
        if record['response'].get('usage',{}).get('completion_tokens_details',{}).get('reasoning_tokens',0):
            raise RuntimeError('Provider did not honor disabled reasoning after retries')
        response=record['response'];choice=response['choices'][0]
        content=choice['message'].get('content')
        if not isinstance(content,str):raise ValueError('OpenRouter returned no final text')
        info=dict(prompt_len=response.get('usage',{}).get('prompt_tokens'),
                  generated_len=[response.get('usage',{}).get('completion_tokens')],
                  usage=response.get('usage'),finish_reason=choice.get('finish_reason'),
                  response_id=response.get('id'),provider=response.get('provider'),
                  request_sha256=signature,native_reasoning_requested=False)
        return messages,content,info

    def generate_batch(self,messages,*,generation_config=None,enable_thinking=False,**kwargs):
        if enable_thinking:raise ValueError('This research client explicitly disables native reasoning')
        config=generation_config or self.generation_config
        with ThreadPoolExecutor(max_workers=4) as pool:
            records=list(pool.map(lambda m:self._generate(m,config),messages))
        return tuple([r[i] for r in records] for i in range(3))


def comparison_report(output):
    """Rebuild the requested Qwen comparison from saved answers, without API calls."""
    from collections import Counter
    import gzip
    from dev.rulaw_proofbench.analyze import scoring, paired_bootstrap
    from dev.rulaw_proofbench.generate import build

    output=Path(output);root=output.parent
    cards,rows,_=build();expected={r['id']:r for r in rows};scored={}
    cards={c['rule_id']:c for c in cards};prompt_reference={}
    report=dict(scorer=scoring.SCORER_NAME,
                scorer_sha256=hashlib.sha256(Path(scoring.__file__).read_bytes()).hexdigest(),
                dataset_sha256=hashlib.sha256((root.parent/'rulaw_proofbench/dataset.jsonl').read_bytes()).hexdigest(),
                records_per_mode=len(rows),primary='article_macro_accuracy',few_shot_count=0,
                temperature=0,reasoning_enabled=False,answer_budget=96,judge_answer_budget=256,
                judge_model='deepseek/deepseek-v4.1-flash',models={},comparisons={})
    for size in (2,9,27):
        model=f'Qwen3.5-{size}B';scored[model]={};entry=dict(backend='local vLLM' if size==2 else 'OpenRouter',modes={})
        if size==2:entry['revision']='15852e8c16360a2fea060d615a32b45270f8a8fc'
        else:entry['revision']='Hosted alias; weights/quantization not pinned'
        for mode in ('closed','grounded'):
            if size==2:
                path=root/'artifacts/vllm'/f'rulaw_proofbench_{mode}.jsonl.gz'
                items=json.loads(gzip.decompress(path.read_bytes()))
                judge_path=root/'openrouter'/f'qwen_{mode}_judge.json'
            else:
                path=output/f'qwen35_{size}b_{mode}.jsonl';items=json.loads(path.read_text())
                judge_path=output/f'qwen35_{size}b_{mode}_judge.json'
            judge=json.loads(judge_path.read_text());decisions=judge['details']['samples']
            assert len(items)==len(decisions)==len(rows)
            assert {r['sample']['id'] for r in items}=={r['id'] for r in decisions}==expected.keys()
            by_id={r['id']:r for r in decisions};strict={};semantic={};disagreements=[];length_stops=0
            for item in items:
                row=item['sample'];identifier=row['id'];original=expected[identifier]
                assert all(row[k]==original[k] for k in ('question','gold','facts','rule_sha256'))
                decision=by_id[identifier];assert decision['prediction']==item['predict']
                if size==2:
                    prompt=item['prompt']
                    judge_cache=root/'openrouter/requests'
                else:
                    cached=json.loads((output/'requests'/(item['info']['request_sha256']+'.json')).read_text())
                    assert cached['request']['messages']==item['prompt']
                    assert cached['response']['choices'][0]['message']['content']==item['predict']
                    assert cached['request']['model']==f'qwen/qwen3.5-{size}b'
                    assert cached['request']['max_tokens']==96 and cached['request']['temperature']==0
                    assert len(item['prompt'])==1 and item['prompt'][0]['role']=='user'
                    prompt=item['prompt'][0]['content'];key=(mode,identifier)
                    if key in prompt_reference:assert prompt_reference[key]==prompt
                    else:prompt_reference[key]=prompt
                    judge_cache=output/'requests'
                assert original['question'] in prompt
                if mode=='grounded':
                    card=cards[original['rule_id']]
                    assert card['source_fragment'] in prompt
                    assert all(fragment in prompt for fragment in card['dependency_fragments'].values())
                cached_judge=json.loads((judge_cache/(decision['info']['request_sha256']+'.json')).read_text())
                assert cached_judge['response']['choices'][0]['message']['content']==decision['raw_response']
                judge_input=json.loads(cached_judge['request']['messages'][1]['content'])
                assert judge_input==dict(question=original['question'],reference=original['gold'],prediction=item['predict'])
                result=scoring.score(original,item['predict']);strict[identifier]=result
                accepted=decision['verdict']['equivalent']
                semantic[identifier]=dict(result,correct=accepted,
                    prediction=result['gold'] if accepted else None,format_valid=True)
                if result['correct']!=accepted:disagreements.append(identifier)
                length_stops+=int(item['info'].get('finish_reason')=='length')
            macro,details=scoring.aggregate(list(strict.values()))
            semantic_macro=scoring.aggregate(list(semantic.values()))[0]
            assert abs(semantic_macro-judge['article_macro_semantic_equivalence'])<1e-12
            entry['modes'][mode]=dict(scorer=macro,judge=semantic_macro,details=details,
                disagreement_ids=disagreements,answer_length_stops=length_stops,
                answers=str(path.relative_to(root)),judgments=str(judge_path.relative_to(root)))
            scored[model][mode]={'scorer':strict,'judge':semantic}
        entry['grounded_gain']={metric:paired_bootstrap(scored[model]['grounded'][metric],scored[model]['closed'][metric])
                                for metric in ('scorer','judge')}
        report['models'][model]=entry
    for larger,smaller in ((9,2),(27,9),(27,2)):
        key=f'{larger}B_minus_{smaller}B'
        report['comparisons'][key]=dict(mixed_backends=smaller==2,modes={
            mode:{metric:paired_bootstrap(scored[f'Qwen3.5-{larger}B'][mode][metric],scored[f'Qwen3.5-{smaller}B'][mode][metric])
                  for metric in ('scorer','judge')} for mode in ('closed','grounded')})
    requests=[json.loads(p.read_text()) for p in (output/'requests').glob('*.json')]
    assert all(r['request']['reasoning']=={'enabled':False} for r in requests)
    assert all(r['response'].get('usage',{}).get('completion_tokens_details',{}).get('reasoning_tokens')==0 for r in requests)
    rejected=[v for r in requests for v in r.get('rejected_reasoning_responses',[])]
    report['new_api_usage']=dict(unique_payloads=len(requests),http_completions=len(requests)+len(rejected),
        rejected_reasoning_responses=len(rejected),total_cost_usd=sum(r['usage_cost_usd'] for r in requests),
        by_model=dict(Counter(r['request']['model'] for r in requests)),
        providers=dict(Counter(r['response'].get('provider') for r in requests)),
        completion_reasons=dict(Counter(r['response']['choices'][0].get('finish_reason') for r in requests)))
    report['limitations']=['2B is an existing local run; backend/template/precision differ from hosted 9B/27B.',
        '4B and 2B were absent from the saved OpenRouter catalog; no new local runs were requested.',
        'One model family and 15 selected articles; no claim of general Russian-law ranking validity.',
        'Reasoning is disabled in requests and verified against reported usage for accepted responses.']
    return report


def write_comparison(output):
    output=Path(output);report=comparison_report(output)
    (output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    lines=['# Qwen3.5: сравнение размеров на RuLaw-ProofBench','',
        'По 150 вопросов без нормы и с нормой. Zero-shot, temperature=0, reasoning выключен.',
        'Метрика — accuracy с равным весом каждой из 15 статей. Судья во всех случаях — DeepSeek V4.1 Flash.', '',
        '| Модель | Запуск | Без нормы: scorer | Без нормы: judge | С нормой: scorer | С нормой: judge |',
        '|---|---|---:|---:|---:|---:|']
    for model,entry in report['models'].items():
        values=[entry['modes'][mode][metric] for mode in ('closed','grounded') for metric in ('scorer','judge')]
        lines.append('| '+model+' | '+entry['backend']+' | '+' | '.join(f'{v*100:.2f}%' for v in values)+' |')
    lines+=['','## Разности и неопределённость','',
        'Парный bootstrap по статьям: 10 000 повторов, seed=555. Интервал относится к выбранным 15 статьям, а не ко всему праву РФ.', '',
        '| Сравнение | Режим | Метрика | Разность, п.п. | 95% интервал, п.п. |',
        '|---|---|---|---:|---|']
    for key,comparison in report['comparisons'].items():
        for mode,metrics in comparison['modes'].items():
            for metric,item in metrics.items():
                lo,hi=item['percentile_95_interval'];lines.append(f"| {key} | {mode} | {metric} | {100*item['macro_difference']:.2f} | [{100*lo:.2f}; {100*hi:.2f}] |")
    lines+=['','## Условия и ограничения','',
        '- 2B — ранее выполненный локальный vLLM-прогон; новых локальных запусков нет. 9B и 27B — новые вызовы OpenRouter. Сравнения с 2B включают различия backend, шаблона и возможной точности весов.',
        '- 2B и 4B отсутствуют в сохранённом каталоге OpenRouter. 4B в таблицу не включена.',
        '- Scorer один: [normalized exact match](../../rulaw_proofbench/SCORING.md). Его грамматика не менялась по результатам 9B/27B. Judge сравнивает с фиксированным gold; решения всех моделей проверяются одним промптом.',
        '- Правильная свободная формулировка может не распознаться scorer. Список расхождений по ID, точность по статьям, неизвестным фактам и формату —в `summary.json`; ответы и объяснения judge —в JSONL/JSON рядом.',
        '- Проверены полные уникальные ID, исходные вопросы/gold/правила, соответствие судейских решений ответам и нулевые reported reasoning tokens у принятых API-ответов.',
        '- API alias не закрепляет веса или квантование. Фактические провайдеры, токены, finish_reason и стоимость сохранены в запросах. Это не независимая экспертная валидация gold и не сравнение нескольких модельных семейств.', '',
        '## Вывод по этому эксперименту', '',
        '27B превосходит 9B при одном способе запуска через OpenRouter: без нормы разность обеих метрик 11,82 п.п. с интервалом [1,33;23,33], с нормой — 21,24 п.п. по scorer и 17,24 п.п. по judge; интервалы также выше нуля. Это положительное свидетельство различающей способности на выбранных статьях.',
        'Между 2B и 9B без нормы устойчивого различия не видно: интервалы обеих разностей включают ноль. С нормой различие значительно больше. Набор различает применение предоставленного правила убедительнее, чем соседние уровни закрытого знания в этой паре. Смешение локального/API исполнения у 2B дополнительно ограничивает вывод.', '',
        'При просмотре всех 24 расхождений новых API-прогонов найдена ошибка judge: ответ 9B без нормы на `gk_186-62c320fe2a4f` «...на срок до одного года» зачтён как эталонный фиксированный срок «1 год». Верхняя граница здесь не равна точному сроку; это противоречит уже заданному критерию judge. В таблице сохранён исходный вердикт, без ручной поправки. Остальные 23 расхождения —однозначные формулировки ничтожности/срока и верхних пределов, не входящие в грамматику scorer. Калибровка 150/150 не гарантирует отсутствие ошибок судьи на новых ответах.', '',
        f"Новые API-затраты, включая judge: ${report['new_api_usage']['total_cost_usd']:.6f}; HTTP completions: {report['new_api_usage']['http_completions']}. Старый локальный 2B и его ранее выполненный judge сюда не входят.", '',
        '```bash','python -m dev.tools.rulaw_openrouter report --output dev/rulaw_proofbench_diagnostics/qwen35_openrouter',
        'python -m dev.rulaw_proofbench_diagnostics.replay','```','',
        'Обе команды используют сохранённые ответы и не требуют ключа или новых API-вызовов.']
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({model:{mode:{metric:entry['modes'][mode][metric] for metric in ('scorer','judge')}
                              for mode in ('closed','grounded')} for model,entry in report['models'].items()},ensure_ascii=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('catalog','probe','solve','calibrate','judge','report'))
    p.add_argument('--model',default='deepseek/deepseek-v4.1-flash')
    p.add_argument('--credential-file',type=Path);p.add_argument('--proxy')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--max-cost',type=float,default=2.0)
    p.add_argument('--predictions',type=Path);p.add_argument('--name')
    args=p.parse_args()
    if args.action=='report':
        write_comparison(args.output);return
    client=OpenRouterResearchClient(args.model,args.credential_file,args.output,args.proxy,args.max_cost)
    if args.action=='catalog':
        models=[{k:m.get(k) for k in ('id','name','context_length','pricing','supported_parameters')}
                for m in client._request('models')['data'] if 'qwen3.5' in m['id'].lower()]
        (args.output/'qwen35_catalog.json').write_text(json.dumps(models,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(models,ensure_ascii=False));return
    if args.action=='probe':
        catalog=client.catalog()
        _,answers,infos=client.generate_batch([[{'role':'user','content':'Правило: до 16 лет предел 24 часа, с 16 до 18 лет предел 35 часов. Работнику 14 лет. Каков предел рабочего времени в часах? Ответьте только числом.'}]])
        print(json.dumps(dict(model=catalog,answers=answers,infos=infos,total_cost_usd=client.cost),ensure_ascii=False));return
    from llmtf.tasks.rulaw_proofbench.data import load_rows
    from llmtf.tasks.rulaw_proofbench.scoring import score,aggregate
    from llmtf.tasks.rulaw_proofbench.judge import judge_records,aggregate_judgments,provenance
    rows = load_rows()
    expected = {row['id']: row for row in rows}
    if args.action=='solve':
        from llmtf.tasks.rulaw_proofbench import RuLawProofBench
        results={}
        for mode in ('closed','grounded'):
            task = RuLawProofBench(mode)
            records = []
            for start in range(0,len(rows),8):
                batch=rows[start:start+8];prompts,predictions,infos=client.generate_batch([task.create_messages(r) for r in batch])
                records.extend(dict(sample=r,prompt=m,predict=y,info=i) for r,m,y,i in zip(batch,prompts,predictions,infos))
            prefix=args.name or args.model.split('/')[-1]
            (args.output/f'{prefix}_{mode}.jsonl').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
            macro,details=aggregate([score(r['sample'],r['predict']) for r in records])
            results[mode]=dict(article_macro_accuracy=macro,details=details)
        results.update(model=client.get_params(),total_cost_usd=client.cost)
        (args.output/f'{prefix}_summary.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    elif args.action=='calibrate':
        from dev.rulaw_proofbench.judge_calibration import build_calibration
        items=build_calibration();decisions=judge_records(client,items)
        details=[dict(**item,judge=decision,correct=decision['verdict']['equivalent']==item['expected_equivalent']) for item,decision in zip(items,decisions)]
        groups={k:[r for r in details if r['kind']==k] for k in {r['kind'] for r in details}}
        results=dict(count=len(details),accuracy=sum(r['correct'] for r in details)/len(details),
                     by_kind={k:dict(count=len(rs),accuracy=sum(r['correct'] for r in rs)/len(rs)) for k,rs in groups.items()},
                     provenance=provenance(client),samples=details,total_cost_usd=client.cost)
        (args.output/'judge_calibration_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    else:
        if not args.predictions:p.error('judge requires --predictions')
        raw=json.loads(args.predictions.read_text());items=[]
        if len(raw) != len(rows) or {row['sample']['id'] for row in raw} != expected.keys():
            raise ValueError(f'Complete unique {len(rows)}-record predictions required')
        for item in raw:
            row=expected[item['sample']['id']]
            if any(item['sample'][k]!=row[k] for k in ('question','gold','rule_sha256')):raise ValueError('Different prediction snapshot')
            items.append(dict(id=row['id'],question=row['question'],reference=row['gold'],prediction=item['predict'],strict=score(row,item['predict'])))
        decisions=judge_records(client,items);macro,details=aggregate_judgments(items,decisions)
        results=dict(article_macro_semantic_equivalence=macro,details=details,provenance=provenance(client),total_cost_usd=client.cost)
        (args.output/((args.name or 'judge')+'_judge.json')).write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in results.items() if k not in ('samples','details','provenance','model')},ensure_ascii=False))


if __name__=='__main__':main()
