"""Versioned corrected scoring alongside the pinned reference; no model dependencies."""
import json
import math
import re
from collections import Counter, defaultdict
from . import reference as ref


def _unique_object(pairs):
    obj = {}
    for k, v in pairs:
        if k in obj:
            raise ValueError('duplicate_key')
        obj[k] = v
    return obj


def _invalid_constant(value):
    raise ValueError('nonfinite_number')


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("nonfinite_number")
    return number


def parse_call(output):
    text = output.strip()
    if text.startswith('```'):
        m = re.fullmatch(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.S)
        if not m:
            return None, 'invalid_fence'
        text = m.group(1).strip()
    try:
        obj = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_invalid_constant, parse_float=_finite_float)
    except (ValueError, RecursionError):
        return None, 'invalid_json'
    if not isinstance(obj, dict) or 'tool' not in obj:
        return None, 'invalid_object'
    if obj['tool'] is not None and not isinstance(obj['tool'], str):
        return None, 'invalid_tool_type'
    if 'args' in obj and not isinstance(obj['args'], dict):
        return None, 'invalid_args_type'
    if obj['tool'] is not None and (not obj['tool'] or 'args' not in obj):
        return None, 'missing_tool_or_args'
    if set(obj) - {'tool', 'args'}:
        return None, 'unexpected_fields'
    if obj['tool'] is None and obj.get('args'):
        return None, 'refusal_with_args'
    return obj, 'ok'


def reference_score(row, output):
    if row['answer_type'] == 'tool_call':
        obj = ref._extract_json_obj(output)
        if obj is not None and ((obj.get('tool') is not None and not isinstance(obj['tool'], str)) or ('args' in obj and not isinstance(obj['args'], dict))):
            return 0.0, 'invalid_field_type'
    return ref.SCORERS[row['answer_type']](row, output), 'reference'


def extract_norms(text):
    """Clause-local code/article association, supporting both writing orders.

    Competing acts between two articles use the closest article; unresolved
    clauses are reported, rather than assigned a convenient act.
    """
    pairs, ambiguous = set(), False
    for clause in re.split(r'[;\n]+', ref.norm(text)):
        # Match long aliases first, avoid acronym substrings inside words.
        aliases = sorted(ref._CODE_ALIASES, key=lambda x: -len(x[0]))
        pattern = '|'.join(r'(?<!\w)' + re.escape(a) + (r'(?!\w)' if len(a) <= 5 else r'\w*') for a, _ in aliases)
        codes = []
        for m in re.finditer(pattern, clause):
            canon = next(c for a, c in aliases if m.group().startswith(a))
            codes.append((m.start(), m.end(), canon))
        for m in ref._FZ_RE.finditer(clause):
            codes.append((m.start(), m.end(), 'ФЗ-' + (m.group(1) or m.group(2))))
        codes.sort()
        articles = list(re.finditer(r'(?:ст\.?|стать[ияею])\s*№?\s*(\d+(?:\.\d+)?(?:\s*[,и]\s*\d+(?:\.\d+)?)*)', clause))
        prefix_order = bool(codes and articles and codes[0][0] < articles[0].start()
                            and codes[-1][1] <= articles[-1].start())
        suffix_order = bool(codes and articles and articles[0].end() <= codes[0][0]
                            and articles[-1].end() <= codes[-1][0])
        for i, article in enumerate(articles):
            if prefix_order or suffix_order:
                candidates = [c for c in codes if c[1] <= article.start()] if prefix_order else [c for c in codes if c[0] >= article.end()]
                if candidates:
                    code = candidates[-1][2] if prefix_order else candidates[0][2]
                    pairs.update((code, n) for n in re.findall(r'\d+(?:\.\d+)?', article.group(1)))
                    continue
            left = [c for c in codes if c[1] <= article.start() and (i == 0 or c[0] >= articles[i-1].end())]
            right = [c for c in codes if c[0] >= article.end() and (i+1 == len(articles) or c[0] < articles[i+1].start())]
            if left and right and left[-1][2] != right[0][2]:
                # A code immediately after a prior article belongs to it.
                if i and left[-1][0] - articles[i-1].end() < article.start() - left[-1][1]:
                    left = []
                else:
                    ambiguous = True
                    continue
            code = right[0][2] if right else left[-1][2] if left else codes[0][2] if len({c[2] for c in codes}) == 1 else None
            if code is None:
                ambiguous = True
            else:
                pairs.update((code, n) for n in re.findall(r'\d+(?:\.\d+)?', article.group(1)))
    return pairs, ambiguous


def extraction_match(gold, output):
    """Preserve lexical answers/identifiers; require numeric token boundaries.

    No numeric conversion, unit conversion, inferred synonyms or dropped zeros.
    Numeric punctuation and sign are part of the matched value.
    """
    gold, output = ref.norm(gold), ref.norm(output)
    if not gold:
        return False
    if re.search(r'\d', gold):
        return re.search(r'(?<![\w.,+−-])' + re.escape(gold) + r'(?![\w]|[.,]\d)', output) is not None
    return gold in output


def _typed_equal(a, b):
    # JSON types are preserved; boolean != number, null != missing. Strings
    # (including dates, identifiers and queries) are case-sensitive, arrays ordered.
    return json.dumps(a, sort_keys=True, ensure_ascii=False) == json.dumps(b, sort_keys=True, ensure_ascii=False)


def score(row, output, catalog=()):
    if not isinstance(output, str):
        raise TypeError('LegalBench-RU expects a single text continuation')
    kind = row['answer_type']
    reference, reference_status = reference_score(row, output)
    result = {k: row[k] for k in ('task', 'id', 'track', 'domain', 'answer_type')}
    result.update(reference_raw=reference, reference_score=round(reference, 3), reference_status=reference_status, parse_status='ok', needs_expert_review=row.get('needs_expert_review', True))
    value = 0.0
    if kind == 'tool_call':
        obj, status = parse_call(output)
        result['parse_status'] = status
        gold = row['answer']; positive = gold['tool'] is not None
        routing = obj is not None and obj['tool'] == gold['tool']
        args = obj.get('args', {}) if obj else {}
        ga = gold.get('args', {})
        recall = sum(k in args and _typed_equal(args[k], v) for k, v in ga.items()) / len(ga) if ga else 1.0
        known = {x['server'] + '.' + x['tool'] for x in catalog}
        result.update(routing_accuracy=float(routing), gold_call_exact=float(routing and _typed_equal(args, ga)), gold_args_recall=recall if routing and positive else None, positive_tool=positive, unannotated_args=len(set(args)-set(ga)), predicted_args_count=len(args), unknown_tool=bool(obj and obj['tool'] is not None and obj['tool'] not in known))
        value = (0.6 + 0.4 * recall if positive else 1.0) if routing else 0.0
    elif kind in ('binary', 'multiple_choice'):
        if kind == 'binary':
            labels = set(re.findall(r'\b(да|нет)\b', ref.norm(output)))
            gold = ref.norm(row['answer'])
        else:
            labels = set(re.findall(r'\b([ABCD])\b', output.upper()))
            if not labels:
                labels = {'ABCD'[i] for i, c in enumerate(row.get('choices', [])) if ref.norm(c) and ref.norm(c) in ref.norm(output)}
            gold = row['answer']
        prediction = next(iter(labels)) if len(labels) == 1 else None
        result.update(gold_label=gold, prediction=prediction)
        result['parse_status'] = 'ambiguous' if len(labels)>1 else 'ok' if labels else 'missing_label'
        value = float(prediction == gold)
    elif kind == 'norm_citation':
        gold = set().union(*(extract_norms(g)[0] for g in row['answer']))
        pred, ambiguous = extract_norms(output)
        value = 2 * len(gold & pred) / (len(gold) + len(pred)) if gold else 0.0
        result.update(citation_exact=float(gold == pred and bool(gold)), parse_status='ambiguous' if ambiguous else 'ok' if pred else 'missing_citation')
    elif kind == 'extraction':
        value = float(any(extraction_match(g, output) for g in [row['answer'], *row.get('accept', [])]))
        if not output.strip():result['parse_status'] = 'empty'
    else:
        raise ValueError(f'Unsupported answer type {kind}')
    result['score'] = value
    return result


def aggregate(records, primary='corrected'):
    if not records:
        raise ValueError('Cannot aggregate empty LegalBench-RU results')
    keys = [(r['task'], r['id']) for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate composite keys in results')
    mean = lambda xs: sum(xs)/len(xs) if xs else None
    details = {'count':len(records), 'selected_keys':keys, 'corrected_score':mean([r['score'] for r in records]), 'reference_score':mean([r['reference_score'] for r in records]), 'parse_status_counts':dict(Counter(r['parse_status'] for r in records)), 'review_required_count':sum(r['needs_expert_review'] for r in records), 'scorer_delta_counts':dict(Counter('increased' if r['score']>r['reference_score'] else 'decreased' if r['score']<r['reference_score'] else 'unchanged' for r in records))}
    for field in ('task', 'track', 'domain', 'answer_type'):
        groups = defaultdict(list)
        for r in records:groups[r[field]].append(r)
        details['by_'+field] = {g:{'count':len(rs), 'score':mean([r['score'] for r in rs]), 'reference_score':mean([r['reference_score'] for r in rs])} for g,rs in sorted(groups.items())}
    binary = [r for r in records if r['answer_type']=='binary']
    classes = {c:[r['score'] for r in binary if r['gold_label']==c] for c in ('да','нет')}
    details['binary'] = {'class_counts':{c:len(v) for c,v in classes.items()},'balanced_accuracy':mean([mean(v) for v in classes.values()]) if all(classes.values()) else None}
    tools = [r for r in records if r['answer_type']=='tool_call']
    details['tools'] = {'count':len(tools)}
    for field in ('routing_accuracy','gold_call_exact','gold_args_recall','unknown_tool'):
        values = [r[field] for r in tools if r[field] is not None]
        details['tools'][field] = {'mean':mean(values),'count':len(values)}
    for positive, name in ((True,'positive_routing'),(False,'negative_refusal')):
        values=[r['routing_accuracy'] for r in tools if r['positive_tool']==positive]
        details['tools'][name]={'mean':mean(values),'count':len(values)}
    vals=[details['tools'][n]['mean'] for n in ('positive_routing','negative_refusal')]
    details['tools']['balanced_routing']=mean(vals) if all(v is not None for v in vals) else None
    return details['reference_score' if primary=='reference' else 'corrected_score'], details
