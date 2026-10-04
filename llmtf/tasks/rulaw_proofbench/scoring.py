"""Normalized exact match under a small, model-independent answer grammar.

No sentence aliases, answer extraction, article-specific rules or fuzzy matching.
See dev/rulaw_proofbench/SCORING.md for the complete accepted grammar.
"""
from collections import defaultdict
from fractions import Fraction
import re
import unicodedata

UNKNOWN = 'недостаточно данных'
SCORER_NAME = 'normalized_exact_match'
CATEGORIES = frozenset({'да', 'нет', 'ничтожна', 'до отмены'})
SPECIAL = frozenset({UNKNOWN, 'не следует из указанной нормы'})
UNITS = {
    'days': r'(?:день|дня|дней)',
    'calendar_days': r'(?:календарный день|календарных дня|календарных дней|день|дня|дней)',
    'hours': r'(?:час|часа|часов)',
    'hours_per_week': r'(?:час|часа|часов)(?: в неделю)?',
}
NUMBER = r'(?:[0-9]+(?:\.[0-9]+)?|[0-9]+\s*/\s*[0-9]+)'


def normalize_text(text):
    if not isinstance(text, str):
        return None
    # Reject adjoining mixed vulgar fractions: Unicode normalization would turn
    # e.g. a whole number immediately followed by ½ into a different fraction.
    if re.search(r'[0-9][¼½¾⅐⅑⅒⅓⅔⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞]', text):
        return None
    value = unicodedata.normalize('NFKC', text).casefold().replace('ё', 'е').replace('⁄', '/')
    value = ' '.join(value.split()).strip(' «»"\'.')
    return value


def _number(value):
    if re.fullmatch(NUMBER, value) is None:
        return None
    try:
        return Fraction(value.replace(' ', ''))
    except (ValueError, ZeroDivisionError):
        return None


def _decimal(value):
    # Exact finite decimal, without floating-point tolerance or rounding.
    denominator = value.denominator
    twos = fives = 0
    while denominator % 2 == 0:
        denominator //= 2
        twos += 1
    while denominator % 5 == 0:
        denominator //= 5
        fives += 1
    if denominator != 1:
        return str(value)
    places = max(twos, fives)
    scaled = value.numerator * 2 ** (places - twos) * 5 ** (places - fives)
    if not places:
        return str(scaled)
    digits = str(scaled).zfill(places + 1)
    return (digits[:-places] + '.' + digits[-places:]).rstrip('0').rstrip('.')


def parse_answer(text, unit):
    value = normalize_text(text)
    if not value:
        return None
    if value in SPECIAL:
        return value
    if unit == 'none':
        if value in CATEGORIES:
            return value
        # A singular unqualified unit denotes one unit; no prose is extracted.
        if value == 'год':
            return '1 год'
        match = re.fullmatch(r'(.+?)\s*год', value)
        if match and _number(match[1].replace(',', '.')) == 1:
            return '1 год'
        return None
    if unit == 'fraction':
        percent = value.endswith('%')
        number = _number((value[:-1].strip() if percent else value).replace(',', '.'))
        return str(number / 100 if percent else number) if number is not None else None
    if unit in UNITS:
        value = re.sub(r'\s*' + UNITS[unit] + r'$', '', value).replace(',', '.')
        number = _number(value)
        return _decimal(number) if number is not None else None
    return None


def score(sample,prediction):
    parsed=parse_answer(prediction,sample['gold']['unit'])
    return dict(id=sample['id'],article_key=sample['article_key'],question_type=sample['question_type'],
                gold=sample['gold']['value'],prediction=parsed,correct=parsed==sample['gold']['value'],
                format_valid=parsed is not None,pair_ids=sample['pair_ids'],
                unknown_but_determinate=sample['unknown_but_determinate'])


def aggregate(records):
    if not records:raise ValueError('Empty RuLaw-ProofBench evaluation')
    if len({r['id'] for r in records})!=len(records):raise ValueError('Duplicate RuLaw-ProofBench result IDs')
    groups=defaultdict(list); types=defaultdict(list); pairs=defaultdict(list)
    for r in records:
        groups[r['article_key']].append(r);types[r['question_type']].append(r)
        for pid in r['pair_ids']:pairs[pid].append(r)
    acc=lambda rs:sum(r['correct'] for r in rs)/len(rs)
    by_article={a:dict(count=len(rs),accuracy=acc(rs)) for a,rs in sorted(groups.items())}
    macro=sum(x['accuracy'] for x in by_article.values())/len(by_article)
    true_unknown=sum(r['gold']==UNKNOWN for r in records)
    predicted_unknown=sum(r['prediction']==UNKNOWN for r in records)
    correct_unknown=sum(r['gold']==UNKNOWN and r['prediction']==UNKNOWN for r in records)
    complete=[rs for rs in pairs.values() if len(rs)==2]
    invariant=[r for r in records if r['unknown_but_determinate']]
    return macro,dict(primary='article_macro_accuracy',articles=by_article,micro_accuracy=acc(records),
        by_type={t:dict(count=len(rs),accuracy=acc(rs)) for t,rs in types.items()},
        format_valid_rate=sum(r['format_valid'] for r in records)/len(records),
        insufficient=dict(gold_count=true_unknown,predicted_count=predicted_unknown,correct=correct_unknown,
                          precision=correct_unknown/predicted_unknown if predicted_unknown else None,
                          recall=correct_unknown/true_unknown if true_unknown else None),
        unknown_but_determinate=dict(count=len(invariant),accuracy=acc(invariant) if invariant else None),
        minimal_pairs=dict(complete=len(complete),partial=sum(len(rs)==1 for rs in pairs.values()),
                           both_correct=sum(all(r['correct'] for r in rs) for rs in complete)/len(complete) if complete else None),
        evaluated_count=len(records))
