from . import (
    darumeru,
    nlpcoreteam,
    rucola,
    shlepa,
    daru_treeway_summ,
    habrqa,
    ruopinionne,
    ruparam,
    nerel
)

########################################
# All tasks
########################################

ruopinionne_default_instruction = '''Твоя задача состоит в том, чтобы проанализировать текст и извлечь из него выражения мнений, представленные в виде кортежа мнений, состоящих из 4 основных составляющих:
1. Источник мнения: автор текста/именованная сущность текста/именованная сущность, не совпадающая с автором текста, но не являющаяся подстрокой исходного текста (AUTHOR/подстрока исходного текста/NULL). Key = Source;
2. Объект мнения: именованная сущность в тексте (неразрывная подстрока исходного текста). Key = Target;
3. Тональность: положительная/негативная (POS/NEG). Key = Polarity; Допустимые значения: [POS;NEG];
4. Языковое выражение: аргумент, на основании которого принята результирующая тональность (неразрывная подстрока исходного текста). Key = Expression;

Если источник мнения отсутствует, то Source = NULL, если автор является источником мнения, то AUTHOR. Ответ необходимо представить в виде json списка, каждый элемент которого является кортежем мнений. Каждый кортеж мнений это словарь, состоящий из четырех значений: Source, Target, Polarity, Expression.
Извлеченная информация обязана быть **полной копией подстроки из текста**: без разрывов и без каких либо изменений.

***Пример***
Текст: 
Президент США подчеркнул, что крайне заинтересован в мирном разрешении кризиса

Ответ: 
***json***
[{"Source": "Президент США", "Target": "мирном разрешении кризиса", 'Polarity': "POS", "Expression": "крайне заинтересован"}]

***Текст***
{text}'''
ruopinionne_default_instruction_short = '''***Текст***\n{text}'''
ruparam_default_instruction = 'Какое из двух предложений является правильным и грамматичным с точки зрения русского языка?\nПредложение 1. {sent_lhs}\nПредложение 2. {sent_rhs}\nОтветь только одной цифрой 1 или 2, ничего не добавляя.'
TASK_REGISTRY = {
    'darumeru/multiq': {'class': darumeru.MultiQ},
    'darumeru/parus': {'class': darumeru.PARus},
    'darumeru/rcb': {'class': darumeru.RCB},
    'darumeru/rummlu': {'class': darumeru.ruMMLU},
    'darumeru/ruopenbookqa': {'class': darumeru.ruOpenBookQA},
    #'darumeru/rutie': {'class': darumeru.ruTiE},
    'darumeru/ruworldtree': {'class': darumeru.ruWorldTree},
    'darumeru/rwsd': {'class': darumeru.RWSD},
    'darumeru/use': {'class': darumeru.USE},
    'nlpcoreteam/rummlu': {'class': nlpcoreteam.ruMMLU},
    'nlpcoreteam/enmmlu': {'class': nlpcoreteam.enMMLU},
    'russiannlp/rucola_custom': {'class': rucola.RuColaCustomTask},
    'shlepa/moviesmc': {'class': shlepa.ShlepaSmallMMLU, 'params': {'dataset_name': 'Vikhrmodels/movie_mc'}},
    'shlepa/musicmc': {'class': shlepa.ShlepaSmallMMLU, 'params': {'dataset_name': 'Vikhrmodels/music_mc'}},
    'shlepa/lawmc': {'class': shlepa.ShlepaSmallMMLU, 'params': {'dataset_name': 'Vikhrmodels/law_mc'}},
    'shlepa/booksmc': {'class': shlepa.ShlepaSmallMMLU, 'params': {'dataset_name': 'Vikhrmodels/books_mc'}},
    'daru/treewayabstractive': {'class': daru_treeway_summ.DaruTreewayAbstractive},
    'daru/treewayextractive': {'class': daru_treeway_summ.DaruTreewayExtractive},
    'darumeru/cp_sent_ru': {'class': darumeru.CopyText, 'params': {'subtask': 'sent', 'lang': 'ru'}},
    'darumeru/cp_sent_en': {'class': darumeru.CopyText, 'params': {'subtask': 'sent', 'lang': 'en'}},
    'darumeru/cp_para_ru': {'class': darumeru.CopyText, 'params': {'subtask': 'para', 'lang': 'ru'}},
    'darumeru/cp_para_en': {'class': darumeru.CopyText, 'params': {'subtask': 'para', 'lang': 'en'}},
    'darumeru/ruscibench_grnti_ru': {'class': darumeru.ruSciBenchGRNTIRu},
    'vikhrmodels/habr_qa_sbs': {'class': habrqa.HabrQASbS},
    'ruopinionne': {'class': ruopinionne.RuOpinionNE, 'params': {'instruction': ruopinionne_default_instruction, 'short_instruction': ruopinionne_default_instruction_short}},
    'ruparam': {'class': ruparam.RuParam, 'params': {'instruction': ruparam_default_instruction}},
    'nerel': {'class': nerel.NestedNER} 
}