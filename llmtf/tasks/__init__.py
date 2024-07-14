from . import (
    darumeru,
    nlpcoreteam,
    rucola,
    shlepa,
    daru_treeway_summ
)

########################################
# All tasks
########################################


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
    #'shlepa/moviesmc': {'class': shlepa.ShlepaSmallMMLU, 'params': {'dataset_name': 'Vikhrmodels/movie_mc'}},
    #'shlepa/musicmc': {'class': shlepa.ShlepaSmallMMLU, 'params': {'dataset_name': 'Vikhrmodels/music_mc'}},
    #'shlepa/lawmc': {'class': shlepa.ShlepaSmallMMLU, 'params': {'dataset_name': 'Vikhrmodels/law_mc'}},
    #'shlepa/booksmc': {'class': shlepa.ShlepaSmallMMLU, 'params': {'dataset_name': 'Vikhrmodels/books_mc'}}
    'daru/treewayabstractive': {'class': daru_treeway_summ.DaruTreewayAbstractive},
    'daru/treewayextractive': {'class': daru_treeway_summ.DaruTreewayExtractive},
    'darumeru/cp_sent_ru': {'class': darumeru.CopyText, 'params': {'subtask': 'sent', 'lang': 'ru'}},
    'darumeru/cp_sent_en': {'class': darumeru.CopyText, 'params': {'subtask': 'sent', 'lang': 'en'}},
    'darumeru/cp_para_ru': {'class': darumeru.CopyText, 'params': {'subtask': 'para', 'lang': 'ru'}},
    'darumeru/cp_para_en': {'class': darumeru.CopyText, 'params': {'subtask': 'para', 'lang': 'en'}}
}