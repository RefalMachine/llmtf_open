from . import (
    darumeru,
    nlpcoreteam,
    rucola
)

########################################
# All tasks
########################################


TASK_REGISTRY = {
    'darumeru/multiq': darumeru.MultiQ,
    'darumeru/parus': darumeru.PARus,
    'darumeru/rcb': darumeru.RCB,
    'darumeru/rummlu': darumeru.ruMMLU,
    'darumeru/ruopenbookqa': darumeru.ruOpenBookQA,
    'darumeru/rutie': darumeru.ruTiE,
    'darumeru/ruworldtree': darumeru.ruWorldTree,
    'darumeru/rwsd': darumeru.RWSD,
    'darumeru/use': darumeru.USE,
    'nlpcoreteam/rummlu': nlpcoreteam.ruMMLU,
    'nlpcoreteam/enmmlu': nlpcoreteam.enMMLU,
    'russiannlp/rucola_custom': rucola.RuColaCustomTask
}