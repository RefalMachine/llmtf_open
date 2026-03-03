import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from datasets import load_dataset

from llmtf.model import ApiVLLMModel
from llmtf.tasks.ner.collection3 import Collection3Json, check_sample, process_sample

from prompt_optimizer.optimizable_task import SimpleOptimizableTask
from prompt_optimizer.evaluator import PseudoGradEval, SimpleFeedbackGenerator
from prompt_optimizer.sampler import PowerLawSampler
from prompt_optimizer.mutator import SimpleMutator
from prompt_optimizer.optimizer import Optimizer
from prompt_optimizer.utils import prompt_to_str


# Models

API_BASE = None
assert API_BASE is not None, "API_BASE is not set. Impossible to connect to api!"

mutator_model_name = "Qwen3-235B-A22B-Instruct-2507"
runner_model_names = [
    "ruadapt-qwen3-4b",
    "qwen3-4b-instruct",
    "tpro-2",
    "RefalMachine/RuadaptQwen2.5-7B-Lite-Beta",
    "Qwen3-32B",
    "RefalMachine/RuadaptQwen3-32B-Instruct-v2"
]

# Mutator model is used to come up with improvements to a prompt.
mutator_model = ApiVLLMModel(api_base=API_BASE)
mutator_model.from_pretrained(mutator_model_name)

# Runner models are used to evaluate suggested prompts on the benchmark.
runner_models = []
for runner_model_name in runner_model_names:
    model = ApiVLLMModel(api_base=API_BASE)
    model.from_pretrained(runner_model_name)
    runner_models.append(model)


# Task

class Collection3JsonOptimizable(Collection3Json, SimpleOptimizableTask):
    # Description of the task, fed to mutator model
    TASK_DESCRIPTION = """Ты — эксперт по извлечению именованных сущностей из русскоязычных текстов. Твоя задача — написать промпт для модели для поиска ВСЕХ именованных сущностей в тексте.
Если присутствует несколько одинаковых вхождений сущности, извлечь необходимо все в порядке встречаемости в тексте.
Сущности могут быть представлены целым словом или последовательностей слов, разделенных пробелами.
Оставь сущности в том виде, в каком они даны в тексте, не изменяй и не склоняй их.
Извлекай строго в порядке встречаемости сущностей в тексте.

# Классы сущностей
PER - человек (с именем).
ORG - организация.
LOC - местоположение.

# Формат вывода модели
Модель должна возвращать список списков в формате JSON: [["TYPE", "text span"], ...].

Текст для обработки в промпте обозначается строкой {text}."""
    
    def __init__(self):
        self.task_description = self.TASK_DESCRIPTION
        # super().__init__()
        Collection3Json.__init__(self)
        SimpleOptimizableTask.__init__(self)

    # Loads and processes train samples of the dataset
    def load_samples(self, max_sample_per_dataset=100000000):
        dataset = load_dataset(**self.dataset_args())
        train_dataset = dataset[self.prompt_split_name()].filter(check_sample)
        test_dataset = dataset[self.test_split_name()].filter(check_sample)
        
        train_dataset = train_dataset.select(
            list(range(min(max_sample_per_dataset, len(dataset[self.prompt_split_name()])))))
        
        train_dataset = train_dataset.map(process_sample)
        test_dataset = test_dataset.map(process_sample)
        return (train_dataset, test_dataset)

    # Fills prompt template with information from a sample
    def fill_prompt(self, prompt, sample, with_answer):
        user_prompt = prompt[0]["content"].replace("{text}", sample["query"])
        if with_answer:
            assistant_prompt = prompt[1]["content"].replace("{answer}", self.get_answer_str(sample))
        else:
            assistant_prompt = prompt[1]["content"].replace("{answer}", "")

        messages = [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_prompt}
        ]
        return messages

task = Collection3JsonOptimizable()

# Initial prompt for runner models
INITIAL_PROMPT_USER = """Извлеки из заданного ниже текста все именованные сущности всех представленных ниже классов.
Сущности могут быть представлены целым словом или последовательностей слов, разделенных пробелами.
Оставь сущности в том виде, в каком они даны в тексте, не изменяй и не склоняй их.

**Классы**
PER - человек (с именем).
ORG - организация.
LOC - местоположение.

**Формат вывода (только json)**
```json
[["класс", "сущность"], ["класс", "сущность"], ...]
```

**Текст**
{text}"""

initial_prompt = [
    {"role": "user", "content": INITIAL_PROMPT_USER},
    {"role": "assistant", "content": "{answer}"}
]


# Feedback generator

# Let's create a custom feedback generator for our task
FEEDBACK_PROMPT = """Текст:
\"\"\"
{query}
\"\"\"
Сгенерированный ответ:
\"\"\"
{y_pred}
\"\"\"
Правильный ответ:
\"\"\"
{y_gold}
\"\"\"
"""

class ShortFeedbackGenerator(SimpleFeedbackGenerator):
    def __init__(self):
        self.feedback_prompt = FEEDBACK_PROMPT
    
    def get_feedback(self, prompt, sample, y_pred):
        y_gold = self.task.get_answer(sample)
        prompt = prompt_to_str(prompt)
        return self.feedback_prompt.format(query=sample["query"], y_pred=y_pred, y_gold=y_gold)


# Evaluator

# Evaluates benchmark and generates score and feedback of the
# prompt, based on 10 worst samples.
worst_samples = 10
evaluator = PseudoGradEval(task, worst_samples, ShortFeedbackGenerator())


# Sampler

# Samples solution from the solution database based on power law.
sampler = PowerLawSampler(alpha=2)


# Mutator

# Mutates prompt based on a feedback
mutator = SimpleMutator(
    mutator_model,
    max_new_tokens=2000
)


# Optimizer

# Runs optimization process
optimizer = Optimizer(
    task,
    runner_models,
    sampler,
    evaluator,
    mutator,
    model_names=runner_model_names,
    batch_size=100,
    max_sample_per_dataset=10,
    few_shot_count=0
)

# Run initial solution
root_solution = optimizer.run_root_solution(initial_prompt)

# Run optimization
best_solution = optimizer.run(max_iterations=1)

# Evaluate on test
root_solution_test = optimizer.run_prompt(initial_prompt, None, "test", "test")
best_solution_test = optimizer.run_prompt(best_solution.prompt, None, "test", "test")

print(f"""=== RESULTS ===

TRAIN IMPROVEMENT: {best_solution.score - root_solution.score}
TEST  IMPROVEMENT: {best_solution_test.score - root_solution_test.score}

= INITIAL SOLUTION =
> TRAIN SCORE: {root_solution.score}
> TEST  SCORE: {root_solution_test.score}
> PROMPT: \n{prompt_to_str(root_solution.prompt)}


= BEST SOLUTION =
> TRAIN SCORE: {best_solution.score}
> TEST  SCORE: {best_solution_test.score}
> PROMPT: \n{prompt_to_str(best_solution.prompt)}
""")
