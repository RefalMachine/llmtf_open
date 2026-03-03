import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from typing import List
from abc import ABC, abstractmethod
import logging
import re

from llmtf.base import LLM

from .solution_database import Solution
from .utils import Message, prompt_to_str

logger = logging.getLogger(__name__)

class Mutator(ABC):
    @abstractmethod
    def mutate(self, parent: Solution) -> List[Message]:
        pass


# Помимо этого тебе будет дан отзыв на промпт - набор триплетов из промпта, сгенерированного ответа и правильного ответа.

DEFAULT_MUTATE_PROMPT_RU = """Ты профессиональный промпт инженер. Ниже дано описание задачи для LLM, \
а также базовый промпт, который тебе будет необходимо усовершенствовать, чтобы повысить качество ответа LLM на эту задачу.
Помимо этого тебе будет дан отзыв на промпт - набор триплетов из текста для обработки, сгенерированного ответа и правильного ответа.

Внимательно прочитай описание задачи и базовый промпт.
Проанализируй отзыв и отметь допущенные моделью ошибки.
Наконец, следуя указанному ниже формату, напиши новый, усовершенствованный промпт.

# Описание задачи
\"\"\"
{task}
\"\"\"

# Состав промпта
Промпт состоит из двух частей - `user` и `assistant`. `assistant` часть является началом ответа агента. \
Она должна содержать в конце строку "{answer}" \
(допускается, что остальная `assistant` часть может быть пустой).
Вместо текста в фигурных скобках будет автоматически подставляться соответствующая информация.

# Базовый промпт
\"\"\"
{prompt}
\"\"\"

# Отзыв на промпт
{feedback}

# Формат
После размышлений напиши улучшенный промпт ниже, строго следуя следующему формату:

[user]:
Пример промта пользователя.

[assistant]:
Пример промпта ассистента {answer}"""

DEFAULT_WRONG_FORMAT_PROMPT_RU = """Некорректный формат вывода! Попробуй ещё раз.
# Формат

[user]:
Пример промта пользователя.

[assistant]:
Пример промпта ассистента {answer}
"""


# [!] outdated:
DEFAULT_MUTATE_PROMPT_EN = """You are a professional prompt engineer tasked with improving a prompt for a LLM.
Prompt consists of two parts - `user` and `assistant`. `assistant` part is a beginning of agent's response. \
It must end with string {answer} \
(it is allowed that the rest of the `assistant` part may be empty).
Text in curly braces will be automatically replaced with corresponding information.

# Task description
\"\"\"
{task}
\"\"\"

# Prompt to improve
\"\"\"
{prompt}
\"\"\"

# Feedback on prompt
{feedback}

# Format
Please, write improved prompt below in the following format:

[user]:
User prompt example.

[assistant]:
Assistant part example {answer}

Do not write anything else!"""

DEFAULT_WRONG_FORMAT_PROMPT_EN = """Incorrect output format! Try again.
# Format

[user]:
User prompt example.

[assistant]:
Assistant part example {answer}
"""

LANG_TO_PROMPTS = {
    "ru": (DEFAULT_MUTATE_PROMPT_RU, DEFAULT_WRONG_FORMAT_PROMPT_RU),
    "en": (DEFAULT_MUTATE_PROMPT_EN, DEFAULT_WRONG_FORMAT_PROMPT_EN)
}


class SimpleMutator(Mutator):
    def __init__(
        self,
        model: LLM,
        lang: str="ru",
        max_retry=5,
        max_new_tokens=512
    ):
        self.model = model
        self.model.generation_config.max_new_tokens = max_new_tokens
        self.mutate_prompt, self.wrong_format_prompt = LANG_TO_PROMPTS[lang]
        self.max_retry = max_retry

    def set_task_description(self, task_description):
        self.mutate_prompt = self.mutate_prompt.replace("{task}", task_description)

    def mutate(self, parent):
        messages = [{
            "role": "user",
            "content": self.mutate_prompt\
                .replace("prompt", prompt_to_str(parent.prompt))\
                .replace("feedback", parent.feedback)
        }]

        for _ in range(self.max_retry):
            prompt, y_pred, info = self.model.generate(messages)
    
            match = re.search(r"\[user\]:\s?(.*)\n\[assistant\]:\s?(.*)", y_pred, flags=re.DOTALL)
            if match and len(match.groups()) == 2:
                user_prompt = match.group(1)
                assistant_prompt = match.group(2)

                new_prompt = [{"role": "user", "content": user_prompt}]
                if assistant_prompt:
                    new_prompt.append({"role": "assistant", "content": assistant_prompt})
                break

            messages += [
                {"role": "assistant", "content": y_pred},
                {"role": "user", "content": self.wrong_format_prompt}
            ]
        else:
            logger.error(f"Mutator failed to follow output format! Messages:\n{messages}")
            raise Exception(f"Mutator failed to follow output format! Messages:\n{messages}")
        return new_prompt
