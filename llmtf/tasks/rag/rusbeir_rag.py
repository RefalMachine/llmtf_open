from llmtf.base import SimpleFewShotHFTask
from typing import Any, Dict
from llmtf.metrics import metric_max_over_ground_truths, rougel, mean
import pandas as pd

def convert_context(context):
    segments = [f'**Сегмент №{i+1}**:\n' + c['chunk'] for i, c in enumerate(context)]
    if len(segments) == 0:
        return 'Поиск не вернул результатов (сегментов).'
    else:
        return '\n\n'.join(segments)
    
class RusbeirRag(SimpleFewShotHFTask):
    def __init__(self, instruction, dataset='bearberry/rubqqa', **kwargs):
        super().__init__(**kwargs)
        self.method = 'generate'
        self.dataset = dataset
        self.instruction = instruction
        self._max_task_new_tokens = 128

    def evaluate(self, sample, y_pred) -> Dict:
        rougel_metric = metric_max_over_ground_truths(lambda x, y: rougel(x, y).fmeasure, y_pred, sample['answers'])
        return {"rougel": rougel_metric}

    def task_name(self):
        return self.dataset

    def aggregation(self) -> Dict:
        return {"rougel": mean}

    def dataset_args(self) -> Dict:
        return {'path': self.dataset}

    def test_split_name(self) -> str:
        return 'train'

    def prompt_split_name(self) -> str:
        return 'train'

    def create_messages(self, sample, with_answer):
        messages = []
        
        answer = sample['answers'][0]
        question = sample['question'].strip()
        segments = convert_context(sample['context']).strip()
        
        instruction_user = self.instruction.replace('{question}', question).replace('{segments}', segments)

        messages.append({'role': 'user', 'content': instruction_user})
        if with_answer:
            messages.append({'role': 'bot', 'content': answer})

        return messages

    def prompt_dataset_start_idx(self) -> int:
        return 0
    
    def get_answer(self, sample):
        return sample['answer'].strip()

llm_judge_instruction_default = [
    {"role": "user", "content": "Твоя задача - оценить, является ли ответ модели семантически эквивалентным хотя бы одному из допустимых ответов. Ответ модели может быть более развернутым или содержать дополнительную информацию, но главное - он должен содержать корректный ответ на вопрос. НЕ сравнивай с тем, что ты считаешь правильным ответом - сравнивай ТОЛЬКО с предложенными допустимыми ответами. Отвечай только Yes или No.\n\nВопрос: В каком городе находится Эйфелева башня?\n\nОтвет модели: Париж\n\nДопустимые ответы:\n1) Москва\n2) Лондон\n3) Берлин"},
    {"role": "assistant", "content": "No"},
    {"role": "user", "content": "Вопрос: Какой тропический фрукт имеет желтую кожуру и растет гроздьями?\n\nОтвет модели: Это банан, тропический фрукт\n\nДопустимые ответы:\n1) банан\n2) Банан"},
    {"role": "assistant", "content": "Yes"},
    {"role": "user", "content": "Вопрос: Сколько будет 8 разделить на 16?\n\nОтвет модели: x=1/2\n\nДопустимые ответы:\n1) 0.5\n2) 4/8\n3) 1/2"},
    {"role": "assistant", "content": "Yes"},
    {"role": "user", "content": "Вопрос: Какой ответ был получен Васей при делении пять на два?\n\nОтвет модели: 2.5\n\nДопустимые ответы:\n1) 2\n2) 2.0\n3) два"},
    {"role": "assistant", "content": "No"},
    {"role": "user", "content": "Вопрос: В какой стране находится Статуя Свободы?\n\nОтвет модели: Ответ: США (Соединенные Штаты Америки)\n\nДопустимые ответы:\n1) Соединенные штаты америки\n2) Америка\n3) USA\n4) United States"},
    {"role": "assistant", "content": "Yes"},
    {"role": "user", "content": "Вопрос: Какая столица России?\n\nОтвет модели: Москва\n\nДопустимые ответы:\n1) Россия\n2) РФ\n3) Russia"},
    {"role": "assistant", "content": "No"},
    {"role": "user", "content": "Вопрос: В каком городе находится Центральный парк, он еще назывался Большое Яблоко?\n\nОтвет модели: NYC - это Нью-Йорк\n\nДопустимые ответы:\n1) Нью-Йорк\n2) New York\n3) New York City\n4) NY"},
    {"role": "assistant", "content": "Yes"},
    {"role": "user", "content": "Вопрос: Упоминается ли Иван Петров в данном тексте?\n\nОтвет модели: Нет, он не в примере\n\nДопустимые ответы:\n1) нет\n2) отсутствует\n3) Нет"},
    {"role": "assistant", "content": "Yes"},
    {"role": "user", "content": "{}"},
]

class RusbeirRagLLMJudge(RusbeirRag):
    ALLOW_BOOTSTRAPPING = False
    
    def __init__(self, model, llm_judge_instruction=llm_judge_instruction_default, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.llm_judge_instruction=llm_judge_instruction

    def evaluate(self, sample, y_pred) -> Dict:
        rougel_metric = metric_max_over_ground_truths(lambda x, y: rougel(x, y).fmeasure, y_pred, sample['answers'])
        return {
            "llm_judge_accuracy": [y_pred, sample['answers'], sample['question']],
            "rougel": rougel_metric
        }

    def llm_judge_accuracy_agg(self, llm_judge_data: Dict):
        messages_batch = []
        for y_pred, answers, question in llm_judge_data:
            answers_formatted = "\n".join([f"{i+1}) {ans}" for i, ans in enumerate(answers)])
            formatted_content = f"Вопрос: {question}\n\nОтвет модели: {y_pred}\n\nДопустимые ответы:\n{answers_formatted}"
            instruction = self.llm_judge_instruction[:-1] + [
                {
                    "role": self.llm_judge_instruction[-1]["role"],
                    "content": self.llm_judge_instruction[-1]["content"].format(formatted_content)
                }
            ]
            messages_batch.append(instruction)
        
        _, responses, _ = self.model.generate_batch(messages_batch, enable_thinking=False)

        scores = []
        details = []
        for i, (response, (y_pred, answers, question)) in enumerate(zip(responses, llm_judge_data)):
            if response[:3] == "Yes":
                score = 1
            elif response[:2] == "No":
                score = 0
            else:
                self.logger.warning(f"LLM judge generated unexpected response. Expected 'Yes' or 'No', got: '{response}'. Setting score to 0.")
                score = 0
            
            scores.append(score)
            details.append({
                'sample_idx': i,
                'question': question,
                'y_pred': y_pred,
                'answers': answers,
                'llm_response': response,
                'llm_score': score
            })
        
        return (mean(scores), details)

    def aggregation(self) -> Dict:
        return {"llm_judge_accuracy": self.llm_judge_accuracy_agg, "rougel": mean}

    def leaderboard_aggregation(self, metrics: Dict) -> float:
        return metrics["llm_judge_accuracy"]
