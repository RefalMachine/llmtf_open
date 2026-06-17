from abc import abstractmethod
from datasets import load_dataset
from PIL import Image
import base64
import io
from llmtf.base import SimpleFewShotHFTask
from llmtf.metrics import mean, rougel

def pil_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

class RussianOcrSmall(SimpleFewShotHFTask):
    DATASET_PATH = "Foximaz/russian_ocr_small"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.method = "generate"

    def prompt_split_name(self):
        return 'test'
    
    def test_split_name(self):
        return 'test'
    
    def create_messages(self, sample, with_answer):
        image = pil_to_base64(sample["image"])
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": self.prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image}"}
                    }
                ]
            }
        ]
        if with_answer:
            messages.append({"role": "assistant", "content": sample["text"]})
        return messages


CAR_PLATE_OCR_INSTRUCTION = """\
Распознай номер автомобиля на этом изображении.
Номер состоит из английских букв и цифр и не содержит пробелов.
Верни только текст номера, без лишних слов и пояснений.\
"""

class CarPlateOcr(RussianOcrSmall):
    DATASET_NAME = "car_plate"

    def task_name(self):
        return "Foximaz/russian_ocr_small/car_plate"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._max_task_new_tokens = 10
        self.prompt = CAR_PLATE_OCR_INSTRUCTION

    def dataset_args(self):
        return {"path": self.DATASET_PATH, "name": self.DATASET_NAME}
    
    def evaluate(self, sample, gen_pred):
        # Exact match
        return {"acc": sample["text"].strip() == gen_pred.strip()}

    def aggregation(self):
        return {"acc": mean}


HANDWRITING_OCR_INSTRUCTION = """\
Распознай рукописный текст на русском языке на изображении.
Выведи только распознанный текст, без пояснений.\
"""

class HandwritingOcr(RussianOcrSmall):
    DATASET_NAME = "handwriting"

    def task_name(self):
        return "Foximaz/russian_ocr_small/handwriting"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._max_task_new_tokens = 50
        self.prompt = HANDWRITING_OCR_INSTRUCTION

    def dataset_args(self):
        return {"path": self.DATASET_PATH, "name": self.DATASET_NAME}
    
    def evaluate(self, sample, gen_pred):
        return {"rougel": rougel(sample["text"].strip(), gen_pred.strip()).fmeasure}

    def aggregation(self):
        return {"rougel": mean}


PRINTED_OCR_INSTRUCTION = """\
Распознай текст на русском языке на изображении.
Выведи только распознанный текст, без пояснений.\
"""

class PrintedOcr(RussianOcrSmall):
    DATASET_NAME = "printed"

    def task_name(self):
        return "Foximaz/russian_ocr_small/printed"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._max_task_new_tokens = 750
        self.prompt = PRINTED_OCR_INSTRUCTION

    def dataset_args(self):
        return {"path": self.DATASET_PATH, "name": self.DATASET_NAME}
    
    def evaluate(self, sample, gen_pred):
        return {"rougel": rougel(sample["text"].strip(), gen_pred.strip()).fmeasure}

    def aggregation(self):
        return {"rougel": mean}
