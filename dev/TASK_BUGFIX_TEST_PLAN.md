# Task layer bugfix test plan

Статус документа: **targeted validation выполнена; полная матрица не запускалась**.

Дата ревизии: 2026-09-21.

## 1. Dependency-free gate

После каждого логического блока и в финале выполнить:

```bash
python3 tests/test_refactor_logic.py
python3 -m unittest discover -s tests -p 'test_*.py' -v
python3 -m compileall -q llmtf evaluate_model.py evaluate_model_api.py benchmark show_results.py dev/tools examples
git diff --check
```

Если установлен pytest:

```bash
python3 -m pytest tests -q
```

## 2. Обязательные regression cases

- RuOpinionNE parser не исполняет Python expressions и возвращает пустой список
  для scalar/invalid payload.
- RuParam создаёт canonical `user`/`assistant` messages, нормализует `order=i`,
  не группирует разные строки по неуникальному исходному `id`, требует
  zero-shot и сохраняет category/part/source/level aggregation details.
- Shlepa сохраняет sample с правильным вариантом `A`.
- Oversized zero-shot prompt выдаёт понятную ошибку; неизвестный API token count
  не интерпретируется как ноль.
- `calculate_logsoftmax` task проходит normal evaluator dispatch на HF-like fake
  backend и отклоняется unsupported backend-ом.
- PPL boundary использует character offsets относительно exact rendered prefix
  и не включает assistant prefill.
- Пустые RuBLiMP/IFEval outputs дают нулевую метрику, а не исключение.
- NER in-place отклоняет усечённую и дополненную копию текста.
- MMLU/RuBLiMP quota helpers никогда не превышают raw sample limit.
- Dataset/messages и backend outputs остаются выровнены.
- Изменение task params или source digest меняет fingerprint.
- Lazy API observations `logprobs`/`detokenize` не меняют fingerprint, а
  старый schema-v2 artifact принимается только при совпадении остальных полей.
- Повторная регистрация существующего task name требует явного override.
- Shlepa при `max_sample_per_dataset=1` берёт distractors из полного split.
- `evaluation_results.txt` включает все завершённые totals своего output-каталога.

## 3. Runtime smoke после pure-logic gate

В целевых Docker profiles выполнить минимум:

1. HF: по одному sample для `ruparam`, одного CopyText task,
   `daru/treewayextractive`, Shlepa и PPL-задачи с assistant prefill.
2. vLLM: `ruparam`, CopyText и Shlepa; extractive/PPL должны вернуть явную
   unsupported-capability error.
3. API: `ruparam` и Shlepa; CopyText должен завершиться ранней понятной
   capability error, не `AttributeError`.
4. LLM-as-a-Judge: один model/reference pair через штатный runner.
5. Повторить affected cells с `--force_recalc` в новых output directories и
   проверить `_params.jsonl`, sample array и `_total.jsonl`.

GPU-доступ проверять только внутри `docker run --gpus all`. Runtime smoke не
заменяется успешным import или pure-logic suite.

Фактически выполненные проверки, версии и ограничения записаны в
[`TASK_BUGFIX_REPORT.md`](TASK_BUGFIX_REPORT.md).
