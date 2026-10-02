# LegalBench-RU (experimental)

Интеграция использует pinned `AlsKozlov/legalbench-ru` revision
`d6e21bfee9a2842ea3eb5b2e97a5c8f1186d41f1` и код upstream
`090163977d0e9a58a5887a38ec144f95cb37e464`. Данные загружаются через
`huggingface_hub` как исходный JSONL; SHA-256 проверяется до чтения gold.
Смешанные string/list/object ответы не проходят через Arrow.

Все 846 размеченных записей учитываются независимо от upstream public/holdout:
30 зарезервированы под demonstrations, 816 остаются в evaluation при любом k.
Canary исключён. Ключ — `(task,id)`, а не bare id. Answers upstream holdout уже
публичны; это не private holdout. Все строки имеют `needs_expert_review=true`;
проверки реализации не подтверждают юридическую корректность разметки.

## Задачи и запуск

| ID | N до лимита | Контекст |
|---|---:|---|
| `legalbench_ru/closed` | 816 | Исходный вопрос, context/options/catalog |
| `legalbench_ru/grounded` | 205 | `norm_text` |
| `legalbench_ru/distractor` | 189 | `distractor_text` |
| `legalbench_ru/temporal` | 8 | `temporal_text` |
| `legalbench_ru/upstream_all_zero_shot` | 846 | Reference, включая demo pool |

У каждого ID есть вариант с суффиксом `_smoke`: замороженная выборка до 8 строк.
Closed smoke охватывает все пять типов ответа, обе binary метки, multi-norm,
положительный tool и отказ. Контекстные smoke имеют собственные exact ids.
Новые ID исключены из `all`, включая PPL. Явный выбор доступен из обеих CLI и
всех трёх benchmark runners. PPL для LegalBench-RU **skipped**, exit code 0 при
отсутствии других ошибок: у задачи нет `get_answer`.

```bash
python evaluate_model.py \
  --model_name_or_path Qwen/Qwen3.5-2B \
  --model_kind hybrid --disable_thinking \
  --dataset_names legalbench_ru/closed \
  --few_shot_count 0 --name_suffix 0shot \
  --model_context_len 32768 --temperature 0 \
  --output_dir results/legalbench/instruct/0shot/closed

python evaluate_model.py \
  --model_name_or_path Qwen/Qwen3.5-2B-Base \
  --is_foundational --conv_path conversation_configs/default_foundational.json \
  --model_kind plain --disable_thinking \
  --dataset_names legalbench_ru/closed \
  --few_shot_count 5 --name_suffix 5shot \
  --model_context_len 32768 --temperature 0 \
  --output_dir results/legalbench/base/5shot/closed
```

Добавьте `--vllm` для local vLLM. Для API используйте `evaluate_model_api.py`,
`--base_url` и `--api_profile vllm`; credentials передавайте через environment.
Base API требует серверного foundational template, совпадающего с локальным;
один клиентский `--is_foundational` этого не обеспечивает.

Thinking-on использует assistant continuation внутри общего reasoning dispatcher.
У vLLM 0.21 exact whitespace-prefill недоступен: API thinking-on требует отдельного
диагностического запуска с явными `--assistant_prefill_policy best_effort` и
`--probe_api_prefill`; такой запуск не подтверждает точную local/API parity.

Общие YAML: `benchmark/llmtf_legal_instruct.yaml` (0-shot) и
`benchmark/llmtf_legal_foundational.yaml` (5-shot). Каждый включает LawMC и
все четыре режима LegalBench-RU, результаты которых сохраняются в один каталог
модели под разными именами. `show_results.py` выводит отдельное значение
каждого режима без категорий и парных сравнений. Standard `Mean` не является
согласованным итоговым баллом юридического бенчмарка.

Параметры общих конфигов соответствуют основным LLMTF пресетам: в частности,
LegalBench-RU Instruct использует temperature 0.3 и лимит 200 на режим, а Base —
temperature 0 и контекст 16 000. Прямые команды выше показывают самостоятельный
детерминированный прогон всей задачи с контекстом 32K. Запуски и таблица всего
юридического бенчмарка описаны в [руководстве](legal_benchmark.md).

## Протокол

Используются шесть фиксированных buckets по answer_type (extraction разделён
по track), по пять demonstrations. k=0..5 выбирает префикс bucket; k>5 запрещён.
Текущий gold не участвует в выборе bucket. Демонстрации всегда closed и
идентичны для парных условий. При переполнении контекста задача падает с ошибкой,
не уменьшает k и не обрезает каталог. При неизвестном API token count все k
сохраняются; ошибка серверного контекста остаётся ошибкой исполнения.

Split manifest содержит exact ids, порядок, hashes demo prompts/answers,
контекстные cohorts и smoke cohorts. Кандидаты ранжировались с seed 555 в порядке
закреплённого export; нормализованные question+context с token Jaccard >=0.72
исключались из demo candidates. Проведён технический просмотр выбранных вопросов
и коротких gold; юридическая экспертная проверка не заявляется. Этот эвристический
аудит не доказывает отсутствие всех семантически близких примеров.
`dev/tools/legalbench_ru_split_candidates.py` воспроизводит исходных кандидатов;
изменение packaged pool требует новой версии протокола и повторной проверки.

Один логический `generate`, исходный текстовый каталог всех 79 tools в каждом
user message, включая demonstrations. Настоящие tools не выполняются; native
function calling, structured decoding и repair retry не используются. Нет
system message/task prefill. Gold demonstrations сериализуется коротко, без
объяснений. Answer budget 512; параметры generation задаются при запуске.
Детерминированная validation использовала temperature 0, одну последовательность,
repetition penalty 1, presence penalty 0. Thinking использует общий dispatcher,
скореру передаётся только answer continuation.

## Scoring и отчёт

Основной experimental score — `legalbench_ru_corrected_v1`, micro mean без
округления per-sample. Рядом сохраняется
`legalbench_ru_upstream_0901639_v1_type_repair`: исходные численные правила,
`round(raw,3)` до mean; неверный тип tool/args даёт 0 вместо падения.
`upstream_all_zero_shot` выбирает reference primary и требует k=0.

Исправления: строгий JSON с одним объектом/необязательным code fence,
запрет duplicate keys и NaN/Infinity, полный server.tool, явный top-level null
для отказа, границы числовых строк, локальная привязка статьи к акту,
однозначные binary/MC labels. Числа/единицы не конвертируются; ведущие нули
идентификаторов сохраняются. Неоднозначность citation отмечается в status.
Tool args сравниваются как типизированный JSON: строки case-sensitive,
arrays ordered, null отличается от missing. Дополнительные args не штрафуют
weighted score, но нарушают annotation exactness. Каталог не содержит полной
JSON schema, поэтому schema validity/semantic correctness поиска не заявляются.

Все изменения описаны в `llmtf/tasks/legalbench_ru/change_ledger.json`.
Sample artifacts сохраняют полный raw output, оба scores, parse status,
routing/exact/args diagnostics и requested/effective shots. Details включают
micro score, count, разрезы task/track/domain/answer_type, balanced binary/tool
routing и counts расхождений скореров. Пустой/invalid output остаётся в denominator;
backend failure не создаёт успешный total. Legacy bootstrap отключён.

Обычные значения всех режимов уже записаны в totals и доступны через
`show_results.py`; дополнительная обработка для юридического бенчмарка не нужна.
Для maintainer-проверки сохранённых результатов доступна необязательная утилита:

```bash
python -m dev.tools.legalbench_ru_report \
  results/legal/instruct/Qwen3.5-2B \
  --output legalbench_replay.json
```

Утилита использует только stdlib, не загружает LLM и не обращается в сеть.
Он replay-ит оба скорера, проверяет totals/params, versions, ids, demonstrations
и cohort membership. Дельты рассчитываются только при полном покрытии заранее
заданной пары. Модели, shots, sampling и budgets должны совпадать. Temporal N=8
показывается явно. На неполной выборке сохраняются matched/missing ids без delta.
CI/bootstrap не реализованы. Нельзя вычитать общий closed score из subset score.

Кэш — только завершённый total с совпадающим fingerprint. В fingerprint входят
локальные hashes protocol helpers, catalog, split, manifest, версии, mode,
selection и фактический hash local data override. k, generation и model/template
записывает общий run config. После прерывания inference задача перезапускается;
sample-level resume не реализован. Старые artifacts с другой версией scorer
offline report отклоняет, автоматического пересчёта baseline нет.

Данные/каталог: CC BY 4.0; перенесённый upstream код: Apache-2.0. Attribution,
notice и тексты лицензий находятся рядом с ресурсами задачи. Протокол не входит
в прежние leaderboard suites. Фактическое покрытие проверок — в
`dev/LEGALBENCH_RU_VALIDATION_REPORT.md`.
