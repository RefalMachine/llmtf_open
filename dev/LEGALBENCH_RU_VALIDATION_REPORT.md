# LegalBench-RU: implementation and validation

Дата: 2026-09-30. Основание: [план интеграции](LEGALBENCH_RU_INTEGRATION_PLAN.md).
Пользовательский контракт: [docs/legalbench_ru.md](../docs/legalbench_ru.md).
После этой validation отдельные пресеты объединены в
`benchmark/llmtf_legal_{instruct,foundational}.yaml` вместе с LawMC;
параметры общих запусков описаны в [руководстве](../docs/legal_benchmark.md).
Результаты ниже относятся к указанным здесь validation settings, а не к новым
Instruct Fast settings с temperature 0.3 и лимитом 200. Offline replay утилита
перенесена в `dev/tools/legalbench_ru_report.py`.
Исходный HEAD: `d36543888b6cc3865cf3a584b5c1bda0b0455567`; интеграция проверена
в незакоммиченном working tree. Существовавшие изменения других планов и
`results_foundational/` не относятся к этой интеграции.

Итог: 61 завершённый total, 2 762 оценённых примера с учётом повторных
измерений между cells. Все 61 total прошли offline replay. Полные Base/Instruct
прогоны составляют 2 436 из этих примеров. Fingerprints, hashes totals/samples,
сводки и проверка Base template сохранены в
[машиночитаемом приложении](LEGALBENCH_RU_VALIDATION_RESULTS.json).

## Реализация

- Pinned JSONL loader без Arrow coercion, проверка SHA-256, canary, composite
  keys, answer types и соответствия gold tool/args каталогу.
- 30 фиксированных demonstrations в шести buckets; остальные 816 записей
  используются при любом k. Upstream public/holdout не фильтрует корпус.
  Контекстные cohorts сохранили 205/189/8 примеров.
- `LegalBenchRU` с generate, k=0..5, строгим prompt budget, одинаковыми closed
  demonstrations во всех контекстных условиях и отсутствием task prefill.
- Побайтово сохранённые upstream prompts и reference scorer; corrected scorer
  с отдельной версией и change ledger. Оба scores сохраняются на одном output.
- Experimental registry IDs исключены из `all` общим resolver для normal/PPL.
  PPL отмечается как skipped; это не unsupported-capability failure.
- Task provenance hook до cache lookup: resources/helpers, split, catalog,
  protocol versions, selection и фактический hash local data override.
- В исходном validation snapshot использовались отдельные YAML для Base/Instruct,
  shots и контекстных условий и отдельная closed category (теперь заменены
  общими legal-конфигами без категорий). Offline report проверяет fingerprint, replay, membership,
  demonstrations и совместимость пар. Неполные пары не получают delta.
- Исправлен existing-API runner: его однопоточный цикл теперь использует
  `queue.Queue`. Прежняя `multiprocessing.Queue` могла вернуть `Empty` сразу
  после `put`, завершив запуск без единственной YAML-задачи и с exit code 0.

Новых зависимостей нет; Dockerfiles и requirements не изменялись. Все приведённые
runtime-результаты получены в существовавших образах. Начатая ранее лишняя
HF-пересборка отменена; дополнительный `legalbench-api-check` образ не использован.
В AGENTS.md уточнено: пересборка нужна только при изменении окружения.

## Данные и среда

| Компонент | Snapshot |
|---|---|
| HF dataset | `AlsKozlov/legalbench-ru`, `d6e21bfee9a2842ea3eb5b2e97a5c8f1186d41f1` |
| Upstream code/catalog | `090163977d0e9a58a5887a38ec144f95cb37e464` |
| Qwen3.5-2B | `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| Qwen3.5-2B-Base | `b1485b2fa6dfa1287294f269f5fb618e03d52d7c` |
| Runtime | torch 2.11.0+cu129, CUDA 12.9, Transformers 5.9.0, vLLM 0.21.0 |
| GPU | NVIDIA GeForce RTX 4090, driver 575.64.03; проверена внутри Docker с `--gpus all` |

Image IDs:

```text
llmtf:api
sha256:d10c2abd2015f960ff251f0eb984ddbab08abb86292adc874cef96d2eb8d7f8f
llmtf:hf-cu129
sha256:2146b9f44aa2964ebda2be52463da49936ec1d6b338baa826125016191611915
llmtf:vllm-cu129
sha256:d32605e9216b680faa60a510628cbc385d4ae58507125e9ea499697f03ec4722
```

GPU generate-проверки использовали 32K context, answer budget 512,
temperature=0, repetition penalty=1, presence penalty=0 и одну последовательность.
Reasoning cells: max=256, min=64, явный end token id 248069, проверенный по
закреплённому tokenizer. Это короткая validation-конфигурация, не рекомендация
оптимального reasoning budget. Batch size: HF=1, local vLLM/API=8.

Для vLLM использованы явные snapshot paths: offline download по имени репозитория
отказался работать из-за отсутствующих README/LICENSE/.gitattributes, хотя веса
и tokenizer присутствовали. Ни веса, ни revision при этом не заменялись.

## Offline gates

Проверки завершились успешно:

| Проверка | Результат |
|---|---|
| Prompt parity с закреплённым upstream | 846 closed + 205 grounded + 189 distractor + 8 temporal |
| Oracle и пустой output | оба scorer: 1 и 0 соответственно на всех 846 строках |
| Reference differential | 5 922 случая, 0 расхождений с upstream на допустимых входах |
| `test_refactor_logic.py` | 40 checks, 0 failures |
| `test_legalbench_ru.py` с pinned corpus | 13 tests, OK |
| `test_legalbench_ru_integration.py` | 7 tests, OK |
| Provenance/evaluator/YAML/show_results/CLI regressions | 7/6/7/4/4 tests, OK |
| compileall и `git diff --check` | OK |

Synthetic tests покрывают JSON escaping/nesting, duplicate keys, nonfinite
numbers, неверные типы tool/args, полный server.tool, отказ, typed args,
numeric boundaries, конфликтующие labels и несколько нормативных актов.
Few-shot проверен для всех k=0..5; k>5 и overflow отклоняются, неизвестный
token count сохраняет demonstrations. Transport failure даёт non-zero summary
без total; пустой ответ остаётся в denominator с нулевым score.

Offline replay отклоняет дубли, изменённый output, несовпадающий total и
несовместимые ресурсы/конфигурации. При отсутствующих cohort members выдаются
missing ids без delta. Стабильный cache hit подтверждён повторным HF-запуском;
изменения mode, k и local data bytes меняют fingerprint.

`pytest` отсутствует на host и в проверенных API/HF образах; зависимости ради
него не устанавливались. Выполнены штатный dependency-free runner и unittest.
API imports и evaluation проверены без torch, Transformers, vLLM и CUDA в
клиентском образе. GPU runtime принадлежит отдельному серверу.

## Runtime matrix

Основной корень итоговых артефактов: `/tmp/legalbench-release/`.
Предварительные каталоги `legalbench-validation*` и `legalbench-validated`
не входят в итоговую матрицу: там были промежуточные scorer/config versions.

| Backend / model | Завершённые cells |
|---|---|
| HF Instruct | initial 1 sample; closed 0/5-shot × thinking off/on по 8; 3 context cells по 8; strict reasoning 1 |
| HF Base | initial 1; closed 0/1/5-shot по 8; 3 context cells по 8 |
| Local vLLM Instruct | та же small matrix + full 816/205/189/8 при 0-shot, thinking off |
| Local vLLM Base | та же small matrix + full 816/205/189/8 при 5-shot |
| API Instruct, auto | initial 1; closed 0/5-shot off по 8; 3 context cells по 8 |
| API Instruct, best_effort | closed 0/5-shot on по 8; strict reasoning 1; отдельный diagnostic output root |
| API Base, auto | initial 1; closed 0/1/5-shot по 8; 3 context cells по 8 |

API thinking-on в `auto` сначала завершился ожидаемым
`PrefillCompatibilityError` без успешного total: reasoning continuation требует
whitespace-ended assistant prefill. vLLM 0.21 не обеспечивает его точное
сохранение. Дополнительные thinking-on/strict cells выполнены с явными
`best_effort` и probe; это **не доказательство exact local/API parity**.

Для Base API сервер использовал `json_to_jinja(default_foundational.json)`.
Реальный `/tokenize` совпал с локальным `render_local_chat_prompt` по всем token
ids восьми 5-shot prompts. Максимум 14 868 tokens; файл проверки:
`api/base/template_verification.json`. Клиентский flag сам по себе не считался
доказательством совпадения. Instruct 5-shot максимум: 14 906 tokens.

Standard `show_results.py` прочитал реальные closed totals; запрос bootstrap
корректно пропущен из-за `ALLOW_BOOTSTRAPPING=False`.

Проверки entry points: real one-sample local CLI, API CLI, local runner,
исправленный existing-API runner и managed-API runner завершились с total и
exit code 0. Managed runner сам поднял vLLM на порту 18766 и остановил его после
проверки. Отдельные тестовые API-серверы на порту 18765 также остановлены.

## Полные результаты local vLLM

Все значения ниже — micro score в [0,1]. Base и Instruct отличаются моделью и
числом shots; это сравнение не изолирует эффект few-shot.

| Model / shots | Mode | N | Corrected | Reference |
|---|---|---:|---:|---:|
| Instruct / 0 | closed | 816 | 0.442047 | 0.467539 |
| Instruct / 0 | grounded | 205 | 0.858537 | 0.858537 |
| Instruct / 0 | distractor | 189 | 0.502646 | 0.513228 |
| Instruct / 0 | temporal | 8 | 0.000000 | 0.000000 |
| Base / 5 | closed | 816 | 0.482396 | 0.497322 |
| Base / 5 | grounded | 205 | 0.775610 | 0.775610 |
| Base / 5 | distractor | 189 | 0.439153 | 0.444444 |
| Base / 5 | temporal | 8 | 0.000000 | 0.000000 |

Closed corrected scores по трекам:

| Track | N | Instruct / 0 | Base / 5 |
|---|---:|---:|---:|
| knowledge | 301 | 0.158915 | 0.179718 |
| reasoning | 333 | 0.675676 | 0.636637 |
| tool-use | 182 | 0.482841 | 0.700769 |

Парные corrected deltas проверены по всем заранее заданным ids:

| Пара | N | Instruct / 0 | Base / 5 |
|---|---:|---:|---:|
| grounded − closed | 205 | +0.112195 | +0.107317 |
| distractor − grounded | 189 | −0.354497 | −0.328042 |
| temporal − grounded | 8 | −0.875000 | −0.875000 |
| temporal − closed | 8 | −0.750000 | −0.625000 |

Missing ids в этих парах нет. Общий closed score не вычитался из subset mean.
Reference/corrected replay совпал с исходными totals. На closed outputs у
Instruct: 781 unchanged / 32 decreased / 3 increased; у Base: 794 / 19 / 3.
Эти counts включают эффект различного порядка округления.

## Ограничения

- Все исходные annotations имеют `needs_expert_review=true`. Проверены
  исполнение и scoring contract, а не юридическая истинность gold.
- Demo duplicate review: normalized question/context token Jaccard >=0.72
  исключал кандидатов; выбранные вопросы/gold технически просмотрены.
  Это не исчерпывающая проверка семантических дубликатов и не expert legal review.
- При full Instruct closed 23 ответа достигли 512 tokens: 18 citations,
  1 extraction, 4 tool-call. У Base — 10: 7 citations, 2 extraction, 1 tool-call.
  Grounded: ещё 1 Instruct и 3 Base extraction. Это результаты с фиксированным
  budget, не утверждение об отсутствии truncation. Backend не записывает
  finish_reason в этих artifacts; проверен `generated_len`.
- Temporal N=8 слишком мал для широких выводов. Full runs выполнены только
  на local vLLM; HF/API проверены малыми cohorts. Старую v4 probability/PPL
  матрицу не повторяли: model/reasoning/backend реализация не менялась.
- Отсутствуют sample-level resume, native tool execution, judge, reject metric
  и bootstrap CI; они не заявляются как реализованные возможности.

## Воспроизведение

Образы не пересобирать для исходного кода. Смонтировать repository, model cache
и каталог артефактов; для GPU-команд добавить `--gpus all --ipc=host`, задать
`CUDA_VISIBLE_DEVICES=0`, использовать `bash -c`. Corpus должен совпасть с
hash из packaged manifest. В этом запуске он находился в
`/tmp/legalbench-ru-hf.jsonl`, внутри контейнеров — `/audit/legalbench-ru-hf.jsonl`.

Основные команды внутри соответствующих существующих контейнеров:

```bash
# HF: два запуска; второй дополнительно имеет --base.
python -m dev.tools.validate_legalbench_ru --backend hf \
  --model Qwen/Qwen3.5-2B --data /audit/legalbench-ru-hf.jsonl \
  --output /audit/legalbench-release/hf/instruct
python -m dev.tools.validate_legalbench_ru --backend hf \
  --model Qwen/Qwen3.5-2B-Base --base --data /audit/legalbench-ru-hf.jsonl \
  --output /audit/legalbench-release/hf/base

# Local vLLM: повторить для Base snapshot с --base.
python -m dev.tools.validate_legalbench_ru --backend vllm \
  --model /root/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc \
  --data /audit/legalbench-ru-hf.jsonl --full \
  --output /audit/legalbench-release/vllm/instruct

# API Instruct: auto/off отдельно от best_effort/on.
python -m dev.tools.validate_legalbench_ru --backend api \
  --model Qwen/Qwen3.5-2B --data /audit/legalbench-ru-hf.jsonl \
  --output /audit/legalbench-release/api/instruct \
  --thinking-modes off --phases smoke matrix conditions
python -m dev.tools.validate_legalbench_ru --backend api \
  --model Qwen/Qwen3.5-2B --data /audit/legalbench-ru-hf.jsonl \
  --output /audit/legalbench-release/api/instruct_best_effort \
  --thinking-modes on --phases matrix strict \
  --assistant-prefill-policy best_effort --probe-api-prefill

# Base API server: --model должен быть фактическим pinned snapshot path.
python -m dev.tools.legalbench_ru_api_smoke --base \
  --model /root/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c \
  --served-name Qwen/Qwen3.5-2B-Base
# В другом контейнере, с доступом к loopback server и tokenizer:
python -m dev.tools.legalbench_ru_api_smoke --verify-template --base \
  --model /root/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c \
  --served-name Qwen/Qwen3.5-2B-Base --data /audit/legalbench-ru-hf.jsonl \
  --output /audit/legalbench-release/api/base/template_verification.json
python -m dev.tools.validate_legalbench_ru --backend api \
  --model Qwen/Qwen3.5-2B-Base --base --data /audit/legalbench-ru-hf.jsonl \
  --output /audit/legalbench-release/api/base

# Offline report; заменить instruct на base для второго полного прогона.
python -m dev.tools.legalbench_ru_report \
  /tmp/legalbench-release/vllm/instruct/full/closed \
  /tmp/legalbench-release/vllm/instruct/full/grounded \
  /tmp/legalbench-release/vllm/instruct/full/distractor \
  /tmp/legalbench-release/vllm/instruct/full/temporal \
  --output /tmp/legalbench-instruct-report.json
```

API client использовал `llmtf:api`; template verification, требующий tokenizer,
использовал HF profile без загрузки модели на GPU. API server использовал
`llmtf:vllm-cu129`, `--network host`, порт 18765, loopback binding и тот же cache.
Credentials в команды и artifacts не включались. Для smoke CLI/runners был
подготовлен YAML из основного конфигурационного файла с `closed_smoke` и
`max_sample_per_dataset: 1`; остальные sampling/budget параметры сохранены.

Команды проверки entry points (пути внутри контейнеров):

```bash
python evaluate_model.py --model_name_or_path Qwen/Qwen3.5-2B \
  --model_kind hybrid --disable_thinking --dataset_names legalbench_ru/closed_smoke \
  --few_shot_count 0 --name_suffix 0shot --max_sample_per_dataset 1 \
  --model_context_len 32768 --output_dir /audit/legalbench-release/cli/local
python evaluate_model_api.py --base_url http://127.0.0.1:18765 \
  --model_name_or_path Qwen/Qwen3.5-2B --api_profile vllm \
  --model_kind hybrid --disable_thinking --dataset_names legalbench_ru/closed_smoke \
  --few_shot_count 0 --name_suffix 0shot --max_sample_per_dataset 1 \
  --model_context_len 32768 --output_dir /audit/legalbench-release/cli/api
python -m benchmark.calculate_benchmark --backend hf \
  --model_dir Qwen/Qwen3.5-2B --benchmark_config /audit/legalbench-cli.yaml \
  --num_gpus 1 --output_dir /audit/legalbench-release/runners/local
python -m benchmark.calculate_benchmark_existing_api \
  --model_name Qwen/Qwen3.5-2B-Base --base_url http://127.0.0.1:18765/v1 \
  --api_profile vllm --benchmark_config /audit/legalbench-cli-base.yaml \
  --output_dir /audit/legalbench-release/runners/existing_api
python -m benchmark.calculate_benchmark_api \
  --model_dir /root/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc \
  --benchmark_config /audit/legalbench-cli.yaml --num_gpus 1 \
  --base_port 18766 --gpu_memory_utilization 0.8 \
  --output_dir /audit/legalbench-release/runners/managed_api
```

Для CLI без local data override исходный pinned JSONL был размещён в HF cache
`hub/datasets--AlsKozlov--legalbench-ru/snapshots/<dataset_revision>/legalbench_ru.jsonl`;
`HF_HUB_OFFLINE=1` исключал сетевую подмену snapshot, loader проверял SHA-256.
Полные raw outputs оставлены в `/tmp/legalbench-release`, а не скопированы в tests
или git. JSON-приложение содержит агрегаты и hashes, без полного текста gold.

## Переход к общему юридическому бенчмарку

Этот раздел фиксирует snapshot от 2026-09-30. После исправления Shlepa от
2026-10-02 LawMC в legal Foundational переключён на настоящие 5-shot;
ниже сохранены результаты прежнего zero-shot запуска. Актуальный протокол и
проверки — в [SHLEPA_FEW_SHOT_FIX_REPORT.md](SHLEPA_FEW_SHOT_FIX_REPORT.md).

По согласованному уточнению scope LegalBench-RU теперь является компонентом
составного legal-бенчмарка вместе с `shlepa/lawmc`. Девять отдельных YAML и
файл категорий заменены двумя `benchmark/llmtf_legal_*.yaml`. Структура YAML,
общие runners, scorers и стандартная отчётность в этом уточнении не изменены.
Offline replay перенесён в `dev/tools/legalbench_ru_report.py` и не требуется
для пользовательского отчёта.

Model/defaults новых конфигов совпадают с текущими
`llmtf_benchmark_instruct_fast.yaml` и `llmtf_benchmark_foundational.yaml`.
Batch size выбирают runners: HF 8, vLLM/API 10 000 000. LegalBench-RU Instruct:
0-shot, temperature 0.3, лимит 200 на режим. Foundational: 5-shot, temperature 0,
контекст 16 000, полный корпус. LawMC: фактический 0-shot и temperature 0 в
обоих пресетах; исходная реализация не использует few-shot demonstrations.

Новые проверки после объединения конфигов:

| Проверка | Результат |
|---|---|
| Dependency-free refactor suite | 40 checks, OK |
| Protocol с pinned corpus + integration | 20 unittest tests, OK |
| Benchmark YAML / show_results / provenance / evaluator / CLI | 7 / 4 / 7 / 6 / 4 tests, OK |
| Перенесённый offline replay на прежних полных Base/Instruct artifacts | OK |
| HF, новые Base/Instruct settings | 5 totals на профиль, 10 реальных примеров, OK |
| Local vLLM, новые Base/Instruct settings | 5 totals на профиль, 10 реальных примеров, OK |
| Стандартные HF/vLLM таблицы без категорий | LawMC и все четыре режима отдельными столбцами, OK |
| HF/vLLM replay новых artifacts | Совпадает с сохранёнными sample/total scores |
| compileall и git diff --check | OK |

Integration tests проверяют оба общих конфига, отсутствие коллизий имён,
передачу всех dataset IDs в HF/vLLM/API команды, совпадение model/generation
параметров с исходными пресетами и выполнение обеих task groups existing-API
runner с унаследованным batch size. API runner в этой проверке использует mock
endpoint/subprocess; нового реального API прогона общих конфигов не заявляется.
`pytest` в проверенных host/HF окружениях отсутствует; использован unittest.

Проверены все LegalBench-RU промпты, выбранные новыми конфигами, с pinned
tokenizers и framework renderer. Для Instruct использован консервативный
local-vLLM context default 8192; HF может иметь больший model context.

| Профиль | Mode | N | Максимум prompt tokens | Prompt budget |
|---|---|---:|---:|---:|
| Instruct | closed | 200 | 277 | 7 680 |
| Instruct | grounded | 200 | 483 | 7 680 |
| Instruct | distractor | 189 | 431 | 7 680 |
| Instruct | temporal | 8 | 166 | 7 680 |
| Foundational | closed | 816 | 14 885 | 15 488 |
| Foundational | grounded | 205 | 1 345 | 15 488 |
| Foundational | distractor | 189 | 1 293 | 15 488 |
| Foundational | temporal | 8 | 587 | 15 488 |

Все 1 815 выбранных промптов помещаются без сокращения demonstrations.
Аудит: `/tmp/llmtf-legal-prompt-budget-audit.json`.

GPU smoke использует те же pinned Qwen3.5 snapshots и существующие Docker
images, что основной validation выше. Временная утилита
`/tmp/validate_llmtf_legal_configs.py` читает новые YAML и переопределяет только
лимит до одного примера на компонент; sampling, thinking, context, shot count,
assistant policy и runner batch size сохраняются. Answer budget совпадает с
обычной CLI: LegalBench-RU 512, LawMC 1. Pinned JSONL передан через local data
override. Артефакты и `smoke_summary.json` находятся в
`/tmp/llmtf-legal-config-validation-final/{hf,vllm}/{instruct,foundational}/<model>`;
таблицы и replay — в `reports/{hf,vllm}/` того же каталога. Всего 20 новых totals
и 20 примеров, отдельно от исходных 61 totals. Новый полный inference
прогон не выполнялся; это проверка исполнения, а не оценка качества модели.

Команды временных проверок внутри существующих HF/vLLM containers
(working tree `/workdir`, host `/tmp` смонтирован как `/audit`, cache как
`/root/.cache/huggingface`; `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`,
`PYTHONPATH=/workdir`, для GPU `--gpus all --ipc=host`):

```bash
# В HF image:
python /audit/validate_llmtf_legal_configs.py --backend hf --profile instruct
python /audit/validate_llmtf_legal_configs.py --backend hf --profile foundational
python /audit/audit_llmtf_legal_prompt_budgets.py
# В vLLM image:
python /audit/validate_llmtf_legal_configs.py --backend vllm --profile instruct
python /audit/validate_llmtf_legal_configs.py --backend vllm --profile foundational
```

Стандартные таблицы проверены для всех четырёх каталогов через `show_results.py`
в API image с соответствующим общим YAML, `--num_proc 1`, без категорий.
Утилита `dev.tools.legalbench_ru_report` также выполнена для всех четырёх
каталогов: неполные cohorts smoke корректно отмечены без парных дельт.

Стандартный `Mean` сохранён для совместимости. Согласованного общего legal
score нет; пользовательский результат — отдельные значения компонентов.
Dockerfiles, requirements и runtime pins не менялись, образы не пересобирались.
