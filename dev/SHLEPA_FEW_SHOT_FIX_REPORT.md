# Исправление few-shot в Shlepa — 2026-10-02

## Результат

Раньше `ShlepaSmallMMLU.load_dataset()` принимал `few_shot_count`, но не
передавал его в построение промпта. Все четыре задачи — `shlepa/moviesmc`,
`shlepa/musicmc`, `shlepa/lawmc`, `shlepa/booksmc` — фактически работали
zero-shot даже при сохранённом параметре 5-shot.

Теперь k означает ровно k полных демонстраций user/assistant перед вопросом.
Демонстрации берутся из первых пригодных различных вопросов train; seed их
перемешивания — 555. Правильная буква учитывает перестановку 12 вариантов.
Оценочный assistant содержит только `Ответ:`, без правильной буквы.

Вопросы демонстраций и дубликаты их текстов после нормализации регистра и
пробелов исключаются из evaluation. Sample limit применяется после исключения.
Поэтому выборка evaluation зависит от k; это не парное сравнение качества
0-shot и 5-shot на неизменном наборе вопросов. Distractors продолжают браться
из полного train. Семантические дубликаты не проверяются.

Отдельный RNG демонстраций не меняет порядок вариантов общих оценочных
вопросов при одинаковом framework seed. Недостаточное число демонстраций,
отсутствие оставшихся evaluation-вопросов и превышение prompt budget дают
ошибку; k не уменьшается молча. При недоступном token counter все k сохраняются.
Sample trace `_shlepa` содержит requested/effective shots, индексы
демонстраций и оценочной записи, длину полного промпта.

Дополнительно новые тесты выявили ошибку текстового `correct_answer`: прежний
fallback зависел от порядка колонок и мог сопоставить ответ с самим полем
`correct_answer`. Теперь проверяются только `answerA`–`answerD`, причём
совпадение должно быть единственным. Непригодные строки пропускаются.
Это исправление может менять и zero-shot baseline. Формула accuracy и
probability scoring для пригодных строк не менялись.

## Конфиги и миграция

- Legal Foundational LawMC использует реальные 5-shot.
- Legal Instruct LawMC сохраняет явные 0-shot.
- Основные Instruct и Instruct Fast YAML не задают число shots для Shlepa:
  их CLI default 5 теперь действительно добавляет демонстрации. Для zero-shot
  нужен явный `few_shot_count: 0`. Их настройки не менялись.
- Структура YAML, batch size, runtime defaults и зависимости не менялись.
  Docker-образы не пересобирались.

В run config добавлены dataset identity и стабильный протокол
`shlepa_train_demonstrations_v1`. Source digest и provenance изменяют
fingerprint: старые artifacts не могут считаться совместимым кэшем.
Для пересчёта используйте отдельный output directory или `--force_recalc`.
Исторические результаты с k>0 следует интерпретировать как zero-shot;
новый few-shot baseline требует нового запуска.

## Проверки

Dependency-free `tests/test_refactor_logic.py`: 40 проверок прошли.
В CPU API-образе прошли 47 unittest-тестов:

| Suite | Тестов |
| --- | ---: |
| `test_shlepa.py` | 12 |
| `test_legalbench_ru_integration.py` | 7 |
| `test_run_provenance.py` | 7 |
| `test_evaluator_integrity.py` | 6 |
| `test_benchmark_config.py` | 7 |
| `test_cli_config.py` | 4 |
| `test_show_results.py` | 4 |

Новые тесты проверяют фактические демонстрации всех четырёх задач, правильные
буквы после shuffle, отсутствие gold в оценочном prefill, исключение
дубликатов, стабильность общих вопросов и sample limits, ошибочные gold,
строгий context budget, неизвестный token count и cache fingerprints.
`pytest` отсутствовал в проверенных host/HF окружениях; unittest выполнен.
`compileall` и `git diff --check` прошли.

## Реальные модели

Каждая probability-ячейка включает все четыре датасета с k=0/1/5 и лимитом
8 исходных evaluation-записей после исключения демонстраций. В `music_mc`
оказалось 7 пригодных строк в каждом таком окне, поэтому probability-серия
содержит 93, а не 96 примеров. HF дополнительно оценивает 8 LawMC 5-shot
примеров через PPL (mean answer-token log probability).

| Backend | Модель | Totals | Оценённых примеров | Максимальный prompt |
| --- | --- | ---: | ---: | ---: |
| HF | Qwen3.5-2B Instruct | 13 | 101 | 1625 |
| HF | Qwen3.5-2B-Base | 13 | 101 | 1597 |
| vLLM local | Qwen3.5-2B Instruct | 12 | 93 | 1625 |
| vLLM local | Qwen3.5-2B-Base | 12 | 93 | 1597 |
| API, CPU client → local vLLM | Qwen3.5-2B-Base | 12 | 93 | 1597 |
| Всего | | 62 | 481 | |

Все серии завершились с exit code 0. Проверены requested/effective shots,
индексы демонстраций и отсутствие оценочного индекса среди них в sample
artifacts, run config и fingerprint в totals/params. API дополнительно
выполнил одну реальную генерацию из LawMC 5-shot промпта и вернул явное
unsupported-capability для PPL. Эти две проверки не входят в 62 totals.
В клиентском API-образе `find_spec('torch')` и `find_spec('vllm')` дали false.
Тестовый API-сервер после проверки остановлен.

LawMC 5-shot accuracy на этих малых выборках: Base 0.375 на HF, vLLM и API;
Instruct 0.625 на HF и vLLM. Это smoke, не оценка полного бенчмарка и не
доказательство общего local/API parity. API probability использует
ограниченный `top_logprobs=20`. Thinking явно выключен; полный reasoning/v4
matrix, полный датасет и API Instruct в этом цикле не запускались.
Local vLLM PPL unsupported-check был добавлен в утилиту после начала его
серий и в этих конкретных local-прогонах не выполнялся.

Runtime: RTX 4090, driver 575.64.03, torch 2.11.0+cu129, CUDA 12.9,
Transformers 5.9.0, vLLM 0.21.0. Использованы существующие образы:
`llmtf:api` (`d10c2abd2015`), `llmtf:hf-cu129` (`2146b9f44aa2`),
`llmtf:vllm-cu129` (`d32605e9216b`).

Model snapshots:

- Instruct: `15852e8c16360a2fea060d615a32b45270f8a8fc`.
- Base: `b1485b2fa6dfa1287294f269f5fb618e03d52d7c`.

## Воспроизведение

Утилита: `python -m dev.tools.validate_shlepa_few_shot`. Запуск из корня
репозитория в соответствующем Docker-профиле; рабочее дерево монтируется
в `/workdir`, Hugging Face cache — в `/root/.cache/huggingface`, `/tmp` —
в `/audit`. Local GPU-команды используют `--gpus all --ipc=host`,
`CUDA_VISIBLE_DEVICES=0`, `HF_HUB_OFFLINE=1`, `PYTHONPATH=/workdir`.
Использовались следующие аргументы; `$instruct_snapshot` и `$base_snapshot`
обозначают полные пути к указанным snapshots внутри смонтированного cache:

```bash
python -m dev.tools.validate_shlepa_few_shot --backend hf --model "$instruct_snapshot" --output /audit/shlepa-fewshot-validation/hf/instruct --ppl
python -m dev.tools.validate_shlepa_few_shot --backend hf --model "$base_snapshot" --base --output /audit/shlepa-fewshot-validation/hf/base --ppl
python -m dev.tools.validate_shlepa_few_shot --backend vllm --model "$instruct_snapshot" --output /audit/shlepa-fewshot-validation/vllm/instruct
python -m dev.tools.validate_shlepa_few_shot --backend vllm --model "$base_snapshot" --base --output /audit/shlepa-fewshot-validation/vllm/base
```

API-сервер запускался в GPU vLLM-образе с `--network host`:

```bash
python -m dev.tools.legalbench_ru_api_smoke --base --port 18795 --model "$base_snapshot" --served-name Qwen/Qwen3.5-2B-Base
```

Клиент запускался в `llmtf:api` с `--network host`, offline cache и без GPU:

```bash
python -m dev.tools.validate_shlepa_few_shot --backend api --model Qwen/Qwen3.5-2B-Base --base --output /audit/shlepa-fewshot-validation/api/base
```

Artifacts находятся в `/tmp/shlepa-fewshot-validation/{hf,vllm,api}/{instruct,base}`;
каждая выполненная серия имеет `validation_summary.json`, sample arrays,
params и totals. API generation trace — `api/base/generation_smoke.json`.
Эти runtime outputs остаются вне репозитория.
