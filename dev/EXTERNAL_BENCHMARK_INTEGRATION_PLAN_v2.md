# RuBookSum / RuWikiBench: план интеграции v2

Дата: 2026-09-23. Статус: **рекомендуемый проект реализации**.
Не является готовой спецификацией опубликованного протокола или отчётом запуска.
Основания: [аудит](EXTERNAL_BENCHMARK_AUDIT.md),
[Task/eval v2](TASK_EVAL_GRAPH_REFACTOR_PLAN_v2.md).
Этот документ заменяет [план v1](EXTERNAL_BENCHMARK_INTEGRATION_PLAN.md).

## 1. Цель и границы

Семь самостоятельных tasks запускаются обычными single-model CLI и всеми
benchmark runners, используют существующие `LLM`/backends и общий lifecycle
artifacts/cache/reporting. Внешние benchmarks разрешено исправлять и
реорганизовывать при сохранении их подхода и воспроизводимости прошлых
экспериментов. Исходные commits остаются доступной исторической точкой отсчёта;
исправления оформляются отдельными commits с проверкой влияния на результаты.
Их CLI, OpenAI clients и файловые side effects в runtime LLMTF не импортируются.

Приоритет первой интеграции — сохранить исходный эксперимент, исправить
подтверждённые дефекты реализации и проверить перенос. Предложенный ниже
`llmtf-v1` — кандидат исправленного протокола, а не разрешение автоматически
изменить scoring или prompts. Разделы 5–7 описывают этот кандидат; отличия от
upstream проходят классификацию из раздела 2.1 до включения в default.
Исторический протокол остаётся запускаемым через закреплённый reference runner
и окружение; не обязательно поддерживать два одинаково развиваемых task API.
Новая версия сама по себе не доказывает совместимость с опубликованными числами.

В первый scope не входят: end-to-end Wiki article agent, coding environment,
публичное зеркало книг/статей, автоматическая установка plugins и изменение
основных leaderboard categories. Gitlink можно обновить на проверенный commit
исправлений отдельным изменением с сохранённым old-to-new соответствием и
отчётом сравнения. Не оставлять единственную копию repairs в dirty submodule
или непубликуемом временном checkout.

## 2. Решения по итогам аудита

| Вопрос v1 | Предлагаемое решение |
|---|---|
| `PipelineTask.execute_batch(model, ...)` | Workflow helper над единым EvaluationPlan/dispatcher; task выдаёт requests |
| Нужно ли ждать весь task graph refactor? | Нет; R1 общий dispatcher/workflow/persistence, R2 coding и массовая миграция |
| Полнота первой поставки | Сначала одна подтверждённая task, затем остальные; семь — конечный scope |
| Resume | Обязателен на границе завершённой книги/статьи; частичная пересчитывается |
| Trace | Полные logical requests/responses локально, большие тексты по content references |
| Few-shot | Строго 0; ненулевой запрос — ранняя ошибка, а не silent override |
| Task flags | Фиксированные protocol variants; общие execution/resource settings через единую schema |
| API ranking | Только endpoint/model с проверенным next-token contract и достаточным coverage |
| Wiki reference hints | Сохранить как явно названные oracle variants; blind mode — будущий новый protocol |
| Categories | Отдельный experimental suite; включение в основной рейтинг отдельным решением |

Это проектные рекомендации для реализации. Непроверенные revisions, права,
ресурсные лимиты и runtime configurations не объявляются согласованными
фактами: ниже у каждого есть конкретный gate.

### 2.1 Исправления внешнего кода и сохранение эксперимента

Уточнение пользователя: изменять внешние benchmarks можно, если это не ломает
их логику/подход и воспроизводимость предыдущих экспериментов. Применять
следующую классификацию; название «bugfix» само по себе не гарантирует parity.

| Класс изменения | Примеры | Условие включения |
|---|---|---|
| Исправление исполнения без изменения успешного эксперимента | Syntax/import fix, корректное завершение one-article run, явные пути, packaging | Replay сохраняет data selection, prompts, calls и scores успешных случаев; repaired snapshot закреплён |
| Исправление, меняющее затронутые результаты | Blueprint list→string, chunk tail, восстановление ошибочного source mapping | Проверка намерения метода, diff affected samples и новая implementation/protocol identity; прежний путь воспроизводим |
| Изменение методики | Новый ranking score, иное взвешивание статей, sampling defaults, обязательная singleton generation | Не входит автоматически в совместимый перенос; отдельный вариант/версия после обоснования |

В частности, удаление chunk tail trimming, YES/NO ratio, article-macro вместо
section-micro и дополнительная генерация singleton пока **кандидаты изменений**.
Singleton passthrough может быть оптимизацией метода; сначала выяснить роль
по описанию/экспериментам, не считать его доказанной ошибкой лишь из-за
отсутствия model call. Oracle source selection Wiki сохраняется как часть
исходного подхода; её явное документирование не требует менять алгоритм.

Для каждой правки: regression fixture → минимальный fix → сравнение before/after
по affected и unaffected cases → commit + change ledger. Для прошлого
эксперимента сохранять code/data/model revisions, prompts/configs, environment
и доступные original outputs. Если его реальный executable snapshot неизвестен,
это отмечается как незакрытая воспроизводимость; одного текущего gitlink мало.

Можно выделить benchmark algorithm в тестируемые модули внешнего проекта и
использовать их через тонкий LLMTF adapter при подходящих зависимостях/условиях
распространения. Это предпочтительнее двух расходящихся копий алгоритма.
Algorithm module получает данные/результаты стадий, но не создаёт собственный
LLM client. Если прямое переиспользование невозможно, adaptation и shared
golden fixtures должны явно фиксировать соответствие реализаций.

## 3. Immutable inputs и data inventory — этап E0

Reference code:

| Проект | Commit |
|---|---|
| RuBookSum | `a997d2154b343bf01d53ce7d7bf1fa2777fc6070` |
| WikiBench | `5dbf8d95c2cced4583943639f9a59e3e22a259d4` |

Исходные кандидаты datasets из v1:

| Dataset | Revision | Что ещё проверить |
|---|---|---|
| `NejimakiTori/literature_sum` | `972f0646a65a63cb513a2eea276766471e98b9f4` | Реальные 634 rows, split/schema, `text: list[str]`, order/content hashes |
| `NejimakiTori/RuWikiBench` | `ae4c23629fa286c404ce22bb751f94648b5dd550` | Доступность revision, 100 articles, format/splits, HTML/source structure и hashes |

Tokenizer `DeepPavlov/rubert-base-cased`, encoders `deepvk/USER-bge-m3` и
`sergeyzh/BERTA` пока имеют только имена. **Точные revisions не закреплены
этим документом.** До E1 требуется manifest с resolved commits, config/weights
identity и проверкой загрузки. Не заполнять отсутствующие hashes догадками.

Inventory выполняется отдельно от дорогой модели и сохраняет:

- полную схему, число строк, пустые/неверные поля, дубликаты и порядок;
- для books: число и ids прошедших `cap_chars=80000`, распределение длины,
  chunks и reference summaries; не называть этот срез полным long-document set;
- для Wiki: original title, sanitized title и collisions, source order,
  source ids/files, reference links и section occurrences; dangling mappings;
- counts snippets на статью и полный corpus digest; число eligible sections,
  пустые references, no-source и no-selected-source случаи;
- версии parser, tokenizers, morphology dictionaries, stopwords resources,
  encoders и фактические embedding norms/sequence limits;
- явный список сохраняемых и исправляемых reference behaviours.

Книга идентифицируется snapshot + исходной позицией/проверенным ключом.
Wiki article/source/section ids строятся из snapshot и структурной позиции;
повторённый заголовок не схлопывается в одну секцию. Filename из dataset
не становится путём записи. Prepared data хранится структурированно, без
воссоздания `Articles/` и без network download источников из HTML.

До включения upstream кода/prompts или публикации fixtures/text caches
зафиксировать конкретные условия и attribution. Отсутствие отдельного LICENSE
в snapshots и dataset metadata перечислены в аудите; это unresolved input,
а не разрешение на перенос. Common executor и синтетические fixtures можно
разрабатывать независимо. Внешним авторам автоматически не писать.

**Выход E0:** `data_inventory.json`, `protocol_manifest.json`, таблица
reference→llmtf изменений, synthetic fixtures и перечень допустимых к
распространению материалов. Названия артефактов здесь проектные, файлы ещё
не созданы. Без E0 нет frozen protocol и полного benchmark run.

## 4. Идентичности задач

Предлагаемые registry ids намеренно отражают версию и oracle/pool semantics:

| ID | Смысл | Primary scalar |
|---|---|---|
| `rubooksum/hierarchical-v1` | Reference-shaped merge, исправленный chunking | Sentence embedding F1 |
| `rubooksum/hierarchical-filtered-v1` | Дополнительная фильтрация chunks/levels | Sentence embedding F1 |
| `rubooksum/blueprint-v1` | Вопросы/ответы и адаптивный merge | Sentence embedding F1 |
| `rubooksum/blueprint-cluster-v1` | Общий blueprint из кластеров вопросов | Sentence embedding F1 |
| `ruwikibench/ranking-pool-v1` | Query generation + BM25 + next-token reranking | `ndcg_pool` |
| `ruwikibench/outline-oracle-v1` | Известные source documents + reference-informed cluster seeds | Heading embedding F1 |
| `ruwikibench/sections-oracle-v1` | Gold sections/mapping/text-assisted source selection | Article-macro sentence embedding F1 |

Имена фиксируются в E0 до первой регистрации. Старых встроенных tasks с этими
ids нет, migration aliases не нужны. Неверсионированный alias не должен молча
переключать протокол. У каждой task собственные outputs и completion status;
общий prepared cache не делает их одним run.

Primary scalar нужен текущему формату totals, но не означает включение в
общий leaderboard. P/R/ROUGE/BLEU, coverage и timing сохраняются отдельно;
среднее по всем этим числам не является общей оценкой качества.

Новые entries не включаются в default `all`: нужны explicit ids или отдельный
experimental YAML. Добавить metadata default selection и regression test
старого состава `all`. Suite пишет в отдельный output namespace: текущий
`Evaluator.create_report` усредняет все totals каталога независимо от categories.
Показывать новые per-task scores отдельно; directory mean обозначать как
техническое среднее данного набора, не как основной LLMTF leaderboard score.

## 5. RuBookSum: точный algorithm contract

### 5.1 Общая подготовка

`text` валидируется как list of strings, соединяется `\n`, затем применяется
`len(text) <= 80000`, затем first-N eligible в исходном порядке. Сохраняются
исходные ids и exclusions. Пустой/невалидный dataset row — data error;
не исключать его только потому, что evaluation на нём неудобен.

Chunking: закреплённый rubert tokenizer без добавления специальных токенов,
window 2000, stride 1800, decode с закреплёнными options. Каждый token window
сохраняется как диапазон и digest decoded text. **Удалить** reference char-based
tail trimming; это явное изменение `llmtf-v1`. Evaluation tokenizer независимо
проверяет полный prompt каждого вызова. Ни число символов, ни rubert tokens
не заменяют эту проверку.

Zero-shot, одна sequence. Sampling defaults нового протокола предлагаются
greedy (`temperature=0`, `top_p=1`, repetition penalty 1); поддержанные run
overrides записываются и образуют отдельный baseline configuration.
Это отличается от внешних near-zero temperature defaults. Скрытые
model-name overrides и текстовое отключение thinking не переносятся.

### 5.2 Hierarchical

Chunk summaries с word-limit instruction 500 и answer cap 2048. Word limit
является инструкцией, не обещанием жёсткой длины результата; превышение
логируется, текст не режется после generation.

Сохраняется reference topology: для шести узлов — merge первой тройки,
затем merge второй тройки с предыдущей summary как контекстом; отдельная
тройка сжимается одним вызовом; хвост переносится. Финальная пара merge-ится,
одиночная summary возвращается. Merge caps 2048. Fixtures на 1, 2, 3, 4, 5,
6, 7 и 12 chunks фиксируют ordered input ids и dependencies.

Filtered сохраняет threshold 0.85 и применение до initial generation и после
уровней. Нормализованные embeddings сравниваются со всеми предыдущими
элементами, включая отфильтрованные; первый сохраняется. Исключённые ids и
scores видны в trace. Изменение на retained-only dedup — отдельный protocol.

### 5.3 Blueprint

Default: questions на chunk (cap 1024), answers на каждый вопрос (256),
chunk summary (2048). Парный merge сначала соединяет тексты; при превышении
500 whitespace-separated words строит новый blueprint и summary. Финальный
дополнительный compression, если требуется, выполняется не более одного раза.

Cluster: вопросы всех chunks кодируются один раз на стадию; cluster count
`min(Q, 15, max(2, floor(sqrt(Q))))` при Q>0. `Q=1` обрабатывается без KMeans;
Q=0 — invalid prediction с нулевым final score, не потерянная книга.
Из каждого кластера выбирается максимум 10 вопросов task-local RNG;
generalization cap 128. Единый blueprint — **строка**, явно передаваемая в
summary/merge, а не Python list representation. Новый общий blueprint
строится только если понадобится следующий уровень/финальное compression.

Зафиксировать `random_state`, `n_init`, clustering implementation и устойчивый
порядок cluster ids/members; общий RNG не должен зависеть от числа активных книг.
Парсер вопросов и bound `max_questions_per_chunk` определяются в E0 по
fixture/inventory, до live run. Превышение bound — explicit invalid prediction,
не silent truncation списка. Book-level call/depth guards защищают execution;
их срабатывание — incomplete run, не successful benchmark total.

### 5.4 Метрики

Общая sentence embedding PRF: razdel sentence segmentation, normalized encoder
embeddings, max cosine matching по строкам/столбцам; arithmetic means дают P/R,
F1 вычисляется отдельно для книги. Не называть это стандартным `bert_score`.
Для непустого reference и пустого prediction — P=R=F1=ROUGE=0.
Пустой reference — data-contract error до generation. Не вводить незаявленное
clipping cosine; правило F1 при неположительном P+R фиксировать явно как 0.

ROUGE-L: выбранный пакет `rouge` с pinned version/options, без добавления
неиспользуемого upstream stemming. Whitespace normalization фиксируется в
manifest. Reasoning удаляется штатным LLM contract, не task regex по model
markers. На book уровне сохраняются все scalar scores, а total — mean books.

## 6. Wiki: preparation и протоколы

### 6.1 Общая подготовка

Весь закреплённый corpus используется для BM25 независимо от evaluation N.
Snippets: 600 whitespace-separated words, overlap 0; deterministic order
article/source/snippet ids. Морфология, стоп-слова, BM25 method/k1/b/dtype и
tie policy фиксируются. Manifest связывает индекс с **точным ordered corpus**.

HTML parser работает offline; сохраняются исходные heading positions,
тексты и source mappings. Все embeddings присоединяются по ids. Не смешивать
original title, sanitized title и identity. Общий source между статьями
сохраняется с исходной принадлежностью; diagnostics считают дубликаты, но
labels не переопределяют без нового протокола.

Нормализация embeddings для cosine явная. Параметры encoder для preparation,
clustering и metrics различаются по operation; metric prompt `Classification`
из upstream Wiki фиксируется отдельно. Проверить, что snippet length не
вызывает незаявленное encoder truncation; явная encoder max-length/truncation
policy входит в protocol manifest. Нельзя объявить all-token coverage только
по word-window размеру.

Prepared cache разделён на content/parser/snippets, lexical index и embeddings.
Ranking не должен вычислять embeddings, которые ему не нужны. Cache keys
содержат input content + implementation/config/runtime hashes, для embeddings
также encoder revision/options. Нужны lock, temporary build, atomic manifest
и проверка corruption. Рабочие каталоги сабмодулей и untrusted pickle не
используются. Отдельная standalone prepare команда допустима, но task также
должна уметь получить/построить этот cache через штатный сервис.

### 6.2 Ranking pool

Два generate requests (RU/EN query, answer cap 512 каждый) не зависят друг от
друга и могут batch-иться. Затем BM25 выбирает
`k=min(3 * article_snippet_count, full_corpus_size)`; empty corpus/unknown
article membership — data errors. Записываются k, candidate ids, BM25 scores,
retrieved positives и positive count в полном корпусе.

Каждый candidate получает next-token YES/NO scores через LLM probability
primitive с существующим centralized surface/max rule. Новый rerank score:
`s = p_yes / (p_yes + p_no)`. Это нормированный candidate score, **не**
калиброванная probability релевантности и не сумма probabilities всех synonyms.
Сохраняются исходные p_yes/p_no и coverage. Zero/invalid denominator — error.
Reference multi-position/`1-p_no` rule намеренно не переносится.

Capability gate имеет две части:

1. До генерации: поддержка operations, проверенная single-token
   представимость candidate surfaces у выбранной модели/deployment. Для API
   без доступного tokenizer/probe нет основания заявлять exact parity;
   неизвестный контракт даёт explicit unsupported result.
2. **На каждом probability response:** scores для обоих кандидатов должны
   быть определены. API top-k допустим только если наблюдения и проверенная
   семантика sorted top-k позволяют восстановить max каждого набора surfaces.
   Наличие одного кандидата или `candidate_ranking_resolved=True` недостаточно.
   Отсутствующие probabilities не превращаются в нули. Если encoder/tokenizer
   знает дополнительные допустимые surfaces, их coverage также учитывается
   общим continuation contract, а не локальной token эвристикой.

Один preflight sample не гарантирует coverage последующих snippets. При
неполноте после bounded transport retries — task failure, diagnostic trace,
без successful total; не повторять запрос с изменённым prompt/temperature или
выключенным thinking. Увеличение server/client top-k — другая явная run config.

Stable sorting: `(-s, bm25_rank, snippet_id)`. Relevance определяется upstream
article membership. Основные `ndcg_pool`, `r_precision_pool` вычисляются с
ideal/R **внутри retrieved pool**; при R=0 оба равны 0, sample сохраняется.
Дополнительные BM25 pool recall и число positives показывают retrieval coverage.
Full-corpus end-to-end IR метрики и человеческая relevance разметка — другой
протокол. Candidate snippet не является независимым bootstrap sample.

### 6.3 Outline oracle

Вход — snippets известной статьи. `neighbor_count=0`, description mode включён.
Reference mapping определяет initial cluster centers как embeddings первых
доступных sources соответствующих header blocks. В provenance:
`reference_access=source_membership+header_source_mapping`.

Cluster count ограничен числом samples; duplicate/degenerate centers имеют
детерминированную policy из fixtures. Пустые sources — data failure. До пяти
ближайших snippets каждого кластера (stable distance/id ties) подаются на
description → outline; финальный combined outline — отдельный generate call.
Caps по 512, topology сохраняется, переполнение context не меняет число
snippets автоматически.

Scorer извлекает непустые Markdown headings, сравнивает titles embedding PRF
с reference headings. Уровни и порядок этой метрикой не оцениваются — это
ограничение должно быть в пользовательской документации. Нет извлечённых
headings — invalid prediction с нулём. Empty reference headings — data error.

### 6.4 Sections oracle

В provenance: `reference_access=section_titles+source_mapping+section_text`.
Eligibility определяется **до model generation**: nonempty filtered reference,
доступные mapped sources и хотя бы один snippet после gold-text similarity
threshold 0.6. Все исключения, включая no-selected-snippet, записываются;
это ограниченный oracle-supported subset, не все секции Википедии.
Если статья не имеет eligible sections — явное dataset exclusion до selection
N, coverage показывает её. Selected group состав после этого неизменен.

Выбранные snippets сортируются по source/snippet position; connected components
по cosine >=0.8 образуют группы. Это transitive grouping, не требование
попарного similarity всех members. Summarization fan-in 5.

Для `llmtf-v1` group description и final section generation должны сделать
минимум один model call даже для singleton. Это исправление bypass в upstream
и явная смена baseline. Intermediate singleton уже сгенерированной summary
можно переносить только по declared topology. Final section cap 512.
Empty/malformed final prediction получает ноль и остаётся в denominator.
Transport/context/encoder failure не превращается в ноль и блокирует total.

Scoring каждой eligible section: sentence embedding PRF, ROUGE-L, sentence
BLEU с razdel tokenization и NLTK smoothing method1, pinned defaults/weights.
First reduce — mean scalar scores секций статьи; second — mean статей.
Section-micro допускается как отдельная diagnostic metric. Primary F1 — mean
section F1 → mean articles, не F1 от усреднённых P/R. Одна статья — один
checkpoint с полным expected section membership.

## 7. Общая aggregation и failure policy

Point estimate — arithmetic mean независимых book/article scores. CI:
предлагается percentile bootstrap, 2000 resamples, confidence 0.95, seed 555.
Для sections resample выбирает статьи со всеми их sections и сохраняет
multiplicity. Для N=1 CI=null с причиной `insufficient_samples`; point estimate
остаётся. Для N=0 successful total отсутствует. Это отличается от upstream
bootstrap median под именем mean и flatten секций.

Ошибки model format отражаются в scorer status и не уменьшают denominator.
Infrastructure failure обязательного sample/scorer делает task failed с
non-zero exit code. Можно сохранить completed raw checkpoints и partial
diagnostics, но не публиковать mean по оставшимся как официальный score.
Нельзя продвигать data exclusions после просмотра model outputs.

## 8. Исполнение, provenance и resources

Использовать T1–T3 из task/eval v2: bounded `AwaitMany`, один dispatcher,
keyed responses, stage-local budgets, resources и atomic completion.
Временные config scopes на одном LLM выполняются последовательно; task не
воспроизводит stop/reasoning/continuation rules.

Thinking request применяется ко всем primary model calls. Effective fallback
hybrid при нехватке бюджета сохраняет framework semantics, но записывается
для каждого invocation; такой run нельзя обозначать как все стадии thinking-on.
Strict reasoning без достаточного бюджета падает. Encoders не получают
LLM thinking flags. Multiple sequences пока отклоняются.

Protocol manifest содержит: schema/version, reference commits и repair/change
ledger, dataset content/revision/selection, tokenizer/encoder revisions,
prompt/parser/scorer/reducer/helper digests, graph/controller version,
stage configs, probability semantics, bootstrap и все algorithm bounds.
Run manifest добавляет deployment/model revisions (unknown явно), runtime,
effective configs, resource placement, batch/queue/deadline и selected-id digest.
Динамические graph nodes и фактические configs сохраняются в trace/completion
manifest; их outputs не нужны заранее для вычисления expected run fingerprint.

Runtime image: optional `external-benchmarks` dependency layer на validated HF
profile, пригодный для remote API primary; vLLM вариант расширяет validated
vLLM profile с тем же dependency overlay. Не копировать pins model stack из
upstream requirements и не менять torch-free базовый API image.

Encoder placement: CPU по явной настройке либо отдельная GPU/подтверждённое
совместное размещение. Сначала измерить memory/time. Не выводить безопасность
совместного запуска encoder и local vLLM из успеха remote API run.
Нужны encoder batch size и max active raw cases отдельно от model batch size;
новый task-specific concurrency/thread pool не нужен.

Общие options: versioned `resources`/`execution` mapping в Python API и YAML
и эквивалентный JSON mapping в обеих CLI. Конкретные названия flags фиксируются
T1 contract tests; неизвестные ключи запрещены. Dataset variants/algorithm
thresholds первоначально остаются fixed registry config. Cache root задаётся
явно/через общий cache service. Все три runners передают эти настройки в
subprocesses и включают sanitized effective settings в provenance.

## 9. Этапы поставки и критерии приёмки

| Этап | Зависимости | Проверяемый результат |
|---|---|---|
| E0 Inventory и freeze | Исходные snapshots | Полные data/resource manifests, change ledger, права на перенос, fixtures |
| E1 Общий runtime | T0–T3, synthetic fixtures; не ждёт лицензирования upstream materials | Generate/proba + encoder dispatch, dynamic fan-out, checkpoints, budgets, чистый report |
| E2 Hierarchical | E0 для books, E1 | Одна книга через CLI HF/vLLM/API; trace всех calls, metrics/replay, resume; затем фиксированные 8 книг |
| E3 Wiki ranking | E0 для Wiki, E1 | Full-corpus preparation при N=1, точные toy metrics, coverage gates, поддержанный API smoke; затем 8 статей |
| E4 Outline и sections | Wiki preparation, E1 | Oracle contract и eligibility фиксированы; singleton и missing-section fixtures; 8 статей каждого task |
| E5 Filtered и Blueprint | E2, dynamic encoder calls | Малые Q, bounded fan-out, string blueprint, deterministic grouping; 8 книг каждого variant |
| E6 Suite и release | E2–E5 | Все семь ids, single-model CLI и три runners, profile validation, docs, полный воспроизводимый report |
| E7 Baseline comparison | Frozen protocols и доступный reference runtime | Paired replay/live report; отдельное основание для baseline-compatibility claim |

E3 идёт раньше сложных Blueprint variants: он проверяет второй тип primitive и
наиболее рискованное API assumption. E4/E5 могут разрабатываться независимо
после общего ядра; последовательность не является разрешением параллельно
писать в одни artifacts. Исполнительный код добавляется небольшими проверяемыми
PR, не одним семизадачным переносом.

Пример будущего manual acceptance для каждого id: явно few-shot 0, fixed
sample manifest и thinking mode; первая попытка завершается, идентичная даёт
cache hit; interrupted attempt восстанавливает completed raw records;
изменение prompt/encoder/corpus hash не переиспользует старый cache.

## 10. Тестовая стратегия

### Offline и contract

- Books: token ranges/decoding; filter-before-limit; merge topology для
  1/2/3/4/5/6/7/12 chunks; duplicate filtering chain; Blueprint Q=0/1/many,
  malformed questions, compression branch, list-vs-string regression.
- Wiki: full corpus при evaluation N=1 и N=8 одинаков; title collisions,
  repeated headings и noncontiguous source ids; reordered embeddings;
  BM25 ties/k>corpus; oracle accesses видны; singleton выполняет generation.
- Probability: один/оба кандидата отсутствуют; eligible surfaces multi-token;
  top-k lower bounds; NaN/zero denominator; stable ranking; unknown tokenizer.
- Метрики: hand-computable embedding matrices, negative similarities, empty
  predictions, known ROUGE/BLEU strings, retrieved-pool vs corpus denominators,
  article macro vs section micro; group bootstrap с повторённой статьёй.
- Runtime: разные stage configs и exception restoration, out-of-order response
  alignment, encoder reused, bounded queue, unknown usage/deadline, no nested
  backend retries, invalid prediction vs infrastructure error.
- Artifacts: kill до/после raw checkpoint и до total, truncated events,
  corrupt/hash-mismatched checkpoint, duplicated writer, force-recalc stale
  total, offline report без model/encoder, credentials sanitization.
- Selection/reporting: прежний состав `all`, explicit opt-in новых ids,
  отсутствие смешивания experimental totals с основным benchmark score.

Synthetic fixtures создаются самостоятельно. Reference trace fixtures
используются только с понятным provenance и условиями хранения; огромные
тексты datasets в tests не копируются.

### Реальные проверки

Сначала inspect environment и Docker GPU passthrough по AGENTS.md. Для каждого
нового id — one-sample smoke на HF, local vLLM и проверенном API deployment,
затем фиксированные 8 **raw books/articles**, а не 8 model requests.
Сохраняются raw ids, число calls, prompts/outputs, coverage, время, memory,
usage с unknown fields, params, totals и complete checkpoint manifests.

Основная матрица — `Qwen/Qwen3.5-2B`, hybrid off/on, все семь tasks на трёх
backend paths, где capabilities подтверждены. Strict reasoning — отдельный
generate workflow и смешанный generate/probability workflow на каждом backend.
Base/foundational — hierarchical и ranking, где continuation verified;
общая гарантия для непроверенных whitespace prefills не заявляется.
Отсутствующий probability contract API — ожидаемый unsupported/non-zero cell,
а не основание объявить весь API path поддержанным. PPL для новых workflow
tasks не определён: ранняя capability error.

Если меняется LLM/budget/probability core, обязательны старые dependency-free
checks и применимые v4 cells, включая HF PPL и неподдерживаемые PPL modes.
Перед общей runtime-safety формулировкой собрать и проверить API/HF/vLLM
profiles по AGENTS.md, плюс optional encoder overlays. Отдельно проверить
torch-free API imports и отсутствие external optional imports для обычных tasks.

После small matrix и измерения стоимости — полный выбранный book subset и
полный eligible Wiki set в согласованной эталонной configuration. Не требовать
семь full datasets на каждой backend/thinking комбинации только ради галочки.
Стоимость полного run сначала оценить по числу chunks/candidates/questions и
измеренному latency/usage; raw sample limit её не ограничивает.

## 11. Reference comparison

Reference code допускается чинить в поддерживаемой ветке внешнего проекта
или fork, с отдельными commits и regression tests. Для сравнения старого и
нового используются отдельные checkouts; scratch checkout — инструмент
проверки, не единственное место хранения исправлений. Syntax/import/single-sample
repairs отделяются от algorithm changes. Поскольку original snapshot не всегда
исполним, такой результат называется
`reference-<commit>-repaired-<digest>`, а не «официальный baseline».

Сравнивать слоями: selected raw ids → chunks/snippets → ordered prompts и
invocation topology → replay outputs → per-unit scores → aggregates.
Для backend сравнения применять fixed model/tokenizer snapshots/configs.
Stochastic live outputs не обязаны совпадать побитово; replay локализует
algorithm/metric differences без повторного inference.

Обязательный ledger кандидатов отличий v1: chunk tail fix, Blueprint shape/
degenerate clustering, Wiki singleton generation, next-token ratio ranking,
явная normalization, deterministic ordering, failure/eligibility denominator,
article-level aggregation/CI, sampling defaults и framework thinking policy.
По каждой строке ledger записывается решение: сохранено исходное поведение,
совместимый repair или отдельная изменённая protocol version. Этот перечень
не означает, что все перечисленные изменения обязательны для интеграции.
Никакой усредняющей «поправки» для перевода новых scores в старые не вводится.

## 12. Изменяемые области и завершение

Планируемые модули: общий execution/requests/resources/artifacts слой;
evaluator integration; provenance и logger sidecars; lazy registry factories;
отдельные task packages для books/wiki и их pure reducers; shared preparation
service; strict runner config; offline reporting; optional dependency overlays;
fixtures/tests и отдельный experimental YAML suite.

Не использовать `llmtf/tasks/rubooksum.py` как скрытый generic scheduler.
Не делать encoder загрузку в constructor и не вычислять metrics внутри
`aggregation`. Не запускать внешние scripts из evaluator subprocess как
замену task integration.

Интеграция завершена, когда все семь tasks имеют frozen manifests, documented
supported configurations, complete standard artifacts, проверяемые resume/
cache/failure semantics и воспроизводимый validation report. Доступность через
Python API без CLIs/runners/reporting не закрывает работу. До включения в
основной leaderboard нужны отдельное решение о составе/весах и новый baseline.

Ближайшая реализуемая работа — E0 inventory вместе с synthetic T0–T1 prototype.
Публичные defaults по неустановленным model revisions, resource limits и правам
фиксируются только по их результатам; менять основные benchmark YAML заранее
не следует.
