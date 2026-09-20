# Assistant continuation и prefill

## Термины и основной контракт

`assistant prefill` — последнее сообщение с ролью `assistant`, текст которого
уже находится во входе модели и должен быть продолжен. Например:

```json
{"role": "assistant", "content": "Ответ: "}
```

В LLMTF действуют следующие правила:

- роль внутри framework, backend'ов и результатов всегда называется
  `assistant`; историческая роль `bot` принимается только на границе загрузки
  старых task-данных и немедленно нормализуется;
- `predict` содержит только текст, реально сгенерированный моделью;
- prefill не добавляется в `predict`;
- полный assistant-ответ при необходимости собирается как
  `prefill + predict`;
- framework не применяет к prefill неявный `strip()` или `rstrip()`.

Вся политика находится в [`llmtf/continuation.py`](../llmtf/continuation.py).
Backend'ы не должны реализовывать собственные варианты этой логики.

Для two-pass reasoning reasoning-текст сохраняется отдельно в
`info.reasoning.text`. Вторая фаза получает его как часть assistant-prefill,
но итоговый `predict` по-прежнему содержит только вновь сгенерированный ответ.
Prefill и reasoning не примешиваются к `predict` постобработкой.

## Два независимых риска API

Первый риск фундаментальный: стандартный chat API может воспринимать последнее
сообщение `assistant` как завершённую реплику истории и начать новый assistant
turn. Поэтому даже prefill `"Ответ:"` нельзя считать переносимым, пока API не
объявляет поддержку continuation.

Второй риск появляется уже у API с continuation: хвостовой whitespace может
измениться при рендеринге chat template.

Распространённая реализация `continue_final_message` сначала ищет в
отрендеренном prompt очищенный текст последнего сообщения. Из-за этого
`"Ответ: "` может фактически превратиться в `"Ответ:"`, а
`"Перевод:\n"` — в `"Перевод:"`.

В таком случае первый пробел или перевод строки уже генерирует модель. Он
правильно появляется в raw `predict`, но фактический prompt отличается от
запрошенного, а простая сборка `prefill + predict` может удвоить разделитель.

Проблема `rstrip` возникает только когда продолженное последнее
assistant-сообщение заканчивается whitespace (обычным или Unicode-пробелом,
переводом строки, табом и так далее). Внутренние пробелы и prefill, который
заканчивается непробельным символом, этому риску не подвержены. Отдельный
пограничный случай — пустой или состоящий только из whitespace prefill. Но это
не отменяет первый риск: сам continuation должен поддерживаться API.

## Политики

Важно не смешивать две независимые настройки:

- `api_profile` определяет транспорт и допустимые поля HTTP payload;
- `assistant_prefill_policy` определяет требуемый уровень достоверности
  continuation.

Одинаковое слово `auto` в этих настройках не означает автоматическое
угадывание возможностей сервера. `api_profile=auto` намеренно остаётся
консервативным и не включает vLLM extensions даже при наличии `/tokenize`.

Политика задаётся опцией:

```text
--assistant_prefill_policy {auto,exact,portable,best_effort}
```

### `auto` (по умолчанию)

- Локальные HF и vLLM сохраняют prefill точно.
- Для API требуется профиль с объявленной поддержкой assistant continuation.
- Для такого API с опасным suffix требуется успешная проверка через
  `--probe_api_prefill`; без неё запуск останавливается до получения метрики.
- Консервативные профили `auto` и `openai` не объявляют continuation: задача
  должна заканчиваться сообщением `user` или использовать `best_effort`.
- Token-level подтверждения достаточно для `auto`, даже если API не позволяет
  восстановить prompt побайтово.

### `exact`

Для опасного whitespace-suffix требуется сильное подтверждение точного prompt.
Локальные backend'ы дают такую гарантию. Для API probe должен позволить
восстановить prompt и подтвердить точный suffix. Простого изменения token count
или token IDs недостаточно. Prefill без хвостового whitespace не запускает
этот специализированный probe: он не подвержен рассматриваемому `rstrip`-риску.

### `portable`

Любой assistant prefill запрещён, в том числе `"Ответ:"` и `"Ответ: ["`.
Запрос должен заканчиваться сообщением `user`, которое явно требует вывести
только ответ. Эта политика полезна для benchmark'ов, которые должны одинаково
запускаться через разнородные закрытые API. Для сравнимости один и тот же
portable prompt следует использовать также на HF и локальном vLLM.
Framework не переписывает такой prompt автоматически: изменение формулировки
меняет эксперимент и должно быть явно внесено и проверено автором task.

### `best_effort`

Framework передаёт сообщения серверу без гарантии continuation и один раз
предупреждает о каждом suffix. Результат помечается как `unverified`. Этот
режим нужно включать явно; он предназначен для осознанной диагностики, а не
для доказательства parity.

## Точный локальный рендеринг

HF и локальный vLLM используют одну реализацию. После полного prefill временно
ставится уникальная внутренняя граница, рендерится завершённый assistant-turn,
после чего prompt обрезается непосредственно перед границей. Граница и
end-of-turn suffix никогда не передаются модели. Исходные сообщения не
изменяются.

Если chat template изменяет или удаляет внутреннюю границу, framework
останавливает запуск вместо перехода к неточному prompt.

## API probe

Probe включается только явно:

```text
--probe_api_prefill
```

Он применим только к профилю, который уже объявляет continuation (сейчас
`vllm`), и запускается лениво: только когда реальный task использует
assistant-prefill с хвостовым whitespace. Для каждого suffix и режима шаблона результат
кешируется на время процесса. В probe используются синтетические сообщения;
данные датасета и credentials в диагностику не попадают.

Проверка сравнивает `/tokenize` для синтетического prefill с suffix и без него:

- одинаковые tokens/count — `verified_stripped`;
- разные tokens/count — `verified_token_effect`;
- token IDs плюс успешный `/detokenize`, восстановивший точный конец prompt —
  `verified_exact`;
- недостаточный ответ или недоступный endpoint — `inconclusive`/`unsupported`.

Ответы модели не сравниваются: одинаковая генерация не доказывает одинаковый
prompt или сам факт continuation. Для полностью закрытого black-box API
точность принципиально нельзя доказать на клиенте; остаются portable prompt
без assistant-prefill или явно выбранный `best_effort`.

В целевом vLLM 0.21 targeted probe уже показал удаление хвостового whitespace
при `continue_final_message`. Поэтому affected prompt нельзя считать parity-run
между local backend и API. Это наблюдение не заменяет полную API-матрицу:
каждый runtime/image всё равно должен пройти probe и сохранить его результат.

## `A` и ` A`: варианты поверхности следующего токена

Это отдельная проблема, не ограниченная trailing whitespace. Многие
tokenizer'ы представляют начало нового слова токеном с ведущим пробелом.
Поэтому семантическая метка `A` может встретиться как следующий токен `"A"`
или `" A"`.

Для `calculate_tokens_proba` LLMTF централизованно строит обе формы:

```python
candidate_surface_forms("A") == ("A", " A")
candidate_surface_forms(" A") == (" A", "A")
```

Локальные backend'ы учитывают только формы, которые действительно состоят из
одного токена. Если ни одной однотокенной формы нет, framework выдаёт ошибку,
а не использует вероятность первого токена многотокенной строки. API сопоставляет
обе декодированные формы с возвращёнными logprobs.

Для совместимости вероятность семантической метки пока равна максимуму
вероятностей её форм. Это записывается как
`candidate_surface_form_aggregation: "max"`. Изменение на сумму будет
изменением определения метрики и должно сопровождаться версионированием.

Raw generation не очищается. Task parser может интерпретировать `"A"` и
`" A"` как одну метку, но `predict` сохраняет исходный текст.

## Краткая таблица решений

| Сценарий | Поведение |
|---|---|
| HF/local vLLM + prefill | Точное локальное продолжение |
| Generic API + user-final prompt | Обычная генерация без tokenizer requirement |
| Generic API + prefill, `auto`/`exact` | Ошибка до генерации |
| vLLM API + prefill без хвостового whitespace | Continuation разрешён профилем |
| vLLM API + whitespace-prefill, `auto` | Требуется успешный token-effect probe |
| vLLM API + whitespace-prefill, `exact` | Требуется exact round-trip подтверждение |
| Любой backend + `portable` + prefill | Ошибка; task должен быть переписан явно |
| API prefill, принятый через `best_effort` | Запуск разрешён, результат отмечен `unverified` |

## Как писать переносимые prompts

Переносимый prompt не должен использовать незавершённый assistant turn.
Например, вместо:

```text
Ответ:␠
```

Здесь `␠` обозначает один хвостовой пробел и не является частью prompt.

задача заканчивается `user`-инструкцией:

```text
Выведите только одну букву A–D. Первым символом ответа должна быть эта буква.
```

Parser задачи должен принимать допустимые surface-формы метки. Для перевода
следует попросить вывести только перевод и не добавлять assistant-prefill.

Нельзя считать, что `max_tokens=1` означает один символ: это один токен
конкретного tokenizer. На API без доступа к tokenizer такую гарантию проверить
невозможно.

## Результаты и воспроизводимость

`_params.jsonl` и fingerprint содержат глобальную конфигурацию:

```json
"_llmtf_continuation": {
  "prefill_policy": "auto",
  "probe_api_prefill": true,
  "token_probability_surface_forms": "candidate_and_single_leading_space",
  "token_probability_aggregation": "max"
}
```

Per-sample `info.assistant_prefill` (либо
`info.response.assistant_prefill` для two-pass reasoning) содержит:

- наличие prefill;
- escaped trailing whitespace, например `"\\n"`;
- выбранную политику;
- фактическую обработку;
- уровень проверки и краткое evidence от API probe.

Prompt и `predict` нужно проверять вместе с этим блоком. Для локальных
backend'ов сохранённый prompt должен заканчиваться точным prefill; для API
prompt остаётся списком сообщений, а уровень доверия отражён в `info`.

## Конфигурация benchmark YAML

Параметры задаются в секции `model` и передаются всем подпроцессам:

```yaml
model:
  api_profile: vllm
  assistant_prefill_policy: auto
  probe_api_prefill: true
```

`probe_api_prefill` влияет только на API entrypoint. Локальные backend'ы
всегда используют общую точную реализацию, если политика не равна `portable`.

Профиль API задаётся отдельно через `--api_profile {auto,openai,vllm}` или
`model.api_profile` в benchmark YAML. Профили `auto` и `openai` используют
консервативный OpenAI payload без continuation; `vllm` включает
`continue_final_message` и другие vLLM-расширения. Подробнее см.
[`api_backend.md`](api_backend.md).

## Требования к изменениям

При изменении continuation-логики необходимо проверить как минимум:

```bash
python3 tests/test_refactor_logic.py
python3 -m unittest discover -s tests -p 'test_*.py'
python3 -m compileall -q llmtf evaluate_model.py evaluate_model_api.py benchmark
git diff --check
```

Runtime matrix должна отдельно покрывать HF, локальный vLLM и API для prefill
без suffix, с обычным пробелом и с переводом строки, а также token-probability
для `A`/` A`.
