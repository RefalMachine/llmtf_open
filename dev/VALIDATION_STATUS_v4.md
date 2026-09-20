# Validation status: refactor v4

> Snapshot-specific development status. Current product behavior is documented
> in `../docs/`.

Последнее обновление: 2026-09-20. Проверялся refactor snapshot поверх commit
`504bd7fc2793c900e97010f124651460afcd4812`. Он стал основой `v0.3.0`, который
выпускается обычным коммитом в `main` без отдельного Git tag.

## Итог

В проверенной матрице refactor стабилен: функциональной деградации в
non-reasoning контролях относительно legacy не обнаружено. Расхождения instruct
объясняются исправлением exact assistant-prefill и candidate surface forms.
Новый reasoning подтверждён независимо от исторически неисправного legacy path.

Это утверждение ограничено перечисленными моделями, задачами и runtime. Exact
local/API parity для whitespace-prefill и Base API generation пока не заявлена.

## Проверено

- 26 dependency-free logic checks; финальный release gate содержит 57
  unittest (56 проходят на минимальном host, один dataset runtime test там
  пропускается и проходит в API image);
- `compileall` и `git diff --check`;
- API image без torch/vLLM/CUDA;
- HF и vLLM images с CUDA passthrough на NVIDIA GeForce RTX 4090;
- torch 2.11.0+cu129, CUDA 12.9, Transformers 5.9.0, vLLM 0.21.0;
- реальные HF/local-vLLM/API generate и token-probability runs;
- HF-only PPL и явный unsupported result для vLLM/API;
- hybrid thinking off/on и отдельный strict reasoning contract;
- foundational Base runs;
- cache/provenance, partial batch failure и API capability failures;
- legacy vLLM non-reasoning comparison.

Полная refactor matrix содержит 26 успешных artifact sets и 208 sample records:
10 HF, 8 local vLLM и 8 API. Все sets содержат params/total sidecars с
совпадающим fingerprint. Подробные метрики и image id находятся в
[`TEST_REPORT_v4.md`](TEST_REPORT_v4.md).

## Legacy comparison

На том же model cache прошли четыре aligned vLLM controls: instruct
thinking-off generate/probability и Base generate/probability. Base probability
совпал полностью, Base generation — 7/8 predictions. Legacy instruct
probability показал известную candidate-id collision, исправленную в refactor.

Legacy reasoning не используется как oracle: обе thinking-enabled команды
свели reasoning budget к нулю из-за двойной семантики `max_prompt_len` в старом
CLI. По решению проекта legacy implementation не исправлялся ради сравнения.

## Известные ограничения

- vLLM 0.21 API удаляет trailing whitespace при assistant continuation.
  Affected instruct API runs выполнены как `best_effort`, а не exact parity.
- Base API не получает foundational stop strings единообразно с local
  backends; exact generation совпала только в 3/8 samples.
- Native provider `reasoning_content`/tool-calling response contract пока не
  реализован.
- Generic закрытый OpenAI-compatible provider отдельно не проверялся; API
  unit-тесты подтверждают conservative transport, но не поведение конкретного
  внешнего сервиса.

Эти пункты находятся в [`BACKLOG.md`](../BACKLOG.md). Численный детерминизм
между разными vLLM engine settings имеет низкий приоритет.

## Контейнерный запуск

Проверяйте GPU внутри target container через `docker run --gpus all`.
Sandbox-shell может не видеть driver. Для scripted checks используйте
`bash -c`, поскольку login shell способен заменить `/opt/venv/bin` в `PATH`.
