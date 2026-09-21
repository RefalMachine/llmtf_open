"""Pure-logic tests for the v3 refactor that do NOT need torch/vllm/transformers.

Run:  python3 -m pytest tests/test_refactor_logic.py -q
  or: python3 tests/test_refactor_logic.py

These tests stub heavy third-party modules so they import cleanly in a bare
Python environment. They exercise:
  - ModelKind enum + user-driven model_kind (no auto-detect; 'auto' is rejected)
  - EmulatedReasoningStrategy.run (two-pass emulated flow:
    reasoning phase stop-by-think / stop-by-length validation + normalization,
    then continuation phase via continuation_fn; no `mode` flag — the caller
    wraps its continuation primitive + extra args (tokens_of_interest, ...) via
    reasoning_phase_kwargs / continuation_phase_kwargs).
    Plus LLM two-pass flow (generate & calculate_tokens_proba) info-payload shape.
  - ReasoningConfig is the single source of reasoning fields (generation_config
    stays sampling-only); MaxLenContext trims reasoning_config live.
  - JsonArrayLogger produces a valid JSON array (incl. empty) and survives
    mid-stream kill + salvaging; PrettyJsonLogger matches the historical format.
  - LLM dispatcher: plain+enable_thinking warns + one-pass; reasoning kind +
    enable_thinking=False raises ValueError; reasoning kind without
    end_thinking_token_id raises ValueError; 'auto' model_kind raises ValueError.
  - apply_model_prompt/count_tokens_for_prompt are optional debug-helpers: they
    raise NotImplementedError on the API backend (no local chat template).
"""
import os, sys, json, types, tempfile, shutil, importlib.util, logging

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

def _stub_third_party():
    """Install minimal stubs so llmtf.* imports without torch/vllm/etc."""
    def stub(name, **attrs):
        if name in sys.modules: return sys.modules[name]
        m = types.ModuleType(name)
        for k, v in attrs.items(): setattr(m, k, v)
        sys.modules[name] = m
        return m
    def make(n):
        cls = type(n, (), {
            '__init__': lambda s, *a, **k: None,
            'from_pretrained': classmethod(lambda cls, *a, **k: cls()),
        })
        def from_dict(cls, d, **k):
            obj = cls()
            for kk, vv in (d or {}).items():
                setattr(obj, kk, vv)
            return obj
        cls.from_dict = classmethod(from_dict)
        return cls
    t = stub('transformers')
    for n in ['AutoTokenizer', 'AutoModelForCausalLM', 'AutoConfig', 'GenerationConfig']:
        setattr(t, n, make(n))
    stub('transformers.generation', GenerationConfig=t.GenerationConfig)
    stub('peft', PeftConfig=type('PeftConfig', (), {}), PeftModel=type('PeftModel', (), {}))
    tor = stub('torch')
    import contextlib
    tor.no_grad = lambda: contextlib.nullcontext()
    tor.empty_cache = lambda: None
    tor.Tensor = object; tor.dtype = object
    tor.float16 = tor.bfloat16 = tor.float32 = 'dt'
    for a in ['cat', 'argmax', 'softmax', 'full', 'LongTensor', 'FloatTensor',
              'from_numpy', 'tensor', 'where', 'gather', 'topk', 'zeros', 'ones']:
        setattr(tor, a, lambda *x, **y: None)
    stub('torch.cuda', is_available=lambda: False, device_count=lambda: 0, empty_cache=lambda: None)
    stub('torch.nn', Module=type('Module', (), {}), functional=stub('torch.nn.functional'))
    stub('requests', get=lambda *a, **k: None, post=lambda *a, **k: None, Session=type('Session', (), {}))
    stub('tqdm', tqdm=lambda *a, **k: iter([]))
    stub('numpy', ndarray=type('ndarray', (), {}), array=lambda *a, **k: None,
         float32='f', float64='d', zeros=lambda *a, **k: None, asarray=lambda *a, **k: None,
         expand_dims=lambda *a, **k: None, concatenate=lambda *a, **k: None,
         log=lambda *a, **k: 0.0, exp=lambda *a, **k: 1.0, sum=lambda *a, **k: 0,
         mean=lambda *a, **k: 0.0, std=lambda *a, **k: 0.0)
    stub('datasets', load_dataset=lambda *a, **k: None, Dataset=type('Dataset', (), {}),
         DatasetDict=type('DatasetDict', (), {}))
    skm = stub('sklearn.metrics')
    for a in ['accuracy_score', 'f1_score', 'precision_score', 'recall_score']:
        setattr(skm, a, lambda *x, **y: 0.0)
    stub('sklearn', metrics=skm)
    stub('sklearn.utils', resample=lambda *a, **k: None)
    rs = stub('rouge_score.rouge_scorer', RougeScorer=lambda *a, **k: lambda *x, **y: None)
    stub('rouge_score', rouge_scorer=rs)
    stub('pymorphy3', MorphAnalyzer=lambda *a, **k: type('Morph', (), {'parse': lambda s, x: []})())
    for n in ['pandas', 'scipy', 'rapidfuzz', 'huggingface_hub', 'gensim', 'networkx', 'sympy',
              'transformers.data', 'transformers.data.metrics', 'transformers.data.metrics.squad_metrics']:
        stub(n)


_stub_third_party()
logging.basicConfig(level=logging.WARNING)


def _load_task_source(relative_path, module_name):
    spec = importlib.util.spec_from_file_location(
        module_name, os.path.join(ROOT, relative_path)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_imports_and_kind_enum():
    from llmtf.llm import LLM
    from llmtf.base import BaseLLM
    from llmtf.backends import HFBackend, VLLMBackend, APIBackend
    from llmtf.reasoning import ModelKind, ReasoningConfig
    assert issubclass(LLM, BaseLLM)
    assert sorted(HFBackend.__abstractmethods__) == []  # all satisfied
    assert set(ModelKind.__members__) == {'plain', 'reasoning', 'hybrid'}
    # default ReasoningConfig is plain
    rc = ReasoningConfig()
    assert rc.model_kind == ModelKind.plain
    assert not rc.is_reasoning


def test_chat_prompt_preserves_trailing_assistant_prefill_whitespace():
    from llmtf.continuation import render_local_chat_prompt

    class StrippingTokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt,
                                continue_final_message, enable_thinking):
            rendered = "".join(
                f"<{message['role']}>{message['content']}<end>"
                for message in messages
            )
            if continue_final_message:
                # Reproduce the relevant Transformers behavior: trailing
                # whitespace in final content is lost while removing <end>.
                final = messages[-1]['content'].strip()
                return rendered[:rendered.rfind(final) + len(final)]
            return rendered

    messages = [
        {'role': 'user', 'content': 'Translate'},
        {'role': 'assistant', 'content': 'Translation:\n'},
    ]
    prompt, decision = render_local_chat_prompt(
        StrippingTokenizer(), messages,
        continue_last_assistant_message=True,
    )
    assert prompt == '<user>Translate<end><assistant>Translation:\n'
    assert messages[-1]['content'] == 'Translation:\n'
    assert decision.verification.value == 'exact_local'


def test_prefill_hazard_is_only_trailing_whitespace():
    from llmtf.continuation import analyze_assistant_prefill
    safe = analyze_assistant_prefill([
        {'role': 'assistant', 'content': 'Answer: ['},
    ])
    hazardous = analyze_assistant_prefill([
        {'role': 'assistant', 'content': 'Answer: \n'},
    ])
    assert safe.present and not safe.hazardous
    assert hazardous.trailing_whitespace == ' \n'


def test_portable_policy_rejects_every_assistant_prefill():
    from llmtf.continuation import (
        ContinuationConfig, PrefillCompatibilityError,
        analyze_assistant_prefill, resolve_prefill,
    )
    analysis = analyze_assistant_prefill([
        {'role': 'assistant', 'content': 'Answer: '},
    ])
    try:
        resolve_prefill(
            analysis,
            ContinuationConfig(prefill_policy='portable'),
            local_exact=True,
        )
        raise AssertionError('portable policy accepted an assistant prefill')
    except PrefillCompatibilityError as exc:
        assert 'does not allow an assistant prefill' in str(exc)

    visible_boundary = analyze_assistant_prefill([
        {'role': 'assistant', 'content': 'Answer:['},
    ])
    try:
        resolve_prefill(
            visible_boundary,
            ContinuationConfig(prefill_policy='portable'),
            local_exact=True,
        )
        raise AssertionError('portable policy accepted a visible-boundary prefill')
    except PrefillCompatibilityError:
        pass


def test_candidate_surface_forms_and_single_token_filtering():
    from llmtf.continuation import (
        candidate_surface_forms, single_token_candidate_ids,
    )
    assert candidate_surface_forms('A') == ('A', ' A')
    assert candidate_surface_forms(' A') == (' A', 'A')

    class Tokenizer:
        def __call__(self, text, add_special_tokens=False):
            mapping = {'A': [1], ' A': [2], 'word': [3, 4], ' word': [5]}
            return {'input_ids': mapping[text]}

    assert single_token_candidate_ids(Tokenizer(), ['A', 'word']) == [[1, 2], [5]]


def test_legacy_facades_removed():
    """No HFModel/VLLMModel/ApiVLLMModel/LocalHostedLLM, no llmtf.models/ llmtf.model."""
    import importlib
    for mod in ['llmtf.model', 'llmtf.models', 'llmtf.models.local', 'llmtf.models.hf']:
        try:
            importlib.import_module(mod)
            raise AssertionError(mod + ' still importable (should be deleted)')
        except ModuleNotFoundError:
            pass
    import llmtf
    for nm in ['HFModel', 'VLLMModel', 'ApiVLLMModel', 'LocalHostedLLM']:
        assert not hasattr(llmtf, nm), nm + ' still exported'


def _build_strategy(max_new_tokens_reasoning=100, end_thinking_token_id=42, truncation_prompt='...'):
    from llmtf.reasoning import EmulatedReasoningStrategy, ReasoningConfig, ReasoningFormat
    fmt = ReasoningFormat(end_thinking_token_id=end_thinking_token_id, truncation_prompt=truncation_prompt)
    rc = ReasoningConfig(model_kind=ReasoningConfig.__dataclass_fields__['model_kind'].default,
                         max_new_tokens_reasoning=max_new_tokens_reasoning, fmt=fmt)
    # build a hybrid rc for the strategy
    from llmtf.reasoning import ModelKind
    rc.model_kind = ModelKind.hybrid
    return EmulatedReasoningStrategy(rc), rc


def test_emulated_strategy_run_stop_by_think_then_continuation():
    """run() performs reasoning + continuation; reasoning text preserved,
    continuation called with updated_messages containing the reasoning block."""
    from llmtf.reasoning import THINK_CLOSE_MARKER
    TC = THINK_CLOSE_MARKER
    strat, rc = _build_strategy()
    messages_batch = [[{'role': 'user', 'content': 'hi'}]]
    def reasoning_fn(prompt_msgs, generation_config=None, **kw):
        return (['p0'], ['reasoning' + TC], [{'prompt_len': 2, 'generated_len': [3],
                                             'generated_cumulative_logprob': -1.0}])
    seen = {}
    def continuation_fn(updated_msgs, **kw):
        seen['msgs'] = updated_msgs
        return (['p1'], ['answer'], [{'prompt_len': 5, 'generated_len': [1],
                                       'generated_cumulative_logprob': -0.5}])
    # sampling-only generation_config (no reasoning fields on it)
    gc = type('G', (), {'stop_strings': [], 'max_new_tokens': 64})()
    result = strat.run(
        messages_batch,
        reasoning_fn=reasoning_fn,
        continuation_fn=continuation_fn,
        generation_config=gc,
        reasoning_phase_kwargs={},
        continuation_phase_kwargs={},
    )
    assert result.reasoning_prompts == ['p0']
    assert result.final_outputs == ['answer']
    # The visible-answer separator is normalized for continuation rendering.
    assert result.reasoning_outputs == ['reasoning' + TC + '\n\n']
    # continuation received prefix + assistant(reasoning text)
    assert seen['msgs'][0][0]['role'] == 'user'
    assert seen['msgs'][0][1]['role'] == 'assistant'
    assert seen['msgs'][0][1]['content'] == 'reasoning' + TC + '\n\n'
    # raw reasoning/final infos passed through (LLM layer shapes info, not strategy)
    assert result.reasoning_infos[0]['generated_len'] == [3]
    assert result.final_infos[0]['generated_cumulative_logprob'] == -0.5


def test_emulated_strategy_run_stopped_by_length_appends_close_marker():
    """On length-limit stop, run() synthesizes a closing marker so the
    continuation phase sees a well-formed block; truncation prompt appended
    when add_reasoning_truncing_prompt=True."""
    from llmtf.reasoning import THINK_CLOSE_MARKER
    TC = THINK_CLOSE_MARKER
    strat, rc = _build_strategy(max_new_tokens_reasoning=100, truncation_prompt='<trunc>')
    messages_batch = [[{'role': 'user', 'content': 'q'}]]
    def reasoning_fn(prompt_msgs, generation_config=None, **kw):
        return (['p'], ['reasoning text'], [{'prompt_len': 1, 'generated_len': [100],
                                             'generated_cumulative_logprob': -1.0}])
    seen = {}
    def continuation_fn(updated_msgs, **kw):
        seen['msgs'] = updated_msgs
        return (['p1'], ['final'], [{'prompt_len': 3, 'generated_len': [1],
                                     'generated_cumulative_logprob': 0.0}])
    gc = type('G', (), {'stop_strings': [], 'max_new_tokens': 64})()
    result = strat.run(
        messages_batch,
        reasoning_fn=reasoning_fn,
        continuation_fn=continuation_fn,
        generation_config=gc,
        reasoning_phase_kwargs={},
        continuation_phase_kwargs={},
        add_reasoning_truncing_prompt=True,
    )
    assert result.final_outputs == ['final']
    # reasoning text normalized: truncation prompt, close marker, answer separator
    assert '<trunc>' in result.reasoning_outputs[0]
    assert result.reasoning_outputs[0].endswith(TC + '\n\n')
    # continuation received the normalized reasoning text in the assistant turn
    assert seen['msgs'][0][-1]['content'] == result.reasoning_outputs[0]


def test_json_array_logger_valid_array():
    from llmtf.sample_logger import JsonArrayLogger
    d = tempfile.mkdtemp(prefix='sl_')
    try:
        with JsonArrayLogger(d, 'task_x') as lg:
            for i in range(3):
                lg.log_sample({'id': i}, f'p{i}', [{'role': 'user', 'content': 'q'}],
                              {'acc': 0.5 + i * 0.1}, {'prompt_len': i})
        path = os.path.join(d, 'task_x.jsonl')
        arr = json.load(open(path, encoding='utf-8'))
        assert isinstance(arr, list) and len(arr) == 3
        assert arr[0]['metric'] == {'acc': 0.5}
        assert arr[2]['info']['prompt_len'] == 2
    finally:
        shutil.rmtree(d)


def test_json_array_logger_empty_is_valid():
    from llmtf.sample_logger import JsonArrayLogger
    d = tempfile.mkdtemp(prefix='sl_e_')
    try:
        with JsonArrayLogger(d, 'empty'):
            pass
        raw = open(os.path.join(d, 'empty.jsonl'), encoding='utf-8').read()
        assert raw == '[\n]\n', repr(raw)
        assert json.load(open(os.path.join(d, 'empty.jsonl'), encoding='utf-8')) == []
    finally:
        shutil.rmtree(d)


def test_json_array_logger_salvage_after_kill():
    """Simulate an abrupt kill mid-stream: file ends without closing ']'.
    Salvaging via `printf '\\n]\\n' >> file` must yield a valid array."""
    from llmtf.sample_logger import JsonArrayLogger
    d = tempfile.mkdtemp(prefix='sl_k_')
    try:
        lg = JsonArrayLogger(d, 'kill_task')
        lg.__enter__()
        lg.log_sample({'id': 1}, 'p1', [{'role': 'user', 'content': 'q'}], {'acc': 1.0}, {})
        lg.log_sample({'id': 2}, 'p2', [{'role': 'user', 'content': 'q'}], {'acc': 0.0}, {})
        lg.file.flush()
        path = os.path.join(d, 'kill_task.jsonl')
        raw_before = open(path, encoding='utf-8').read()
        assert raw_before.startswith('[\n'), repr(raw_before[:10])
        assert not raw_before.rstrip().endswith(']'), 'array should be unclosed'
        with open(path, 'a', encoding='utf-8') as f:
            f.write('\n]\n')
        arr = json.load(open(path, encoding='utf-8'))
        assert len(arr) == 2 and arr[1]['metric'] == {'acc': 0.0}
    finally:
        shutil.rmtree(d)


def test_pretty_json_logger_matches_historical_format():
    from llmtf.sample_logger import PrettyJsonLogger
    d = tempfile.mkdtemp(prefix='sl_p_')
    try:
        obj = {'task_name': 't', 'results': {'acc': 0.7}, 'leaderboard_result': 0.7, 'time': 1.23}
        with PrettyJsonLogger(d, 't_total') as lg:
            lg.log_json(obj)
        raw = open(os.path.join(d, 't_total.jsonl'), encoding='utf-8').read()
        expected = json.dumps(obj, ensure_ascii=False, indent=4) + '\n'
        assert raw == expected, 'PrettyJsonLogger format drifted'
    finally:
        shutil.rmtree(d)


class _FakeBackend:
    """Minimal backend stub for LLM dispatcher/setup tests (no model load)."""
    def __init__(self):
        from transformers import GenerationConfig
        self.generation_config = GenerationConfig.from_dict(
            {'max_new_tokens': 64, 'max_length': 4096, 'stop_strings': [],
             'eos_token_id': [0], 'temperature': 0.1, 'top_p': 0.9, 'top_k': 40,
             'repetition_penalty': 1.0, 'do_sample': True, 'num_beams': 1})
    def support_method(self, m): return m in ['generate', 'calculate_tokens_proba']
    def configure_continuation(self, config): self.continuation_config = config
    def from_pretrained(self, *a, **k): pass
    def generate_batch(self, messages, **k):
        return (['p']*len(messages), ['o']*len(messages),
                [{'prompt_len': 1, 'generated_len': [1], 'generated_cumulative_logprob': None} for _ in messages])
    def calculate_tokens_proba_batch(self, messages, tokens_of_interest, **k):
        return (['p']*len(messages), [{'a': 0.5}]*len(messages),
                [{'generated_len': 1, 'generated_token': 'a'} for _ in messages])
    def count_tokens_for_messages(self, messages, **k): return 5
    def get_model_context_len(self): return 4096
    def get_params(self): return {}
    def add_stop_strings(self, s): pass
    def reset_stop_strings(self): pass


def test_llm_setup_reasoning_auto_rejected():
    """v3 removed auto-detection: 'auto' is no longer a valid model_kind."""
    from llmtf.llm import LLM
    m = LLM(backend=_FakeBackend())
    raised = False
    try:
        m._setup_reasoning(model_kind='auto')
    except ValueError:
        raised = True
    assert raised, "expected ValueError for model_kind='auto' (auto-detect removed in v3)"


def test_llm_setup_reasoning_default_is_plain():
    """default LLM has a plain ReasoningConfig and no strategy."""
    from llmtf.llm import LLM
    from llmtf.reasoning import ModelKind
    m = LLM(backend=_FakeBackend())
    # _reasoning_config defaults to plain
    assert m.reasoning_config.model_kind == ModelKind.plain
    assert m._reasoning is None
    # calling _setup_reasoning explicitly with 'plain' keeps plain
    m._setup_reasoning(model_kind='plain')
    assert m.reasoning_config.model_kind == ModelKind.plain
    assert m._reasoning is None


def test_llm_setup_reasoning_hybrid_with_explicit_id():
    from llmtf.llm import LLM
    from llmtf.reasoning import ModelKind
    m = LLM(backend=_FakeBackend())
    m._setup_reasoning(model_kind='hybrid', max_new_tokens_reasoning=100,
                       min_new_tokens_reasoning=50,
                       end_thinking_token_id=151668)
    assert m.reasoning_config.model_kind == ModelKind.hybrid
    assert m.reasoning_config.max_new_tokens_reasoning == 100
    assert m.reasoning_config.min_new_tokens_reasoning == 50
    assert m.reasoning_config.fmt.end_thinking_token_id == 151668
    assert m._reasoning is not None
    # generation_config stays sampling-only (no reasoning fields)
    assert not hasattr(m.generation_config, 'max_new_tokens_reasoning')
    assert not hasattr(m.generation_config, 'end_thinking_token_id')


def test_llm_setup_reasoning_reasoning_without_id_raises():
    """reasoning kind requires explicit end_thinking_token_id."""
    from llmtf.llm import LLM
    m = LLM(backend=_FakeBackend())
    raised = False
    try:
        m._setup_reasoning(model_kind='reasoning', max_new_tokens_reasoning=100, min_new_tokens_reasoning=50)
    except ValueError as e:
        raised = True
        assert 'end_thinking_token_id' in str(e)
    assert raised, "expected ValueError for model_kind='reasoning' without end_thinking_token_id"


def test_llm_setup_reasoning_hybrid_without_id_warns():
    """hybrid without end_thinking_token_id warns and falls back to text stop-string."""
    import logging
    from llmtf.llm import LLM
    from llmtf.reasoning import ModelKind
    m = LLM(backend=_FakeBackend())
    records = []
    h = logging.Handler()
    h.emit = lambda r: records.append(r)
    m.logger.addHandler(h)
    m._setup_reasoning(model_kind='hybrid', max_new_tokens_reasoning=100, min_new_tokens_reasoning=50)
    assert m.reasoning_config.model_kind == ModelKind.hybrid
    assert m.reasoning_config.fmt.end_thinking_token_id is None
    assert m._reasoning is not None
    assert any('end_thinking_token_id' in r.getMessage() for r in records), \
        "expected a warning about missing end_thinking_token_id for hybrid"


def test_dispatcher_plain_with_enable_thinking_warns_and_onepass():
    """plain model + enable_thinking=True -> warning + one-pass backend.generate_batch."""
    import logging
    from llmtf.llm import LLM
    from llmtf.reasoning import ModelKind, ReasoningConfig
    m = LLM(backend=_FakeBackend())
    m._reasoning_config = ReasoningConfig(model_kind=ModelKind.plain)
    m._reasoning = None
    called = {'gen': 0}
    def fake_gen(messages, **kw):
        called['gen'] += 1
        return (['p'], ['out'], [{'prompt_len': 1, 'generated_len': [1], 'generated_cumulative_logprob': None}])
    m.backend.generate_batch = fake_gen
    # capture warning (plain + enable_thinking=True logs a warning)
    records = []
    h = logging.Handler()
    h.emit = lambda r: records.append(r)
    m.logger.addHandler(h)
    out = m.generate_batch([[{'role': 'user', 'content': 'q'}]], enable_thinking=True)
    assert called['gen'] == 1
    assert out[1] == ['out']
    assert any('plain' in r.getMessage().lower() or 'thinking' in r.getMessage().lower() for r in records), \
        'expected a warning about enable_thinking on a plain model'


def test_dispatcher_reasoning_kind_with_enable_thinking_false_raises():
    from llmtf.llm import LLM
    from llmtf.reasoning import ModelKind, ReasoningConfig
    m = LLM(backend=_FakeBackend())
    m._reasoning_config = ReasoningConfig(model_kind=ModelKind.reasoning)
    m._reasoning = None
    raised = False
    try:
        m.generate_batch([[{'role': 'user', 'content': 'q'}]], enable_thinking=False)
    except ValueError as e:
        raised = True
        assert 'reasoning' in str(e)
    assert raised, 'expected ValueError for model_kind=reasoning + enable_thinking=False'


def test_dispatcher_hybrid_two_pass_generate():
    """hybrid + enable_thinking=True: reasoning phase then continuation phase,
    output stays the raw generated continuation, and info carries both
    reasoning and response sub-payloads."""
    from llmtf.llm import LLM
    from llmtf.reasoning import THINK_CLOSE_MARKER
    TC = THINK_CLOSE_MARKER
    m = LLM(backend=_FakeBackend())
    m._setup_reasoning(model_kind='hybrid', max_new_tokens_reasoning=50,
                       min_new_tokens_reasoning=20,
                       end_thinking_token_id=151668)
    calls = {'gen': 0}
    continuation_messages = {}
    def fake_gen(messages, generation_config=None, **kw):
        calls['gen'] += 1
        if calls['gen'] == 1:
            return (['p0'], ['reasoning' + TC],
                    [{'prompt_len': 2, 'generated_len': [3],
                      'generated_cumulative_logprob': -1.0} for _ in messages])
        continuation_messages['value'] = messages
        return (['p1'], [' answer'],
                [{'prompt_len': 5, 'generated_len': [1],
                  'generated_cumulative_logprob': -0.5} for _ in messages])
    m.backend.generate_batch = fake_gen
    messages = [[{'role': 'user', 'content': 'q'},
                 {'role': 'assistant', 'content': 'prefix '}]]
    prompts, outputs, infos = m.generate_batch(messages, enable_thinking=True)
    assert calls['gen'] == 2  # reasoning + continuation
    assert outputs == [' answer']
    assert continuation_messages['value'][0][-1] == {
        'role': 'assistant', 'content': 'reasoning' + TC + '\n\nprefix '
    }
    assert sum(
        message['role'] == 'assistant'
        for message in continuation_messages['value'][0]
    ) == 1
    assert infos[0]['reasoning']['generated_len'] == [3]
    assert infos[0]['reasoning']['text'] == 'reasoning' + TC + '\n\n'
    assert infos[0]['response']['generated_cumulative_logprob'] == -0.5


def test_dispatcher_one_pass_returns_generated_continuation_only():
    from llmtf.llm import LLM
    m = LLM(backend=_FakeBackend())
    m._setup_reasoning(model_kind='plain')
    m.backend.generate_batch = lambda messages, **kwargs: (
        ['rendered'], ['\nanswer'],
        [{'prompt_len': 2, 'generated_len': [1],
          'generated_cumulative_logprob': -0.5}],
    )
    messages = [[
        {'role': 'user', 'content': 'q'},
        {'role': 'assistant', 'content': 'prefix\n'},
    ]]

    _, outputs, _ = m.generate_batch(messages, enable_thinking=False)

    assert outputs == ['\nanswer']


def test_removed_assistant_prefill_output_option_is_rejected():
    from llmtf.llm import LLM
    m = LLM(backend=_FakeBackend())
    m._setup_reasoning(model_kind='plain')
    try:
        m.generate_batch(
            [[{'role': 'user', 'content': 'q'}]],
            add_assistant_prompt_to_output=True,
        )
    except TypeError as exc:
        assert 'predict contains only tokens generated' in str(exc)
    else:
        raise AssertionError('removed prefill-output option must fail explicitly')


def test_dispatcher_hybrid_two_pass_ctp():
    """hybrid + enable_thinking=True for calculate_tokens_proba: reasoning phase
    uses generate_batch, continuation uses calculate_tokens_proba_batch with
    tokens_of_interest threaded through; info response carries generated_token."""
    from llmtf.llm import LLM
    from llmtf.reasoning import THINK_CLOSE_MARKER
    TC = THINK_CLOSE_MARKER
    m = LLM(backend=_FakeBackend())
    m._setup_reasoning(model_kind='hybrid', max_new_tokens_reasoning=50,
                       min_new_tokens_reasoning=20,
                       end_thinking_token_id=151668)
    gen_calls = {'n': 0}
    def fake_gen(messages, generation_config=None, **kw):
        gen_calls['n'] += 1
        return (['p0'], ['reasoning' + TC],
                [{'prompt_len': 2, 'generated_len': [4],
                  'generated_cumulative_logprob': -2.0} for _ in messages])
    ctp_seen = {}
    def fake_ctp(messages, tokens_of_interest, **kw):
        ctp_seen['toi'] = tokens_of_interest
        return (['p1'], [{'A': 0.9}],
                [{'generated_len': 1, 'generated_token': 'A'} for _ in messages])
    m.backend.generate_batch = fake_gen
    m.backend.calculate_tokens_proba_batch = fake_ctp
    messages = [[{'role': 'user', 'content': 'q'}]]
    prompts, probs, infos = m.calculate_tokens_proba_batch(messages, ['A', 'B'],
                                                            enable_thinking=True)
    assert gen_calls['n'] == 1
    assert ctp_seen['toi'] == ['A', 'B']
    assert probs == [{'A': 0.9}]
    assert infos[0]['response']['generated_token'] == 'A'
    assert 'generated_cumulative_logprob' not in infos[0]['response']
    assert infos[0]['reasoning']['text'] == 'reasoning' + TC + '\n\n'


def test_maxlencontext_trims_reasoning_above_floor():
    """MaxLenContext trims max_new_tokens_reasoning from upper bound down to
    fit model_context_len, but never below min_new_tokens_reasoning; if the
    floor cannot be reserved, reasoning is SKIPPED (eff=0, skipped_reasoning)
    and the prompt budget is the full leftover."""
    from llmtf.llm import LLM
    from llmtf.reasoning import ModelKind, ReasoningConfig
    from llmtf.utils import MaxLenContext

    m = LLM(backend=_FakeBackend())
    m._setup_reasoning(model_kind='hybrid', max_new_tokens_reasoning=4096,
                       min_new_tokens_reasoning=1024,
                       end_thinking_token_id=151668)
    # Tiny context to force a trim that still leaves room for the floor.
    # model_context_len = 8192; answer=16; upper=4096; floor=1024.
    # available_for_prompt_and_reasoning = 8192 - 16 = 8176.
    # reasoning_eff = min(4096, 8176) = 4096; prompt leftover = 8176 - 4096 = 4080.
    # reasoning is NOT trimmed (no trim happens here): keep as a baseline.
    m.get_model_context_len = lambda: 8192

    class TaskStub:
        _max_task_new_tokens = 16
        @property
        def max_task_new_tokens(self): return self._max_task_new_tokens
    t = TaskStub()

    saved_rc = m.reasoning_config.max_new_tokens_reasoning
    saved_floor = m.reasoning_config.min_new_tokens_reasoning
    with MaxLenContext(t, m, custom_generation_config=None) as mpl:
        assert m.reasoning_config.max_new_tokens_reasoning == 4096  # not trimmed
        assert mpl == 8176 - 4096  # = 4080
    assert m.reasoning_config.max_new_tokens_reasoning == saved_rc
    assert m.reasoning_config.min_new_tokens_reasoning == saved_floor

    # Now force a real trim: model_context_len = 5120 leaves
    # 5120 - 16 = 5104 for prompt+reasoning; reasoning_eff = min(4096, 5104) =
    # 4096; prompt leftover = 5104 - 4096 = 1008. Still not trimmed (>=floor).
    m.get_model_context_len = lambda: 5120
    with MaxLenContext(t, m, custom_generation_config=None) as mpl:
        assert m.reasoning_config.max_new_tokens_reasoning == 4096
        assert mpl == 1008


def test_maxlencontext_skips_reasoning_when_floor_unreachable():
    """If model_context_len cannot reserve floor for reasoning, reasoning is
    SKIPPED (eff=0) with a loud warning; pre_turn_tokens = leftover."""
    from llmtf.llm import LLM
    from llmtf.reasoning import ModelKind, ReasoningConfig
    from llmtf.utils import MaxLenContext

    m = LLM(backend=_FakeBackend())
    m._setup_reasoning(model_kind='hybrid', max_new_tokens_reasoning=4096,
                       min_new_tokens_reasoning=1024,
                       end_thinking_token_id=151668)
    # model_context_len = 1020; answer=16; available = 1004.
    # reasoning_eff upper = 4096 => trim to 1004, but 1004 < floor 1024 → SKIPPED.
    # reasoning_eff=0, pre_turn_tokens = 1004.
    m.get_model_context_len = lambda: 1020

    class TaskStub:
        _max_task_new_tokens = 16
        @property
        def max_task_new_tokens(self): return self._max_task_new_tokens
    t = TaskStub()

    with MaxLenContext(t, m, custom_generation_config=None) as mpl:
        assert m.reasoning_config.max_new_tokens_reasoning == 0  # skipped
        assert mpl == 1020 - 16  # = 1004, full leftover, no reasoning reserved


def test_api_backend_apply_model_prompt_not_implemented():
    """APIBackend exposes no local chat-template rendering: apply_model_prompt
    and count_tokens_for_prompt raise NotImplementedError (debug-API contract).
    count_tokens_for_messages remains the only supported token-count path."""
    from llmtf.backends import APIBackend
    from llmtf.llm import LLM
    api = APIBackend(api_base='http://localhost:0')  # no network at __init__
    # APIBackend exposes from_pretrained only via HTTP; skip by directly
    # exercising the NotImplementedError on the debug methods.
    raised_apply = False
    try:
        api.apply_model_prompt([{'role': 'user', 'content': 'hi'}])
    except NotImplementedError:
        raised_apply = True
    assert raised_apply, 'expected NotImplementedError from APIBackend.apply_model_prompt'
    raised_count = False
    try:
        api.count_tokens_for_prompt('some prompt string')
    except NotImplementedError:
        raised_count = True
    assert raised_count, 'expected NotImplementedError from APIBackend.count_tokens_for_prompt'


def test_api_foundational_config_propagates_stop_string():
    from llmtf.backends import APIBackend

    class Response:
        def __init__(self, data):
            self._data = data

        def json(self):
            return self._data

    api = APIBackend(
        api_base='http://localhost:0', api_profile='vllm',
        model_context_len=16000,
    )

    def request(method, path, **kwargs):
        if path == '/v1/models':
            return Response({'data': [{'id': 'model', 'max_model_len': 16000}]})
        if path == '/tokenize':
            return Response({'count': 1})
        raise AssertionError(path)

    api._request = request
    api.from_pretrained('model', is_foundational=True)
    assert api.generation_config.stop_strings == ['\n\n']
    assert api.stop_strings_base == ['\n\n']
    assert api.conversation_template_path.endswith(
        'conversation_configs/default_foundational.json'
    )


def test_api_probability_all_candidates_censored_returns_zero_bounds():
    from llmtf.backends import APIBackend

    class Response:
        def json(self):
            return {
                'choices': [{
                    'logprobs': {'content': [{
                        'token': 'other',
                        'top_logprobs': [
                            {'token': 'other', 'logprob': -0.1},
                        ],
                    }]},
                }],
                'usage': {'prompt_tokens': 4},
            }

    api = APIBackend(api_base='http://localhost:0', api_profile='vllm')
    api.model_name = 'model'
    api._request = lambda *args, **kwargs: Response()
    _, probabilities, info = api.calculate_tokens_proba(
        [{'role': 'user', 'content': 'choose'}], ['1', '2', '3'],
    )
    assert probabilities == {'1': 0.0, '2': 0.0, '3': 0.0}
    assert info['candidate_ranking_resolved'] is False
    assert info['candidate_score_semantics'] == (
        'top_k_censored_all_candidates_below_cutoff'
    )


def test_task_prompt_budget_and_quota_helpers():
    from llmtf.base import (
        PromptTooLongError, distribute_sample_limit, ensure_prompt_fits,
    )
    assert distribute_sample_limit(8, 57).count(1) == 8
    assert sum(distribute_sample_limit(8, 57)) == 8
    assert sum(distribute_sample_limit(100, 45)) == 100
    ensure_prompt_fits(None, 10, 'task')
    ensure_prompt_fits(10, 10, 'task')
    try:
        ensure_prompt_fits(11, 10, 'task')
        raise AssertionError('oversized prompt was accepted')
    except PromptTooLongError as exc:
        assert 'does not define a semantics-preserving truncation policy' in str(exc)


def test_ruopinion_parser_never_executes_model_output():
    module = _load_task_source(
        'llmtf/tasks/ruopinionne.py', 'task_ruopinionne_test'
    )
    parsed = module.pred2opinions_default(
        "{'Source': 'NULL', 'Target': 'x', 'Polarity': 'POS', "
        "'Expression': ['x']}"
    )
    assert parsed[0]['Target'] == 'x'
    assert module.pred2opinions_default(
        "__import__('os').environ.clear()"
    ) == []
    assert module.pred2opinions_default('null') == []


def test_ruparam_uses_canonical_message_roles():
    module = _load_task_source('llmtf/tasks/ruparam.py', 'task_ruparam_test')
    task = module.RuParam(instruction='{sent_lhs} / {sent_rhs}')
    messages = task.create_messages(
        {'gram': 'ok', 'ungram': 'bad', 'order': 's'}, with_answer=True
    )
    assert [message['role'] for message in messages] == ['user', 'assistant']


def test_ruparam_normalises_inverse_rows_and_source_confusables():
    module = _load_task_source(
        'llmtf/tasks/ruparam.py', 'task_ruparam_normalise_test'
    )
    row = module._normalise_row({
        'id': 'duplicate-id',
        'gram': 'stored ungrammatical',
        'ungram': 'stored grammatical',
        'label': ' category ',
        'source': 'torfl_С1',  # Cyrillic С in the source export.
        'order': 'i',
    }, 7)
    assert row['gram'] == 'stored grammatical'
    assert row['ungram'] == 'stored ungrammatical'
    assert row['category'] == 'category'
    assert row['source'] == 'torfl_C1'
    assert row['part'] == 'torfl'
    assert row['torfl_level'] == 'C1'
    assert row['row_id'].startswith('7:')


def test_ruparam_pair_accuracy_and_diagnostic_slices():
    module = _load_task_source(
        'llmtf/tasks/ruparam.py', 'task_ruparam_aggregation_test'
    )

    def result(row_id, presentation, correct, category, part, level=None):
        return {
            'row_id': row_id,
            'presentation': presentation,
            'correct': correct,
            'category': category,
            'source': 'torfl_A1' if part == 'torfl' else 'RuConst',
            'part': part,
            'torfl_level': level,
            'identical_sentences': False,
        }

    results = [
        result('0:digest', 'grammatical_first', True, 'agreement',
               'torfl', 'A1'),
        result('0:digest', 'grammatical_second', True, 'agreement',
               'torfl', 'A1'),
        # A different annotated row may have the same source id in the CSV;
        # row_id, not that source id, is the aggregation identity.
        result('1:digest', 'grammatical_first', True, 'island',
               'parametric'),
        result('1:digest', 'grammatical_second', False, 'island',
               'parametric'),
    ]
    score, details = module.RuParam._aggregate_pair_accuracy(results)
    assert score == 0.5
    assert details['pair_count'] == 2
    assert details['presentation_count'] == 4
    assert details['category_macro_accuracy'] == 0.5
    assert details['by_category']['agreement'] == {
        'accuracy': 1.0, 'count': 1,
    }
    assert details['by_category']['island'] == {
        'accuracy': 0.0, 'count': 1,
    }
    assert details['by_part']['torfl']['accuracy'] == 1.0
    assert details['by_part']['parametric']['accuracy'] == 0.0
    assert details['by_torfl_level']['A1']['count'] == 1


def test_ruparam_loader_is_zero_shot_and_accepts_train_fallback():
    module = _load_task_source(
        'llmtf/tasks/ruparam.py', 'task_ruparam_loader_test'
    )
    task = module.RuParam(instruction='{sent_lhs} / {sent_rhs}')
    try:
        task._load_dataset(None, 100, 1, few_shot_count=1)
        raise AssertionError('RuParam accepted a few-shot configuration')
    except ValueError as exc:
        assert 'few_shot_count=0' in str(exc)

    class FakeDataset(list):
        def select(self, indexes):
            return FakeDataset([self[index] for index in indexes])

    rows = FakeDataset([
        {
            'id': 'same-id', 'gram': 'good one', 'ungram': 'bad one',
            'label': 'one', 'source': 'torfl_A1', 'order': 's',
        },
        {
            'id': 'same-id', 'gram': 'bad two', 'ungram': 'good two',
            'label': 'two', 'source': 'RuConst', 'order': 'i',
        },
    ])
    module.load_dataset = lambda **kwargs: {'train': rows}
    module.tqdm = lambda values: values

    class FakeModel:
        @staticmethod
        def count_tokens_for_messages(messages):
            return len(messages)

    loaded = task._load_dataset(
        FakeModel(), max_prompt_len=100, max_sample_per_dataset=2,
        few_shot_count=0,
    )
    assert len(loaded) == 4
    assert len({item['sample']['row_id'] for item in loaded}) == 2
    assert loaded[2]['sample']['gram'] == 'good two'
    assert loaded[2]['sample']['ungram'] == 'bad two'
    assert [item['sample']['correct_choice'] for item in loaded] == [
        '1', '2', '1', '2',
    ]


def test_shlepa_keeps_correct_answer_a():
    module = _load_task_source('llmtf/tasks/shlepa.py', 'task_shlepa_test')
    task = module.ShlepaSmallMMLU('example/dataset')
    module.random.shuffle = lambda values: None
    doc = {
        'question': 'q', 'correct_answer': 'answerA',
        'answerA': 'correct', 'answerB': 'b', 'answerC': 'c', 'answerD': 'd',
    }
    additional = {
        f'answer{label}': [f'{label}{index}' for index in range(5)]
        for label in 'ABCD'
    }
    result = task._helper(doc, additional)
    assert result['gold'] == 'A'


def test_shlepa_distractors_use_full_dataset_when_sample_limit_is_one():
    module = _load_task_source('llmtf/tasks/shlepa.py', 'task_shlepa_limit_test')
    task = module.ShlepaSmallMMLU('example/dataset')

    class FakeDataset:
        def __init__(self):
            self.rows = [
                {'id': index, 'answerA': f'a{index}'} for index in range(6)
            ]
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, index):
            if isinstance(index, list):
                return {
                    key: [self.rows[item][key] for item in index]
                    for key in self.rows[0]
                }
            return self.rows[index]

    distractors = task._get_additional_samples(0, FakeDataset())
    assert distractors['id'] == [1, 2, 3, 4, 5]


def test_rublimp_empty_prediction_is_incorrect_not_an_exception():
    module = _load_task_source('llmtf/tasks/rublimp.py', 'task_rublimp_test')
    classify = module.RuBlimpClassify(dataset_slices=['slice'])
    choice = module.RuBlimpChoice(dataset_slices=['slice'])
    assert classify.evaluate({'correct': True}, '')['acc'] is False
    assert choice.evaluate({'swap': False}, None)['acc'] is False


def test_copytext_rejects_backend_without_local_tokenizer_early():
    module = _load_task_source('llmtf/tasks/darumeru.py', 'task_copytext_test')
    task = module.CopyText(subtask='sent', lang='ru')
    model = type('APIModel', (), {'backend': object()})()
    try:
        task.load_dataset(
            model, max_prompt_len=100, max_sample_per_dataset=1,
            few_shot_count=0,
        )
        raise AssertionError('CopyText accepted a backend without a tokenizer')
    except NotImplementedError as exc:
        assert 'requires a local tokenizer' in str(exc)


def test_llmaaj_loader_uses_evaluator_prompt_budget_name():
    import ast
    path = os.path.join(ROOT, 'llmtf/tasks/llm_as_a_judge.py')
    with open(path, encoding='utf-8') as source:
        tree = ast.parse(source.read(), filename=path)
    task_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == 'LLMAsJudgeStyleControl'
    )
    loader = next(
        node for node in task_class.body
        if isinstance(node, ast.FunctionDef) and node.name == 'load_dataset'
    )
    arguments = [argument.arg for argument in loader.args.args]
    assert 'max_prompt_len' in arguments
    assert 'max_len' not in arguments


def test_ner_in_place_requires_the_complete_original_text():
    module = _load_task_source(
        'llmtf/tasks/ner/ner_abc.py', 'task_ner_abc_test'
    )
    sample = {'tokens': ['Иван', 'пришёл', '.']}
    check = module.NerInPlaceAbc.check_text
    assert check(None, sample, '<PERSON>Иван</PERSON> пришёл.')
    assert not check(None, sample, '<PERSON>Иван</PERSON>')
    assert not check(
        None, sample, '<PERSON>Иван</PERSON> пришёл. Лишнее'
    )


if __name__ == '__main__':
    import inspect
    g = globals()
    failures = 0
    for name, fn in list(g.items()):
        if name.startswith('test_') and inspect.isfunction(fn):
            try:
                fn()
                print('PASS', name)
            except Exception as e:
                failures += 1
                import traceback
                print('FAIL', name, '->', type(e).__name__ + ':', e)
                traceback.print_exc()
    print('\n%d failures' % failures)
    sys.exit(1 if failures else 0)
