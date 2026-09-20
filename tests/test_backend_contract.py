import unittest
import inspect
from unittest import mock

from test_refactor_logic import _stub_third_party
_stub_third_party()

from llmtf.backends.base import normalize_stop_token_ids, validate_batch_result
from llmtf.config import DEFAULT_VLLM_GPU_MEMORY_UTILIZATION
from llmtf.llm import LLM
from llmtf.utils import MaxLenContext


class BackendContractTests(unittest.TestCase):
    def test_vllm_gpu_memory_default_is_shared_runtime_value(self):
        from llmtf.backends.vllm import VLLMBackend

        default = inspect.signature(VLLMBackend.__init__).parameters[
            'gpu_memory_utilization'
        ].default
        self.assertEqual(default, DEFAULT_VLLM_GPU_MEMORY_UTILIZATION)
        self.assertEqual(default, 0.92)

    def test_continuation_provenance_is_not_redacted(self):
        from llmtf.provenance import sanitize
        value = {
            'token_probability_surface_forms': 'candidate_and_single_leading_space',
            'token_probability_aggregation': 'max',
        }
        self.assertEqual(sanitize(value), value)

    def test_normalize_stop_token_ids(self):
        cases = ((None, None), (3, [3]), ([3, 4], [3, 4]), ((3,), [3]))
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(normalize_stop_token_ids(value), expected)

    def test_batch_result_alignment_is_validated(self):
        with self.assertRaisesRegex(ValueError, 'misaligned'):
            validate_batch_result((['p'], [], [{}]), 1, method='generate')

    def test_two_pass_multiple_sequences_fails_before_backend_call(self):
        class Backend:
            def __init__(self):
                self.generation_config = type('Config', (), {
                    'max_new_tokens': 8, 'max_length': 100, 'stop_strings': [],
                    'eos_token_id': [1], 'num_return_sequences': 2,
                })()
                self.calls = 0
            def generate_batch(inner_self, *args, **kwargs):
                inner_self.calls += 1
                raise AssertionError('must not be called')
            def get_params(inner_self): return {}
            def support_method(inner_self, method): return True
            def add_stop_strings(inner_self, values): pass
            def reset_stop_strings(inner_self): pass
            def get_model_context_len(inner_self): return 100
        backend = Backend()
        model = LLM(backend=backend)
        model._setup_reasoning('hybrid', 10, 2, end_thinking_token_id=7)
        with self.assertRaisesRegex(NotImplementedError, 'num_return_sequences=1'):
            model.generate_batch(
                [[{'role': 'user', 'content': 'q'}]], enable_thinking=True
            )
        self.assertEqual(backend.calls, 0)

    def test_hybrid_disabled_does_not_reserve_reasoning_budget(self):
        model, task = self._budget_model()
        with MaxLenContext(
            task, model, None, reasoning_enabled=False
        ) as prompt_budget:
            self.assertEqual(prompt_budget, 900)
            self.assertEqual(model.reasoning_config.max_new_tokens_reasoning, 400)

    def test_budget_state_restored_after_exception(self):
        model, task = self._budget_model()
        with self.assertRaisesRegex(RuntimeError, 'inside'):
            with MaxLenContext(task, model, None, reasoning_enabled=True):
                self.assertEqual(model.generation_config.max_new_tokens, 100)
                raise RuntimeError('inside')
        self.assertEqual(model.generation_config.max_new_tokens, 64)
        self.assertEqual(model.reasoning_config.max_new_tokens_reasoning, 400)

    def test_strict_reasoning_budget_failure_is_explicit(self):
        model, task = self._budget_model()
        model._setup_reasoning('reasoning', 400, 100, end_thinking_token_id=7)
        model.backend.get_model_context_len = lambda: 150
        with self.assertRaisesRegex(ValueError, 'Strict reasoning'):
            with MaxLenContext(task, model, None, reasoning_enabled=True):
                pass

    def _budget_model(self):
        class Backend:
            generation_config = type('Config', (), {'max_new_tokens': 64})()
            def get_model_context_len(inner_self): return 1000
            def get_params(inner_self): return {}
        class Task:
            max_task_new_tokens = 100
        model = LLM(backend=Backend())
        model._setup_reasoning('hybrid', 400, 100, end_thinking_token_id=7)
        return model, Task()

    def test_vllm_sampling_uses_effective_phase_stop_ids(self):
        import llmtf.backends.vllm as module
        captured = {}
        class SamplingParams:
            def __init__(inner_self, **kwargs):
                captured.update(kwargs)
        output = type('Output', (), {
            'token_ids': [9], 'cumulative_logprob': -1.0, 'text': 'answer'
        })()
        response = type('Response', (), {
            'prompt_token_ids': [1, 2], 'outputs': [output]
        })()
        backend = object.__new__(module.VLLMBackend)
        backend.generation_config = type('Config', (), {
            'max_length': 100, 'eos_token_id': [1],
        })()
        class Tokenizer:
            def __call__(self, *args, **kwargs):
                return {'input_ids': [1, 2]}
            def decode(self, ids):
                return 'prompt'
            def apply_chat_template(self, *args, **kwargs):
                return 'prompt'
        tokenizer = Tokenizer()
        backend.tokenizer = tokenizer
        backend.model = type('Model', (), {
            'generate': lambda inner_self, **kwargs: [response]
        })()
        backend._get_lora_request = lambda: None
        phase = type('Phase', (), {
            'temperature': 0.0, 'top_p': 1.0, 'top_k': -1,
            'max_new_tokens': 10, 'repetition_penalty': 1.0,
            'presence_penalty': 0.0, 'stop_strings': [],
            'eos_token_id': [42], 'num_return_sequences': 1,
        })()
        with mock.patch.object(module, 'SamplingParams', SamplingParams):
            backend.generate_batch(
                [[{'role': 'user', 'content': 'q'}]], generation_config=phase
            )
        self.assertEqual(captured['stop_token_ids'], [42])
        self.assertEqual(backend.generation_config.eos_token_id, [1])


if __name__ == '__main__':
    unittest.main()
