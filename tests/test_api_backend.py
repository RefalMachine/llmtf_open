import copy
import unittest
from unittest import mock

from test_refactor_logic import _stub_third_party
_stub_third_party()

from llmtf.backends.api import APIBackend
from llmtf.backends.base import BackendBatchError, BackendRequestError
from llmtf.config import SamplingConfig
from llmtf.continuation import ContinuationConfig, PrefillCompatibilityError
from llmtf.llm import LLM
from llmtf.reasoning import THINK_CLOSE_MARKER


class Response:
    status_code = 200
    text = ''
    def __init__(self, payload):
        self.payload = payload
    def json(self):
        return self.payload


class APIBackendTests(unittest.TestCase):
    def _backend(self):
        backend = APIBackend(
            api_base='http://server', api_key='secret', max_retries=0,
            num_procs=2, api_profile='vllm',
        )
        backend.model_name = 'model'
        backend.generation_config = SamplingConfig()
        return backend

    def test_api_uses_phase_local_stop_ids_without_mutating_base(self):
        backend = self._backend()
        base = copy.deepcopy(backend.generation_config)
        phase = copy.deepcopy(base)
        phase.eos_token_id = [42]
        payloads = []
        def request(method, path, **kwargs):
            payloads.append(kwargs['json'])
            return Response({
                'choices': [{'message': {'content': 'ok'}}],
                'usage': {'prompt_tokens': 2, 'completion_tokens': 1},
            })
        backend._request = request
        backend.generate(
            [{'role': 'user', 'content': 'q'}], generation_config=phase
        )
        self.assertEqual(payloads[0]['stop_token_ids'], [42])
        self.assertEqual(
            backend.generation_config.eos_token_id, base.eos_token_id
        )

    def test_two_pass_payload_restores_answer_stop_ids(self):
        import llmtf.backends.api as api_module
        backend = self._backend()
        backend.generation_config.eos_token_id = [1]
        payloads = []
        def request(method, path, **kwargs):
            payload = kwargs['json']
            payloads.append(payload)
            is_reasoning = payload['stop_token_ids'] == [42]
            return Response({
                'choices': [{'message': {'content': (
                    'reasoning' + THINK_CLOSE_MARKER if is_reasoning else 'answer'
                )}}],
                'usage': {'prompt_tokens': 2, 'completion_tokens': 1},
            })
        backend._request = request
        model = LLM(backend=backend, assistant_prefill_policy='best_effort')
        model._setup_reasoning(
            'hybrid', 10, 2, end_thinking_token_id=42
        )
        with mock.patch.object(
            api_module.tqdm, 'tqdm', lambda iterable, **kwargs: iterable
        ):
            model.generate_batch(
                [[{'role': 'user', 'content': 'q'}]], enable_thinking=True
            )
        self.assertEqual([payload['stop_token_ids'] for payload in payloads], [[42], [1]])
        self.assertEqual(backend.generation_config.eos_token_id, [1])

    def test_failed_middle_request_fails_closed_with_original_index(self):
        backend = self._backend()
        def generate(self, messages, **kwargs):
            index = int(messages[0]['content'])
            if index == 1:
                raise RuntimeError('boom')
            return messages, str(index), {'generated_len': [1]}
        import llmtf.backends.api as api_module
        with mock.patch.object(APIBackend, 'generate', generate), \
                mock.patch.object(api_module.tqdm, 'tqdm', lambda iterable, **kwargs: iterable):
            with self.assertRaises(BackendBatchError) as caught:
                backend.generate_batch([
                    [{'role': 'user', 'content': '0'}],
                    [{'role': 'user', 'content': '1'}],
                    [{'role': 'user', 'content': '2'}],
                ])
        self.assertEqual(list(caught.exception.failures), [1])

    def test_tokenize_failure_is_not_cached_as_zero(self):
        backend = self._backend()
        calls = {'count': 0}
        def request(*args, **kwargs):
            calls['count'] += 1
            raise RuntimeError('tokenizer unavailable')
        backend._request = request
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, 'unavailable'):
                backend.count_tokens_for_messages([
                    {'role': 'user', 'content': 'q'}
                ])
        self.assertEqual(calls['count'], 2)

    def test_openai_profile_needs_no_tokenize_and_uses_standard_payload(self):
        backend = APIBackend(
            api_base='http://server', api_key='secret', max_retries=0,
            api_profile='openai', model_context_len=4096,
        )
        backend.model_name = 'closed-model'
        backend.generation_config = SamplingConfig()
        payloads = []

        def request(method, path, **kwargs):
            payloads.append((path, kwargs.get('json')))
            return Response({
                'choices': [{'message': {'content': 'answer'}}],
            })

        backend._request = request
        self.assertIsNone(backend.count_tokens_for_messages([
            {'role': 'user', 'content': 'q'},
        ]))
        _, output, info = backend.generate([
            {'role': 'user', 'content': 'q'},
        ])
        self.assertEqual(output, 'answer')
        self.assertIsNone(info['prompt_len'])
        payload = payloads[0][1]
        for extension in (
            'top_k', 'repetition_penalty', 'stop_token_ids',
            'add_generation_prompt', 'continue_final_message',
            'chat_template_kwargs', 'skip_special_tokens',
        ):
            self.assertNotIn(extension, payload)

    def test_openai_profile_rejects_any_prefill_unless_best_effort(self):
        backend = APIBackend(
            api_base='http://server', api_key='secret', max_retries=0,
            api_profile='openai', model_context_len=4096,
        )
        backend.model_name = 'closed-model'
        backend.generation_config = SamplingConfig()
        with self.assertRaisesRegex(
            PrefillCompatibilityError, 'does not declare support'
        ):
            backend.generate([
                {'role': 'user', 'content': 'q'},
                {'role': 'assistant', 'content': 'Answer:'},
            ])

    def test_auto_profile_tolerates_missing_discovery_and_tokenize(self):
        backend = APIBackend(
            api_base='http://server', api_key='secret', max_retries=0,
            api_profile='auto', model_context_len=4096,
        )

        def request(method, path, **kwargs):
            raise BackendRequestError(method, '<redacted>' + path, 404)

        backend._request = request
        backend.from_pretrained('closed-model')
        self.assertEqual(backend.model_name, 'closed-model')
        self.assertIsNone(backend.count_tokens_for_messages([
            {'role': 'user', 'content': 'q'},
        ]))
        self.assertFalse(backend.server_capabilities['model_discovery'])
        self.assertFalse(backend.server_capabilities['tokenize'])
        self.assertFalse(backend.supports_assistant_continuation)

    def test_top_logprobs_are_reported_as_censored_ranking(self):
        backend = self._backend()
        backend._request = lambda *args, **kwargs: Response({
            'choices': [{
                'logprobs': {'content': [{
                    'token': ' A',
                    'top_logprobs': [
                        {'token': ' A', 'logprob': -0.1},
                        {'token': 'other', 'logprob': -0.2},
                    ],
                }]},
            }],
            'usage': {'prompt_tokens': 3, 'completion_tokens': 1},
        })
        _, scores, info = backend.calculate_tokens_proba(
            [{'role': 'user', 'content': 'q'}], ['A', 'B']
        )
        self.assertGreater(scores['A'], 0)
        self.assertEqual(scores['B'], 0.0)
        self.assertEqual(
            info['candidate_score_semantics'], 'top_k_censored_ranking'
        )
        self.assertEqual(info['candidate_surface_coverage']['A'], [' A'])
        self.assertEqual(info['candidate_surface_coverage']['B'], [])

    def test_top_logprobs_fail_when_no_candidate_is_visible(self):
        backend = self._backend()
        backend._request = lambda *args, **kwargs: Response({
            'choices': [{
                'logprobs': {'content': [{
                    'token': 'other',
                    'top_logprobs': [
                        {'token': 'other', 'logprob': -0.1},
                    ],
                }]},
            }],
        })
        with self.assertRaisesRegex(
            BackendRequestError, 'none of the requested candidate'
        ):
            backend.calculate_tokens_proba(
                [{'role': 'user', 'content': 'q'}], ['A', 'B']
            )

    def test_two_pass_reasoning_fails_before_request_without_continuation(self):
        backend = APIBackend(
            api_base='http://server', api_key='secret', max_retries=0,
            api_profile='openai', model_context_len=4096,
        )
        backend.model_name = 'closed-model'
        backend.generation_config = SamplingConfig()
        backend._request = mock.Mock(side_effect=AssertionError('unexpected request'))
        model = LLM(backend=backend)
        model._setup_reasoning('hybrid', 10, 2, end_thinking_token_id=42)
        with self.assertRaisesRegex(
            NotImplementedError, 'requires assistant continuation'
        ):
            model.generate_batch(
                [[{'role': 'user', 'content': 'q'}]], enable_thinking=True
            )
        backend._request.assert_not_called()

    def test_params_redact_endpoint_and_never_include_key(self):
        backend = self._backend()
        backend.max_model_len = 1024
        params = backend.get_params()
        self.assertEqual(params['api_base'], '<redacted>')
        self.assertNotIn('api_key', params)
        self.assertNotIn('secret', str(params))

    def test_auto_rejects_unprobed_trailing_whitespace(self):
        backend = self._backend()
        with self.assertRaisesRegex(PrefillCompatibilityError, 'probe_api_prefill'):
            backend.generate([
                {'role': 'user', 'content': 'q'},
                {'role': 'assistant', 'content': 'Answer: '},
            ])

    def test_api_probe_detects_stripped_suffix(self):
        backend = self._backend()
        backend.configure_continuation(ContinuationConfig(probe_api_prefill=True))
        backend._request = lambda *args, **kwargs: Response({
            'count': 1, 'tokens': [10],
        })
        with self.assertRaisesRegex(PrefillCompatibilityError, 'suffix is stripped'):
            backend.generate([
                {'role': 'user', 'content': 'q'},
                {'role': 'assistant', 'content': 'Answer:\n'},
            ])

    def test_auto_accepts_probe_verified_token_effect_and_logs_it(self):
        backend = self._backend()
        backend.configure_continuation(ContinuationConfig(probe_api_prefill=True))
        tokenize_calls = {'count': 0}

        def request(method, path, **kwargs):
            if path == '/tokenize':
                tokenize_calls['count'] += 1
                content = kwargs['json']['messages'][-1]['content']
                tokens = [10, 11] if content.endswith('\n') else [10]
                return Response({'count': len(tokens), 'tokens': tokens})
            if path == '/detokenize':
                raise RuntimeError('not supported')
            return Response({
                'choices': [{'message': {'content': 'answer'}}],
                'usage': {'prompt_tokens': 2, 'completion_tokens': 1},
            })

        backend._request = request
        _, output, info = backend.generate([
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': 'Answer:\n'},
        ])
        self.assertEqual(output, 'answer')
        self.assertEqual(
            info['assistant_prefill']['verification'],
            'verified_token_effect',
        )
        self.assertEqual(tokenize_calls['count'], 2)

    def test_exact_accepts_detokenize_verified_suffix(self):
        backend = self._backend()
        backend.configure_continuation(ContinuationConfig(
            prefill_policy='exact', probe_api_prefill=True,
        ))

        def request(method, path, **kwargs):
            if path == '/tokenize':
                content = kwargs['json']['messages'][-1]['content']
                tokens = [10, 11] if content.endswith('\n') else [10]
                return Response({'count': len(tokens), 'tokens': tokens})
            if path == '/detokenize':
                return Response({'prompt': 'LLMTF assistant-prefill probe:\n'})
            return Response({
                'choices': [{'message': {'content': 'answer'}}],
                'usage': {'prompt_tokens': 2, 'completion_tokens': 1},
            })

        backend._request = request
        _, _, info = backend.generate([
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': 'Answer:\n'},
        ])
        self.assertEqual(
            info['assistant_prefill']['verification'], 'verified_exact'
        )


if __name__ == '__main__':
    unittest.main()
