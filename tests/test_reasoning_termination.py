import unittest

from test_refactor_logic import _stub_third_party, _build_strategy

_stub_third_party()

from llmtf.llm import LLM
from llmtf.reasoning import THINK_CLOSE_MARKER
from llmtf.config import SamplingConfig
from llmtf.backends.base import vllm_generation_termination


class ReasoningTerminationTests(unittest.TestCase):
    def test_explicit_stop_on_known_native_eos_is_recognized(self):
        for stop_reason, expected in ((99, 'eos'), (42, 'stop'), ('STOP', 'stop'), (None, 'eos')):
            with self.subTest(stop_reason=stop_reason):
                info = vllm_generation_termination('stop', stop_reason, native_eos_token_ids=[99])
                self.assertEqual(info['reason'], expected)
        info = vllm_generation_termination('stop', None, has_stop_reason=False, native_eos_token_ids=[99])
        self.assertEqual(info['reason'], 'unknown')

    def test_eos_closes_without_budget_phrase_and_preserves_prefill(self):
        strategy, _ = _build_strategy(truncation_prompt='[budget phrase]')
        cases = [('reason[native-eos]', 5), ('[native-eos]', 1),
                 ('reason[native-eos]', 100), ('reason', 5)]
        for text, generated_len in cases:
            with self.subTest(text=text, generated_len=generated_len):
                seen = []

                def reasoning_fn(messages, **kwargs):
                    return ['p'], [text], [{
                        'prompt_len': 2, 'generated_len': [generated_len],
                        'generated_cumulative_logprob': None,
                        'generation_termination': [{'reason': 'eos', 'terminal_text': '[native-eos]'}],
                    }]

                def continuation_fn(messages, **kwargs):
                    seen.extend(messages)
                    return ['p'], ['answer'], [{'generated_len': [1]}]

                result = strategy.run(
                    [[{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': 'Ответ: '}]],
                    reasoning_fn=reasoning_fn, continuation_fn=continuation_fn,
                    generation_config=SamplingConfig(), reasoning_phase_kwargs={},
                    continuation_phase_kwargs={},
                )
                self.assertNotIn('[native-eos]', seen[0][-1]['content'])
                self.assertNotIn('[budget phrase]', seen[0][-1]['content'])
                self.assertTrue(seen[0][-1]['content'].endswith(THINK_CLOSE_MARKER + '\n\nОтвет: '))
                info = LLM._reasoning_info_entry(result.reasoning_infos[0], result.reasoning_outputs[0], True)
                self.assertEqual(info['termination_reason'], 'eos')
                self.assertTrue(info['synthetic_close'])
                self.assertEqual(info['eos_removed'], text.endswith('[native-eos]'))
                self.assertEqual(info['generated_len'], [generated_len])

    def test_close_marker_then_eos_does_not_duplicate_close(self):
        strategy, _ = _build_strategy()
        text, info = strategy._normalize_reasoning(
            'reason' + THINK_CLOSE_MARKER + '[native-eos]', 5,
            {'reason': 'eos', 'terminal_text': '[native-eos]'}, True,
        )
        self.assertEqual(text, 'reason\n' + THINK_CLOSE_MARKER)
        self.assertEqual(info['termination_reason'], 'think_close')
        self.assertFalse(info['synthetic_close'])

    def test_close_marker_used_as_native_eos_still_counts_as_natural_close(self):
        strategy, _ = _build_strategy()
        text, info = strategy._normalize_reasoning(
            'reason' + THINK_CLOSE_MARKER, 5,
            {'reason': 'eos', 'terminal_text': THINK_CLOSE_MARKER}, True,
        )
        self.assertEqual(text, 'reason\n' + THINK_CLOSE_MARKER)
        self.assertEqual(info['termination_reason'], 'think_close')
        self.assertFalse(info['synthetic_close'])
        self.assertFalse(info['eos_removed'])

    def test_unknown_short_stop_is_not_treated_as_eos(self):
        strategy, _ = _build_strategy()
        for reason in ('unknown', 'stop', 'error', 'abort'):
            with self.subTest(reason=reason), self.assertRaises(ValueError):
                strategy._normalize_reasoning('short', 2, {'reason': reason}, True)

    def test_error_is_not_normalized_as_a_budget_stop(self):
        strategy, _ = _build_strategy()
        for reason in ('error', 'abort', 'cancelled', 'content_filter'):
            for key in ('reason', 'finish_reason'):
                with self.subTest(reason=reason, key=key), self.assertRaises(ValueError):
                    strategy._normalize_reasoning('partial', 100, {key: reason}, True)

    def test_eos_in_middle_is_not_deleted(self):
        strategy, _ = _build_strategy()
        text, _ = strategy._normalize_reasoning(
            'reason[native-eos]quoted', 5,
            {'reason': 'eos', 'terminal_text': '[native-eos]'}, True,
        )
        self.assertIn('[native-eos]quoted', text)

    def test_native_stops_survive_reasoning_config_without_mutation(self):
        strategy, _ = _build_strategy(end_thinking_token_id=42)
        base = SamplingConfig(eos_token_id=[1, 2], stop_strings=['STOP'])
        phase = strategy._build_reasoning_config(base)
        self.assertEqual(phase.eos_token_id, [1, 2, 42])
        self.assertEqual(base.eos_token_id, [1, 2])
        self.assertEqual(base.stop_strings, ['STOP'])

    def test_length_keeps_budget_phrase(self):
        strategy, _ = _build_strategy(truncation_prompt='[budget phrase]')
        text, info = strategy._normalize_reasoning('reason', 100, {'reason': 'length'}, True)
        self.assertIn('[budget phrase]', text)
        self.assertEqual(info['termination_reason'], 'length')

    def test_natural_close_normalizes_only_trailing_whitespace(self):
        strategy, _ = _build_strategy()
        body = 'First paragraph.\n\n    Indented second paragraph.'
        for tail in ('', '\n', '\n\n', '\n\n\n', ' \t\r\n\n'):
            with self.subTest(tail=repr(tail)):
                text, info = strategy._normalize_reasoning(
                    body + tail + THINK_CLOSE_MARKER, 5, {'reason': 'stop'}, True,
                )
                self.assertEqual(text, body + '\n' + THINK_CLOSE_MARKER)
                self.assertFalse(info['synthetic_close'])
                again, _ = strategy._normalize_reasoning(text, 5, {'reason': 'stop'}, True)
                self.assertEqual(again, text)

    def test_budget_and_eos_use_the_same_close_boundary(self):
        for phrase in ('', '\nBudget phrase.\n\n'):
            strategy, _ = _build_strategy(truncation_prompt=phrase)
            for reason, length in (('length', 100), ('eos', 5)):
                for add_phrase in (False, True):
                    with self.subTest(phrase=phrase, reason=reason, add_phrase=add_phrase):
                        body = 'First paragraph.\n\nSecond paragraph.\n\n'
                        text, info = strategy._normalize_reasoning(
                            body, length, {'reason': reason}, add_phrase,
                        )
                        suffix = phrase if reason == 'length' and add_phrase else ''
                        self.assertEqual(text, (body + suffix).rstrip() + '\n' + THINK_CLOSE_MARKER)
                        self.assertEqual(info['termination_reason'], reason)
                        self.assertTrue(info['synthetic_close'])

    def test_empty_reasoning_has_a_single_newline_before_close(self):
        strategy, _ = _build_strategy()
        for body, reason in ((THINK_CLOSE_MARKER, 'stop'), ('\n\n', 'eos')):
            with self.subTest(reason=reason):
                text, _ = strategy._normalize_reasoning(body, 1, {'reason': reason}, True)
                self.assertEqual(text, '\n' + THINK_CLOSE_MARKER)

    def test_continuation_keeps_prefill_and_answer_separator(self):
        strategy, _ = _build_strategy()
        raw = 'First paragraph.\n\nSecond paragraph.\n\n' + THINK_CLOSE_MARKER
        seen = []

        def reasoning_fn(messages, **kwargs):
            return ['prompt'], [raw], [{
                'prompt_len': 2, 'generated_len': [7],
                'generated_cumulative_logprob': None,
            }]

        def continuation_fn(messages, **kwargs):
            seen.extend(messages)
            return ['prompt'], ['A'], [{'generated_len': [1]}]

        result = strategy.run(
            [[{'role': 'user', 'content': 'Question'}, {'role': 'assistant', 'content': 'Ответ: '}]],
            reasoning_fn=reasoning_fn, continuation_fn=continuation_fn,
            generation_config=SamplingConfig(), reasoning_phase_kwargs={},
            continuation_phase_kwargs={},
        )
        expected = 'First paragraph.\n\nSecond paragraph.\n' + THINK_CLOSE_MARKER + '\n\n'
        self.assertEqual(seen[0][-1]['content'], expected + 'Ответ: ')
        self.assertEqual(result.reasoning_outputs, [expected])
        self.assertEqual(result.reasoning_infos[0]['generated_len'], [7])

    def test_normalization_policy_is_recorded_for_reasoning_models(self):
        from unittest.mock import Mock

        backend = Mock()
        backend.get_params.return_value = {}
        model = LLM(backend=backend)
        model._setup_reasoning('hybrid', 100, 10, end_thinking_token_id=42)
        self.assertEqual(
            model.get_params()['_llmtf_reasoning']['normalization_policy'],
            'single_newline_before_close_v1',
        )
        model._setup_reasoning('plain', 100, 10)
        self.assertNotIn('normalization_policy', model.get_params()['_llmtf_reasoning'])


if __name__ == '__main__':
    unittest.main()
