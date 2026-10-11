"""Native Qwen chat continuation; opt in with LLMTF_REASONING_TEMPLATE_MODEL.

Run separately in the HF image with the pinned Qwen3.5-2B tokenizer mounted.
No weights, GPU or network are needed for this regression test.
"""

import os
import unittest


@unittest.skipUnless(os.getenv('LLMTF_REASONING_TEMPLATE_MODEL'), 'Local Qwen tokenizer not selected')
class ReasoningChatTemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        from llmtf.reasoning import THINK_CLOSE_MARKER, ReasoningConfig, EmulatedReasoningStrategy

        cls.tokenizer = AutoTokenizer.from_pretrained(
            os.environ['LLMTF_REASONING_TEMPLATE_MODEL'], local_files_only=True,
        )
        cls.close = THINK_CLOSE_MARKER
        cls.body = cls.close.replace('/', '') + '\nFirst paragraph.\n\nSecond paragraph.'
        cls.config = ReasoningConfig(max_new_tokens_reasoning=32, min_new_tokens_reasoning=32)
        cls.strategy = EmulatedReasoningStrategy(cls.config)

    def render(self, content):
        return self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': 'Question'}, {'role': 'assistant', 'content': content}],
            tokenize=False, add_generation_prompt=False, continue_final_message=True,
            enable_thinking=False,
        )

    def test_original_extra_newline_reproduces_chat_error(self):
        with self.assertRaisesRegex(ValueError, 'continue_final_message'):
            self.render(self.body + '\n\n' + self.close + '\n\nAnswer:')

    def test_normalized_reasoning_passes_native_chat_continuation(self):
        for reason in ('stop', 'length', 'eos'):
            for tail in ('', '\n', '\n\n', ' \t\r\n\n'):
                for phrase in ('', '\nBudget phrase.\n\n'):
                    with self.subTest(reason=reason, tail=repr(tail), phrase=bool(phrase)):
                        self.config.fmt.truncation_prompt = phrase
                        source = self.body + tail + (self.close if reason == 'stop' else '')
                        length = 32 if reason == 'length' else 5
                        normalized, _ = self.strategy._normalize_reasoning(
                            source, length, {'reason': reason}, True,
                        )
                        content = normalized + '\n\nAnswer:'
                        rendered = self.render(content)
                        self.assertTrue(rendered.endswith(content))
                        self.assertIn('First paragraph.\n\nSecond paragraph.', rendered)

    def test_empty_reasoning_passes_native_chat_continuation(self):
        for source, reason in ((self.close, 'stop'), ('\n\n', 'eos')):
            with self.subTest(reason=reason):
                normalized, _ = self.strategy._normalize_reasoning(
                    source, 1, {'reason': reason}, True,
                )
                content = normalized + '\n\nAnswer:'
                self.assertTrue(self.render(content).endswith(content))


if __name__ == '__main__':
    unittest.main()
