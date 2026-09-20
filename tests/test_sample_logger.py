import unittest

from llmtf.utils import normalize_message_roles, remove_image


class RemoveImageTests(unittest.TestCase):
    def test_rendered_prompt_is_preserved(self):
        sample, prompt = remove_image(
            {"id": 1, "image": "not-serializable"},
            "rendered prompt",
        )

        self.assertEqual(sample, {"id": 1})
        self.assertEqual(prompt, "rendered prompt")

    def test_empty_prompt_returns_sample_and_prompt(self):
        sample, prompt = remove_image({"id": 1, "image": object()}, [])

        self.assertEqual(sample, {"id": 1})
        self.assertEqual(prompt, [])

    def test_multimodal_chat_drops_only_image_items(self):
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "question"},
                {"type": "image_url", "image_url": {"url": "data:..."}},
                "plain content",
            ],
        }]

        _, safe_messages = remove_image({"id": 1}, messages)

        self.assertEqual(
            safe_messages[0]["content"],
            [{"type": "text", "text": "question"}, "plain content"],
        )
        self.assertEqual(len(messages[0]["content"]), 3)

    def test_legacy_bot_role_is_canonicalized_in_sample_and_prompt(self):
        sample, prompt = remove_image(
            {"messages": [{"role": "bot", "content": "prefix"}]},
            [{"role": "bot", "content": "prefix"}],
        )

        self.assertEqual(sample["messages"][0]["role"], "assistant")
        self.assertEqual(prompt[0]["role"], "assistant")

    def test_message_role_normalization_does_not_mutate_input(self):
        messages = [{"role": "bot", "content": "prefix"}]

        normalized = normalize_message_roles(messages)

        self.assertEqual(normalized[0]["role"], "assistant")
        self.assertEqual(messages[0]["role"], "bot")


if __name__ == "__main__":
    unittest.main()
