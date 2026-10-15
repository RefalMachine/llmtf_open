"""JSON chat templates must preserve their literal text and whitespace."""
import importlib.util
import json
from pathlib import Path
import unittest

from jinja2.sandbox import ImmutableSandboxedEnvironment

from llmtf.utils import json_to_jinja


CONFIGS = Path(__file__).resolve().parents[1] / "conversation_configs"


def direct_render(config, messages, generation):
    roles = {
        config.get("system_role", "system"): config.get("system_message_template", ""),
        config.get("user_role", "user"): config.get("user_message_template", ""),
        config.get("bot_role", "assistant"): config.get("bot_message_template", ""),
    }
    parts = [config.get("global_prefix", "")]
    for message in messages:
        parts.append(roles.get(message["role"], "").replace("{role}", message["role"])
                     .replace("{content}", message["content"]))
    if generation:
        parts.append(config.get("suffix", ""))
    return "".join(parts)


def render(config, messages, generation, **options):
    template, _ = json_to_jinja(config)
    env = ImmutableSandboxedEnvironment(**options)
    return env.from_string(template).render(messages=messages, add_generation_prompt=generation)


class JsonTemplateTests(unittest.TestCase):
    def test_all_repository_templates_match_literal_composition(self):
        for path in sorted(CONFIGS.glob("*.json")):
            config = json.loads(path.read_text())
            roles = [config.get(key + "_role", default) for key, default in
                     [("system", "system"), ("user", "user"), ("bot", "assistant")]]
            messages = [dict(role=role, content="  Текст\n\t ") for role in roles]
            for generation in (False, True):
                sequence = messages + ([dict(role=roles[1], content="Вопрос")] if generation else [])
                for trimming in (False, True):
                    with self.subTest(config=path.name, generation=generation, trimming=trimming):
                        actual = render(config, sequence, generation, trim_blocks=trimming, lstrip_blocks=trimming)
                        # Keep model-specific template markers out of failure output.
                        self.assertTrue(actual == direct_render(config, sequence, generation), path.name)

    def test_foundational_generation_has_no_unconfigured_newline(self):
        config = json.loads((CONFIGS / "default_foundational.json").read_text())
        messages = [dict(role="user", content="Пример"), dict(role="assistant", content="да"),
                    dict(role="user", content="Вопрос")]
        self.assertEqual(render(config, messages, True, trim_blocks=True, lstrip_blocks=True),
                         "Query: Пример\n\nResponse: да\n\nQuery: Вопрос\n\nResponse:")

    def test_explicit_whitespace_is_preserved_at_every_boundary(self):
        for whitespace in ("", " ", "\n", "\n\n", "\r\n", "\t  \n"):
            config = dict(global_prefix=whitespace + "Prefix" + whitespace,
                          system_message_template=whitespace + "System {content}" + whitespace,
                          user_message_template=whitespace + "User {content}" + whitespace,
                          bot_message_template=whitespace + "Assistant {content}" + whitespace,
                          suffix=whitespace + "Response:" + whitespace)
            messages = [dict(role="system", content="S"), dict(role="user", content=""),
                        dict(role="assistant", content="\n да\t"), dict(role="user", content="Q")]
            for generation in (False, True):
                with self.subTest(whitespace=whitespace, generation=generation):
                    self.assertEqual(render(config, messages, generation, trim_blocks=True, lstrip_blocks=True),
                                     direct_render(config, messages, generation))

    def test_literal_jinja_syntax_quotes_and_backslashes_are_not_executed(self):
        text = "{{ 7 * 7 }} {% if false %}literal{% endif %} {# comment #} 'quote' \\\n"
        config = dict(global_prefix=text, user_role="reader's role",
                      user_message_template=text + "{role} {content} / {content}", suffix=text)
        messages = [dict(role="reader's role", content="{{ untouched }}\n")]
        self.assertEqual(render(config, messages, True, trim_blocks=True, lstrip_blocks=True),
                         direct_render(config, messages, True))

    def test_missing_templates_and_eos_contract(self):
        self.assertEqual(render({}, [dict(role="user", content="Q")], True, trim_blocks=True), "")
        for value, expected in [(None, None), ("END", "END"), (["END", "OTHER"], "END"), ([], [])]:
            self.assertEqual(json_to_jinja(dict(eos_token=value))[1], expected)


@unittest.skipUnless(importlib.util.find_spec("transformers"), "Run in the project container for Transformers integration")
class TransformersTemplateTests(unittest.TestCase):
    def test_actual_tokenizer_and_assistant_continuation(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import PreTrainedTokenizerFast
        from llmtf.continuation import render_local_chat_prompt

        tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"unknown": 0}, unk_token="unknown")))
        for path in sorted(CONFIGS.glob("*.json")):
            config = json.loads(path.read_text())
            tokenizer.chat_template = json_to_jinja(config)[0]
            messages = [dict(role=config.get("user_role", "user"), content="Вопрос\n")]
            for generation in (False, True):
                actual = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation)
                self.assertTrue(actual == direct_render(config, messages, generation), path.name)

        config = json.loads((CONFIGS / "default_foundational.json").read_text())
        config["suffix"] = "\nResponse: "
        tokenizer.chat_template = json_to_jinja(config)[0]
        messages = [dict(role="user", content="Вопрос")]
        actual, _ = render_local_chat_prompt(tokenizer, messages)
        self.assertEqual(actual, "Query: Вопрос\n\n\nResponse: ")
        messages.append(dict(role="assistant", content="Ответ: \n "))
        actual, _ = render_local_chat_prompt(tokenizer, messages)
        self.assertEqual(actual, "Query: Вопрос\n\nResponse: Ответ: \n ")


if __name__ == "__main__":
    unittest.main()
