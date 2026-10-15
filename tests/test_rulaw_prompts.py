"""Dependency-free contracts for deterministic expanded RuLaw presentation."""

import copy
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1] / "llmtf/tasks/rulaw_proofbench"


def load_module(name):
    spec = importlib.util.spec_from_file_location("rulaw_prompt_test_" + name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prompts = load_module("prompts")
scoring = load_module("scoring")


def sample(kind="numeric", unit="days", family="test:1"):
    return {
        "id": "fixture", "prompt_version": "expanded_v1", "answer_kind": kind,
        "title": "Act", "article": "1", "article_key": family, "as_of": "2025-01-01",
        "case_text": "Act, статья 1. Применяется выбранное положение в состоянии на 2025-01-01.\n"
                     "Исходная область. Не рассматривается X.\n"
                     "Первое обстоятельство не указано; варианты X и Y.\n"
                     "Второе обстоятельство установлено.\nИсходный target.",
        # Deliberately different order: rendering must not zip these with phrases.
        "facts": {"second": True, "first": None},
        "question": "Old instructions", "context": "Полная норма.\n\nСвязанная норма.",
        "gold": {"value": "3", "unit": unit}, "proof": "PRIVATE_PROOF",
        "question_type": "unknown_ambiguous", "unknown_but_determinate": False,
        "mcq": {
            "gold_label": "B", "question": "Old MCQ instructions",
            "options": [{"label": "A", "text": "недостаточно данных", "value": "недостаточно данных"},
                        {"label": "B", "text": "3", "value": "3"}],
        },
    }


class PromptTests(unittest.TestCase):
    def test_all_views_preserve_public_text_without_reading_answers(self):
        row = sample()
        original = copy.deepcopy(row)
        for mode in ("closed", "grounded"):
            for mcq in (False, True):
                with self.subTest(mode=mode, mcq=mcq):
                    text = prompts.render_prompt(row, mode, mcq)
                    parts = row["case_text"].split("\n")
                    self.assertIn("ОБЛАСТЬ РАССМОТРЕНИЯ\n" + "\n".join(parts[:2]), text)
                    self.assertIn("ОБСТОЯТЕЛЬСТВА\n— " + "\n— ".join(parts[2:-1]), text)
                    self.assertIn("ЗАДАНИЕ\n" + parts[-1], text)
                    self.assertEqual(row["context"] in text, mode == "grounded")
                    headers = ["ПРАВИЛА РЕШЕНИЯ", "ФОРМАТ ОТВЕТА"]
                    if mode == "grounded":
                        headers.append("НОРМАТИВНЫЙ ТЕКСТ")
                    headers += ["ОБЛАСТЬ РАССМОТРЕНИЯ", "ОБСТОЯТЕЛЬСТВА", "ЗАДАНИЕ"]
                    self.assertEqual([text.index(h) for h in headers], sorted(text.index(h) for h in headers))
                    self.assertNotIn("Old", text)
                    self.assertNotIn("Ответ:", text)
                    changed = copy.deepcopy(row)
                    changed["gold"]["value"] = "PRIVATE_GOLD"
                    changed["mcq"]["gold_label"] = "PRIVATE_LABEL"
                    changed["proof"] = "OTHER_PRIVATE_PROOF"
                    changed["question_type"] = "complete"
                    changed["unknown_but_determinate"] = True
                    changed["facts"] = {"unrelated_a": False, "unrelated_b": False}
                    self.assertEqual(text, prompts.render_prompt(changed, mode, mcq))
        self.assertEqual(row, original)

    def test_numeric_format_explicitly_allows_special_answers(self):
        for unit in prompts.UNITS:
            row = sample(unit=unit)
            text = prompts.answer_format(row, False)
            for answer in ("не следует из указанной нормы", "недостаточно данных"):
                self.assertIn(answer, text)
                self.assertEqual(scoring.parse_answer(answer, unit), answer)
                self.assertLess(text.index(answer), text.index("Только для числового ответа"))
            self.assertIn(prompts.UNITS[unit], text)
            self.assertEqual("в процентах" in text, unit == "fraction")
            self.assertEqual("Не переводите срок в дни" in text, unit in ("years", "months"))
            for value in ("3", "1,5", "3/2"):
                self.assertIsNotNone(scoring.parse_answer(value, unit))
        self.assertEqual(scoring.parse_answer("25%", "fraction"), "1/4")

    def test_binary_and_categorical_formats(self):
        text = prompts.render_prompt(sample("binary", "none"), "closed")
        self.assertNotIn("не следует из указанной нормы", text)
        self.assertIn("отрицательный", text)
        for family, answers in {
            "gk:163": ["ничтожна"], "gk:186": ["1 год", "ничтожна", "до отмены"],
        }.items():
            row = sample("categorical", "none", family)
            text = prompts.answer_format(row, False)
            for answer in answers:
                self.assertIn(answer, text)
                self.assertEqual(scoring.parse_answer(answer, "none"), answer)
        with self.assertRaisesRegex(ValueError, "categorical family.*fixture"):
            prompts.render_prompt(sample("categorical", "none", "unseen:1"), "closed")

    def test_mcq_options_and_instruction_do_not_require_text_answers(self):
        row = sample()
        text = prompts.render_prompt(row, "grounded", True)
        self.assertTrue(text.endswith("A. недостаточно данных\nB. 3\n\n" + prompts.MCQ_END))
        self.assertNotIn("Запишите число", text)
        self.assertNotIn("Допустимы три формы", text)
        self.assertNotIn(prompts.OPEN_END, text)

    def test_invalid_source_structure_fails_with_id(self):
        row = sample()
        cases = [
            {"prompt_version": "unknown"}, {"case_text": None},
            {"case_text": row["case_text"] + "\nextra"},
            {"case_text": row["case_text"].replace("Исходная область. Не рассматривается X.", "")},
            {"title": "different"}, {"as_of": "2026-01-01"},
            {"facts": {}}, {"facts": []}, {"answer_kind": "unknown"},
            {"gold": {"unit": "unknown"}},
        ]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "fixture"):
                prompts.render_prompt(dict(row, **changes), "closed")
        del row["case_text"]
        with self.assertRaisesRegex(ValueError, "fixture"):
            prompts.render_prompt(row, "closed")


if __name__ == "__main__":
    unittest.main()
