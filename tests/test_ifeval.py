import unittest
from unittest import mock

import langdetect

from llmtf.tasks.ifeval import ru_instructions


class RuIFEvalLanguageTests(unittest.TestCase):
    def test_response_language_uses_local_langdetect(self):
        checker = ru_instructions.ResponseLanguageChecker(
            'language:response_language'
        )
        checker.build_description(language='ru')
        with mock.patch.object(
            ru_instructions.langdetect, 'detect', return_value='ru'
        ) as detect:
            self.assertTrue(checker.check_following('Русский ответ'))
        detect.assert_called_once_with('Русский ответ')

        with mock.patch.object(
            ru_instructions.langdetect, 'detect', return_value='en'
        ):
            self.assertFalse(checker.check_following('English answer'))

    def test_undetectable_language_does_not_pass(self):
        checker = ru_instructions.ResponseLanguageChecker(
            'language:response_language'
        )
        checker.build_description(language='ru')
        with mock.patch.object(
            ru_instructions.langdetect,
            'detect',
            side_effect=langdetect.LangDetectException(0, 'no features'),
        ):
            self.assertFalse(checker.check_following('...'))

    def test_case_checkers_require_russian_and_requested_case(self):
        upper = ru_instructions.CapitalLettersEnglishChecker(
            'change_case:english_capital'
        )
        lower = ru_instructions.LowercaseLettersEnglishChecker(
            'change_case:english_lowercase'
        )
        with mock.patch.object(
            ru_instructions.langdetect, 'detect', return_value='ru'
        ):
            self.assertTrue(upper.check_following('РУССКИЙ ОТВЕТ'))
            self.assertFalse(upper.check_following('Русский ответ'))
            self.assertTrue(lower.check_following('русский ответ'))
            self.assertFalse(lower.check_following('Русский ответ'))
        with mock.patch.object(
            ru_instructions.langdetect, 'detect', return_value='en'
        ):
            self.assertFalse(upper.check_following('ENGLISH ANSWER'))
            self.assertFalse(lower.check_following('english answer'))


if __name__ == '__main__':
    unittest.main()
