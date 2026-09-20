import unittest

from llmtf.cli import merge_backend_kwargs, parse_json_object


class ExampleBackend:
    def __init__(self, alpha=1, enabled=True):
        self.alpha = alpha
        self.enabled = enabled


class CliConfigTests(unittest.TestCase):
    def test_json_only_value_survives_cli_defaults(self):
        self.assertEqual(
            merge_backend_kwargs(ExampleBackend, '{"alpha": 9}', {}),
            {'alpha': 9},
        )

    def test_explicit_cli_value_wins_over_json(self):
        self.assertEqual(
            merge_backend_kwargs(
                ExampleBackend, '{"alpha": 9}', {'alpha': 3}
            ),
            {'alpha': 3},
        )

    def test_backend_kwargs_must_be_json_object(self):
        for raw in ('[]', '1', '"value"', '{broken'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_json_object(raw)

    def test_unknown_backend_key_fails_before_construction(self):
        with self.assertRaisesRegex(ValueError, 'typo'):
            merge_backend_kwargs(ExampleBackend, '{"typo": 1}', {})


if __name__ == '__main__':
    unittest.main()
