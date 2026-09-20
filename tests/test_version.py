import unittest

import llmtf


class VersionTests(unittest.TestCase):
    def test_release_version(self):
        self.assertEqual(llmtf.__version__, "0.3.0")


if __name__ == "__main__":
    unittest.main()
