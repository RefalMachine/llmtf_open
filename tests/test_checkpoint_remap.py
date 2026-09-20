import unittest

from dev.tools.remap_qwen35_checkpoint import remap_key


class CheckpointRemapTests(unittest.TestCase):
    def test_language_model_prefix(self):
        self.assertEqual(
            remap_key(
                "model.language_model.language_model.language_model.layers.0.weight"
            ),
            "model.language_model.layers.0.weight",
        )

    def test_visual_prefix(self):
        self.assertEqual(
            remap_key("model.language_model.visual.patch_embed.weight"),
            "model.visual.patch_embed.weight",
        )

    def test_unrelated_key_is_unchanged(self):
        self.assertEqual(remap_key("lm_head.weight"), "lm_head.weight")


if __name__ == "__main__":
    unittest.main()
