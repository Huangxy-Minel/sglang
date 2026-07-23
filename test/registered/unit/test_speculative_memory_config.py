"""Unit tests for speculative memory configuration helpers."""

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "python"
    / "sglang"
    / "srt"
    / "speculative"
    / "memory_config.py"
)
SPEC = importlib.util.spec_from_file_location("speculative_memory_config", MODULE_PATH)
memory_config = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = memory_config
SPEC.loader.exec_module(memory_config)


class TestDraftTokenCapacity(unittest.TestCase):
    def test_dense_target_reuses_device_capacity(self):
        self.assertEqual(
            memory_config.resolve_draft_token_capacity(
                target_device_capacity=32832,
                target_logical_capacity=None,
            ),
            32832,
        )

    def test_hisparse_target_uses_logical_capacity(self):
        self.assertEqual(
            memory_config.resolve_draft_token_capacity(
                target_device_capacity=32832,
                target_logical_capacity=295488,
            ),
            295488,
        )

    def test_capacity_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            memory_config.resolve_draft_token_capacity(
                target_device_capacity=0,
                target_logical_capacity=None,
            )


if __name__ == "__main__":
    unittest.main()
