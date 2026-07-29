import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
COMPAT_MODULE_PATH = (
    REPO_ROOT
    / "python/sglang/srt/layers/moe/token_dispatcher/deepep_compat.py"
)


def load_compat_module():
    if not COMPAT_MODULE_PATH.exists():
        return None

    spec = importlib.util.spec_from_file_location(
        "sglang_deepep_compat", COMPAT_MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LegacyBuffer:
    @staticmethod
    def get_dispatch_config(num_ranks):
        return ("legacy", num_ranks)


class BlackwellBuffer:
    @staticmethod
    def get_dispatch_config(num_ranks, real_hidden_bytes=0):
        return ("blackwell", num_ranks, real_hidden_bytes)


class TestDeepEPDispatchConfigCompatibility(unittest.TestCase):
    def setUp(self):
        self.compat = load_compat_module()
        self.assertIsNotNone(
            self.compat,
            "DeepEP compatibility module must provide dispatch config selection",
        )

    def test_passes_real_hidden_bytes_to_blackwell_deepep(self):
        config = self.compat.get_dispatch_config(
            BlackwellBuffer,
            num_ranks=8,
            real_hidden_bytes=6144,
        )

        self.assertEqual(config, ("blackwell", 8, 6144))

    def test_keeps_legacy_deepep_signature_compatible(self):
        config = self.compat.get_dispatch_config(
            LegacyBuffer,
            num_ranks=8,
            real_hidden_bytes=6144,
        )

        self.assertEqual(config, ("legacy", 8))


if __name__ == "__main__":
    unittest.main()
