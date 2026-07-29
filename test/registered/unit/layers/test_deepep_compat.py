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

    @staticmethod
    def get_combine_config(num_ranks):
        return ("legacy-combine", num_ranks)


class BlackwellBuffer:
    @staticmethod
    def get_dispatch_config(num_ranks, real_hidden_bytes=0):
        return ("blackwell", num_ranks, real_hidden_bytes)

    @staticmethod
    def get_combine_config(num_ranks, real_hidden_bytes=0):
        return ("blackwell-combine", num_ranks, real_hidden_bytes)


class DispatchHintConfig:
    def get_nvl_buffer_size_hint(self, hidden_bytes, num_ranks):
        return hidden_bytes + 10_000 + num_ranks

    def get_rdma_buffer_size_hint(self, hidden_bytes, num_ranks):
        return hidden_bytes + num_ranks


class CombineHintConfig:
    def get_nvl_buffer_size_hint(self, hidden_bytes, num_ranks):
        return hidden_bytes + num_ranks

    def get_rdma_buffer_size_hint(self, hidden_bytes, num_ranks):
        return hidden_bytes + 10_000 + num_ranks


class TestDeepEPDispatchConfigCompatibility(unittest.TestCase):
    def setUp(self):
        self.compat = load_compat_module()
        self.assertIsNotNone(
            self.compat,
            "DeepEP compatibility module must provide dispatch config selection",
        )

    def require_helper(self, name):
        helper = getattr(self.compat, name, None)
        self.assertIsNotNone(helper, f"DeepEP compatibility helper {name} must exist")
        return helper

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

    def test_passes_real_hidden_bytes_to_blackwell_combine_config(self):
        get_combine_config = self.require_helper("get_combine_config")

        config = get_combine_config(
            BlackwellBuffer,
            num_ranks=8,
            real_hidden_bytes=12288,
        )

        self.assertEqual(config, ("blackwell-combine", 8, 12288))

    def test_keeps_legacy_combine_signature_compatible(self):
        get_combine_config = self.require_helper("get_combine_config")

        config = get_combine_config(
            LegacyBuffer,
            num_ranks=8,
            real_hidden_bytes=12288,
        )

        self.assertEqual(config, ("legacy-combine", 8))

    def test_enables_fp8_for_deepgemm_normal_dispatch(self):
        use_fp8_normal_dispatch = self.require_helper("use_fp8_normal_dispatch")

        self.assertTrue(
            use_fp8_normal_dispatch(
                enable_jit_deepgemm=True,
                is_cutlass=False,
                force_bf16_dispatch=False,
            )
        )

    def test_disables_fp8_when_bf16_dispatch_is_forced(self):
        use_fp8_normal_dispatch = self.require_helper("use_fp8_normal_dispatch")

        self.assertFalse(
            use_fp8_normal_dispatch(
                enable_jit_deepgemm=True,
                is_cutlass=False,
                force_bf16_dispatch=True,
            )
        )

    def test_disables_fp8_for_cutlass_runner(self):
        use_fp8_normal_dispatch = self.require_helper("use_fp8_normal_dispatch")

        self.assertFalse(
            use_fp8_normal_dispatch(
                enable_jit_deepgemm=True,
                is_cutlass=True,
                force_bf16_dispatch=False,
            )
        )

    def test_disables_fp8_without_deepgemm(self):
        use_fp8_normal_dispatch = self.require_helper("use_fp8_normal_dispatch")

        self.assertFalse(
            use_fp8_normal_dispatch(
                enable_jit_deepgemm=False,
                is_cutlass=False,
                force_bf16_dispatch=False,
            )
        )

    def test_computes_fp8_dispatch_payload_bytes(self):
        get_hidden_bytes = self.require_helper("get_normal_dispatch_hidden_bytes")

        self.assertEqual(
            get_hidden_bytes(hidden_size=6144, use_fp8_dispatch=True),
            6144,
        )

    def test_computes_bf16_dispatch_payload_bytes(self):
        get_hidden_bytes = self.require_helper("get_normal_dispatch_hidden_bytes")

        self.assertEqual(
            get_hidden_bytes(hidden_size=6144, use_fp8_dispatch=False),
            12288,
        )

    def test_sizes_dispatch_and_combine_hints_with_independent_payloads(self):
        get_size_hints = self.require_helper("get_normal_buffer_size_hints")

        nvl_bytes, rdma_bytes = get_size_hints(
            dispatch_config=DispatchHintConfig(),
            combine_config=CombineHintConfig(),
            dispatch_hidden_bytes=6144,
            combine_hidden_bytes=12288,
            num_ranks=8,
        )

        self.assertEqual(nvl_bytes, 16152)
        self.assertEqual(rdma_bytes, 22296)


if __name__ == "__main__":
    unittest.main()
