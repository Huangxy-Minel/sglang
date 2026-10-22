"""CPU tests of the real v1 methods, without GPU-only module imports."""

import ast
import copy
import importlib.util
import logging
import unittest
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]

# Register CPU CI without importing sglang's GPU/model dependencies.
_ci_spec = importlib.util.spec_from_file_location(
    "ci_register", ROOT / "python/sglang/test/ci/ci_register.py"
)
_ci = importlib.util.module_from_spec(_ci_spec)
_ci_spec.loader.exec_module(_ci)
register_cpu_ci = _ci.register_cpu_ci
register_cpu_ci(est_time=1, suite="base-a-test-cpu")
SOURCE = ROOT / "python/sglang/srt/layers/moe/token_dispatcher/deepep.py"


def load_class(name, namespace):
    node = next(
        n
        for n in ast.parse(SOURCE.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == name
    )
    # Base behavior is tested separately; avoid loading unrelated GPU imports.
    node.bases = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    return namespace[name]


class Mode(Enum):
    NORMAL = "normal"
    LOW_LATENCY = "low_latency"
    AUTO = "auto"

    def enable_normal(self):
        return self != Mode.LOW_LATENCY

    def enable_low_latency(self):
        return self != Mode.NORMAL


class Dtype(Enum):
    FP8 = "fp8"
    BF16 = "bf16"
    INT8 = "int8"
    MXFP4 = "mxfp4"
    MXFP8 = "mxfp8"
    NVFP4 = "nvfp4"


class Hint:
    def __init__(self, hidden_bytes, dispatch):
        self.hidden_bytes = hidden_bytes
        self.dispatch = dispatch

    def get_nvl_buffer_size_hint(self, hidden_bytes, ranks):
        assert self.hidden_bytes == hidden_bytes
        return hidden_bytes + (10000 if self.dispatch else 0)

    def get_rdma_buffer_size_hint(self, hidden_bytes, ranks):
        assert self.hidden_bytes == hidden_bytes
        return hidden_bytes + (10000 if not self.dispatch else 0)


class FakeBuffer:
    @staticmethod
    def get_dispatch_config(ranks, real_hidden_bytes=0):
        return Hint(real_hidden_bytes, True)

    @staticmethod
    def get_combine_config(ranks, real_hidden_bytes=0):
        return Hint(real_hidden_bytes, False)

    @staticmethod
    def get_low_latency_rdma_size_hint(*args):
        return 500

    def __init__(self, group, nvl, rdma, **kwargs):
        self.nvl, self.rdma = nvl, rdma
        self.low_latency_mode = kwargs["low_latency_mode"]


class TestV1Payload(unittest.TestCase):
    def setUp(self):
        self.resources = SimpleNamespace(buffers={})
        self.config = SimpleNamespace(
            normal_dispatch_config=None, normal_combine_config=None, num_sms=24
        )
        self.wrapper = SimpleNamespace(ENABLE_JIT_DEEPGEMM=True)
        self.namespace = dict(
            get_resources=lambda: self.resources,
            Buffer=FakeBuffer,
            DeepEPConfig=SimpleNamespace(get_instance=lambda: self.config),
            DeepEPMode=Mode,
            DeepEPDispatchMode=Mode,
            DispatcherOutputDtype=Dtype,
            _is_npu=True,
            _is_mnnvl_fabric_supported=lambda: False,
            get_cuda_version=lambda: (12, 0),
            deep_gemm_wrapper=self.wrapper,
            copy=copy.copy,
            logger=logging.getLogger(__name__),
            get_server_args=lambda: True,
            get_exec=lambda: self.exec_config,
            get_moe_runner_backend=lambda: SimpleNamespace(
                is_flashinfer_cutedsl=lambda: False,
                is_cutlass=lambda: self.cutlass,
                is_humming=lambda: False,
            ),
            envs=SimpleNamespace(
                SGLANG_DEEPEP_BF16_DISPATCH=SimpleNamespace(get=lambda: False)
            ),
        )
        compat_path = SOURCE.with_name("deepep_compat.py")
        if compat_path.exists():
            spec = importlib.util.spec_from_file_location("compat", compat_path)
            compat = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(compat)
            self.namespace.update(
                {
                    name: getattr(compat, name)
                    for name in dir(compat)
                    if not name.startswith("_")
                }
            )
        self.exec_config = SimpleNamespace(
            moe=SimpleNamespace(deepep_dispatcher_output_dtype="auto")
        )
        self.cutlass = False
        utils_path = SOURCE.parents[1] / "utils.py"
        dtype_function = next(
            n
            for n in ast.parse(utils_path.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == "get_deepep_output_dtype"
        )
        exec(
            compile(
                ast.Module(body=[dtype_function], type_ignores=[]),
                str(utils_path),
                "exec",
            ),
            self.namespace,
        )
        self.buffer = load_class("DeepEPBuffer", self.namespace)
        self.base = load_class("_DeepEPDispatcherImplBase", self.namespace)

    def impl(self, name):
        cls = load_class(name, self.namespace)
        obj = cls.__new__(cls)
        obj._normal_dispatch_hidden_bytes = (
            self.base._normal_dispatch_hidden_bytes.__get__(obj)
        )
        obj.group = SimpleNamespace(size=lambda: 8)
        obj.hidden_size, obj.params_bytes = 6144, 2
        obj.deepep_mode = Mode.AUTO
        obj.num_max_dispatch_tokens_per_rank, obj.num_experts = 256, 256
        obj.quant_config = {
            "normal_dispatcher_output_dtype": "fp8",
            "low_latency_dispatcher_output_dtype": "bf16",
        }
        return obj

    def test_auto_ll_first_uses_normal_dtype_for_buffer_and_reuses_resource(self):
        self.assertTrue(
            hasattr(self.base, "_normal_dispatch_hidden_bytes"),
            "normal wire payload must be derived independently of LL dtype",
        )
        ll = self.impl("_DeepEPDispatcherImplLowLatency")
        result = ll._get_buffer()
        self.assertEqual((result.nvl, result.rdma), (16144, 22288))
        self.assertEqual(self.buffer.configured_buffer_bytes(), 38432)
        normal = self.impl("_DeepEPDispatcherImplNormal")
        self.assertIs(normal._get_buffer(), result)
        self.assertIs(self.resources.buffers["deepep_ep_state"].buffer, result)
        self.assertEqual(ll.dispatch_mode, Mode.LOW_LATENCY)

    def test_explicit_bf16_and_disabled_deepgemm_keep_two_byte_normal_payload(self):
        self.assertTrue(
            hasattr(self.base, "_normal_dispatch_hidden_bytes"),
            "normal wire payload helper missing",
        )
        for dtype, jit in (("bf16", True), ("fp8", False), ("nvfp4", True)):
            with self.subTest(dtype=dtype, jit=jit):
                self.resources.buffers.clear()
                self.wrapper.ENABLE_JIT_DEEPGEMM = jit
                obj = self.impl("_DeepEPDispatcherImplLowLatency")
                obj.quant_config["normal_dispatcher_output_dtype"] = dtype
                result = obj._get_buffer()
                self.assertEqual((result.nvl, result.rdma), (22288, 22288))

    def test_server_dtype_override_takes_precedence_over_normal_quant_override(self):
        obj = self.impl("_DeepEPDispatcherImplLowLatency")
        self.exec_config.moe.deepep_dispatcher_output_dtype = "bf16"
        result = obj._get_buffer()
        self.assertEqual((result.nvl, result.rdma), (22288, 22288))

    def test_cutlass_default_uses_bf16_without_quant_override(self):
        self.cutlass = True
        obj = self.impl("_DeepEPDispatcherImplLowLatency")
        obj.quant_config = {}
        result = obj._get_buffer()
        self.assertEqual((result.nvl, result.rdma), (22288, 22288))

    def test_custom_configs_remain_authoritative_and_hints_are_separate(self):
        self.config.normal_dispatch_config = Hint(6144, True)
        self.config.normal_combine_config = Hint(12288, False)
        # Deliberately distinct dispatch and combine payload widths.
        result = self.buffer.get_deepep_buffer(
            SimpleNamespace(size=lambda: 8),
            6144,
            2,
            Mode.NORMAL,
            dispatch_hidden_bytes=6144,
        )
        self.assertEqual((result.nvl, result.rdma), (16144, 22288))

    def test_byte_accounting_is_zero_before_allocation_and_follows_resources(self):
        self.assertTrue(
            hasattr(self.buffer, "configured_buffer_bytes"),
            "resource-owned byte accounting missing",
        )
        self.assertEqual(self.buffer.configured_buffer_bytes(), 0)
        self.buffer.get_deepep_buffer(
            SimpleNamespace(size=lambda: 8), 6144, 2, Mode.NORMAL
        )
        self.assertEqual(self.buffer.configured_buffer_bytes(), 44576)
        self.resources = SimpleNamespace(buffers={})
        self.assertEqual(self.buffer.configured_buffer_bytes(), 0)


if __name__ == "__main__":
    unittest.main()
