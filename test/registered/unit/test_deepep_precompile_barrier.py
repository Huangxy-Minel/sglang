"""CPU-only tests for the DeepEP precompile barrier helper."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace


DEEPEP_PATH = (
    Path(__file__).resolve().parents[3]
    / "python"
    / "sglang"
    / "srt"
    / "layers"
    / "moe"
    / "token_dispatcher"
    / "deepep.py"
)


def load_barrier_helper(enabled, barrier):
    tree = ast.parse(DEEPEP_PATH.read_text())
    helper = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_deepep_precompile_tp_barrier"
        ),
        None,
    )
    if helper is None:
        raise AssertionError("DeepEP precompile barrier helper is missing")

    envs = SimpleNamespace(
        SGLANG_IN_DEEPGEMM_PRECOMPILE_STAGE=SimpleNamespace(get=lambda: enabled)
    )
    namespace = {
        "envs": envs,
        "get_tp_group": lambda: SimpleNamespace(barrier=barrier),
    }
    module = ast.Module(body=[helper], type_ignores=[])
    exec(compile(module, str(DEEPEP_PATH), "exec"), namespace)
    return namespace["_deepep_precompile_tp_barrier"]


class TestDeepEPPrecompileBarrier(unittest.TestCase):
    def test_disabled_precompile_stage_does_not_barrier(self):
        calls = []
        load_barrier_helper(False, lambda: calls.append("barrier"))()
        self.assertEqual(calls, [])

    def test_enabled_precompile_stage_barriers_once(self):
        calls = []
        load_barrier_helper(True, lambda: calls.append("barrier"))()
        self.assertEqual(calls, ["barrier"])


if __name__ == "__main__":
    unittest.main()
