"""Unit tests for HiSparse allocator transaction bookkeeping."""

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "python"
    / "sglang"
    / "srt"
    / "mem_cache"
    / "hisparse_allocator_transaction.py"
)
SPEC = importlib.util.spec_from_file_location(
    "hisparse_allocator_transaction", MODULE_PATH
)
transaction_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = transaction_module
SPEC.loader.exec_module(transaction_module)


class FakeTensor(list):
    def clone(self):
        return FakeTensor(self)


class FakeMapping:
    def __init__(self, values):
        self.values = list(values)

    def __getitem__(self, indices):
        return FakeTensor(self.values[index] for index in indices)

    def __setitem__(self, indices, values):
        for index, value in zip(indices, values):
            self.values[index] = value


class TestHiSparseAllocatorTransaction(unittest.TestCase):
    def test_restore_reverts_only_touched_mapping_entries(self):
        mapping = FakeMapping([0, 11, 0, 0, 44])
        state = transaction_module.HiSparseAllocatorTransaction(
            logical_state=("logical-free", "logical-release"),
            hot_state=("hot-free", "hot-release"),
        )

        state.record_mapping_update(mapping, FakeTensor([2, 3]))
        mapping[FakeTensor([2, 3])] = FakeTensor([22, 33])
        mapping[FakeTensor([4])] = FakeTensor([55])
        state.restore_mapping(mapping)

        self.assertEqual(mapping.values, [0, 11, 0, 0, 55])
        self.assertEqual(
            state.logical_state, ("logical-free", "logical-release")
        )
        self.assertEqual(state.hot_state, ("hot-free", "hot-release"))

    def test_multiple_updates_restore_in_reverse_order(self):
        mapping = FakeMapping([0, 0, 0])
        state = transaction_module.HiSparseAllocatorTransaction("logical", "hot")

        state.record_mapping_update(mapping, FakeTensor([1]))
        mapping[FakeTensor([1])] = FakeTensor([10])
        state.record_mapping_update(mapping, FakeTensor([1]))
        mapping[FakeTensor([1])] = FakeTensor([20])
        state.restore_mapping(mapping)

        self.assertEqual(mapping.values, [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
