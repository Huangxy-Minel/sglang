import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.mem_cache.base_prefix_cache import DecLockRefResult, IncLockRefResult
from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseTokenToKVPoolAllocator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


class FakeSubAllocator:
    def __init__(self, size: int, available: int):
        self.size = size
        self._available = available

    def available_size(self):
        return self._available


class FakeHostPool:
    def __init__(self, size: int, available: int, size_per_token: int = 1):
        self.size = size
        self._available = available
        self.size_per_token = size_per_token

    def available_size(self):
        return self._available


class TestHiSparseCapacity(unittest.TestCase):
    def make_hisparse_allocator(
        self,
        *,
        logical_size=1000,
        logical_available=1000,
        hot_size=100,
        hot_available=100,
    ):
        allocator = HiSparseTokenToKVPoolAllocator.__new__(
            HiSparseTokenToKVPoolAllocator
        )
        allocator.logical_attn_allocator = FakeSubAllocator(
            logical_size, logical_available
        )
        allocator.hisparse_attn_allocator = FakeSubAllocator(hot_size, hot_available)
        allocator.host_pool = None
        return allocator

    def make_tree_cache(self, evictable_size=0):
        tree_cache = MagicMock()
        tree_cache.evictable_size.return_value = evictable_size
        tree_cache.full_evictable_size.return_value = evictable_size
        tree_cache.supports_mamba.return_value = False
        tree_cache.supports_swa.return_value = False
        tree_cache.is_tree_cache.return_value = False
        tree_cache.disable = False
        tree_cache.inc_lock_ref.return_value = IncLockRefResult()
        tree_cache.dec_lock_ref.return_value = DecLockRefResult()
        return tree_cache

    def make_running_batch(self):
        batch = MagicMock()
        batch.reqs = []
        batch.batch_size.return_value = 0
        return batch

    def make_req(self, *, extend_input_len, max_new_tokens):
        req = MagicMock()
        req.extend_input_len = extend_input_len
        req.host_hit_length = 0
        req.output_ids = []
        req.prefix_indices = []
        req.fill_ids = list(range(extend_input_len))
        req.last_node = MagicMock()
        req.sampling_params = SimpleNamespace(
            max_new_tokens=max_new_tokens, ignore_eos=False
        )
        return req

    def make_adder(self, allocator, *, tree_cache=None):
        return PrefillAdder(
            page_size=1,
            tree_cache=tree_cache or self.make_tree_cache(),
            token_to_kv_pool_allocator=allocator,
            running_batch=self.make_running_batch(),
            new_token_ratio=1.0,
            rem_input_tokens=10000,
            rem_chunk_tokens=None,
        )

    def test_available_size_keeps_hot_pool_semantics(self):
        allocator = self.make_hisparse_allocator(
            logical_size=1000, logical_available=800, hot_size=100, hot_available=60
        )

        self.assertEqual(allocator.available_size(), 60)

    def test_scheduling_available_size_uses_logical_and_host_capacity(self):
        allocator = self.make_hisparse_allocator(
            logical_size=1000, logical_available=700, hot_size=100, hot_available=30
        )
        allocator.attach_host_pool(FakeHostPool(size=900, available=650))

        self.assertEqual(allocator.hot_available_size(), 30)
        self.assertEqual(allocator.logical_available_size(), 700)
        self.assertEqual(allocator.host_available_size(), 650)
        self.assertEqual(allocator.scheduling_available_size(), 650)
        self.assertEqual(allocator.scheduling_available_size(evictable_size=100), 650)

    def test_scheduling_available_size_falls_back_to_logical_without_host_pool(self):
        allocator = self.make_hisparse_allocator(
            logical_size=1000, logical_available=700, hot_size=100, hot_available=30
        )

        self.assertEqual(allocator.scheduling_available_size(), 700)
        self.assertEqual(allocator.scheduling_available_size(evictable_size=50), 750)

    def test_capacity_stats_reports_hot_logical_and_host_breakdown(self):
        allocator = self.make_hisparse_allocator(
            logical_size=1000, logical_available=700, hot_size=100, hot_available=30
        )
        allocator.attach_host_pool(
            FakeHostPool(size=900, available=650, size_per_token=4)
        )

        stats = allocator.capacity_stats()

        self.assertEqual(
            stats["hot"],
            {"used": 70, "available": 30, "total": 100, "usage": 0.7},
        )
        self.assertEqual(
            stats["logical"],
            {"used": 300, "available": 700, "total": 1000, "usage": 0.3},
        )
        self.assertEqual(
            stats["host"],
            {
                "used": 250,
                "available": 650,
                "total": 900,
                "usage": 250 / 900,
                "bytes": 3600,
            },
        )
        self.assertEqual(stats["effective_available"], 30)
        self.assertEqual(stats["scheduling_available"], 650)

    def test_coordinator_capacity_stats_adds_hisparse_config(self):
        allocator = self.make_hisparse_allocator(
            logical_size=1000, logical_available=700, hot_size=100, hot_available=30
        )
        allocator.attach_host_pool(
            FakeHostPool(size=900, available=650, size_per_token=4)
        )
        coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
        coordinator.token_to_kv_pool_allocator = allocator
        coordinator.host_to_device_ratio = 6
        coordinator.top_k = 2048
        coordinator.device_buffer_size = 4096
        coordinator.ack_staging_queue = [object(), object()]

        stats = coordinator.capacity_stats()

        self.assertEqual(stats["host"]["bytes"], 3600)
        self.assertEqual(stats["host_to_device_ratio"], 6)
        self.assertEqual(stats["top_k"], 2048)
        self.assertEqual(stats["device_buffer_size"], 4096)
        self.assertEqual(stats["staging_queue_len"], 2)

    def test_prefill_admission_uses_scheduling_capacity_not_hot_capacity(self):
        allocator = self.make_hisparse_allocator(
            logical_size=1000, logical_available=1000, hot_size=100, hot_available=100
        )
        allocator.attach_host_pool(FakeHostPool(size=900, available=900))
        adder = self.make_adder(allocator)

        result = adder.add_one_req(
            self.make_req(extend_input_len=80, max_new_tokens=400),
            has_chunked_req=False,
            truncation_align_size=None,
        )

        self.assertEqual(result, AddReqResult.CONTINUE)
        self.assertEqual(len(adder.can_run_list), 1)

    def test_prefill_records_no_token_when_scheduling_capacity_is_exhausted(self):
        allocator = self.make_hisparse_allocator(
            logical_size=1000, logical_available=1000, hot_size=100, hot_available=100
        )
        allocator.attach_host_pool(FakeHostPool(size=900, available=900))
        adder = self.make_adder(allocator)

        result = adder.add_one_req(
            self.make_req(extend_input_len=80, max_new_tokens=900),
            has_chunked_req=False,
            truncation_align_size=None,
        )

        self.assertEqual(result, AddReqResult.NO_TOKEN)
        self.assertEqual(adder.no_token_reject_count, 1)
        self.assertEqual(adder.last_no_token_total_tokens, 980)
        self.assertEqual(adder.last_no_token_effective_available, 900)

    def test_decode_mem_uses_scheduling_capacity_for_hisparse_allocator(self):
        allocator = self.make_hisparse_allocator(
            logical_size=1000, logical_available=1000, hot_size=100, hot_available=0
        )
        allocator.attach_host_pool(FakeHostPool(size=900, available=900))
        batch = ScheduleBatch.__new__(ScheduleBatch)
        batch.token_to_kv_pool_allocator = allocator
        batch.tree_cache = MagicMock()
        batch.new_tokens_required_next_decode = MagicMock(return_value=1)
        batch.is_spec_v2 = False

        self.assertTrue(ScheduleBatch.check_decode_mem(batch))


if __name__ == "__main__":
    unittest.main()
