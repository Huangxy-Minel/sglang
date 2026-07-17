"""Pure unit tests for the standalone one-batch benchmark helpers."""

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "python"
    / "sglang"
    / "bench_one_batch_utils.py"
)
SPEC = importlib.util.spec_from_file_location("bench_one_batch_utils", MODULE_PATH)
bench_utils = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = bench_utils
SPEC.loader.exec_module(bench_utils)


class TestLocalRankAssignments(unittest.TestCase):
    def test_two_nodes_split_global_ranks_across_local_gpus(self):
        self.assertEqual(
            bench_utils.get_local_rank_assignments(16, 2, 0),
            [(rank, rank) for rank in range(8)],
        )
        self.assertEqual(
            bench_utils.get_local_rank_assignments(16, 2, 1),
            [(rank, rank - 8) for rank in range(8, 16)],
        )

    def test_rank_assignment_rejects_invalid_topology(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            bench_utils.get_local_rank_assignments(15, 2, 0)
        with self.assertRaisesRegex(ValueError, "node_rank"):
            bench_utils.get_local_rank_assignments(16, 2, 2)


class TestChunkPlan(unittest.TestCase):
    def test_unspecified_chunk_size_keeps_one_shot_prefill(self):
        plan = bench_utils.build_chunk_plan(
            input_lengths=[4096] * 32,
            requested_chunk_size=None,
            effective_chunk_size=8192,
            page_size=64,
        )
        self.assertFalse(plan.enabled)
        self.assertEqual(plan.per_request_chunk_size, 4096)
        self.assertEqual(plan.num_chunks, 1)
        self.assertEqual(plan.bounds, ((0, 4096),))

    def test_negative_chunk_size_keeps_one_shot_prefill(self):
        plan = bench_utils.build_chunk_plan(
            input_lengths=[4096] * 32,
            requested_chunk_size=-1,
            effective_chunk_size=-1,
            page_size=64,
        )
        self.assertFalse(plan.enabled)
        self.assertEqual(plan.bounds, ((0, 4096),))

    def test_explicit_chunk_size_uses_effective_dp_adjusted_budget(self):
        plan = bench_utils.build_chunk_plan(
            input_lengths=[4096] * 32,
            requested_chunk_size=65536,
            effective_chunk_size=4096,
            page_size=64,
        )
        self.assertTrue(plan.enabled)
        self.assertEqual(plan.requested_chunk_size, 65536)
        self.assertEqual(plan.effective_chunk_size, 4096)
        self.assertEqual(plan.per_request_chunk_size, 128)
        self.assertEqual(plan.num_chunks, 32)
        self.assertEqual(plan.bounds[0], (0, 128))
        self.assertEqual(plan.bounds[-1], (3968, 4096))

    def test_chunk_size_is_page_aligned_and_final_chunk_keeps_remainder(self):
        plan = bench_utils.build_chunk_plan(
            input_lengths=[1000] * 3,
            requested_chunk_size=1000,
            effective_chunk_size=1000,
            page_size=64,
        )
        self.assertEqual(plan.per_request_chunk_size, 320)
        self.assertEqual(plan.bounds, ((0, 320), (320, 640), (640, 960), (960, 1000)))

    def test_too_small_chunk_budget_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one page per request"):
            bench_utils.build_chunk_plan(
                input_lengths=[4096] * 32,
                requested_chunk_size=1024,
                effective_chunk_size=1024,
                page_size=64,
            )

    def test_unequal_input_lengths_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "equal-length"):
            bench_utils.build_chunk_plan(
                input_lengths=[4096, 2048],
                requested_chunk_size=4096,
                effective_chunk_size=4096,
                page_size=64,
            )

    def test_prepare_chunk_reuses_request_slot_and_committed_prefix(self):
        class CloneableList(list):
            def __init__(self, values=(), dtype="int32"):
                super().__init__(values)
                self.dtype = dtype

            def clone(self):
                return CloneableList(self, dtype=self.dtype)

            def long(self):
                return CloneableList(self, dtype="int64")

        class FakeReqToToken:
            def __init__(self):
                self.rows = [CloneableList(range(512))]

            def __getitem__(self, key):
                row, column_slice = key
                return CloneableList(self.rows[row][column_slice])

        class FakeReq:
            req_pool_idx = 0
            fill_ids = []
            prefix_indices = CloneableList()
            extend_input_len = 0

            def set_extend_input_len(self, value):
                self.extend_input_len = value

        req = FakeReq()
        full_input_ids = [list(range(512))]
        bench_utils.prepare_chunk_requests(
            reqs=[req],
            full_input_ids=full_input_ids,
            start=128,
            end=256,
            req_to_token=FakeReqToToken(),
        )
        self.assertEqual(req.req_pool_idx, 0)
        self.assertEqual(req.prefix_indices, list(range(128)))
        self.assertEqual(req.prefix_indices.dtype, "int64")
        self.assertEqual(req.fill_ids, list(range(256)))
        self.assertEqual(req.extend_input_len, 128)


class TestClusterMetrics(unittest.TestCase):
    def test_metrics_use_global_batch_and_slowest_rank_latencies(self):
        metrics = bench_utils.build_cluster_metrics(
            batch_size=32,
            dp_size=16,
            input_len=4096,
            output_len=3,
            cluster_prefill_latency=4.0,
            cluster_median_decode_latency=1.0,
            cluster_total_latency=5.25,
        )
        self.assertEqual(metrics["global_batch_size"], 512)
        self.assertEqual(metrics["cluster_prefill_latency"], 4.0)
        self.assertEqual(metrics["cluster_prefill_throughput"], 524288.0)
        self.assertEqual(metrics["cluster_median_decode_latency"], 1.0)
        self.assertAlmostEqual(metrics["cluster_median_decode_throughput"], 512.0)
        self.assertEqual(metrics["cluster_total_latency"], 5.25)
        self.assertAlmostEqual(
            metrics["cluster_overall_throughput"],
            (4096 + 3) * 512 / 5.25,
        )


class TestWarmupPrecompileBarriers(unittest.TestCase):
    def test_barriers_are_enabled_only_inside_warmup_scope(self):
        class FakeFlag:
            def __init__(self):
                self.value = False

            def override(self, value):
                flag = self

                class Override:
                    def __enter__(self):
                        self.original = flag.value
                        flag.value = value

                    def __exit__(self, exc_type, exc_value, traceback):
                        flag.value = self.original

                return Override()

        flag = FakeFlag()
        self.assertFalse(flag.value)
        with bench_utils.enable_deepep_precompile_barriers_for_warmup(flag):
            self.assertTrue(flag.value)
        self.assertFalse(flag.value)

    def test_barrier_flag_is_restored_when_warmup_fails(self):
        class FakeFlag:
            value = False

            def override(self, value):
                flag = self

                class Override:
                    def __enter__(self):
                        flag.value = value

                    def __exit__(self, exc_type, exc_value, traceback):
                        flag.value = False

                return Override()

        flag = FakeFlag()
        with self.assertRaisesRegex(RuntimeError, "warmup failed"):
            with bench_utils.enable_deepep_precompile_barriers_for_warmup(flag):
                self.assertTrue(flag.value)
                raise RuntimeError("warmup failed")
        self.assertFalse(flag.value)


if __name__ == "__main__":
    unittest.main()
