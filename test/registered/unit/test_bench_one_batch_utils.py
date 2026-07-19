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


class TestDeepEPMicroWarmup(unittest.TestCase):
    def test_auto_and_normal_modes_get_page_aligned_prefill_only_shape(self):
        self.assertEqual(
            bench_utils.build_deepep_micro_warmup_shape(
                moe_a2a_backend="deepep",
                deepep_mode="auto",
                page_size=64,
            ),
            (1, 64, 1),
        )
        self.assertEqual(
            bench_utils.build_deepep_micro_warmup_shape(
                moe_a2a_backend="deepep",
                deepep_mode="normal",
                page_size=128,
            ),
            (1, 128, 1),
        )

    def test_micro_warmup_is_disabled_without_normal_deepep(self):
        self.assertIsNone(
            bench_utils.build_deepep_micro_warmup_shape(
                moe_a2a_backend="deepep",
                deepep_mode="low_latency",
                page_size=64,
            )
        )
        self.assertIsNone(
            bench_utils.build_deepep_micro_warmup_shape(
                moe_a2a_backend="none",
                deepep_mode="auto",
                page_size=64,
            )
        )

    def test_micro_input_has_at_least_64_tokens_and_is_page_aligned(self):
        self.assertEqual(
            bench_utils.build_deepep_micro_warmup_shape(
                moe_a2a_backend="deepep",
                deepep_mode="auto",
                page_size=24,
            ),
            (1, 72, 1),
        )


class TestClusterMetrics(unittest.TestCase):
    def test_phase_breakdown_comes_from_the_slowest_prefill_rank(self):
        values = bench_utils.values_from_slowest_rank(
            [
                (5.0, 3.0, 1.0, 1.0),
                (6.0, 2.0, 3.0, 1.0),
                (4.0, 2.0, 1.0, 1.0),
            ]
        )
        self.assertEqual(values, (6.0, 2.0, 3.0, 1.0))

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

    def test_metrics_distinguish_requested_and_admitted_decode_batch(self):
        metrics = bench_utils.build_cluster_metrics(
            batch_size=21,
            requested_batch_size=128,
            dp_size=16,
            input_len=4096,
            output_len=128,
            cluster_prefill_latency=8.0,
            cluster_median_decode_latency=0.1,
            cluster_total_latency=9.0,
        )
        self.assertEqual(metrics["requested_batch_size"], 128)
        self.assertEqual(metrics["batch_size"], 21)
        self.assertEqual(metrics["requested_global_batch_size"], 2048)
        self.assertEqual(metrics["global_batch_size"], 336)


class TestHiSparseWaveCapacity(unittest.TestCase):
    def _snapshot(self, **overrides):
        values = dict(
            hot_total=20000,
            hot_available=20000,
            logical_total=200000,
            logical_available=200000,
            host_total=200000,
            host_available=200000,
            request_slots_available=128,
            max_context_len=32768,
        )
        values.update(overrides)
        return bench_utils.HiSparseCapacitySnapshot(**values)

    def test_wave_chunk_budget_is_per_dp_group_request(self):
        dp16 = bench_utils.build_wave_chunk_plan(
            input_len=32768,
            requested_chunk_size=65536,
            effective_chunk_size=4096,
            page_size=64,
        )
        dp8 = bench_utils.build_wave_chunk_plan(
            input_len=32768,
            requested_chunk_size=65536,
            effective_chunk_size=8192,
            page_size=64,
        )
        self.assertEqual(dp16.per_request_chunk_size, 4096)
        self.assertEqual(dp16.num_chunks, 8)
        self.assertEqual(dp8.per_request_chunk_size, 8192)
        self.assertEqual(dp8.num_chunks, 4)

    def test_capacity_reserves_prefill_peak_and_future_decode(self):
        decision = bench_utils.evaluate_hisparse_wave_admission(
            snapshot=self._snapshot(),
            ready_count=1,
            requested_batch_size=4,
            input_len=4096,
            output_len=128,
            page_size=64,
            device_buffer_size=4096,
        )
        self.assertTrue(decision.can_admit)
        self.assertEqual(decision.stop_reason, "admitted")
        self.assertEqual(decision.requirements.hot_per_ready_request, 4160)
        self.assertEqual(decision.requirements.hot_prefill_peak, 4160)
        self.assertEqual(decision.requirements.logical_for_next_wave, 4352)
        self.assertEqual(decision.requirements.host_for_next_wave, 4352)

    def test_hot_prefill_peak_stops_before_launching_partial_wave(self):
        decision = bench_utils.evaluate_hisparse_wave_admission(
            snapshot=self._snapshot(hot_total=8256, hot_available=4096),
            ready_count=1,
            requested_batch_size=4,
            input_len=4096,
            output_len=128,
            page_size=64,
            device_buffer_size=4096,
        )
        self.assertFalse(decision.can_admit)
        self.assertEqual(decision.stop_reason, "hot_prefill_peak")

    def test_hot_decode_reserve_is_checked_even_if_current_pool_is_free(self):
        decision = bench_utils.evaluate_hisparse_wave_admission(
            snapshot=self._snapshot(hot_total=8256, hot_available=8192),
            ready_count=1,
            requested_batch_size=4,
            input_len=1024,
            output_len=4096,
            page_size=64,
            device_buffer_size=4096,
        )
        self.assertFalse(decision.can_admit)
        self.assertEqual(decision.stop_reason, "hot_decode_reserve")

    def test_logical_host_request_and_context_failures_are_distinct(self):
        common = dict(
            ready_count=1,
            requested_batch_size=4,
            input_len=4096,
            output_len=128,
            page_size=64,
            device_buffer_size=4096,
        )
        cases = [
            (self._snapshot(logical_available=4000), "logical_pool"),
            (self._snapshot(host_available=4000), "host_pool"),
            (self._snapshot(request_slots_available=0), "request_pool"),
            (self._snapshot(max_context_len=4095), "max_context_len"),
        ]
        for snapshot, expected in cases:
            with self.subTest(expected=expected):
                decision = bench_utils.evaluate_hisparse_wave_admission(
                    snapshot=snapshot, **common
                )
                self.assertFalse(decision.can_admit)
                self.assertEqual(decision.stop_reason, expected)

    def test_logical_capacity_includes_extend_safety_page(self):
        decision = bench_utils.evaluate_hisparse_wave_admission(
            snapshot=self._snapshot(logical_available=1024),
            ready_count=0,
            requested_batch_size=1,
            input_len=1000,
            output_len=1,
            page_size=64,
            device_buffer_size=4096,
        )
        self.assertFalse(decision.can_admit)
        self.assertEqual(decision.stop_reason, "logical_pool")
        self.assertEqual(decision.requirements.logical_for_next_wave, 1064)

    def test_requested_batch_is_a_per_dp_group_target(self):
        decision = bench_utils.evaluate_hisparse_wave_admission(
            snapshot=self._snapshot(),
            ready_count=128,
            requested_batch_size=128,
            input_len=4096,
            output_len=128,
            page_size=64,
            device_buffer_size=4096,
        )
        self.assertFalse(decision.can_admit)
        self.assertEqual(decision.stop_reason, "target_reached")
        self.assertEqual(bench_utils.global_requested_batch_size(128, 16), 2048)
        self.assertEqual(bench_utils.global_requested_batch_size(128, 8), 1024)

    def test_dp_seed_is_shared_by_tp_ranks_but_changes_across_dp_groups(self):
        self.assertEqual(
            bench_utils.seed_for_attention_dp_group(1234, 7),
            bench_utils.seed_for_attention_dp_group(1234, 7),
        )
        self.assertNotEqual(
            bench_utils.seed_for_attention_dp_group(1234, 7),
            bench_utils.seed_for_attention_dp_group(1234, 8),
        )


class TestUnifiedWaveCapacity(unittest.TestCase):
    def test_wave_chunk_plan_gives_the_budget_to_one_request_per_dp_group(self):
        baseline = bench_utils.build_wave_chunk_plan(
            input_len=32768,
            requested_chunk_size=65536,
            effective_chunk_size=4096,
            page_size=64,
        )
        hisparse = bench_utils.build_wave_chunk_plan(
            input_len=32768,
            requested_chunk_size=65536,
            effective_chunk_size=4096,
            page_size=64,
        )

        self.assertEqual(baseline, hisparse)
        self.assertEqual(baseline.per_request_chunk_size, 4096)
        self.assertEqual(baseline.num_chunks, 8)

    def test_device_capacity_reserves_output_growth_for_ready_requests(self):
        snapshot = bench_utils.DeviceCapacitySnapshot(
            device_total=20000,
            device_available=4479,
            request_slots_available=8,
            max_context_len=32768,
        )
        rejected = bench_utils.evaluate_device_wave_admission(
            snapshot=snapshot,
            ready_count=2,
            requested_batch_size=4,
            input_len=4096,
            output_len=128,
            page_size=64,
        )
        admitted = bench_utils.evaluate_device_wave_admission(
            snapshot=bench_utils.DeviceCapacitySnapshot(
                device_total=20000,
                device_available=4480,
                request_slots_available=8,
                max_context_len=32768,
            ),
            ready_count=2,
            requested_batch_size=4,
            input_len=4096,
            output_len=128,
            page_size=64,
        )

        self.assertFalse(rejected.can_admit)
        self.assertEqual(rejected.stop_reason, "device_pool")
        self.assertEqual(rejected.required_tokens, 4480)
        self.assertTrue(admitted.can_admit)

    def test_device_capacity_reports_non_memory_stop_reasons(self):
        common = dict(
            ready_count=0,
            requested_batch_size=4,
            input_len=4096,
            output_len=128,
            page_size=64,
        )
        cases = [
            (
                bench_utils.DeviceCapacitySnapshot(20000, 20000, 0, 32768),
                "request_pool",
            ),
            (
                bench_utils.DeviceCapacitySnapshot(20000, 20000, 8, 4095),
                "max_context_len",
            ),
        ]
        for snapshot, reason in cases:
            with self.subTest(reason=reason):
                decision = bench_utils.evaluate_device_wave_admission(
                    snapshot=snapshot, **common
                )
                self.assertFalse(decision.can_admit)
                self.assertEqual(decision.stop_reason, reason)


if __name__ == "__main__":
    unittest.main()
