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

    def test_worker_ranks_cover_dp16_tp1_and_dp8_tp2(self):
        dp16 = bench_utils.derive_worker_ranks(
            tp_rank=7,
            tp_size=16,
            dp_size=16,
            attn_cp_size=1,
            moe_dp_size=1,
            ep_size=16,
            enable_dp_attention=True,
        )
        self.assertEqual(dp16.attn_dp_rank, 7)
        self.assertEqual(dp16.attn_tp_rank, 0)
        self.assertEqual(dp16.attn_tp_size, 1)
        self.assertEqual(dp16.moe_ep_rank, 7)

        dp8_tp2_rank0 = bench_utils.derive_worker_ranks(
            tp_rank=6,
            tp_size=16,
            dp_size=8,
            attn_cp_size=1,
            moe_dp_size=1,
            ep_size=16,
            enable_dp_attention=True,
        )
        dp8_tp2_rank1 = bench_utils.derive_worker_ranks(
            tp_rank=7,
            tp_size=16,
            dp_size=8,
            attn_cp_size=1,
            moe_dp_size=1,
            ep_size=16,
            enable_dp_attention=True,
        )
        self.assertEqual(dp8_tp2_rank0.attn_dp_rank, 3)
        self.assertEqual(dp8_tp2_rank1.attn_dp_rank, 3)
        self.assertEqual(dp8_tp2_rank0.attn_tp_rank, 0)
        self.assertEqual(dp8_tp2_rank1.attn_tp_rank, 1)
        self.assertEqual(dp8_tp2_rank0.attn_tp_size, 2)


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


class TestDecodeProfilePlan(unittest.TestCase):
    def test_gpu_profile_also_captures_cpu_ranges(self):
        self.assertEqual(
            bench_utils.normalize_profile_activities(("GPU",)),
            ("CPU", "GPU"),
        )

    def test_profile_window_skips_16_steps_then_profiles_8(self):
        plan = bench_utils.build_decode_profile_plan(
            output_len=128,
            profile_enabled=True,
            profile_stage="decode",
            profile_start_step=16,
            profile_steps=8,
            execution_mode="eager",
            exit_after_capture=True,
        )

        self.assertFalse(plan.action_for_step(15).profile)
        self.assertFalse(plan.action_for_step(15).force_eager)
        self.assertTrue(plan.action_for_step(16).profile)
        self.assertTrue(plan.action_for_step(23).force_eager)
        self.assertTrue(plan.action_for_step(23).exit_after_step)
        self.assertFalse(plan.action_for_step(24).profile)
        self.assertEqual(plan.profiled_steps, 8)
        self.assertEqual(plan.execution_mode, "eager")

    def test_explicit_cuda_graph_limit_is_preserved(self):
        self.assertEqual(
            bench_utils.resolve_one_batch_cuda_graph_max_bs(32, (128,)), 32
        )

    def test_missing_cuda_graph_limit_uses_requested_batch(self):
        self.assertEqual(
            bench_utils.resolve_one_batch_cuda_graph_max_bs(None, (16, 64)), 64
        )

    def test_profile_trace_filename_uses_admitted_batch_and_parallelism(self):
        self.assertEqual(
            bench_utils.build_profile_trace_filename(
                output_dir="/tmp/traces",
                prefix="glm51",
                batch_size=7,
                input_len=32768,
                output_len=128,
                stage="decode",
                tp_size=16,
                dp_size=16,
                ep_size=16,
            ),
            "/tmp/traces/glm51_tp16_dp16_ep16_batch7_input32768_output128_decode.trace.json.gz",
        )

    def test_runtime_mode_profiles_the_same_window_without_forcing_eager(self):
        plan = bench_utils.build_decode_profile_plan(
            output_len=128,
            profile_enabled=True,
            profile_stage="decode",
            profile_start_step=16,
            profile_steps=8,
            execution_mode="runtime",
            exit_after_capture=False,
        )

        self.assertFalse(plan.action_for_step(15).profile)
        self.assertTrue(plan.action_for_step(16).profile)
        self.assertFalse(plan.action_for_step(16).force_eager)
        self.assertTrue(plan.action_for_step(23).profile)
        self.assertFalse(plan.action_for_step(23).force_eager)

    def test_profile_window_rejects_invalid_bounds(self):
        with self.assertRaisesRegex(ValueError, "decode iterations"):
            bench_utils.build_decode_profile_plan(
                output_len=24,
                profile_enabled=True,
                profile_stage="decode",
                profile_start_step=16,
                profile_steps=8,
                execution_mode="eager",
                exit_after_capture=True,
            )


class TestMTPHelpers(unittest.TestCase):
    def test_speculative_slot_reserve_is_page_aligned(self):
        self.assertEqual(
            bench_utils.speculative_slot_reserve(
                num_steps=3,
                topk=1,
                num_draft_tokens=4,
                page_size=64,
            ),
            64,
        )
        self.assertEqual(
            bench_utils.speculative_slot_reserve(
                num_steps=3,
                topk=4,
                num_draft_tokens=4,
                page_size=8,
            ),
            16,
        )

    def test_mtp_cycle_metrics_count_bonus_tokens(self):
        metrics = bench_utils.build_mtp_cycle_metrics(
            accepted_draft_tokens=[0, 2, 3],
            speculative_num_steps=3,
            process_latency=0.012,
            core_cycle_latency=0.009,
        )
        self.assertEqual(metrics["active_batch_size"], 3)
        self.assertEqual(metrics["accepted_draft_tokens"], 5)
        self.assertEqual(metrics["accepted_tokens"], 8)
        self.assertAlmostEqual(metrics["average_accepted_length"], 8 / 3)
        self.assertAlmostEqual(metrics["normalized_tpot_ms"], 3.375)
        self.assertAlmostEqual(metrics["throughput_per_dp"], 8 / 0.009)
        self.assertAlmostEqual(metrics["draft_acceptance_rate"], 5 / 9)

    def test_cluster_mtp_metrics_use_unique_dp_tokens_and_slowest_rank(self):
        metrics = bench_utils.build_mtp_cluster_cycle_metrics(
            unique_dp_accepted_tokens=57,
            max_core_cycle_latency=0.01,
        )
        self.assertEqual(metrics["cluster_accepted_tokens"], 57)
        self.assertEqual(metrics["cluster_throughput"], 5700)

    def test_draft_inputs_merge_in_wave_order(self):
        class FakeDraftInput:
            def __init__(self, values):
                self.values = list(values)

            def merge_batch(self, other):
                self.values.extend(other.values)

        merged = bench_utils.merge_speculative_inputs(
            [FakeDraftInput([0]), FakeDraftInput([1]), FakeDraftInput([2])]
        )
        self.assertEqual(merged.values, [0, 1, 2])

    def test_speculative_mode_rejects_unsupported_combinations(self):
        for kwargs, message in (
            ({"enable_hisparse": True}, "HiSparse"),
            ({"correctness_test": True}, "correctness"),
            (
                {"profile_enabled": True, "profile_execution_mode": "eager"},
                "runtime",
            ),
            ({"draft_model_path": "/draft"}, "external draft"),
        ):
            defaults = dict(
                spec_algorithm="EAGLE",
                enable_hisparse=False,
                correctness_test=False,
                profile_enabled=False,
                profile_execution_mode="runtime",
                model_path="/target",
                draft_model_path="/target",
            )
            defaults.update(kwargs)
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    bench_utils.validate_one_batch_speculative_mode(**defaults)

    def test_exact_output_filter_uses_per_request_lengths(self):
        self.assertEqual(
            bench_utils.unfinished_request_indices([128, 125, 128, 127], 128),
            [1, 3],
        )

    def test_profile_window_rejects_invalid_execution_mode(self):
        with self.assertRaisesRegex(ValueError, "profile execution mode"):
            bench_utils.build_decode_profile_plan(
                output_len=128,
                profile_enabled=True,
                profile_stage="decode",
                profile_start_step=16,
                profile_steps=8,
                execution_mode="graph",
                exit_after_capture=False,
            )

    def test_profile_controls_require_decode_profiling(self):
        with self.assertRaisesRegex(ValueError, "require --profile"):
            bench_utils.build_decode_profile_plan(
                output_len=128,
                profile_enabled=False,
                profile_stage="decode",
                profile_start_step=16,
                profile_steps=8,
                execution_mode="eager",
                exit_after_capture=True,
            )
        with self.assertRaisesRegex(ValueError, "decode profile stage"):
            bench_utils.build_decode_profile_plan(
                output_len=128,
                profile_enabled=True,
                profile_stage="prefill",
                profile_start_step=16,
                profile_steps=8,
                execution_mode="eager",
                exit_after_capture=True,
            )


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


class TestTargetWarmupControls(unittest.TestCase):
    def test_target_warmup_reuses_prefill_log_interval(self):
        self.assertEqual(
            bench_utils.resolve_target_warmup_log_interval(
                skip_target_warmup=False,
                log_prefill_wave=8,
            ),
            8,
        )

    def test_target_warmup_can_be_skipped_without_disabling_micro_warmup(self):
        self.assertIsNone(
            bench_utils.resolve_target_warmup_log_interval(
                skip_target_warmup=True,
                log_prefill_wave=8,
            )
        )


class TestClusterMetrics(unittest.TestCase):
    def test_unique_storage_bytes_deduplicates_shared_cuda_storage(self):
        class FakeStorage:
            def __init__(self, ptr, size):
                self._ptr = ptr
                self._size = size

            def data_ptr(self):
                return self._ptr

            def nbytes(self):
                return self._size

        class FakeTensor:
            def __init__(self, device, storage):
                self.device = device
                self._storage = storage

            def untyped_storage(self):
                return self._storage

        shared = FakeStorage(100, 4096)
        tensors = [
            FakeTensor("cuda:0", shared),
            FakeTensor("cuda:0", shared),
            FakeTensor("cuda:0", FakeStorage(200, 1024)),
            FakeTensor("cpu", FakeStorage(300, 8192)),
        ]

        self.assertEqual(bench_utils.unique_cuda_storage_bytes(tensors), 5120)

        draft_tensors = [
            FakeTensor("cuda:0", shared),
            FakeTensor("cuda:0", FakeStorage(400, 2048)),
        ]
        self.assertEqual(
            bench_utils.exclusive_cuda_storage_bytes(draft_tensors, tensors),
            2048,
        )

    def test_kv_pool_usage_splits_indexer_without_double_counting(self):
        self.assertEqual(
            bench_utils.split_kv_pool_bytes(1000, 250),
            {"kv_data_bytes": 750, "kv_indexer_bytes": 250},
        )
        with self.assertRaisesRegex(ValueError, "indexer KV storage"):
            bench_utils.split_kv_pool_bytes(100, 101)

    def test_hbm_usage_separates_reclaimable_cache_from_active_memory(self):
        usage = bench_utils.build_hbm_usage(
            bench_utils.HBMUsageSnapshot(
                total_bytes=1000,
                free_bytes=100,
                torch_active_bytes=700,
                torch_reserved_bytes=800,
                model_bytes=400,
                kv_data_bytes=200,
                kv_indexer_bytes=50,
                cuda_graph_bytes=100,
                draft_model_bytes=20,
                draft_kv_data_bytes=10,
                draft_kv_indexer_bytes=5,
                deepep_configured_bytes=80,
            )
        )

        self.assertEqual(usage["used_bytes"], 900)
        self.assertEqual(usage["torch_active_other_bytes"], 15)
        self.assertEqual(usage["torch_inactive_cache_bytes"], 100)
        self.assertEqual(usage["native_external_bytes"], 100)
        self.assertEqual(usage["effective_available_bytes"], 200)
        self.assertEqual(usage["deepep_configured_bytes"], 80)
        self.assertEqual(
            usage["model_bytes"]
            + usage["kv_data_bytes"]
            + usage["kv_indexer_bytes"]
            + usage["draft_model_bytes"]
            + usage["draft_kv_data_bytes"]
            + usage["draft_kv_indexer_bytes"]
            + usage["torch_active_other_bytes"]
            + usage["native_external_bytes"]
            + usage["effective_available_bytes"],
            usage["total_bytes"],
        )

    def test_hbm_usage_rejects_impossible_accounting(self):
        with self.assertRaisesRegex(ValueError, "known active HBM"):
            bench_utils.build_hbm_usage(
                bench_utils.HBMUsageSnapshot(
                    total_bytes=1000,
                    free_bytes=100,
                    torch_active_bytes=700,
                    torch_reserved_bytes=800,
                    model_bytes=700,
                    kv_data_bytes=200,
                    kv_indexer_bytes=50,
                    cuda_graph_bytes=100,
                )
            )

    def test_hbm_usage_summary_reports_per_gpu_min_avg_max(self):
        summary = bench_utils.summarize_hbm_usage(
            [
                bench_utils.HBMUsageSnapshot(
                    total_bytes=1000,
                    free_bytes=100,
                    torch_active_bytes=700,
                    torch_reserved_bytes=800,
                    model_bytes=400,
                    kv_data_bytes=200,
                    kv_indexer_bytes=50,
                    cuda_graph_bytes=100,
                    deepep_configured_bytes=80,
                ),
                bench_utils.HBMUsageSnapshot(
                    total_bytes=1000,
                    free_bytes=80,
                    torch_active_bytes=710,
                    torch_reserved_bytes=820,
                    model_bytes=400,
                    kv_data_bytes=200,
                    kv_indexer_bytes=50,
                    cuda_graph_bytes=100,
                    deepep_configured_bytes=80,
                ),
            ]
        )

        self.assertEqual(summary["num_ranks"], 2)
        self.assertEqual(summary["free_bytes"], {"min": 80, "avg": 90, "max": 100})
        self.assertEqual(
            summary["torch_inactive_cache_bytes"],
            {"min": 100, "avg": 105, "max": 110},
        )
        self.assertEqual(
            summary["effective_available_bytes"],
            {"min": 190, "avg": 195, "max": 200},
        )

    def test_prefill_wave_log_selection_covers_start_interval_and_final(self):
        self.assertTrue(bench_utils.should_log_prefill_wave(1, 16))
        self.assertTrue(bench_utils.should_log_prefill_wave(5, 16))
        self.assertFalse(bench_utils.should_log_prefill_wave(6, 16))
        self.assertTrue(bench_utils.should_log_prefill_wave(16, 16))
        self.assertTrue(
            bench_utils.should_log_prefill_wave(21, 16, is_final=True)
        )
        self.assertFalse(bench_utils.should_log_prefill_wave(1, 0))
        self.assertFalse(
            bench_utils.should_log_prefill_wave(21, 0, is_final=True)
        )

    def test_decode_step_metrics_report_tpot_and_both_throughputs(self):
        metrics = bench_utils.build_decode_step_metrics(
            batch_size=32,
            dp_size=16,
            process_latency=0.085,
            core_forward_tpot=0.08,
        )
        self.assertEqual(metrics["process_latency_ms"], 85.0)
        self.assertEqual(metrics["tpot_ms"], 80.0)
        self.assertEqual(metrics["throughput_per_dp"], 400.0)
        self.assertEqual(metrics["cluster_throughput"], 6400.0)

    def test_prefill_wave_metrics_use_cumulative_ready_tokens(self):
        metrics = bench_utils.build_prefill_wave_metrics(
            ready_batch_size=8,
            dp_size=16,
            input_len=4096,
            elapsed=4.0,
        )
        self.assertEqual(metrics["throughput_per_dp"], 8192.0)
        self.assertEqual(metrics["cluster_throughput"], 131072.0)

    def test_capacity_usage_reports_device_and_hisparse_pools(self):
        device = bench_utils.build_capacity_usage(
            bench_utils.DeviceCapacitySnapshot(1000, 250, 8, 32768)
        )
        self.assertEqual(
            device,
            {
                "device": {
                    "used": 750,
                    "available": 250,
                    "total": 1000,
                    "usage": 0.75,
                }
            },
        )

        hisparse = bench_utils.build_capacity_usage(
            bench_utils.HiSparseCapacitySnapshot(
                hot_total=100,
                hot_available=25,
                logical_total=1000,
                logical_available=600,
                host_total=2000,
                host_available=1500,
                request_slots_available=8,
                max_context_len=32768,
            )
        )
        self.assertEqual(hisparse["hot"]["usage"], 0.75)
        self.assertEqual(hisparse["logical"]["used"], 400)
        self.assertEqual(hisparse["host"]["used"], 500)

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
            cluster_median_core_forward_tpot=0.8,
            cluster_total_latency=5.25,
        )
        self.assertEqual(metrics["global_batch_size"], 512)
        self.assertEqual(metrics["cluster_prefill_latency"], 4.0)
        self.assertEqual(metrics["cluster_prefill_throughput"], 524288.0)
        self.assertEqual(metrics["cluster_median_decode_latency"], 1.0)
        self.assertEqual(metrics["cluster_median_core_forward_tpot"], 0.8)
        self.assertAlmostEqual(
            metrics["cluster_median_decode_throughput_per_dp"], 40.0
        )
        self.assertAlmostEqual(metrics["cluster_median_decode_throughput"], 640.0)
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
            cluster_median_core_forward_tpot=0.08,
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

    def test_deferred_hydration_reaches_decode_capacity_after_long_prefill(self):
        snapshot = self._snapshot(
            hot_total=37056,
            hot_available=37056,
            max_context_len=65536,
        )
        ready_count = 0
        while True:
            decision = bench_utils.evaluate_hisparse_wave_admission(
                snapshot=snapshot,
                ready_count=ready_count,
                requested_batch_size=128,
                input_len=32768,
                output_len=8192,
                page_size=64,
                device_buffer_size=4096,
            )
            if not decision.can_admit:
                break
            ready_count += 1

        self.assertEqual(ready_count, 8)
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

    def test_device_capacity_adds_speculative_peak_for_all_decode_requests(self):
        decision = bench_utils.evaluate_device_wave_admission(
            snapshot=bench_utils.DeviceCapacitySnapshot(
                device_total=20000,
                device_available=4671,
                request_slots_available=8,
                max_context_len=32768,
            ),
            ready_count=2,
            requested_batch_size=4,
            input_len=4096,
            output_len=128,
            page_size=64,
            speculative_reserve_per_request=64,
        )

        self.assertFalse(decision.can_admit)
        self.assertEqual(decision.required_tokens, 4672)

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
