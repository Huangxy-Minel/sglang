"""Execute actual benchmark/runtime methods with CPU-only protocol doubles.

AST loading avoids importing the CUDA runtime; the extracted production bodies
are compiled unchanged (except decorators) and exercised through their APIs.
"""

import ast
import contextlib
import inspect
import runpy
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "python/sglang/benchmark/one_batch.py"
RUNNER = ROOT / "python/sglang/srt/model_executor/model_runner.py"


def load_function(path, name, env, class_name=None):
    tree = ast.parse(path.read_text())
    nodes = tree.body
    if class_name:
        nodes = next(
            n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name
        ).body
    node = next(
        n
        for n in nodes
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
    )
    node.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), env)
    return env[name]


# Load the stdlib-only registry without importing the GPU runtime package.
register_cpu_ci = runpy.run_path(
    str(Path(__file__).resolve().parents[3] / "python/sglang/test/ci/ci_register.py")
)["register_cpu_ci"]
register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestForwardBatchContracts(unittest.TestCase):
    def test_decode_passes_schedule_batch_and_int64_input(self):
        token = NS(to=lambda dtype: ("cast", dtype))
        batch = NS(prepare_for_decode=lambda: None)
        calls = []

        def init_new(actual, runner, *, return_hidden_states_before_norm):
            self.assertIs(actual, batch)
            self.assertFalse(return_hidden_states_before_norm)
            calls.append(actual.input_ids)
            return "forward"

        env = {
            "torch": NS(int64="int64"),
            "ForwardBatch": NS(init_new=init_new),
            "_maybe_prepare_mlp_sync_batch": lambda *a: None,
        }
        fn = load_function(BENCH, "prepare_decode_forward_batch", env)
        self.assertEqual(fn(token, batch, object()), "forward")
        self.assertEqual(calls, [("cast", "int64")])


class TestForceEager(unittest.TestCase):
    def make_raw(self, mode):
        events = []
        graph = NS(
            can_run_graph=lambda b: True,
            execute=lambda *a, **k: events.append("graph") or "logits",
        )
        batch = NS(
            forward_mode=NS(
                is_cpu_graph=lambda: True,
                is_cuda_graph=lambda: True,
                is_decode=lambda: mode == "decode",
                is_split_prefill=lambda: False,
                is_extend=lambda **k: mode == "prefill",
                is_target_verify=lambda: False,
            ),
            token_indices_to_pool=None,
            global_num_tokens_cpu=[1],
            post_forward_mlp_sync_batch=lambda ret: events.append("sync"),
        )
        runner = NS(
            device="cuda",
            attn_backend=None,
            decode_cuda_graph_runner=graph if mode == "decode" else None,
            prefill_cuda_graph_runner=graph,
            hisparse_coordinator=None,
            _prepare_eager_forward_batch=lambda b: events.append("prepare"),
            _maybe_execute_deferred_mamba_cow_and_clear=lambda b: None,
            eager_runner=NS(execute=lambda *a, **k: events.append("eager") or "logits"),
            pp_group=NS(is_last_rank=True),
            _extend_forward_kwargs=lambda *a: {},
            device_timer=None,
        )
        env = {
            "contextlib": contextlib,
            "has_forward_context": lambda: True,
            "get_global_dwdp_manager": lambda: None,
            "EagerRunner": type("EagerRunner", (), {}),
            "_prefill_cuda_graph_allows_context_parallel": lambda *a: True,
            "device_timer_ctx": lambda *a: contextlib.nullcontext(),
            "ModelRunnerOutput": lambda **k: NS(**k),
        }
        return (
            load_function(RUNNER, "_forward_raw", env, "ModelRunner"),
            runner,
            batch,
            events,
        )

    def test_force_eager_gates_both_graph_phases_and_keeps_sync(self):
        for phase in ("decode", "prefill"):
            with self.subTest(phase=phase):
                fn, runner, batch, events = self.make_raw(phase)
                result = fn(runner, batch, None, force_eager=True)
                self.assertFalse(result.can_run_graph)
                self.assertEqual(events, ["prepare", "eager", "sync"])

    def test_default_graph_execution_unchanged(self):
        for phase in ("decode", "prefill"):
            fn, runner, batch, events = self.make_raw(phase)
            self.assertTrue(fn(runner, batch, None).can_run_graph)
            self.assertIn("graph", events)
            self.assertNotIn("eager", events)

    def test_public_forward_force_eager_is_keyword_only_default_false(self):
        fn = load_function(RUNNER, "forward", {}, "ModelRunner")
        param = inspect.signature(fn).parameters.get("force_eager")
        self.assertIsNotNone(param)
        self.assertEqual(param.kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIs(param.default, False)

    def test_rank_fault_retry_preserves_force_eager(self):
        calls = []
        runner = NS(
            eplb_manager=None, _forward_raw=lambda *a, **k: calls.append(k) or "retried"
        )
        fn = load_function(
            RUNNER,
            "_maybe_rebalance_after_rank_fault",
            {"maybe_rebalance_after_rank_fault": lambda **k: True},
            "ModelRunner",
        )
        self.assertEqual(
            fn(runner, "old", "batch", None, False, 1, force_eager=True), "retried"
        )
        self.assertEqual(calls, [{"force_eager": True}])


class TestRequestConstruction(unittest.TestCase):
    def test_synthetic_requests_use_arrays_and_new_extend_range(self):
        from array import array

        def req(**kwargs):
            self.assertIsInstance(kwargs["origin_input_ids"], array)
            obj = NS(**kwargs, prefix_indices=[])
            obj.set_extend_range = lambda start, end: setattr(
                obj, "extend_range", (start, end)
            )
            return obj

        fn = load_function(
            BENCH,
            "prepare_synthetic_inputs_for_latency_test",
            {
                "array": array,
                "Req": req,
                "SamplingParams": lambda **k: NS(**k),
                "BenchArgs": NS(output_len=16),
            },
        )
        reqs = fn(1, 3, custom_inputs=[[1, 2, 3]], rid_offset=7)
        self.assertEqual(reqs[0].rid, 7)
        self.assertEqual(reqs[0].extend_range, (0, 3))
        self.assertEqual(list(reqs[0].full_untruncated_fill_ids), [1, 2, 3])


class TestWorkerLifecycle(unittest.TestCase):
    def test_publication_precedes_initializers_and_cleanup_runs_on_failure(self):
        events = []

        def work(*args):
            events.append("work")
            raise RuntimeError("worker failure")

        env = {
            "publish": lambda *a, **k: events.append(
                ("publish", k["ranks"].world_rank)
            ),
            "SpawnRanks": NS,
            "spawn_world_rank": lambda *a, **k: k["tp_rank"],
            "initialize_moe_config": lambda: events.append("moe"),
            "initialize_fp8_gemm_config": lambda: events.append("fp8"),
            "initialize_fp4_gemm_config": lambda: events.append("fp4"),
            "dist": NS(is_initialized=lambda: True),
            "destroy_model_parallel": lambda: events.append("destroy_model"),
            "destroy_distributed_environment": lambda: events.append("destroy_dist"),
        }
        fn = load_function(BENCH, "run_worker", env)
        with self.assertRaisesRegex(RuntimeError, "worker failure"):
            fn(work, object(), object(), object(), 1, 9)
        self.assertEqual(
            events,
            [
                ("publish", 9),
                "moe",
                "fp8",
                "fp4",
                "work",
                "destroy_model",
                "destroy_dist",
            ],
        )

    def test_load_model_preserves_new_bootstrap_pool_graph_order(self):
        events = []
        runner = NS(max_total_num_tokens=100, model_config="model_config")
        for name in (
            "start_startup_weight_load",
            "alloc_memory_pool",
            "init_attention_backends",
            "init_cuda_graphs",
            "finalize_startup_weight_load",
        ):
            setattr(runner, name, lambda n=name: events.append(n))
        cfg = NS(
            mem_fraction_static=0.8,
            tokenizer_path="tokenizer",
            tokenizer_mode="auto",
            trust_remote_code=False,
            tp_size=1,
        )
        env = {
            "resolving_view": lambda s: cfg,
            "suppress_other_loggers": lambda: None,
            "ModelConfig": NS(from_server_args=lambda s: "model_config"),
            "bootstrap": NS(
                init_parallel_runtime=lambda **k: events.append("parallel"),
                init_layer_runtime=lambda **k: events.append("layer"),
            ),
            "get_device": lambda: NS(device="cuda"),
            "use_mlx": lambda: False,
            "ModelRunner": lambda **k: events.append("runner") or runner,
            "get_model": lambda: NS(is_startup_weight_load_overlap=True),
            "mambaish_config": lambda c: None,
            "get_tokenizer": lambda *a, **k: "tokenizer",
            "_TorchBenchRunner": lambda r: r,
        }
        fn = load_function(BENCH, "load_model", env)
        result = fn(object(), NS(nccl_port=123), 1, 1)
        self.assertEqual(result, (runner, "tokenizer"))
        self.assertEqual(
            events,
            [
                "parallel",
                "layer",
                "runner",
                "start_startup_weight_load",
                "alloc_memory_pool",
                "init_attention_backends",
                "init_cuda_graphs",
                "finalize_startup_weight_load",
            ],
        )


class TestMemoryAccounting(unittest.TestCase):
    def test_graph_memory_aggregates_new_per_phase_measurements(self):
        cuda = NS(
            synchronize=lambda x: None,
            mem_get_info=lambda x: (400, 1000),
            memory_allocated=lambda x: 300,
            memory_reserved=lambda x: 500,
        )
        runner = NS(
            gpu_id=0,
            model=NS(parameters=lambda: [], buffers=lambda: []),
            token_to_kv_pool_allocator=NS(
                get_kvcache=lambda: NS(get_kv_size_bytes=lambda: 100)
            ),
            server_args=NS(moe_a2a_backend="none"),
            graph_memory_usage={"prefill": 0.125, "decode": 0.25},
        )
        env = {
            "torch": NS(cuda=cuda),
            "resolving_view": lambda s: s,
            "unique_cuda_storage_bytes": lambda t: 0,
            "split_kv_pool_bytes": lambda a, b: {
                "kv_data_bytes": a,
                "kv_indexer_bytes": b,
            },
            "HBMUsageSnapshot": NS,
        }
        fn = load_function(BENCH, "_hbm_usage_snapshot", env, "_TorchBenchRunner")
        result = fn(NS(torch_runner=runner))
        self.assertEqual(result.cuda_graph_bytes, 3 * (1 << 27))


class TestMainConfiguration(unittest.TestCase):
    def invoke(
        self, *, raw=None, resolved=None, max_bs=None, locked=(), algorithm=None
    ):
        class Graph:
            def __init__(self, config):
                self.config = config

            def to_dict(self):
                return {k: dict(v) for k, v in self.config.items()}

            @classmethod
            def from_dict(cls, config):
                return cls(config)

        server = NS(
            cuda_graph_config=raw,
            cuda_graph_max_bs_decode=max_bs,
            _resolution_finished=resolved is not None,
            _cuda_graph_config_locked=set(locked),
            speculative_algorithm=algorithm,
            pp_size=1,
            model_path="model",
            tp_size=1,
            nnodes=1,
            node_rank=0,
            device="cuda",
            resolve_once=lambda: None,
        )
        declarations = []
        import runpy

        helpers = runpy.run_path(
            str(ROOT / "python/sglang/benchmark/one_batch_utils.py")
        )
        env = {
            "CudaGraphConfig": Graph,
            "Phase": NS(DECODE="decode"),
            "resolving_view": lambda s: s,
            "resolution_result": lambda *a: Graph(resolved),
            "merge_one_batch_cuda_graph_config": helpers[
                "merge_one_batch_cuda_graph_config"
            ],
            "declare_resolution": lambda *a, **k: declarations.append(
                k["cuda_graph_config"]
            ),
            "_set_envs_and_config": lambda s: None,
            "latency_test": object(),
            "PortArgs": NS(init_new=lambda s: object()),
            "get_local_rank_assignments": lambda *a: [(0, 0)],
            "run_worker": lambda *a: None,
            "generate_decode_cuda_graph_batch_sizes": lambda server, maximum: [
                1,
                maximum,
            ],
        }
        fn = load_function(BENCH, "main", env)
        fn(server, NS(batch_size=[128], correctness_test=False))
        result = declarations[-1]
        return result.to_dict() if isinstance(result, Graph) else result

    def test_resolved_default_is_replaced_with_benchmark_inference(self):
        result = self.invoke(
            resolved={
                "decode": {"max_bs": 160, "bs": [1, 160]},
                "prefill": {"backend": "disabled"},
            }
        )
        self.assertEqual(result["decode"], {"max_bs": 128, "bs": [1, 128]})
        self.assertEqual(result["prefill"], {"backend": "disabled"})

    def test_resolved_explicit_40_keeps_capture_limit_and_buckets(self):
        result = self.invoke(
            raw={"decode": {"max_bs": 40}},
            resolved={"decode": {"max_bs": 40, "bs": [1, 40]}},
            locked=[("decode", "max_bs")],
        )
        self.assertEqual(result["decode"], {"max_bs": 40, "bs": [1, 40]})

    def test_raw_json_priority_over_convenience(self):
        self.assertEqual(
            self.invoke(raw={"decode": {"max_bs": 40}}, max_bs=64)["decode"]["max_bs"],
            40,
        )

    def test_speculative_execution_is_rejected(self):
        for algorithm in ("EAGLE", "NEXTN", "STANDALONE"):
            with (
                self.subTest(algorithm=algorithm),
                self.assertRaisesRegex(ValueError, "speculative/MTP"),
            ):
                self.invoke(algorithm=algorithm)


class TestAllocationAndFailureContracts(unittest.TestCase):
    def test_decode_batch_has_host_row_mirror_and_input_ids(self):
        class Tensor(list):
            pass

        reqs = [
            NS(kv=NS(req_pool_idx=slot), origin_input_ids=[1, 2, 3], output_ids=[4])
            for slot in (3, 7)
        ]
        runner = NS(
            device="cuda",
            req_to_token_pool=object(),
            token_to_kv_pool_allocator=object(),
            model_config=NS(vocab_size=100),
        )
        env = {
            "torch": NS(
                tensor=lambda values, **k: Tensor(values), int64="int64", int32="int32"
            ),
            "get_schedule": lambda: NS(page_size=64),
            "TreeCacheNamespace": NS,
            "ScheduleBatch": NS(init_new=lambda **k: NS(**k)),
            "SpeculativeAlgorithm": NS(NONE="none"),
            "SamplingBatchInfo": NS(from_schedule_batch=lambda *a: "sampling"),
        }
        fn = load_function(BENCH, "_build_decode_batch", env, "_TorchBenchRunner")
        batch = fn(NS(torch_runner=runner, is_hisparse=False), reqs)
        self.assertEqual(batch.req_pool_indices_cpu, [3, 7])
        self.assertEqual(batch.req_pool_indices, [3, 7])
        self.assertEqual(batch.input_ids, [4, 4])
        self.assertEqual(batch.seq_lens, [3, 3])

    def test_release_resets_kv_ownership_and_is_idempotent(self):
        events = []

        class Rows:
            def __getitem__(self, key):
                events.append(("row", key[0], key[1].stop))
                return NS(clone=lambda: [10, 11, 12])

        req = NS(kv=NS(req_pool_idx=7, kv_allocated_len=3))
        req.kv.mark_kv_released = lambda: setattr(req.kv, "kv_allocated_len", 0)

        def release_row(r):
            events.append("row_free")
            r.kv.req_pool_idx = None

        runner = NS(
            req_to_token_pool=NS(req_to_token=Rows(), free=release_row),
            token_to_kv_pool_allocator=NS(
                free=lambda indices: events.append(("free", indices))
            ),
        )
        fn = load_function(BENCH, "_release_request", {}, "_TorchBenchRunner")
        wrapped = NS(torch_runner=runner, is_hisparse=False)
        fn(wrapped, req)
        fn(wrapped, req)
        self.assertEqual(events, [("row", 7, 3), ("free", [10, 11, 12]), "row_free"])
        self.assertEqual(req.kv.kv_allocated_len, 0)

    def test_worker_failure_terminates_peers_without_waiting_for_completion(self):
        events = []
        failed = NS(pid=9, exitcode=2, join=lambda **k: None, is_alive=lambda: False)
        peer = NS(
            pid=10,
            exitcode=None,
            join=lambda **k: events.append("join"),
            is_alive=lambda: True,
            terminate=lambda: events.append("terminate"),
        )
        fn = load_function(BENCH, "wait_for_workers", {})
        with self.assertRaisesRegex(RuntimeError, "pid=9.*exit code 2"):
            fn([peer, failed])
        self.assertEqual(events, ["join", "terminate", "join"])

    def test_extend_resolves_deferred_prefill_tensor_before_forward(self):
        events = []
        batch = NS(
            input_ids=None,
            prefill_input_ids_cpu=NS(
                to=lambda device, **k: events.append("h2d") or "device_ids"
            ),
            device="cuda",
            prepare_for_extend=lambda: events.append("prepare"),
        )

        def init_new(actual, runner, *, return_hidden_states_before_norm):
            self.assertIs(actual, batch)
            self.assertEqual(actual.input_ids, "device_ids")
            self.assertIsNone(actual.prefill_input_ids_cpu)
            self.assertFalse(return_hidden_states_before_norm)
            return "forward"

        runner = NS(
            device="cuda",
            req_to_token_pool=object(),
            token_to_kv_pool_allocator=object(),
            model_config=object(),
            forward=lambda b: NS(logits_output=NS(next_token_logits="logits")),
            sample=lambda *a: events.append("sample") or "sampled",
        )
        env = {
            "get_schedule": lambda: NS(page_size=64),
            "TreeCacheNamespace": NS,
            "ScheduleBatch": NS(init_new=lambda **k: batch),
            "SpeculativeAlgorithm": NS(NONE="none"),
            "_maybe_prepare_mlp_sync_batch": lambda *a: None,
            "ForwardBatch": NS(init_new=init_new),
        }
        fn = load_function(BENCH, "extend", env)
        self.assertEqual(fn([], runner, sample=False), (None, "logits", batch))
        self.assertEqual(events, ["prepare", "h2d"])


class TestForwardWrapper(unittest.TestCase):
    def test_forward_threads_eager_mode_through_raw_and_fault_retry(self):
        calls = []
        output = NS(can_run_graph=False)
        runner = NS(
            forward_pass_id=0,
            msprobe_debugger=None,
            is_draft_worker=False,
            canary_manager=None,
            enable_elastic_ep=True,
            eplb_manager=None,
            _forward_raw=lambda *a, **k: calls.append(("raw", k)) or output,
            _maybe_rebalance_after_rank_fault=lambda *a, **k: (
                calls.append(("retry", k)) or output
            ),
        )
        batch = NS(
            apply_deprecated_skip_attn_backend_init=lambda value: calls.append(
                ("deprecated", value)
            )
        )
        env = {
            "contextlib": contextlib,
            "profile_range": lambda *a: contextlib.nullcontext(),
            "build_step_span_name": lambda *a, **k: "step",
            "get_global_expert_distribution_recorder": lambda: NS(
                with_forward_pass=lambda *a: contextlib.nullcontext(
                    {"metrics": "metrics"}
                )
            ),
            "get_schedule": lambda: NS(disable_overlap_schedule=True),
            "get_global_experts_capturer": lambda: None,
            "get_global_indexer_capturer": lambda: None,
            "dumper": NS(may_enable=False),
            "get_exec": lambda: NS(moe=NS(elastic_ep_backend=None)),
        }
        fn = load_function(RUNNER, "forward", env, "ModelRunner")
        self.assertIs(fn(runner, batch, force_eager=True), output)
        self.assertEqual(
            calls,
            [
                ("deprecated", None),
                ("raw", {"force_eager": True}),
                ("retry", {"force_eager": True}),
            ],
        )
        self.assertEqual(output.expert_distribution_metrics, "metrics")


class TestCompositeKVAccounting(unittest.TestCase):
    @staticmethod
    def v4_pool_without_allocation():
        # Compile the real, complete inheritance chain without module imports or
        # GPU allocation. Only external values used in default arguments need
        # placeholders; all class bodies/methods and inheritance are unchanged.
        import abc

        nodes = [
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            )
        ]
        for filename, name in (
            ("memory_pool.py", "KVCache"),
            ("base_swa_memory_pool.py", "BaseSWAKVPool"),
            ("deepseek_v4_memory_pool.py", "DeepSeekV4TokenToKVPool"),
        ):
            tree = ast.parse(
                (ROOT / "python/sglang/srt/mem_cache" / filename).read_text()
            )
            nodes.append(
                next(
                    n
                    for n in tree.body
                    if isinstance(n, ast.ClassDef) and n.name == name
                )
            )
        env = {
            "abc": abc,
            "KVLayout": NS(V4=object()),
            "DeepSeekV4SingleKVPool": object,
        }
        module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
        exec(compile(module, "<actual KV pool classes>", "exec"), env)
        return object.__new__(env["DeepSeekV4TokenToKVPool"])

    @staticmethod
    def snapshot_for_pool(pool):
        helpers = runpy.run_path(
            str(ROOT / "python/sglang/benchmark/one_batch_utils.py")
        )
        cuda = NS(
            synchronize=lambda x: None,
            mem_get_info=lambda x: (400, 1000),
            memory_allocated=lambda x: 300,
            memory_reserved=lambda x: 500,
        )
        runner = NS(
            gpu_id=0,
            model=NS(parameters=lambda: [], buffers=lambda: []),
            token_to_kv_pool_allocator=NS(get_kvcache=lambda: pool),
            server_args=NS(moe_a2a_backend="none"),
            graph_memory_usage={"decode": 0.125},
        )
        env = {
            **helpers,
            "torch": NS(cuda=cuda),
            "resolving_view": lambda s: s,
            "unique_cuda_storage_bytes": lambda tensors: sum(tensors),
        }
        fn = load_function(BENCH, "_hbm_usage_snapshot", env, "_TorchBenchRunner")
        return fn(NS(torch_runner=runner)), helpers

    def test_actual_v4_composite_without_size_method_reports_unknown(self):
        pool = self.v4_pool_without_allocation()
        snapshot, helpers = self.snapshot_for_pool(pool)
        self.assertIsNone(snapshot.kv_data_bytes)
        self.assertIsNone(snapshot.kv_indexer_bytes)
        usage = helpers["build_hbm_usage"](snapshot)
        self.assertIsNone(usage["torch_active_other_bytes"])
        self.assertEqual(usage["torch_active_bytes"], 300)
        self.assertEqual(usage["used_bytes"], 600)
        self.assertEqual(usage["effective_available_bytes"], 600)

    def test_post_capture_virtual_upper_bound_is_not_physical_usage(self):
        def virtual_size():
            self.fail("virtual address upper bound is not allocated backing")

        pool = NS(post_capture_active=True, get_kv_size_bytes=virtual_size)
        snapshot, _ = self.snapshot_for_pool(pool)
        self.assertIsNone(snapshot.kv_data_bytes)
        self.assertIsNone(snapshot.kv_indexer_bytes)

    def test_legacy_exact_size_and_indexer_split_are_retained(self):
        pool = NS(get_kv_size_bytes=lambda: (70, 30), index_k_with_scale_buffer=[20])
        snapshot, helpers = self.snapshot_for_pool(pool)
        self.assertEqual(snapshot.kv_data_bytes, 80)
        self.assertEqual(snapshot.kv_indexer_bytes, 20)
        self.assertEqual(
            helpers["build_hbm_usage"](snapshot)["torch_active_other_bytes"], 200
        )

    def test_unknown_rank_still_gathers_and_renders_complete_report(self):
        import dataclasses

        unknown, helpers = self.snapshot_for_pool(self.v4_pool_without_allocation())
        known, _ = self.snapshot_for_pool(NS(get_kv_size_bytes=lambda: 100))
        fields = [field.name for field in dataclasses.fields(unknown)]

        class Tensor(list):
            def tolist(self):
                return list(self)

        for local, remote in ((unknown, known), (known, unknown)):
            calls = []

            def gather(output, value, *, group):
                calls.append((list(value), group))
                output[0][:] = value
                output[1][:] = [
                    getattr(remote, field) if getattr(remote, field) is not None else -1
                    for field in fields
                ]

            env = {
                **helpers,
                "dataclasses": dataclasses,
                "torch": NS(
                    tensor=lambda values, **k: Tensor(values),
                    empty_like=lambda value: Tensor([0] * len(value)),
                    int64="int64",
                ),
                "dist": NS(
                    is_initialized=lambda: True,
                    get_world_size=lambda **k: 2,
                    all_gather=gather,
                ),
                "get_parallel": lambda: NS(tp_group=NS(cpu_group="model_tp")),
            }
            fn = load_function(BENCH, "hbm_usage_summary", env, "_TorchBenchRunner")
            summary = fn(NS(_hbm_usage_snapshot=lambda: local))
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][1], "model_tp")
            self.assertEqual(summary["num_ranks"], 2)
            for name in (
                "kv_data_bytes",
                "kv_indexer_bytes",
                "torch_active_other_bytes",
            ):
                self.assertEqual(summary[name], {"min": None, "avg": None, "max": None})
            self.assertEqual(
                summary["used_bytes"], {"min": 600, "avg": 600, "max": 600}
            )
            lines = []
            render = load_function(BENCH, "print_hbm_usage_report", {})
            render(summary, lines.append)
            for label in ("KV data pool", "KV indexer pool", "PyTorch active other"):
                self.assertTrue(
                    any(label in line and "unknown" in line for line in lines)
                )


if __name__ == "__main__":
    unittest.main()
