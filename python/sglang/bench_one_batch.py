"""
Benchmark the latency of running a single static batch without a server.

This script does not launch a server and uses the low-level APIs.
It accepts server arguments (the same as launch_server.py) and benchmark arguments (e.g., batch size, input lengths).

# Usage (latency test)
## with dummy weights:
python -m sglang.bench_one_batch --model-path meta-llama/Meta-Llama-3-8B-Instruct --load-format dummy
## sweep through multiple data points and store (append) the results in a jsonl file:
python -m sglang.bench_one_batch --model-path meta-llama/Meta-Llama-3-8B-Instruct --batch 1 12 14 --input-len 256 512 --output-len 32 256 --run-name test_run
## run with profiling:
python -m sglang.bench_one_batch --model-path meta-llama/Meta-Llama-3-8B-Instruct --batch 1 12 14 --input-len 256 512 --profile
## run with profiling to custom directory:
export SGLANG_TORCH_PROFILER_DIR=/root/sglang/profile_log
python -m sglang.bench_one_batch --model-path meta-llama/Meta-Llama-3-8B-Instruct --batch 1 --input-len 256 --profile
## run with CUDA profiler (nsys):
nsys profile --force-overwrite=true -o bench_one_batch python -m sglang.bench_one_batch --model-path meta-llama/Meta-Llama-3-8B-Instruct --batch 1 --input-len 256 --profile --profile-activities CUDA_PROFILER
# Usage (correctness test):
python -m sglang.bench_one_batch --model-path TinyLlama/TinyLlama-1.1B-Chat-v0.4 --correct

## Reference output (of the correctness test above, can be gpu dependent):
input_ids=[[1, 450, 7483, 310, 3444, 338], [1, 450, 7483, 310, 278, 3303, 13187, 290, 338], [1, 20628, 338, 263, 6575, 1460, 2462, 322, 306, 763]]

prefill logits (first half): tensor([[-10.0312,  -9.5000,   0.8931,  ...,  -4.9414,  -3.2422,  -3.3633],
        [-10.0312,  -9.5000,   0.8931,  ...,  -4.9414,  -3.2422,  -3.3633],
        [ -9.1875, -10.2500,   2.7129,  ...,  -4.3359,  -4.0664,  -4.1328]],
       device='cuda:0')

prefill logits (final): tensor([[-8.3125, -7.1172,  3.3457,  ..., -4.9570, -4.1328, -3.4141],
        [-8.9141, -9.0156,  4.1445,  ..., -4.9922, -4.4961, -4.0781],
        [-9.6328, -9.0547,  4.0195,  ..., -5.3047, -4.7148, -4.4570]],
       device='cuda:0')

========== Prompt 0 ==========
<s> The capital of France is Paris.
The capital of the United States is Washington, D.C.


========== Prompt 1 ==========
<s> The capital of the United Kindom is London.
The capital of the United Kingdom is London.
The capital of the

========== Prompt 2 ==========
<s> Today is a sunny day and I like to go for a walk in the park.
I'm going to the park
"""

import argparse
import copy
import dataclasses
import itertools
import json
import logging
import multiprocessing
import os
import time
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist

from sglang.bench_one_batch_utils import (
    ChunkPlan,
    DeviceCapacitySnapshot,
    HiSparseCapacitySnapshot,
    build_capacity_usage,
    build_chunk_plan,
    build_cluster_metrics,
    build_decode_step_metrics,
    build_deepep_micro_warmup_shape,
    build_prefill_wave_metrics,
    build_wave_chunk_plan,
    evaluate_device_wave_admission,
    evaluate_hisparse_wave_admission,
    get_local_rank_assignments,
    prepare_chunk_requests,
    seed_for_attention_dp_group,
    should_log_prefill_wave,
    values_from_slowest_rank,
)
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_tp_group,
)
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.layers.dp_attention import (
    get_attention_dp_rank,
    get_attention_tp_size,
)
from sglang.srt.layers.moe import initialize_moe_config
from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler_dp_attn_mixin import prepare_mlp_sync_batch_raw
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import (
    configure_logger,
    get_bool_env_var,
    kill_process_tree,
    maybe_reindex_device_id,
    require_mlp_sync,
    require_mlp_tp_gather,
    set_gpu_proc_affinity,
    suppress_other_loggers,
)
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.srt.utils.tensor_bridge import use_mlx


def start_profile(profile_activities, profile_record_shapes=False, rank_print=print):
    """
    Abstracted function to start profiling based on profile_activities.
    Returns profiler object (or None).
    """
    if "CUDA_PROFILER" in profile_activities:
        try:
            torch.cuda.cudart().cudaProfilerStart()
            rank_print("CUDA Profiler started (nsys will begin capturing)")
        except Exception as e:
            rank_print(f"Failed to start CUDA profiler: {e}")
        return None
    else:
        activities = []
        if "CPU" in profile_activities:
            activities.append(torch.profiler.ProfilerActivity.CPU)
        if "GPU" in profile_activities:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        if "XPU" in profile_activities:
            activities.append(torch.profiler.ProfilerActivity.XPU)
        if activities:
            profiler = torch.profiler.profile(
                activities=activities,
                with_stack=True,
                record_shapes=profile_record_shapes,
            )
            profiler.start()
            return profiler
        return None


def stop_profile(
    profiler,
    profile_activities,
    rank_print=print,
    save_trace=False,
    trace_filename=None,
    stage=None,
):
    """
    Abstracted function to stop profiling based on profile_activities.
    Optionally saves trace results and prints completion messages.
    """
    if "CUDA_PROFILER" in profile_activities:
        try:
            torch.cuda.cudart().cudaProfilerStop()
            rank_print("CUDA Profiler stopped (nsys should dump traces)")
        except Exception as e:
            rank_print(f"Failed to stop CUDA profiler: {e}")
    elif profiler is not None:
        profiler.stop()

    if save_trace:
        if profiler is not None:
            if trace_filename:
                _save_profile_trace_results(profiler, trace_filename)
                stage_desc = f"for {stage}" if stage else ""
                rank_print(
                    f"torch profiler chrome trace {stage_desc} saved to {trace_filename}"
                )
        if "CUDA_PROFILER" in profile_activities:
            rank_print(f"CUDA profiler trace for {stage} completed")


@contextmanager
def trace_range(name: str, enabled: bool):
    """Emit matching torch-profiler and NVTX ranges when profiling is enabled."""
    if not enabled:
        yield
        return

    use_nvtx = torch.cuda.is_available()
    if use_nvtx:
        torch.cuda.nvtx.range_push(name)
    try:
        with torch.profiler.record_function(name):
            yield
    finally:
        if use_nvtx:
            torch.cuda.nvtx.range_pop()


def trace_mark(name: str, enabled: bool):
    if enabled and torch.cuda.is_available():
        torch.cuda.nvtx.mark(name)


@dataclasses.dataclass
class BenchArgs:
    run_name: str = "default"
    batch_size: Tuple[int] = (1,)
    input_len: Tuple[int] = (1024,)
    output_len: Tuple[int] = (16,)
    prompt_filename: str = ""
    result_filename: str = "result.jsonl"
    correctness_test: bool = False
    # This is only used for correctness test
    cut_len: int = 4
    log_prefill_wave: int = 0
    log_decode_step: int = 0
    profile: bool = False
    profile_record_shapes: bool = False
    profile_activities: Tuple[str] = ("CPU", "GPU")
    profile_stage: str = "all"
    profile_filename_prefix: str = "profile"
    profile_start_step: Optional[int] = None
    profile_steps: Optional[int] = None
    # This option is registered by ServerArgs. Keeping the raw CLI value here
    # lets one-batch distinguish an explicit request from an automatic default.
    chunked_prefill_size: Optional[int] = None

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--run-name", type=str, default=BenchArgs.run_name)
        parser.add_argument(
            "--batch-size",
            type=int,
            nargs="+",
            default=BenchArgs.batch_size,
            help=(
                "Desired decode batch size per attention DP group. The cluster "
                "target is batch_size multiplied by dp_size."
            ),
        )
        parser.add_argument(
            "--input-len", type=int, nargs="+", default=BenchArgs.input_len
        )
        parser.add_argument(
            "--output-len", type=int, nargs="+", default=BenchArgs.output_len
        )
        parser.add_argument(
            "--prompt-filename", type=str, default=BenchArgs.prompt_filename
        )
        parser.add_argument(
            "--result-filename", type=str, default=BenchArgs.result_filename
        )
        parser.add_argument("--correctness-test", action="store_true")
        parser.add_argument("--cut-len", type=int, default=BenchArgs.cut_len)
        parser.add_argument(
            "--log-prefill-wave",
            type=int,
            default=BenchArgs.log_prefill_wave,
            help=(
                "Log prefill progress for the first five waves, every N waves, "
                "and the final admitted wave. Zero disables progress logs."
            ),
        )
        parser.add_argument(
            "--log-decode-step",
            type=int,
            default=BenchArgs.log_decode_step,
            help="Log decode latency by step, default is set to zero to disable.",
        )
        parser.add_argument("--profile", action="store_true", help="Enable profiling.")
        parser.add_argument(
            "--profile-record-shapes",
            action="store_true",
            help="Record tensor shapes in profiling results.",
        )
        parser.add_argument(
            "--profile-activities",
            type=str,
            nargs="+",
            default=["CPU", "GPU"],
            choices=["CPU", "GPU", "CUDA_PROFILER", "XPU"],
            help="Profiler activities: CPU, GPU, XPU, CUDA_PROFILER. If CPU/GPU/XPU, use torch profiler. If CUDA_PROFILER, use CUDA profiler.",
        )
        parser.add_argument(
            "--profile-stage",
            type=str,
            default=BenchArgs.profile_stage,
            choices=["all", "prefill", "decode"],
            help="Which stage to profile: all, prefill, or decode only.",
        )
        parser.add_argument(
            "--profile-filename-prefix",
            type=str,
            default=BenchArgs.profile_filename_prefix,
            help="Prefix of the profiling file names. The full profiling result file(s) be "
            '"[profile_filename_prefix]_batch[batch_size]_input[input_len]_output[output_len].trace.json.gz"',
        )
        parser.add_argument(
            "--profile-start-step",
            type=int,
            default=None,
            help="Decode step at which to start profiling (0-indexed). If not specified, defaults to output_len // 2.",
        )
        parser.add_argument(
            "--profile-steps",
            type=int,
            default=None,
            help="Number of decode steps to profile starting from profile-start-step. If not specified, profiles only one step.",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # use the default value's type to cast the args into correct types.
        attrs = [(attr.name, type(attr.default)) for attr in dataclasses.fields(cls)]
        result = {}
        for attr, attr_type in attrs:
            value = getattr(args, attr)
            # Handle None values - don't try to cast them
            if value is None or attr_type == type(None):
                result[attr] = value
            else:
                result[attr] = attr_type(value)
        return cls(**result)


def load_model(server_args, port_args, gpu_id, tp_rank):
    suppress_other_loggers()
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None
    moe_ep_rank = tp_rank // (server_args.tp_size // server_args.ep_size)

    model_config = ModelConfig.from_server_args(server_args)
    runner_kwargs = dict(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=gpu_id,
        tp_rank=tp_rank,
        tp_size=server_args.tp_size,
        moe_ep_rank=moe_ep_rank,
        moe_ep_size=server_args.ep_size,
        pp_rank=0,
        pp_size=1,
        nccl_port=port_args.nccl_port,
        server_args=server_args,
    )

    _use_mlx = use_mlx()
    if _use_mlx:
        from sglang.srt.hardware_backend.mlx.model_runner_stub import (
            MlxModelRunnerStub,
        )

        model_runner = MlxModelRunnerStub(**runner_kwargs)
    else:
        model_runner = ModelRunner(**runner_kwargs)
    rank_print(f"max_total_num_tokens={model_runner.max_total_num_tokens}")
    tokenizer = get_tokenizer(
        server_args.tokenizer_path,
        tokenizer_mode=server_args.tokenizer_mode,
        trust_remote_code=server_args.trust_remote_code,
    )
    if server_args.tp_size > 1:
        dist.barrier()

    if _use_mlx:
        model_runner = _MlxBenchRunner(model_runner, server_args)
    else:
        model_runner = _TorchBenchRunner(model_runner)

    return model_runner, tokenizer


def prepare_inputs_for_correctness_test(bench_args, tokenizer, custom_prompts):
    if custom_prompts:
        custom_input_len = len(custom_prompts)
        bs = bench_args.batch_size[0]
        if custom_input_len > bs:
            logging.warning(
                f"Custom input size ({custom_input_len}) is larger than batch_size ({bs}). "
                f"Using the first {bs} prompts."
            )
            custom_prompts = custom_prompts[:bs]

    prompts = (
        custom_prompts
        if custom_prompts
        else [
            "The capital of France is",
            "The capital of the United Kindom is",
            "Today is a sunny day and I like",
        ]
    )
    input_ids = [tokenizer.encode(p) for p in prompts]
    sampling_params = SamplingParams(
        temperature=0,
        max_new_tokens=BenchArgs.output_len,
    )

    reqs = []
    for i in range(len(prompts)):
        assert len(input_ids[i]) > bench_args.cut_len

        tmp_input_ids = input_ids[i][: bench_args.cut_len]
        req = Req(
            rid=i,
            origin_input_text=prompts[i],
            origin_input_ids=tmp_input_ids,
            sampling_params=sampling_params,
        )
        req.fill_ids = req.origin_input_ids
        req.logprob_start_len = -1
        req.set_extend_input_len(len(req.fill_ids) - len(req.prefix_indices))
        reqs.append(req)

    return input_ids, reqs


def prepare_extend_inputs_for_correctness_test(
    bench_args, input_ids, reqs, model_runner
):
    for i in range(len(reqs)):
        req: Req = reqs[i]
        req.fill_ids += input_ids[i][bench_args.cut_len :]
        if model_runner is not None:
            req.prefix_indices = model_runner.req_to_token_pool.req_to_token[
                i, : bench_args.cut_len
            ].to(req.prefix_indices.dtype)
            req.logprob_start_len = -1
            req.set_extend_input_len(len(req.fill_ids) - len(req.prefix_indices))
    return reqs


def prepare_synthetic_inputs_for_latency_test(
    batch_size,
    input_len,
    custom_inputs=None,
    *,
    seed=None,
    rid_offset=0,
):
    if custom_inputs:
        input_ids = custom_inputs
    elif seed is None:
        input_ids = np.random.randint(
            0, 10000, (batch_size, input_len), dtype=np.int32
        )
    else:
        input_ids = np.random.default_rng(seed).integers(
            0, 10000, (batch_size, input_len), dtype=np.int32
        )
    sampling_params = SamplingParams(
        temperature=0,
        max_new_tokens=BenchArgs.output_len,
    )

    reqs = []
    for i in range(len(input_ids)):
        req = Req(
            rid=rid_offset + i,
            origin_input_text="",
            origin_input_ids=list(input_ids[i]),
            sampling_params=sampling_params,
        )
        req.fill_ids = req.origin_input_ids
        req.logprob_start_len = -1
        req.set_extend_input_len(len(req.fill_ids) - len(req.prefix_indices))
        reqs.append(req)

    return reqs


class TreeCacheNamespace(SimpleNamespace):
    def supports_swa(self) -> bool:
        return False

    def supports_mamba(self) -> bool:
        return False

    def is_chunk_cache(self) -> bool:
        return False

    def is_tree_cache(self) -> bool:
        return not self.is_chunk_cache()

    def evict(self, params: EvictParams):
        pass


@torch.no_grad
def extend(reqs, model_runner, sample: bool = True):
    # Create dummy tree_cache for benchmarks (no prefix caching, just allocation)
    dummy_tree_cache = TreeCacheNamespace(
        page_size=model_runner.server_args.page_size,
        device=model_runner.device,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
    )

    batch = ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        tree_cache=dummy_tree_cache,
        model_config=model_runner.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )
    batch.prepare_for_extend()
    batch.is_extend_in_batch = True
    _maybe_prepare_mlp_sync_batch(batch, model_runner)
    model_worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    logits_output = model_runner.forward(forward_batch).logits_output
    next_token_ids = (
        model_runner.sample(logits_output, forward_batch) if sample else None
    )
    return next_token_ids, logits_output.next_token_logits, batch


@torch.no_grad
def decode(input_token_ids, batch, model_runner):
    batch.output_ids = input_token_ids
    batch.prepare_for_decode()
    batch.is_extend_in_batch = False
    _maybe_prepare_mlp_sync_batch(batch, model_runner)
    model_worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    logits_output = model_runner.forward(forward_batch).logits_output
    next_token_ids = model_runner.sample(logits_output, forward_batch)
    return next_token_ids, logits_output.next_token_logits


def _maybe_prepare_mlp_sync_batch(batch: ScheduleBatch, model_runner):
    if require_mlp_sync(model_runner.server_args):
        prepare_mlp_sync_batch_raw(
            batch,
            dp_size=model_runner.server_args.dp_size,
            attn_tp_size=get_attention_tp_size(),
            attn_cp_size=model_runner.attn_cp_size,
            tp_group=model_runner.tp_group,
            get_idle_batch=None,
            disable_cuda_graph=model_runner.server_args.disable_cuda_graph,
            require_mlp_tp_gather=require_mlp_tp_gather(model_runner.server_args),
            disable_overlap_schedule=model_runner.server_args.disable_overlap_schedule,
            offload_tags=set(),
        )


class _TorchBenchRunner:
    """Wraps ModelRunner for the standard PyTorch benchmark path."""

    def __init__(self, model_runner):
        self.torch_runner = model_runner
        self._active_batch = None

    def clear(self):
        if self.is_hisparse:
            coordinator = self.torch_runner.hisparse_coordinator
            if coordinator.ack_staging_queue:
                raise RuntimeError(
                    "cannot clear one-batch while HiSparse staging is active"
                )
            if (
                coordinator.mem_pool_host.available_size()
                != coordinator.mem_pool_host.size
            ):
                raise RuntimeError(
                    "HiSparse host KV pool was not fully released by the previous run"
                )
        self.torch_runner.req_to_token_pool.clear()
        self.torch_runner.token_to_kv_pool_allocator.clear()

    def extend(self, reqs, sample: bool = True):
        return extend(reqs, self.torch_runner, sample=sample)

    def prefill(
        self,
        reqs,
        chunk_plan: ChunkPlan,
        trace_enabled: bool,
        trace_prefix: str = "prefill",
    ):
        if not chunk_plan.enabled:
            with trace_range(f"{trace_prefix}/chunk_0", trace_enabled):
                return self.extend(reqs)

        full_input_ids = [list(req.fill_ids) for req in reqs]
        result = None
        for chunk_index, (start, end) in enumerate(chunk_plan.bounds):
            prepare_chunk_requests(
                reqs=reqs,
                full_input_ids=full_input_ids,
                start=start,
                end=end,
                req_to_token=self.torch_runner.req_to_token_pool.req_to_token,
            )

            is_final_chunk = chunk_index == chunk_plan.num_chunks - 1
            with trace_range(
                f"{trace_prefix}/chunk_{chunk_index}", trace_enabled
            ):
                result = self.extend(reqs, sample=is_final_chunk)

        assert result is not None and result[0] is not None
        return result

    def _hisparse_capacity_snapshot(self) -> HiSparseCapacitySnapshot:
        allocator = self.torch_runner.token_to_kv_pool_allocator
        coordinator = self.torch_runner.hisparse_coordinator
        return HiSparseCapacitySnapshot(
            hot_total=allocator.hisparse_attn_allocator.size,
            hot_available=allocator.hisparse_attn_allocator.available_size(),
            logical_total=allocator.logical_attn_allocator.size,
            logical_available=allocator.logical_attn_allocator.available_size(),
            host_total=coordinator.mem_pool_host.size,
            host_available=coordinator.mem_pool_host.available_size(),
            request_slots_available=self.torch_runner.req_to_token_pool.available_size(),
            max_context_len=self.torch_runner.req_to_token_pool.max_context_len,
        )

    def _device_capacity_snapshot(self) -> DeviceCapacitySnapshot:
        allocator = self.torch_runner.token_to_kv_pool_allocator
        return DeviceCapacitySnapshot(
            device_total=allocator.size,
            device_available=allocator.available_size(),
            request_slots_available=(
                self.torch_runner.req_to_token_pool.available_size()
            ),
            max_context_len=self.torch_runner.req_to_token_pool.max_context_len,
        )

    def _capacity_snapshot(self):
        if self.is_hisparse:
            return self._hisparse_capacity_snapshot()
        return self._device_capacity_snapshot()

    def _cluster_min_capacity_snapshot(self, snapshot):
        names = [field.name for field in dataclasses.fields(snapshot)]
        values = [getattr(snapshot, name) for name in names]
        if dist.is_initialized() and dist.get_world_size() > 1:
            # Each wave enters EP collectives spanning the model TP world. A
            # per-attention-TP decision could let DP groups execute different
            # wave counts and deadlock those collectives.
            tensor = torch.tensor(values, dtype=torch.int64, device="cpu")
            dist.all_reduce(
                tensor,
                op=dist.ReduceOp.MIN,
                group=get_tp_group().cpu_group,
            )
            values = tensor.tolist()
        return type(snapshot)(**dict(zip(names, values)))

    def _cluster_limiting_rank(
        self,
        local_snapshot,
        cluster_snapshot,
        admission,
        ready_count: int,
    ) -> Optional[int]:
        if admission.can_admit or admission.stop_reason == "target_reached":
            return None

        reason = admission.stop_reason
        if reason == "device_pool":
            local_value, cluster_value = (
                local_snapshot.device_available,
                cluster_snapshot.device_available,
            )
        elif reason == "hot_prefill_peak":
            hot_per_req = admission.requirements.hot_per_ready_request
            local_value = min(
                local_snapshot.hot_available,
                local_snapshot.hot_total - ready_count * hot_per_req,
            )
            cluster_value = min(
                cluster_snapshot.hot_available,
                cluster_snapshot.hot_total - ready_count * hot_per_req,
            )
        elif reason == "hot_decode_reserve":
            local_value, cluster_value = (
                local_snapshot.hot_total,
                cluster_snapshot.hot_total,
            )
        elif reason == "logical_pool":
            local_value, cluster_value = (
                local_snapshot.logical_available,
                cluster_snapshot.logical_available,
            )
        elif reason == "host_pool":
            local_value, cluster_value = (
                local_snapshot.host_available,
                cluster_snapshot.host_available,
            )
        elif reason == "request_pool":
            local_value, cluster_value = (
                local_snapshot.request_slots_available,
                cluster_snapshot.request_slots_available,
            )
        elif reason == "max_context_len":
            local_value, cluster_value = (
                local_snapshot.max_context_len,
                cluster_snapshot.max_context_len,
            )
        else:
            return None

        if not dist.is_initialized() or dist.get_world_size() == 1:
            return 0
        candidate = (
            dist.get_rank()
            if local_value == cluster_value
            else dist.get_world_size()
        )
        candidate_tensor = torch.tensor(candidate, dtype=torch.int64, device="cpu")
        dist.all_reduce(
            candidate_tensor,
            op=dist.ReduceOp.MIN,
            group=get_tp_group().cpu_group,
        )
        return int(candidate_tensor.item())

    def _build_decode_batch(self, reqs):
        runner = self.torch_runner
        tree_cache = TreeCacheNamespace(
            page_size=runner.server_args.page_size,
            device=runner.device,
            req_to_token_pool=runner.req_to_token_pool,
            token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
        )
        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=runner.req_to_token_pool,
            token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
            tree_cache=tree_cache,
            model_config=runner.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        batch.req_pool_indices = torch.tensor(
            [req.req_pool_idx for req in reqs],
            dtype=torch.int64,
            device=runner.device,
        )
        seq_lens = [
            len(req.origin_input_ids) + len(req.output_ids) - 1 for req in reqs
        ]
        batch.seq_lens = torch.tensor(
            seq_lens, dtype=torch.int64, device=runner.device
        )
        batch.seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
        batch.orig_seq_lens = torch.tensor(
            seq_lens, dtype=torch.int32, device=runner.device
        )
        batch.seq_lens_sum = sum(seq_lens)
        batch.output_ids = torch.tensor(
            [req.output_ids[-1] for req in reqs],
            dtype=torch.int64,
            device=runner.device,
        )
        batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
            batch, runner.model_config.vocab_size
        )
        if self.is_hisparse:
            batch.hisparse_coordinator = runner.hisparse_coordinator
        return batch

    def prefill_waves(
        self,
        reqs,
        requested_batch_size,
        input_len,
        output_len,
        requested_chunk_size,
        effective_chunk_size,
        dp_size,
        log_prefill_wave,
        rank_print,
        trace_enabled,
    ):
        chunk_plan = build_wave_chunk_plan(
            input_len=input_len,
            requested_chunk_size=requested_chunk_size,
            effective_chunk_size=effective_chunk_size,
            page_size=self.page_size,
        )
        coordinator = (
            self.torch_runner.hisparse_coordinator if self.is_hisparse else None
        )
        ready_reqs = []
        capacity_snapshots = []
        prefill_compute_latency = 0.0
        staging_latency = 0.0
        control_latency = 0.0
        stop_reason = "target_reached"
        progress_metrics = []
        last_logged_ready_count = 0
        prefill_tic = time.perf_counter()

        def emit_progress(progress, final_reason=None):
            nonlocal last_logged_ready_count
            capacity_parts = []
            for pool_name, pool in progress["capacity"].items():
                label = {
                    "device": "device KV",
                    "hot": "hot KV",
                    "logical": "logical KV",
                    "host": "host KV",
                }[pool_name]
                capacity_parts.append(
                    f"{label}: {pool['used']}/{pool['total']} slots "
                    f"({pool['usage']:.2%})"
                )
            suffix = f", stop reason: {final_reason}" if final_reason else ""
            rank_print(
                f"Prefill wave {progress['wave']}. "
                f"BS/DP: {progress['batch_size']}, "
                f"global BS: {progress['global_batch_size']}, "
                f"wave latency: {progress['wave_latency_s']:.5f} s, "
                f"cumulative throughput/DP: "
                f"{progress['throughput_per_dp']:.2f} token/s, "
                f"cluster throughput (est.): "
                f"{progress['cluster_throughput']:.2f} token/s, "
                + ", ".join(capacity_parts)
                + suffix
            )
            last_logged_ready_count = progress["batch_size"]

        try:
            while len(ready_reqs) < requested_batch_size:
                control_tic = time.perf_counter()
                local_snapshot = self._capacity_snapshot()
                cluster_snapshot = self._cluster_min_capacity_snapshot(local_snapshot)
                if self.is_hisparse:
                    admission = evaluate_hisparse_wave_admission(
                        snapshot=cluster_snapshot,
                        ready_count=len(ready_reqs),
                        requested_batch_size=requested_batch_size,
                        input_len=input_len,
                        output_len=output_len,
                        page_size=self.page_size,
                        device_buffer_size=coordinator.device_buffer_size,
                    )
                    requirements = dataclasses.asdict(admission.requirements)
                else:
                    admission = evaluate_device_wave_admission(
                        snapshot=cluster_snapshot,
                        ready_count=len(ready_reqs),
                        requested_batch_size=requested_batch_size,
                        input_len=input_len,
                        output_len=output_len,
                        page_size=self.page_size,
                    )
                    requirements = {"required_tokens": admission.required_tokens}
                limiting_rank = self._cluster_limiting_rank(
                    local_snapshot,
                    cluster_snapshot,
                    admission,
                    len(ready_reqs),
                )
                control_latency += time.perf_counter() - control_tic
                capacity_snapshots.append(
                    {
                        "wave": len(ready_reqs),
                        "ready_count": len(ready_reqs),
                        **dataclasses.asdict(cluster_snapshot),
                        "requirements": requirements,
                        "decision": admission.stop_reason,
                        "limiting_rank": limiting_rank,
                    }
                )
                if not admission.can_admit:
                    stop_reason = admission.stop_reason
                    break

                req = reqs[len(ready_reqs)]
                wave_index = len(ready_reqs)
                wave_tic = time.perf_counter()
                self.synchronize()
                compute_tic = time.perf_counter()
                next_token_ids, _, _ = self.prefill(
                    [req],
                    chunk_plan=chunk_plan,
                    trace_enabled=trace_enabled,
                    trace_prefix=f"prefill/wave_{wave_index}",
                )
                self.synchronize()
                prefill_compute_latency += time.perf_counter() - compute_tic
                req.output_ids.append(int(next_token_ids[0].item()))

                if self.is_hisparse:
                    # Stage once after all chunks of this request have completed.
                    self.synchronize()
                    staging_tic = time.perf_counter()
                    with trace_range(
                        f"hisparse/staging/wave_{wave_index}", trace_enabled
                    ):
                        coordinator.admit_request_into_staging(req)
                        coordinator.write_staging_stream.synchronize()
                        wave_ready_reqs = coordinator.collect_ready_reqs()
                        self.synchronize()
                    staging_latency += time.perf_counter() - staging_tic
                    if len(wave_ready_reqs) != 1 or wave_ready_reqs[0] is not req:
                        raise RuntimeError(
                            "HiSparse one-batch expected exactly one ready request per "
                            "wave, got "
                            f"{[ready_req.rid for ready_req in wave_ready_reqs]}"
                        )
                else:
                    wave_ready_reqs = [req]
                ready_reqs.extend(wave_ready_reqs)

                control_tic = time.perf_counter()
                self.barrier()
                control_latency += time.perf_counter() - control_tic

                elapsed = time.perf_counter() - prefill_tic
                wave_metrics = build_prefill_wave_metrics(
                    ready_batch_size=len(ready_reqs),
                    dp_size=dp_size,
                    input_len=input_len,
                    elapsed=elapsed,
                )
                progress = {
                    "wave": wave_index + 1,
                    "batch_size": len(ready_reqs),
                    "global_batch_size": len(ready_reqs) * dp_size,
                    "wave_latency_s": time.perf_counter() - wave_tic,
                    "elapsed_s": elapsed,
                    **wave_metrics,
                    "capacity": build_capacity_usage(self._capacity_snapshot()),
                }
                progress_metrics.append(progress)
                is_final = len(ready_reqs) >= requested_batch_size
                if should_log_prefill_wave(
                    len(ready_reqs), log_prefill_wave, is_final=is_final
                ):
                    emit_progress(
                        progress,
                        final_reason="target_reached" if is_final else None,
                    )

            if (
                progress_metrics
                and log_prefill_wave > 0
                and last_logged_ready_count != len(ready_reqs)
            ):
                emit_progress(progress_metrics[-1], final_reason=stop_reason)

            if not ready_reqs:
                raise RuntimeError(
                    "KV capacity rejected the first prefill wave: "
                    f"reason={stop_reason}, snapshot={capacity_snapshots[-1]}"
                )

            batch = self._build_decode_batch(ready_reqs)
            self._active_batch = batch
        except Exception:
            for req in reqs:
                self._release_request(req)
            self._active_batch = None
            raise
        return batch.output_ids.clone(), batch, {
            "requested_batch_size": requested_batch_size,
            "batch_size": len(ready_reqs),
            "num_waves": len(ready_reqs),
            "stop_reason": stop_reason,
            "prefill_compute_latency": prefill_compute_latency,
            "staging_latency": staging_latency,
            "prefill_control_latency": control_latency,
            "capacity_snapshots": capacity_snapshots,
            "progress_metrics": progress_metrics,
            "chunk_plan": chunk_plan,
        }

    def decode(self, next_token_ids, batch):
        return decode(next_token_ids, batch, self.torch_runner)

    def _release_request(self, req):
        if req.req_pool_idx is None:
            return

        runner = self.torch_runner
        allocator = runner.token_to_kv_pool_allocator
        allocated_locs = runner.req_to_token_pool.req_to_token[
            req.req_pool_idx, : req.kv_allocated_len
        ].clone()

        if self.is_hisparse:
            coordinator = runner.hisparse_coordinator
            if req.hisparse_staging:
                coordinator.abort_staging_request(req)
            elif int(coordinator.req_device_buffer_size[req.req_pool_idx]) > 0:
                coordinator.request_finished(req)

        allocator.free(allocated_locs)
        runner.req_to_token_pool.free(req)

    def _assert_pools_restored(self):
        runner = self.torch_runner
        allocator = runner.token_to_kv_pool_allocator
        if (
            runner.req_to_token_pool.available_size()
            != runner.req_to_token_pool.size
        ):
            raise RuntimeError("request pool leaked after one-batch cleanup")
        if self.is_hisparse:
            coordinator = runner.hisparse_coordinator
            if coordinator.ack_staging_queue:
                raise RuntimeError(
                    "HiSparse staging queue was not drained during cleanup"
                )
            if (
                coordinator.mem_pool_host.available_size()
                != coordinator.mem_pool_host.size
            ):
                raise RuntimeError(
                    "HiSparse host KV pool leaked after one-batch cleanup"
                )
            if (
                allocator.logical_attn_allocator.available_size()
                != allocator.logical_attn_allocator.size
            ):
                raise RuntimeError(
                    "HiSparse logical KV pool leaked after one-batch cleanup"
                )
            if (
                allocator.hisparse_attn_allocator.available_size()
                != allocator.hisparse_attn_allocator.size
            ):
                raise RuntimeError(
                    "HiSparse hot KV pool leaked after one-batch cleanup"
                )
        elif allocator.available_size() != allocator.size:
            raise RuntimeError("device KV pool leaked after one-batch cleanup")

    def cleanup(self, batch):
        if batch is None:
            return

        try:
            for req in batch.reqs:
                self._release_request(req)
            self._assert_pools_restored()
        finally:
            if batch is self._active_batch:
                self._active_batch = None

    def cleanup_active_batch(self):
        if self._active_batch is not None:
            self.cleanup(self._active_batch)

    def synchronize(self):
        synchronize(self.torch_runner.device)

    def barrier(self):
        if dist.is_initialized() and dist.get_world_size() > 1:
            get_tp_group().barrier()

    def max_reduce(self, values):
        if not values or not dist.is_initialized() or dist.get_world_size() == 1:
            return list(values)
        tensor = torch.tensor(values, dtype=torch.float64, device="cpu")
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=get_tp_group().cpu_group)
        return tensor.tolist()

    def values_from_slowest_rank(self, values):
        if not values or not dist.is_initialized() or dist.get_world_size() == 1:
            return list(values)
        tensor = torch.tensor(values, dtype=torch.float64, device="cpu")
        gathered = [
            torch.empty_like(tensor)
            for _ in range(dist.get_world_size(group=get_tp_group().cpu_group))
        ]
        dist.all_gather(gathered, tensor, group=get_tp_group().cpu_group)
        return list(values_from_slowest_rank([item.tolist() for item in gathered]))

    @property
    def is_hisparse(self):
        return self.torch_runner.hisparse_coordinator is not None

    @property
    def page_size(self):
        return self.torch_runner.server_args.page_size

class _MlxBenchRunner:
    """Wraps MlxModelRunner for the MLX benchmark path."""

    def __init__(self, model_runner, server_args):
        from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner

        self.mlx_runner = MlxModelRunner(
            model_path=server_args.model_path,
            trust_remote_code=server_args.trust_remote_code,
        )
        self.fake_torch_runner = model_runner

    def clear(self):
        self.mlx_runner.clear()

    def extend(self, reqs):
        req_ids = [str(req.rid) for req in reqs]
        token_ids_list = [[int(t) for t in req.fill_ids] for req in reqs]
        next_token_ids = self.mlx_runner.prefill_batch(req_ids, token_ids_list)
        return torch.tensor(next_token_ids), None, req_ids

    def prefill(self, reqs, chunk_plan: ChunkPlan, trace_enabled: bool):
        if chunk_plan.enabled:
            raise ValueError("chunked one-batch prefill is not supported on MLX")
        return self.extend(reqs)

    def prefill_waves(self, *args, **kwargs):
        raise ValueError("wave one-batch prefill is not supported on MLX")

    def decode(self, next_token_ids, req_ids):
        next_token_ids = self.mlx_runner.decode_batch(req_ids)
        return torch.tensor(next_token_ids), None

    def cleanup(self, batch):
        if isinstance(batch, list):
            for req_id in batch:
                self.mlx_runner.remove_request(req_id)

    def cleanup_active_batch(self):
        pass

    def synchronize(self):
        pass

    def barrier(self):
        pass

    def max_reduce(self, values):
        return list(values)

    def values_from_slowest_rank(self, values):
        return list(values)

    @property
    def page_size(self):
        return self.fake_torch_runner.server_args.page_size

def _read_prompts_from_file(prompt_file, rank_print):
    """Read custom prompts from the file specified by `--prompt-filename`."""
    if not prompt_file:
        return []
    if not os.path.exists(prompt_file):
        rank_print(
            f"Custom prompt file {prompt_file} not found. Using default inputs..."
        )
        return []
    with open(prompt_file, "r") as pf:
        return pf.readlines()


def _get_torch_profiler_output_dir():
    return os.environ.get("SGLANG_TORCH_PROFILER_DIR", "/tmp")


def _create_torch_profiler_filename(
    profile_filename_prefix, batch_size, input_len, output_len, stage
):
    output_dir = _get_torch_profiler_output_dir()
    filename = f"{profile_filename_prefix}_batch{batch_size}_input{input_len}_output{output_len}_{stage}.trace.json.gz"
    return os.path.join(output_dir, filename)


def _save_profile_trace_results(profiler, filename):
    parent_dir = os.path.dirname(os.path.abspath(filename))
    os.makedirs(parent_dir, exist_ok=True)
    profiler.export_chrome_trace(filename)
    print(
        profiler.key_averages(group_by_input_shape=True).table(
            sort_by="self_cpu_time_total"
        )
    )


def correctness_test(
    server_args,
    port_args,
    bench_args,
    gpu_id,
    tp_rank,
):
    if (
        bench_args.chunked_prefill_size is not None
        and bench_args.chunked_prefill_size > 0
    ):
        raise ValueError(
            "--chunked-prefill-size is only supported by the one-batch latency test"
        )

    # Configure the logger
    configure_logger(server_args, prefix=f" TP{tp_rank}")
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    # Load the model
    model_runner, tokenizer = load_model(server_args, port_args, gpu_id, tp_rank)
    # Prepare inputs
    custom_prompts = _read_prompts_from_file(bench_args.prompt_filename, rank_print)
    input_ids, reqs = prepare_inputs_for_correctness_test(
        bench_args, tokenizer, custom_prompts
    )
    rank_print(f"\n{input_ids=}\n")

    batch = None
    try:
        if bench_args.cut_len > 0:
            # Prefill
            next_token_ids, next_token_logits, batch = model_runner.extend(reqs)
            rank_print(f"prefill logits (first half): {next_token_logits} \n")

            # Prepare extend inputs
            torch_runner = getattr(model_runner, "torch_runner", None)
            reqs = prepare_extend_inputs_for_correctness_test(
                bench_args, input_ids, reqs, torch_runner
            )

        # Extend (prefill w/ KV cache)
        next_token_ids, next_token_logits, batch = model_runner.extend(reqs)
        rank_print(f"prefill logits (final): {next_token_logits} \n")

        # Decode
        output_ids = [
            input_ids[i] + [next_token_ids[i]] for i in range(len(input_ids))
        ]
        for _ in range(bench_args.output_len[0] - 1):
            next_token_ids, _ = model_runner.decode(next_token_ids, batch)
            next_token_ids_list = next_token_ids.tolist()
            for i in range(len(reqs)):
                output_ids[i].append(next_token_ids_list[i])

        # Print output texts
        for i in range(len(reqs)):
            rank_print(f"========== Prompt {i} ==========")
            rank_print(tokenizer.decode(output_ids[i]), "\n")
    finally:
        model_runner.cleanup(batch)


def synchronize(device):
    torch.get_device_module(device).synchronize()


def get_deepep_phase_modes(server_args):
    if server_args.moe_a2a_backend != "deepep":
        return None

    configured = server_args.deepep_mode
    if configured == "auto":
        prefill_mode = "normal"
        decode_mode = "low_latency"
    else:
        prefill_mode = configured
        decode_mode = configured
    return {
        "configured": configured,
        "prefill": prefill_mode,
        "decode": decode_mode,
    }


def latency_test_run_once(
    run_name,
    model_runner,
    server_args,
    rank_print,
    reqs,
    batch_size,
    input_len,
    output_len,
    log_prefill_wave,
    log_decode_step,
    profile,
    profile_record_shapes,
    profile_activities,
    profile_filename_prefix,
    profile_stage,
    tp_rank,
    profile_start_step=None,
    profile_steps=None,
    requested_chunked_prefill_size=None,
):
    requested_batch_size = batch_size
    is_hisparse = getattr(model_runner, "is_hisparse", False)
    input_lengths = {len(req.fill_ids) for req in reqs}
    if input_lengths != {input_len}:
        raise ValueError(
            "wave one-batch requires equal-length inputs matching "
            f"--input-len={input_len}, got {sorted(input_lengths)}"
        )
    chunk_plan = build_wave_chunk_plan(
        input_len=input_len,
        requested_chunk_size=requested_chunked_prefill_size,
        effective_chunk_size=server_args.chunked_prefill_size,
        page_size=model_runner.page_size,
    )

    model_runner.clear()

    measurement_results = {
        "run_name": run_name,
        "requested_batch_size": requested_batch_size,
        "batch_size": requested_batch_size,
        "requested_global_batch_size": requested_batch_size * server_args.dp_size,
        "global_batch_size": requested_batch_size * server_args.dp_size,
        "input_len": input_len,
        "output_len": output_len,
        "chunked_prefill": {
            "enabled": chunk_plan.enabled,
            "requested_size": chunk_plan.requested_chunk_size,
            "effective_size": chunk_plan.effective_chunk_size,
            "per_request_chunk_size": chunk_plan.per_request_chunk_size,
            "num_chunks": chunk_plan.num_chunks,
        },
    }
    deepep_phase_modes = get_deepep_phase_modes(server_args)
    if deepep_phase_modes is not None:
        measurement_results["deepep_mode"] = deepep_phase_modes
        rank_print(
            "DeepEP phases. "
            f"configured={deepep_phase_modes['configured']}, "
            f"prefill={deepep_phase_modes['prefill']}, "
            f"decode={deepep_phase_modes['decode']}"
        )
    rank_print(
        "Chunked prefill. "
        f"enabled={chunk_plan.enabled}, "
        f"requested={chunk_plan.requested_chunk_size}, "
        f"effective={chunk_plan.effective_chunk_size}, "
        f"per_request={chunk_plan.per_request_chunk_size}, "
        f"chunks={chunk_plan.num_chunks}"
    )

    tot_latency = 0

    # No rank may start prefill before every rank has finished setup.
    model_runner.barrier()
    profiler = None
    enable_profile_prefill = profile and profile_stage in ["all", "prefill"]
    if enable_profile_prefill:
        profiler = start_profile(
            profile_activities,
            profile_record_shapes=profile_record_shapes,
            rank_print=rank_print,
        )

    trace_mark("phase/PREFILL_START", bool(profile))
    model_runner.synchronize()
    tic = time.perf_counter()
    next_token_ids, batch, wave_prefill = model_runner.prefill_waves(
        reqs=reqs,
        requested_batch_size=requested_batch_size,
        input_len=input_len,
        output_len=output_len,
        requested_chunk_size=requested_chunked_prefill_size,
        effective_chunk_size=server_args.chunked_prefill_size,
        dp_size=server_args.dp_size,
        log_prefill_wave=log_prefill_wave,
        rank_print=rank_print,
        trace_enabled=enable_profile_prefill,
    )
    batch_size = wave_prefill["batch_size"]
    measurement_results.update(
        {
            "batch_size": batch_size,
            "global_batch_size": batch_size * server_args.dp_size,
            "waves": {
                key: value
                for key, value in wave_prefill.items()
                if key != "chunk_plan"
            },
        }
    )
    if is_hisparse:
        measurement_results["hisparse_waves"] = measurement_results["waves"]
    model_runner.synchronize()
    prefill_latency = time.perf_counter() - tic

    if enable_profile_prefill:
        trace_filename = _create_torch_profiler_filename(
            profile_filename_prefix,
            requested_batch_size,
            input_len,
            output_len,
            "prefill",
        )
        stop_profile(
            profiler,
            profile_activities,
            rank_print=rank_print,
            save_trace=True,
            trace_filename=trace_filename,
            stage="prefill",
        )

    # Stop the profiler before cross-rank synchronization so barriers and
    # diagnostic reductions do not contaminate kernel timing.
    model_runner.barrier()
    trace_mark("phase_transition/PREFILL_DONE", bool(profile))
    local_control_latency = max(
        0.0,
        prefill_latency
        - wave_prefill["prefill_compute_latency"]
        - wave_prefill["staging_latency"],
    )
    prefill_phase_values = [
        prefill_latency,
        wave_prefill["prefill_compute_latency"],
        wave_prefill["staging_latency"],
        local_control_latency,
    ]
    cluster_prefill_values = model_runner.values_from_slowest_rank(
        prefill_phase_values
    )
    cluster_prefill_latency = cluster_prefill_values[0]

    tot_latency += prefill_latency
    throughput = input_len * batch_size / prefill_latency
    rank_print(
        f"Prefill. latency: {prefill_latency:6.5f} s, throughput: {throughput:9.2f} token/s"
    )
    measurement_results["prefill_latency"] = prefill_latency
    measurement_results["prefill_throughput"] = throughput
    measurement_results["prefill_compute_latency"] = cluster_prefill_values[1]
    measurement_results["staging_latency"] = cluster_prefill_values[2]
    measurement_results["prefill_control_latency"] = cluster_prefill_values[3]
    rank_print(
        "Wave prefill. "
        f"requested decode BS={requested_batch_size}, "
        f"admitted decode BS={batch_size}, "
        f"requested global BS={requested_batch_size * server_args.dp_size}, "
        f"admitted global BS={batch_size * server_args.dp_size}, "
        f"waves={wave_prefill['num_waves']}, "
        f"stop_reason={wave_prefill['stop_reason']}, "
        f"compute={measurement_results['prefill_compute_latency']:6.5f} s, "
        f"staging={measurement_results['staging_latency']:6.5f} s, "
        f"control={measurement_results['prefill_control_latency']:6.5f} s"
    )

    # This second gate makes decode start a distinct cluster-wide phase.
    model_runner.barrier()
    trace_mark("phase_transition/DECODE_START", bool(profile))

    decode_latencies = []
    # Determine profiling start step and end step
    profile_start = (
        profile_start_step if profile_start_step is not None else (output_len // 2)
    )
    profile_end = profile_start + (profile_steps if profile_steps is not None else 1)
    enable_profile_decode = profile and profile_stage in ["all", "decode"]
    profiler = None
    decode_profile_started = False
    for i in range(output_len - 1):
        model_runner.synchronize()
        # Start profiler at the specified step
        if enable_profile_decode and i == profile_start:
            profiler = start_profile(
                profile_activities,
                profile_record_shapes=profile_record_shapes,
                rank_print=rank_print,
            )
            decode_profile_started = True

        tic = time.perf_counter()
        with trace_range(
            f"decode/step_{i}",
            enable_profile_decode and profile_start <= i < profile_end,
        ):
            next_token_ids, _ = model_runner.decode(next_token_ids, batch)
        model_runner.synchronize()
        latency = time.perf_counter() - tic

        # Stop profiler after the specified number of steps
        if enable_profile_decode and decode_profile_started and i >= profile_end - 1:
            trace_filename = _create_torch_profiler_filename(
                profile_filename_prefix,
                requested_batch_size,
                input_len,
                output_len,
                "decode",
            )
            stop_profile(
                profiler,
                profile_activities,
                rank_print=rank_print,
                save_trace=True,
                trace_filename=trace_filename,
                stage="decode",
            )
            profiler = None
            decode_profile_started = False

        tot_latency += latency
        decode_metrics = build_decode_step_metrics(
            batch_size=batch_size,
            dp_size=server_args.dp_size,
            latency=latency,
        )
        decode_latencies.append(latency)
        if i < 5 or (log_decode_step > 0 and i % log_decode_step == 0):
            rank_print(
                f"Decode {i}. BS/DP: {batch_size}, "
                f"global BS: {batch_size * server_args.dp_size}, "
                f"TPOT: {decode_metrics['tpot_ms']:.3f} ms/token, "
                f"throughput/DP: "
                f"{decode_metrics['throughput_per_dp']:.2f} token/s, "
                f"cluster throughput (est.): "
                f"{decode_metrics['cluster_throughput']:.2f} token/s"
            )

    if decode_profile_started:
        trace_filename = _create_torch_profiler_filename(
            profile_filename_prefix,
            requested_batch_size,
            input_len,
            output_len,
            "decode",
        )
        stop_profile(
            profiler,
            profile_activities,
            rank_print=rank_print,
            save_trace=True,
            trace_filename=trace_filename,
            stage="decode",
        )

    trace_mark("phase_transition/DECODE_DONE", bool(profile))

    # Record decode timing from 2nd output
    med_decode_latency = None
    if output_len > 1:
        med_decode_latency = float(np.median(decode_latencies))
        med_decode_metrics = build_decode_step_metrics(
            batch_size=batch_size,
            dp_size=server_args.dp_size,
            latency=med_decode_latency,
        )
        rank_print(
            f"Decode median. BS/DP: {batch_size}, "
            f"global BS: {batch_size * server_args.dp_size}, "
            f"TPOT: {med_decode_metrics['tpot_ms']:.3f} ms/token, "
            f"throughput/DP: "
            f"{med_decode_metrics['throughput_per_dp']:.2f} token/s, "
            f"cluster throughput (est.): "
            f"{med_decode_metrics['cluster_throughput']:.2f} token/s"
        )
        measurement_results["median_decode_latency"] = med_decode_latency
        measurement_results["median_decode_tpot_ms"] = med_decode_metrics["tpot_ms"]
        measurement_results["median_decode_throughput"] = med_decode_metrics[
            "throughput_per_dp"
        ]

    throughput = (input_len + output_len) * batch_size / tot_latency
    rank_print(
        f"Total. latency: {tot_latency:6.3f} s, throughput: {throughput:9.2f} token/s"
    )
    measurement_results["total_latency"] = tot_latency
    measurement_results["overall_throughput"] = throughput

    cluster_latency_values = [tot_latency]
    if med_decode_latency is not None:
        cluster_latency_values.append(med_decode_latency)
    cluster_latency_values = model_runner.max_reduce(cluster_latency_values)
    cluster_total_latency = cluster_latency_values[0]
    cluster_median_decode_latency = (
        cluster_latency_values[1] if med_decode_latency is not None else None
    )
    cluster_metrics = build_cluster_metrics(
        batch_size=batch_size,
        requested_batch_size=requested_batch_size,
        dp_size=server_args.dp_size,
        input_len=input_len,
        output_len=output_len,
        cluster_prefill_latency=cluster_prefill_latency,
        cluster_median_decode_latency=cluster_median_decode_latency,
        cluster_total_latency=cluster_total_latency,
    )
    measurement_results.update(cluster_metrics)
    rank_print(
        "Cluster. "
        f"global batch size: {cluster_metrics['global_batch_size']}, "
        f"prefill latency: {cluster_metrics['cluster_prefill_latency']:6.5f} s, "
        f"total latency: {cluster_metrics['cluster_total_latency']:6.3f} s, "
        f"overall throughput: {cluster_metrics['cluster_overall_throughput']:9.2f} token/s"
    )
    if "cluster_median_decode_latency" in cluster_metrics:
        rank_print(
            "Cluster decode. "
            f"BS/DP: {batch_size}, "
            f"global BS: {cluster_metrics['global_batch_size']}, "
            f"TPOT: "
            f"{cluster_metrics['cluster_median_decode_latency'] * 1000:.3f} ms/token, "
            f"throughput/DP: "
            f"{batch_size / cluster_metrics['cluster_median_decode_latency']:.2f} token/s, "
            f"cluster throughput: "
            f"{cluster_metrics['cluster_median_decode_throughput']:.2f} token/s"
        )

    model_runner.cleanup(batch)
    return measurement_results


def latency_test(
    server_args,
    port_args,
    bench_args,
    gpu_id,
    tp_rank,
):
    initialize_moe_config(server_args)
    initialize_fp8_gemm_config(server_args)
    initialize_fp4_gemm_config(server_args)

    # Set CPU affinity
    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(
            server_args.pp_size, server_args.tp_size, server_args.nnodes, tp_rank
        )

    # Configure the logger
    configure_logger(server_args, prefix=f" TP{tp_rank}")
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    # Load the model
    model_runner, tokenizer = load_model(server_args, port_args, gpu_id, tp_rank)
    attention_dp_rank = (
        get_attention_dp_rank() if server_args.enable_dp_attention else 0
    )
    attention_dp_seed = seed_for_attention_dp_group(
        server_args.random_seed, attention_dp_rank
    )

    micro_warmup_shape = build_deepep_micro_warmup_shape(
        moe_a2a_backend=server_args.moe_a2a_backend,
        deepep_mode=server_args.deepep_mode,
        page_size=model_runner.page_size,
    )
    if micro_warmup_shape is not None:
        micro_batch_size, micro_input_len, micro_output_len = micro_warmup_shape
        micro_reqs = prepare_synthetic_inputs_for_latency_test(
            micro_batch_size,
            micro_input_len,
            custom_inputs=[list(range(micro_input_len))],
        )
        micro_chunk_plan = build_chunk_plan(
            input_lengths=[micro_input_len] * micro_batch_size,
            requested_chunk_size=None,
            effective_chunk_size=server_args.chunked_prefill_size,
            page_size=model_runner.page_size,
        )
        rank_print(
            "DeepEP normal micro warmup (unmeasured). "
            f"batch_size={micro_batch_size}, input_len={micro_input_len}, "
            f"output_len={micro_output_len}"
        )
        model_runner.clear()
        model_runner.synchronize()
        micro_tic = time.perf_counter()
        model_runner.prefill(
            micro_reqs,
            chunk_plan=micro_chunk_plan,
            trace_enabled=False,
        )
        model_runner.synchronize()
        rank_print(
            "DeepEP normal micro warmup finished. "
            f"latency={time.perf_counter() - micro_tic:6.5f} s"
        )
        # Keep synchronization outside the measured target warmup/benchmark.
        model_runner.barrier()
        model_runner.clear()

    # Prepare inputs for warm up
    reqs = prepare_synthetic_inputs_for_latency_test(
        bench_args.batch_size[0],
        bench_args.input_len[0],
        seed=attention_dp_seed,
        rid_offset=attention_dp_rank * bench_args.batch_size[0],
    )

    # Warm up
    rank_print("Warmup ...")
    try:
        latency_test_run_once(
            bench_args.run_name,
            model_runner,
            server_args,
            rank_print,
            reqs,
            bench_args.batch_size[0],
            bench_args.input_len[0],
            min(
                32, bench_args.output_len[0]
            ),  # shorter decoding to speed up the warmup
            log_prefill_wave=0,
            log_decode_step=0,
            profile=False,
            profile_record_shapes=False,
            profile_activities=("CPU", "GPU"),
            profile_filename_prefix="",
            profile_stage="all",
            tp_rank=tp_rank,
            profile_start_step=None,
            profile_steps=None,
            requested_chunked_prefill_size=bench_args.chunked_prefill_size,
        )
    finally:
        model_runner.cleanup_active_batch()

    rank_print("Benchmark ...")

    custom_inputs = _read_prompts_from_file(bench_args.prompt_filename, rank_print)
    custom_inputs = [tokenizer.encode(p.strip()) for p in custom_inputs]
    custom_input_len = len(custom_inputs)

    # Run the sweep
    result_list = []
    for bs, il, ol in itertools.product(
        bench_args.batch_size, bench_args.input_len, bench_args.output_len
    ):
        bs_aligned_inputs = []
        if custom_inputs:
            if custom_input_len == bs:
                bs_aligned_inputs = custom_inputs
            elif custom_input_len > bs:
                rank_print(
                    f"Custom input size ({custom_input_len}) is larger than batch_size ({bs}). "
                    f"Using the first {bs} prompts."
                )
                bs_aligned_inputs = copy.deepcopy(custom_inputs[:bs])
            else:
                rank_print(
                    f"Custom input size ({custom_input_len}) is smaller than batch_size ({bs}). "
                    f"Pad to the desired batch_size with the last prompt."
                )
                bs_aligned_inputs = copy.deepcopy(custom_inputs)
                bs_aligned_inputs.extend(
                    [bs_aligned_inputs[-1]] * (bs - custom_input_len)
                )

        reqs = prepare_synthetic_inputs_for_latency_test(
            bs,
            il,
            bs_aligned_inputs,
            seed=attention_dp_seed,
            rid_offset=attention_dp_rank * bs,
        )
        try:
            ret = latency_test_run_once(
                bench_args.run_name,
                model_runner,
                server_args,
                rank_print,
                reqs,
                bs,
                il,
                ol,
                bench_args.log_prefill_wave,
                bench_args.log_decode_step,
                bench_args.profile if tp_rank == 0 else None,
                bench_args.profile_record_shapes if tp_rank == 0 else None,
                bench_args.profile_activities,
                bench_args.profile_filename_prefix,
                bench_args.profile_stage,
                tp_rank,
                bench_args.profile_start_step,
                bench_args.profile_steps,
                bench_args.chunked_prefill_size,
            )
        finally:
            model_runner.cleanup_active_batch()
        if ret is not None:
            result_list.append(ret)

    # Write results in jsonlines format on rank 0.
    if tp_rank == 0 and bench_args.result_filename:
        with open(bench_args.result_filename, "a") as fout:
            for result in result_list:
                fout.write(json.dumps(result) + "\n")


def run_worker(work_func, server_args, port_args, bench_args, gpu_id, tp_rank):
    try:
        work_func(server_args, port_args, bench_args, gpu_id, tp_rank)
    finally:
        if dist.is_initialized():
            for cleanup in (destroy_model_parallel, destroy_distributed_environment):
                try:
                    cleanup()
                except Exception:
                    logging.exception(
                        "Failed to run distributed cleanup: %s", cleanup.__name__
                    )


def wait_for_workers(workers):
    pending = list(workers)
    failure = None
    while pending and failure is None:
        for proc in list(pending):
            proc.join(timeout=0.1)
            if proc.is_alive():
                continue
            pending.remove(proc)
            if proc.exitcode != 0:
                failure = (proc.pid, proc.exitcode)
                break

    if failure is not None:
        for proc in pending:
            proc.terminate()
        for proc in pending:
            proc.join()
        pid, exitcode = failure
        raise RuntimeError(
            f"one-batch worker pid={pid} failed with exit code {exitcode}"
        )


def main(server_args, bench_args):
    server_args.cuda_graph_max_bs = max(bench_args.batch_size)

    _set_envs_and_config(server_args)

    if server_args.model_path:
        if bench_args.correctness_test:
            work_func = correctness_test
        else:
            work_func = latency_test
    else:
        raise ValueError(
            "Provide --model-path for running the tests or "
            "provide --result-filename for plotting the results"
        )

    port_args = PortArgs.init_new(server_args)

    assignments = get_local_rank_assignments(
        server_args.tp_size, server_args.nnodes, server_args.node_rank
    )
    if server_args.tp_size == 1:
        run_worker(work_func, server_args, port_args, bench_args, 0, 0)
    else:
        workers = []
        for tp_rank, local_gpu_id in assignments:
            with maybe_reindex_device_id(local_gpu_id) as gpu_id:
                proc = multiprocessing.Process(
                    target=run_worker,
                    args=(
                        work_func,
                        server_args,
                        port_args,
                        bench_args,
                        gpu_id,
                        tp_rank,
                    ),
                )
                proc.start()
                workers.append(proc)

        wait_for_workers(workers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    bench_args = BenchArgs.from_cli_args(args)

    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    try:
        main(server_args, bench_args)
    finally:
        if server_args.tp_size != 1:
            kill_process_tree(os.getpid(), include_parent=False)
