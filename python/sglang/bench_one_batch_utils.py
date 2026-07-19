"""Pure helpers for ``sglang.bench_one_batch``.

This module intentionally has no torch or SGLang runtime imports so its rank,
chunk, and metric calculations can be unit tested on CPU-only machines.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, fields as dataclass_fields
from typing import Any, Optional, Sequence


@dataclass(frozen=True)
class ChunkPlan:
    enabled: bool
    requested_chunk_size: Optional[int]
    effective_chunk_size: Optional[int]
    per_request_chunk_size: int
    num_chunks: int
    bounds: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class DecodeProfileStepAction:
    profile: bool
    force_eager: bool
    exit_after_step: bool


@dataclass(frozen=True)
class DecodeProfilePlan:
    enabled: bool
    start_step: int
    end_step: int
    force_eager: bool
    exit_after_capture: bool

    @property
    def profiled_steps(self) -> int:
        return self.end_step - self.start_step if self.enabled else 0

    @property
    def executed_steps_after_capture(self) -> int:
        return self.end_step if self.enabled and self.exit_after_capture else 0

    def action_for_step(self, step: int) -> DecodeProfileStepAction:
        in_window = self.enabled and self.start_step <= step < self.end_step
        return DecodeProfileStepAction(
            profile=in_window,
            force_eager=in_window and self.force_eager,
            exit_after_step=(
                in_window
                and self.exit_after_capture
                and step == self.end_step - 1
            ),
        )


@dataclass(frozen=True)
class HiSparseCapacitySnapshot:
    hot_total: int
    hot_available: int
    logical_total: int
    logical_available: int
    host_total: int
    host_available: int
    request_slots_available: int
    max_context_len: int


@dataclass(frozen=True)
class HiSparseCapacityRequirements:
    hot_per_ready_request: int
    hot_prefill_peak: int
    logical_for_next_wave: int
    host_for_next_wave: int


@dataclass(frozen=True)
class HiSparseAdmissionDecision:
    can_admit: bool
    stop_reason: str
    requirements: HiSparseCapacityRequirements


@dataclass(frozen=True)
class DeviceCapacitySnapshot:
    device_total: int
    device_available: int
    request_slots_available: int
    max_context_len: int


@dataclass(frozen=True)
class DeviceAdmissionDecision:
    can_admit: bool
    stop_reason: str
    required_tokens: int


@dataclass(frozen=True)
class HBMUsageSnapshot:
    total_bytes: int
    free_bytes: int
    model_bytes: int
    kv_data_bytes: int
    kv_indexer_bytes: int
    cuda_graph_bytes: int
    deepep_configured_bytes: int = 0


def normalize_profile_activities(
    profile_activities: Sequence[str],
) -> tuple[str, ...]:
    """Keep CPU ranges whenever CUDA kernels are captured by torch profiler."""
    activities = list(dict.fromkeys(profile_activities))
    if "GPU" in activities and "CPU" not in activities:
        activities.insert(0, "CPU")
    return tuple(activities)


@contextmanager
def disable_cuda_graph_replay(model_runner: Any, enabled: bool):
    """Temporarily make both SGLang CUDA graph runners unavailable."""
    if not enabled:
        yield
        return

    graph_runner = model_runner.graph_runner
    piecewise_graph_runner = model_runner.piecewise_cuda_graph_runner
    model_runner.graph_runner = None
    model_runner.piecewise_cuda_graph_runner = None
    try:
        yield
    finally:
        model_runner.graph_runner = graph_runner
        model_runner.piecewise_cuda_graph_runner = piecewise_graph_runner


def build_decode_profile_plan(
    output_len: int,
    profile_enabled: bool,
    profile_stage: str,
    profile_start_step: Optional[int],
    profile_steps: Optional[int],
    force_eager: bool,
    exit_after_capture: bool,
) -> DecodeProfilePlan:
    """Validate and describe the decode profiler capture window."""
    if output_len <= 0:
        raise ValueError(f"output_len must be positive, got {output_len}")

    has_profile_control = force_eager or exit_after_capture
    if has_profile_control and not profile_enabled:
        raise ValueError(
            "--profile-force-eager and --profile-exit-after-capture require --profile"
        )

    decode_enabled = profile_enabled and profile_stage in ("all", "decode")
    if has_profile_control and not decode_enabled:
        raise ValueError(
            "profile force-eager and early-exit controls require a decode profile stage"
        )

    if not decode_enabled:
        return DecodeProfilePlan(
            enabled=False,
            start_step=0,
            end_step=0,
            force_eager=False,
            exit_after_capture=False,
        )

    start_step = (
        profile_start_step if profile_start_step is not None else output_len // 2
    )
    steps = profile_steps if profile_steps is not None else 1
    if start_step < 0:
        raise ValueError(f"profile_start_step must be non-negative, got {start_step}")
    if steps <= 0:
        raise ValueError(f"profile_steps must be positive, got {steps}")

    decode_iterations = output_len - 1
    end_step = start_step + steps
    if end_step > decode_iterations:
        raise ValueError(
            "decode profile window exceeds available decode iterations: "
            f"start={start_step}, steps={steps}, "
            f"decode_iterations={decode_iterations}"
        )

    return DecodeProfilePlan(
        enabled=True,
        start_step=start_step,
        end_step=end_step,
        force_eager=force_eager,
        exit_after_capture=exit_after_capture,
    )


def unique_cuda_storage_bytes(tensors: Sequence[object]) -> int:
    """Count CUDA tensor storage once even when tensors share a backing store."""
    storages: dict[tuple[str, int], int] = {}
    for tensor in tensors:
        device = str(getattr(tensor, "device", ""))
        if not device.startswith("cuda"):
            continue
        storage = tensor.untyped_storage()
        key = (device, storage.data_ptr())
        storages[key] = max(storages.get(key, 0), storage.nbytes())
    return sum(storages.values())


def split_kv_pool_bytes(total_bytes: int, indexer_bytes: int) -> dict[str, int]:
    if total_bytes < 0 or indexer_bytes < 0:
        raise ValueError("KV storage values must be non-negative")
    if indexer_bytes > total_bytes:
        raise ValueError(
            "indexer KV storage cannot exceed total KV pool storage: "
            f"indexer={indexer_bytes}, total={total_bytes}"
        )
    return {
        "kv_data_bytes": total_bytes - indexer_bytes,
        "kv_indexer_bytes": indexer_bytes,
    }


def build_hbm_usage(snapshot: HBMUsageSnapshot) -> dict[str, int]:
    """Build a non-overlapping per-GPU HBM ledger."""
    values = {
        field.name: getattr(snapshot, field.name)
        for field in dataclass_fields(HBMUsageSnapshot)
    }
    if any(value < 0 for value in values.values()):
        raise ValueError("HBM usage values must be non-negative")
    if snapshot.free_bytes > snapshot.total_bytes:
        raise ValueError("free HBM cannot exceed total HBM")

    used_bytes = snapshot.total_bytes - snapshot.free_bytes
    known_used_bytes = (
        snapshot.model_bytes
        + snapshot.kv_data_bytes
        + snapshot.kv_indexer_bytes
        + snapshot.cuda_graph_bytes
    )
    if known_used_bytes > used_bytes:
        raise ValueError(
            "known HBM categories exceed actual used HBM: "
            f"known={known_used_bytes}, used={used_bytes}"
        )

    return {
        **values,
        "used_bytes": used_bytes,
        "other_bytes": used_bytes - known_used_bytes,
    }


def summarize_hbm_usage(
    snapshots: Sequence[HBMUsageSnapshot],
) -> dict[str, object]:
    """Summarize per-GPU HBM ledgers across ranks as min/avg/max."""
    if not snapshots:
        raise ValueError("at least one HBM snapshot is required")

    usages = [build_hbm_usage(snapshot) for snapshot in snapshots]
    summary: dict[str, object] = {"num_ranks": len(usages)}
    for name in usages[0]:
        values = [usage[name] for usage in usages]
        summary[name] = {
            "min": min(values),
            "avg": sum(values) / len(values),
            "max": max(values),
        }
    return summary


def build_deepep_micro_warmup_shape(
    moe_a2a_backend: str,
    deepep_mode: str,
    page_size: int,
) -> Optional[tuple[int, int, int]]:
    """Return a small prefill-only shape for initializing normal DeepEP."""
    if moe_a2a_backend != "deepep" or deepep_mode not in ("auto", "normal"):
        return None
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")

    input_len = ((max(64, page_size) + page_size - 1) // page_size) * page_size
    return 1, input_len, 1


def get_local_rank_assignments(
    tp_size: int, nnodes: int, node_rank: int
) -> list[tuple[int, int]]:
    """Return ``(global_rank, local_gpu_id)`` pairs for one node."""
    if nnodes <= 0:
        raise ValueError(f"nnodes must be positive, got {nnodes}")
    if tp_size <= 0 or tp_size % nnodes != 0:
        raise ValueError(
            f"tp_size must be positive and divisible by nnodes, got "
            f"tp_size={tp_size}, nnodes={nnodes}"
        )
    if node_rank < 0 or node_rank >= nnodes:
        raise ValueError(
            f"node_rank must be in [0, {nnodes}), got node_rank={node_rank}"
        )

    ranks_per_node = tp_size // nnodes
    global_start = node_rank * ranks_per_node
    return [
        (global_start + local_gpu_id, local_gpu_id)
        for local_gpu_id in range(ranks_per_node)
    ]


def build_chunk_plan(
    input_lengths: Sequence[int],
    requested_chunk_size: Optional[int],
    effective_chunk_size: Optional[int],
    page_size: int,
) -> ChunkPlan:
    """Build an equal-length static-batch chunk plan.

    ``requested_chunk_size`` preserves whether the CLI option was explicitly
    supplied. ``effective_chunk_size`` is the value after ServerArgs applies
    hardware defaults and DP-attention adjustment.
    """
    if not input_lengths:
        raise ValueError("input_lengths must not be empty")
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if any(length <= 0 for length in input_lengths):
        raise ValueError("all input lengths must be positive")

    enabled = requested_chunk_size is not None and requested_chunk_size > 0
    if not enabled:
        input_len = max(input_lengths)
        return ChunkPlan(
            enabled=False,
            requested_chunk_size=requested_chunk_size,
            effective_chunk_size=effective_chunk_size,
            per_request_chunk_size=input_len,
            num_chunks=1,
            bounds=((0, input_len),),
        )

    if len(set(input_lengths)) != 1:
        raise ValueError(
            "chunked one-batch prefill currently requires equal-length inputs"
        )
    if effective_chunk_size is None or effective_chunk_size <= 0:
        raise ValueError(
            "an explicitly enabled chunk size must remain positive after "
            "ServerArgs adjustment"
        )

    batch_size = len(input_lengths)
    minimum_budget = batch_size * page_size
    if effective_chunk_size < minimum_budget:
        raise ValueError(
            "chunked one-batch prefill needs at least one page per request: "
            f"effective_chunk_size={effective_chunk_size}, "
            f"minimum={minimum_budget}"
        )

    per_request_chunk_size = (
        effective_chunk_size // batch_size // page_size * page_size
    )
    input_len = input_lengths[0]
    bounds = tuple(
        (start, min(start + per_request_chunk_size, input_len))
        for start in range(0, input_len, per_request_chunk_size)
    )
    return ChunkPlan(
        enabled=True,
        requested_chunk_size=requested_chunk_size,
        effective_chunk_size=effective_chunk_size,
        per_request_chunk_size=per_request_chunk_size,
        num_chunks=len(bounds),
        bounds=bounds,
    )


def build_wave_chunk_plan(
    input_len: int,
    requested_chunk_size: Optional[int],
    effective_chunk_size: Optional[int],
    page_size: int,
) -> ChunkPlan:
    """Build the chunk plan for one request in each attention DP group."""
    return build_chunk_plan(
        input_lengths=[input_len],
        requested_chunk_size=requested_chunk_size,
        effective_chunk_size=effective_chunk_size,
        page_size=page_size,
    )


def global_requested_batch_size(batch_size: int, dp_size: int) -> int:
    if batch_size <= 0 or dp_size <= 0:
        raise ValueError("batch_size and dp_size must be positive")
    return batch_size * dp_size


def seed_for_attention_dp_group(random_seed: int, attention_dp_rank: int) -> int:
    if attention_dp_rank < 0:
        raise ValueError("attention_dp_rank must be non-negative")
    return random_seed + attention_dp_rank


def should_log_prefill_wave(
    ready_batch_size: int, log_interval: int, is_final: bool = False
) -> bool:
    """Return whether a completed prefill wave should emit a progress log."""
    if log_interval <= 0:
        return False
    return ready_batch_size <= 5 or ready_batch_size % log_interval == 0 or is_final


def build_decode_step_metrics(
    batch_size: int,
    dp_size: int,
    process_latency: float,
    core_forward_tpot: float,
) -> dict[str, float]:
    """Build full-step latency and core-forward TPOT metrics."""
    if batch_size <= 0 or dp_size <= 0:
        raise ValueError("batch_size and dp_size must be positive")
    if process_latency <= 0 or core_forward_tpot <= 0:
        raise ValueError("process_latency and core_forward_tpot must be positive")
    return {
        "process_latency_ms": process_latency * 1000,
        "tpot_ms": core_forward_tpot * 1000,
        "throughput_per_dp": batch_size / core_forward_tpot,
        "cluster_throughput": batch_size * dp_size / core_forward_tpot,
    }


def build_prefill_wave_metrics(
    ready_batch_size: int, dp_size: int, input_len: int, elapsed: float
) -> dict[str, float]:
    """Build cumulative throughput metrics after a completed prefill wave."""
    if ready_batch_size <= 0 or dp_size <= 0 or input_len <= 0:
        raise ValueError("ready_batch_size, dp_size, and input_len must be positive")
    if elapsed <= 0:
        raise ValueError("elapsed must be positive")
    throughput_per_dp = ready_batch_size * input_len / elapsed
    return {
        "throughput_per_dp": throughput_per_dp,
        "cluster_throughput": throughput_per_dp * dp_size,
    }


def _pool_usage(total: int, available: int) -> dict[str, int | float]:
    if total <= 0 or available < 0 or available > total:
        raise ValueError(
            f"invalid pool capacity: total={total}, available={available}"
        )
    used = total - available
    return {
        "used": used,
        "available": available,
        "total": total,
        "usage": used / total,
    }


def build_capacity_usage(
    snapshot: DeviceCapacitySnapshot | HiSparseCapacitySnapshot,
) -> dict[str, dict[str, int | float]]:
    """Convert a capacity snapshot to used/available/total pool metrics."""
    if isinstance(snapshot, DeviceCapacitySnapshot):
        return {
            "device": _pool_usage(snapshot.device_total, snapshot.device_available)
        }
    if isinstance(snapshot, HiSparseCapacitySnapshot):
        return {
            "hot": _pool_usage(snapshot.hot_total, snapshot.hot_available),
            "logical": _pool_usage(
                snapshot.logical_total, snapshot.logical_available
            ),
            "host": _pool_usage(snapshot.host_total, snapshot.host_available),
        }
    raise TypeError(f"unsupported capacity snapshot: {type(snapshot).__name__}")


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def evaluate_device_wave_admission(
    snapshot: DeviceCapacitySnapshot,
    ready_count: int,
    requested_batch_size: int,
    input_len: int,
    output_len: int,
    page_size: int,
) -> DeviceAdmissionDecision:
    """Check whether a device-only KV pool can admit the next request wave.

    The current free count already excludes the input KV of ready requests. The
    additional requirement reserves their remaining decode growth plus the
    complete input/output lifetime of the request in the next wave.
    """
    if ready_count < 0:
        raise ValueError("ready_count must be non-negative")
    if requested_batch_size <= 0:
        raise ValueError("requested_batch_size must be positive")
    if input_len <= 0 or output_len <= 0:
        raise ValueError("input_len and output_len must be positive")
    if page_size <= 0:
        raise ValueError("page_size must be positive")

    aligned_input_len = _align_up(input_len, page_size)
    aligned_full_len = _align_up(input_len + output_len, page_size)
    next_request_peak = max(aligned_full_len, input_len + page_size)
    required_tokens = (
        ready_count * (aligned_full_len - aligned_input_len) + next_request_peak
    )

    def decision(can_admit: bool, reason: str) -> DeviceAdmissionDecision:
        return DeviceAdmissionDecision(can_admit, reason, required_tokens)

    if ready_count >= requested_batch_size:
        return decision(False, "target_reached")
    if input_len + output_len > snapshot.max_context_len:
        return decision(False, "max_context_len")
    if snapshot.request_slots_available < 1:
        return decision(False, "request_pool")
    if snapshot.device_available < required_tokens:
        return decision(False, "device_pool")
    return decision(True, "admitted")


def evaluate_hisparse_wave_admission(
    snapshot: HiSparseCapacitySnapshot,
    ready_count: int,
    requested_batch_size: int,
    input_len: int,
    output_len: int,
    page_size: int,
    device_buffer_size: int,
) -> HiSparseAdmissionDecision:
    """Check whether one more request per attention DP group can be prefetched.

    Allocator free counts only describe allocations made so far. This check also
    reserves the future decode growth of requests that have already completed
    prefill, so a large requested decode batch cannot overcommit the hot pool.
    """
    if ready_count < 0:
        raise ValueError("ready_count must be non-negative")
    if requested_batch_size <= 0:
        raise ValueError("requested_batch_size must be positive")
    if input_len <= 0 or output_len <= 0:
        raise ValueError("input_len and output_len must be positive")
    if page_size <= 0 or device_buffer_size <= 0:
        raise ValueError("page_size and device_buffer_size must be positive")

    full_len = input_len + output_len
    aligned_input_len = _align_up(input_len, page_size)
    aligned_full_len = _align_up(full_len, page_size)
    next_request_logical_peak = max(
        aligned_full_len,
        input_len + page_size,
    )
    hot_per_ready_request = min(aligned_full_len, device_buffer_size)
    if hot_per_ready_request == device_buffer_size:
        hot_per_ready_request += page_size

    requirements = HiSparseCapacityRequirements(
        hot_per_ready_request=hot_per_ready_request,
        hot_prefill_peak=input_len + page_size,
        logical_for_next_wave=(
            ready_count * (aligned_full_len - aligned_input_len)
            + next_request_logical_peak
        ),
        host_for_next_wave=ready_count * output_len + full_len,
    )

    def decision(can_admit: bool, reason: str) -> HiSparseAdmissionDecision:
        return HiSparseAdmissionDecision(can_admit, reason, requirements)

    if ready_count >= requested_batch_size:
        return decision(False, "target_reached")
    if full_len > snapshot.max_context_len:
        return decision(False, "max_context_len")
    if snapshot.request_slots_available < 1:
        return decision(False, "request_pool")

    # One-batch keeps completed requests host-only until all prefill waves are
    # done. Their decode buffers are hydrated together immediately before
    # decode, so they do not consume hot slots during the next prefill wave.
    if snapshot.hot_available < requirements.hot_prefill_peak:
        return decision(False, "hot_prefill_peak")
    if snapshot.hot_total < (ready_count + 1) * hot_per_ready_request:
        return decision(False, "hot_decode_reserve")
    if snapshot.logical_available < requirements.logical_for_next_wave:
        return decision(False, "logical_pool")
    if snapshot.host_available < requirements.host_for_next_wave:
        return decision(False, "host_pool")
    return decision(True, "admitted")


def prepare_chunk_requests(
    reqs,
    full_input_ids: Sequence[Sequence[int]],
    start: int,
    end: int,
    req_to_token,
) -> None:
    """Advance static requests to one chunk while preserving their pool slots."""
    for req, input_ids in zip(reqs, full_input_ids):
        if start > 0:
            # write_cache_indices reads prefix tensor pointers as int64.
            req.prefix_indices = (
                req_to_token[req.req_pool_idx, :start].clone().long()
            )
        req.fill_ids = list(input_ids[:end])
        req.set_extend_input_len(end - start)


def values_from_slowest_rank(
    per_rank_values: Sequence[Sequence[float]],
) -> tuple[float, ...]:
    """Return the complete metric row whose first value is largest."""
    if not per_rank_values or not per_rank_values[0]:
        raise ValueError("per_rank_values must contain non-empty rows")
    width = len(per_rank_values[0])
    if any(len(values) != width for values in per_rank_values):
        raise ValueError("all per-rank metric rows must have the same width")
    return tuple(max(per_rank_values, key=lambda values: values[0]))


def build_cluster_metrics(
    batch_size: int,
    dp_size: int,
    input_len: int,
    output_len: int,
    cluster_prefill_latency: float,
    cluster_median_decode_latency: Optional[float],
    cluster_total_latency: float,
    requested_batch_size: Optional[int] = None,
    cluster_median_core_forward_tpot: Optional[float] = None,
) -> dict[str, float | int]:
    """Calculate metrics from latencies already max-reduced across ranks."""
    if cluster_prefill_latency <= 0:
        raise ValueError("cluster_prefill_latency must be positive")
    if cluster_total_latency <= 0:
        raise ValueError("cluster_total_latency must be positive")
    if (
        cluster_median_decode_latency is not None
        and cluster_median_decode_latency <= 0
    ):
        raise ValueError("cluster_median_decode_latency must be positive")
    if (
        cluster_median_core_forward_tpot is not None
        and cluster_median_core_forward_tpot <= 0
    ):
        raise ValueError("cluster_median_core_forward_tpot must be positive")

    requested_batch_size = requested_batch_size or batch_size
    global_batch_size = batch_size * dp_size
    result: dict[str, float | int] = {
        "requested_batch_size": requested_batch_size,
        "batch_size": batch_size,
        "requested_global_batch_size": global_requested_batch_size(
            requested_batch_size, dp_size
        ),
        "global_batch_size": global_batch_size,
        "cluster_prefill_latency": cluster_prefill_latency,
        "cluster_prefill_throughput": (
            input_len * global_batch_size / cluster_prefill_latency
        ),
        "cluster_total_latency": cluster_total_latency,
        "cluster_overall_throughput": (
            (input_len + output_len) * global_batch_size / cluster_total_latency
        ),
    }
    if cluster_median_decode_latency is not None:
        core_tpot = (
            cluster_median_core_forward_tpot
            if cluster_median_core_forward_tpot is not None
            else cluster_median_decode_latency
        )
        result.update(
            {
                "cluster_median_decode_latency": cluster_median_decode_latency,
                "cluster_median_decode_latency_ms": (
                    cluster_median_decode_latency * 1000
                ),
                "cluster_median_core_forward_tpot": core_tpot,
                "cluster_median_core_forward_tpot_ms": core_tpot * 1000,
                "cluster_median_decode_throughput_per_dp": (
                    batch_size / core_tpot
                ),
                "cluster_median_decode_throughput": (
                    global_batch_size / core_tpot
                ),
            }
        )
    return result
