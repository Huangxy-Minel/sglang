"""Pure helpers for ``sglang.bench_one_batch``.

This module intentionally has no torch or SGLang runtime imports so its rank,
chunk, and metric calculations can be unit tested on CPU-only machines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class ChunkPlan:
    enabled: bool
    requested_chunk_size: Optional[int]
    effective_chunk_size: Optional[int]
    per_request_chunk_size: int
    num_chunks: int
    bounds: tuple[tuple[int, int], ...]


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

    virtual_hot_available = min(
        snapshot.hot_available,
        snapshot.hot_total - ready_count * hot_per_ready_request,
    )
    if virtual_hot_available < requirements.hot_prefill_peak:
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
        result.update(
            {
                "cluster_median_decode_latency": cluster_median_decode_latency,
                "cluster_median_decode_throughput": (
                    global_batch_size / cluster_median_decode_latency
                ),
            }
        )
    return result
