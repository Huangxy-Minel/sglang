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


def build_cluster_metrics(
    batch_size: int,
    dp_size: int,
    input_len: int,
    output_len: int,
    cluster_prefill_latency: float,
    cluster_median_decode_latency: Optional[float],
    cluster_total_latency: float,
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

    global_batch_size = batch_size * dp_size
    result: dict[str, float | int] = {
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
