"""Pure helpers for ``sglang.bench_one_batch``.

This module intentionally has no torch or SGLang runtime imports so its rank,
chunk, and metric calculations can be unit tested on CPU-only machines.
"""

from __future__ import annotations

from dataclasses import dataclass, fields as dataclass_fields
from pathlib import Path
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
class DecodeProfileStepAction:
    profile: bool
    force_eager: bool
    exit_after_step: bool


@dataclass(frozen=True)
class DecodeProfilePlan:
    enabled: bool
    start_step: int
    end_step: int
    execution_mode: str
    exit_after_capture: bool

    @property
    def profiled_steps(self) -> int:
        return self.end_step - self.start_step if self.enabled else 0

    def action_for_step(self, step: int) -> DecodeProfileStepAction:
        in_window = self.enabled and self.start_step <= step < self.end_step
        return DecodeProfileStepAction(
            profile=in_window,
            force_eager=in_window and self.execution_mode == "eager",
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
class WorkerRanks:
    attn_dp_rank: int
    attn_tp_rank: int
    attn_tp_size: int
    attn_cp_rank: int
    moe_dp_rank: int
    moe_ep_rank: int


@dataclass(frozen=True)
class HBMUsageSnapshot:
    total_bytes: int
    free_bytes: int
    torch_active_bytes: int
    torch_reserved_bytes: int
    model_bytes: int
    kv_data_bytes: int
    kv_indexer_bytes: int
    cuda_graph_bytes: int
    draft_model_bytes: int = 0
    draft_kv_data_bytes: int = 0
    draft_kv_indexer_bytes: int = 0
    deepep_configured_bytes: int = 0


def normalize_profile_activities(
    profile_activities: Sequence[str],
) -> tuple[str, ...]:
    """Keep CPU ranges whenever CUDA kernels are captured by torch profiler."""
    activities = list(dict.fromkeys(profile_activities))
    if "GPU" in activities and "CPU" not in activities:
        activities.insert(0, "CPU")
    return tuple(activities)


def resolve_one_batch_cuda_graph_max_bs(
    configured_max_bs: Optional[int], batch_sizes: Sequence[int]
) -> int:
    """Preserve an explicit graph limit and infer one only when absent."""
    if configured_max_bs is not None:
        return configured_max_bs
    if not batch_sizes:
        raise ValueError("batch_sizes must not be empty")
    return max(batch_sizes)


def build_profile_trace_filename(
    *,
    output_dir: str,
    prefix: str,
    batch_size: int,
    input_len: int,
    output_len: int,
    stage: str,
    tp_size: int,
    dp_size: int,
    ep_size: int,
) -> str:
    """Build a trace name from the admitted batch and runtime topology."""
    filename = (
        f"{prefix}_tp{tp_size}_dp{dp_size}_ep{ep_size}_batch{batch_size}_"
        f"input{input_len}_output{output_len}_{stage}.trace.json.gz"
    )
    return str(Path(output_dir) / filename)


def build_decode_profile_plan(
    output_len: int,
    profile_enabled: bool,
    profile_stage: str,
    profile_start_step: Optional[int],
    profile_steps: Optional[int],
    execution_mode: str,
    exit_after_capture: bool,
) -> DecodeProfilePlan:
    """Validate and describe the decode profiler capture window."""
    if output_len <= 0:
        raise ValueError(f"output_len must be positive, got {output_len}")
    if execution_mode not in ("runtime", "eager"):
        raise ValueError(
            "profile execution mode must be 'runtime' or 'eager', "
            f"got {execution_mode!r}"
        )

    has_profile_control = execution_mode != "runtime" or exit_after_capture
    if has_profile_control and not profile_enabled:
        raise ValueError(
            "--profile-execution-mode eager and --profile-exit-after-capture "
            "require --profile"
        )

    decode_enabled = profile_enabled and profile_stage in ("all", "decode")
    if has_profile_control and not decode_enabled:
        raise ValueError(
            "profile execution-mode and early-exit controls require a decode "
            "profile stage"
        )

    if not decode_enabled:
        return DecodeProfilePlan(
            enabled=False,
            start_step=0,
            end_step=0,
            execution_mode="runtime",
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
        execution_mode=execution_mode,
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


def exclusive_cuda_storage_bytes(
    tensors: Sequence[object], excluded_tensors: Sequence[object]
) -> int:
    """Count CUDA storage in ``tensors`` that is not shared with exclusions."""

    excluded = set()
    for tensor in excluded_tensors:
        device = str(getattr(tensor, "device", ""))
        if device.startswith("cuda"):
            storage = tensor.untyped_storage()
            excluded.add((device, storage.data_ptr()))

    storages: dict[tuple[str, int], int] = {}
    for tensor in tensors:
        device = str(getattr(tensor, "device", ""))
        if not device.startswith("cuda"):
            continue
        storage = tensor.untyped_storage()
        key = (device, storage.data_ptr())
        if key not in excluded:
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
    """Build a non-overlapping per-GPU HBM capacity ledger."""
    values = {
        field.name: getattr(snapshot, field.name)
        for field in dataclass_fields(HBMUsageSnapshot)
    }
    if any(value < 0 for value in values.values()):
        raise ValueError("HBM usage values must be non-negative")
    if snapshot.free_bytes > snapshot.total_bytes:
        raise ValueError("free HBM cannot exceed total HBM")

    used_bytes = snapshot.total_bytes - snapshot.free_bytes
    if snapshot.torch_active_bytes > snapshot.torch_reserved_bytes:
        raise ValueError("PyTorch active HBM cannot exceed reserved HBM")
    if snapshot.torch_reserved_bytes > used_bytes:
        raise ValueError("PyTorch reserved HBM cannot exceed driver-used HBM")

    known_active_bytes = (
        snapshot.model_bytes
        + snapshot.kv_data_bytes
        + snapshot.kv_indexer_bytes
        + snapshot.draft_model_bytes
        + snapshot.draft_kv_data_bytes
        + snapshot.draft_kv_indexer_bytes
    )
    if known_active_bytes > snapshot.torch_active_bytes:
        raise ValueError(
            "known active HBM categories exceed PyTorch active HBM: "
            f"known={known_active_bytes}, active={snapshot.torch_active_bytes}"
        )

    torch_active_other_bytes = snapshot.torch_active_bytes - known_active_bytes
    torch_inactive_cache_bytes = (
        snapshot.torch_reserved_bytes - snapshot.torch_active_bytes
    )
    native_external_bytes = used_bytes - snapshot.torch_reserved_bytes
    effective_available_bytes = snapshot.free_bytes + torch_inactive_cache_bytes

    return {
        **values,
        "used_bytes": used_bytes,
        "torch_active_other_bytes": torch_active_other_bytes,
        "torch_inactive_cache_bytes": torch_inactive_cache_bytes,
        "native_external_bytes": native_external_bytes,
        "effective_available_bytes": effective_available_bytes,
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


def resolve_target_warmup_log_interval(
    skip_target_warmup: bool, log_prefill_wave: int
) -> Optional[int]:
    """Return the target-shape warmup log interval, or None when skipped."""
    return None if skip_target_warmup else log_prefill_wave


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


def derive_worker_ranks(
    *,
    tp_rank: int,
    tp_size: int,
    dp_size: int,
    attn_cp_size: int,
    moe_dp_size: int,
    ep_size: int,
    enable_dp_attention: bool,
) -> WorkerRanks:
    """Mirror the online controller's attention and MoE rank layout."""
    values = (tp_size, dp_size, attn_cp_size, moe_dp_size, ep_size)
    if any(value <= 0 for value in values):
        raise ValueError("parallel sizes must be positive")
    if tp_rank < 0 or tp_rank >= tp_size:
        raise ValueError(f"tp_rank must be in [0, {tp_size}), got {tp_rank}")

    attn_dp_size = dp_size if enable_dp_attention else 1
    attn_denominator = attn_dp_size * attn_cp_size
    if tp_size % attn_denominator != 0:
        raise ValueError(
            "tp_size must be divisible by dp_size * attn_cp_size for DP attention"
        )
    attn_tp_size = tp_size // attn_denominator
    attn_tp_rank = tp_rank % attn_tp_size
    attn_cp_rank = (tp_rank // attn_tp_size) % attn_cp_size
    attn_dp_rank = (
        tp_rank // (attn_tp_size * attn_cp_size)
        if enable_dp_attention
        else 0
    )

    if tp_size % moe_dp_size != 0:
        raise ValueError("tp_size must be divisible by moe_dp_size")
    ranks_per_moe_dp = tp_size // moe_dp_size
    if ranks_per_moe_dp % ep_size != 0:
        raise ValueError("ranks per MoE DP group must be divisible by ep_size")
    moe_dp_rank = tp_rank // ranks_per_moe_dp
    moe_ep_rank = (tp_rank % ranks_per_moe_dp) // (ranks_per_moe_dp // ep_size)
    return WorkerRanks(
        attn_dp_rank=attn_dp_rank,
        attn_tp_rank=attn_tp_rank,
        attn_tp_size=attn_tp_size,
        attn_cp_rank=attn_cp_rank,
        moe_dp_rank=moe_dp_rank,
        moe_ep_rank=moe_ep_rank,
    )


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


def speculative_slot_reserve(
    *, num_steps: int, topk: int, num_draft_tokens: int, page_size: int
) -> int:
    """Return the page-aligned peak temporary target-KV reserve per request."""
    if min(num_steps, topk, num_draft_tokens, page_size) <= 0:
        raise ValueError("speculative decoding sizes must be positive")
    return _align_up(max(num_steps * topk, num_draft_tokens), page_size)


def merge_speculative_inputs(spec_inputs: Sequence[object]):
    """Merge per-wave draft states in request order."""
    if not spec_inputs:
        raise ValueError("at least one speculative input is required")
    merged = spec_inputs[0]
    for spec_input in spec_inputs[1:]:
        merged.merge_batch(spec_input)
    return merged


def unfinished_request_indices(
    generated_lengths: Sequence[int], output_len: int
) -> list[int]:
    if output_len <= 0:
        raise ValueError("output_len must be positive")
    if any(length < 0 or length > output_len for length in generated_lengths):
        raise ValueError("generated lengths must be between zero and output_len")
    return [
        index
        for index, generated_length in enumerate(generated_lengths)
        if generated_length < output_len
    ]


def validate_one_batch_speculative_mode(
    *,
    spec_algorithm: Optional[str],
    enable_hisparse: bool,
    correctness_test: bool,
    profile_enabled: bool,
    profile_execution_mode: str,
    model_path: str,
    draft_model_path: Optional[str],
) -> None:
    if spec_algorithm is None:
        return
    if spec_algorithm.upper() != "EAGLE":
        raise ValueError("one-batch speculative decoding currently supports EAGLE only")
    if enable_hisparse:
        raise ValueError("one-batch MTP does not yet support HiSparse")
    if correctness_test:
        raise ValueError("one-batch MTP does not support correctness mode")
    if profile_enabled and profile_execution_mode != "runtime":
        raise ValueError("one-batch MTP profiling requires runtime execution mode")
    if draft_model_path is not None and draft_model_path != model_path:
        raise ValueError("one-batch MTP does not support an external draft model")


def seed_for_attention_dp_group(random_seed: int, attention_dp_rank: int) -> int:
    if attention_dp_rank < 0:
        raise ValueError("attention_dp_rank must be non-negative")
    return random_seed + attention_dp_rank


def build_mtp_draft_extend_prefix_lens(
    pre_verify_seq_lens: Sequence[int],
) -> list[int]:
    """Snapshot the sequence prefix consumed before an EAGLE verify cycle."""
    prefix_lens = [int(seq_len) for seq_len in pre_verify_seq_lens]
    if any(seq_len < 0 for seq_len in prefix_lens):
        raise ValueError("MTP draft-extend prefix lengths must be non-negative")
    return prefix_lens


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


def build_mtp_cycle_metrics(
    *,
    accepted_draft_tokens: Sequence[int],
    raw_accepted_draft_tokens: Optional[Sequence[int]] = None,
    speculative_num_steps: int,
    process_latency: float,
    core_cycle_latency: float,
) -> dict[str, float | int]:
    """Build metrics for one synchronous draft/verify/commit/extend cycle.

    ``accepted_draft_tokens`` contains tokens retained by the benchmark after
    fixed-length clipping. ``raw_accepted_draft_tokens`` contains the verify
    result before clipping and is used only for acceptance diagnostics.
    """
    active_batch_size = len(accepted_draft_tokens)
    if active_batch_size <= 0:
        raise ValueError("an MTP cycle must contain at least one active request")
    if speculative_num_steps <= 0:
        raise ValueError("speculative_num_steps must be positive")
    if process_latency <= 0 or core_cycle_latency <= 0:
        raise ValueError("MTP cycle latencies must be positive")
    if any(
        accepted < 0 or accepted > speculative_num_steps
        for accepted in accepted_draft_tokens
    ):
        raise ValueError("accepted draft token counts are out of range")
    if raw_accepted_draft_tokens is None:
        raw_accepted_draft_tokens = accepted_draft_tokens
    if len(raw_accepted_draft_tokens) != active_batch_size:
        raise ValueError("raw and committed acceptance counts must have equal length")
    if any(
        raw < committed or raw > speculative_num_steps
        for raw, committed in zip(
            raw_accepted_draft_tokens, accepted_draft_tokens
        )
    ):
        raise ValueError("raw accepted draft token counts are out of range")

    accepted_draft = sum(accepted_draft_tokens)
    raw_accepted_draft = sum(raw_accepted_draft_tokens)
    accepted_tokens = accepted_draft + active_batch_size
    raw_accepted_tokens = raw_accepted_draft + active_batch_size
    trimmed_tokens = raw_accepted_draft - accepted_draft
    average_accepted_length = accepted_tokens / active_batch_size
    return {
        "active_batch_size": active_batch_size,
        "accepted_draft_tokens": accepted_draft,
        "accepted_tokens": accepted_tokens,
        "raw_accepted_draft_tokens": raw_accepted_draft,
        "raw_accepted_tokens": raw_accepted_tokens,
        "trimmed_tokens": trimmed_tokens,
        "average_accepted_length": average_accepted_length,
        "raw_average_accepted_length": (
            raw_accepted_tokens / active_batch_size
        ),
        "process_latency_ms": process_latency * 1000,
        "core_cycle_latency_ms": core_cycle_latency * 1000,
        "normalized_tpot_ms": (
            core_cycle_latency / average_accepted_length * 1000
        ),
        "throughput_per_dp": accepted_tokens / core_cycle_latency,
        "draft_acceptance_rate": (
            raw_accepted_draft
            / (active_batch_size * speculative_num_steps)
        ),
        "committed_draft_acceptance_rate": (
            accepted_draft / (active_batch_size * speculative_num_steps)
        ),
    }


def build_mtp_acceptance_accounting(
    *,
    raw_accepted_draft_before: Sequence[int],
    raw_accepted_draft_after: Sequence[int],
    committed_accepted_draft_tokens: Sequence[int],
) -> dict[str, tuple[int, ...] | int]:
    """Separate raw EAGLE acceptance from fixed-length committed work."""
    if not (
        len(raw_accepted_draft_before)
        == len(raw_accepted_draft_after)
        == len(committed_accepted_draft_tokens)
    ):
        raise ValueError("MTP acceptance vectors must have equal length")

    raw_per_req = tuple(
        after - before
        for before, after in zip(
            raw_accepted_draft_before, raw_accepted_draft_after
        )
    )
    committed_per_req = tuple(committed_accepted_draft_tokens)
    if any(raw < 0 for raw in raw_per_req):
        raise ValueError("raw MTP acceptance counters must be monotonic")
    if any(committed < 0 for committed in committed_per_req):
        raise ValueError("committed MTP acceptance counts must be non-negative")
    if any(
        committed > raw
        for raw, committed in zip(raw_per_req, committed_per_req)
    ):
        raise ValueError("committed MTP acceptance cannot exceed raw acceptance")

    trimmed_per_req = tuple(
        raw - committed
        for raw, committed in zip(raw_per_req, committed_per_req)
    )
    active_batch_size = len(raw_per_req)
    return {
        "raw_accepted_draft_tokens_per_req": raw_per_req,
        "committed_accepted_draft_tokens_per_req": committed_per_req,
        "trimmed_tokens_per_req": trimmed_per_req,
        "raw_accepted_tokens": sum(raw_per_req) + active_batch_size,
        "committed_accepted_tokens": (
            sum(committed_per_req) + active_batch_size
        ),
        "trimmed_tokens": sum(trimmed_per_req),
    }


def fixed_output_length_reached(
    *, current_output_len: int, target_output_len: int
) -> bool:
    """Return whether a fixed-length benchmark request is exactly complete."""
    if target_output_len <= 0:
        raise ValueError("target output length must be positive")
    if current_output_len < 0:
        raise ValueError("current output length must be non-negative")
    if current_output_len > target_output_len:
        raise RuntimeError(
            "fixed-length benchmark request exceeded its target output length"
        )
    return current_output_len == target_output_len


def build_mtp_cluster_cycle_metrics(
    *, unique_dp_accepted_tokens: int, max_core_cycle_latency: float
) -> dict[str, float | int]:
    if unique_dp_accepted_tokens < 0:
        raise ValueError("unique DP accepted tokens must be non-negative")
    if max_core_cycle_latency <= 0:
        raise ValueError("max core cycle latency must be positive")
    return {
        "cluster_accepted_tokens": unique_dp_accepted_tokens,
        "cluster_core_cycle_latency_ms": max_core_cycle_latency * 1000,
        "cluster_throughput": (
            unique_dp_accepted_tokens / max_core_cycle_latency
        ),
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
    speculative_reserve_per_request: int = 0,
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
    if speculative_reserve_per_request < 0:
        raise ValueError("speculative_reserve_per_request must be non-negative")

    aligned_input_len = _align_up(input_len, page_size)
    aligned_full_len = _align_up(input_len + output_len, page_size)
    next_request_peak = max(aligned_full_len, input_len + page_size)
    required_tokens = (
        ready_count * (aligned_full_len - aligned_input_len) + next_request_peak
        + (ready_count + 1) * speculative_reserve_per_request
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
