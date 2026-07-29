import inspect
from typing import Any


def use_fp8_normal_dispatch(
    enable_jit_deepgemm: bool,
    is_cutlass: bool,
    force_bf16_dispatch: bool,
) -> bool:
    return enable_jit_deepgemm and not is_cutlass and not force_bf16_dispatch


def get_normal_dispatch_hidden_bytes(
    hidden_size: int,
    use_fp8_dispatch: bool,
) -> int:
    return hidden_size * (1 if use_fp8_dispatch else 2)


def get_dispatch_config(
    buffer_cls: Any,
    num_ranks: int,
    real_hidden_bytes: int,
):
    """Get a dispatch config across legacy and Blackwell DeepEP APIs."""
    parameters = inspect.signature(buffer_cls.get_dispatch_config).parameters
    if "real_hidden_bytes" in parameters:
        return buffer_cls.get_dispatch_config(
            num_ranks,
            real_hidden_bytes=real_hidden_bytes,
        )

    return buffer_cls.get_dispatch_config(num_ranks)


def get_normal_buffer_size_hints(
    dispatch_config: Any,
    combine_config: Any,
    dispatch_hidden_bytes: int,
    combine_hidden_bytes: int,
    num_ranks: int,
) -> tuple[int, int]:
    dispatch_nvl_bytes = dispatch_config.get_nvl_buffer_size_hint(
        dispatch_hidden_bytes,
        num_ranks,
    )
    dispatch_rdma_bytes = dispatch_config.get_rdma_buffer_size_hint(
        dispatch_hidden_bytes,
        num_ranks,
    )
    combine_nvl_bytes = combine_config.get_nvl_buffer_size_hint(
        combine_hidden_bytes,
        num_ranks,
    )
    combine_rdma_bytes = combine_config.get_rdma_buffer_size_hint(
        combine_hidden_bytes,
        num_ranks,
    )
    return (
        max(dispatch_nvl_bytes, combine_nvl_bytes),
        max(dispatch_rdma_bytes, combine_rdma_bytes),
    )
