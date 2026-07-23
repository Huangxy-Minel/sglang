"""Pure helpers for speculative worker memory configuration."""

from typing import Optional


def resolve_draft_token_capacity(
    *,
    target_device_capacity: int,
    target_logical_capacity: Optional[int],
) -> int:
    capacity = (
        target_logical_capacity
        if target_logical_capacity is not None
        else target_device_capacity
    )
    if capacity <= 0:
        raise ValueError("draft token capacity must be positive")
    return capacity
