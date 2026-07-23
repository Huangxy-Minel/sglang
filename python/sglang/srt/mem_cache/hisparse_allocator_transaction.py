"""Transaction bookkeeping for temporary HiSparse allocator mutations."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class HiSparseAllocatorTransaction:
    logical_state: Any
    hot_state: Any
    mapping_updates: list[tuple[Any, Any]] = field(default_factory=list)

    def record_mapping_update(self, mapping, indices) -> None:
        self.mapping_updates.append(
            (indices.clone(), mapping[indices].clone())
        )

    def restore_mapping(self, mapping) -> None:
        for indices, previous_values in reversed(self.mapping_updates):
            mapping[indices] = previous_values
