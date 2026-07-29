import inspect
from typing import Any


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
