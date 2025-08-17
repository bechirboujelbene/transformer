"""Utility helpers under the transformer_core namespace."""

from transformer_core.util.transformer_util import create_causal_mask
# Optional: expose count_parameters for convenience
try:
    from transformer_core.util.notebook_util import count_parameters  # type: ignore
except Exception:  # pragma: no cover
    count_parameters = None  # not required for library usage

__all__ = [
    "create_causal_mask",
    "count_parameters",
]
