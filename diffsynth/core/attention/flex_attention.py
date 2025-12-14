import torch
from einops import rearrange
from typing import Callable

try:
    from torch.nn.attention import flex_attention as torch_flex_attention
    FLEX_ATTENTION_AVAILABLE = True
except Exception:
    FLEX_ATTENTION_AVAILABLE = False


def make_score_mod_from_mask(mask: torch.Tensor) -> Callable:
    """Create a score modifier for flex_attention from a boolean mask."""
    mask = mask.bool()

    def score_mod(batch: int, head: int, q_idx: int, k_idx: int):
        return 0.0 if mask[q_idx, k_idx] else float("-inf")

    return score_mod


def flex_cross_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask_fn: Callable,
    num_heads: int,
) -> torch.Tensor:
    """
    Apply flex_attention with a custom score modifier.
    """
    if not FLEX_ATTENTION_AVAILABLE:
        raise ImportError("torch.nn.attention.flex_attention is not available in this environment.")
    q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
    k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
    v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
    out = torch_flex_attention(q, k, v, score_mod=mask_fn)
    out = rearrange(out, "b n s d -> b s (n d)", n=num_heads)
    return out
