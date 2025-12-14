import math
import torch


def compute_mllm_cross_attention_mask(
    num_dit_tokens: int,
    num_text_tokens: int,
    num_mllm_video_tokens: int,
    s_dit: int,
    s_mllm: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Build a causal mask so DiT tokens only see past MLLM video tokens.
    """
    total_blocks = math.ceil(num_dit_tokens / s_dit) if s_dit > 0 else 0
    max_k = num_text_tokens + num_mllm_video_tokens
    mask = torch.zeros((num_dit_tokens, max_k), device=device, dtype=torch.bool)
    if total_blocks == 0 or max_k == 0:
        return mask
    for block_id in range(total_blocks):
        q_start = block_id * s_dit
        q_end = min(num_dit_tokens, q_start + s_dit)
        allowed_video_tokens = max(0, (block_id - 1) * s_mllm)
        allowed_video_tokens = min(allowed_video_tokens, num_mllm_video_tokens)
        k_end = num_text_tokens + allowed_video_tokens
        if k_end > 0:
            mask[q_start:q_end, :k_end] = True
    return mask
