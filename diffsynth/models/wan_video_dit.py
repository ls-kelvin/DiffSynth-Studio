import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from einops import rearrange
from .wan_video_camera_controller import SimpleAdapter
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextDecoderLayer,
    Qwen3VLTextRMSNorm,
    Qwen3VLTextRotaryEmbedding,
)

try:
    import os
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    
    USE_FLEX_ATTENTION = os.environ.get("DISABLE_FLEX_ATTENTION", "0") != "1"
    
    # 默认开启 compile（可用 FLEX_ATTENTION_COMPILE=0 关闭）
    _FLEX_COMPILE_ENABLED = os.environ.get("FLEX_ATTENTION_COMPILE", "1") == "1"
    _FLEX_ATTN_COMPILED = None

    def _flex_attention_call(q, k, v, *, block_mask=None, score_mod=None, kernel_options=None):
        global _FLEX_ATTN_COMPILED
        if _FLEX_COMPILE_ENABLED:
            if _FLEX_ATTN_COMPILED is None:
                _FLEX_ATTN_COMPILED = torch.compile(flex_attention)
            return _FLEX_ATTN_COMPILED(q, k, v, block_mask=block_mask, score_mod=score_mod, kernel_options=kernel_options)
        return flex_attention(q, k, v, block_mask=block_mask, score_mod=score_mod, kernel_options=kernel_options)

except ImportError:
    USE_FLEX_ATTENTION = False
    create_block_mask = None

    def _flex_attention_call(*args, **kwargs):
        raise RuntimeError("flex_attention is not available in this PyTorch build.")


try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False
    
    
def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False):
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        if isinstance(x,tuple):
            x = x[0]
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
    def forward(self, q, k, v):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs, block_mask: Optional[object] = None):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        if USE_FLEX_ATTENTION and (block_mask is not None):
            q_h = rearrange(q, "b s (n d) -> b n s d", n=self.num_heads).contiguous()
            k_h = rearrange(k, "b s (n d) -> b n s d", n=self.num_heads).contiguous()
            v_h = rearrange(v, "b s (n d) -> b n s d", n=self.num_heads).contiguous()
            x = _flex_attention_call(q_h, k_h, v_h, block_mask=block_mask)
            x = rearrange(x, "b n s d -> b s (n d)", n=self.num_heads)
        else:
            x = self.attn(q, k, v)
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False, has_mllm_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        self.has_mllm_input = has_mllm_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)
        if has_mllm_input:
            # K/V projection for already-projected MLLM embeddings
            self.k_mllm = nn.Linear(dim, dim)
            self.v_mllm = nn.Linear(dim, dim)
            self.norm_k_mllm = RMSNorm(dim, eps=eps)
            self.fuse_linear = nn.Linear(dim, dim)
            nn.init.zeros_(self.fuse_linear.weight)
            nn.init.zeros_(self.fuse_linear.bias)
            
        self.attn = AttentionModule(self.num_heads)
        

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mllm_embeddings: Optional[torch.Tensor] = None,
        mllm_mask: Optional[torch.Tensor] = None,          # 仍保留给 SDPA fallback 用
        mllm_block_mask: Optional[object] = None,          # flex 用：在 WanModel.forward 里只构建一次
    ):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y

        q = self.norm_q(self.q(x))

        # T5 cross-attention
        k_t5 = self.norm_k(self.k(ctx))
        v_t5 = self.v(ctx)
        x_t5 = self.attn(q, k_t5, v_t5)

        x = x_t5

        # MLLM decoupled attention
        if self.has_mllm_input and (mllm_embeddings is not None):
            k_m = self.norm_k_mllm(self.k_mllm(mllm_embeddings))
            v_m = self.v_mllm(mllm_embeddings)

            q_h = rearrange(q,   "b s (n d) -> b n s d", n=self.num_heads).contiguous()
            k_h = rearrange(k_m, "b s (n d) -> b n s d", n=self.num_heads).contiguous()
            v_h = rearrange(v_m, "b s (n d) -> b n s d", n=self.num_heads).contiguous()

            # Flex 路径：使用“外部一次性构建”的 block_mask
            if USE_FLEX_ATTENTION and (mllm_block_mask is not None):
                x_mllm = _flex_attention_call(q_h, k_h, v_h, block_mask=mllm_block_mask)
                x_mllm = rearrange(x_mllm, "b n s d -> b s (n d)", n=self.num_heads)
            elif os.getenv("CAUSAL_MASK", "1") == "0":
                # 非因果遮罩路径
                x_mllm = flash_attention(q, k_m, v_m, num_heads=self.num_heads)
            else:
                # SDPA fallback（使用 prefix 长度向量构造允许的 KV）
                attn_mask = None
                if mllm_mask is not None:
                    kv_len = v_h.shape[2]
                    allowed = torch.arange(kv_len, device=q.device) < mllm_mask.unsqueeze(-1).to(device=q.device)
                    attn_mask = (~allowed).unsqueeze(1).expand(q_h.shape[0], self.num_heads, q_h.shape[2], k_h.shape[2])
                x_mllm = F.scaled_dot_product_attention(q_h, k_h, v_h, attn_mask=attn_mask)
                x_mllm = rearrange(x_mllm, "b n s d -> b s (n d)", n=self.num_heads)
                
            x = x + self.fuse_linear(x_mllm)

        # Image cross-attention (if available)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y_img = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y_img
            
        return self.o(x)

class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual

class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6, has_mllm_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input, has_mllm_input=has_mllm_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward(
        self,
        x,
        context,
        t_mod,
        freqs,
        mllm_embeddings: Optional[torch.Tensor] = None,
        mllm_mask: Optional[torch.Tensor] = None,
        mllm_block_mask: Optional[object] = None,
        dit_block_mask: Optional[object] = None,
    ):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=chunk_dim)

        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )

        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs, block_mask=dit_block_mask))

        x = x + self.cross_attn(
            self.norm3(x),
            context,
            mllm_embeddings=mllm_embeddings,
            mllm_mask=mllm_mask,
            mllm_block_mask=mllm_block_mask,
        )

        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x



class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Qwen3VLMllmEmbedding(nn.Module):
    def __init__(self, out_dim: int, num_layers: int = 2):
        super().__init__()
        self.config = Qwen3VLTextConfig(
            hidden_size=2560,
            intermediate_size=9728,
            num_hidden_layers=num_layers,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            rope_theta=5000000,
            max_position_embeddings=262144,
            attention_bias=False,
            attention_dropout=0.0,
            rope_scaling={
                "rope_type": "default",
                "mrope_interleaved": True,
                "mrope_section": [24, 20, 20],
            },
            attn_implementation="flex_attention" if USE_FLEX_ATTENTION else None,
        )
        self.layers = nn.ModuleList(
            [Qwen3VLTextDecoderLayer(self.config, layer_idx) for layer_idx in range(num_layers)]
        )
        self.norm = Qwen3VLTextRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(self.config)
        self.proj = MLP(self.config.hidden_size, out_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        mllm_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # Align position ids with Qwen3VL text model expectations
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device, dtype=torch.long).unsqueeze(0)
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]
        position_ids = position_ids.to(device=hidden_states.device)
        text_position_ids = text_position_ids.to(device=hidden_states.device)
        if text_position_ids.shape[0] != batch_size:
            text_position_ids = text_position_ids.expand(batch_size, -1)

        if mllm_mask is not None:
            prefix_cross = mllm_mask.to(device=hidden_states.device, dtype=torch.int64)
            if prefix_cross.shape[0] != batch_size:
                prefix_cross = prefix_cross.expand(batch_size, -1)

            self_prefix = self.compute_self_prefix(prefix_cross, seq_len, hidden_states.device)

            kv_idx = torch.arange(seq_len, device=hidden_states.device).view(1, 1, seq_len)
            allowed = kv_idx < self_prefix.unsqueeze(-1)  # (B, Q, KV)
            attn_mask = (~allowed).unsqueeze(1)  # (B,1,Q,KV)

        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attn_mask,
                position_ids=text_position_ids,
            )
        hidden_states = self.norm(hidden_states)
        return self.proj(hidden_states)

    @staticmethod
    def compute_self_prefix(cross_prefix: torch.Tensor, seq_len: int, device) -> torch.Tensor:
        """
        Convert cross prefix (B, Lc) into self prefix (B, seq_len) such that:
        for each unique rising value v in cross_prefix, positions [prev, v) can see up to v kv tokens.
        Example: cross=[1,1,4,4,7,7] -> self=[1,4,4,4,7,7,7] when seq_len>=7.
        """
        batch_size = cross_prefix.shape[0]
        self_prefix = torch.full((batch_size, seq_len), seq_len, device=device, dtype=torch.int64)
        for b in range(batch_size):
            uniques = torch.unique_consecutive(cross_prefix[b]).tolist()
            prev = 0
            last_v = seq_len
            for v in uniques:
                v_int = int(v)
                end = min(v_int, seq_len)
                if end > prev:
                    self_prefix[b, prev:end] = v_int
                prev = min(end, seq_len)
                last_v = v_int
                if prev >= seq_len:
                    break
            if prev < seq_len:
                self_prefix[b, prev:] = min(last_v, seq_len)
        return self_prefix


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        if len(t_mod.shape) == 3:
            shift, scale = (self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)).chunk(2, dim=2)
            x = (self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2)))
        else:
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


class WanModel(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
        has_mllm_input: bool = False,
        mllm_embed_num_layers: int = 2,
    ):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.has_mllm_input = has_mllm_input
        if has_mllm_input:
            # Adapt MLLM hidden states via lightweight Qwen3VL text layers, then map to model dim.
            self.mllm_embedding = Qwen3VLMllmEmbedding(dim, num_layers=mllm_embed_num_layers)
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps, has_mllm_input=has_mllm_input)
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        if add_control_adapter:
            self.control_adapter = SimpleAdapter(in_dim_control_adapter, dim, kernel_size=patch_size[1:], stride=patch_size[1:])
        else:
            self.control_adapter = None

    def patchify(self, x: torch.Tensor, control_camera_latents_input: Optional[torch.Tensor] = None):
        x = self.patch_embedding(x)
        if self.control_adapter is not None and control_camera_latents_input is not None:
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        return x

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                mllm_hidden_states: Optional[torch.Tensor] = None,
                mllm_mask: Optional[torch.Tensor] = None,
                mllm_kv_len: Optional[int] = None,
                mllm_position_ids: Optional[torch.Tensor] = None,
                dit_block_mask: Optional[object] = None,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                **kwargs,
                ):
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep).to(x.dtype))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)
        
        if self.has_image_input:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)
        
        mllm_embeddings = None
        # Process MLLM hidden states to MLLM embeddings (outer layer responsibility)
        if self.has_mllm_input and mllm_hidden_states is not None:
            # Expect mllm_hidden_states shape (B, L, Hdim)
            mllm_embeddings = self.mllm_embedding(
                mllm_hidden_states,
                position_ids=mllm_position_ids,
                mllm_mask=mllm_mask,
            )

        # Build flex block_mask once (if flex available) for cross-attn
        mllm_block_mask = None
        if (
            USE_FLEX_ATTENTION
            and create_block_mask is not None
            and mllm_embeddings is not None
            and mllm_mask is not None
        ):
            B, Q_LEN = mllm_mask.shape
            KV_LEN = mllm_kv_len if mllm_kv_len is not None else int(mllm_mask.max().item())

            def mask_mod(b, h, q_idx, kv_idx):
                return kv_idx < mllm_mask[b, q_idx]

            mllm_block_mask = create_block_mask(
                mask_mod,
                B=B,
                H=None,
                Q_LEN=Q_LEN,
                KV_LEN=KV_LEN,
                device=str(x.device),
            )

        x, (f, h, w) = self.patchify(x)
        
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
        
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        if (
            dit_block_mask is None
            and USE_FLEX_ATTENTION
            and create_block_mask is not None
            and x.dim() == 3
        ):
            B, Q_LEN = x.shape[0], x.shape[1]
            KV_LEN = x.shape[1]

            def mask_mod(b, h, q_idx, kv_idx):
                return ((q_idx // (1560*8)) * (1560*8) <= kv_idx) & (kv_idx < (q_idx // (1560*8) + 1) * (1560*8))

            dit_block_mask = create_block_mask(
                mask_mod,
                B=B,
                H=None,
                Q_LEN=Q_LEN,
                KV_LEN=KV_LEN,
                device=str(x.device),
            )

        for block in self.blocks:
            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod, freqs, mllm_embeddings, mllm_mask, mllm_block_mask, dit_block_mask,
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs, mllm_embeddings, mllm_mask, mllm_block_mask, dit_block_mask,
                        use_reentrant=False,
                    )
            else:
                x = block(
                    x, context, t_mod, freqs,
                    mllm_embeddings=mllm_embeddings,
                    mllm_mask=mllm_mask,
                    mllm_block_mask=mllm_block_mask,
                    dit_block_mask=dit_block_mask,
                )

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x

    def load_state_dict(self, state_dict, assign: bool = False, strict: bool = True):
        """Custom load_state_dict to support partial loading for backward compatibility.

        When strict=False, missing keys in the provided state_dict are ignored,
        preventing errors when loading old checkpoints that lack newly added MLLM parameters.
        """
        # Simply delegate to parent with strict=False to allow missing keys
        return super().load_state_dict(state_dict, assign=assign, strict=False)
