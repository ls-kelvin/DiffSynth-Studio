"""
Autoregressive inference for Wan Video with MLLM condition.

During training, dit_block_mask ensures that the i-th block can only see 
its own region's DiT hidden states, while mllm_mask ensures it can only 
see previous regions' MLLM conditions.

This inference code mimics that behavior by generating blocks sequentially:
1. First denoise block 0 with only T5 text + MLLM text (no video condition)
2. Feed the generated video block 0 to MLLM to get its condition
3. Denoise block 1 with T5 text + MLLM text + block 0's MLLM condition
4. Continue for subsequent blocks...

Key optimization: Each block is denoised independently with only its own latents,
avoiding redundant computation on the full sequence.
"""

import os
import torch
import math
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional, List, Union
from einops import rearrange

from .wan_video import (
    WanVideoPipeline,
    WanVideoUnit_ShapeChecker,
    WanVideoUnit_NoiseInitializer,
    WanVideoUnit_PromptEmbedder,
    WanVideoUnit_MLLMEmbedder,
    BLOCK_DURATION,
)
from ..models.wan_video_dit import WanModel, sinusoidal_embedding_1d
from ..core import ModelConfig

_DEBUG_COMPARE_MASKS = os.environ.get("DEBUG_COMPARE_MASKS", "0") == "1"

class WanVideoAutoregressivePipeline(WanVideoPipeline):
    """
    Autoregressive inference pipeline that generates video blocks sequentially,
    using MLLM conditions from previously generated blocks.
    
    This pipeline generates video in a block-by-block manner:
    - Block 0: Generated with text-only MLLM condition
    - Block 1: Generated with MLLM condition from block 0's video
    - Block N: Generated with MLLM condition from blocks 0 to N-1's video
    
    Each block corresponds to approximately 2 seconds of video (BLOCK_DURATION).
    """
    
    def __init__(self, device="cuda", torch_dtype=torch.bfloat16):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.mllm_embedder = WanVideoUnit_MLLMEmbedder()
    
    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
        mllm_processor_config: ModelConfig = None,
        redirect_common_files: bool = False,
        use_usp: bool = False,
        vram_limit: float = None,
    ):
        """Load pretrained models for autoregressive inference."""
        # Use parent's from_pretrained but return our pipeline type
        parent_pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch_dtype,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            mllm_processor_config=mllm_processor_config,
            redirect_common_files=redirect_common_files,
            use_usp=use_usp,
            vram_limit=vram_limit,
        )
        
        # Create autoregressive pipeline with parent's attributes
        pipe = WanVideoAutoregressivePipeline(device=device, torch_dtype=torch_dtype)
        
        # Copy all model references
        for attr in ['tokenizer', 'text_encoder', 'image_encoder', 'dit', 'dit2',
                     'vae', 'motion_controller', 'vace', 'vace2', 'vap',
                     'animate_adapter', 'audio_encoder', 'mllm_encoder', 
                     'mllm_processor', 'audio_processor', 'scheduler']:
            if hasattr(parent_pipe, attr):
                setattr(pipe, attr, getattr(parent_pipe, attr))
        
        pipe.vram_management_enabled = parent_pipe.vram_management_enabled
        return pipe
    
    def calculate_num_blocks(self, num_frames: int, height: int, width: int) -> int:
        """
        Calculate number of temporal blocks based on video dimensions.
        
        Each block contains 4 * BLOCK_DURATION latent frames (8 latent frames for BLOCK_DURATION=2).
        """
        num_dit_frames = 1 + (num_frames - 1) // 4
        latent_frames_per_block = 4 * BLOCK_DURATION
        num_blocks = math.ceil(num_dit_frames / latent_frames_per_block)
        return num_blocks

    def get_block_latent_range(self, block_idx: int, num_frames: int) -> tuple:
        """
        Get the latent frame range for a specific block.
        Returns (start_latent, end_latent) indices.
        """
        num_dit_frames = 1 + (num_frames - 1) // 4
        latent_frames_per_block = 4 * BLOCK_DURATION
        
        start_latent = block_idx * latent_frames_per_block
        end_latent = min((block_idx + 1) * latent_frames_per_block, num_dit_frames)
        
        return start_latent, end_latent

    def get_block_video_frame_range(self, block_idx: int, num_frames: int) -> tuple:
        """
        Get the video frame range for a specific block.
        Returns (start_frame, end_frame) where end_frame is exclusive.
        """
        start_latent, end_latent = self.get_block_latent_range(block_idx, num_frames)
        
        # Convert latent frames to video frames
        # Latent frame 0 -> video frames 0-3
        # Latent frame n (n>0) -> video frames 1 + (n-1)*4 to 1 + (n-1)*4 + 3
        if start_latent == 0:
            start_frame = 0
        else:
            start_frame = 1 + (start_latent - 1) * 4
        
        if end_latent <= 1:
            end_frame = 1
        else:
            end_frame = min(1 + (end_latent - 1) * 4, num_frames)
        
        return start_frame, end_frame
    
    def encode_mllm_for_block(
        self,
        prompt: str,
        generated_video_frames: List[Image.Image],
        height: int,
        width: int,
        num_frames: int,
        current_block: int,
    ) -> dict:
        """
        Encode MLLM condition for the current block.
        
        For block 0: Uses text-only mode (no video context)
        For block N (N>0): Uses video from blocks 0 to N-1 as context
        
        Returns dict with mllm_hidden_states, mllm_mask, mllm_kv_len, mllm_position_ids
        """
        if current_block == 0 or len(generated_video_frames) == 0:
            # First block: text-only MLLM condition
            result = self.mllm_embedder.process(
                self, prompt, None, height, width, num_frames,
                use_mllm_condition=True, mode="text"
            )
        else:
            # Subsequent blocks: use accumulated video frames
            result = self.mllm_embedder.process(
                self, prompt, generated_video_frames, height, width, num_frames,
                use_mllm_condition=True, mode="full"
            )
        return result

    def denoise_block(
        self,
        block_idx: int,
        num_blocks: int,
        full_latents: torch.Tensor,
        dit: WanModel,
        context_posi: torch.Tensor,
        context_nega: torch.Tensor,
        mllm_embeddings: Optional[torch.Tensor],
        cfg_scale: float,
        height: int,
        width: int,
        num_frames: int,
        progress_bar_cmd=tqdm,
    ) -> torch.Tensor:
        """
        Denoise a single block of the video efficiently.
        
        Only processes the current block's latents, not the full sequence.
        """
        start_latent, end_latent = self.get_block_latent_range(block_idx, num_frames)
        
        block_latents = full_latents[:, :, start_latent:end_latent].clone()  # Tensor, shape (B, C, F_block, H, W).
        clean_latent_frame = None  # Optional[Tensor], shape (B, C, 1, H, W) when available.
        if start_latent > 0:  # bool, blocks after the first get clean condition.
            clean_latent_frame = full_latents[:, :, start_latent - 1:start_latent].clone()  # (B, C, 1, H, W).
        
        # Calculate spatial dimensions
        lat_h = height // 16
        lat_w = width // 16
        S_dit = lat_h * lat_w
        
        for progress_id, timestep in enumerate(progress_bar_cmd(
            self.scheduler.timesteps, 
            desc=f"Block {block_idx + 1}/{num_blocks}"
        )):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            
            # Forward pass for this block only (positive)
            noise_pred_posi = model_fn_block(
                dit=dit,
                block_latents=block_latents,
                timestep=timestep,
                context=context_posi,
                mllm_embeddings=mllm_embeddings,
                clean_latent_frame=clean_latent_frame,  # Optional[Tensor], (B, C, 1, H, W) clean condition.
                start_latent_frame=start_latent,
                height=height,
                width=width,
            )
            
            # Forward pass (negative) for CFG
            if cfg_scale != 1.0:
                noise_pred_nega = model_fn_block(
                    dit=dit,
                    block_latents=block_latents,
                    timestep=timestep,
                    context=context_nega,
                    mllm_embeddings=mllm_embeddings,  # No MLLM for negative
                    clean_latent_frame=clean_latent_frame,  # Optional[Tensor], (B, C, 1, H, W) clean condition.
                    start_latent_frame=start_latent,
                    height=height,
                    width=width,
                )
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi
            
            # Apply scheduler step
            block_latents = self.scheduler.step(
                noise_pred, self.scheduler.timesteps[progress_id], block_latents
            )
        
        # Update full latents with denoised block
        full_latents[:, :, start_latent:end_latent] = block_latents
        return full_latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: Optional[str] = "",
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames: int = 81,
        cfg_scale: Optional[float] = 5.0,
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple] = (30, 52),
        tile_stride: Optional[tuple] = (15, 26),
        progress_bar_cmd=tqdm,
    ):
        """
        Autoregressive video generation.
        
        Generates video blocks sequentially:
        1. Generate block 0 with text-only MLLM condition
        2. Decode block 0 and encode with MLLM
        3. Generate block 1 with block 0's MLLM condition
        4. Continue until all blocks are generated
        
        Args:
            prompt: Text prompt for video generation
            negative_prompt: Negative prompt for CFG
            seed: Random seed for reproducibility
            rand_device: Device for random number generation
            height: Video height (will be adjusted to valid dimensions)
            width: Video width (will be adjusted to valid dimensions)
            num_frames: Number of video frames
            cfg_scale: Classifier-free guidance scale
            num_inference_steps: Number of denoising steps per block
            sigma_shift: Scheduler sigma shift
            tiled: Use tiled VAE encoding/decoding
            tile_size: Tile size for VAE
            tile_stride: Tile stride for VAE
            progress_bar_cmd: Progress bar function
            
        Returns:
            List of PIL Images representing the generated video
        """
        # Validate and adjust dimensions
        height, width, num_frames = self.check_resize_height_width(height, width, num_frames)
        
        # Calculate block count
        num_blocks = self.calculate_num_blocks(num_frames, height, width)
        num_dit_frames = 1 + (num_frames - 1) // 4
        
        print(f"=== Autoregressive Video Generation ===")
        print(f"Video: {width}x{height}, {num_frames} frames ({num_dit_frames} latent frames)")
        print(f"Blocks: {num_blocks} (BLOCK_DURATION={BLOCK_DURATION})")
        
        # Initialize scheduler
        self.scheduler.set_timesteps(
            num_inference_steps, 
            denoising_strength=1.0, 
            shift=sigma_shift
        )
        
        # Initialize noise
        latent_height = height // self.vae.upsampling_factor
        latent_width = width // self.vae.upsampling_factor
        shape = (1, self.vae.model.z_dim, num_dit_frames, latent_height, latent_width)
        latents = self.generate_noise(shape, seed=seed, rand_device=rand_device)
        
        # Encode T5 text prompts (shared across all blocks)
        self.load_models_to_device(["text_encoder"])
        prompt_embedder = WanVideoUnit_PromptEmbedder()
        context_posi = prompt_embedder.encode_prompt(self, prompt)
        context_nega = prompt_embedder.encode_prompt(self, negative_prompt)
        
        # Embed text context once
        self.load_models_to_device(self.in_iteration_models)
        context_posi_emb = self.dit.text_embedding(context_posi)
        context_nega_emb = self.dit.text_embedding(context_nega)
        
        # Accumulated video frames for MLLM conditioning
        generated_video_frames: List[Image.Image] = []
        
        # Process each block sequentially
        for block_idx in range(num_blocks):
            start_latent, end_latent = self.get_block_latent_range(block_idx, num_frames)
            start_frame, end_frame = self.get_block_video_frame_range(block_idx, num_frames)
            
            print(f"\n--- Block {block_idx + 1}/{num_blocks} ---")
            print(f"  Latent frames: [{start_latent}, {end_latent})")
            print(f"  Video frames: [{start_frame}, {end_frame})")
            print(f"  MLLM context: {len(generated_video_frames)} frames from previous blocks")
            
            # Step 1: Encode MLLM condition for this block
            self.load_models_to_device(["mllm_encoder"])
            mllm_output = self.encode_mllm_for_block(
                prompt=prompt,
                generated_video_frames=generated_video_frames,
                height=height,
                width=width,
                num_frames=num_frames,
                current_block=block_idx,
            )
            
            # Compute MLLM embeddings if available
            mllm_embeddings = None
            mllm_mask = mllm_output["mllm_mask"]
            if _DEBUG_COMPARE_MASKS:
                hs = mllm_output.get("mllm_hidden_states")
                pid = mllm_output.get("mllm_position_ids")
                kv_len = mllm_output.get("mllm_kv_len")
                print(
                    f"[wan_video_ar] block={block_idx} mllm_hidden_states={None if hs is None else tuple(hs.shape)} "
                    f"mllm_position_ids={None if pid is None else tuple(pid.shape)} "
                    f"mllm_mask={None if mllm_mask is None else tuple(mllm_mask.shape)} mllm_kv_len={kv_len}"
                )
                if mllm_mask is not None:
                    vals = [int(mllm_mask[0, i].item()) for i in [0, min(1, mllm_mask.shape[1]-1), min(10, mllm_mask.shape[1]-1)]]
                    print(f"[wan_video_ar] block={block_idx} mllm_mask samples (q=0/1/10)={vals}")
            if (
                mllm_output.get("mllm_hidden_states") is not None
                and hasattr(self.dit, "has_mllm_input")
                and self.dit.has_mllm_input
            ):
                self.load_models_to_device(self.in_iteration_models)
                mllm_embeddings = self.dit.mllm_embedding(
                    mllm_output["mllm_hidden_states"],
                    position_ids=mllm_output["mllm_position_ids"],
                    mllm_mask=mllm_mask,
                )
                if _DEBUG_COMPARE_MASKS:
                    print(f"[wan_video_ar] block={block_idx} mllm_embeddings={tuple(mllm_embeddings.shape)}")
            
            # Step 2: Denoise this block (only this block's latents)
            self.load_models_to_device(self.in_iteration_models)
            latents = self.denoise_block(
                block_idx=block_idx,
                num_blocks=num_blocks,
                full_latents=latents,
                dit=self.dit,
                context_posi=context_posi_emb,
                context_nega=context_nega_emb,
                mllm_embeddings=mllm_embeddings,
                cfg_scale=cfg_scale,
                height=height,
                width=width,
                num_frames=num_frames,
                progress_bar_cmd=progress_bar_cmd,
            )
            
            # Step 3: Decode all latents up to current block for MLLM context
            # Note: VAE is causal, so we must decode from the beginning
            self.load_models_to_device(['vae'])
            
            # Decode all latents up to and including current block
            latents_to_decode = latents[:, :, :end_latent]
            all_video = self.vae.decode(
                latents_to_decode, 
                device=self.device, 
                tiled=tiled, 
                tile_size=tile_size, 
                tile_stride=tile_stride
            )
            all_video_frames = self.vae_output_to_video(all_video)
            
            # Update accumulated frames (replace with properly decoded frames)
            generated_video_frames = all_video_frames
            print(f"  Decoded {len(all_video_frames)} frames up to block {block_idx}")
        
        # Final full decode
        print(f"\n=== Final Decoding ===")
        self.load_models_to_device(['vae'])
        video = self.vae.decode(
            latents, 
            device=self.device, 
            tiled=tiled, 
            tile_size=tile_size, 
            tile_stride=tile_stride
        )
        video = self.vae_output_to_video(video)
        self.load_models_to_device([])
        
        print(f"Generation complete: {len(video)} frames")
        return video


def model_fn_block(
    dit: WanModel,
    block_latents: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    mllm_embeddings: Optional[torch.Tensor] = None,
    clean_latent_frame: Optional[torch.Tensor] = None,  # (B, C, 1, H, W) clean VAE latent frame.
    start_latent_frame: int = 0,
    height: int = 480,
    width: int = 832,
) -> torch.Tensor:
    """
    Forward pass for a single block's latents only.
    
    This is an optimized version that only processes the current block,
    avoiding redundant computation on the full sequence.
    
    Args:
        dit: The DiT model
        block_latents: Latents for this block only, shape (B, C, T_block, H, W)
        timestep: Current timestep
        context: Text embeddings (already processed through text_embedding)
        mllm_embeddings: MLLM embeddings (already processed through mllm_embedding)
        mllm_mask: MLLM attention mask for this block's tokens
        start_latent_frame: Starting latent frame index for RoPE positioning
        height: Video height
        width: Video width
        
    Returns:
        Noise prediction for this block, same shape as block_latents
    """
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))  # Tensor, shape (B, D), global timestep embed.
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))  # Tensor, shape (B, 6, D), global AdaLN params.
    
    x = block_latents
    x = dit.patchify(x, None)
    f, h, w = x.shape[2:]
    x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
    if _DEBUG_COMPARE_MASKS and (mllm_embeddings is not None):
        print(
            f"[wan_video_ar] model_fn_block start_latent_frame={start_latent_frame} f={int(f)} h={int(h)} w={int(w)} "
            f"Q_LEN={int(x.shape[1])} mllm_embeddings_len={int(mllm_embeddings.shape[1])}"
        )
    
    # Compute RoPE frequencies with correct position offset
    # The frequencies need to account for the block's position in the full sequence
    freqs = torch.cat([
        dit.freqs[0][start_latent_frame:start_latent_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

    keep_mask = None
    if start_latent_frame > 0:
        clean_tokens = dit.patchify(clean_latent_frame, None)
        clean_tokens = rearrange(clean_tokens, 'b c f h w -> b (f h w) c').contiguous()
        seg_len = int(clean_tokens.shape[1])
        
        clean_timestep = t.new_zeros((t.shape[0],))
        t_clean = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, clean_timestep))
        t_mod_clean = dit.time_projection(t_clean).unflatten(1, (6, dit.dim))
        t_noisy, t_mod_noisy = t, t_mod
        
        prev_frame = start_latent_frame - 1
        clean_freqs = torch.cat([
            dit.freqs[0][prev_frame:prev_frame + 1].view(1, 1, 1, -1).expand(1, h, w, -1),
            dit.freqs[1][:h].view(1, h, 1, -1).expand(1, h, w, -1),
            dit.freqs[2][:w].view(1, 1, w, -1).expand(1, h, w, -1)
        ], dim=-1).reshape(h * w, 1, -1).to(x.device)
        
        x = torch.cat([clean_tokens, x], dim=1)
        freqs = torch.cat([clean_freqs, freqs], dim=0)
        t = torch.cat([
            t_clean[:, None, :].expand(t_clean.shape[0], seg_len, t_clean.shape[-1]),
            t_noisy[:, None, :].expand(t_noisy.shape[0], f * h * w, t_noisy.shape[-1]),
        ], dim=1)
        t_mod = torch.cat([
            t_mod_clean[:, None, :, :].expand(t_mod_clean.shape[0], seg_len, t_mod_clean.shape[-2], t_mod_clean.shape[-1]),
            t_mod_noisy[:, None, :, :].expand(t_mod_noisy.shape[0], f * h * w, t_mod_noisy.shape[-2], t_mod_noisy.shape[-1]),
        ], dim=1)
        keep_mask = torch.cat([
            torch.zeros(clean_tokens.shape[1], dtype=torch.bool, device=x.device),
            torch.ones(f * h * w, dtype=torch.bool, device=x.device),
        ], dim=0)
    
    for block in dit.blocks:
        x = block(x, context, t_mod, freqs, mllm_embeddings=mllm_embeddings, mllm_mask=None,
                 mllm_block_mask=None, dit_block_mask=None)
    
    if keep_mask is not None:
        x = x[:, keep_mask]
        if t.dim() == 3:
            t = t[:, keep_mask]
    
    x = dit.head(x, t)
    x = dit.unpatchify(x, (f, h, w))
    
    return x
