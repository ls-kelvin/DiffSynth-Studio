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

Key insight: Training uses dit_block_mask to isolate blocks, but during inference
we generate each block independently with only the relevant MLLM conditions.
This ensures temporal consistency while enabling autoregressive generation.
"""

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
    model_fn_wan_video,
)
from ..models.wan_video_dit import sinusoidal_embedding_1d
from ..core import ModelConfig


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
        models: dict,
        inputs_shared: dict,
        inputs_posi: dict,
        inputs_nega: dict,
        cfg_scale: float,
        height: int,
        width: int,
        num_frames: int,
        progress_bar_cmd=tqdm,
    ) -> torch.Tensor:
        """
        Denoise a single block of the video.
        
        Uses full model forward pass but only updates the current block's latents.
        The dit_block_mask ensures the DiT only attends within the block.
        """
        start_latent, end_latent = self.get_block_latent_range(block_idx, num_frames)
        
        # Clone full latents to preserve other blocks
        current_latents = full_latents.clone()
        
        for progress_id, timestep in enumerate(progress_bar_cmd(
            self.scheduler.timesteps, 
            desc=f"Block {block_idx + 1}/{num_blocks}"
        )):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            
            # Update inputs with current latents
            inputs_shared["latents"] = current_latents
            
            # Forward pass (positive)
            noise_pred_posi = model_fn_wan_video(
                **models, **inputs_shared, **inputs_posi, 
                timestep=timestep,
            )
            
            # Forward pass (negative) for CFG
            if cfg_scale != 1.0:
                noise_pred_nega = model_fn_wan_video(
                    **models, **inputs_shared, **inputs_nega,
                    timestep=timestep,
                )
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi
            
            # Apply scheduler step
            new_latents = self.scheduler.step(
                noise_pred, self.scheduler.timesteps[progress_id], current_latents
            )
            
            # Only update current block's latents, keep others unchanged
            current_latents[:, :, start_latent:end_latent] = new_latents[:, :, start_latent:end_latent]
        
        return current_latents

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
        
        # Prepare DiT models
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        
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
            
            # Prepare inputs
            inputs_shared = {
                "latents": latents,
                "height": height,
                "width": width,
                "num_frames": num_frames,
                "cfg_scale": cfg_scale,
                "cfg_merge": False,
                **mllm_output,
            }
            inputs_posi = {"context": context_posi}
            inputs_nega = {"context": context_nega}
            
            # Step 2: Denoise this block
            self.load_models_to_device(self.in_iteration_models)
            latents = self.denoise_block(
                block_idx=block_idx,
                num_blocks=num_blocks,
                full_latents=latents,
                models=models,
                inputs_shared=inputs_shared,
                inputs_posi=inputs_posi,
                inputs_nega=inputs_nega,
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
