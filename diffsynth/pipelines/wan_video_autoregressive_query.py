"""
Autoregressive inference for Wan Video with MetaQuery MLLM condition.

This pipeline generates video blocks sequentially using fixed-length MetaQuery
embeddings as MLLM condition, instead of variable-length history-based KV.
"""

from typing import Optional, List, Union

import torch
from PIL import Image
from tqdm import tqdm

from .wan_video_inter_3 import (
    WanVideoInterPipeline_MetaQuery,
    WanVideoUnit_BlockScheduler,
    WanVideoUnit_PromptEmbedder,
    WanVideoUnit_MLLMEmbedder_MetaQuery,
    BLOCK_DURATION,
    compute_noise_pred_per_block_metaquery,
    sample_frames_with_constraints,
)
from ..models.wan_video_dit import WanModel, sinusoidal_embedding_1d
from ..core import ModelConfig


class WanVideoAutoregressiveQueryPipeline(WanVideoInterPipeline_MetaQuery):
    """
    Autoregressive inference pipeline using MetaQuery MLLM conditioning.

    Each block is generated sequentially. The MLLM condition for each block
    uses fixed-length query embeddings that encode prompts and video frames
    visible up to that block.
    """

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.last_block_videos = []

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
        num_metaqueries: int = 64,
    ):
        """Load pretrained models for autoregressive MetaQuery inference."""
        parent_pipe = WanVideoInterPipeline_MetaQuery.from_pretrained(
            torch_dtype=torch_dtype,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            mllm_processor_config=mllm_processor_config,
            redirect_common_files=redirect_common_files,
            use_usp=use_usp,
            vram_limit=vram_limit,
        )

        pipe = WanVideoAutoregressiveQueryPipeline(device=device, torch_dtype=torch_dtype)
        for attr in ["tokenizer", "text_encoder", "dit", "vae", "mllm_encoder", "mllm_processor", "scheduler"]:
            if hasattr(parent_pipe, attr):
                setattr(pipe, attr, getattr(parent_pipe, attr))
        pipe.vram_management_enabled = parent_pipe.vram_management_enabled
        
        # Initialize metaquery tokens if not already done
        if hasattr(pipe, 'mllm_encoder') and pipe.mllm_encoder is not None:
            if pipe.mllm_encoder.num_metaqueries == 0:
                pipe.mllm_encoder.num_metaqueries = num_metaqueries
                pipe.mllm_encoder._init_metaquery_tokens()
        
        return pipe

    def _build_block_info(self, prompt_list: list[str], clip_frames: list[int], num_frames: int) -> list[dict]:
        block_scheduler = WanVideoUnit_BlockScheduler()
        return block_scheduler.process(self, prompt_list, clip_frames, num_frames)["block_info"]

    def _collect_video_blocks(
        self,
        video_frames: list[Image.Image],
        block_info: list[dict],
        total_num_frames: int,
    ) -> tuple[list, list, list[int]]:
        video_blocks = []
        video_metadata_blocks = []
        sampled_counts = []

        for block in block_info:
            block_frames = video_frames[block["start_frame"]:block["end_frame"]]
            total_frames_in_block = len(block_frames)
            local_indices = sample_frames_with_constraints(total_frames_in_block, target_stride=8)
            sampled_counts.append(len(local_indices))
            global_indices = [block["start_frame"] + i for i in local_indices]
            sampled_frames = [video_frames[i] for i in global_indices] if global_indices else []
            metadata = {
                "fps": 16,
                "frames_indices": global_indices,
                "total_num_frames": total_num_frames,
            }
            video_blocks.append(sampled_frames)
            video_metadata_blocks.append(metadata)

        return video_blocks, video_metadata_blocks, sampled_counts

    def encode_mllm_for_block_metaquery(
        self,
        prompt_list: list[str],
        block_info: list[dict],
        generated_video_frames: list[Image.Image],
        current_block: int,
    ) -> Optional[torch.Tensor]:
        """Encode MLLM condition for a single block using MetaQuery.
        
        Returns:
            query_embeds: (1, num_metaqueries, hidden_dim) or None
        """
        mllm_embedder = WanVideoUnit_MLLMEmbedder_MetaQuery()
        
        # Collect video blocks for history (only previous blocks)
        if current_block > 0:
            video_blocks, video_metadata_blocks, _ = self._collect_video_blocks(
                generated_video_frames, block_info[:current_block], 
                total_num_frames=len(generated_video_frames)
            )
        else:
            video_blocks = []
            video_metadata_blocks = []
        
        # Build query tokens string
        query_tokens = ["<|begin_of_query|>"]
        query_tokens.extend([f"<|query_{i}|>" for i in range(self.mllm_encoder.num_metaqueries)])
        query_tokens.append("<|end_of_query|>")
        query_str = "".join(query_tokens)
        
        # Build full text with history
        full_text_parts = [f"<|im_start|>system\n{mllm_embedder.system_prompt}<|im_end|>\n"]
        all_videos = []
        all_metadata = []
        
        prev_prompt_idx = None
        for block_idx in range(current_block + 1):
            block = block_info[block_idx]
            user_content_parts = []
            prompt_idx = block["prompt_idx"]
            
            # Add prompt if changed
            if prompt_idx != prev_prompt_idx:
                user_content_parts.append(prompt_list[prompt_idx])
                prev_prompt_idx = prompt_idx
            
            # Add video for previous blocks only
            if block_idx > 0:
                if len(user_content_parts) > 0:
                    user_content_parts.append(" ")
                user_content_parts.append(" <|vision_start|><|video_pad|><|vision_end|>")
                all_videos.append(video_blocks[block_idx - 1])
                all_metadata.append(video_metadata_blocks[block_idx - 1])
            
            user_content = "".join(user_content_parts)
            full_text_parts.append(f"<|im_start|>user\n{user_content}<|im_end|>\n")
            full_text_parts.append(f"<|im_start|>assistant\n{query_str}<|im_end|>\n")
        
        full_text = "".join(full_text_parts)
        
        # Process with MLLM
        if all_videos:
            model_inputs = self.mllm_processor(
                text=[full_text],
                videos=all_videos,
                padding=True,
                video_metadata=all_metadata,
                return_tensors="pt",
                do_resize=False,
                do_sample_frames=False
            ).to(self.device)
        else:
            model_inputs = self.mllm_processor(
                text=[full_text],
                videos=None,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
        
        # Get hidden states
        hidden_states = self.mllm_encoder(**model_inputs)[-1]
        input_ids = model_inputs["input_ids"]
        
        # Extract query embeddings for current block
        boq = self.mllm_encoder.boq_token_id
        eoq = self.mllm_encoder.eoq_token_id
        if boq is None or eoq is None:
            raise RuntimeError("MetaQuery tokens are not initialized.")
        
        boq_positions = (input_ids[0] == boq).nonzero(as_tuple=True)[0].tolist()
        eoq_positions = (input_ids[0] == eoq).nonzero(as_tuple=True)[0].tolist()
        
        if len(boq_positions) != current_block + 1 or len(eoq_positions) != current_block + 1:
            raise RuntimeError(
                f"Expected {current_block + 1} MetaQuery segments, got {len(boq_positions)} BOQ and {len(eoq_positions)} EOQ."
            )
        
        # Get the query embeddings for the current block (last one)
        b_pos = boq_positions[current_block]
        e_pos = eoq_positions[current_block]
        if e_pos <= b_pos:
            raise RuntimeError("Invalid MetaQuery token positions.")
        
        query_embeds = hidden_states[:, b_pos + 1:e_pos, :]
        
        return query_embeds

    def denoise_block(
        self,
        block: dict,
        num_blocks: int,
        full_latents: torch.Tensor,
        dit: WanModel,
        context_posi: torch.Tensor,
        context_nega: torch.Tensor,
        mllm_embeddings: Optional[torch.Tensor],
        freqs_full: torch.Tensor,
        tokens_per_latent_frame: int,
        use_gradient_checkpointing: bool,
        cfg_scale: float,
        mllm_cfg_scale: float = 1.0,
        clean_latents_source: Optional[torch.Tensor] = None,
        progress_bar_cmd=tqdm,
    ) -> torch.Tensor:
        """Denoise a single block using MetaQuery MLLM embeddings."""
        latent_start = block["latent_start"]
        latent_end = block["latent_end"]
        block_latents = full_latents[:, :, latent_start:latent_end].clone()

        for progress_id, timestep in enumerate(progress_bar_cmd(
            self.scheduler.timesteps,
            desc=f"Block {block['global_block_idx'] + 1}/{num_blocks}",
        )):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

            full_latents[:, :, latent_start:latent_end] = block_latents
            t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
            t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))

            clean_timestep = t.new_zeros((t.shape[0],))
            t_clean = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, clean_timestep))
            t_mod_clean = dit.time_projection(t_clean).unflatten(1, (6, dit.dim))

            clean_source = clean_latents_source if clean_latents_source is not None else full_latents

            noise_pred_posi = compute_noise_pred_per_block_metaquery(
                dit=dit,
                block_idx=block["global_block_idx"],
                block_info=block,
                x_full=full_latents,
                input_latents=None,
                clean_input_latents=clean_source,
                freqs_full=freqs_full,
                context_per_block={block["global_block_idx"]: context_posi},
                t=t,
                t_mod=t_mod,
                t_clean=t_clean,
                t_mod_clean=t_mod_clean,
                mllm_embeddings=mllm_embeddings,
                tokens_per_latent_frame=tokens_per_latent_frame,
                use_gradient_checkpointing=use_gradient_checkpointing,
                device=self.device,
                timestep_value=timestep,
            )

            w_t = cfg_scale - 1.0
            w_m = mllm_cfg_scale - 1.0

            if w_t != 0.0 or w_m != 0.0:
                noise_pred_nega = noise_pred_posi
                if w_t != 0.0:
                    noise_pred_nega = compute_noise_pred_per_block_metaquery(
                        dit=dit,
                        block_idx=block["global_block_idx"],
                        block_info=block,
                        x_full=full_latents,
                        input_latents=None,
                        clean_input_latents=clean_source,
                        freqs_full=freqs_full,
                        context_per_block={block["global_block_idx"]: context_nega},
                        t=t,
                        t_mod=t_mod,
                        t_clean=t_clean,
                        t_mod_clean=t_mod_clean,
                        mllm_embeddings=mllm_embeddings,
                        tokens_per_latent_frame=tokens_per_latent_frame,
                        use_gradient_checkpointing=use_gradient_checkpointing,
                        device=self.device,
                        timestep_value=timestep,
                    )

                noise_pred_text_only = noise_pred_posi
                if w_m != 0.0:
                    noise_pred_text_only = compute_noise_pred_per_block_metaquery(
                        dit=dit,
                        block_idx=block["global_block_idx"],
                        block_info=block,
                        x_full=full_latents,
                        input_latents=None,
                        clean_input_latents=clean_source,
                        freqs_full=freqs_full,
                        context_per_block={block["global_block_idx"]: context_posi},
                        t=t,
                        t_mod=t_mod,
                        t_clean=t_clean,
                        t_mod_clean=t_mod_clean,
                        mllm_embeddings=mllm_embeddings,
                        tokens_per_latent_frame=tokens_per_latent_frame,
                        use_gradient_checkpointing=use_gradient_checkpointing,
                        device=self.device,
                        timestep_value=timestep,
                        mllm_cfg_drop=1.0,
                    )

                noise_pred = (1.0 + w_t + w_m) * noise_pred_posi - w_t * noise_pred_nega - w_m * noise_pred_text_only
            else:
                noise_pred = noise_pred_posi

            block_latents = self.scheduler.step(
                noise_pred,
                self.scheduler.timesteps[progress_id],
                block_latents,
            )

        full_latents[:, :, latent_start:latent_end] = block_latents
        return full_latents

    @torch.no_grad()
    def __call__(
        self,
        prompt_list: list[str],
        clip_frames: list[int],
        negative_prompt_list: Optional[list[str]] = None,
        input_video: Optional[list[Image.Image]] = None,
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames: int = 81,
        use_mllm_condition: Optional[bool] = True,
        use_gt_mllm: bool = False,
        use_gt_vae: bool = False,
        gt_decode: bool = True,
        cfg_scale: Optional[float] = 5.0,
        mllm_cfg_scale: Optional[float] = 1.0,
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple[int, int]] = (30, 52),
        tile_stride: Optional[tuple[int, int]] = (15, 26),
        use_gradient_checkpointing: bool = False,
        progress_bar_cmd=tqdm,
    ):
        if len(prompt_list) != len(clip_frames):
            raise ValueError(
                f"prompt_list length ({len(prompt_list)}) must equal clip_frames length ({len(clip_frames)})."
            )
        if sum(clip_frames) != num_frames:
            raise ValueError(
                f"Sum of clip_frames ({sum(clip_frames)}) must equal num_frames ({num_frames})."
            )
        if negative_prompt_list is None:
            negative_prompt_list = [""] * len(prompt_list)
        if len(negative_prompt_list) != len(prompt_list):
            raise ValueError(
                f"negative_prompt_list length ({len(negative_prompt_list)}) must equal prompt_list length ({len(prompt_list)})."
            )

        height, width, num_frames = self.check_resize_height_width(height, width, num_frames)
        block_info = self._build_block_info(prompt_list, clip_frames, num_frames)
        num_blocks = len(block_info)
        num_dit_frames = 1 + (num_frames - 1) // 4
        self.last_block_videos = []

        print("=== Autoregressive MetaQuery Video Generation ===")
        print(f"Video: {width}x{height}, {num_frames} frames ({num_dit_frames} latent frames)")
        print(f"Blocks: {num_blocks} (BLOCK_DURATION={BLOCK_DURATION})")
        print(f"MetaQuery tokens: {self.mllm_encoder.num_metaqueries}")

        self.scheduler.set_timesteps(num_inference_steps, shift=sigma_shift)

        latent_height = height // self.vae.upsampling_factor
        latent_width = width // self.vae.upsampling_factor
        shape = (1, self.vae.model.z_dim, num_dit_frames, latent_height, latent_width)
        latents = self.generate_noise(shape, seed=seed, rand_device=rand_device)

        self.load_models_to_device(["text_encoder"])
        prompt_embedder = WanVideoUnit_PromptEmbedder()
        prompt_embeddings_map = prompt_embedder.process(self, prompt_list, block_info)["prompt_embeddings_map"]
        negative_embeddings_map = prompt_embedder.process(self, negative_prompt_list, block_info)["prompt_embeddings_map"]

        self.load_models_to_device(self.in_iteration_models)
        context_per_prompt = {idx: self.dit.text_embedding(emb) for idx, emb in prompt_embeddings_map.items()}
        context_per_prompt_nega = {idx: self.dit.text_embedding(emb) for idx, emb in negative_embeddings_map.items()}

        lat_h = height // 16
        lat_w = width // 16
        tokens_per_latent_frame = lat_h * lat_w
        freqs_full = torch.cat([
            self.dit.freqs[0][:num_dit_frames].view(num_dit_frames, 1, 1, -1).expand(num_dit_frames, lat_h, lat_w, -1),
            self.dit.freqs[1][:lat_h].view(1, lat_h, 1, -1).expand(num_dit_frames, lat_h, lat_w, -1),
            self.dit.freqs[2][:lat_w].view(1, 1, lat_w, -1).expand(num_dit_frames, lat_h, lat_w, -1),
        ], dim=-1).reshape(num_dit_frames * lat_h * lat_w, 1, -1).to(self.device)

        generated_video_frames: List[Image.Image] = []

        # Encode input_video to latents if requested
        input_video_latents = None
        if use_gt_vae and input_video is not None:
            self.load_models_to_device(["vae"])
            input_video_tensor = self.preprocess_video(input_video)
            input_video_latents = self.vae.encode(
                input_video_tensor,
                device=self.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            ).to(dtype=self.torch_dtype, device=self.device)
            print(f"Encoded input_video to latents: {input_video_latents.shape}")

        for block in block_info:
            block_idx = block["global_block_idx"]
            start_frame = block["start_frame"]
            end_frame = block["end_frame"]
            latent_start = block["latent_start"]
            latent_end = block["latent_end"]

            print(f"\n--- Block {block_idx + 1}/{num_blocks} ---")
            print(f"  Latent frames: [{latent_start}, {latent_end})")
            print(f"  Video frames: [{start_frame}, {end_frame})")
            print(f"  MLLM context: {len(generated_video_frames)} frames from previous blocks")

            mllm_embeddings = None

            if use_mllm_condition:
                self.load_models_to_device(["mllm_encoder"])
                query_embeds = self.encode_mllm_for_block_metaquery(
                    prompt_list=prompt_list,
                    block_info=block_info,
                    generated_video_frames=generated_video_frames,
                    current_block=block_idx,
                )

                if query_embeds is not None and hasattr(self.dit, "has_mllm_input") and self.dit.has_mllm_input:
                    self.load_models_to_device(self.in_iteration_models)
                    # Process through DiT's mllm_embedding (bidirectional attention)
                    mllm_embeddings = self.dit.mllm_embedding(
                        query_embeds,
                        position_ids=None,
                        mllm_mask=None,
                    )

            self.load_models_to_device(self.in_iteration_models)
            context_posi = context_per_prompt[block["prompt_idx"]]
            context_nega = context_per_prompt_nega[block["prompt_idx"]]
            latents = self.denoise_block(
                block=block,
                num_blocks=num_blocks,
                full_latents=latents,
                dit=self.dit,
                context_posi=context_posi,
                context_nega=context_nega,
                mllm_embeddings=mllm_embeddings,
                freqs_full=freqs_full,
                tokens_per_latent_frame=tokens_per_latent_frame,
                use_gradient_checkpointing=use_gradient_checkpointing,
                cfg_scale=cfg_scale,
                mllm_cfg_scale=mllm_cfg_scale,
                clean_latents_source=input_video_latents,
                progress_bar_cmd=progress_bar_cmd,
            )

            if gt_decode and input_video_latents is not None:
                gt_prefix = input_video_latents[:, :, :latent_start]
                pred_suffix = latents[:, :, latent_start:latent_end]
                latents_to_decode = torch.cat([gt_prefix, pred_suffix], dim=2)

                self.load_models_to_device(["vae"])
                merged_video = self.vae.decode(
                    latents_to_decode,
                    device=self.device,
                    tiled=tiled,
                    tile_size=tile_size,
                    tile_stride=tile_stride,
                )
                merged_video_frames = self.vae_output_to_video(merged_video)
                self.last_block_videos.append({
                    "block_idx": block_idx,
                    "prompt_idx": block["prompt_idx"],
                    "end_frame": end_frame,
                    "frames": merged_video_frames,
                })
                print(f"  Saved block-switch video for block {block_idx}")

            if use_mllm_condition:
                if use_gt_mllm and input_video is not None:
                    generated_video_frames = input_video[:end_frame]
                    print(f"  Using input_video frames up to block {block_idx}")
                else:
                    self.load_models_to_device(["vae"])
                    latents_to_decode = latents[:, :, :latent_end]
                    all_video = self.vae.decode(
                        latents_to_decode,
                        device=self.device,
                        tiled=tiled,
                        tile_size=tile_size,
                        tile_stride=tile_stride,
                    )
                    all_video_frames = self.vae_output_to_video(all_video)
                    generated_video_frames = all_video_frames
                    print(f"  Decoded {len(all_video_frames)} frames up to block {block_idx}")

        print("\n=== Final Decoding ===")
        self.load_models_to_device(["vae"])
        video = self.vae.decode(
            latents,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        video = self.vae_output_to_video(video)
        self.load_models_to_device([])
        print(f"Generation complete: {len(video)} frames")
        return video
