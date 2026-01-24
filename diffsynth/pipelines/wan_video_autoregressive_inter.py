"""
Autoregressive inference for Wan Video Inter with MLLM condition.

This pipeline generates video blocks sequentially. For each block, it uses
the prompts and the previously generated video frames (or provided input_video)
as MLLM context, then denoises only the current block's latents.
"""

from typing import Optional, List, Union

import torch
from PIL import Image
from tqdm import tqdm

from .wan_video_inter import (
    WanVideoInterPipeline,
    WanVideoUnit_BlockScheduler,
    WanVideoUnit_PromptEmbedder,
    BLOCK_DURATION,
    compute_noise_pred_per_block,
    sample_frames_with_constraints,
)
from ..models.wan_video_dit import WanModel, sinusoidal_embedding_1d
from ..core import ModelConfig


_MLLM_TEMPLATE = (
    "<|im_start|>system\n"
    "Analyze the user's full video instruction and the provided partial video sequence. "
    "First, concisely describe the key elements, actions, and scene of the existing video segment. "
    "Then, predict the precise visual content for the next segment of video. "
    "The prediction must strictly follow the user's full instruction while ensuring seamless temporal "
    "continuity in motion, camera work, lighting, and object interactions with the existing frames. "
    "For the initial frame (when no video exists), use the instruction as the sole basis to generate "
    "the starting scene.<|im_end|>\n<|im_start|>user\n"
)
_MLLM_DROP_IDX = 111


class WanVideoAutoregressiveInterPipeline(WanVideoInterPipeline):
    """
    Autoregressive inference pipeline for Wan Video Inter.

    Each block is generated sequentially. The MLLM condition for a block
    is built from prompts and video frames up to the previous block.
    """

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16):
        super().__init__(device=device, torch_dtype=torch_dtype)

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
        """Load pretrained models for autoregressive inter inference."""
        parent_pipe = WanVideoInterPipeline.from_pretrained(
            torch_dtype=torch_dtype,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            mllm_processor_config=mllm_processor_config,
            redirect_common_files=redirect_common_files,
            use_usp=use_usp,
            vram_limit=vram_limit,
        )

        pipe = WanVideoAutoregressiveInterPipeline(device=device, torch_dtype=torch_dtype)
        for attr in ["tokenizer", "text_encoder", "dit", "vae", "mllm_encoder", "mllm_processor", "scheduler"]:
            if hasattr(parent_pipe, attr):
                setattr(pipe, attr, getattr(parent_pipe, attr))
        pipe.vram_management_enabled = parent_pipe.vram_management_enabled
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

    def _build_mllm_text(self, prompt_list: list[str], block_info: list[dict]) -> str:
        text_parts = []
        prev_prompt_idx = None
        last_prompt_idx = None
        for block in block_info:
            prompt_idx = block["prompt_idx"]
            if prompt_idx != prev_prompt_idx:
                if text_parts:
                    text_parts.append(" ")
                text_parts.append(prompt_list[prompt_idx])
                prev_prompt_idx = prompt_idx
            text_parts.append(" <|vision_start|><|video_pad|><|vision_end|>")
            last_prompt_idx = prompt_idx

        return _MLLM_TEMPLATE + "".join(text_parts)

    def _encode_mllm_with_video(
        self,
        prompt_list: list[str],
        block_info: list[dict],
        video_blocks: list[list[Image.Image]],
        video_metadata_blocks: list[dict],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        full_text = self._build_mllm_text(prompt_list, block_info)
        model_inputs = self.mllm_processor(
            text=[full_text],
            videos=video_blocks,
            padding=True,
            video_metadata=video_metadata_blocks,
            return_tensors="pt",
            do_resize=False,
            do_sample_frames=False,
        ).to(self.device)
        position_ids, _ = self.mllm_encoder.model.get_rope_index(
            input_ids=model_inputs["input_ids"],
            video_grid_thw=model_inputs["video_grid_thw"],
            attention_mask=model_inputs["attention_mask"],
        )
        hidden_states = self.mllm_encoder(**model_inputs)[-1]
        hidden_states = hidden_states[:, _MLLM_DROP_IDX:]
        input_ids = model_inputs["input_ids"][:, _MLLM_DROP_IDX:]
        position_ids = position_ids[..., _MLLM_DROP_IDX:]
        return hidden_states, input_ids, position_ids

    def _build_mllm_mask(
        self,
        num_frames: int,
        height: int,
        width: int,
        input_ids: torch.Tensor,
        block_info: list[dict],
        sampled_counts: list[int],
    ) -> tuple[torch.Tensor, int, torch.Tensor]:
        lat_h = height // 16
        lat_w = width // 16
        s_dit = lat_h * lat_w
        num_dit_frames = 1 + (num_frames - 1) // 4
        num_dit_tokens = num_dit_frames * s_dit
        mllm_seq_len = input_ids.shape[1]

        vision_start_token_id = self.mllm_encoder.config.vision_start_token_id
        vision_end_token_id = self.mllm_encoder.config.vision_end_token_id
        vision_start_positions = (input_ids[0] == vision_start_token_id).nonzero(as_tuple=True)[0]
        vision_end_positions = (input_ids[0] == vision_end_token_id).nonzero(as_tuple=True)[0]

        mllm_mask = torch.full(
            (1, num_dit_tokens),
            0,
            device=input_ids.device,
            dtype=torch.int32,
        )
        mllm_vision_ranges = torch.zeros(
            (1, num_dit_tokens, 2),
            device=input_ids.device,
            dtype=torch.int32,
        )

        text_end = mllm_seq_len
        if len(vision_start_positions) > 0:
            text_end = vision_start_positions[0].item()

        vision_seg_offset = 0
        for block_idx, block in enumerate(block_info):
            num_sampled = sampled_counts[block_idx] if block_idx < len(sampled_counts) else 0
            num_vision_segs = num_sampled // 2

            latent_start = block["latent_start"]
            latent_end = block["latent_end"]
            start_dit_token = latent_start * s_dit
            end_dit_token = latent_end * s_dit

            if block_idx == 0:
                mllm_mask[:, start_dit_token:end_dit_token] = 0
            elif (
                len(vision_end_positions) == 0
                or len(vision_start_positions) == 0
                or vision_seg_offset <= 0
            ):
                mllm_mask[:, start_dit_token:end_dit_token] = 0
            else:
                end_pos_idx = min(vision_seg_offset - 1, len(vision_end_positions) - 1)
                end_mllm_prefix = vision_end_positions[end_pos_idx].item() + 1
                prefix_end = min(end_mllm_prefix, mllm_seq_len)
                mllm_mask[:, start_dit_token:end_dit_token] = prefix_end

                vision_range_start = vision_start_positions[0].item()
                vision_range_end = vision_end_positions[end_pos_idx].item() + 1
                mllm_vision_ranges[:, start_dit_token:end_dit_token, 0] = vision_range_start
                mllm_vision_ranges[:, start_dit_token:end_dit_token, 1] = vision_range_end

            vision_seg_offset += num_vision_segs

        return mllm_mask, mllm_seq_len, mllm_vision_ranges

    def encode_mllm_for_block(
        self,
        prompt_list: list[str],
        block_info: list[dict],
        generated_video_frames: list[Image.Image],
        height: int,
        width: int,
        num_frames: int,
        current_block: int,
    ) -> dict:
        if current_block == 0 or not generated_video_frames:
            return {}

        prev_blocks = [b for b in block_info if b["global_block_idx"] < current_block]
        if not prev_blocks:
            return {}

        max_prev_prompt_idx = max(b["prompt_idx"] for b in prev_blocks)
        prompt_slice = prompt_list[:max_prev_prompt_idx + 1]
        video_blocks, video_metadata_blocks, sampled_counts = self._collect_video_blocks(
            generated_video_frames, prev_blocks, total_num_frames=num_frames
        )
        hidden_states, input_ids, position_ids = self._encode_mllm_with_video(
            prompt_slice, prev_blocks, video_blocks, video_metadata_blocks
        )
        current_block_info = next(
            block for block in block_info if block["global_block_idx"] == current_block
        )
        mask_block_info = prev_blocks + [current_block_info]
        sampled_counts = sampled_counts + [0]

        mllm_mask, mllm_kv_len, mllm_vision_ranges = self._build_mllm_mask(
            num_frames=num_frames,
            height=height,
            width=width,
            input_ids=input_ids,
            block_info=mask_block_info,
            sampled_counts=sampled_counts,
        )
        return {
            "mllm_hidden_states": hidden_states,
            "mllm_mask": mllm_mask,
            "mllm_kv_len": mllm_kv_len,
            "mllm_position_ids": position_ids,
            "mllm_vision_ranges": mllm_vision_ranges,
        }

    def denoise_block(
        self,
        block: dict,
        num_blocks: int,
        full_latents: torch.Tensor,
        dit: WanModel,
        context_posi: torch.Tensor,
        context_nega: torch.Tensor,
        mllm_embeddings: Optional[torch.Tensor],
        mllm_mask: Optional[torch.Tensor],
        mllm_kv_len: Optional[int],
        mllm_vision_ranges: Optional[torch.Tensor],
        freqs_full: torch.Tensor,
        tokens_per_latent_frame: int,
        use_gradient_checkpointing: bool,
        cfg_scale: float,
        clean_latents_source: Optional[torch.Tensor] = None,
        progress_bar_cmd=tqdm,
    ) -> torch.Tensor:
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

            # Use clean_latents_source if provided, otherwise use full_latents
            clean_source = clean_latents_source if clean_latents_source is not None else full_latents

            noise_pred_posi = compute_noise_pred_per_block(
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
                mllm_mask_full=mllm_mask,
                mllm_vision_ranges=mllm_vision_ranges,
                mllm_kv_len=mllm_kv_len,
                tokens_per_latent_frame=tokens_per_latent_frame,
                use_gradient_checkpointing=use_gradient_checkpointing,
                device=self.device,
            )

            if cfg_scale != 1.0:
                noise_pred_nega = compute_noise_pred_per_block(
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
                    mllm_mask_full=mllm_mask,
                    mllm_vision_ranges=mllm_vision_ranges,
                    mllm_kv_len=mllm_kv_len,
                    tokens_per_latent_frame=tokens_per_latent_frame,
                    use_gradient_checkpointing=use_gradient_checkpointing,
                    device=self.device,
                )
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
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
        cfg_scale: Optional[float] = 5.0,
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

        print("=== Autoregressive Inter Video Generation ===")
        print(f"Video: {width}x{height}, {num_frames} frames ({num_dit_frames} latent frames)")
        print(f"Blocks: {num_blocks} (BLOCK_DURATION={BLOCK_DURATION})")

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
            input_video_tensor = self.vae.video_to_vae_input(input_video)
            input_video_latents = self.vae.encode(
                input_video_tensor,
                device=self.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
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
            mllm_mask = None
            mllm_kv_len = None
            mllm_vision_ranges = None

            if use_mllm_condition and block_idx > 0:
                self.load_models_to_device(["mllm_encoder"])
                mllm_output = self.encode_mllm_for_block(
                    prompt_list=prompt_list,
                    block_info=block_info,
                    generated_video_frames=generated_video_frames,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    current_block=block_idx,
                )

                if (
                    mllm_output.get("mllm_hidden_states") is not None
                    and hasattr(self.dit, "has_mllm_input")
                    and self.dit.has_mllm_input
                ):
                    self.load_models_to_device(self.in_iteration_models)
                    mllm_mask = mllm_output["mllm_mask"]
                    mllm_kv_len = mllm_output["mllm_kv_len"]
                    mllm_vision_ranges = mllm_output.get("mllm_vision_ranges")
                    mllm_embeddings = self.dit.mllm_embedding(
                        mllm_output["mllm_hidden_states"],
                        position_ids=mllm_output["mllm_position_ids"],
                        mllm_mask=mllm_mask,
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
                mllm_mask=mllm_mask,
                mllm_kv_len=mllm_kv_len,
                mllm_vision_ranges=mllm_vision_ranges,
                freqs_full=freqs_full,
                tokens_per_latent_frame=tokens_per_latent_frame,
                use_gradient_checkpointing=use_gradient_checkpointing,
                cfg_scale=cfg_scale,
                clean_latents_source=input_video_latents,
                progress_bar_cmd=progress_bar_cmd,
            )

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
