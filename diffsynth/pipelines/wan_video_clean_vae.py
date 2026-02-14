import torch, types, math, os
from PIL import Image
from typing import Optional, Union
from einops import rearrange
from tqdm import tqdm

from ..diffusion import FlowMatchScheduler
from ..core import ModelConfig
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit

from ..models.wan_video_dit import WanModel, sinusoidal_embedding_1d
from ..models.wan_video_text_encoder import WanTextEncoder, HuggingfaceTokenizer
from ..models.wan_video_vae import WanVideoVAE
from ..models.wan_video_mllm_encoder import WanMLLMEncoder
from ..models.wan_video_mllm_encoder import Qwen3VLProcessor

try:
    from torch.nn.attention.flex_attention import create_block_mask
    FLEX_ATTENTION_AVAILABLE = os.environ.get("DISABLE_FLEX_ATTENTION", "0") != "1"
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    create_block_mask = None

BLOCK_DURATION = 5
CLEAN_FRAME_COUNT = int(os.getenv("CLEAN_FRAME_COUNT", "2"))
LATENT_STRIDE = int(os.getenv("LATENT_STRIDE", "4"))


class WanVideoCleanVAEPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16, time_division_factor=4, time_division_remainder=1
        )
        self.scheduler = FlowMatchScheduler("Wan")
        self.tokenizer: HuggingfaceTokenizer = None
        self.text_encoder: WanTextEncoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.mllm_encoder: WanMLLMEncoder = None
        self.mllm_processor: Qwen3VLProcessor = None
        self.in_iteration_models = ("dit",)
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_CleanVAEBlockScheduler(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_PromptEmbedder(),
        ]
        self.post_units = []
        self.model_fn = model_fn_wan_video_clean_vae


    def enable_usp(self):
        from ..utils.xfuser import get_sequence_parallel_world_size, usp_attn_forward, usp_dit_forward

        for block in self.dit.blocks:
            block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
        self.dit.forward = types.MethodType(usp_dit_forward, self.dit)
        self.sp_size = get_sequence_parallel_world_size()
        self.use_unified_sequence_parallel = True


    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
        mllm_processor_config: ModelConfig = None,
        redirect_common_files: bool = False,
        use_usp: bool = False,
        vram_limit: float = None,
    ):
        # Redirect model path
        if redirect_common_files:
            redirect_dict = {
                "models_t5_umt5-xxl-enc-bf16.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "models_t5_umt5-xxl-enc-bf16.safetensors"),
                "Wan2.1_VAE.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "Wan2.1_VAE.safetensors"),
                "Wan2.2_VAE.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "Wan2.2_VAE.safetensors"),
            }
            for model_config in model_configs:
                if model_config.origin_file_pattern is None or model_config.model_id is None:
                    continue
                if model_config.origin_file_pattern in redirect_dict and model_config.model_id != redirect_dict[model_config.origin_file_pattern][0]:
                    print(f"To avoid repeatedly downloading model files, ({model_config.model_id}, {model_config.origin_file_pattern}) is redirected to {redirect_dict[model_config.origin_file_pattern]}. You can use `redirect_common_files=False` to disable file redirection.")
                    model_config.model_id = redirect_dict[model_config.origin_file_pattern][0]
                    model_config.origin_file_pattern = redirect_dict[model_config.origin_file_pattern][1]
        
        # Initialize pipeline
        pipe = WanVideoCleanVAEPipeline(device=device, torch_dtype=torch_dtype)
        if use_usp:
            from ..utils.xfuser import initialize_usp
            initialize_usp()
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)
        
        # Fetch models
        pipe.text_encoder = model_pool.fetch_model("wan_video_text_encoder")
        pipe.dit = model_pool.fetch_model("wan_video_dit", index=2)
        pipe.vae = model_pool.fetch_model("wan_video_vae")
        pipe.mllm_encoder = model_pool.fetch_model("wan_mllm_encoder")

        # Size division factor
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        # Initialize tokenizer and processor
        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            pipe.tokenizer = HuggingfaceTokenizer(name=tokenizer_config.path, seq_len=512, clean='whitespace')
        if mllm_processor_config is not None:
            mllm_processor_config.download_if_necessary()
            pipe.mllm_processor = Qwen3VLProcessor.from_pretrained(mllm_processor_config.path)
        
        # Unified Sequence Parallel
        if use_usp:
            pipe.enable_usp()
        
        # VRAM Management
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe


    @torch.no_grad()
    def __call__(
        self,
        prompt_list: list[str],
        clip_frames: list[int],
        input_video: list[Image.Image],
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames: int = 81,
        use_mllm_condition: Optional[bool] = True,
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
        if input_video is None:
            raise ValueError("input_video is required for interactive training.")

        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, shift=sigma_shift)
        
        inputs_shared = {
            "prompt_list": prompt_list,
            "clip_frames": clip_frames,
            "input_video": input_video,
            "seed": seed,
            "rand_device": rand_device,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "use_mllm_condition": use_mllm_condition,
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
            "use_gradient_checkpointing": use_gradient_checkpointing,
        }
        inputs_posi = {}
        inputs_nega = {}
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            noise_pred = self.model_fn(**models, **inputs_shared, timestep=timestep)
            inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])

        # Decode
        self.load_models_to_device(["vae"])
        video = self.vae.decode(inputs_shared["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video = self.vae_output_to_video(video)
        self.load_models_to_device([])
        return video


class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames"),
            output_params=("height", "width", "num_frames"),
        )

    def process(self, pipe: WanVideoCleanVAEPipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}


class WanVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames", "seed", "rand_device"),
            output_params=("noise",)
        )

    def process(self, pipe: WanVideoCleanVAEPipeline, height, width, num_frames, seed, rand_device):
        length = (num_frames - 1) // 4 + 1
        shape = (1, pipe.vae.model.z_dim, length, height // pipe.vae.upsampling_factor, width // pipe.vae.upsampling_factor)
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        return {"noise": noise}


class WanVideoUnit_CleanVAEBlockScheduler(PipelineUnit):
    """
    Block scheduler for clean VAE history training.
    Uses standard BLOCK_DURATION for block scheduling (same as original).
    History frames are sampled from 0 with fixed stride, all < latent_start, plus latent_start - 1.
    """
    def __init__(self):
        super().__init__(
            input_params=("prompt_list", "clip_frames", "num_frames", "latent_stride"),
            output_params=("block_info",)
        )
    
    def process(self, pipe, prompt_list, clip_frames, num_frames, latent_stride=LATENT_STRIDE):
        if len(prompt_list) != len(clip_frames):
            raise ValueError(
                f"prompt_list length ({len(prompt_list)}) must equal clip_frames length ({len(clip_frames)})."
            )
        if sum(clip_frames) != num_frames:
            raise ValueError(
                f"Sum of clip_frames ({sum(clip_frames)}) must equal num_frames ({num_frames})."
            )
        
        # Use standard block scheduling (same as original WanVideoUnit_BlockScheduler)
        max_block_latent_frames = 4 * BLOCK_DURATION
        block_info = []
        current_video_frame = 0
        current_latent_frame = 0
        global_block_idx = 0
        
        for clip_idx, clip_frame_count in enumerate(clip_frames):
            clip_start_frame = current_video_frame
            clip_end_frame = current_video_frame + clip_frame_count
            
            if clip_idx == 0:
                clip_num_latent = 1 + (clip_frame_count - 1) // 4
            else:
                clip_num_latent = clip_frame_count // 4
            
            num_blocks_in_clip = (clip_num_latent + max_block_latent_frames - 1) // max_block_latent_frames
            
            for block_idx_in_clip in range(num_blocks_in_clip):
                local_latent_start = block_idx_in_clip * max_block_latent_frames
                local_latent_end = min((block_idx_in_clip + 1) * max_block_latent_frames, clip_num_latent)
                block_num_latent = local_latent_end - local_latent_start
                
                global_latent_start = current_latent_frame + local_latent_start
                global_latent_end = current_latent_frame + local_latent_end
                
                if clip_idx == 0:
                    if local_latent_start == 0:
                        block_start_frame = clip_start_frame
                        block_end_frame = clip_start_frame + 1 + (local_latent_end - 1) * 4
                    else:
                        block_start_frame = clip_start_frame + 1 + (local_latent_start - 1) * 4
                        block_end_frame = clip_start_frame + 1 + (local_latent_end - 1) * 4
                else:
                    block_start_frame = clip_start_frame + local_latent_start * 4
                    block_end_frame = clip_start_frame + local_latent_end * 4
                
                block_end_frame = min(block_end_frame, clip_end_frame)
                block_num_frames = block_end_frame - block_start_frame
                
                # Calculate history indices: sample from 0 with stride, all < latent_start, plus latent_start - 1
                history_indices = []
                
                # Sample from 0 with stride, all indices < global_latent_start
                idx = 0
                while idx < global_latent_start:
                    history_indices.append(idx)
                    idx += latent_stride
                
                # Add latent_start - 1 if not already included
                if global_latent_start > 0 and (global_latent_start - 1) not in history_indices:
                    history_indices.append(global_latent_start - 1)
                
                history_indices = sorted(set(history_indices))
                
                block_info.append({
                    "prompt_idx": clip_idx,
                    "clip_idx": clip_idx,
                    "block_idx_in_clip": block_idx_in_clip,
                    "global_block_idx": global_block_idx,
                    "start_frame": block_start_frame,
                    "end_frame": block_end_frame,
                    "latent_start": global_latent_start,
                    "latent_end": global_latent_end,
                    "num_frames": block_num_frames,
                    "num_latent_frames": block_num_latent,
                    "history_indices": history_indices,
                })
                
                global_block_idx += 1
            
            current_video_frame = clip_end_frame
            current_latent_frame += clip_num_latent
        
        return {"block_info": block_info}


class WanVideoUnit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "noise", "tiled", "tile_size", "tile_stride", "use_mllm_condition"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoCleanVAEPipeline, input_video, noise, tiled, tile_size, tile_stride, use_mllm_condition=False):
        if input_video is None:
            return {"latents": noise}
        pipe.load_models_to_device(self.onload_model_names)
        input_video = pipe.preprocess_video(input_video)
        input_latents = pipe.vae.encode(input_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        elif use_mllm_condition:
            return {"latents": noise}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents}


class WanVideoUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("prompt_list", "block_info"),
            output_params=("prompt_embeddings_map",),
            onload_model_names=("text_encoder",)
        )
    
    def encode_prompt(self, pipe: WanVideoCleanVAEPipeline, prompt):
        ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(pipe.device)
        mask = mask.to(pipe.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_emb = pipe.text_encoder(ids, mask)
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        return prompt_emb

    def process(self, pipe: WanVideoCleanVAEPipeline, prompt_list, block_info) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        unique_prompt_indices = sorted(set(block["prompt_idx"] for block in block_info))
        prompt_embeddings_map = {}
        for prompt_idx in unique_prompt_indices:
            prompt_text = prompt_list[prompt_idx]
            prompt_embeddings_map[prompt_idx] = self.encode_prompt(pipe, prompt_text)
        return {"prompt_embeddings_map": prompt_embeddings_map}


def model_fn_wan_video_clean_vae(
    dit: WanModel,
    latents: torch.Tensor,
    input_latents: torch.Tensor,
    block_info: list[dict],
    timestep: torch.Tensor,
    prompt_embeddings_map: dict,
    use_gradient_checkpointing: bool = False,
    clean_timestep: torch.Tensor = None,
    clean_input_latents: torch.Tensor = None,
    t5_cfg_drop: float = 0.0,
    latent_stride: int = LATENT_STRIDE,
    **kwargs,
) -> torch.Tensor:
    """
    Model function for clean VAE history training (no MLLM).
    Uses clean frame VAE embeddings as history, sampled with fixed stride.
    """
    # Process text embeddings per block with CFG drop
    context_per_prompt = {idx: dit.text_embedding(emb) for idx, emb in prompt_embeddings_map.items()}
    context_per_block = {}
    for block in block_info:
        block_idx = block["global_block_idx"]
        prompt_idx = block["prompt_idx"]
        
        # Apply t5_cfg_drop per block (use torch.rand for distributed sync)
        if t5_cfg_drop > 0 and torch.rand(1).item() < t5_cfg_drop:
            # Zero out t5 embedding for this block
            context_per_block[block_idx] = torch.zeros_like(context_per_prompt[prompt_idx])
        else:
            context_per_block[block_idx] = context_per_prompt[prompt_idx]
    
    # No MLLM embeddings
    mllm_embeddings = None
    
    # Time embeddings
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    
    if clean_timestep is None:
        clean_timestep = t.new_zeros((t.shape[0],))
    t_clean = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, clean_timestep))
    t_mod_clean = dit.time_projection(t_clean).unflatten(1, (6, dit.dim))
    
    # Prepare full latents and freqs
    x = latents
    x_patched_test = dit.patchify(x)
    f_total, h, w = x_patched_test.shape[2:]
    
    freqs = torch.cat([
        dit.freqs[0][:f_total].view(f_total, 1, 1, -1).expand(f_total, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f_total, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f_total, h, w, -1)
    ], dim=-1).reshape(f_total * h * w, 1, -1).to(x.device)
    
    tokens_per_latent_frame = h * w
    
    # Process each block
    noise_preds = []
    for block in block_info:
        block_idx = block["global_block_idx"]
        
        # Use compute_noise_pred_per_block_clean_vae with history indices
        noise_pred_block = compute_noise_pred_per_block_clean_vae(
            dit=dit,
            block_idx=block_idx,
            block_info=block,
            x_full=x,
            input_latents=input_latents,
            clean_input_latents=clean_input_latents if clean_input_latents is not None else input_latents,
            freqs_full=freqs,
            context_per_block=context_per_block,
            t=t,
            t_mod=t_mod,
            t_clean=t_clean,
            t_mod_clean=t_mod_clean,
            tokens_per_latent_frame=tokens_per_latent_frame,
            use_gradient_checkpointing=use_gradient_checkpointing,
            device=x.device,
        )
        noise_preds.append(noise_pred_block)
    
    noise_pred = torch.cat(noise_preds, dim=2)
    return noise_pred


def compute_noise_pred_per_block_clean_vae(
    dit: WanModel,
    block_idx: int,
    block_info: dict,
    x_full: torch.Tensor,
    input_latents: torch.Tensor,
    clean_input_latents: torch.Tensor,
    freqs_full: torch.Tensor,
    context_per_block: dict,
    t: torch.Tensor,
    t_mod: torch.Tensor,
    t_clean: torch.Tensor,
    t_mod_clean: torch.Tensor,
    tokens_per_latent_frame: int,
    use_gradient_checkpointing: bool,
    device: torch.device,
    timestep_value: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute noise prediction for a block using clean VAE history.
    For each block, prepend history frames (indices < latent_start + latent_start-1)
    from clean_input_latents.
    """
    latent_start = block_info["latent_start"]
    latent_end = block_info["latent_end"]
    history_indices = block_info.get("history_indices", [])
    
    context = context_per_block[block_idx]
    
    # Extract noisy block
    x_block = x_full[:, :, latent_start:latent_end, :, :]
    x_patched = dit.patchify(x_block)
    f, h, w = x_patched.shape[2:]
    x_noisy = rearrange(x_patched, 'b c f h w -> b (f h w) c').contiguous()
    
    # Extract clean history frames
    if len(history_indices) > 0 and clean_input_latents is not None:
        clean_latents_list = []
        for hist_idx in history_indices:
            if hist_idx < clean_input_latents.shape[2]:
                clean_latents_list.append(clean_input_latents[:, :, hist_idx:hist_idx+1, :, :])
        
        if len(clean_latents_list) > 0:
            clean_latents = torch.cat(clean_latents_list, dim=2)
            clean_patched = dit.patchify(clean_latents)
            clean_tokens = rearrange(clean_patched, 'b c f h w -> b (f h w) c').contiguous()
            
            # Combine clean history + noisy current
            x_combined = torch.cat([clean_tokens, x_noisy], dim=1)
            
            num_clean_tokens = clean_tokens.shape[1]
            num_noisy_tokens = x_noisy.shape[1]
            
            # Create masks
            keep_mask = torch.cat([
                torch.zeros(num_clean_tokens, dtype=torch.bool, device=device),
                torch.ones(num_noisy_tokens, dtype=torch.bool, device=device)
            ], dim=0)
            
            block_ids = torch.cat([
                torch.full((num_clean_tokens,), -1, dtype=torch.int32, device=device),
                torch.full((num_noisy_tokens,), 0, dtype=torch.int32, device=device)
            ], dim=0)
            
            # Prepare freqs
            freqs_noisy = freqs_full[latent_start * tokens_per_latent_frame:latent_end * tokens_per_latent_frame]
            freqs_clean_list = []
            for hist_idx in history_indices:
                start_token = hist_idx * tokens_per_latent_frame
                end_token = (hist_idx + 1) * tokens_per_latent_frame
                if end_token <= freqs_full.shape[0]:
                    freqs_clean_list.append(freqs_full[start_token:end_token])
            freqs_clean = torch.cat(freqs_clean_list, dim=0) if freqs_clean_list else freqs_full[:0]
            freqs_combined = torch.cat([freqs_clean, freqs_noisy], dim=0)
            
            # Prepare time embeddings
            t_combined = torch.cat([
                t_clean[:, None, :].expand(t_clean.shape[0], num_clean_tokens, t_clean.shape[-1]),
                t[:, None, :].expand(t.shape[0], num_noisy_tokens, t.shape[-1])
            ], dim=1)
            t_mod_combined = torch.cat([
                t_mod_clean[:, None, :, :].expand(t_mod_clean.shape[0], num_clean_tokens, t_mod_clean.shape[-2], t_mod_clean.shape[-1]),
                t_mod[:, None, :, :].expand(t_mod.shape[0], num_noisy_tokens, t_mod.shape[-2], t_mod.shape[-1])
            ], dim=1)
            
            x_input = x_combined
            freqs_input = freqs_combined
            t_input = t_combined
            t_mod_input = t_mod_combined
        else:
            # No valid history
            x_input = x_noisy
            freqs_input = freqs_full[latent_start * tokens_per_latent_frame:latent_end * tokens_per_latent_frame]
            t_input = t
            t_mod_input = t_mod
            keep_mask = None
    else:
        # No history
        x_input = x_noisy
        freqs_input = freqs_full[latent_start * tokens_per_latent_frame:latent_end * tokens_per_latent_frame]
        t_input = t
        t_mod_input = t_mod
        keep_mask = None
    
    # Forward through DiT blocks (no MLLM)
    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)
        return custom_forward
    
    for dit_block in dit.blocks:
        dit_block._current_block_idx = block_idx
        dit_block._current_timestep = timestep_value
        dit_block._norm_stats = getattr(dit, "cross_attn_norm_stats", None)
        if use_gradient_checkpointing:
            x_input = torch.utils.checkpoint.checkpoint(
                create_custom_forward(dit_block),
                x_input, context, t_mod_input, freqs_input,
                None, None, None, None, None, False,  # No MLLM
                use_reentrant=False,
            )
        else:
            x_input = dit_block(
                x_input, context, t_mod_input, freqs_input,
                mllm_embeddings=None,
                mllm_mask=None,
                mllm_block_mask=None,
                dit_block_mask=None,
                mllm_zero_out=False,
            )
    
    x_output = dit.head(x_input, t_input if t_input.dim() == 3 else t_input)
    
    # Keep only noisy tokens if we had history
    if keep_mask is not None:
        x_output = x_output[:, keep_mask]
    
    noise_pred = dit.unpatchify(x_output, (f, h, w))
    
    return noise_pred
