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
CLEAN_FRAME_COUNT = 2


class WanVideoInterPipeline(BasePipeline):

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
            WanVideoUnit_BlockScheduler(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_PromptEmbedder(),
            WanVideoUnit_MLLMEmbedder(),
        ]
        self.post_units = []
        self.model_fn = model_fn_wan_video_inter


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
        pipe = WanVideoInterPipeline(device=device, torch_dtype=torch_dtype)
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

    def process(self, pipe: WanVideoInterPipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}


class WanVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames", "seed", "rand_device"),
            output_params=("noise",)
        )

    def process(self, pipe: WanVideoInterPipeline, height, width, num_frames, seed, rand_device):
        length = (num_frames - 1) // 4 + 1
        shape = (1, pipe.vae.model.z_dim, length, height // pipe.vae.upsampling_factor, width // pipe.vae.upsampling_factor)
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        return {"noise": noise}


class WanVideoUnit_BlockScheduler(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("prompt_list", "clip_frames", "num_frames"),
            output_params=("block_info",)
        )
    
    def process(self, pipe: WanVideoInterPipeline, prompt_list, clip_frames, num_frames):
        if len(prompt_list) != len(clip_frames):
            raise ValueError(
                f"prompt_list length ({len(prompt_list)}) must equal clip_frames length ({len(clip_frames)})."
            )
        if sum(clip_frames) != num_frames:
            raise ValueError(
                f"Sum of clip_frames ({sum(clip_frames)}) must equal num_frames ({num_frames})."
            )
        
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

    def process(self, pipe: WanVideoInterPipeline, input_video, noise, tiled, tile_size, tile_stride, use_mllm_condition=False):
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
    
    def encode_prompt(self, pipe: WanVideoInterPipeline, prompt):
        ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(pipe.device)
        mask = mask.to(pipe.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_emb = pipe.text_encoder(ids, mask)
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        return prompt_emb

    def process(self, pipe: WanVideoInterPipeline, prompt_list, block_info) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        unique_prompt_indices = sorted(set(block["prompt_idx"] for block in block_info))
        prompt_embeddings_map = {}
        for prompt_idx in unique_prompt_indices:
            prompt_text = prompt_list[prompt_idx]
            prompt_embeddings_map[prompt_idx] = self.encode_prompt(pipe, prompt_text)
        return {"prompt_embeddings_map": prompt_embeddings_map}


def sample_frames_with_constraints(total_frames, target_stride=8):
    if total_frames <= 0:
        return []

    last_frame = total_frames - 1
    n = math.ceil(total_frames / target_stride)
    if n <= 0:
        n = 1
    if n % 2 == 0:
        n += 1

    if n == 1:
        return [last_frame]

    indices = []
    for i in range(n):
        pos = i * (last_frame) / (n - 1) if n - 1 != 0 else float(last_frame)
        idx = int(round(pos))
        indices.append(idx)

    indices = sorted(set(indices))
    if indices[-1] != last_frame:
        indices.append(last_frame)
        indices = sorted(set(indices))

    if indices and indices[0] == 0:
        indices = indices[1:]

    return indices


class WanVideoUnit_MLLMEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "height", "width", "num_frames", "use_mllm_condition", "prompt_list", "block_info"),
            output_params=("mllm_hidden_states", "mllm_mask", "mllm_kv_len", "mllm_position_ids", "mllm_vision_ranges"),
            onload_model_names=("mllm_encoder",)
        )
    
    def process_video_for_mllm(self, pipe: WanVideoInterPipeline, input_video, block_info):
        video_blocks = []
        video_metadata_blocks = []
        sampled_counts = []
        
        for block in block_info:
            block_frames = input_video[block["start_frame"]:block["end_frame"]]
            total_frames_in_block = len(block_frames)
            local_indices = sample_frames_with_constraints(total_frames_in_block, target_stride=8)
            sampled_counts.append(len(local_indices))
            global_indices = [block["start_frame"] + i for i in local_indices]
            sampled_frames = [input_video[i] for i in global_indices]
            metadata = {
                "fps": 16,
                "frames_indices": global_indices,
                "total_num_frames": len(input_video)
            }
            video_blocks.append(sampled_frames)
            video_metadata_blocks.append(metadata)
        
        return video_blocks, video_metadata_blocks, sampled_counts
        
    def encode_prompt(self, pipe: WanVideoInterPipeline, prompt_list, block_info, video_blocks, video_metadata_blocks):
        template = "<|im_start|>system\nAnalyze the user's full video instruction and the provided partial video sequence. First, concisely describe the key elements, actions, and scene of the existing video segment. Then, predict the precise visual content for the next segment of video. The prediction must strictly follow the user's full instruction while ensuring seamless temporal continuity in motion, camera work, lighting, and object interactions with the existing frames. For the initial frame (when no video exists), use the instruction as the sole basis to generate the starting scene.<|im_end|>\n<|im_start|>user\n"
        drop_idx = 111
        
        text_parts = []
        all_videos = []
        all_metadata = []
        
        prev_prompt_idx = None
        for block_idx, block in enumerate(block_info):
            prompt_idx = block["prompt_idx"]
            if prompt_idx != prev_prompt_idx:
                if len(text_parts) > 0:
                    text_parts.append(" ")
                text_parts.append(prompt_list[prompt_idx])
                prev_prompt_idx = prompt_idx
            
            text_parts.append(" <|vision_start|><|video_pad|><|vision_end|>")
            all_videos.append(video_blocks[block_idx])
            all_metadata.append(video_metadata_blocks[block_idx])
        
        full_text = template + "".join(text_parts)
        
        txt = [full_text]
        model_inputs = pipe.mllm_processor(
            text=txt,
            videos=all_videos,
            padding=True,
            video_metadata=all_metadata,
            return_tensors="pt",
            do_resize=False,
            do_sample_frames=False
        ).to(pipe.device)
        
        position_ids, _ = pipe.mllm_encoder.model.get_rope_index(
            input_ids=model_inputs["input_ids"],
            video_grid_thw=model_inputs["video_grid_thw"],
            attention_mask=model_inputs["attention_mask"],
        )
        
        hidden_states = pipe.mllm_encoder(**model_inputs)[-1]
        hidden_states = hidden_states[:, drop_idx:]
        input_ids = model_inputs["input_ids"][:, drop_idx:]
        position_ids = position_ids[..., drop_idx:]
        
        return hidden_states, input_ids, position_ids
    
    def calculate_mllm_mask(self, pipe: WanVideoInterPipeline, num_frames, height, width, input_ids, block_info, sampled_counts):
        lat_h = height // 16
        lat_w = width // 16
        s_dit = lat_h * lat_w
        num_dit_frames = 1 + (num_frames - 1) // 4
        num_dit_tokens = num_dit_frames * s_dit
        mllm_seq_len = input_ids.shape[1]
        
        vision_start_token_id = pipe.mllm_encoder.config.vision_start_token_id
        vision_end_token_id = pipe.mllm_encoder.config.vision_end_token_id
        
        vision_start_positions = (input_ids[0] == vision_start_token_id).nonzero(as_tuple=True)[0]
        vision_end_positions = (input_ids[0] == vision_end_token_id).nonzero(as_tuple=True)[0]
        
        mllm_mask = torch.zeros((1, num_dit_tokens), device=input_ids.device, dtype=torch.int32)
        mllm_vision_ranges = torch.zeros((1, num_dit_tokens, 2), device=input_ids.device, dtype=torch.int32)
        
        vision_seg_offset = 0
        for block_idx, block in enumerate(block_info):
            num_sampled = sampled_counts[block_idx]
            num_vision_segs = num_sampled // 2
            
            latent_start = block["latent_start"]
            latent_end = block["latent_end"]
            start_dit_token = latent_start * s_dit
            end_dit_token = latent_end * s_dit
            
            if block_idx == 0 or len(vision_end_positions) == 0 or len(vision_start_positions) == 0:
                mllm_mask[:, start_dit_token:end_dit_token] = 0
                mllm_vision_ranges[:, start_dit_token:end_dit_token, 0] = 0
                mllm_vision_ranges[:, start_dit_token:end_dit_token, 1] = 0
            else:
                end_pos_idx = min(max(vision_seg_offset - 1, 0), len(vision_end_positions) - 1)
                end_mllm_prefix = vision_end_positions[end_pos_idx].item() + 1
                mllm_mask[:, start_dit_token:end_dit_token] = min(end_mllm_prefix, mllm_seq_len)
                
                vision_range_start = vision_start_positions[0].item()
                vision_range_end = vision_end_positions[end_pos_idx].item() + 1
                mllm_vision_ranges[:, start_dit_token:end_dit_token, 0] = vision_range_start
                mllm_vision_ranges[:, start_dit_token:end_dit_token, 1] = vision_range_end
            
            vision_seg_offset += num_vision_segs
            
        return mllm_mask, mllm_seq_len, mllm_vision_ranges

    def process(self, pipe: WanVideoInterPipeline, prompt_list, input_video, height, width, num_frames, block_info, use_mllm_condition=False):
        if not use_mllm_condition:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        
        video_blocks, video_metadata_blocks, sampled_counts = self.process_video_for_mllm(
            pipe, input_video, block_info
        )
        mllm_hidden_states, input_ids, position_ids = self.encode_prompt(
            pipe, prompt_list, block_info, video_blocks, video_metadata_blocks
        )
        mllm_mask, mllm_kv_len, mllm_vision_ranges = self.calculate_mllm_mask(
            pipe, num_frames, height, width, input_ids, block_info, sampled_counts
        )
        return {
            "mllm_hidden_states": mllm_hidden_states,
            "mllm_mask": mllm_mask,
            "mllm_kv_len": mllm_kv_len,
            "mllm_position_ids": position_ids,
            "mllm_vision_ranges": mllm_vision_ranges,
        }


def compute_noise_pred_per_block(
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
    mllm_embeddings: torch.Tensor,
    mllm_mask_full: torch.Tensor,
    mllm_vision_ranges: torch.Tensor,
    mllm_kv_len: int,
    tokens_per_latent_frame: int,
    use_gradient_checkpointing: bool,
    device: torch.device,
) -> torch.Tensor:
    latent_start = block_info["latent_start"]
    latent_end = block_info["latent_end"]
    
    context = context_per_block[block_idx]
    
    x_block = x_full[:, :, latent_start:latent_end, :, :]
    x_patched = dit.patchify(x_block)
    f, h, w = x_patched.shape[2:]
    x_noisy = rearrange(x_patched, 'b c f h w -> b (f h w) c').contiguous()
    
    clean_frame_count = CLEAN_FRAME_COUNT
    keep_mask = None
    block_ids = None
    mllm_mask_combined = None
    
    clean_latents_source = clean_input_latents if clean_input_latents is not None else input_latents

    if block_idx > 0 and clean_frame_count > 0 and clean_latents_source is not None:
        prev_latent_end = latent_start
        prev_latent_start = max(0, prev_latent_end - clean_frame_count)
        actual_clean_count = prev_latent_end - prev_latent_start
        if actual_clean_count > 0:
            clean_latents = clean_latents_source[:, :, prev_latent_start:prev_latent_end, :, :]
            clean_patched = dit.patchify(clean_latents)
            clean_tokens = rearrange(clean_patched, 'b c f h w -> b (f h w) c').contiguous()
            
            x_combined = torch.cat([clean_tokens, x_noisy], dim=1)
            
            num_clean_tokens = clean_tokens.shape[1]
            num_noisy_tokens = x_noisy.shape[1]
            keep_mask = torch.cat([
                torch.zeros(num_clean_tokens, dtype=torch.bool, device=device),
                torch.ones(num_noisy_tokens, dtype=torch.bool, device=device)
            ], dim=0)
            block_ids = torch.cat([
                torch.full((num_clean_tokens,), -1, dtype=torch.int32, device=device),
                torch.full((num_noisy_tokens,), 0, dtype=torch.int32, device=device)
            ], dim=0)
            
            start_token_noisy = latent_start * tokens_per_latent_frame
            end_token_noisy = latent_end * tokens_per_latent_frame
            freqs_noisy = freqs_full[start_token_noisy:end_token_noisy]
            
            start_token_clean = prev_latent_start * tokens_per_latent_frame
            end_token_clean = prev_latent_end * tokens_per_latent_frame
            freqs_clean = freqs_full[start_token_clean:end_token_clean]
            freqs_combined = torch.cat([freqs_clean, freqs_noisy], dim=0)
            
            t_combined = torch.cat([
                t_clean[:, None, :].expand(t_clean.shape[0], num_clean_tokens, t_clean.shape[-1]),
                t[:, None, :].expand(t.shape[0], num_noisy_tokens, t.shape[-1])
            ], dim=1)
            t_mod_combined = torch.cat([
                t_mod_clean[:, None, :, :].expand(t_mod_clean.shape[0], num_clean_tokens, t_mod_clean.shape[-2], t_mod_clean.shape[-1]),
                t_mod[:, None, :, :].expand(t_mod.shape[0], num_noisy_tokens, t_mod.shape[-2], t_mod.shape[-1])
            ], dim=1)
            
            if mllm_mask_full is not None:
                mllm_mask_noisy = mllm_mask_full[:, start_token_noisy:end_token_noisy]
                clean_prefix = mllm_mask_full[:, start_token_noisy:start_token_noisy + 1]
                mllm_mask_clean = clean_prefix.expand(mllm_mask_full.shape[0], num_clean_tokens)
                mllm_mask_combined = torch.cat([mllm_mask_clean, mllm_mask_noisy], dim=1)
            
            x_input = x_combined
            freqs_input = freqs_combined
            t_input = t_combined
            t_mod_input = t_mod_combined
            mllm_mask_input = mllm_mask_combined
        else:
            x_input = x_noisy
            freqs_input = freqs_full[latent_start * tokens_per_latent_frame:latent_end * tokens_per_latent_frame]
            t_input = t
            t_mod_input = t_mod
            mllm_mask_input = mllm_mask_full[:, latent_start * tokens_per_latent_frame:latent_end * tokens_per_latent_frame] if mllm_mask_full is not None else None
    else:
        x_input = x_noisy
        freqs_input = freqs_full[latent_start * tokens_per_latent_frame:latent_end * tokens_per_latent_frame]
        t_input = t
        t_mod_input = t_mod
        mllm_mask_input = mllm_mask_full[:, latent_start * tokens_per_latent_frame:latent_end * tokens_per_latent_frame] if mllm_mask_full is not None else None

    mllm_block_mask = None
    if FLEX_ATTENTION_AVAILABLE and create_block_mask is not None and \
       mllm_embeddings is not None and mllm_mask_input is not None and \
       mllm_vision_ranges is not None:
        start_token_global = latent_start * tokens_per_latent_frame
        vision_range_start = mllm_vision_ranges[0, start_token_global, 0].item()
        vision_range_end = mllm_vision_ranges[0, start_token_global, 1].item()
        
        B = x_input.shape[0]
        Q_LEN = x_input.shape[1]
        KV_LEN = mllm_kv_len
        
        def mask_mod(b, h, q_idx, kv_idx):
            in_vision_range = (kv_idx >= vision_range_start) & (kv_idx < vision_range_end)
            if mllm_mask_combined is not None:
                prefix_ok = kv_idx < mllm_mask_combined[b, q_idx]
            else:
                prefix_ok = kv_idx < mllm_mask_input[b, q_idx]
            return in_vision_range & prefix_ok
        
        mllm_block_mask = create_block_mask(
            mask_mod, B=B, H=None, Q_LEN=Q_LEN, KV_LEN=KV_LEN,
            device=str(device)
        )
    
    dit_block_mask = None
    if FLEX_ATTENTION_AVAILABLE and create_block_mask is not None and \
       block_ids is not None and keep_mask is not None:
        B = x_input.shape[0]
        Q_LEN = x_input.shape[1]
        KV_LEN = Q_LEN
        
        def mask_mod_dit(b, h, q_idx, kv_idx):
            same_block = block_ids[q_idx] == block_ids[kv_idx]
            q_is_clean = ~keep_mask[q_idx]
            kv_is_noisy = keep_mask[kv_idx]
            invalid_clean_look = q_is_clean & kv_is_noisy
            return same_block & (~invalid_clean_look)
        
        dit_block_mask = create_block_mask(
            mask_mod_dit, B=B, H=None, Q_LEN=Q_LEN, KV_LEN=KV_LEN,
            device=str(device)
        )
    
    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)
        return custom_forward
    
    for dit_block in dit.blocks:
        if use_gradient_checkpointing:
            x_input = torch.utils.checkpoint.checkpoint(
                create_custom_forward(dit_block),
                x_input, context, t_mod_input, freqs_input,
                mllm_embeddings, mllm_mask_input, mllm_block_mask, dit_block_mask,
                use_reentrant=False,
            )
        else:
            x_input = dit_block(
                x_input, context, t_mod_input, freqs_input,
                mllm_embeddings=mllm_embeddings,
                mllm_mask=mllm_mask_input,
                mllm_block_mask=mllm_block_mask,
                dit_block_mask=dit_block_mask,
            )
    
    x_output = dit.head(x_input, t_input if t_input.dim() == 3 else t_input)
    
    if keep_mask is not None:
        x_output = x_output[:, keep_mask]
    
    noise_pred = dit.unpatchify(x_output, (f, h, w))
    
    return noise_pred


def model_fn_wan_video_inter(
    dit: WanModel,
    latents: torch.Tensor,
    input_latents: torch.Tensor,
    block_info: list[dict],
    timestep: torch.Tensor,
    prompt_embeddings_map: dict,
    mllm_hidden_states: torch.Tensor,
    mllm_mask: torch.Tensor,
    mllm_kv_len: int,
    mllm_position_ids: torch.Tensor,
    mllm_vision_ranges: torch.Tensor,
    use_gradient_checkpointing: bool = False,
    clean_timestep: torch.Tensor = None,
    clean_input_latents: torch.Tensor = None,
    **kwargs,
) -> torch.Tensor:
    context_per_prompt = {idx: dit.text_embedding(emb) for idx, emb in prompt_embeddings_map.items()}
    context_per_block = {}
    for block in block_info:
        block_idx = block["global_block_idx"]
        prompt_idx = block["prompt_idx"]
        context_per_block[block_idx] = context_per_prompt[prompt_idx]
    
    if hasattr(dit, "has_mllm_input") and dit.has_mllm_input and mllm_hidden_states is not None:
        mllm_embeddings = dit.mllm_embedding(
            mllm_hidden_states,
            position_ids=mllm_position_ids,
            mllm_mask=mllm_mask
        )
    else:
        mllm_embeddings = None
    
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    
    if clean_timestep is None:
        clean_timestep = t.new_zeros((t.shape[0],))
    t_clean = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, clean_timestep))
    t_mod_clean = dit.time_projection(t_clean).unflatten(1, (6, dit.dim))
    
    x = latents
    x_patched_test = dit.patchify(x)
    f_total, h, w = x_patched_test.shape[2:]
    
    freqs = torch.cat([
        dit.freqs[0][:f_total].view(f_total, 1, 1, -1).expand(f_total, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f_total, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f_total, h, w, -1)
    ], dim=-1).reshape(f_total * h * w, 1, -1).to(x.device)
    
    tokens_per_latent_frame = h * w
    
    noise_preds = []
    for block in block_info:
        block_idx = block["global_block_idx"]
        noise_pred_block = compute_noise_pred_per_block(
            dit=dit,
            block_idx=block_idx,
            block_info=block,
            x_full=x,
            input_latents=input_latents,
            clean_input_latents=clean_input_latents,
            freqs_full=freqs,
            context_per_block=context_per_block,
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
            device=x.device,
        )
        noise_preds.append(noise_pred_block)
    
    noise_pred = torch.cat(noise_preds, dim=2)
    return noise_pred
