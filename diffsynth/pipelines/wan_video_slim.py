import torch, types, math, os
import numpy as np
from PIL import Image
from einops import repeat
from typing import Optional, Union
from einops import rearrange
from tqdm import tqdm
from typing_extensions import Literal

from ..diffusion import FlowMatchScheduler
from ..core import ModelConfig
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit

from ..models.wan_video_dit import WanModel, sinusoidal_embedding_1d
from ..models.wan_video_text_encoder import WanTextEncoder, HuggingfaceTokenizer
from ..models.wan_video_vae import WanVideoVAE
from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.wan_video_mllm_encoder import WanMLLMEncoder
from ..models.wan_video_mllm_encoder import Qwen3VLProcessor

try:
    from torch.nn.attention.flex_attention import create_block_mask
    FLEX_ATTENTION_AVAILABLE = os.environ.get("DISABLE_FLEX_ATTENTION", "0") != "1"
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    create_block_mask = None

BLOCK_DURATION = 2
CLEAN_FRAME_COUNT = 1

class WanVideoPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16, time_division_factor=4, time_division_remainder=1
        )
        self.scheduler = FlowMatchScheduler("Wan")
        self.tokenizer: HuggingfaceTokenizer = None
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.dit2: WanModel = None
        self.vae: WanVideoVAE = None
        self.mllm_encoder: WanMLLMEncoder = None
        self.mllm_processor: Qwen3VLProcessor = None
        self.in_iteration_models = ("dit",)
        self.in_iteration_models_2 = ("dit2",)
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_PromptEmbedder(),
            WanVideoUnit_MLLMEmbedder(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_ImageEmbedderVAE(),
            WanVideoUnit_ImageEmbedderCLIP(),
            WanVideoUnit_ImageEmbedderFused(),
            WanVideoUnit_TeaCache(),
            WanVideoUnit_CfgMerger(),
        ]
        self.post_units = []
        self.model_fn = model_fn_wan_video


    def enable_usp(self):
        from ..utils.xfuser import get_sequence_parallel_world_size, usp_attn_forward, usp_dit_forward

        for block in self.dit.blocks:
            block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
        self.dit.forward = types.MethodType(usp_dit_forward, self.dit)
        if self.dit2 is not None:
            for block in self.dit2.blocks:
                block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
            self.dit2.forward = types.MethodType(usp_dit_forward, self.dit2)
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
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "models_clip_open-clip-xlm-roberta-large-vit-huge-14.safetensors"),
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
        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype)
        if use_usp:
            from ..utils.xfuser import initialize_usp
            initialize_usp()
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)
        
        # Fetch models
        pipe.text_encoder = model_pool.fetch_model("wan_video_text_encoder")
        dit = model_pool.fetch_model("wan_video_dit", index=2)
        if isinstance(dit, list):
            pipe.dit, pipe.dit2 = dit
        else:
            pipe.dit = dit
        pipe.vae = model_pool.fetch_model("wan_video_vae")
        pipe.image_encoder = model_pool.fetch_model("wan_video_image_encoder")
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
        if use_usp: pipe.enable_usp()
        
        # VRAM Management
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe


    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: Optional[str] = "",
        # Image-to-video
        input_image: Optional[Image.Image] = None,
        # Video-to-video
        input_video: Optional[list[Image.Image]] = None,
        denoising_strength: Optional[float] = 1.0,
        # Randomness
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        # Shape
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames=81,
        use_mllm_condition: Optional[bool] = False,
        mllm_pos_mode: Optional[Literal["text", "full"]] = "full",
        mllm_neg_mode: Optional[Literal["text", "full"]] = "full",
        # Classifier-free guidance
        cfg_scale: Optional[float] = 5.0,
        cfg_merge: Optional[bool] = False,
        # Boundary
        switch_DiT_boundary: Optional[float] = 0.875,
        # Scheduler
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        # VAE tiling
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple[int, int]] = (30, 52),
        tile_stride: Optional[tuple[int, int]] = (15, 26),
        # Sliding window
        sliding_window_size: Optional[int] = None,
        sliding_window_stride: Optional[int] = None,
        # Teacache
        tea_cache_l1_thresh: Optional[float] = None,
        tea_cache_model_id: Optional[str] = "",
        # progress_bar
        progress_bar_cmd=tqdm,
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)
        
        # Inputs
        inputs_posi = {
            "prompt": prompt, "mllm_pos_mode": mllm_pos_mode,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt, "mllm_neg_mode": mllm_neg_mode,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }
        inputs_shared = {
            "mllm_prompt": prompt,
            "input_image": input_image,
            "input_video": input_video, "denoising_strength": denoising_strength,
            "seed": seed, "rand_device": rand_device,
            "height": height, "width": width, "num_frames": num_frames,
            "use_mllm_condition": use_mllm_condition,
            "cfg_scale": cfg_scale, "cfg_merge": cfg_merge,
            "sigma_shift": sigma_shift,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "sliding_window_size": sliding_window_size, "sliding_window_stride": sliding_window_stride,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            # Switch DiT if necessary
            if timestep.item() < switch_DiT_boundary * 1000 and self.dit2 is not None and not models["dit"] is self.dit2:
                self.load_models_to_device(self.in_iteration_models_2)
                models["dit"] = self.dit2
                
            # Timestep
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            
            # Inference
            noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
            if cfg_scale != 1.0:
                if cfg_merge:
                    noise_pred_posi, noise_pred_nega = noise_pred_posi.chunk(2, dim=0)
                else:
                    noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])
            if "first_frame_latents" in inputs_shared:
                inputs_shared["latents"][:, :, 0:1] = inputs_shared["first_frame_latents"]
        
        # post-denoising, pre-decoding processing logic
        for unit in self.post_units:
            inputs_shared, _, _ = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)
        # Decode
        self.load_models_to_device(['vae'])
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

    def process(self, pipe: WanVideoPipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}



class WanVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames", "seed", "rand_device"),
            output_params=("noise",)
        )

    def process(self, pipe: WanVideoPipeline, height, width, num_frames, seed, rand_device):
        length = (num_frames - 1) // 4 + 1
        shape = (1, pipe.vae.model.z_dim, length, height // pipe.vae.upsampling_factor, width // pipe.vae.upsampling_factor)
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        return {"noise": noise}
    


class WanVideoUnit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "noise", "tiled", "tile_size", "tile_stride", "use_mllm_condition"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_video, noise, tiled, tile_size, tile_stride, use_mllm_condition=False):
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
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "positive": "positive"},
            input_params_nega={"prompt": "negative_prompt", "positive": "positive"},
            output_params=("context",),
            onload_model_names=("text_encoder",)
        )
    
    def encode_prompt(self, pipe: WanVideoPipeline, prompt):
        ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(pipe.device)
        mask = mask.to(pipe.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_emb = pipe.text_encoder(ids, mask)
        for i, v in enumerate(seq_lens):
            prompt_emb[:, v:] = 0
        return prompt_emb

    def process(self, pipe: WanVideoPipeline, prompt, positive) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        prompt_emb = self.encode_prompt(pipe, prompt)
        return {"context": prompt_emb}


class WanVideoUnit_MLLMEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "height", "width", "num_frames", "use_mllm_condition", "mllm_prompt"),
            output_params=("mllm_hidden_states", "mllm_mask", "mllm_kv_len", "mllm_position_ids"),
            onload_model_names=("mllm_encoder",)
        )
    
    def process_video_for_mllm(self, pipe: WanVideoPipeline, input_video):
        indices = list(range(4, len(input_video), 8))
        if len(indices) % 2 != 0:
            indices = indices[:-1]
        video_metadata = [{"fps": 16, "frames_indices": indices, "total_num_frames": len(input_video)}]
        input_video = [input_video[i] for i in indices]
        return input_video, video_metadata
        
    def encode_prompt(self, pipe: WanVideoPipeline, prompt, input_video, video_metadata, mode="full"):
        template = "<|im_start|>system\nAnalyze the user's full video instruction and the provided partial video sequence. First, concisely describe the key elements, actions, and scene of the existing video segment. Then, predict the precise visual content for the next segment of video. The prediction must strictly follow the user's full instruction while ensuring seamless temporal continuity in motion, camera work, lighting, and object interactions with the existing frames. For the initial frame (when no video exists), use the instruction as the sole basis to generate the starting scene.<|im_end|>\n<|im_start|>user\n{}"
        drop_idx = 111 
        if mode == "text":
            txt = [template.format(prompt)]
            model_inputs = pipe.mllm_processor(text=txt, return_tensors="pt", do_resize=False, do_sample_frames=False).to(pipe.device)
            position_ids, _ = pipe.mllm_encoder.model.get_rope_index(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
            )
        else:
            txt = [template.format(prompt + " <|vision_start|><|video_pad|><|vision_end|>")]
            model_inputs = pipe.mllm_processor(text=txt, videos=input_video, padding=True, video_metadata=video_metadata, return_tensors="pt", do_resize=False, do_sample_frames=False).to(pipe.device)
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
    
    def calculate_mllm_mask(self, pipe: WanVideoPipeline, num_frames, height, width, input_ids):
        lat_h = height // 16
        lat_w = width // 16
        S_dit = lat_h * lat_w 
        num_dit_frames = 1 + (num_frames - 1) // 4
        num_dit_tokens = num_dit_frames * S_dit
        mllm_seq_len = input_ids.shape[1]
        
        vision_start_token_id = pipe.mllm_encoder.config.vision_start_token_id
        vision_end_token_id = pipe.mllm_encoder.config.vision_end_token_id
        
        vision_start_positions = (input_ids[0] == vision_start_token_id).nonzero(as_tuple=True)[0]
        vision_end_positions = (input_ids[0] == vision_end_token_id).nonzero(as_tuple=True)[0]
        
        prefix_lengths = torch.zeros((1, num_dit_tokens), device=input_ids.device, dtype=torch.int32)
        
        if len(vision_start_positions) > 0:
            text_end = vision_start_positions[0].item() - 6
        else:
            text_end = mllm_seq_len
        
        for i in range(0, num_dit_frames):
            start_dit = i * S_dit
            end_dit = (i + 1) * S_dit
            visible_segments = i // (4 * BLOCK_DURATION)
            
            if visible_segments == 0:
                end_mllm = text_end
            elif visible_segments > 0 and visible_segments*BLOCK_DURATION <= len(vision_end_positions):
                end_mllm = vision_end_positions[visible_segments*BLOCK_DURATION-1].item() + 1
            else:
                end_mllm = mllm_seq_len
            
            prefix_lengths[:, start_dit:end_dit] = min(end_mllm, mllm_seq_len)
        
        return prefix_lengths, mllm_seq_len

    def process(self, pipe: WanVideoPipeline, mllm_prompt, input_video, height, width, num_frames, use_mllm_condition=False, mode="full"):
        if not use_mllm_condition:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        if mode == "text":
            mllm_hidden_states, input_ids, position_ids = self.encode_prompt(pipe, mllm_prompt, None, None, mode="text")
        else:
            mllm_video, video_metadata = self.process_video_for_mllm(pipe, input_video)
            mllm_hidden_states, input_ids, position_ids = self.encode_prompt(pipe, mllm_prompt, mllm_video, video_metadata, mode="full")
        mllm_mask, mllm_kv_len = self.calculate_mllm_mask(pipe, num_frames, height, width, input_ids)
        return {
            "mllm_hidden_states": mllm_hidden_states,
            "mllm_mask": mllm_mask,
            "mllm_kv_len": mllm_kv_len,
            "mllm_position_ids": position_ids,
        }



class WanVideoUnit_ImageEmbedderCLIP(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "height", "width"),
            output_params=("clip_feature",),
            onload_model_names=("image_encoder",)
        )

    def process(self, pipe: WanVideoPipeline, input_image, height, width):
        if input_image is None or pipe.image_encoder is None or not pipe.dit.require_clip_embedding:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        clip_context = pipe.image_encoder.encode_image([image])
        clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"clip_feature": clip_context}
    


class WanVideoUnit_ImageEmbedderVAE(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            output_params=("y",),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_image, num_frames, height, width, tiled, tile_size, tile_stride):
        if input_image is None or not pipe.dit.require_vae_embedding:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        msk = torch.ones(1, num_frames, height//8, width//8, device=pipe.device)
        msk[:, 1:] = 0
        vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)

        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]
        
        y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"y": y}



class WanVideoUnit_ImageEmbedderFused(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "latents", "height", "width", "tiled", "tile_size", "tile_stride"),
            output_params=("latents", "fuse_vae_embedding_in_latents", "first_frame_latents"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_image, latents, height, width, tiled, tile_size, tile_stride):
        if input_image is None or not pipe.dit.fuse_vae_embedding_in_latents:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).transpose(0, 1)
        z = pipe.vae.encode([image], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        latents[:, :, 0: 1] = z
        return {"latents": latents, "fuse_vae_embedding_in_latents": True, "first_frame_latents": z}



class WanVideoUnit_TeaCache(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"num_inference_steps": "num_inference_steps", "tea_cache_l1_thresh": "tea_cache_l1_thresh", "tea_cache_model_id": "tea_cache_model_id"},
            input_params_nega={"num_inference_steps": "num_inference_steps", "tea_cache_l1_thresh": "tea_cache_l1_thresh", "tea_cache_model_id": "tea_cache_model_id"},
            output_params=("tea_cache",)
        )

    def process(self, pipe: WanVideoPipeline, num_inference_steps, tea_cache_l1_thresh, tea_cache_model_id):
        if tea_cache_l1_thresh is None:
            return {}
        return {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id)}



class WanVideoUnit_CfgMerger(PipelineUnit):
    def __init__(self):
        super().__init__(take_over=True)
        self.concat_tensor_names = ["context", "clip_feature", "y"]

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if not inputs_shared["cfg_merge"]:
            return inputs_shared, inputs_posi, inputs_nega
        for name in self.concat_tensor_names:
            tensor_posi = inputs_posi.get(name)
            tensor_nega = inputs_nega.get(name)
            tensor_shared = inputs_shared.get(name)
            if tensor_posi is not None and tensor_nega is not None:
                inputs_shared[name] = torch.concat((tensor_posi, tensor_nega), dim=0)
            elif tensor_shared is not None:
                inputs_shared[name] = torch.concat((tensor_shared, tensor_shared), dim=0)
        inputs_posi.clear()
        inputs_nega.clear()
        return inputs_shared, inputs_posi, inputs_nega


class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh, model_id):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None
        
        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02],
            "Wan2.1-T2V-14B": [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01],
            "Wan2.1-I2V-14B-480P": [2.57151496e+05, -3.54229917e+04,  1.40286849e+03, -1.35890334e+01, 1.32517977e-01],
            "Wan2.1-I2V-14B-720P": [ 8.10705460e+03,  2.13393892e+03, -3.72934672e+02,  1.66203073e+01, -4.17769401e-02],
        }
        if model_id not in self.coefficients_dict:
            supported_model_ids = ", ".join([i for i in self.coefficients_dict])
            raise ValueError(f"{model_id} is not a supported TeaCache model id. Please choose a valid model id in ({supported_model_ids}).")
        self.coefficients = self.coefficients_dict[model_id]

    def check(self, dit: WanModel, x, t_mod):
        modulated_inp = t_mod.clone()
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = self.coefficients
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = x.clone()
        return not should_calc

    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states



class TemporalTiler_BCTHW:
    def __init__(self):
        pass

    def build_1d_mask(self, length, left_bound, right_bound, border_width):
        x = torch.ones((length,))
        if border_width == 0:
            return x
        
        shift = 0.5
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + shift) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip((torch.arange(border_width) + shift) / border_width, dims=(0,))
        return x

    def build_mask(self, data, is_bound, border_width):
        _, _, T, _, _ = data.shape
        t = self.build_1d_mask(T, is_bound[0], is_bound[1], border_width[0])
        mask = repeat(t, "T -> 1 1 T 1 1")
        return mask
    
    def run(self, model_fn, sliding_window_size, sliding_window_stride, computation_device, computation_dtype, model_kwargs, tensor_names, batch_size=None):
        tensor_names = [tensor_name for tensor_name in tensor_names if model_kwargs.get(tensor_name) is not None]
        tensor_dict = {tensor_name: model_kwargs[tensor_name] for tensor_name in tensor_names}
        B, C, T, H, W = tensor_dict[tensor_names[0]].shape
        if batch_size is not None:
            B *= batch_size
        data_device, data_dtype = tensor_dict[tensor_names[0]].device, tensor_dict[tensor_names[0]].dtype
        value = torch.zeros((B, C, T, H, W), device=data_device, dtype=data_dtype)
        weight = torch.zeros((1, 1, T, 1, 1), device=data_device, dtype=data_dtype)
        for t in range(0, T, sliding_window_stride):
            if t - sliding_window_stride >= 0 and t - sliding_window_stride + sliding_window_size >= T:
                continue
            t_ = min(t + sliding_window_size, T)
            model_kwargs.update({
                tensor_name: tensor_dict[tensor_name][:, :, t: t_:, :].to(device=computation_device, dtype=computation_dtype) \
                    for tensor_name in tensor_names
            })
            model_output = model_fn(**model_kwargs).to(device=data_device, dtype=data_dtype)
            mask = self.build_mask(
                model_output,
                is_bound=(t == 0, t_ == T),
                border_width=(sliding_window_size - sliding_window_stride,)
            ).to(device=data_device, dtype=data_dtype)
            value[:, :, t: t_, :, :] += model_output * mask
            weight[:, :, t: t_, :, :] += mask
        value /= weight
        model_kwargs.update(tensor_dict)
        return value


def model_fn_wan_video(
    dit: WanModel,
    latents: torch.Tensor = None,
    input_latents: Optional[torch.Tensor] = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    mllm_hidden_states: torch.Tensor = None,
    mllm_mask: torch.Tensor = None,
    mllm_kv_len: int = None,
    mllm_position_ids: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    tea_cache: TeaCache = None,
    use_unified_sequence_parallel: bool = False,
    sliding_window_size: Optional[int] = None,
    sliding_window_stride: Optional[int] = None,
    cfg_merge: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    fuse_vae_embedding_in_latents: bool = False,
    clean_timestep: torch.Tensor = None, # 修改点：加入 clean_timestep 参数
    **kwargs,
):
    if sliding_window_size is not None and sliding_window_stride is not None:
        model_kwargs = dict(
            dit=dit,
            latents=latents,
            input_latents=input_latents,
            timestep=timestep,
            context=context,
            mllm_hidden_states=mllm_hidden_states,
            mllm_mask=mllm_mask,
            mllm_kv_len=mllm_kv_len,
            mllm_position_ids=mllm_position_ids,
            clip_feature=clip_feature,
            y=y,
            tea_cache=tea_cache,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
            clean_timestep=clean_timestep, # 传递 clean_timestep
        )
        return TemporalTiler_BCTHW().run(
            model_fn_wan_video,
            sliding_window_size, sliding_window_stride,
            latents.device, latents.dtype,
            model_kwargs=model_kwargs,
            tensor_names=["latents", "y"],
            batch_size=2 if cfg_merge else 1
        )

    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                            get_sequence_parallel_world_size,
                                            get_sp_group)

    # Timestep
    if dit.seperated_timestep and fuse_vae_embedding_in_latents:
        timestep = torch.concat([
            torch.zeros((1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device),
            torch.ones((latents.shape[2] - 1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device) * timestep
        ]).flatten()
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0))
        if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
            t_chunks = torch.chunk(t, get_sequence_parallel_world_size(), dim=1)
            t_chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, t_chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in t_chunks]
            t = t_chunks[get_sequence_parallel_rank()]
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
        
    # Text Embedding
    context = dit.text_embedding(context)

    x = latents
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)

    # Image Embedding
    if y is not None and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)
    if clip_feature is not None and dit.require_clip_embedding:
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)

    # MLLM embeddings
    if hasattr(dit, "has_mllm_input") and dit.has_mllm_input and mllm_hidden_states is not None:
        mllm_embeddings = dit.mllm_embedding(
            mllm_hidden_states,
            position_ids=mllm_position_ids,
            mllm_mask=mllm_mask
        )
    else:
        mllm_embeddings = None
        
    # Patchify
    x = dit.patchify(x)
    f, h, w = x.shape[2:]
    x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
    
    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
    
    tokens_per_latent_frame = int(h * w)
    keep_mask = None
    block_ids = None
    block_latent_frames = 4 * BLOCK_DURATION
    num_blocks = math.ceil(f / block_latent_frames) if block_latent_frames > 0 else 1
    
    if input_latents is not None and num_blocks > 1:
        frame_offset = max(f - int(input_latents.shape[2]), 0)
        x_frames = x.view(x.shape[0], f, tokens_per_latent_frame, x.shape[2])
        freqs_frames = freqs.view(f, tokens_per_latent_frame, 1, freqs.shape[-1])
        t_frames = t.view(t.shape[0], f, tokens_per_latent_frame, t.shape[-1]) if t.dim() == 3 else None
        t_mod_frames = t_mod.view(t_mod.shape[0], f, tokens_per_latent_frame, t_mod.shape[-2], t_mod.shape[-1]) if t_mod.dim() == 4 else None
        
        # 修改点 1：使用传入的 clean_timestep，默认 fallback 到 0
        if clean_timestep is None:
            clean_timestep = t.new_zeros((t.shape[0],))
        t_clean = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, clean_timestep))
        t_mod_clean = dit.time_projection(t_clean).unflatten(1, (6, dit.dim))
        
        t_noisy = t if t.dim() == 2 else None
        t_mod_noisy = t_mod if t_mod.dim() == 3 else None
        
        if t_frames is None:
            t_frames = t_noisy[:, None, None, :].expand(t_noisy.shape[0], f, tokens_per_latent_frame, t_noisy.shape[-1])
        if t_mod_frames is None:
            t_mod_frames = t_mod_noisy[:, None, None, :, :].expand(t_mod_noisy.shape[0], f, tokens_per_latent_frame, t_mod_noisy.shape[-2], t_mod_noisy.shape[-1])
        
        mllm_frames = mllm_mask.view(mllm_mask.shape[0], f, tokens_per_latent_frame) if mllm_mask is not None else None

        x_segments, freqs_segments, block_id_segments, keep_mask_segments = [], [], [], []
        t_segments, t_mod_segments = [], []
        mllm_segments = [] if mllm_frames is not None else None
        clean_cache = {}

        def get_clean_tokens(seq_frame_idx: int):
            if seq_frame_idx in clean_cache:
                return clean_cache[seq_frame_idx]
            clean_frame_idx = seq_frame_idx - frame_offset
            clean_latent = input_latents[:, :, clean_frame_idx:clean_frame_idx + 1]
            clean_tokens = dit.patchify(clean_latent)
            clean_tokens = rearrange(clean_tokens, 'b c f h w -> b (f h w) c').contiguous()
            clean_cache[seq_frame_idx] = clean_tokens
            return clean_tokens

        # 修改点 2：支持环境变量自定义 clean 帧数
        clean_frame_count = CLEAN_FRAME_COUNT

        for block_idx in range(num_blocks):
            block_start = block_idx * block_latent_frames
            block_end = min((block_idx + 1) * block_latent_frames, f)
            
            if block_idx > 0:
                # 获取前一块最后的若干个 clean 帧
                prev_frames_start = max(0, block_start - clean_frame_count)
                for prev_frame in range(prev_frames_start, block_start):
                    clean_tokens = get_clean_tokens(prev_frame)
                    seg_len = clean_tokens.shape[1]
                    x_segments.append(clean_tokens)
                    freqs_segments.append(freqs_frames[prev_frame].reshape(seg_len, 1, freqs_frames.shape[-1]))
                    block_id_segments.append(torch.full((seg_len,), block_idx, device=x.device, dtype=torch.int32))
                    keep_mask_segments.append(torch.zeros((seg_len,), device=x.device, dtype=torch.bool)) # False 表示是 clean tokens
                    t_segments.append(t_clean[:, None, :].expand(t_clean.shape[0], seg_len, t_clean.shape[-1]))
                    t_mod_segments.append(t_mod_clean[:, None, :, :].expand(t_mod_clean.shape[0], seg_len, t_mod_clean.shape[-2], t_mod_clean.shape[-1]))
                    if mllm_segments is not None:
                        block_prefix = mllm_frames[:, block_start, 0]
                        mllm_segments.append(block_prefix[:, None].expand(mllm_frames.shape[0], seg_len))

            noisy_tokens = x_frames[:, block_start:block_end].reshape(x.shape[0], -1, x.shape[2])
            seg_len = noisy_tokens.shape[1]
            x_segments.append(noisy_tokens)
            freqs_segments.append(freqs_frames[block_start:block_end].reshape(seg_len, 1, freqs_frames.shape[-1]))
            block_id_segments.append(torch.full((seg_len,), block_idx, device=x.device, dtype=torch.int32))
            keep_mask_segments.append(torch.ones((seg_len,), device=x.device, dtype=torch.bool)) # True 表示是 noisy tokens
            t_segments.append(t_frames[:, block_start:block_end].reshape(t.shape[0], seg_len, t.shape[-1]))
            t_mod_segments.append(t_mod_frames[:, block_start:block_end].reshape(t_mod.shape[0], seg_len, t_mod.shape[-2], t_mod.shape[-1]))
            if mllm_segments is not None:
                mllm_segments.append(mllm_frames[:, block_start:block_end].reshape(mllm_frames.shape[0], seg_len))

        x = torch.cat(x_segments, dim=1)
        freqs = torch.cat(freqs_segments, dim=0)
        block_ids = torch.cat(block_id_segments, dim=0)
        keep_mask = torch.cat(keep_mask_segments, dim=0)
        t = torch.cat(t_segments, dim=1)
        t_mod = torch.cat(t_mod_segments, dim=1)
        if mllm_segments is not None:
            mllm_mask = torch.cat(mllm_segments, dim=1)

    mllm_block_mask = None
    if FLEX_ATTENTION_AVAILABLE and create_block_mask is not None and mllm_embeddings is not None and mllm_mask is not None:
        B, Q_LEN = mllm_mask.shape
        KV_LEN = mllm_kv_len
        def mask_mod(b, h, q_idx, kv_idx):
            return kv_idx < mllm_mask[b, q_idx]
        mllm_block_mask = create_block_mask(mask_mod, B=B, H=None, Q_LEN=Q_LEN, KV_LEN=KV_LEN, device=str(x.device))
        
    dit_block_mask = None

    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod)
    else:
        tea_cache_update = False
        
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            chunks = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)
            pad_shape = chunks[0].shape[1] - chunks[-1].shape[1]
            chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in chunks]
            x = chunks[get_sequence_parallel_rank()]
            
    if tea_cache_update:
        x = tea_cache.update(x)
    else:
        if FLEX_ATTENTION_AVAILABLE and create_block_mask is not None and x is not None:
            B, Q_LEN, KV_LEN = x.shape[0], x.shape[1], x.shape[1]
            if keep_mask is not None:
                # 重新计算 block_ids 列表以匹配拼接后的 x
                block_ids_list = []
                for block_idx in range(num_blocks):
                    block_start = block_idx * block_latent_frames
                    block_end = min((block_idx + 1) * block_latent_frames, f)
                    noisy_frames = block_end - block_start
                    if block_idx > 0:
                        # 插入 clean 帧的 ids (对应环境变量定义的数量)
                        actual_clean_count = min(block_start, clean_frame_count)
                        block_ids_list.append(torch.full((actual_clean_count * tokens_per_latent_frame,), block_idx, device=x.device, dtype=torch.int32))
                    block_ids_list.append(torch.full((noisy_frames * tokens_per_latent_frame,), block_idx, device=x.device, dtype=torch.int32))
                block_ids = torch.cat(block_ids_list, dim=0)
            elif block_ids is None:
                frame_ids = torch.arange(f, device=x.device, dtype=torch.int32)
                block_ids = (frame_ids // block_latent_frames).repeat_interleave(tokens_per_latent_frame)
            
            # 修改点 3：Clean tokens 不关注到当前块的 noisy tokens
            def mask_mod(b, h, q_idx, kv_idx):
                same_block = block_ids[q_idx] == block_ids[kv_idx]
                if keep_mask is not None:
                    q_is_clean = ~keep_mask[q_idx]
                    kv_is_noisy = keep_mask[kv_idx]
                    # 如果查询是 clean 帧且键是 noisy 帧，则不可见
                    is_invalid_clean_look = q_is_clean & kv_is_noisy
                    return same_block & (~is_invalid_clean_look)
                return same_block
                
            dit_block_mask = create_block_mask(mask_mod, B=B, H=None, Q_LEN=Q_LEN, KV_LEN=KV_LEN, device=str(x.device))

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        for block_id, block in enumerate(dit.blocks):
            if use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs, mllm_embeddings, mllm_mask, mllm_block_mask, dit_block_mask,
                        use_reentrant=False,
                    )
            elif use_gradient_checkpointing:
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
        if tea_cache is not None:
            tea_cache.store(x)
            
    if keep_mask is not None:
        x = x[:, keep_mask]
        if t.dim() == 3:
            t = t[:, keep_mask]

    x = dit.head(x, t)
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = get_sp_group().all_gather(x, dim=1)
            x = x[:, :-pad_shape] if pad_shape > 0 else x

    x = dit.unpatchify(x, (f, h, w))
    return x
