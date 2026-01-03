import os

os.environ.setdefault("MLLM_NO_CFG", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import torch

try:
    import gradio as gr
except Exception as e:  # pragma: no cover
    raise RuntimeError("Gradio is required. Please `pip install gradio`.") from e

from diffsynth.core.data.unified_dataset import UnifiedDataset
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.pipelines.wan_video_autoregressive import WanVideoAutoregressivePipeline
from diffsynth.utils.data import save_video


DEFAULT_NEG_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，"
    "残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，"
    "三条腿，背景人很多，倒着走"
)


def _parse_pair(text: str, default: Tuple[int, int]) -> Tuple[int, int]:
    if text is None:
        return default
    raw = str(text).strip()
    if not raw:
        return default
    parts = [p.strip() for p in raw.replace("x", ",").split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"Expected two integers like '30,52', got: {text!r}")
    return int(parts[0]), int(parts[1])


def _lora_tag(lora_path: str) -> str:
    if not lora_path:
        return "no-lora"
    parts = Path(lora_path).parts
    tail = parts[-2:] if len(parts) >= 2 else parts
    safe = "-".join(tail)
    return safe.replace(os.sep, "-")


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_input_video(
    input_video_path: str,
    *,
    height: int,
    width: int,
    num_frames: int,
    fps: int,
) -> list:
    if not input_video_path:
        raise ValueError("input_video_path is required in normal mode.")
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"input_video_path not found: {input_video_path}")

    video_op = UnifiedDataset.default_video_operator(
        base_path="",
        height=height,
        width=width,
        num_frames=num_frames,
        fps=fps,
        height_division_factor=16,
        width_division_factor=16,
        time_division_factor=4,
        time_division_remainder=1,
    )
    frames = video_op(input_video_path)
    if not frames:
        raise ValueError(f"Failed to decode any frames from: {input_video_path}")
    return frames


@dataclass
class ModelPaths:
    dit_path: str
    text_encoder_path: str
    vae_path: str
    mllm_shard_paths: Tuple[str, ...]
    tokenizer_path: str
    mllm_processor_path: str


class _PipeHolder:
    def __init__(self):
        self.device: Optional[str] = None
        self.torch_dtype: Optional[torch.dtype] = None
        self.model_paths: Optional[ModelPaths] = None
        self.base_pipe = None
        self.ar_pipe = None
        self.last_lora_path: Optional[str] = None
        self.lora_hotload: bool = False

    def _build_base_pipe(self, device: str, torch_dtype: torch.dtype, model_paths: ModelPaths):
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch_dtype,
            device=device,
            model_configs=[
                ModelConfig(path=model_paths.dit_path),
                ModelConfig(path=model_paths.text_encoder_path),
                ModelConfig(path=model_paths.vae_path),
                ModelConfig(path=list(model_paths.mllm_shard_paths)),
            ],
            tokenizer_config=ModelConfig(path=model_paths.tokenizer_path),
            mllm_processor_config=ModelConfig(path=model_paths.mllm_processor_path),
        )
        lora_hotload = bool(getattr(getattr(pipe, "dit", None), "vram_management_enabled", True))
        return pipe, lora_hotload

    def _build_ar_pipe_from_base(self, base_pipe: WanVideoPipeline, device: str, torch_dtype: torch.dtype):
        # Match WanVideoAutoregressivePipeline.from_pretrained(): share model refs to avoid re-loading weights.
        pipe = WanVideoAutoregressivePipeline(device=device, torch_dtype=torch_dtype)
        for attr in [
            "tokenizer",
            "text_encoder",
            "image_encoder",
            "dit",
            "dit2",
            "vae",
            "motion_controller",
            "vace",
            "vace2",
            "vap",
            "animate_adapter",
            "audio_encoder",
            "mllm_encoder",
            "mllm_processor",
            "audio_processor",
            "scheduler",
        ]:
            if hasattr(base_pipe, attr):
                setattr(pipe, attr, getattr(base_pipe, attr))
        pipe.vram_management_enabled = base_pipe.vram_management_enabled
        return pipe

    def get(
        self,
        *,
        mode: str,
        device: str,
        torch_dtype: torch.dtype,
        model_paths: ModelPaths,
        lora_path: str,
        lora_alpha: float,
        force_reload: bool = False,
    ):
        need_rebuild = (
            force_reload
            or self.base_pipe is None
            or self.device != device
            or self.torch_dtype != torch_dtype
            or self.model_paths != model_paths
        )
        if need_rebuild:
            self.base_pipe, self.lora_hotload = self._build_base_pipe(device, torch_dtype, model_paths)
            self.ar_pipe = None
            self.device = device
            self.torch_dtype = torch_dtype
            self.model_paths = model_paths
            self.last_lora_path = None

        if lora_path and lora_path != self.last_lora_path:
            if self.lora_hotload:
                self.base_pipe.clear_lora()
                self.base_pipe.load_lora(self.base_pipe.dit, lora_path, alpha=lora_alpha, hotload=True)
            else:
                # Non-hotload LoRA is fused into the base weights, so we need a clean rebuild.
                self.base_pipe, self.lora_hotload = self._build_base_pipe(device, torch_dtype, model_paths)
                self.base_pipe.load_lora(self.base_pipe.dit, lora_path, alpha=lora_alpha)
                self.ar_pipe = None
            self.last_lora_path = lora_path

        if not lora_path:
            self.last_lora_path = None

        if mode == "normal":
            return self.base_pipe
        if mode == "ar":
            if self.ar_pipe is None:
                self.ar_pipe = self._build_ar_pipe_from_base(self.base_pipe, device=device, torch_dtype=torch_dtype)
            return self.ar_pipe
        raise ValueError(f"Unknown mode: {mode}")


PIPE_HOLDER = _PipeHolder()


def _generate(
    *,
    mode: str,
    prompt: str,
    negative_prompt: str,
    input_video_path: str,
    lora_path: str,
    lora_alpha: float,
    seed: Optional[int],
    height: int,
    width: int,
    num_frames: int,
    cfg_scale: float,
    num_inference_steps: int,
    sigma_shift: float,
    tiled: bool,
    tile_size_text: str,
    tile_stride_text: str,
    use_mllm_condition: bool,
    mllm_neg_mode: str,
    fps: int,
    quality: int,
    device: str,
    torch_dtype: str,
    model_paths: ModelPaths,
    force_reload: bool,
) -> Tuple[str, str]:
    if not prompt or not str(prompt).strip():
        raise ValueError("prompt is required.")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[torch_dtype]
    pipe = PIPE_HOLDER.get(
        mode=mode,
        device=device,
        torch_dtype=dtype,
        model_paths=model_paths,
        lora_path=lora_path,
        lora_alpha=lora_alpha,
        force_reload=force_reload,
    )

    tile_size = _parse_pair(tile_size_text, (30, 52))
    tile_stride = _parse_pair(tile_stride_text, (15, 26))

    if mode == "normal":
        input_video = _load_input_video(
            input_video_path,
            height=height,
            width=width,
            num_frames=num_frames,
            fps=fps,
        )
        out_frames = WanVideoPipeline.__call__(
            pipe,
            prompt=prompt,
            negative_prompt=negative_prompt,
            input_video=input_video,
            seed=seed,
            rand_device=device,
            height=height,
            width=width,
            num_frames=len(input_video),
            cfg_scale=cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
            use_mllm_condition=use_mllm_condition,
            mllm_neg_mode=mllm_neg_mode,
        )
    elif mode == "ar":
        out_frames = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            rand_device=device,
            height=height,
            width=width,
            num_frames=num_frames,
            cfg_scale=cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    out_dir = os.path.join("output_videos", "gradio")
    _ensure_dir(out_dir)
    save_path = os.path.join(out_dir, f"{mode}-{_lora_tag(lora_path)}-{_timestamp()}.mp4")
    save_video(out_frames, save_path, fps=fps, quality=quality)
    return save_path, f"Saved: {save_path}"


def build_demo(default_device: str, default_dtype: str, model_paths: ModelPaths) -> gr.Blocks:
    with gr.Blocks(title="Wan Video (Normal / AR)") as demo:
        gr.Markdown("Wan video generation UI (normal + autoregressive) with LoRA switching.")

        with gr.Row():
            mode = gr.Dropdown(choices=["normal", "ar"], value="normal", label="Mode")
            device = gr.Textbox(value=default_device, label="Device (cuda/cpu)")
            torch_dtype = gr.Dropdown(choices=["bf16", "fp16", "fp32"], value=default_dtype, label="DType")

        prompt = gr.Textbox(lines=3, label="Prompt")
        negative_prompt = gr.Textbox(lines=2, value=DEFAULT_NEG_PROMPT, label="Negative prompt")

        with gr.Row():
            input_video_path = gr.Textbox(label="Input video path (normal mode)", placeholder="/path/to/input.mp4")
            lora_path = gr.Textbox(label="LoRA ckpt path (.safetensors)", placeholder="./models/train/.../step-xxxx.safetensors")

        with gr.Row():
            lora_alpha = gr.Slider(minimum=0.0, maximum=2.0, value=1.0, step=0.05, label="LoRA alpha")
            seed = gr.Number(value=1, precision=0, label="Seed (int, empty=random)")

        with gr.Row():
            height = gr.Number(value=480, precision=0, label="Height")
            width = gr.Number(value=832, precision=0, label="Width")
            num_frames = gr.Number(value=81, precision=0, label="Num frames")

        with gr.Row():
            cfg_scale = gr.Slider(minimum=1.0, maximum=12.0, value=5.0, step=0.5, label="CFG scale")
            num_inference_steps = gr.Slider(minimum=1, maximum=100, value=50, step=1, label="Inference steps")
            sigma_shift = gr.Slider(minimum=0.0, maximum=10.0, value=5.0, step=0.5, label="Sigma shift")

        with gr.Row():
            tiled = gr.Checkbox(value=True, label="Tiled VAE")
            tile_size_text = gr.Textbox(value="30,52", label="Tile size (H,W)")
            tile_stride_text = gr.Textbox(value="15,26", label="Tile stride (H,W)")

        with gr.Row():
            use_mllm_condition = gr.Checkbox(value=False, label="Use MLLM condition (normal mode)")
            mllm_neg_mode = gr.Dropdown(choices=["full", "text"], value="full", label="MLLM negative mode")

        with gr.Row():
            fps = gr.Slider(minimum=1, maximum=60, value=15, step=1, label="Output FPS")
            quality = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Video quality (imageio)")

        with gr.Accordion("Advanced: model paths (restart app to change)", open=False):
            gr.Textbox(value=model_paths.dit_path, label="DiT path", interactive=False)
            gr.Textbox(value=model_paths.text_encoder_path, label="Text encoder path", interactive=False)
            gr.Textbox(value=model_paths.vae_path, label="VAE path", interactive=False)
            gr.Textbox(value=",".join(model_paths.mllm_shard_paths), label="MLLM shard paths", interactive=False)
            gr.Textbox(value=model_paths.tokenizer_path, label="Tokenizer path", interactive=False)
            gr.Textbox(value=model_paths.mllm_processor_path, label="MLLM processor path", interactive=False)

        with gr.Row():
            force_reload = gr.Checkbox(value=False, label="Force reload pipeline this run")
            run_btn = gr.Button(value="Generate", variant="primary")

        output_video = gr.Video(label="Output video")
        status = gr.Textbox(label="Status", interactive=False)

        def _seed_or_none(x):
            if x is None:
                return None
            try:
                s = int(x)
            except Exception:
                return None
            return s

        def _run(
            mode_,
            prompt_,
            negative_prompt_,
            input_video_path_,
            lora_path_,
            lora_alpha_,
            seed_,
            height_,
            width_,
            num_frames_,
            cfg_scale_,
            num_inference_steps_,
            sigma_shift_,
            tiled_,
            tile_size_text_,
            tile_stride_text_,
            use_mllm_condition_,
            mllm_neg_mode_,
            fps_,
            quality_,
            device_,
            torch_dtype_,
            force_reload_,
        ):
            save_path, msg = _generate(
                mode=mode_,
                prompt=prompt_,
                negative_prompt=negative_prompt_,
                input_video_path=input_video_path_,
                lora_path=lora_path_,
                lora_alpha=float(lora_alpha_),
                seed=_seed_or_none(seed_),
                height=int(height_),
                width=int(width_),
                num_frames=int(num_frames_),
                cfg_scale=float(cfg_scale_),
                num_inference_steps=int(num_inference_steps_),
                sigma_shift=float(sigma_shift_),
                tiled=bool(tiled_),
                tile_size_text=tile_size_text_,
                tile_stride_text=tile_stride_text_,
                use_mllm_condition=bool(use_mllm_condition_),
                mllm_neg_mode=mllm_neg_mode_,
                fps=int(fps_),
                quality=int(quality_),
                device=device_,
                torch_dtype=torch_dtype_,
                model_paths=model_paths,
                force_reload=bool(force_reload_),
            )
            return save_path, msg

        run_btn.click(
            fn=_run,
            inputs=[
                mode,
                prompt,
                negative_prompt,
                input_video_path,
                lora_path,
                lora_alpha,
                seed,
                height,
                width,
                num_frames,
                cfg_scale,
                num_inference_steps,
                sigma_shift,
                tiled,
                tile_size_text,
                tile_stride_text,
                use_mllm_condition,
                mllm_neg_mode,
                fps,
                quality,
                device,
                torch_dtype,
                force_reload,
            ],
            outputs=[output_video, status],
        )

    return demo


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--device", type=str, default=None, help="cuda/cpu; default auto")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])

    parser.add_argument(
        "--dit_path",
        type=str,
        default="/root/workspace/zzt/models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default="/root/workspace/zzt/models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default="/root/workspace/zzt/models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
    )
    parser.add_argument(
        "--mllm_shards",
        type=str,
        default="/root/workspace/zzt/models/Qwen/Qwen3-VL-4B-Instruct/model-00001-of-00002.safetensors,"
        "/root/workspace/zzt/models/Qwen/Qwen3-VL-4B-Instruct/model-00002-of-00002.safetensors",
        help="Comma-separated paths",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="/root/workspace/zzt/models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl",
    )
    parser.add_argument(
        "--mllm_processor_path",
        type=str,
        default="/root/workspace/zzt/models/Qwen/Qwen3-VL-4B-Instruct",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    shards = tuple([p.strip() for p in args.mllm_shards.split(",") if p.strip()])
    if len(shards) == 0:
        raise ValueError("--mllm_shards is empty")

    model_paths = ModelPaths(
        dit_path=args.dit_path,
        text_encoder_path=args.text_encoder_path,
        vae_path=args.vae_path,
        mllm_shard_paths=shards,
        tokenizer_path=args.tokenizer_path,
        mllm_processor_path=args.mllm_processor_path,
    )

    demo = build_demo(default_device=device, default_dtype=args.dtype, model_paths=model_paths)
    demo.queue(max_size=8)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
