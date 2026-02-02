import os

os.environ["MLLM_NO_CFG"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import re
from pathlib import Path
from PIL import ImageFilter

import torch
from accelerate import Accelerator

from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video_autoregressive_inter import (
    WanVideoAutoregressiveInterPipeline,
    ModelConfig,
)
from diffsynth.core.data.unified_dataset import WanVideoInterDataset
from diffsynth.core.loader.file import load_state_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", type=str, default="/root/workspace/zzt/data/AgiBotWorld-Alpha/agibot_result_sample.jsonl")
    parser.add_argument("--base_path", type=str, default="")
    parser.add_argument("--lora_step", type=int, default=7200)
    parser.add_argument("--run_cate", type=str, default="mllm")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--disable_mllm", action="store_true", help="Disable MLLM condition (default: False)")
    parser.add_argument("--use_gt_mllm", action="store_true", help="Use input_video for MLLM context")
    parser.add_argument("--use_gt_vae", action="store_true", help="Use input_video for VAE clean latents")
    parser.add_argument("--gt_decode", action=argparse.BooleanOptionalAction, default=True, help="Decode with GT prefix latents when saving prompt-switch videos")
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--sigma_shift", type=float, default=5.0)
    parser.add_argument("--target_fps", type=int, default=6)
    parser.add_argument("--source_fps", type=int, default=30)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--max_frames", type=int, default=241)
    return parser.parse_args()


def main():
    args = parse_args()
    accelerator = Accelerator()
    rank = accelerator.process_index
    world_size = accelerator.num_processes

    NEG_PROMPT = (
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
        "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
        "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
        "杂乱的背景，三条腿，背景人很多，倒着走"
    )

    dataset = WanVideoInterDataset(
        base_path=args.base_path,
        metadata_path=args.jsonl_path,
        target_fps=args.target_fps,
        source_fps=args.source_fps,
        height=args.height,
        width=args.width,
        num_frames=args.max_frames,
    )

    total_items = len(dataset.data) if hasattr(dataset, "data") else len(dataset)
    indices = list(range(rank, total_items, world_size))
    accelerator.print(f"[Rank {rank}] Assigned {len(indices)} / {total_items} items.")

    pipe = WanVideoAutoregressiveInterPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=accelerator.device,
        model_configs=[
            ModelConfig(path="/root/workspace/zzt/models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"),
            ModelConfig(path="/root/workspace/zzt/models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(path="/root/workspace/zzt/models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"),
            ModelConfig(path=[
                "/root/workspace/zzt/models/Qwen/Qwen3-VL-4B-Instruct/model-00001-of-00002.safetensors",
                "/root/workspace/zzt/models/Qwen/Qwen3-VL-4B-Instruct/model-00002-of-00002.safetensors",
            ]),
        ],
        tokenizer_config=ModelConfig(path="/root/workspace/zzt/models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"),
        mllm_processor_config=ModelConfig(path="/root/workspace/zzt/models/Qwen/Qwen3-VL-4B-Instruct"),
    )

    # Load LoRA
    lora_path = f"./models/train2/Wan2.1-T2V-1.3B_lora_agibot-alpha_{args.run_cate}/step-{args.lora_step}.safetensors"
    if args.lora_step != 0:
        state_dict = load_state_dict(lora_path, torch_dtype=pipe.torch_dtype, device=accelerator.device)
        dit_state_dict = {k.replace("dit.", ""): v for k, v in state_dict.items() if k.startswith("dit.")}
        mllm_state_dict = {k.replace("mllm_encoder.", ""): v for k, v in state_dict.items() if k.startswith("mllm_encoder.")}
        pipe.dit.load_state_dict(dit_state_dict, strict=False)
        pipe.mllm_encoder.load_state_dict(mllm_state_dict, strict=False)
        pipe.load_lora(pipe.dit, state_dict=dit_state_dict, alpha=1.0)
        if rank == 0:
            print(f"✅ LoRA loaded: {lora_path}")

    output_dir = args.output_dir or f"output_videos/{args.lora_step}/{args.run_cate}"
    os.makedirs(output_dir, exist_ok=True)

    for idx in indices:
        try:
            item = dataset[idx]
        except Exception as e:
            accelerator.print(f"❌ [Rank {rank}] Failed to load item {idx}: {e}")
            continue

        prompt_list = item.get("prompt_list", [])
        clip_frames = item.get("clip_frames", [])
        input_video = item.get("video", None)
        
        # input_video = [
        #     frame.filter(ImageFilter.GaussianBlur(radius=2))
        #     for frame in input_video
        # ]

        if not prompt_list or not clip_frames or input_video is None:
            accelerator.print(f"❌ [Rank {rank}] Missing fields for item {idx}")
            continue

        negative_prompt_list = [NEG_PROMPT] * len(prompt_list)
        height = input_video[0].size[1]
        width = input_video[0].size[0]
        num_frames = len(input_video)

        try:
            output_video = pipe(
                prompt_list=prompt_list,
                negative_prompt_list=negative_prompt_list,
                clip_frames=clip_frames,
                input_video=input_video if (args.use_gt_mllm or args.use_gt_vae) else None,
                seed=args.seed,
                height=height,
                width=width,
                num_frames=num_frames,
                use_mllm_condition=not args.disable_mllm,
                use_gt_mllm=args.use_gt_mllm,
                use_gt_vae=args.use_gt_vae,
                gt_decode=args.gt_decode,
                cfg_scale=args.cfg_scale,
                num_inference_steps=args.num_inference_steps,
                sigma_shift=args.sigma_shift,
                tiled=args.tiled,
            )

            video_id = item["video_id"]
            filename = f"{video_id}_ar_inter.mp4"
            save_path = os.path.join(output_dir, filename)

            counter = 1
            orig_save_path = save_path
            while os.path.exists(save_path):
                name, ext = os.path.splitext(orig_save_path)
                save_path = f"{name}_{counter}{ext}"
                counter += 1

            save_video(output_video, save_path, fps=args.fps, quality=args.quality)
            accelerator.print(f"✅ [Rank {rank}] Saved: {save_path}")

            block_videos = getattr(pipe, "last_block_videos", [])
            for block_video in block_videos:
                block_idx = block_video["block_idx"]
                prompt_idx = block_video["prompt_idx"]
                frames = block_video["frames"]
                block_filename = f"{video_id}_ar_inter_gt_block{block_idx}_prompt{prompt_idx}.mp4"
                block_save_path = os.path.join(output_dir, block_filename)
                counter = 1
                orig_block_save_path = block_save_path
                while os.path.exists(block_save_path):
                    name, ext = os.path.splitext(orig_block_save_path)
                    block_save_path = f"{name}_{counter}{ext}"
                    counter += 1
                save_video(frames, block_save_path, fps=args.fps, quality=args.quality)
                accelerator.print(f"✅ [Rank {rank}] Saved block-switch: {block_save_path}")

        except Exception as e:
            accelerator.print(f"❌ [Rank {rank}] Error on item {idx}: {e}")
            continue

    accelerator.wait_for_everyone()
    if rank == 0:
        print("🎉 All done!")


if __name__ == "__main__":
    main()
