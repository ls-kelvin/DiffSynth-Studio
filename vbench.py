import os

os.environ["MLLM_NO_CFG"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import argparse
import re
from pathlib import Path
import torch
from accelerate import Accelerator
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video_autoregressive import WanVideoAutoregressivePipeline, ModelConfig
from diffsynth.core.data.unified_dataset import UnifiedDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", type=str, default="/root/workspace/zzt/data/UltraVideo/vbench_expand.jsonl",
                        help="Path to JSONL file with fields: {'prompt': str, 'video_file': str}")
    parser.add_argument("--prompt_type", type=str, default="extended_prompt")
    parser.add_argument("--lora_step", type=int, default=48800)
    parser.add_argument("--run_cate", type=str, default="local_vae")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--disable_mllm", action="store_false", help="Enable MLLM condition (default: True)")
    parser.add_argument("--tiled", action="store_true")
    return parser.parse_args()


def sanitize_filename(s, max_len=40):
    # 移除/替换非法字符，保留中英文、数字、空格、常见标点
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', s)
    s = s.strip().replace(' ', '_')
    return s[:max_len]


def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line.strip())
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"⚠️ Line {line_num} JSON error: {e}")
    return data


def main():
    args = parse_args()
    accelerator = Accelerator()
    rank = accelerator.process_index
    world_size = accelerator.num_processes

    # Fixed negative prompt
    NEG_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

    # Load data
    all_items = load_jsonl(args.jsonl_path)
    local_items = all_items[rank::world_size]  # Shard by rank

    accelerator.print(f"[Rank {rank}] Assigned {len(local_items)} / {len(all_items)} items.")

    pipe = WanVideoAutoregressivePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=accelerator.device,
        model_configs=[
            ModelConfig(path="/root/workspace/zzt/models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"),
            ModelConfig(path="/root/workspace/zzt/models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(path="/root/workspace/zzt/models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"),
            ModelConfig(path=[
                "/root/workspace/zzt/models/Qwen/Qwen3-VL-4B-Instruct/model-00001-of-00002.safetensors",
                "/root/workspace/zzt/models/Qwen/Qwen3-VL-4B-Instruct/model-00002-of-00002.safetensors"
            ])
        ],
        tokenizer_config=ModelConfig(path="/root/workspace/zzt/models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"),
        mllm_processor_config=ModelConfig(path="/root/workspace/zzt/models/Qwen/Qwen3-VL-4B-Instruct")
    )

    # Load LoRA
    lora_path = f"./models/train/Wan2.1-T2V-1.3B_lora_ultravideo_{args.run_cate}/step-{args.lora_step}.safetensors"
    if args.lora_step != 0:
        pipe.load_lora(pipe.dit, lora_path, alpha=1.0)
        if rank == 0:
            print(f"✅ LoRA loaded: {lora_path}")

    # Output dir
    output_dir = args.output_dir or f"output_videos/{args.lora_step}/{args.run_cate}/vbench_expand"
    os.makedirs(output_dir, exist_ok=True)

    # Process one-by-one
    for i, item in enumerate(local_items):
        prompt = item[args.prompt_type]
        
        filename = f"{item['prompt']}.mp4"
        save_path = os.path.join(output_dir, f"{item['prompt']}-0.mp4")
        
        if os.path.exists(save_path):
            continue

        try:
            # Generate single video
            output_video = pipe(
                prompt=prompt,
                negative_prompt=NEG_PROMPT,
                # input_video=input_video,
                seed=args.seed,
                tiled=args.tiled,
                # use_mllm_condition=args.use_mllm,
                # mllm_neg_mode="full",
                num_frames=93
            )

            # Safe filename: {video_stem}_{sanitized_prompt_head}_ori.mp4


            # Avoid overwrite: add _1, _2, ... if exists
            counter = 1
            while os.path.exists(save_path):
                name, ext = os.path.splitext(filename)
                save_path = f"{output_dir}/{name}-{counter}{ext}"
                counter += 1

            save_video(output_video, save_path, fps=args.fps, quality=args.quality)
            accelerator.print(f"✅ [Rank {rank}] Saved: {save_path}")

        except Exception as e:
            accelerator.print(f"❌ [Rank {rank}] Error on item {i} (video: {item['prompt']}): {e}")
            continue

    accelerator.wait_for_everyone()
    if rank == 0:
        print("🎉 All done!")


if __name__ == "__main__":
    main()