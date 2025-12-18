#!/bin/bash

export DISABLE_FLEX_ATTENTION=0
export WANDB_API_KEY="c034c199c0ac6fe718bd148a2fc8c84602cba136"
export WANDB_MODE="offline"
export CUDA_LAUNCH_BLOCKING=1 

# Stage 2: Train DiT with LoRA using cached embeddings
accelerate launch examples/wanvideo/model_training/train_mllm.py \
  --dataset_base_path data/UltraVideo \
  --dataset_metadata_path data/UltraVideo/metadata.csv \
  --dataset_repeat 1 \
  --height 480 \
  --width 832 \
  --num_frames 161 \
  --model_path '[
    [
      "/root/workspace/zzt/models/Qwen/Qwen3-VL-4B-Instruct/model-00001-of-00002.safetensors",
      "/root/workspace/zzt/models/Qwen/Qwen3-VL-4B-Instruct/model-00002-of-00002.safetensors"
    ],
      "/root/workspace/zzt/models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
      "/root/workspace/zzt/models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
      "/root/workspace/zzt/models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"
  ]' \
  --tokenizer_path "/root/workspace/zzt/models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl" \
  --mllm_processor_path "/root/workspace/zzt/models/Qwen/Qwen3-VL-4B-Instruct" \
  --learning_rate 1e-4 \
  --gradient_accumulation_steps 8 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.1-T2V-1.3B_lora_ultravideo" \
  --task "sft" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 64 \
  --use_mllm_condition \
  --num_epochs 10 \
  --use_wandb \
  --wandb_project "SSD" \
  --wandb_run_name "wan2.1-1.3b-t2v-ultravideo" \
  --save_steps 100