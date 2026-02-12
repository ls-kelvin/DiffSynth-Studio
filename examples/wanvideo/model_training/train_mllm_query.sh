#!/bin/bash

export DISABLE_FLEX_ATTENTION=0
export WANDB_API_KEY="c034c199c0ac6fe718bd148a2fc8c84602cba136"
export WANDB_MODE="offline"
# export DISABLE_MLLM=1
# export MLLM_INIT=1
# export MLLM_TRANSFORMER=0
export MLLM_QEURY=256
export MLLM_TRANSFORMER_LAYER=2
export MLLM_MODE="decoupled_kv"

# Stage 2: Train DiT with LoRA using cached embeddings
accelerate launch examples/wanvideo/model_training/train_mllm_query.py \
  --dataset_base_path "" \
  --dataset_metadata_path /root/workspace/zzt/data/AgiBotWorld-Alpha/agirobot_result_.jsonl \
  --dataset_repeat 1 \
  --height 480 \
  --width 640 \
  --num_frames 241 \
  --target_fps 6 \
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
  --learning_rate 2e-5 \
  --gradient_accumulation_steps 4 \
  --remove_prefix_in_ckpt "pipe." \
  --output_path "./models/train2/Wan2.1-T2V-1.3B_lora_agibot-alpha_mllm_query_frozen" \
  --task "sft" \
  --trainable_models "mllm_encoder,dit" \
  --preset_lora_path "/root/workspace/zzt/Diff5/models/train2/Wan2.1-T2V-1.3B_lora_agibot-alpha_mllm_4/step-15200.safetensors" \
  --preset_lora_model "dit" \
  --lora_rank 64 \
  --num_epochs 100 \
  --use_wandb \
  --wandb_project "SSD" \
  --wandb_run_name "wan2.1-1.3b-t2v_agibot-alpha_mllm_query_frozen" \
  --save_steps 800 \
  --t5_cfg_drop 0.1 \
  --use_mllm_condition