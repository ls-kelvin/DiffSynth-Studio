#!/bin/bash

export DISABLE_FLEX_ATTENTION=0
export WANDB_API_KEY="c034c199c0ac6fe718bd148a2fc8c84602cba136"
export WANDB_MODE="offline"

# Stage 2: Train DiT with LoRA using cached embeddings
accelerate launch examples/wanvideo/model_training/train_mllm_inter.py \
  --dataset_base_path "./models/train2/agibot-alpha" \
  --dataset_repeat 1 \
  --model_path '[
    "/root/workspace/zzt/models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"
  ]' \
  --tokenizer_path "/root/workspace/zzt/models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl" \
  --mllm_processor_path "/root/workspace/zzt/models/Qwen/Qwen3-VL-4B-Instruct" \
  --learning_rate 1e-5 \
  --gradient_accumulation_steps 4 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train2/Wan2.1-T2V-1.3B_lora_agibot-alpha_mllm" \
  --task "sft:train" \
  --lora_base_model "dit" \
  --lora_target_modules "self_attn.q,self_attn.k,self_attn.v,self_attn.o,cross_attn.q,cross_attn.k,cross_attn.v,cross_attn.o,ffn.0,ffn.2" \
  --lora_rank 64 \
  --use_mllm_condition \
  --num_epochs 100 \
  --use_wandb \
  --wandb_project "SSD" \
  --wandb_run_name "wan2.1-1.3b-t2v_agibot-alpha_mllm" \
  --save_steps 800 \
  --cfg_drop 0.1