#!/bin/bash
# Stage 2: Train DiT with LoRA using cached embeddings
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch examples/wanvideo/model_training/train_mllm.py \
  --dataset_base_path "./models/train/Wan2.1-T2V-1.3B_lora_split_cache" \
  --dataset_repeat 50 \
  --model_path '[
    "lib/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
  ]' \
  --learning_rate 1e-4 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.1-T2V-1.3B_lora" \
  --task "sft:train" \
  --lora_base_model "dit" \
  --lora_target_modules "self_attn.q,self_attn.k,self_attn.v,self_attn.o,cross_attn.q,cross_attn.k,cross_attn.v,cross_attn.o,mlp.c_fc,mlp.c_proj" \
  --lora_rank 32 \
  --use_mllm_condition \
  --use_flex_attention \
  --num_epochs 10