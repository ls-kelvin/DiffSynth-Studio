#!/bin/bash

export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

# Stage 1: Precompute embeddings for MLLM-conditioned Wan
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 accelerate launch \
  --num_processes 8 \
  --config_file examples/wanvideo/model_training/distributed_eval.yaml \
  examples/wanvideo/model_training/train_mllm_inter.py \
  --dataset_base_path "" \
  --dataset_metadata_path /root/workspace/zzt/VideoCaption/output/agirobot_result.jsonl \
  --height 480 \
  --width 832 \
  --num_frames 125 \
  --target_fps 6 \
  --dataset_repeat 1 \
  --model_path '[
    [
      "/root/workspace/zzt/models/Qwen/Qwen3-VL-4B-Instruct/model-00001-of-00002.safetensors",
      "/root/workspace/zzt/models/Qwen/Qwen3-VL-4B-Instruct/model-00002-of-00002.safetensors"
    ],
      "/root/workspace/zzt/models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
      "/root/workspace/zzt/models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
  ]' \
  --tokenizer_path "/root/workspace/zzt/models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl" \
  --mllm_processor_path "/root/workspace/zzt/models/Qwen/Qwen3-VL-4B-Instruct" \
  --learning_rate 1e-4 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train2/agibot-alpha" \
  --task "sft:data_process" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --use_mllm_condition 