#!/bin/bash
# Stage 1: Precompute embeddings for MLLM-conditioned Wan
accelerate launch examples/wanvideo/model_training/train_mllm.py \
  --dataset_base_path data/example_video_dataset \
  --dataset_metadata_path data/example_video_dataset/metadata.csv \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-T2V-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-1.3B:Wan2.1_VAE.pth,Qwen/Qwen3-VL-4B-Instruct:*.safetensors" \
  --output_path "./models/train/Wan2.1-T2V-1.3B_lora_split_cache" \
  --task "sft:data_process" \
  --use_mllm_condition \
  --num_frames 81 \
  --height 480 \
  --width 832
