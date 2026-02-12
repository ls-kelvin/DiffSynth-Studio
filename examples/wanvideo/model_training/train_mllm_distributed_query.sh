#!/bin/bash

export NCCL_IB_DISABLE=0

# 设置每个节点的进程数，默认为8（如果未设置环境变量）
PET_NPROC_PER_NODE=${PET_NPROC_PER_NODE:-8}
# 总节点数 (默认为2，请根据实际修改)
PET_NNODES=${PET_NNODES:-2}
# 当前节点的 Rank (0 为主节点，1 为从节点...，必须在不同节点上设置不同值)
PET_NODE_RANK=${PET_NODE_RANK:-0}
# 计算总进程数 (Total World Size)
TOTAL_PROCESSES=$((PET_NPROC_PER_NODE * PET_NNODES))

echo "=================================================="
echo "Distributed Training Config:"
echo "  Nodes: ${PET_NNODES}"
echo "  GPU per Node: ${PET_NPROC_PER_NODE}"
echo "  Total Processes (World Size): ${TOTAL_PROCESSES}"
echo "  Current Node Rank: ${PET_NODE_RANK}"
echo "  Master Addr: ${MASTER_ADDR}:${MASTER_PORT}"
echo "=================================================="

export DISABLE_FLEX_ATTENTION=0
export WANDB_API_KEY="c034c199c0ac6fe718bd148a2fc8c84602cba136"
export WANDB_MODE="offline"
# export DISABLE_MLLM=1
# export MLLM_INIT=1
# export MLLM_TRANSFORMER=0
export MLLM_QEURY=256
export MLLM_TRANSFORMER_LAYER=6
export MLLM_MODE="decoupled_kv"
export CLEAN_FRAME_COUNT=0

# Stage 2: Train DiT with LoRA using cached embeddings
accelerate launch \
  --num_processes ${TOTAL_PROCESSES} \
  --num_machines ${PET_NNODES} \
  --machine_rank ${PET_NODE_RANK} \
  --main_process_ip ${MASTER_ADDR} \
  --main_process_port ${MASTER_PORT} \
  --config_file examples/wanvideo/model_training/distributed.yaml \
  examples/wanvideo/model_training/train_mllm_query.py \
  --dataset_base_path "" \
  --dataset_metadata_path /root/workspace/zzt/VideoCaption/agibot-alpha_result.jsonl \
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
  --output_path "./models/train2/Wan2.1-T2V-1.3B_lora_agibot-alpha_mllm_query_mllm" \
  --task "sft" \
  --trainable_models "mllm_encoder" \
  --lora_base_model "dit" \
  --lora_target_modules "self_attn.q,self_attn.k,self_attn.v,self_attn.o,cross_attn.q,cross_attn.k,cross_attn.v,cross_attn.o,ffn.0,ffn.2" \
  --lora_rank 64 \
  --num_epochs 100 \
  --use_wandb \
  --wandb_project "SSD" \
  --wandb_run_name "wan2.1-1.3b-t2v_agibot-alpha_mllm_query_mllm" \
  --save_steps 800 \
  --t5_cfg_drop 0.1 \
  --mllm_cfg_drop 0.1 \
  --use_mllm_condition