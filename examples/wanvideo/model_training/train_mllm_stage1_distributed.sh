#!/bin/bash

# export NCCL_P2P_DISABLE="1"
# export NCCL_IB_DISABLE="1"

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

# Stage 1: Precompute embeddings for MLLM-conditioned Wan
accelerate launch \
  --num_processes ${TOTAL_PROCESSES} \
  --num_machines ${PET_NNODES} \
  --machine_rank ${PET_NODE_RANK} \
  --main_process_ip ${MASTER_ADDR} \
  --main_process_port ${MASTER_PORT} \
  --config_file examples/wanvideo/model_training/distributed_eval.yaml \
  examples/wanvideo/model_training/train_mllm.py \
  --dataset_base_path data/UltraVideo \
  --dataset_metadata_path data/UltraVideo/detailed.jsonl \
  --height 480 \
  --width 832 \
  --num_frames 125 \
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
  --output_path "./models/train/UltraVideo_detailed" \
  --task "sft:data_process" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --use_mllm_condition 

