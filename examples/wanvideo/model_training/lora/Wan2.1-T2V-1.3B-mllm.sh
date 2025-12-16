CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch examples/wanvideo/model_training/train_mllm.py \
  --dataset_base_path data/example_video_dataset \
  --dataset_metadata_path data/example_video_dataset/metadata.csv \
  --height 480 \
  --width 832 \
  --dataset_repeat 100 \
  --model_path '[
    [
      "lib/Qwen3-VL-4B-Instruct/model-00001-of-00002.safetensors",
      "lib/Qwen3-VL-4B-Instruct/model-00002-of-00002.safetensors"
    ],
      "lib/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
      "lib/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
      "lib/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
  ]' \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.1-T2V-1.3B_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --use_mllm_condition 
