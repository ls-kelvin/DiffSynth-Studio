# Weights & Biases (wandb) Logging for WanVideo Training

This guide explains how to use wandb logging with the WanVideo training pipeline.

## Installation

First, install wandb:

```bash
pip install wandb
```

## Setup

Login to wandb (first time only):

```bash
wandb login
```

You'll be prompted to enter your API key from https://wandb.ai/authorize

## Usage

### Basic Usage

Add the `--use_wandb` flag to your training command:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path data/example_video_dataset \
  --dataset_metadata_path data/example_video_dataset/metadata.csv \
  --height 480 \
  --width 832 \
  --dataset_repeat 100 \
  --model_path '[
      "/path/to/diffusion_pytorch_model.safetensors",
      "/path/to/models_t5_umt5-xxl-enc-bf16.pth",
      "/path/to/Wan2.1_VAE.pth"
  ]' \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.1-T2V-1.3B_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --use_wandb
```

### Custom Project and Run Name

Specify custom project and run names:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch examples/wanvideo/model_training/train.py \
  ... [other args] ... \
  --use_wandb \
  --wandb_project "my-custom-project" \
  --wandb_run_name "wan2.1-lora-experiment-001"
```

## Logged Metrics

The following metrics are automatically logged to wandb:

- `train/loss`: Training loss at each step

Additional config parameters are also logged:
- Learning rate
- Weight decay
- Number of epochs
- Batch size (gradient accumulation steps)
- LoRA rank and target modules
- Video dimensions (height, width, num_frames)
- Task type

## Viewing Results

After starting training, you can view your results at:
- https://wandb.ai/YOUR_USERNAME/YOUR_PROJECT_NAME

The wandb dashboard provides:
- Real-time loss curves
- System metrics (GPU, CPU, memory usage)
- Hyperparameter comparison across runs
- Run comparison and analysis tools

## Command-line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--use_wandb` | flag | False | Enable wandb logging |
| `--wandb_project` | str | "diffsynth-wanvideo" | Project name in wandb |
| `--wandb_run_name` | str | None | Custom run name (auto-generated if None) |

## Troubleshooting

### wandb not installed
If you see "Warning: wandb is not installed", install it with:
```bash
pip install wandb
```

### Not logged in
If you see authentication errors, run:
```bash
wandb login
```

### Multi-GPU Training
Wandb logging is automatically handled for multi-GPU training - only the main process will log to wandb to avoid duplicate entries.

## Example Bash Script

You can also use the existing shell scripts with wandb. For example, modify `lora/Wan2.1-T2V-1.3B.sh`:

```bash
accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path "data/example_video_dataset" \
  --dataset_metadata_path "data/example_video_dataset/metadata.csv" \
  --height 480 \
  --width 832 \
  --dataset_repeat 100 \
  --model_path '[
      "/path/to/your/models/diffusion_pytorch_model.safetensors",
      "/path/to/your/models/models_t5_umt5-xxl-enc-bf16.pth",
      "/path/to/your/models/Wan2.1_VAE.pth"
  ]' \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.1-T2V-1.3B_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --use_wandb \
  --wandb_project "wanvideo-experiments" \
  --wandb_run_name "wan21-t2v-lora32"
```
