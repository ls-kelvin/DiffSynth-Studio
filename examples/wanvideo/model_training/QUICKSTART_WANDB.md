# Quick Start: Wandb Logging for WanVideo

This is a quick start guide to enable Weights & Biases logging in your WanVideo training.

## 3-Step Setup

### Step 1: Install wandb
```bash
pip install wandb
```

### Step 2: Login (first time only)
```bash
wandb login
```
Enter your API key from: https://wandb.ai/authorize

### Step 3: Add flag to training
Add `--use_wandb` to your existing training command:

```bash
accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path data/example_video_dataset \
  --dataset_metadata_path data/example_video_dataset/metadata.csv \
  --height 480 \
  --width 832 \
  --num_frames 81 \
  --model_path '[
      "path/to/diffusion_pytorch_model.safetensors",
      "path/to/models_t5_umt5-xxl-enc-bf16.pth",
      "path/to/Wan2.1_VAE.pth"
  ]' \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --lora_base_model "dit" \
  --lora_rank 32 \
  --output_path "./models/output" \
  --use_wandb
```

That's it! Your training metrics will now be logged to wandb.

## View Your Results

After starting training, open: https://wandb.ai

## Optional: Custom Project/Run Names

```bash
# Add these flags:
--use_wandb \
--wandb_project "my-awesome-project" \
--wandb_run_name "experiment-001"
```

## What Gets Logged

- **Training loss** at every step
- **Hyperparameters**: learning rate, batch size, LoRA config, etc.
- **System metrics**: GPU usage, memory (automatic)

## Troubleshooting

**"wandb is not installed"**
```bash
pip install wandb
```

**"Login required"**
```bash
wandb login
```

**Multi-GPU training**
Wandb logging is automatically handled - no extra config needed!

## More Info

For detailed documentation, see: [README_wandb.md](./README_wandb.md)

---
**Note**: Existing training scripts work unchanged. Wandb is completely optional!
