# Weights & Biases (wandb) Integration for WanVideo Pipeline

## Overview

This document describes the implementation of Weights & Biases (wandb) logging for the DiffSynth-Studio WanVideo training pipeline.

## Changes Made

### 1. Core Files Modified

#### `diffsynth/diffusion/logger.py`
- Added wandb initialization and logging support to `ModelLogger` class
- New parameters:
  - `use_wandb`: Boolean flag to enable/disable wandb logging
  - `wandb_project`: Project name in wandb
  - `wandb_run_name`: Custom run name (optional)
  - `wandb_config`: Dictionary of hyperparameters to log
- New methods:
  - `init_wandb()`: Initialize wandb run (only on main process)
  - `log_metrics()`: Log metrics to wandb
- Modified methods:
  - `on_step_end()`: Now accepts optional `loss` parameter and logs it to wandb
  - `on_training_end()`: Properly closes wandb run

#### `diffsynth/diffusion/runner.py`
- Modified `launch_training_task()` to:
  - Initialize wandb before training loop
  - Pass loss value to `model_logger.on_step_end()`

#### `diffsynth/diffusion/parsers.py`
- Added `add_wandb_config()` function with command-line arguments:
  - `--use_wandb`: Enable wandb logging (flag)
  - `--wandb_project`: Set project name (default: "diffsynth-wanvideo")
  - `--wandb_run_name`: Set run name (optional)
- Integrated wandb config into `add_general_config()`

#### `examples/wanvideo/model_training/train.py`
- Updated to pass wandb configuration to `ModelLogger`
- Automatically collects training hyperparameters for wandb config:
  - Learning rate, weight decay, num epochs
  - LoRA configuration (rank, target modules)
  - Video dimensions (height, width, num_frames)
  - Task type

### 2. Documentation Added

#### `examples/wanvideo/model_training/README_wandb.md`
- Comprehensive guide on using wandb with WanVideo training
- Installation and setup instructions
- Usage examples with command-line arguments
- Troubleshooting section
- Example bash scripts

## Features

### Automatic Logging
- **Training Loss**: Logged at every training step as `train/loss`
- **Hyperparameters**: Automatically collected and logged to wandb config
- **Multi-GPU Support**: Only the main process logs to prevent duplicates

### Graceful Degradation
- If wandb is not installed, training continues with a warning
- No breaking changes to existing training scripts
- Backward compatible - existing scripts work without modification

### Easy Integration
Simply add `--use_wandb` to any training command:

```bash
accelerate launch examples/wanvideo/model_training/train.py \
  [existing args...] \
  --use_wandb \
  --wandb_project "my-project" \
  --wandb_run_name "experiment-001"
```

## Installation

To use wandb logging:

```bash
pip install wandb
wandb login
```

## Usage Examples

### Basic Usage (with defaults)
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path data/example_video_dataset \
  --dataset_metadata_path data/example_video_dataset/metadata.csv \
  --height 480 \
  --width 832 \
  --num_frames 81 \
  --model_path '[...]' \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --lora_base_model "dit" \
  --lora_rank 32 \
  --output_path "./models/output" \
  --use_wandb
```

### Custom Project and Run Name
```bash
accelerate launch examples/wanvideo/model_training/train.py \
  [other args...] \
  --use_wandb \
  --wandb_project "video-generation-research" \
  --wandb_run_name "wan21-t2v-lora32-exp01"
```

### Without wandb (backward compatible)
```bash
# Existing scripts work unchanged
accelerate launch examples/wanvideo/model_training/train.py \
  [existing args...]
  # No --use_wandb flag
```

## Logged Metrics

### Real-time Metrics
- `train/loss`: Training loss at each step

### Configuration (logged at start)
- `learning_rate`: Optimizer learning rate
- `weight_decay`: Optimizer weight decay
- `num_epochs`: Total training epochs
- `batch_size`: Effective batch size (gradient accumulation)
- `lora_rank`: LoRA rank parameter
- `lora_target_modules`: LoRA target layers
- `height`: Video frame height
- `width`: Video frame width
- `num_frames`: Number of frames per video
- `task`: Training task type (sft, direct_distill, etc.)

## Architecture

### Flow Diagram
```
Training Script (train.py)
    ↓
ModelLogger.__init__() [wandb config stored]
    ↓
launch_training_task() 
    ↓
ModelLogger.init_wandb() [wandb.init() called on main process]
    ↓
Training Loop
    ├→ Model forward pass
    ├→ Loss computation
    ├→ Backward pass
    └→ ModelLogger.on_step_end(loss=loss) [wandb.log() called]
    ↓
ModelLogger.on_training_end() [wandb.finish() called]
```

### Design Principles

1. **Non-invasive**: Minimal changes to existing code
2. **Optional**: Can be enabled/disabled via command-line flag
3. **Robust**: Gracefully handles missing wandb installation
4. **Multi-GPU aware**: Only main process logs to wandb
5. **Extensible**: Easy to add more metrics in the future

## Testing

The implementation has been tested with:
- ✓ Import without wandb installed (graceful degradation)
- ✓ ModelLogger initialization with wandb disabled
- ✓ ModelLogger initialization with wandb enabled
- ✓ Backward compatibility with existing training scripts

## Future Enhancements

Potential additions for future iterations:
1. Log validation metrics
2. Log learning rate schedule
3. Log gradient norms
4. Upload sample generated videos
5. Log system metrics (GPU utilization, memory usage)
6. Support for wandb sweeps (hyperparameter tuning)
7. Log model checkpoints as wandb artifacts

## Compatibility

- **Python**: 3.8+
- **Dependencies**: wandb (optional)
- **DiffSynth-Studio**: Current version
- **Accelerate**: Compatible with existing multi-GPU setup

## Migration Guide

For existing training scripts, no changes are required. To enable wandb:

1. Install wandb: `pip install wandb`
2. Login: `wandb login`
3. Add flags to training command: `--use_wandb --wandb_project "your-project"`

## Support

For issues or questions:
1. Check the README_wandb.md in `examples/wanvideo/model_training/`
2. Verify wandb installation: `python -c "import wandb; print(wandb.__version__)"`
3. Check wandb status: `wandb status`

## Summary

The wandb integration provides production-ready experiment tracking for the WanVideo pipeline with minimal code changes and maximum flexibility. The implementation follows best practices for distributed training and provides a foundation for future monitoring and visualization enhancements.
